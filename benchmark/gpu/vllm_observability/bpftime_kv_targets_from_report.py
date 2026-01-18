#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _iter_runs(report: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    runs = report.get("runs")
    if not isinstance(runs, list):
        return []
    for r in runs:
        if isinstance(r, dict):
            yield r


def _categorize_kernel(name: str) -> str:
    n = name.lower()
    if "reshape_and_cache" in n:
        return "kv_write"
    if "flash_fwd_splitkv_kernel" in n or "flash_fwd_splitkv_combine_kernel" in n or n.startswith("_zn5flash"):
        return "flash_attention"
    if "paged" in n and "attn" in n:
        return "paged_attention"
    if "swap" in n or "copy" in n:
        return "swap_or_copy_kernel"
    if "kv" in n and "cache" in n:
        return "kv_related"
    return "other"


def _suggest_substring(full_name: str) -> str:
    # Prefer stable “semantic” substrings rather than full mangled names.
    for s in [
        "reshape_and_cache_flash_kernel",
        "flash_fwd_splitkv_kernel",
        "flash_fwd_splitkv_combine_kernel",
        "paged_attention",
        "swap",
        "copy",
    ]:
        if s in full_name:
            return s
    # Fallback: keep a shorter prefix of the mangled name (still works as substring).
    return full_name[:64]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract recommended KV/attn/offload observation targets from a kv_*_coverage_report.json (no vLLM changes)."
    )
    ap.add_argument("--report", required=True, help="Path to kv_coverage_report.json or kv_server_coverage_report.json")
    ap.add_argument("--top", type=int, default=10, help="Top-N kernels per category (by count).")
    ap.add_argument(
        "--profiles",
        default="",
        help="Comma-separated run.profile names to include (default includes all rc==0 runs).",
    )
    ap.add_argument("--json", action="store_true", help="Output JSON.")
    args = ap.parse_args()

    report = json.load(open(args.report, "r", encoding="utf-8"))

    want = {p.strip() for p in args.profiles.split(",") if p.strip()}
    kernel_counts_by_cat: Dict[str, Counter] = defaultdict(Counter)
    substring_counts_by_cat: Dict[str, Counter] = defaultdict(Counter)
    substring_example_name: Dict[Tuple[str, str], str] = {}

    memcpy_bytes = Counter()
    memcpy_counts = Counter()

    for run in _iter_runs(report):
        if run.get("rc") != 0:
            continue
        prof = run.get("profile")
        if want and prof not in want:
            continue

        pk = (run.get("picked_kernels") or {}).get("kernels") or []
        for row in pk:
            if not isinstance(row, dict):
                continue
            name = row.get("name")
            cnt = row.get("count")
            if not isinstance(name, str) or not name:
                continue
            if not isinstance(cnt, int):
                try:
                    cnt = int(cnt)
                except Exception:
                    continue
            cat = _categorize_kernel(name)
            kernel_counts_by_cat[cat][name] += cnt
            sub = _suggest_substring(name)
            substring_counts_by_cat[cat][sub] += cnt
            key = (cat, sub)
            # Keep one example full name for reference.
            if key not in substring_example_name:
                substring_example_name[key] = name

        mem_ops = run.get("mem_ops") or {}
        # We treat memcpy (DtoH/HtoD) as the best non-invasive “swap/offload/copy” signal.
        for row in mem_ops.get("memcpy_top_by_bytes") or []:
            if not isinstance(row, dict):
                continue
            kind = row.get("kind")
            val = row.get("value")
            if isinstance(kind, str) and isinstance(val, int):
                memcpy_bytes[kind] += val
        for row in mem_ops.get("memcpy_top_by_count") or []:
            if not isinstance(row, dict):
                continue
            kind = row.get("kind")
            val = row.get("value")
            if isinstance(kind, str) and isinstance(val, int):
                memcpy_counts[kind] += val

    def top_k(cat: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for sub, cnt in substring_counts_by_cat.get(cat, Counter()).most_common(max(1, args.top)):
            out.append(
                {
                    "count": int(cnt),
                    "suggest_substring": sub,
                    "example_name": substring_example_name.get((cat, sub), ""),
                }
            )
        return out

    out = {
        "report": args.report,
        "profiles": sorted(list(want)) if want else "ALL(rc==0)",
        "recommended_kernel_substrings": {
            "kv_write": top_k("kv_write"),
            "flash_attention": top_k("flash_attention"),
            "paged_attention": top_k("paged_attention"),
            "swap_or_copy_kernel": top_k("swap_or_copy_kernel"),
            "kv_related": top_k("kv_related"),
        },
        "memcpy_summary": {
            "total_bytes_by_kind": memcpy_bytes.most_common(),
            "total_count_by_kind": memcpy_counts.most_common(),
        },
    }

    if args.json:
        print(json.dumps(out, indent=2, sort_keys=False))
        return 0

    print(f"report={out['report']}")
    print(f"profiles={out['profiles']}")
    print("recommended kernel substrings:")
    for cat in ["kv_write", "flash_attention", "paged_attention", "swap_or_copy_kernel", "kv_related"]:
        rows = out["recommended_kernel_substrings"][cat]
        print(f"- {cat}:")
        if not rows:
            print("  (none)")
            continue
        for r in rows:
            print(f"  - {r['suggest_substring']}  # count={r['count']}")
    print("memcpy bytes by kind (sum across runs):", out["memcpy_summary"]["total_bytes_by_kind"])
    print("memcpy count by kind (sum across runs):", out["memcpy_summary"]["total_count_by_kind"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
