#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _compile_rx(pat: str) -> Optional[re.Pattern]:
    if not pat:
        return None
    return re.compile(pat, re.IGNORECASE)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Pick candidate KV/attention-related kernels from a bpftime CUDA trace JSONL (top-N by count)."
    )
    ap.add_argument("--trace", required=True, help="bpftime CUDA trace JSONL path.")
    ap.add_argument(
        "--include",
        default="vllm|flash|paged|kv|cache|reshape|swap|block",
        help="Regex include filter for kernel names (default: vLLM/KV/attn-ish keywords).",
    )
    ap.add_argument(
        "--exclude",
        default="cublas|cudnn|nccl|triton_per_",
        help="Regex exclude filter for kernel names (default: cublas/cudnn/nccl/triton_per_).",
    )
    ap.add_argument(
        "--only-vllm",
        action="store_true",
        help="Only keep kernels whose name contains 'vllm' or starts with '_ZN4vllm'.",
    )
    ap.add_argument("--top", type=int, default=30, help="Top-N kernels to print.")
    ap.add_argument(
        "--sort",
        choices=["count", "gpu_ms"],
        default="count",
        help="Sort by launch count (default) or total kernel GPU time (requires kernel_timing records).",
    )
    ap.add_argument("--json", action="store_true", help="Output JSON.")
    args = ap.parse_args()

    rx_in = _compile_rx(args.include)
    rx_ex = _compile_rx(args.exclude)

    counts = Counter()
    gpu_ms_sum = Counter()
    examples: Dict[str, Dict[str, Any]] = {}
    for obj in _iter_jsonl(args.trace):
        if obj.get("type") != "launch":
            # Optional: timing records may exist when BPFTIME_CUDA_KERNEL_TIMING=1.
            if obj.get("type") == "kernel_timing":
                name = obj.get("name")
                ms = obj.get("gpu_ms")
                if isinstance(name, str) and name and isinstance(ms, (int, float)):
                    gpu_ms_sum[name] += float(ms)
            continue
        name = obj.get("name")
        if not isinstance(name, str) or not name:
            continue
        if rx_in and not rx_in.search(name):
            continue
        if rx_ex and rx_ex.search(name):
            continue
        if args.only_vllm and ("vllm" not in name and not name.startswith("_ZN4vllm")):
            continue
        counts[name] += 1
        if name not in examples:
            examples[name] = {
                "name": name,
                "count": 0,
                "example_grid": [obj.get("grid_x"), obj.get("grid_y"), obj.get("grid_z")],
                "example_block": [
                    obj.get("block_x"),
                    obj.get("block_y"),
                    obj.get("block_z"),
                ],
                "example_shared_mem": obj.get("shared_mem"),
            }

    out: List[Dict[str, Any]] = []
    if args.sort == "gpu_ms":
        items = gpu_ms_sum.most_common(max(1, args.top))
    else:
        items = counts.most_common(max(1, args.top))
    for name, n in items:
        e = dict(examples.get(name) or {"name": name})
        if args.sort == "gpu_ms":
            e["gpu_ms_total"] = float(n)
            e["count"] = int(counts.get(name, 0))
        else:
            e["count"] = int(n)
            e["gpu_ms_total"] = float(gpu_ms_sum.get(name, 0.0))
        out.append(e)

    if args.json:
        print(json.dumps({"trace": args.trace, "kernels": out}, indent=2))
        return 0

    print(f"trace={args.trace}")
    print(f"include={args.include!r} exclude={args.exclude!r} top={args.top} sort={args.sort}")
    for row in out:
        print(
            f"- count={row['count']} gpu_ms_total={row.get('gpu_ms_total')} name={row['name']} "
            f"block={row.get('example_block')} grid={row.get('example_grid')} shmem={row.get('example_shared_mem')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
