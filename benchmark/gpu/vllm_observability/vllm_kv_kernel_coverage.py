#!/usr/bin/env python3
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from dataclasses import dataclass
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Profile:
    name: str
    desc: str
    vllm_args: List[str]
    extra_env: Dict[str, str]


def _run(cmd: Sequence[str], env: Dict[str, str]) -> int:
    p = subprocess.run(cmd, env=env, check=False)
    return int(p.returncode)


def _read_json_from_cmd(cmd: Sequence[str]) -> Dict[str, Any]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(
            f"command failed rc={p.returncode}: {' '.join(cmd)}\n"
            f"stdout:\n{p.stdout}\n\nstderr:\n{p.stderr}"
        )
    return json.loads(p.stdout or "{}")


def _categorize_kernel(name: str) -> str:
    n = name.lower()
    if "reshape_and_cache" in n or ("cache" in n and "reshape" in n):
        return "kv_write"
    if n.startswith("_zn5flash") or "flash_fwd" in n or "flash_bwd" in n:
        return "flash_attention"
    if "paged" in n and "attention" in n:
        return "paged_attention"
    if "swap" in n or "copy" in n:
        return "swap_or_copy"
    if "kv" in n and ("cache" in n or "paged" in n):
        return "kv_related"
    return "other"


def _pick_mem_ops(trace_path: str, top: int) -> Dict[str, Any]:
    # Best-effort: swap/copy paths may show up as memcpy/memset events (not kernels).
    memcpy_count = Counter()
    memcpy_bytes = Counter()
    memcpy_top_ops: List[Dict[str, Any]] = []

    memset_count = Counter()
    memset_bytes = Counter()
    memset_top_ops: List[Dict[str, Any]] = []

    def _maybe_add_top(dst: List[Dict[str, Any]], row: Dict[str, Any]) -> None:
        dst.append(row)
        dst.sort(key=lambda x: int(x.get("bytes", 0) or 0), reverse=True)
        if len(dst) > max(1, top):
            del dst[top:]

    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = obj.get("type")
            if t == "memcpy":
                kind = obj.get("kind") or "unknown"
                b = obj.get("bytes")
                if not isinstance(b, int):
                    continue
                memcpy_count[kind] += 1
                memcpy_bytes[kind] += b
                _maybe_add_top(
                    memcpy_top_ops,
                    {
                        "seq": obj.get("seq"),
                        "kind": kind,
                        "bytes": b,
                        "src": obj.get("src"),
                        "dst": obj.get("dst"),
                        "stream": obj.get("stream"),
                        "async": obj.get("async"),
                    },
                )
            elif t == "memset":
                kind = obj.get("kind") or "unknown"
                b = obj.get("bytes")
                if not isinstance(b, int):
                    continue
                memset_count[kind] += 1
                memset_bytes[kind] += b
                _maybe_add_top(
                    memset_top_ops,
                    {
                        "seq": obj.get("seq"),
                        "kind": kind,
                        "bytes": b,
                        "dst": obj.get("dst"),
                        "stream": obj.get("stream"),
                        "async": obj.get("async"),
                    },
                )

    def _topk_counter(c: Counter) -> List[Dict[str, Any]]:
        return [{"kind": k, "value": int(v)} for (k, v) in c.most_common(max(1, top))]

    return {
        "memcpy_top_by_count": _topk_counter(memcpy_count),
        "memcpy_top_by_bytes": _topk_counter(memcpy_bytes),
        "memcpy_top_ops": memcpy_top_ops,
        "memset_top_by_count": _topk_counter(memset_count),
        "memset_top_by_bytes": _topk_counter(memset_bytes),
        "memset_top_ops": memset_top_ops,
    }


def _default_profiles(model: str) -> List[Profile]:
    # Notes:
    # - We intentionally keep each run short; the goal is to expose different kernel
    #   shapes/paths so we can expand the target list (non-invasive).
    # - swap/copy may still not trigger unless the workload actually hits memory pressure.
    return [
        Profile(
            name="baseline_decode",
            desc="Small prompt + moderate decode to exercise KV write + attention decode.",
            vllm_args=[
                "--model",
                model,
                "--prompt",
                "Write a short haiku about GPUs.",
                "--max-tokens",
                "128",
                "--batch-size",
                "1",
            ],
            extra_env={},
        ),
        Profile(
            name="batch_decode",
            desc="Higher batch to increase concurrency (more CTAs, more streams).",
            vllm_args=[
                "--model",
                model,
                "--prompt",
                "Summarize: KV cache, flash attention, and paged attention.",
                "--max-tokens",
                "128",
                "--batch-size",
                "32",
            ],
            extra_env={},
        ),
        Profile(
            name="long_prefill_short_decode",
            desc="Long prompt to push prefill, but short decode (prefill-heavy).",
            vllm_args=[
                "--model",
                model,
                "--prompt",
                ("Explain GPU memory hierarchy. " * 512).strip(),
                "--max-tokens",
                "1",
                "--batch-size",
                "1",
            ],
            extra_env={},
        ),
        Profile(
            name="kv_pressure",
            desc="Try to push memory pressure via KV override + larger max_model_len (may trigger more KV path variants).",
            vllm_args=[
                "--model",
                model,
                "--prompt",
                ("A" * 4096),
                "--max-tokens",
                "128",
                "--batch-size",
                "8",
                "--max-model-len",
                "2048",
                "--kv-cache-memory-bytes",
                str(1 << 30),  # 1 GiB
                "--gpu-memory-utilization",
                "0.60",
                "--swap-space",
                "8",
            ],
            extra_env={},
        ),
        Profile(
            name="cpu_offload",
            desc="Enable cpu_offload + low gpu_memory_utilization to try to expose swap/copy paths (best-effort).",
            vllm_args=[
                "--model",
                model,
                "--prompt",
                ("B" * 4096),
                "--max-tokens",
                "128",
                "--batch-size",
                "16",
                "--max-model-len",
                "2048",
                "--gpu-memory-utilization",
                "0.40",
                "--swap-space",
                "16",
                "--cpu-offload-gb",
                "8",
            ],
            extra_env={},
        ),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run a small vLLM workload matrix and pick KV-related kernels from bpftime trace (no vLLM changes)."
    )
    ap.add_argument(
        "--model",
        default=os.environ.get("VLLM_MODEL", "Qwen/Qwen3-0.6B"),
        help="HuggingFace model id or local path. Default: $VLLM_MODEL or Qwen/Qwen3-0.6B",
    )
    ap.add_argument(
        "--out-dir",
        default=os.environ.get("BPFTIME_VLLM_KV_COVERAGE_DIR", "/tmp/bpftime-vllm-kv-coverage"),
        help="Output directory. Default: $BPFTIME_VLLM_KV_COVERAGE_DIR or /tmp/bpftime-vllm-kv-coverage",
    )
    ap.add_argument(
        "--profiles",
        default="",
        help="Comma-separated profile names to run (default runs all).",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=30,
        help="Top-N kernels to keep per trace (by count).",
    )
    ap.add_argument(
        "--include",
        default="vllm|flash|paged|kv|cache|reshape|swap|block",
        help="Regex include filter passed to bpftime_trace_pick_kernels.py",
    )
    ap.add_argument(
        "--exclude",
        default="cublas|cudnn|nccl|triton_per_",
        help="Regex exclude filter passed to bpftime_trace_pick_kernels.py",
    )
    ap.add_argument(
        "--only-vllm",
        action="store_true",
        help="Only keep vLLM kernels (name contains 'vllm' or starts with _ZN4vllm).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs without executing.",
    )
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profile_list = _default_profiles(args.model)
    want = {p.strip() for p in (args.profiles.split(",") if args.profiles else []) if p.strip()}
    if want:
        profile_list = [p for p in profile_list if p.name in want]
        missing = sorted(want - {p.name for p in profile_list})
        if missing:
            raise SystemExit(f"unknown profiles: {missing}")

    vllm_script = str(
        pathlib.Path(__file__).resolve().parent.joinpath("vllm_inprocess_generate.py")
    )
    pick_script = str(
        pathlib.Path(__file__).resolve().parent.joinpath("bpftime_trace_pick_kernels.py")
    )
    summarize_script = str(
        pathlib.Path(__file__).resolve().parent.joinpath("bpftime_trace_summarize.py")
    )

    runs_out: List[Dict[str, Any]] = []
    for prof in profile_list:
        ts = time.strftime("%Y%m%d-%H%M%S")
        trace_path = str(out_dir.joinpath(f"trace-{prof.name}-{ts}.jsonl"))
        env = dict(os.environ)
        env.update(
            {
                # Force in-process mode (no extra worker processes).
                "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
                # bpftime tracing
                "BPFTIME_ALLOW_NO_SHM": "1",
                "BPFTIME_LOG_OUTPUT": env.get("BPFTIME_LOG_OUTPUT", "console"),
                "BPFTIME_CUDA_TRACE_PATH": trace_path,
                # Ensure users can run without re-downloading.
                "HF_HOME": env.get("HF_HOME", "/tmp/hf"),
                # LD_PRELOAD: default to in-tree agent if not set.
                "LD_PRELOAD": env.get(
                    "LD_PRELOAD",
                    str(pathlib.Path("build/runtime/agent/libbpftime-agent.so").resolve()),
                ),
            }
        )
        env.update(prof.extra_env)

        cmd = [sys.executable, vllm_script, *prof.vllm_args]
        print(f"[kv_coverage] profile={prof.name} trace={trace_path}")
        print(f"[kv_coverage] desc: {prof.desc}")
        print(f"[kv_coverage] cmd: {' '.join(cmd)}")
        if args.dry_run:
            continue

        rc = _run(cmd, env=env)
        if rc != 0:
            runs_out.append(
                {
                    "profile": prof.name,
                    "desc": prof.desc,
                    "trace": trace_path,
                    "rc": rc,
                    "error": "vllm run failed",
                }
            )
            continue

        # Summaries
        summary = _read_json_from_cmd(
            [sys.executable, summarize_script, "--trace", trace_path, "--json"]
        )
        picked = _read_json_from_cmd(
            [
                sys.executable,
                pick_script,
                "--trace",
                trace_path,
                "--include",
                args.include,
                "--exclude",
                args.exclude,
                "--top",
                str(args.top),
                *(["--only-vllm"] if args.only_vllm else []),
                "--json",
            ]
        )
        for k in picked.get("kernels", []) or []:
            if isinstance(k, dict) and isinstance(k.get("name"), str):
                k["category_guess"] = _categorize_kernel(k["name"])

        runs_out.append(
            {
                "profile": prof.name,
                "desc": prof.desc,
                "trace": trace_path,
                "rc": 0,
                "summary": summary,
                "picked_kernels": picked,
                "mem_ops": _pick_mem_ops(trace_path, top=10),
            }
        )

    out_path = out_dir.joinpath("kv_coverage_report.json")
    out_path.write_text(json.dumps({"runs": runs_out}, indent=2, sort_keys=False))
    print(f"[kv_coverage] wrote {out_path}")
    if runs_out:
        # Print a short, human-friendly tail: last run's picked kernels.
        last = runs_out[-1]
        ks = (((last.get("picked_kernels") or {}).get("kernels")) or [])[:10]
        print("[kv_coverage] last run: top picked kernels (up to 10):")
        for row in ks:
            if not isinstance(row, dict):
                continue
            print(
                f"- count={row.get('count')} category={row.get('category_guess')} name={row.get('name')}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
