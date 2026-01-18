#!/usr/bin/env python3
import argparse
import os
import pathlib
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence


def _run(cmd: Sequence[str], env: Dict[str, str]) -> int:
    p = subprocess.run(cmd, env=env, check=False)
    return int(p.returncode)


def _env_common(out_dir: pathlib.Path, trace_path: Optional[str]) -> Dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "BPFTIME_ALLOW_NO_SHM": "1",
            "BPFTIME_LOG_OUTPUT": env.get("BPFTIME_LOG_OUTPUT", "console"),
            "HF_HOME": env.get("HF_HOME", "/tmp/hf"),
            "LD_PRELOAD": env.get(
                "LD_PRELOAD",
                str(pathlib.Path("build/runtime/agent/libbpftime-agent.so").resolve()),
            ),
        }
    )
    if trace_path:
        env["BPFTIME_CUDA_TRACE_PATH"] = trace_path
    return env


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Bring-up one KV/attn kernel in real vLLM (no vLLM changes): timing / marker / thread-map(device)."
    )
    ap.add_argument(
        "--mode",
        required=True,
        choices=[
            "trace",
            "timing",
            "marker",
            "thread_lane0",
            "thread_warp0",
            "thread_full",
        ],
        help="Bring-up mode.",
    )
    ap.add_argument(
        "--kernel",
        default="",
        help="Kernel substring filter (required for timing/marker/thread_*).",
    )
    ap.add_argument(
        "--model",
        default=os.environ.get("VLLM_MODEL", "Qwen/Qwen3-0.6B"),
        help="HuggingFace model id or local path. Default: $VLLM_MODEL or Qwen/Qwen3-0.6B",
    )
    ap.add_argument(
        "--prompt",
        default=os.environ.get("VLLM_PROMPT", "Write a short haiku about GPUs."),
        help="Prompt text.",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("VLLM_MAX_TOKENS", "32")),
        help="Max new tokens.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("VLLM_BATCH_SIZE", "1")),
        help="Batch size in one generate() call.",
    )
    ap.add_argument(
        "--out-dir",
        default=os.environ.get("BPFTIME_VLLM_KV_BRINGUP_DIR", "/tmp/bpftime-vllm-kv-bringup"),
        help="Output directory for trace/dumps.",
    )
    ap.add_argument(
        "--max-records",
        type=int,
        default=1,
        help="BPFTIME_CUDA_SASS_SAMPLE_MAX_RECORDS (thread-map) or max_ctas clamp (marker).",
    )
    ap.add_argument(
        "--marker-offsets",
        default="0,0x10",
        help="Comma/space separated marker offsets (default: 0,0x10).",
    )
    ap.add_argument(
        "--marker-ring",
        type=int,
        default=4096,
        help="BPFTIME_CUDA_SASS_MARKER_RING_ENTRIES (default: 4096).",
    )
    ap.add_argument(
        "--no-cta-clamp",
        action="store_true",
        help="Disable CTA clamp for marker/thread-map (not recommended during bring-up).",
    )
    ap.add_argument(
        "--timing-sample-every",
        type=int,
        default=10,
        help="BPFTIME_CUDA_KERNEL_TIMING_SAMPLE_EVERY (default: 10).",
    )
    args = ap.parse_args()

    if args.mode != "trace" and not args.kernel:
        ap.error("--kernel is required for this mode")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    trace_path = str(out_dir.joinpath(f"trace-{args.mode}-{ts}.jsonl"))

    vllm_script = str(
        pathlib.Path(__file__).resolve().parent.joinpath("vllm_inprocess_generate.py")
    )
    analyze_trace = str(
        pathlib.Path(__file__).resolve().parent.joinpath("bpftime_trace_analyze.py")
    )
    analyze_thread = str(
        pathlib.Path(__file__).resolve().parent.joinpath("bpftime_sass_thread_analyze.py")
    )
    analyze_marker = str(
        pathlib.Path(__file__).resolve().parent.joinpath("bpftime_sass_marker_analyze.py")
    )

    vllm_cmd = [
        sys.executable,
        vllm_script,
        "--model",
        args.model,
        "--prompt",
        args.prompt,
        "--max-tokens",
        str(args.max_tokens),
        "--batch-size",
        str(args.batch_size),
    ]

    env = _env_common(out_dir, trace_path if args.mode in ("trace", "timing") else None)

    if args.mode == "timing":
        env["BPFTIME_CUDA_KERNEL_TIMING"] = "1"
        env["BPFTIME_CUDA_KERNEL_TIMING_FILTER"] = args.kernel
        env["BPFTIME_CUDA_KERNEL_TIMING_SAMPLE_EVERY"] = str(max(1, args.timing_sample_every))

    if args.mode.startswith("thread_"):
        dump_path = str(out_dir.joinpath(f"sass-thread-{ts}.jsonl"))
        env.update(
            {
                "BPFTIME_CUDA_SASS_DETOUR": "1",
                "BPFTIME_CUDA_SASS_SAMPLE": "1",
                "BPFTIME_CUDA_SASS_SAMPLE_MODE": "thread",
                "BPFTIME_CUDA_SASS_DETOUR_FILTER": args.kernel,
                "BPFTIME_CUDA_SASS_SAMPLE_MAX_RECORDS": str(max(1, args.max_records)),
                "BPFTIME_CUDA_SASS_SAMPLE_SYNC_AFTER_LAUNCH": "1",
                "BPFTIME_CUDA_SASS_SAMPLE_DUMP_ON_SYNC": "1",
                "BPFTIME_CUDA_SASS_SAMPLE_DUMP_PATH": dump_path,
                # device-per-thread: accurate tid/warp/lane
                "BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE": "1",
                "BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_STRIDE4": "1",
            }
        )
        if not args.no_cta_clamp:
            env["BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_CTA_CLAMP"] = "1"
        if args.mode == "thread_lane0":
            env["BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_LANE0_ONLY"] = "1"
        elif args.mode == "thread_warp0":
            env["BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_WARP0_ONLY"] = "1"
        else:
            # full: no density gate
            pass

    if args.mode == "marker":
        dump_path = str(out_dir.joinpath(f"sass-marker-{ts}.jsonl"))
        env.update(
            {
                "BPFTIME_CUDA_SASS_DETOUR": "1",
                "BPFTIME_CUDA_SASS_SAMPLE": "1",
                "BPFTIME_CUDA_SASS_SAMPLE_MODE": "marker",
                "BPFTIME_CUDA_SASS_DETOUR_FILTER": args.kernel,
                "BPFTIME_CUDA_SASS_MARKER_OFFSETS": args.marker_offsets,
                "BPFTIME_CUDA_SASS_MARKER_RING_ENTRIES": str(max(1, args.marker_ring)),
                "BPFTIME_CUDA_SASS_SAMPLE_DUMP_ON_SYNC": "1",
                "BPFTIME_CUDA_SASS_SAMPLE_DUMP_PATH": dump_path,
            }
        )
        if args.no_cta_clamp:
            env["BPFTIME_CUDA_SASS_MARKER_CTA_CLAMP_DISABLE"] = "1"

    print(f"[bringup] mode={args.mode} kernel={args.kernel!r}")
    print(f"[bringup] out_dir={out_dir}")
    print(f"[bringup] vllm_cmd={' '.join(vllm_cmd)}")
    rc = _run(vllm_cmd, env=env)
    if rc != 0:
        print(f"[bringup] vllm run failed rc={rc}", file=sys.stderr)
        return rc

    if args.mode == "timing":
        # Best-effort: show sync attribution filtered to the kernel substring.
        _run(
            [
                sys.executable,
                analyze_trace,
                "--trace",
                trace_path,
                "--kernel-filter",
                args.kernel,
                "--top",
                "10",
            ],
            env=dict(os.environ),
        )

    if args.mode.startswith("thread_"):
        _run([sys.executable, analyze_thread, "--dump", dump_path], env=dict(os.environ))

    if args.mode == "marker":
        _run([sys.executable, analyze_marker, "--dump", dump_path], env=dict(os.environ))

    print("[bringup] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

