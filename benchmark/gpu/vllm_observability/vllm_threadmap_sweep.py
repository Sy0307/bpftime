#!/usr/bin/env python3
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class Mode:
    name: str
    extra_env: Dict[str, str]


def _run(cmd: Sequence[str], env: Dict[str, str], timeout_s: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=timeout_s,
    )


def _read_json_best_effort(path: str) -> Dict[str, Any]:
    # The analyzer prints JSON when --json, so we can parse it directly.
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Systematically verify SM120 SASS thread-map(device) on multiple vLLM hotspot kernels (non-invasive)."
    )
    ap.add_argument(
        "--model",
        default=os.environ.get("VLLM_MODEL", "Qwen/Qwen3-0.6B"),
        help="HuggingFace model id or local path. Default: $VLLM_MODEL or Qwen/Qwen3-0.6B",
    )
    ap.add_argument(
        "--kernels",
        default="rms_norm_kernel,fused_add_rms_norm_kernel,rotary_embedding_kernel,act_and_mul_kernel,unrolled_elementwise_kernel,cutlass_80_wmma_tensorop",
        help="Comma-separated kernel substrings to test (each run patches one filter).",
    )
    ap.add_argument(
        "--modes",
        default="lane0",
        help="Comma-separated density modes: lane0,warp0,full (default: lane0).",
    )
    ap.add_argument(
        "--out-dir",
        default=os.environ.get("BPFTIME_VLLM_THREADMAP_SWEEP_DIR", "/tmp/bpftime-vllm-threadmap-sweep"),
        help="Output directory. Default: $BPFTIME_VLLM_THREADMAP_SWEEP_DIR or /tmp/bpftime-vllm-threadmap-sweep",
    )
    ap.add_argument("--max-records", type=int, default=1, help="BPFTIME_CUDA_SASS_SAMPLE_MAX_RECORDS (default: 1).")
    ap.add_argument("--cta-clamp", action="store_true", default=True, help="Enable CTA clamp (default: on).")
    ap.add_argument("--no-cta-clamp", action="store_true", help="Disable CTA clamp.")
    ap.add_argument(
        "--prefer-exit",
        action="store_true",
        help="Prefer detouring at tail EXIT for thread-map(device) (stability knob for some kernels).",
    )
    ap.add_argument(
        "--retry-prefer-exit-on-failure",
        action="store_true",
        default=True,
        help="If a run fails (nonzero exit), retry once with --prefer-exit. Default: on.",
    )
    ap.add_argument(
        "--no-retry-prefer-exit-on-failure",
        action="store_true",
        help="Disable retry with --prefer-exit.",
    )
    ap.add_argument("--max-tokens", type=int, default=32, help="vLLM max new tokens (default: 32).")
    ap.add_argument("--batch-size", type=int, default=4, help="vLLM batch size (default: 4).")
    ap.add_argument("--timeout-s", type=float, default=300.0, help="Per-run timeout seconds.")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kernels = [k.strip() for k in args.kernels.split(",") if k.strip()]
    if not kernels:
        ap.error("empty --kernels")

    want_modes = {m.strip() for m in args.modes.split(",") if m.strip()}
    known: Dict[str, Mode] = {
        "lane0": Mode("lane0", {"BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_LANE0_ONLY": "1"}),
        "warp0": Mode("warp0", {"BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_WARP0_ONLY": "1"}),
        "full": Mode("full", {}),
    }
    modes: List[Mode] = []
    for m in ["lane0", "warp0", "full"]:
        if m in want_modes:
            modes.append(known[m])
    if not modes:
        ap.error(f"unknown/empty --modes: {args.modes}")

    vllm_script = str(
        pathlib.Path(__file__).resolve().parent.joinpath("vllm_inprocess_generate.py")
    )
    analyze_script = str(
        pathlib.Path(__file__).resolve().parent.joinpath("bpftime_sass_thread_analyze.py")
    )

    common_env = dict(os.environ)
    common_env.update(
        {
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "BPFTIME_ALLOW_NO_SHM": "1",
            "BPFTIME_LOG_OUTPUT": common_env.get("BPFTIME_LOG_OUTPUT", "console"),
            "LD_PRELOAD": common_env.get(
                "LD_PRELOAD",
                str(pathlib.Path("build/runtime/agent/libbpftime-agent.so").resolve()),
            ),
            "HF_HOME": common_env.get("HF_HOME", "/tmp/hf"),
        }
    )

    vllm_cmd = [
        sys.executable,
        vllm_script,
        "--model",
        args.model,
        "--enforce-eager",
        "--max-tokens",
        str(args.max_tokens),
        "--batch-size",
        str(args.batch_size),
    ]

    report: Dict[str, Any] = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "batch_size": args.batch_size,
        "max_records": args.max_records,
        "cta_clamp": bool(args.cta_clamp and not args.no_cta_clamp),
        "prefer_exit": bool(args.prefer_exit),
        "retry_prefer_exit_on_failure": bool(
            args.retry_prefer_exit_on_failure and not args.no_retry_prefer_exit_on_failure
        ),
        "runs": [],
    }

    for k in kernels:
        for mode in modes:
            def one_attempt(prefer_exit: bool) -> Dict[str, Any]:
                ts = time.strftime("%Y%m%d-%H%M%S")
                dump_path = str(out_dir.joinpath(f"sass-thread-{k}-{mode.name}-{'exit' if prefer_exit else 'entry'}-{ts}.jsonl"))
                analyze_path = str(out_dir.joinpath(f"analyze-{k}-{mode.name}-{'exit' if prefer_exit else 'entry'}-{ts}.json"))

                env = dict(common_env)
                env.update(
                    {
                        "BPFTIME_CUDA_SASS_DETOUR": "1",
                        "BPFTIME_CUDA_SASS_SAMPLE": "1",
                        "BPFTIME_CUDA_SASS_SAMPLE_MODE": "thread",
                        "BPFTIME_CUDA_SASS_DETOUR_FILTER": k,
                        "BPFTIME_CUDA_SASS_SAMPLE_FILTER": k,
                        "BPFTIME_CUDA_SASS_SAMPLE_MAX_RECORDS": str(max(1, int(args.max_records))),
                        # Bring-up helper: force a one-time sync+dump after the first
                        # *instrumented* matching kernel launch (avoid early-init syncs).
                        "BPFTIME_CUDA_SASS_SAMPLE_SYNC_AFTER_LAUNCH": "1",
                        "BPFTIME_CUDA_SASS_SAMPLE_SYNC_AFTER_LAUNCH_REQUIRE_SAMPLED": "1",
                        "BPFTIME_CUDA_SASS_SAMPLE_DUMP_PATH": dump_path,
                        "BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE": "1",
                        "BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_STRIDE4": "1",
                    }
                )
                env.update(mode.extra_env)
                if prefer_exit:
                    env["BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_PREFER_EXIT"] = "1"
                if args.no_cta_clamp:
                    env["BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_CTA_CLAMP"] = "0"
                elif args.cta_clamp:
                    env["BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_CTA_CLAMP"] = "1"

                p = _run(vllm_cmd, env=env, timeout_s=float(args.timeout_s))
                ok = p.returncode == 0
                vllm_out_tail = (p.stdout or "")[-4000:]

                analysis: Optional[Dict[str, Any]] = None
                if os.path.exists(dump_path):
                    pa = subprocess.run(
                        [sys.executable, analyze_script, "--dump", dump_path, "--json"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                    )
                    if pa.returncode == 0:
                        try:
                            analysis = json.loads(pa.stdout or "")
                            pathlib.Path(analyze_path).write_text(
                                json.dumps(analysis, indent=2, sort_keys=False)
                            )
                        except Exception:
                            analysis = None

                return {
                    "prefer_exit": prefer_exit,
                    "dump": dump_path,
                    "analyze": analyze_path if analysis else None,
                    "ok": ok,
                    "rc": int(p.returncode),
                    "analysis_ok": analysis is not None,
                    "thread_records": (analysis or {}).get("thread_records") if isinstance(analysis, dict) else None,
                    "inferred_blockdim_x": (analysis or {}).get("inferred_blockdim_x") if isinstance(analysis, dict) else None,
                    "smid_lo8_counts": (analysis or {}).get("smid_lo8_counts") if isinstance(analysis, dict) else None,
                    "vllm_out_tail": vllm_out_tail,
                }

            attempts: List[Dict[str, Any]] = []
            attempts.append(one_attempt(prefer_exit=bool(args.prefer_exit)))

            retry_enabled = bool(args.retry_prefer_exit_on_failure and not args.no_retry_prefer_exit_on_failure)
            if retry_enabled and not args.prefer_exit and not attempts[0]["ok"]:
                attempts.append(one_attempt(prefer_exit=True))

            # Pick the best attempt for the top-level summary.
            best = max(
                attempts,
                key=lambda a: (
                    1 if a.get("ok") else 0,
                    1 if a.get("analysis_ok") else 0,
                    int(a.get("thread_records") or 0),
                ),
            )

            report["runs"].append(
                {
                    "kernel_filter": k,
                    "mode": mode.name,
                    "attempts": attempts,
                    "best": best,
                }
            )

            tr = best.get("thread_records")
            print(
                f"[sweep] kernel={k} mode={mode.name} best_ok={best.get('ok')} thread_records={tr} prefer_exit={best.get('prefer_exit')} dump={best.get('dump')}"
            )

    out_path = out_dir.joinpath("threadmap_sweep_report.json")
    out_path.write_text(json.dumps(report, indent=2, sort_keys=False))
    print(f"[sweep] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
