#!/usr/bin/env python3
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class Profile:
    name: str
    desc: str
    extra_env: Dict[str, str]


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


def _pick_mem_ops(trace_path: str, top: int) -> Dict[str, Any]:
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

    for obj in _iter_jsonl(trace_path):
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


def _read_json_from_cmd(cmd: Sequence[str], env: Dict[str, str]) -> Dict[str, Any]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, check=False)
    if p.returncode != 0:
        raise RuntimeError(
            f"command failed rc={p.returncode}: {' '.join(cmd)}\n"
            f"stdout:\n{p.stdout}\n\nstderr:\n{p.stderr}"
        )
    # load test prints a single JSON object at the end, but bpftime logs may also
    # be interleaved on stdout. The JSON itself is pretty-printed (multi-line).
    # Best-effort: find a suffix that is a valid JSON object by trying each '{'
    # candidate from the end.
    out = p.stdout or ""
    brace_positions: List[int] = []
    i = out.find("{")
    while i != -1:
        brace_positions.append(i)
        i = out.find("{", i + 1)
    for i in reversed(brace_positions):
        # Heuristic: only consider candidates that look like a top-level JSON start
        # (beginning of output or at a line start).
        if i != 0 and out[i - 1] not in ("\n", "\r"):
            continue
        try:
            obj = json.loads(out[i:])
        except Exception:
            continue
        if isinstance(obj, dict) and (
            "trace_path" in obj
            or "trace" in obj
            or "load" in obj
            or "top_kernels" in obj
            or "kernels" in obj
        ):
            return obj
    raise RuntimeError(
        f"failed to parse JSON from stdout (expected final JSON object); cmd={' '.join(cmd)}\n"
        f"stdout_tail:\n{out[-4000:]}\n\nstderr_tail:\n{(p.stderr or '')[-4000:]}"
    )


def _default_profiles() -> List[Profile]:
    # Design intent:
    # - Cover decode-heavy vs prefill-heavy shapes.
    # - Try to trigger host<->device memcpy (swap/offload), best-effort.
    # - Keep defaults conservative to avoid OOM; users can tune env for their GPU.
    return [
        Profile(
            name="server_baseline",
            desc="Baseline openai server load (small prompts, moderate decode).",
            extra_env={
                "VLLM_CONCURRENCY": "8",
                "VLLM_TOTAL_REQUESTS": "64",
                "VLLM_MAX_TOKENS_MIN": "16",
                "VLLM_MAX_TOKENS_MAX": "128",
                "VLLM_PROMPT_LEN": "512",
                "VLLM_PROMPT2_LEN": "512",
                "VLLM_GPU_MEM_UTIL": "0.20",
            },
        ),
        Profile(
            name="server_prefill_heavy",
            desc="Long prompts (prefill-heavy), tiny decode.",
            extra_env={
                "VLLM_CONCURRENCY": "8",
                "VLLM_TOTAL_REQUESTS": "32",
                "VLLM_MAX_TOKENS_MIN": "1",
                "VLLM_MAX_TOKENS_MAX": "8",
                "VLLM_PROMPT_LEN": "8192",
                "VLLM_PROMPT2_LEN": "8192",
                "VLLM_GPU_MEM_UTIL": "0.20",
            },
        ),
        Profile(
            name="server_decode_heavy",
            desc="Short prompts (minimal prefill), long decode.",
            extra_env={
                "VLLM_CONCURRENCY": "16",
                "VLLM_TOTAL_REQUESTS": "64",
                "VLLM_MAX_TOKENS_MIN": "256",
                "VLLM_MAX_TOKENS_MAX": "512",
                "VLLM_PROMPT_LEN": "128",
                "VLLM_PROMPT2_LEN": "128",
                "VLLM_GPU_MEM_UTIL": "0.20",
            },
        ),
        Profile(
            name="server_kv_offload_pressure",
            desc="Try to force KV offload/swap: low GPU mem util + cpu offload + swap + smaller KV cache.",
            extra_env={
                "VLLM_CONCURRENCY": "32",
                "VLLM_TOTAL_REQUESTS": "128",
                "VLLM_MAX_TOKENS_MIN": "128",
                "VLLM_MAX_TOKENS_MAX": "256",
                "VLLM_PROMPT_LEN": "4096",
                "VLLM_PROMPT2_LEN": "4096",
                "VLLM_GPU_MEM_UTIL": "0.10",
                "VLLM_MAX_MODEL_LEN": "2048",
                # Try to cap GPU KV to encourage swapping/offload.
                "VLLM_KV_CACHE_MEMORY_BYTES": str(512 * 1024 * 1024),  # 512 MiB
                "VLLM_SWAP_SPACE": "16",
                "VLLM_CPU_OFFLOAD_GB": "8",
                # Note: vLLM's newer kv-offloading connector can require additional
                # configs; we keep this profile on the older swap/cpu-offload knobs.
            },
        ),
        Profile(
            name="server_paged_attention_triton",
            desc="Force a non-default attention backend (best-effort) to try to surface paged-attention-like kernels.",
            extra_env={
                "VLLM_CONCURRENCY": "8",
                "VLLM_TOTAL_REQUESTS": "32",
                "VLLM_MAX_TOKENS_MIN": "64",
                "VLLM_MAX_TOKENS_MAX": "128",
                "VLLM_PROMPT_LEN": "1024",
                "VLLM_PROMPT2_LEN": "1024",
                "VLLM_GPU_MEM_UTIL": "0.20",
                # vLLM supports this flag; exact effect depends on build.
                "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
            },
        ),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run real vLLM OpenAI server load profiles and collect KV-related kernel/memcpy coverage from bpftime trace."
    )
    ap.add_argument(
        "--model",
        default=os.environ.get("VLLM_MODEL", "Qwen/Qwen3-0.6B"),
        help="HuggingFace model id or local path (server). Default: $VLLM_MODEL or Qwen/Qwen3-0.6B",
    )
    ap.add_argument(
        "--out-dir",
        default=os.environ.get("BPFTIME_VLLM_KV_SERVER_COVERAGE_DIR", "/tmp/bpftime-vllm-kv-server-coverage"),
        help="Output directory. Default: $BPFTIME_VLLM_KV_SERVER_COVERAGE_DIR or /tmp/bpftime-vllm-kv-server-coverage",
    )
    ap.add_argument(
        "--profiles",
        default="",
        help="Comma-separated profile names to run (default runs all).",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=40,
        help="Top-N kernels to keep per trace (by count).",
    )
    ap.add_argument(
        "--include",
        default="vllm|flash|paged|kv|cache|reshape|swap|block|triton",
        help="Regex include filter passed to bpftime_trace_pick_kernels.py.",
    )
    ap.add_argument(
        "--exclude",
        default="cublas|cudnn|nccl",
        help="Regex exclude filter passed to bpftime_trace_pick_kernels.py.",
    )
    ap.add_argument(
        "--only-vllm",
        action="store_true",
        help="Only keep kernels whose name contains 'vllm' or starts with _ZN4vllm.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs without executing.",
    )
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profile_list = _default_profiles()
    want = {p.strip() for p in (args.profiles.split(",") if args.profiles else []) if p.strip()}
    if want:
        profile_list = [p for p in profile_list if p.name in want]
        missing = sorted(want - {p.name for p in profile_list})
        if missing:
            raise SystemExit(f"unknown profiles: {missing}")

    load_test = str(
        pathlib.Path(__file__).resolve().parent.joinpath("vllm_openai_server_load_test.py")
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
        log_path = str(out_dir.joinpath(f"server-{prof.name}-{ts}.log"))

        env = dict(os.environ)
        env.update(
            {
                "VLLM_MODEL": args.model,
                "VLLM_SERVER_LOG": log_path,
                "BPFTIME_CUDA_TRACE_PATH": trace_path,
                "HF_HOME": env.get("HF_HOME", "/tmp/hf"),
                "LD_PRELOAD": env.get(
                    "LD_PRELOAD",
                    str(pathlib.Path("build/runtime/agent/libbpftime-agent.so").resolve()),
                ),
            }
        )
        env.update(prof.extra_env)

        cmd = [sys.executable, load_test]
        print(f"[kv_server_coverage] profile={prof.name} trace={trace_path}")
        print(f"[kv_server_coverage] desc: {prof.desc}")
        if args.dry_run:
            continue

        try:
            out = _read_json_from_cmd(cmd, env=env)
        except Exception as e:
            runs_out.append(
                {
                    "profile": prof.name,
                    "desc": prof.desc,
                    "trace": trace_path,
                    "server_log": log_path,
                    "rc": 1,
                    "error": str(e),
                }
            )
            continue

        # Attach extra analyzers to the same trace.
        summary = _read_json_from_cmd(
            [sys.executable, summarize_script, "--trace", trace_path, "--top", str(args.top), "--json"],
            env=dict(os.environ),
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
            ],
            env=dict(os.environ),
        )
        mem_ops = _pick_mem_ops(trace_path, top=10)

        runs_out.append(
            {
                "profile": prof.name,
                "desc": prof.desc,
                "trace": trace_path,
                "server_log": log_path,
                "rc": 0,
                "server_out": out,
                "summary": summary,
                "picked_kernels": picked,
                "mem_ops": mem_ops,
            }
        )

    out_path = out_dir.joinpath("kv_server_coverage_report.json")
    out_path.write_text(json.dumps({"runs": runs_out}, indent=2, sort_keys=False))
    print(f"[kv_server_coverage] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
