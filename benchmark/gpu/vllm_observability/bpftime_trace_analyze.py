#!/usr/bin/env python3
import argparse
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
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
                # Best-effort: tolerate truncated tail lines.
                continue


@dataclass
class Launch:
    seq: int
    name: str
    stream: str
    ts_ns: int


@dataclass
class Timing:
    launch_seq: int
    name: str
    stream: str
    gpu_ms: float
    flushed_by_seq: int
    flushed_by_api: str


def _as_int(x: Any) -> Optional[int]:
    if isinstance(x, int):
        return x
    if isinstance(x, str):
        try:
            return int(x, 0)
        except Exception:
            return None
    return None


def _as_float(x: Any) -> Optional[float]:
    if isinstance(x, (int, float)):
        return float(x)
    return None


def _as_str(x: Any) -> Optional[str]:
    if isinstance(x, str):
        return x
    return None


def analyze_sync_waits(
    trace_path: str,
    kernel_name_filter: Optional[str],
    top_k: int,
) -> Dict[str, Any]:
    rx = re.compile(kernel_name_filter, re.IGNORECASE) if kernel_name_filter else None
    launches: Dict[int, Launch] = {}
    timings_by_launch: Dict[int, Timing] = {}

    # Per-stream launch ordering (by launch seq, which is the trace sequence number).
    stream_launch_seqs: Dict[str, List[int]] = defaultdict(list)

    # Boundary markers:
    # - For stream sync: last seen sync seq per stream.
    # - For ctx sync: last seen ctx sync trace seq.
    last_stream_sync_seq: Dict[str, int] = defaultdict(lambda: -1)
    last_ctx_sync_seq: int = -1

    out_sync: List[Dict[str, Any]] = []

    # First pass: we stream through events and compute sync attribution online.
    for obj in _iter_jsonl(trace_path):
        t = obj.get("type")

        if t == "launch":
            seq = _as_int(obj.get("seq"))
            name = _as_str(obj.get("name"))
            stream = _as_str(obj.get("stream"))
            ts_ns = _as_int(obj.get("ts_ns"))
            if seq is None or name is None or stream is None or ts_ns is None:
                continue
            launches[seq] = Launch(seq=seq, name=name, stream=stream, ts_ns=ts_ns)
            stream_launch_seqs[stream].append(seq)
            continue

        if t == "kernel_timing":
            launch_seq = _as_int(obj.get("launch_seq"))
            name = _as_str(obj.get("name"))
            stream = _as_str(obj.get("stream"))
            gpu_ms = _as_float(obj.get("gpu_ms"))
            flushed_by_seq = _as_int(obj.get("flushed_by_seq"))
            flushed_by_api = _as_str(obj.get("flushed_by_api"))
            if (
                launch_seq is None
                or name is None
                or stream is None
                or gpu_ms is None
                or flushed_by_seq is None
                or flushed_by_api is None
            ):
                continue
            timings_by_launch[launch_seq] = Timing(
                launch_seq=launch_seq,
                name=name,
                stream=stream,
                gpu_ms=gpu_ms,
                flushed_by_seq=flushed_by_seq,
                flushed_by_api=flushed_by_api,
            )
            continue

        if t != "sync":
            continue

        api = _as_str(obj.get("api")) or "unknown"
        sync_seq = _as_int(obj.get("seq"))
        duration_ns = _as_int(obj.get("duration_ns"))
        obj_stream = _as_str(obj.get("obj"))
        if sync_seq is None or duration_ns is None:
            continue

        is_ctx = api == "cuCtxSynchronize"
        is_stream = api == "cuStreamSynchronize"
        if not (is_ctx or is_stream):
            continue

        if is_stream and not obj_stream:
            continue

        if is_ctx:
            # All streams since last ctx sync.
            begin_seq = last_ctx_sync_seq
            last_ctx_sync_seq = sync_seq
            # Reset per-stream markers too (ctx sync implies all work is complete).
            for s in list(last_stream_sync_seq.keys()):
                last_stream_sync_seq[s] = sync_seq
            candidates: List[Launch] = []
            for s, seqs in stream_launch_seqs.items():
                for ls in seqs:
                    if begin_seq < ls < sync_seq:
                        candidates.append(launches[ls])
        else:
            begin_seq = last_stream_sync_seq[obj_stream]
            last_stream_sync_seq[obj_stream] = sync_seq
            candidates = [
                launches[ls]
                for ls in stream_launch_seqs.get(obj_stream, [])
                if begin_seq < ls < sync_seq
            ]

        # Optional kernel name filter: only count kernels matching regex.
        if rx:
            candidates = [c for c in candidates if rx.search(c.name)]

        kernel_count = Counter()
        kernel_gpu_ms = Counter()
        untimed = 0
        timed = 0
        for c in candidates:
            kernel_count[c.name] += 1
            ti = timings_by_launch.get(c.seq)
            if ti is None:
                untimed += 1
                continue
            timed += 1
            kernel_gpu_ms[c.name] += float(ti.gpu_ms)

        top_by_gpu = [
            {"name": name, "total_gpu_ms": float(ms), "count": int(kernel_count[name])}
            for (name, ms) in kernel_gpu_ms.most_common(max(1, top_k))
        ]
        top_by_count = [
            {"name": name, "count": int(n), "timed_gpu_ms": float(kernel_gpu_ms.get(name, 0.0))}
            for (name, n) in kernel_count.most_common(max(1, top_k))
        ]

        out_sync.append(
            {
                "sync_seq": sync_seq,
                "api": api,
                "obj": obj_stream,
                "duration_ms": float(duration_ns) / 1e6,
                "launches_considered": len(candidates),
                "timed_launches": timed,
                "untimed_launches": untimed,
                "top_by_total_gpu_ms": top_by_gpu,
                "top_by_count": top_by_count,
            }
        )

    # Global summary
    total_syncs = len(out_sync)
    total_timing = len(timings_by_launch)
    total_launch = len(launches)
    return {
        "trace_path": trace_path,
        "kernel_name_filter": kernel_name_filter,
        "total_launch_records": total_launch,
        "total_kernel_timing_records": total_timing,
        "total_syncs_analyzed": total_syncs,
        "syncs": out_sync,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Analyze bpftime CUDA trace JSONL and attribute sync time to kernels (best-effort, no vLLM changes)."
    )
    ap.add_argument(
        "--trace",
        default=os.environ.get("BPFTIME_CUDA_TRACE_PATH", ""),
        help="Trace JSONL path (default: $BPFTIME_CUDA_TRACE_PATH).",
    )
    ap.add_argument(
        "--kernel-filter",
        default="",
        help="Regex filter for kernel names when attributing sync waits (optional).",
    )
    ap.add_argument("--top", type=int, default=10, help="Top-K kernels to show per sync.")
    ap.add_argument("--json", action="store_true", help="Output full JSON (default prints a short summary).")
    args = ap.parse_args()

    if not args.trace:
        ap.error("missing --trace (or set BPFTIME_CUDA_TRACE_PATH)")
    if not os.path.exists(args.trace):
        ap.error(f"trace file not found: {args.trace}")

    out = analyze_sync_waits(
        trace_path=args.trace,
        kernel_name_filter=(args.kernel_filter or None),
        top_k=max(1, args.top),
    )

    if args.json:
        print(json.dumps(out, indent=2, sort_keys=False))
        return 0

    syncs = out["syncs"]
    print(f"trace={out['trace_path']}")
    print(
        f"launch={out['total_launch_records']} timing={out['total_kernel_timing_records']} "
        f"syncs={out['total_syncs_analyzed']} filter={out['kernel_name_filter']!r}"
    )
    if not syncs:
        return 0
    # Print last few syncs (more useful than the early init ones).
    tail = syncs[-min(10, len(syncs)) :]
    for s in tail:
        api = s["api"]
        obj = s["obj"]
        dur = s["duration_ms"]
        considered = s["launches_considered"]
        timed = s["timed_launches"]
        print(f"- {api} obj={obj} cpu_ms={dur:.3f} launches={considered} timed={timed}")
        for row in s["top_by_total_gpu_ms"][: min(5, args.top)]:
            print(f"  - gpu_ms={row['total_gpu_ms']:.3f} count={row['count']} {row['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
