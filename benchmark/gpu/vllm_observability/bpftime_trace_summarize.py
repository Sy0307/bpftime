#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Summary:
    trace_path: str
    total_lines: int
    unique_kernels: int
    top_kernels: Tuple[Tuple[str, int], ...]
    matched_kernels: Tuple[Tuple[str, int], ...]
    timed_kernels: Tuple[Tuple[str, Dict[str, float]], ...]


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Best-effort: tolerate a truncated last line if the process
                # terminates without flushing/closing the trace stream.
                continue


def summarize_trace(path: str, top_n: int, pattern: Optional[str]) -> Summary:
    kernel_counts: Counter[str] = Counter()
    kernel_timing_count: Counter[str] = Counter()
    kernel_timing_sum_ms: Counter[str] = Counter()
    total = 0
    for obj in _iter_jsonl(path):
        total += 1
        t = obj.get("type")
        if t == "launch":
            name = obj.get("name")
            if isinstance(name, str) and name:
                kernel_counts[name] += 1
        elif t == "kernel_timing":
            name = obj.get("name")
            ms = obj.get("gpu_ms")
            if isinstance(name, str) and name and isinstance(ms, (int, float)):
                kernel_timing_count[name] += 1
                kernel_timing_sum_ms[name] += float(ms)

    top = tuple(kernel_counts.most_common(top_n))
    if pattern:
        rx = re.compile(pattern, re.IGNORECASE)
        matched = tuple((k, v) for (k, v) in kernel_counts.most_common() if rx.search(k))
    else:
        matched = tuple()

    timed: List[Tuple[str, Dict[str, float]]] = []
    for name, cnt in kernel_timing_count.most_common(top_n):
        total_ms = float(kernel_timing_sum_ms.get(name, 0.0))
        avg_ms = float(total_ms / cnt) if cnt else 0.0
        timed.append((name, {"count": float(cnt), "total_ms": total_ms, "avg_ms": avg_ms}))

    return Summary(
        trace_path=path,
        total_lines=total,
        unique_kernels=len(kernel_counts),
        top_kernels=top,
        matched_kernels=matched,
        timed_kernels=tuple(timed),
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Summarize bpftime CUDA launch trace JSONL (top kernels + optional regex filter)."
    )
    ap.add_argument(
        "--trace",
        default=os.environ.get("BPFTIME_CUDA_TRACE_PATH", ""),
        help="Path to bpftime CUDA trace JSONL (default: $BPFTIME_CUDA_TRACE_PATH).",
    )
    ap.add_argument("--top", type=int, default=30, help="Top-N kernel names to print.")
    ap.add_argument(
        "--grep",
        default="",
        help="Regex to filter kernel names (case-insensitive). Example: 'cache|kv|paged|reshape'.",
    )
    ap.add_argument("--json", action="store_true", help="Output JSON instead of text.")
    args = ap.parse_args()

    if not args.trace:
        ap.error("missing --trace (or set BPFTIME_CUDA_TRACE_PATH)")
    if not os.path.exists(args.trace):
        ap.error(f"trace file not found: {args.trace}")

    s = summarize_trace(args.trace, top_n=max(1, args.top), pattern=(args.grep or None))
    if args.json:
        print(
            json.dumps(
                {
                    "trace_path": s.trace_path,
                    "total_lines": s.total_lines,
                    "unique_kernels": s.unique_kernels,
                    "top_kernels": s.top_kernels,
                    "matched_kernels": s.matched_kernels,
                    "timed_kernels": s.timed_kernels,
                },
                indent=2,
                sort_keys=False,
            )
        )
        return 0

    print(f"trace={s.trace_path}")
    print(f"total_lines={s.total_lines} unique_kernels={s.unique_kernels}")
    print("\nTop kernels:")
    for name, n in s.top_kernels:
        print(f"{n:8d}  {name}")
    if args.grep:
        print(f"\nMatched kernels (/{args.grep}/i):")
        for name, n in s.matched_kernels:
            print(f"{n:8d}  {name}")
    if s.timed_kernels:
        print("\nTimed kernels (top by count):")
        for name, st in s.timed_kernels:
            print(
                f"{int(st['count']):8d}  total_ms={st['total_ms']:.3f} avg_ms={st['avg_ms']:.6f}  {name}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
