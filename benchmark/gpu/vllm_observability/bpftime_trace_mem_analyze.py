#!/usr/bin/env python3
import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple
from collections import deque


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


def _as_int(x: Any) -> Optional[int]:
    if isinstance(x, int):
        return x
    if isinstance(x, str):
        try:
            return int(x, 0)
        except Exception:
            return None
    return None


def _as_str(x: Any) -> Optional[str]:
    if isinstance(x, str):
        return x
    return None


@dataclass
class Segment:
    begin_seq: int
    end_seq: int
    stream: str
    sync_api: str
    sync_duration_ns: int
    memcpy_bytes: Counter
    memcpy_count: Counter
    memset_bytes: Counter
    memset_count: Counter
    kernels: Counter


def _new_segment(stream: str, begin_seq: int) -> Segment:
    return Segment(
        begin_seq=begin_seq,
        end_seq=begin_seq,
        stream=stream,
        sync_api="",
        sync_duration_ns=0,
        memcpy_bytes=Counter(),
        memcpy_count=Counter(),
        memset_bytes=Counter(),
        memset_count=Counter(),
        kernels=Counter(),
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Analyze bpftime CUDA trace JSONL and correlate memcpy/memset with kernel windows (best-effort, no vLLM changes)."
    )
    ap.add_argument(
        "--trace",
        default=os.environ.get("BPFTIME_CUDA_TRACE_PATH", ""),
        help="Trace JSONL path (default: $BPFTIME_CUDA_TRACE_PATH).",
    )
    ap.add_argument("--top", type=int, default=10, help="Top-K kernels/mem ops per segment.")
    ap.add_argument(
        "--min-bytes",
        type=int,
        default=1 << 20,
        help="Only print segments whose total memcpy+memset bytes >= this threshold (default: 1 MiB).",
    )
    ap.add_argument(
        "--require-memcpy-kind",
        default="",
        help="Only keep segments that have this memcpy kind (e.g. DtoHAsync/HtoDAsync/DtoDAsync).",
    )
    ap.add_argument(
        "--require-memcpy-bytes",
        type=int,
        default=0,
        help="If --require-memcpy-kind is set, require at least this many bytes of that kind in the segment.",
    )
    ap.add_argument(
        "--large-memcpy-kind",
        default="",
        help="If set, report individual large memcpy ops of this kind (e.g. DtoHAsync/HtoDAsync).",
    )
    ap.add_argument(
        "--large-memcpy-min-bytes",
        type=int,
        default=0,
        help="Minimum bytes for --large-memcpy-kind (0 disables).",
    )
    ap.add_argument(
        "--neighbors",
        type=int,
        default=5,
        help="Number of neighboring kernel launches to capture before/after a large memcpy (default: 5).",
    )
    ap.add_argument(
        "--max-large",
        type=int,
        default=20,
        help="Max number of large memcpy ops to report (default: 20).",
    )
    ap.add_argument("--json", action="store_true", help="Output full JSON summary.")
    args = ap.parse_args()

    if not args.trace:
        ap.error("missing --trace (or set BPFTIME_CUDA_TRACE_PATH)")
    if not os.path.exists(args.trace):
        ap.error(f"trace file not found: {args.trace}")

    # Per-stream rolling segment (between stream syncs; ctx sync resets all).
    seg_by_stream: Dict[str, Segment] = {}

    global_memcpy_bytes = Counter()
    global_memcpy_count = Counter()
    global_memset_bytes = Counter()
    global_memset_count = Counter()
    global_kernels = Counter()

    out_segments: List[Dict[str, Any]] = []
    large_memcpy_out: List[Dict[str, Any]] = []

    # Per-stream context for "large memcpy" neighbor capture.
    last_launches: Dict[str, Deque[Dict[str, Any]]] = defaultdict(
        lambda: deque(maxlen=max(0, args.neighbors))
    )
    pending_large: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def classify_segment(top_kernels: List[Tuple[str, int]]) -> str:
        # Very coarse, best-effort (non-invasive).
        s = " ".join([n for (n, _) in top_kernels[:5]]).lower()
        if "reshape_and_cache" in s:
            return "kv_write"
        if "flash_" in s or "flash::" in s:
            return "attention_flash"
        if "cublas" in s or "cutlass" in s or "gemm" in s:
            return "gemm"
        if "rms_norm" in s or "layer_norm" in s:
            return "norm"
        if "rotary" in s:
            return "rotary"
        return "other"

    def flush_segment(s: Segment) -> None:
        total_bytes = sum(s.memcpy_bytes.values()) + sum(s.memset_bytes.values())
        if total_bytes < max(0, args.min_bytes):
            return
        if args.require_memcpy_kind:
            k = args.require_memcpy_kind
            if int(s.memcpy_bytes.get(k, 0)) < int(max(0, args.require_memcpy_bytes)):
                return
            if int(s.memcpy_count.get(k, 0)) == 0:
                return
        top_k = s.kernels.most_common(max(1, args.top))
        out_segments.append(
            {
                "begin_seq": s.begin_seq,
                "end_seq": s.end_seq,
                "stream": s.stream,
                "sync_api": s.sync_api,
                "sync_duration_ms": float(s.sync_duration_ns) / 1e6,
                "total_bytes": int(total_bytes),
                "class_guess": classify_segment(top_k),
                "memcpy_top_by_bytes": [
                    {"kind": k, "bytes": int(v), "count": int(s.memcpy_count[k])}
                    for (k, v) in s.memcpy_bytes.most_common(max(1, args.top))
                ],
                "memset_top_by_bytes": [
                    {"kind": k, "bytes": int(v), "count": int(s.memset_count[k])}
                    for (k, v) in s.memset_bytes.most_common(max(1, args.top))
                ],
                "top_kernels_by_count": [
                    {"name": n, "count": int(c)} for (n, c) in s.kernels.most_common(max(1, args.top))
                ],
            }
        )

    for obj in _iter_jsonl(args.trace):
        t = obj.get("type")

        if t == "launch":
            name = _as_str(obj.get("name")) or ""
            stream = _as_str(obj.get("stream")) or "0x0"
            seq = _as_int(obj.get("seq")) or 0
            if not name:
                continue
            global_kernels[name] += 1
            # Feed "after" neighbors for pending large memcpys on this stream.
            if pending_large.get(stream):
                for rec in list(pending_large[stream]):
                    rec["after"].append({"seq": seq, "name": name})
                    if len(rec["after"]) >= max(0, args.neighbors):
                        large_memcpy_out.append(rec)
                        pending_large[stream].remove(rec)
                        if len(large_memcpy_out) >= max(0, args.max_large):
                            break
            # Track last launches.
            last_launches[stream].append({"seq": seq, "name": name})
            seg = seg_by_stream.get(stream)
            if seg is None:
                seg = _new_segment(stream=stream, begin_seq=seq)
                seg_by_stream[stream] = seg
            seg.end_seq = seq
            seg.kernels[name] += 1
            continue

        if t == "memcpy":
            kind = _as_str(obj.get("kind")) or "unknown"
            stream = _as_str(obj.get("stream")) or "0x0"
            seq = _as_int(obj.get("seq")) or 0
            b = _as_int(obj.get("bytes")) or 0
            global_memcpy_bytes[kind] += b
            global_memcpy_count[kind] += 1
            # Large memcpy neighbor capture (best-effort).
            if (
                args.large_memcpy_kind
                and kind == args.large_memcpy_kind
                and b >= int(max(0, args.large_memcpy_min_bytes))
                and len(large_memcpy_out) < max(0, args.max_large)
            ):
                rec = {
                    "seq": seq,
                    "stream": stream,
                    "kind": kind,
                    "bytes": int(b),
                    "src": obj.get("src"),
                    "dst": obj.get("dst"),
                    "before": list(last_launches[stream]),
                    "after": [],
                }
                if max(0, args.neighbors) == 0:
                    large_memcpy_out.append(rec)
                else:
                    pending_large[stream].append(rec)
            seg = seg_by_stream.get(stream)
            if seg is None:
                seg = _new_segment(stream=stream, begin_seq=seq)
                seg_by_stream[stream] = seg
            seg.end_seq = seq
            seg.memcpy_bytes[kind] += b
            seg.memcpy_count[kind] += 1
            continue

        if t == "memset":
            kind = _as_str(obj.get("kind")) or "unknown"
            stream = _as_str(obj.get("stream")) or "0x0"
            seq = _as_int(obj.get("seq")) or 0
            b = _as_int(obj.get("bytes")) or 0
            global_memset_bytes[kind] += b
            global_memset_count[kind] += 1
            seg = seg_by_stream.get(stream)
            if seg is None:
                seg = _new_segment(stream=stream, begin_seq=seq)
                seg_by_stream[stream] = seg
            seg.end_seq = seq
            seg.memset_bytes[kind] += b
            seg.memset_count[kind] += 1
            continue

        if t != "sync":
            continue

        api = _as_str(obj.get("api")) or "unknown"
        sync_seq = _as_int(obj.get("seq")) or 0
        duration_ns = _as_int(obj.get("duration_ns")) or 0
        obj_stream = _as_str(obj.get("obj")) or ""

        is_ctx = api == "cuCtxSynchronize"
        is_stream = api == "cuStreamSynchronize"
        if not (is_ctx or is_stream):
            continue

        if is_ctx:
            # Flush all active stream segments and reset.
            for s in list(seg_by_stream.values()):
                s.end_seq = sync_seq
                s.sync_api = api
                s.sync_duration_ns = duration_ns
                flush_segment(s)
            seg_by_stream.clear()
        else:
            if not obj_stream:
                continue
            s = seg_by_stream.get(obj_stream)
            if s is None:
                # Nothing happened on this stream in our view.
                continue
            s.end_seq = sync_seq
            s.sync_api = api
            s.sync_duration_ns = duration_ns
            flush_segment(s)
            # Reset segment for next window.
            seg_by_stream[obj_stream] = _new_segment(stream=obj_stream, begin_seq=sync_seq)

    out = {
        "trace": args.trace,
        "min_bytes": int(max(0, args.min_bytes)),
        "global": {
            "memcpy_bytes_by_kind": global_memcpy_bytes.most_common(),
            "memcpy_count_by_kind": global_memcpy_count.most_common(),
            "memset_bytes_by_kind": global_memset_bytes.most_common(),
            "memset_count_by_kind": global_memset_count.most_common(),
            "top_kernels_by_count": global_kernels.most_common(20),
        },
        "segments": out_segments,
        "large_memcpy": large_memcpy_out,
    }

    if args.json:
        print(json.dumps(out, indent=2, sort_keys=False))
        return 0

    print(f"trace={out['trace']}")
    print("global memcpy_bytes_by_kind:", out["global"]["memcpy_bytes_by_kind"][:10])
    print("global memset_bytes_by_kind:", out["global"]["memset_bytes_by_kind"][:10])
    print(f"segments_printed={len(out_segments)} min_bytes={out['min_bytes']}")
    for seg in out_segments[: min(len(out_segments), 10)]:
        print(
            f"- stream={seg['stream']} sync={seg['sync_api']} total_bytes={seg['total_bytes']} "
            f"class={seg.get('class_guess')} sync_ms={seg['sync_duration_ms']:.3f} seq=[{seg['begin_seq']},{seg['end_seq']}]"
        )
        for row in seg["memcpy_top_by_bytes"][:3]:
            print(f"    memcpy {row['kind']} bytes={row['bytes']} count={row['count']}")
        for row in seg["top_kernels_by_count"][:3]:
            print(f"    kernel {row['count']} {row['name'][:80]}")
    if args.large_memcpy_kind and args.large_memcpy_min_bytes > 0:
        print(
            f"large_memcpy_found={len(large_memcpy_out)} kind={args.large_memcpy_kind} "
            f"min_bytes={args.large_memcpy_min_bytes} neighbors={args.neighbors}"
        )
        for rec in large_memcpy_out[: min(len(large_memcpy_out), 10)]:
            print(f"- memcpy seq={rec['seq']} stream={rec['stream']} bytes={rec['bytes']} kind={rec['kind']}")
            for b in rec["before"][-3:]:
                print(f"    before {b['seq']} {b['name'][:80]}")
            for a in rec["after"][:3]:
                print(f"    after {a['seq']} {a['name'][:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
