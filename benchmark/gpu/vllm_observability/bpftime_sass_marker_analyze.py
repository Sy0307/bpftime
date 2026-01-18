#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Optional


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


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Analyze bpftime SM120 SASS pc-marker JSONL dumps (marker_off hotspot counts)."
    )
    ap.add_argument("--dump", required=True, help="Marker dump JSONL path.")
    ap.add_argument("--top", type=int, default=20, help="Top-K marker offsets to show.")
    ap.add_argument("--tag", default="", help="Optional numeric tag filter (hex or dec).")
    ap.add_argument("--json", action="store_true", help="Output full JSON summary.")
    args = ap.parse_args()

    tag_filter: Optional[int] = None
    if args.tag:
        tag_filter = int(args.tag, 0)

    meta = None
    total_markers = 0
    seq_min = None
    seq_max = None
    by_off = Counter()
    by_tag = Counter()
    by_tag_off = defaultdict(Counter)

    for obj in _iter_jsonl(args.dump):
        t = obj.get("type")
        if t == "bpftime_sass_sample_meta":
            meta = obj
            continue
        if t != "bpftime_sass_marker":
            continue
        tag = obj.get("tag")
        if tag_filter is not None and tag != tag_filter:
            continue
        total_markers += 1
        by_off[obj.get("marker_off")] += 1
        by_tag[tag] += 1
        by_tag_off[tag][obj.get("marker_off")] += 1
        seq = obj.get("seq")
        if isinstance(seq, int):
            seq_min = seq if seq_min is None else min(seq_min, seq)
            seq_max = seq if seq_max is None else max(seq_max, seq)

    out = {
        "dump": args.dump,
        "meta": meta,
        "total_markers": total_markers,
        "seq_min": seq_min,
        "seq_max": seq_max,
        "top_marker_off": [
            {"marker_off": off, "count": int(cnt)}
            for off, cnt in by_off.most_common(max(1, args.top))
        ],
        "top_tags": [{"tag": t, "count": int(cnt)} for t, cnt in by_tag.most_common(10)],
    }

    if args.json:
        print(json.dumps(out, indent=2, sort_keys=False))
        return 0

    print(f"dump={args.dump}")
    if meta:
        print(
            f"ring_entries={meta.get('ring_entries')} write_idx={meta.get('write_idx')} "
            f"max_ctas={meta.get('max_ctas')} lane0_only={meta.get('lane0_only')} "
            f"warp0_only={meta.get('warp0_only')}"
        )
    print(f"markers={total_markers} seq_min={seq_min} seq_max={seq_max} tag_filter={tag_filter}")
    for row in out["top_marker_off"]:
        print(f"- marker_off={row['marker_off']} count={row['count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
