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
        description="Analyze bpftime SM120 SASS thread-map JSONL dumps (ctaid/tid distributions and SMID histogram)."
    )
    ap.add_argument("--dump", required=True, help="Thread-map dump JSONL path.")
    ap.add_argument("--json", action="store_true", help="Output JSON summary.")
    args = ap.parse_args()

    meta = None
    thread_cnt = 0
    by_cta = Counter()
    by_warp = Counter()
    by_lane = Counter()
    by_smid_lo8 = Counter()
    cta_blockdim = defaultdict(set)

    for obj in _iter_jsonl(args.dump):
        t = obj.get("type")
        if t == "bpftime_sass_sample_meta":
            meta = obj
            continue
        if t != "bpftime_sass_thread":
            continue
        thread_cnt += 1
        cta = obj.get("ctaid_x")
        tid = obj.get("tid_x")
        by_cta[cta] += 1
        by_warp[obj.get("warp_id")] += 1
        by_lane[obj.get("lane_id")] += 1
        by_smid_lo8[obj.get("smid_lo8")] += 1
        if isinstance(cta, int) and isinstance(tid, int):
            cta_blockdim[cta].add(tid)

    inferred_blockdim = None
    if cta_blockdim:
        # Best-effort blockDim.x inference under gates:
        # - full: max(tid_x)+1
        # - lane0-only: lane_id==0 only, infer (max(warp_id)+1)*32
        # - warp0-only: warp_id==0 only, infer 32
        lane_ids = {k for (k, v) in by_lane.items() if v != 0}
        warp_ids = {k for (k, v) in by_warp.items() if v != 0}
        if lane_ids == {0} and warp_ids:
            inferred_blockdim = (max(warp_ids) + 1) * 32
        elif warp_ids == {0} and lane_ids:
            inferred_blockdim = 32
        else:
            inferred_blockdim = max((max(v) + 1) for v in cta_blockdim.values() if v)

    out = {
        "dump": args.dump,
        "meta": meta,
        "thread_records": thread_cnt,
        "inferred_blockdim_x": inferred_blockdim,
        "ctaid_counts": by_cta.most_common(20),
        "smid_lo8_counts": by_smid_lo8.most_common(20),
        "warp_id_counts": by_warp.most_common(10),
        "lane_id_counts": by_lane.most_common(10),
    }

    if args.json:
        print(json.dumps(out, indent=2, sort_keys=False))
        return 0

    print(f"dump={args.dump}")
    if meta:
        print(
            f"mode={meta.get('mode')} thread_device={meta.get('thread_device')} "
            f"cta_clamp={meta.get('cta_clamp')} max_ctas={meta.get('max_ctas')} "
            f"record_bytes={meta.get('record_bytes')}"
        )
    print(f"thread_records={thread_cnt} inferred_blockdim_x={inferred_blockdim}")
    print("ctaid_counts(top20):", by_cta.most_common(20))
    print("smid_lo8_counts(top10):", by_smid_lo8.most_common(10))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
