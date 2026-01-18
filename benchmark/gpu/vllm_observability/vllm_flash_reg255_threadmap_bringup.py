#!/usr/bin/env python3

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
import glob
from typing import Any, Dict, Optional, Sequence, Tuple


def _run(cmd: Sequence[str], env: Dict[str, str], timeout_s: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=timeout_s,
    )


def _iter_dump_files(dump_path: str) -> Sequence[str]:
    # `BPFTIME_CUDA_SASS_SAMPLE_DUMP_PATH` supports `%p` expansion on the agent side.
    # When vLLM spawns helper processes, using `%p` avoids JSONL clobbering.
    if "%p" in dump_path:
        return sorted(glob.glob(dump_path.replace("%p", "*")))
    if os.path.exists(dump_path):
        return [dump_path]
    return []


def _extract_target_func_id(dump_path: str) -> Optional[int]:
    # We expect a bpftime_sass_sample_meta line with control.target_func_id in identify closure.
    candidates: Dict[int, int] = {}
    for path in _iter_dump_files(dump_path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if obj.get("type") != "bpftime_sass_sample_meta":
                        continue
                    ctrl = obj.get("control")
                    if not isinstance(ctrl, dict):
                        continue
                    v = ctrl.get("target_func_id")
                    if isinstance(v, int) and v != 0:
                        candidates[v] = candidates.get(v, 0) + 1
                        continue
                    v2 = ctrl.get("reserved1")
                    if isinstance(v2, int) and v2 != 0:
                        candidates[v2] = candidates.get(v2, 0) + 1
                        continue
                    slots = ctrl.get("slots")
                    if isinstance(slots, list) and slots:
                        counts: Dict[int, int] = {}
                        for x in slots:
                            if isinstance(x, int) and x != 0:
                                counts[x] = counts.get(x, 0) + 1
                        if counts:
                            chosen = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
                            candidates[chosen] = candidates.get(chosen, 0) + 1
        except Exception:
            continue
    if not candidates:
        return None
    # Prefer the most frequently observed candidate across (possibly multiple) JSONLs.
    return sorted(candidates.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _count_thread_records(dump_path: str) -> int:
    n = 0
    for path in _iter_dump_files(dump_path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if obj.get("type") == "bpftime_sass_thread":
                        n += 1
        except Exception:
            continue
    return n


def _env_common(agent_so: str, hf_home: str) -> Dict[str, str]:
    env = dict(os.environ)
    # Avoid permission/ownership issues with the default shared-memory name
    # `/dev/shm/bpftime_maps_shm` on shared containers. Use a per-run unique name
    # so the agent can initialize shared memory instead of falling back to the
    # (less tested) CUDA-standalone mode.
    if "BPFTIME_GLOBAL_SHM_NAME" not in env:
        env["BPFTIME_GLOBAL_SHM_NAME"] = f"bpftime_maps_shm_{os.getuid()}_{int(time.time())}"
    env.update(
        {
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "BPFTIME_ALLOW_NO_SHM": "1",
            "BPFTIME_LOG_OUTPUT": env.get("BPFTIME_LOG_OUTPUT", "console"),
            "LD_PRELOAD": env.get("LD_PRELOAD", agent_so),
            "HF_HOME": env.get("HF_HOME", hf_home),
        }
    )
    return env


def _extract_target_func_id_from_log(log_path: str) -> Optional[int]:
    if not os.path.exists(log_path):
        return None
    # Example line (only when BPFTIME_CUDA_SASS_DETOUR_DEBUG=1):
    #   SASS control: set target func_id=269 stream=0x0
    best: Optional[int] = None
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if "SASS control: set target func_id=" not in line:
                    continue
                try:
                    # Best-effort parse.
                    s = line.split("set target func_id=", 1)[1]
                    num = ""
                    for ch in s:
                        if ch.isdigit():
                            num += ch
                        else:
                            break
                    if not num:
                        continue
                    v = int(num)
                    if v > 0:
                        best = v
                        break
                except Exception:
                    continue
    except Exception:
        return None
    if best:
        return best
    # Fallback: parse identify dumps and extract the dominant slot value.
    # Example:
    #   SASS control: identify dump ... slots=[269,0,0,0,0,0,0,0] unique_ids=1 ...
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if "SASS control: identify dump" not in line or "slots=[" not in line:
                    continue
                try:
                    s = line.split("slots=[", 1)[1]
                    s = s.split("]", 1)[0]
                    vals = []
                    for tok in s.split(","):
                        tok = tok.strip()
                        if not tok:
                            continue
                        if tok.isdigit():
                            vals.append(int(tok))
                    for v in vals:
                        if v != 0:
                            return v
                except Exception:
                    continue
    except Exception:
        return None
    return None


def _extract_image_id_from_log(log_path: str) -> Optional[int]:
    if not os.path.exists(log_path):
        return None
    # Example line (only when BPFTIME_CUDA_SASS_DETOUR_DEBUG=1):
    #   SASS control: identify dump ... image_id=0x1234abcd) ...
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if "SASS control: identify dump" not in line or "image_id=" not in line:
                    continue
                s = line.split("image_id=", 1)[1]
                num = ""
                for ch in s:
                    if ch in "0123456789abcdefABCDEFxX":
                        num += ch
                    else:
                        break
                if not num:
                    continue
                return int(num, 0)
    except Exception:
        return None
    return None


def _vllm_cmd(python: str, model: str, max_tokens: int, batch_size: int) -> Sequence[str]:
    script = str(pathlib.Path(__file__).resolve().parent.joinpath("vllm_inprocess_generate.py"))
    return [
        python,
        script,
        "--model",
        model,
        "--enforce-eager",
        "--max-tokens",
        str(max_tokens),
        "--batch-size",
        str(batch_size),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Bring up regcount=255 flashattention thread-map(device) on real vLLM (non-invasive)."
    )
    ap.add_argument(
        "--python",
        default=os.environ.get("BPFTIME_VLLM_PYTHON", ""),
        help="Python executable to run vLLM (default: BPFTIME_VLLM_PYTHON or .venv-vllm/bin/python if present).",
    )
    ap.add_argument("--model", default=os.environ.get("VLLM_MODEL", "Qwen/Qwen3-0.6B"))
    ap.add_argument("--kernel-filter", default=os.environ.get("BPFTIME_FLASH_FILTER", "flash"))
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--timeout-s", type=float, default=900.0)
    ap.add_argument(
        "--out-dir",
        default=os.environ.get("BPFTIME_FLASH_BRINGUP_DIR", "/tmp/bpftime-vllm-flash-reg255-threadmap"),
    )
    ap.add_argument(
        "--agent-so",
        default=os.environ.get(
            "BPFTIME_AGENT_SO",
            str(pathlib.Path("build/runtime/agent/libbpftime-agent.so").resolve()),
        ),
    )
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME", "/tmp/hf"))
    ap.add_argument(
        "--prefer-exit",
        action="store_true",
        default=True,
        help="Enable prefer-exit (recommended). Default: on.",
    )
    ap.add_argument("--no-prefer-exit", action="store_true", help="Disable prefer-exit.")
    ap.add_argument(
        "--pre-exit",
        action="store_true",
        default=False,
        help="Enable reg255 pre-exit detour (lower-risk transition point).",
    )
    ap.add_argument(
        "--use-dead-regs",
        action="store_true",
        default=True,
        help="Use dead-reg derived scratch regs for EXIT stub. Default: on.",
    )
    ap.add_argument("--no-use-dead-regs", action="store_true")
    ap.add_argument(
        "--dump-on-sync",
        action="store_true",
        default=True,
        help="Dump sampling on sync. Default: on.",
    )
    ap.add_argument("--no-dump-on-sync", action="store_true")
    ap.add_argument(
        "--max-records",
        type=int,
        default=1,
        help="BPFTIME_CUDA_SASS_SAMPLE_MAX_RECORDS (default: 1 for CTA0 clamp).",
    )
    ap.add_argument(
        "--stride4",
        action="store_true",
        default=True,
        help="Use stride4 stores (more robust). Default: on.",
    )
    ap.add_argument("--no-stride4", action="store_true")
    ap.add_argument(
        "--cta-clamp",
        action="store_true",
        default=True,
        help="Enable CTA clamp (CTA0-only when max_records=1). Default: on.",
    )
    ap.add_argument("--no-cta-clamp", action="store_true")
    ap.add_argument(
        "--density",
        default="tid0,lane0,warp0,full",
        help="Comma-separated density stages: tid0,lane0,warp0,full (default: all).",
    )
    ap.add_argument(
        "--threadmap-via-closure-upgrade",
        action="store_true",
        default=True,
        help="Run each thread-map stage in a single process using identify closure + on-demand upgrade (recommended).",
    )
    ap.add_argument(
        "--no-threadmap-via-closure-upgrade",
        action="store_true",
        help="Use the legacy two-run flow: identify first, then patch by (image_id, func_id).",
    )
    ap.add_argument(
        "--legacy-use-image-id",
        action="store_true",
        default=False,
        help="(Legacy flow only) Also set BPFTIME_CUDA_SASS_DETOUR_FILTER_IMAGE_ID from the identify run. "
        "Note: some JIT/link paths can produce different code objects across runs; this can reduce hit rate.",
    )
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    python = args.python
    if not python:
        cand = pathlib.Path(".venv-vllm/bin/python")
        if cand.exists():
            python = str(cand.resolve())
        else:
            python = sys.executable

    prefer_exit = bool(args.prefer_exit and not args.no_prefer_exit)
    use_dead_regs = bool(args.use_dead_regs and not args.no_use_dead_regs)
    dump_on_sync = bool(args.dump_on_sync and not args.no_dump_on_sync)
    stride4 = bool(args.stride4 and not args.no_stride4)
    cta_clamp = bool(args.cta_clamp and not args.no_cta_clamp)

    # Step 4.1: identify/closure to learn a precise target func_id for this run.
    ts = time.strftime("%Y%m%d-%H%M%S")
    identify_dump = str(out_dir.joinpath(f"identify-{ts}-%p.jsonl"))
    env0 = _env_common(args.agent_so, args.hf_home)
    identify_log = str(out_dir / f"identify-out-{ts}.log")
    env0.update(
        {
            "BPFTIME_CUDA_SASS_DETOUR": "1",
            "BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE": "1",
            "BPFTIME_CUDA_SASS_DETOUR_FILTER": args.kernel_filter,
            # IMPORTANT for strip/func_id scenarios:
            # leave BPFTIME_CUDA_SASS_SAMPLE_FILTER empty so sampling "follows
            # selection" (i.e., follow the detour's func_id/image_id mapping),
            # instead of relying on `.text.<name>` substring matches.
            "BPFTIME_CUDA_SASS_SAMPLE_FILTER": "",
            # Single-run closure helper for stripped raw-ELF SM120 code objects
            # (common in flashattention): allow a gated patch-all fallback on miss
            # so Identify can collect func_id(s) even when `.text.<name>` is absent.
            "BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE_PATCH_ALL_ON_MISS": "1",
            # Bring-up: enable debug logs so we can recover `target_func_id` even
            # if JSONL is clobbered by repeated identify windows.
            "BPFTIME_CUDA_SASS_DETOUR_DEBUG": "1",
            "BPFTIME_CUDA_SASS_SAMPLE": "1",
            "BPFTIME_CUDA_SASS_SAMPLE_MODE": "warp",
            "BPFTIME_CUDA_SASS_SAMPLE_MAX_RECORDS": "1",
            "BPFTIME_CUDA_SASS_SAMPLE_DUMP_PATH": identify_dump,
            # Bring-up: force a one-time sync+dump after a matching sampled launch.
            # This avoids relying on workload-specific sync points.
            "BPFTIME_CUDA_SASS_SAMPLE_SYNC_AFTER_LAUNCH": "1",
        }
    )
    if dump_on_sync:
        env0["BPFTIME_CUDA_SASS_SAMPLE_DUMP_ON_SYNC"] = "1"
    if prefer_exit:
        env0["BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_PREFER_EXIT"] = "1"
    if args.pre_exit:
        env0["BPFTIME_CUDA_SASS_DETOUR_REG255_PRE_EXIT"] = "1"
    if use_dead_regs:
        env0["BPFTIME_CUDA_SASS_DETOUR_EXIT_STUB_USE_DEAD_REGS"] = "1"

    p0 = _run(_vllm_cmd(python, args.model, args.max_tokens, args.batch_size), env0, args.timeout_s)
    pathlib.Path(identify_log).write_text(p0.stdout or "", encoding="utf-8")
    if p0.returncode != 0:
        print("identify run failed; see log:", identify_log, file=sys.stderr)
        return 2

    target_func_id = _extract_target_func_id(identify_dump)
    if not target_func_id:
        target_func_id = _extract_target_func_id_from_log(identify_log)
    if not target_func_id:
        print("failed to extract target_func_id from:", identify_dump, file=sys.stderr)
        return 3
    image_id = _extract_image_id_from_log(identify_log)
    if image_id is None:
        print("failed to extract image_id from:", identify_log, file=sys.stderr)
        return 3
    print("target_func_id=", target_func_id)
    print("image_id=0x%08x" % int(image_id))

    # Step 4.3: run thread-map(device) on the precise func_id with increasing density.
    use_closure_upgrade = bool(args.threadmap_via_closure_upgrade and not args.no_threadmap_via_closure_upgrade)
    want = {s.strip() for s in str(args.density).split(",") if s.strip()}
    stages = []
    if "tid0" in want:
        stages.append(("tid0", {"BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_LANE0_ONLY": "1", "BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_WARP0_ONLY": "1"}))
    if "lane0" in want:
        stages.append(("lane0", {"BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_LANE0_ONLY": "1"}))
    if "warp0" in want:
        stages.append(("warp0", {"BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_WARP0_ONLY": "1"}))
    if "full" in want:
        stages.append(("full", {}))

    for name, extra in stages:
        dump_path = str(out_dir.joinpath(f"threadmap-{name}-{ts}-%p.jsonl"))
        env = _env_common(args.agent_so, args.hf_home)
        if use_closure_upgrade:
            env.update(
                {
                    "BPFTIME_CUDA_SASS_DETOUR": "1",
                    # Keep identify-closure enabled for the thread-map stages:
                    # - many flashattention kernels are loaded via fatbinc/JIT paths
                    #   where we still rely on the closure-aware loader to obtain a
                    #   patchable SM120 ELF cubin.
                    # - but disable `cuLaunchKernel_on_demand_patch_all` to avoid
                    #   repeated patch attempts in-process (stability).
                    "BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE": "1",
                    # Enable the steady-state path: identify/force gives us a stable func_id;
                    # on-demand upgrade reloads a func-id-targeted module so we avoid patch-all
                    # and reduce cross-module interference.
                    "BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE_ON_DEMAND_UPGRADE": "1",
                    # We already ran a dedicated identify pass above and extracted a stable
                    # target_func_id for this bring-up. Force the closure state into
                    # "targeted" mode so we can skip per-launch Identify windows (more stable
                    # for reg255 EXIT stubs) and let the device-side EXIT stub consult the
                    # control header to decide whether to write.
                    "BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE_FORCE_TARGET_FUNC_ID": str(
                        int(target_func_id)
                    ),
                    # Avoid cross-image func_id collisions during forced-target bring-up:
                    # func_id values are only meaningful within a code object, so constrain
                    # load-time patching to the image_id we observed in the identify pass.
                    "BPFTIME_CUDA_SASS_DETOUR_FILTER_IMAGE_ID": "0x%08x" % int(image_id),
                    # Avoid load-time patch-all fallback during bring-up stages; it is too
                    # invasive on large libraries. We rely on on-demand upgrade instead.
                    "BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE_NO_PATCH_ALL_ON_MISS": "1",
                    "BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE_NO_ON_DEMAND_PATCH_ALL": "1",
                    "BPFTIME_CUDA_SASS_DETOUR_FILTER": args.kernel_filter,
                    "BPFTIME_CUDA_SASS_DETOUR_DEBUG": env.get("BPFTIME_CUDA_SASS_DETOUR_DEBUG", "1"),
                    "BPFTIME_CUDA_SASS_SAMPLE": "1",
                    # For strip/func_id selection, keep SAMPLE_FILTER empty so
                    # sampling "follows selection" instead of `.text.<name>` matches.
                    "BPFTIME_CUDA_SASS_SAMPLE_FILTER": "",
                    "BPFTIME_CUDA_SASS_SAMPLE_MODE": "thread",
                    "BPFTIME_CUDA_SASS_SAMPLE_MAX_RECORDS": str(max(1, int(args.max_records))),
                    "BPFTIME_CUDA_SASS_SAMPLE_DUMP_PATH": dump_path,
                    "BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE": "1",
                    # Bring-up: allow the no-regcount EXIT stub even when vendor code objects
                    # don't carry readable `.nv.info/.nv.merc` regcount tables.
                    "BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_EXIT_STUB_ON_NO_REGCOUNT": "1",
                    # Ensure we get a dump on real workloads with long async chains.
                    "BPFTIME_CUDA_SASS_SAMPLE_DUMP_ON_SYNC": "1",
                }
            )
        else:
            env.update(
                {
                    "BPFTIME_CUDA_SASS_DETOUR": "1",
                    # For stripped SM120 flashattention, selection is driven by (image_id, func_id),
                    # not by `.text.<name>` substring matches.
                    "BPFTIME_CUDA_SASS_DETOUR_FILTER": "",
                    "BPFTIME_CUDA_SASS_DETOUR_FILTER_FUNC_IDS": str(int(target_func_id)),
                    "BPFTIME_CUDA_SASS_SAMPLE": "1",
                    # For strip/func_id selection, keep SAMPLE_FILTER empty so
                    # sampling "follows selection" instead of `.text.<name>` matches.
                    "BPFTIME_CUDA_SASS_SAMPLE_FILTER": "",
                    "BPFTIME_CUDA_SASS_SAMPLE_MODE": "thread",
                    "BPFTIME_CUDA_SASS_SAMPLE_MAX_RECORDS": str(max(1, int(args.max_records))),
                    "BPFTIME_CUDA_SASS_SAMPLE_DUMP_PATH": dump_path,
                    "BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE": "1",
                    # Bring-up: allow the no-regcount EXIT stub even when vendor code objects
                    # don't carry readable `.nv.info/.nv.merc` regcount tables.
                    "BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_EXIT_STUB_ON_NO_REGCOUNT": "1",
                    # Bring-up: ensure we get a dump even if the workload doesn't
                    # hit a convenient sync point immediately.
                    "BPFTIME_CUDA_SASS_SAMPLE_SYNC_AFTER_LAUNCH": "1",
                    # Apply cached func_id sets immediately to raw-ELF code objects.
                    # This reduces dependence on “fatbinc must appear first”.
                    "BPFTIME_CUDA_SASS_DETOUR_USE_CACHED_FUNC_IDS_FOR_RAW_ELF": "1",
                }
            )
            if args.legacy_use_image_id:
                env["BPFTIME_CUDA_SASS_DETOUR_FILTER_IMAGE_ID"] = "0x%08x" % int(image_id)
        if stride4:
            env["BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_STRIDE4"] = "1"
        if cta_clamp:
            env["BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_CTA_CLAMP"] = "1"
        if dump_on_sync:
            env["BPFTIME_CUDA_SASS_SAMPLE_DUMP_ON_SYNC"] = "1"
        if prefer_exit:
            env["BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_PREFER_EXIT"] = "1"
        if args.pre_exit:
            env["BPFTIME_CUDA_SASS_DETOUR_REG255_PRE_EXIT"] = "1"
        if use_dead_regs:
            env["BPFTIME_CUDA_SASS_DETOUR_EXIT_STUB_USE_DEAD_REGS"] = "1"

        env.update(extra)

        p = _run(_vllm_cmd(python, args.model, args.max_tokens, args.batch_size), env, args.timeout_s)
        (out_dir / f"threadmap-{name}-out-{ts}.log").write_text(p.stdout or "", encoding="utf-8")
        if p.returncode != 0:
            print(f"stage {name} failed (rc={p.returncode}); see log:", out_dir / f"threadmap-{name}-out-{ts}.log", file=sys.stderr)
            return 4
        nrec = _count_thread_records(dump_path)
        print(f"stage {name}: thread_records={nrec} dump={dump_path}")
        if nrec == 0:
            return 5

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
