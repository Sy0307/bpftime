# vLLM observability test (OpenAI server + /metrics)

This is a more realistic vLLM validation than a single `LLM.generate()` call:

- starts the **real** vLLM OpenAI-compatible server
- sends a completion request
- scrapes vLLM Prometheus metrics at `/metrics`
- (when running under bpftime) summarizes bpftime’s CUDA JSONL trace

## Setup (suggested venv)

```bash
python3 -m venv /tmp/vllm-venv
source /tmp/vllm-venv/bin/activate

pip install -U pip wheel setuptools
pip install --index-url https://download.pytorch.org/whl/cu124 torch --extra-index-url https://pypi.org/simple
pip install vllm
```

## Run without bpftime

```bash
source /tmp/vllm-venv/bin/activate
HF_HOME=/tmp/hf \
python benchmark/gpu/vllm_observability/vllm_openai_server_test.py
```

## Run a more realistic server scenario (load test)

This runs multiple concurrent requests (optionally streaming) and reports p50/p90/p99 latency,
TTFT percentiles (if streaming), `/metrics` deltas, and bpftime trace summary.

```bash
source /tmp/vllm-venv/bin/activate
HF_HOME=/tmp/hf \
VLLM_CONCURRENCY=4 \
VLLM_TOTAL_REQUESTS=20 \
VLLM_STREAM=1 \
python benchmark/gpu/vllm_observability/vllm_openai_server_load_test.py
```

## In-process vLLM (single-process smoke)

This runs real vLLM inference in-process (no OpenAI server). It is useful when you
want the process to exit cleanly so bpftime can dump sampling results on exit.

```bash
source /tmp/vllm-venv/bin/activate
HF_HOME=/tmp/hf \
python benchmark/gpu/vllm_observability/vllm_inprocess_generate.py --model Qwen/Qwen3-0.6B
```

Tips (to cover more KV/cache-related paths without modifying vLLM):
- Use `--enforce-eager` to avoid CUDA graphs and make tracing/sampling timing more predictable.
- Use `--batch-size` to increase concurrent prompts in one `generate()` call.
- Use `--max-model-len` + `--kv-cache-memory-bytes` to shrink KV capacity (can trigger different cache pressure behavior).

## Flashattention (strip/JIT): single-run identify closure (no vLLM changes)

Some flashattention kernels are loaded via `cuLibrary*` paths and can be stripped or JIT-produced,
so `kernel_name` filtering is not enough to locate a `.text.<name>` section in SM120 cubins.

Enable identify-closure so bpftime can “learn” the target `func_id` and guarantee **at least 1 record**
in the same run:

```bash
source /tmp/vllm-venv/bin/activate
HF_HOME=/tmp/hf \
BPFTIME_CUDA_SASS_DETOUR=1 \
BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE=1 \
BPFTIME_CUDA_SASS_DETOUR_FILTER=flash \
BPFTIME_CUDA_SASS_SAMPLE=1 \
BPFTIME_CUDA_SASS_SAMPLE_MODE=warp \
BPFTIME_CUDA_SASS_SAMPLE_MAX_RECORDS=1 \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_ON_SYNC=1 \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_PATH=/tmp/bpftime-vllm-flash-sync.jsonl \
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
python benchmark/gpu/vllm_observability/vllm_inprocess_generate.py \
  --model Qwen/Qwen3-0.6B --max-tokens 1 --enforce-eager
```

## Flashattention regcount=255: thread-map(device) bring-up (SM120, non-invasive)

This script runs real vLLM in-process, uses identify-closure to learn `func_id`, then brings up
`thread-map(device)` density gates step-by-step (tid0 → lane0 → warp0 → full).

```bash
python3 benchmark/gpu/vllm_observability/vllm_flash_reg255_threadmap_bringup.py \
  --python .venv-vllm/bin/python \
  --kernel-filter flash_fwd_splitkv_kernel \
  --max-tokens 2 --batch-size 1 \
  --out-dir /tmp/bpftime-vllm-flash-reg255-threadmap
```

Notes:
- Default is “single-run closure + on-demand upgrade” (recommended for strip/JIT paths).
- Legacy two-run flow: add `--no-threadmap-via-closure-upgrade`.
- In legacy mode, `--legacy-use-image-id` also sets `BPFTIME_CUDA_SASS_DETOUR_FILTER_IMAGE_ID`, but some JIT/link paths may emit different code objects across runs (hit rate can drop).

Expected:
- log: `SASS control: identify dump ... slots=[...]`
- dump: `/tmp/bpftime-vllm-flash-sync.jsonl` has 2 lines (meta + `bpftime_sass_warp`)

Knobs (advanced):
- Disable “closure default patch-all fallback”: `BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE_NO_PATCH_ALL_FALLBACK_DEFAULT=1`
- Disable “closure default JIT-link PTX→cubin”: `BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE_NO_JITLINK_PTX_DEFAULT=1`

## KV cache kernel: trace → pick target → minimal SASS sampling

The KV cache update path in vLLM often shows up as a kernel similar to:
`vllm::reshape_and_cache_flash_kernel<...>`.

### 1) Collect a kernel launch trace

```bash
source /tmp/vllm-venv/bin/activate

VLLM_ENABLE_V1_MULTIPROCESSING=0 \
BPFTIME_ALLOW_NO_SHM=1 \
BPFTIME_LOG_OUTPUT=console \
BPFTIME_CUDA_TRACE_PATH=/tmp/bpftime-vllm-trace-kv.jsonl \
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
HF_HOME=/tmp/hf \
python benchmark/gpu/vllm_observability/vllm_inprocess_generate.py --model Qwen/Qwen3-0.6B --max-tokens 1
```

Summarize and filter likely KV-related kernels:

```bash
python benchmark/gpu/vllm_observability/bpftime_trace_summarize.py \
  --trace /tmp/bpftime-vllm-trace-kv.jsonl \
  --grep 'cache|kv|paged|reshape|swap|block' \
  --top 30
```

Pick vLLM/KV-related kernels (top-N by count) from a trace:

## JIT-link PTX/IR route (no dead-reg hunting): sampler-bound thread-map

Some vLLM/PyTorch stacks emit kernels via NVRTC/Triton and link them through `cuLink`/`nvJitLink`.
If the loaded code object still has PTX/NVVM available, bpftime can inject thread-map logic at the
PTX level (toolchain handles regs/spills/layout) and bind its output to the normal bpftime sampler
buffer, so you still get standard JSONL (`bpftime_sass_thread`).

Workflow:
1) Use the launch trace to find a stable kernel-name substring that actually appears in your run.
2) Set `BPFTIME_CUDA_SASS_DETOUR_FILTER=<that_substring>`.
3) Enable JIT-link PTX injection + thread-map sampler.

Example (lane0-only + CTA0 clamp, dump-on-sync):

```bash
source /tmp/vllm-venv/bin/activate
HF_HOME=/tmp/hf \
BPFTIME_CUDA_SASS_DETOUR=1 \
BPFTIME_CUDA_SASS_DETOUR_FILTER='<kernel_substring>' \
BPFTIME_CUDA_SASS_DETOUR_JITLINK_PTX=1 \
BPFTIME_CUDA_SASS_DETOUR_JITLINK_PTX_THREADMAP=1 \
BPFTIME_CUDA_SASS_SAMPLE=1 \
BPFTIME_CUDA_SASS_SAMPLE_MODE=thread \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE=1 \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_STRIDE4=1 \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_LANE0_ONLY=1 \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_CTA_CLAMP=1 \
BPFTIME_CUDA_SASS_SAMPLE_MAX_RECORDS=1 \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_ON_SYNC=1 \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_PATH=/tmp/bpftime-vllm-ptx-threadmap.jsonl \
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
python benchmark/gpu/vllm_observability/vllm_inprocess_generate.py --model Qwen/Qwen3-0.6B --max-tokens 1 --enforce-eager
```

Notes:
- This only helps for kernels that actually go through JIT-link with PTX/NVVM available; cubin-only
  flashattention on sm_120 still needs the SASS route.
- Implementation/progress is tracked in `sass_update.md`.

```bash
python benchmark/gpu/vllm_observability/bpftime_trace_pick_kernels.py \
  --trace /tmp/bpftime-vllm-trace-kv.jsonl \
  --only-vllm --top 20
```

### 1.1) Expand KV coverage (workload matrix → more kernels)

Some KV-related kernels (e.g. paged attention variants, swap/copy paths) only appear under
specific workload / memory-pressure settings. To discover a broader set of KV/attn kernels
**without modifying vLLM**, run a small workload matrix and auto-pick candidates per trace:

```bash
source /tmp/vllm-venv/bin/activate

python benchmark/gpu/vllm_observability/vllm_kv_kernel_coverage.py \
  --model Qwen/Qwen3-0.6B \
  --out-dir /tmp/bpftime-vllm-kv-coverage \
  --top 40 --only-vllm
```

Profiles (use `--profiles a,b,c` to select): `baseline_decode`, `batch_decode`, `long_prefill_short_decode`, `kv_pressure`, `cpu_offload`.

Output:
- traces: `/tmp/bpftime-vllm-kv-coverage/trace-<profile>-<ts>.jsonl`
- report: `/tmp/bpftime-vllm-kv-coverage/kv_coverage_report.json` (picked kernels + `memcpy/memset` summaries per run)

Once you pick a target kernel substring, you can bring it up quickly on real vLLM:

```bash
# timing (GPU duration), filtered to one kernel
python benchmark/gpu/vllm_observability/vllm_kv_kernel_bringup.py \
  --mode timing --kernel reshape_and_cache_flash_kernel \
  --model Qwen/Qwen3-0.6B --max-tokens 32

# marker (pc-marker/hotspot), clamp+lane0-only by default
python benchmark/gpu/vllm_observability/vllm_kv_kernel_bringup.py \
  --mode marker --kernel reshape_and_cache_flash_kernel \
  --model Qwen/Qwen3-0.6B --max-tokens 32

# thread-map(device), lane0-only bring-up
python benchmark/gpu/vllm_observability/vllm_kv_kernel_bringup.py \
  --mode thread_lane0 --kernel reshape_and_cache_flash_kernel \
  --model Qwen/Qwen3-0.6B --max-tokens 32
```

### 1.2) Expand KV coverage via real server load (paged/offload/swap best-effort)

Some kernels and paths are easier to trigger with a real OpenAI-compatible server under load.
This runner starts the vLLM OpenAI server and collects a bpftime trace per profile:

```bash
source /tmp/vllm-venv/bin/activate

python benchmark/gpu/vllm_observability/vllm_kv_server_coverage.py \
  --model Qwen/Qwen3-0.6B \
  --out-dir /tmp/bpftime-vllm-kv-server-coverage
```

Output:
- report: `/tmp/bpftime-vllm-kv-server-coverage/kv_server_coverage_report.json`
- per-run trace/log paths are embedded in the report.

Notes:
- **swap/copy path** is usually visible as large `memcpy` volumes (especially `DtoHAsync`) in `mem_ops` for the
  `server_kv_offload_pressure` profile (this is non-invasive: no SASS needed).
- You may not see a kernel name containing `paged_attention` on CUDA even though vLLM uses a paged KV layout;
  attention compute often appears as `flash_*` kernels (FLASH_ATTN path) or GEMM/ATen kernels (non-flash backend).

### 1.3) Pick “KV-related kernel substrings” from a report

For non-invasive bring-up, we use **kernel substring filters** (for timing/sampling) and **memcpy kind/bytes**
(for offload/swap-like signals). You can extract recommended target substrings from a coverage report:

```bash
python benchmark/gpu/vllm_observability/bpftime_kv_targets_from_report.py \
  --report /tmp/bpftime-vllm-kv-server-coverage/kv_server_coverage_report.json --top 5
```

For a typical vLLM run, KV-related targets usually include:
- `reshape_and_cache_flash_kernel` (KV write/update)
- `flash_fwd_splitkv_kernel` / `flash_fwd_splitkv_combine_kernel` (attention read/compute)
- `memcpy` volumes (e.g. `DtoHAsync/HtoDAsync`) as the best non-invasive “swap/offload/copy” evidence

## Roadmap (non-invasive)

This section is a concrete plan for what’s next in bpftime GPU observability for real vLLM workloads,
without modifying vLLM.

### A) GPU compute layer: expand bring-up to top-N hot kernels

Goal: move beyond “KV/flash only” and cover vLLM’s real hotspots (norm/rotary + cutlass/cublasLt + ATen).

Plan:
1) Run a workload profile (in-process or server) and produce a coverage report.
2) Pick top-N targets:
   - By count: `bpftime_trace_pick_kernels.py --sort count`
   - If kernel timing is enabled for a subset: `bpftime_trace_pick_kernels.py --sort gpu_ms`
3) For each target substring, apply a stable “template” (env preset):
   - `kernel_timing`: enable + sampling (e.g. `SAMPLE_EVERY=10`)
   - `pc-marker`: start with offsets `0,0x10` and clamp/limit
   - `thread-map(device)`: only for selected kernels; start `lane0-only` + `CTA_CLAMP` + small `MAX_RECORDS`

Practical tip: time multiple hotspot kernels in a single run (best-effort):

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
BPFTIME_ALLOW_NO_SHM=1 \
BPFTIME_LOG_OUTPUT=console \
BPFTIME_CUDA_TRACE_PATH=/tmp/bpftime-vllm-hot-timing.jsonl \
BPFTIME_CUDA_KERNEL_TIMING=1 \
BPFTIME_CUDA_KERNEL_TIMING_FILTERS='reshape_and_cache_flash_kernel,flash_fwd_splitkv_kernel,flash_fwd_splitkv_combine_kernel,rms_norm_kernel,fused_add_rms_norm_kernel,rotary_embedding_kernel,act_and_mul_kernel,cutlass_80_wmma_tensorop,cublasLt,unrolled_elementwise_kernel' \
BPFTIME_CUDA_KERNEL_TIMING_SAMPLE_EVERY=50 \
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
HF_HOME=/tmp/hf \
python benchmark/gpu/vllm_observability/vllm_inprocess_generate.py \
  --model Qwen/Qwen3-0.6B --enforce-eager --max-tokens 32 --batch-size 4

python benchmark/gpu/vllm_observability/bpftime_trace_pick_kernels.py \
  --trace /tmp/bpftime-vllm-hot-timing.jsonl --include '' --exclude '' --sort gpu_ms --top 20
```

### A.1) Systematic thread-map(device) validation (rate-limit templates)

`thread-map(device)` is riskier than timing/marker because it injects a longer SASS stub and writes per-thread data.
For real vLLM bring-up, we validate kernel-by-kernel and keep density limited.

Sweep runner (each run spawns a fresh in-process vLLM):

```bash
source /tmp/vllm-venv/bin/activate
HF_HOME=/tmp/hf \
python benchmark/gpu/vllm_observability/vllm_threadmap_sweep.py \
  --model Qwen/Qwen3-0.6B \
  --kernels 'reshape_and_cache_flash_kernel,rms_norm_kernel,fused_add_rms_norm_kernel,rotary_embedding_kernel,act_and_mul_kernel,cutlass_80_wmma_tensorop,unrolled_elementwise_kernel' \
  --modes lane0,warp0 \
  --max-records 1 \
  --out-dir /tmp/bpftime-vllm-threadmap-sweep
```

Output:
- report: `/tmp/bpftime-vllm-threadmap-sweep/threadmap_sweep_report.json`
- dumps/analyzes per attempt: `sass-thread-*.jsonl` + `analyze-*.json`

**Stable rate-limit templates (recommended)**
- `lane0-only` (lowest overhead): `BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_LANE0_ONLY=1`
- `warp0-only` (moderate): `BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_WARP0_ONLY=1`
- Always keep: `BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_CTA_CLAMP=1` + small `BPFTIME_CUDA_SASS_SAMPLE_MAX_RECORDS` first.

**Known unstable class (needs extra work)**
- Some ATen/torch kernels (e.g. `unrolled_elementwise_kernel`) can crash under entry detour (CUDA illegal instruction).
  The sweep runner will retry once with `BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_PREFER_EXIT=1` as a stability knob.
  Current prefer-exit implementation patches multiple EXIT sites (including `@P0 EXIT`) and uses UR-base stores by default to avoid cublas instability (details in `vllm_issues.md`).

### B) Memory/movement layer: stronger offline correlation (memcpy ↔ kernel windows)

Goal: make “offload/swap/copy” and “which phase it correlates with” visible without vLLM changes.

Plan:
1) Use `mem_ops` in coverage reports to confirm that large `DtoHAsync/HtoDAsync` volumes happen in a given profile
   (`server_kv_offload_pressure` is a good stress profile).
2) Use `bpftime_trace_mem_analyze.py` to correlate memcpy/memset with the nearest kernel windows (per-stream segments):

```bash
python benchmark/gpu/vllm_observability/bpftime_trace_mem_analyze.py \
  --trace /tmp/bpftime-vllm-kv-server-coverage/trace-...jsonl --min-bytes $((16*1024*1024))
```

To focus on offload/copy-like traffic, filter segments that contain `DtoHAsync` and exceed a threshold:

```bash
python benchmark/gpu/vllm_observability/bpftime_trace_mem_analyze.py \
  --trace /tmp/bpftime-vllm-kv-server-coverage/trace-...jsonl \
  --require-memcpy-kind DtoHAsync --require-memcpy-bytes $((1*1024*1024)) \
  --min-bytes $((1*1024*1024)) --top 5
```

Future enhancement (still non-invasive): enable `BPFTIME_CUDA_TRACE_ARGS=1` for selected kernels and use pointer
fingerprints to cluster memcpy events into “likely KV” vs “likely weights/activations” buckets.

### C) Library/runtime layer (record only; not implemented yet)

Potential future hooks (still non-invasive):
- `cublasLt` API calls: matmul shapes (m/n/k), algorithm id, workspace bytes
- `nccl` API calls: collective type, bytes, stream, duration
- `flashinfer` / other attention backends: API-level semantics if available

### 2) Minimal SASS sampling on the chosen KV kernel

Start with the stable `thread` mode in **host-expanded** form (per-warp write, host expands to per-thread):

```bash
source /tmp/vllm-venv/bin/activate

VLLM_ENABLE_V1_MULTIPROCESSING=0 \
BPFTIME_ALLOW_NO_SHM=1 \
BPFTIME_LOG_OUTPUT=console \
BPFTIME_CUDA_SASS_DETOUR=1 \
BPFTIME_CUDA_SASS_SAMPLE=1 \
BPFTIME_CUDA_SASS_SAMPLE_MODE=thread \
BPFTIME_CUDA_SASS_DETOUR_FILTER=reshape_and_cache_flash_kernel \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE=0 \
BPFTIME_CUDA_SASS_SAMPLE_MAX_RECORDS=1 \
BPFTIME_CUDA_SASS_SAMPLE_SYNC_AFTER_LAUNCH=1 \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_ON_SYNC=1 \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_PATH=/tmp/bpftime-vllm-kv-threadmap-warp.jsonl \
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
HF_HOME=/tmp/hf \
python benchmark/gpu/vllm_observability/vllm_inprocess_generate.py --model Qwen/Qwen3-0.6B --max-tokens 1
```

Notes:
- In this mode, each CTA writes **32 warp slots**, and the host expands them to **1024 thread records** (32 warps × 32 lanes).
- For accuracy on kernels whose `blockDim.x` is not 1024, switch to device-per-thread writes later
  (`BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE=1`) with density gating and CTA clamp.

### 3) KV kernel: device-per-thread bring-up (lane0-only → warp0-only → full)

This avoids the host-expanded assumption and makes `tid_x/warp/lane` accurate for kernels whose
`blockDim.x` is not 1024.

Start with the most conservative gate (lane0-only), and clamp to a single CTA:

```bash
source /tmp/vllm-venv/bin/activate

VLLM_ENABLE_V1_MULTIPROCESSING=0 \
BPFTIME_ALLOW_NO_SHM=1 \
BPFTIME_LOG_OUTPUT=console \
BPFTIME_CUDA_SASS_DETOUR=1 \
BPFTIME_CUDA_SASS_SAMPLE=1 \
BPFTIME_CUDA_SASS_SAMPLE_MODE=thread \
BPFTIME_CUDA_SASS_DETOUR_FILTER=reshape_and_cache_flash_kernel \
BPFTIME_CUDA_SASS_SAMPLE_FILTER=reshape_and_cache_flash_kernel \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE=1 \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_STRIDE4=1 \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_CTA_CLAMP=1 \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_LANE0_ONLY=1 \
BPFTIME_CUDA_SASS_SAMPLE_MAX_RECORDS=1 \
BPFTIME_CUDA_SASS_SAMPLE_SYNC_AFTER_LAUNCH=1 \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_ON_SYNC=1 \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_PATH=/tmp/bpftime-vllm-kv-device-lane0.jsonl \
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
HF_HOME=/tmp/hf \
python benchmark/gpu/vllm_observability/vllm_inprocess_generate.py --model Qwen/Qwen3-0.6B --max-tokens 1
```

Then increase density:
- warp0-only: `BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_WARP0_ONLY=1` and `LANE0_ONLY=0`
- full: `WARP0_ONLY=0` and `LANE0_ONLY=0`

Finally, expand CTAs by increasing `BPFTIME_CUDA_SASS_SAMPLE_MAX_RECORDS=1 → 4`
(keep `CTA_CLAMP=1` until stable).

#### CTA expansion + output limiting (recommended sequence)

When you increase `BPFTIME_CUDA_SASS_SAMPLE_MAX_RECORDS`, keep `BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_CTA_CLAMP=1`
to avoid modulo collisions. Use density gates to limit write volume:

- lane0-only + many CTAs: records ≈ `max_records * warp_count` (cheap)
- warp0-only + many CTAs: records ≈ `max_records * 32` (moderate)
- full + few CTAs: records ≈ `max_records * blockDim.x` (expensive)

Quickly analyze a dump (ctaid distribution + inferred blockDim.x + SMID histogram):

```bash
python benchmark/gpu/vllm_observability/bpftime_sass_thread_analyze.py \
  --dump /tmp/bpftime-vllm-kv-device-full.jsonl
```

Note: if you clamp to `MAX_RECORDS=1`, it is normal to see only `smid_lo8=0` (CTA0 may be scheduled on SM0).
Increase CTAs (e.g. `MAX_RECORDS=64`) to observe a distribution of SMIDs.

## Run with bpftime (kernel launch tracing)

```bash
source /tmp/vllm-venv/bin/activate

BPFTIME_ALLOW_NO_SHM=1 \
BPFTIME_LOG_OUTPUT=console \
BPFTIME_CUDA_TRACE_PATH=/tmp/bpftime-vllm-trace.jsonl \
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
HF_HOME=/tmp/hf \
python benchmark/gpu/vllm_observability/vllm_openai_server_test.py
```

Expected:
- `/tmp/bpftime-vllm-trace.jsonl` exists and is non-empty
- script prints a JSON summary including `metrics_delta`, `top_kernels`, and `trace_type_counts`
- `trace_type_counts` should include `launch` and usually also `memcpy`/`memset`/`alloc`/`free`/`sync`

### Optional: kernel GPU duration (non-intrusive, via CUDA events)

This measures **GPU time** (not CPU launch time) for a filtered subset of kernels by recording
two CUDA events around the launch and flushing durations at synchronization points.

It emits additional JSONL records of type `kernel_timing` into the same trace file.

```bash
source /tmp/vllm-venv/bin/activate

VLLM_ENABLE_V1_MULTIPROCESSING=0 \
BPFTIME_LOG_OUTPUT=console \
BPFTIME_CUDA_TRACE_PATH=/tmp/bpftime-vllm-trace-kv-timing.jsonl \
BPFTIME_CUDA_KERNEL_TIMING=1 \
BPFTIME_CUDA_KERNEL_TIMING_FILTER=reshape_and_cache_flash_kernel \
BPFTIME_CUDA_KERNEL_TIMING_SAMPLE_EVERY=10 \
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
HF_HOME=/tmp/hf \
python benchmark/gpu/vllm_observability/vllm_inprocess_generate.py --model Qwen/Qwen3-0.6B --max-tokens 1

python benchmark/gpu/vllm_observability/bpftime_trace_summarize.py \
  --trace /tmp/bpftime-vllm-trace-kv-timing.jsonl \
  --grep reshape_and_cache_flash_kernel --top 5
```

Notes:
- Timing is disabled automatically during CUDA graph capture.
- `kernel_timing` records are flushed on `cuStreamSynchronize` / `cuCtxSynchronize`.
- If you want more complete timing coverage, run `vllm_inprocess_generate.py` with `--enforce-eager` to avoid CUDA graphs.

### Offline: attribute sync waits to kernels (best-effort)

When `kernel_timing` is enabled, you can attribute each `cuStreamSynchronize` / `cuCtxSynchronize`
CPU wait to the set of kernel launches that happened since the previous sync.

```bash
python benchmark/gpu/vllm_observability/bpftime_trace_analyze.py \
  --trace /tmp/bpftime-vllm-trace-kv-timing.jsonl \
  --kernel-filter 'reshape_and_cache_flash_kernel' \
  --top 5
```

### Optional: launch args fingerprint (best-effort, filtered)

For driver calls using `kernelParams`, bpftime can compute a best-effort `args_fingerprint`
by safely sampling a few u64 words from the launch argument values (to group “same-name kernels”
by call shape/pointer patterns).

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
BPFTIME_LOG_OUTPUT=console \
BPFTIME_CUDA_TRACE_PATH=/tmp/bpftime-vllm-trace-args.jsonl \
BPFTIME_CUDA_TRACE_ARGS=1 \
BPFTIME_CUDA_TRACE_ARGS_FILTER=reshape_and_cache_flash_kernel \
BPFTIME_CUDA_TRACE_ARGS_MAX=6 \
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
HF_HOME=/tmp/hf \
python benchmark/gpu/vllm_observability/vllm_inprocess_generate.py --model Qwen/Qwen3-0.6B --max-tokens 1
```

### With bpftime + load test

```bash
source /tmp/vllm-venv/bin/activate

BPFTIME_ALLOW_NO_SHM=1 \
BPFTIME_LOG_OUTPUT=console \
BPFTIME_CUDA_TRACE_PATH=/tmp/bpftime-vllm-trace.jsonl \
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
HF_HOME=/tmp/hf \
VLLM_CONCURRENCY=4 \
VLLM_TOTAL_REQUESTS=20 \
VLLM_STREAM=1 \
python benchmark/gpu/vllm_observability/vllm_openai_server_load_test.py
```

## Run with bpftime (experimental SASS detour)

When the workload loads **cubin-only** images (no PTX), you can enable a minimal
SASS-level detour to validate that CUBINs are patchable and (optionally) run a
tiny SASS sampler inside kernels:

```bash
source /tmp/vllm-venv/bin/activate

BPFTIME_ALLOW_NO_SHM=1 \
BPFTIME_LOG_OUTPUT=console \
BPFTIME_CUDA_TRACE_PATH=/tmp/bpftime-vllm-trace.jsonl \
BPFTIME_CUDA_SASS_DETOUR=1 \
BPFTIME_CUDA_SASS_DETOUR_DEBUG=1 \
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
HF_HOME=/tmp/hf \
python benchmark/gpu/vllm_observability/vllm_openai_server_test.py
```

Notes:
- This currently targets `sm_120` (CUDA 12 + cc 12.0).
- Use `BPFTIME_CUDA_SASS_DETOUR_DUMP_DIR=/tmp/bpftime-sass` to dump detoured CUBINs
  for offline `cuobjdump` inspection.

### Enable SASS sampling (inside cubin-only kernels)

Sampling modes (SM120-only for now):

- `smid_bitmap`: marks which SMIDs executed the selected kernel (coarse)
- `cta` / `records`: CTA→SMID map (`buffer[ctaid.x] = smid_raw`) for the selected kernel
- `warp`: per-CTA per-warp SMID slots (writes `smid_lo8` into `slots[ctaid.x*32 + warp_id]`)
- `thread`: defaults to **host-expanded** per-thread records by expanding each sampled warp slot into
  32 lanes (`tid_x = warp_id*32 + lane_id`); optionally supports experimental device-side per-thread writes
  via `BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE=1`
- `marker`: pc-marker / hotspot events: writes per-hit records into a ring buffer
  (see below; requires `BPFTIME_CUDA_SASS_MARKER_OFFSETS` to select points).

Example (recommended to start with a single kernel filter):

```bash
source /tmp/vllm-venv/bin/activate

BPFTIME_ALLOW_NO_SHM=1 \
BPFTIME_LOG_OUTPUT=console \
BPFTIME_CUDA_SASS_DETOUR=1 \
BPFTIME_CUDA_SASS_DETOUR_FILTER=cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_tn_align8 \
BPFTIME_CUDA_SASS_DETOUR_DUMP_DIR=/tmp/bpftime-sass-qwen3 \
BPFTIME_CUDA_SASS_SAMPLE=1 \
BPFTIME_CUDA_SASS_SAMPLE_MODE=smid_bitmap \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_PATH=/tmp/bpftime-sass-samples.jsonl \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_ON_SYNC=1 \
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
HF_HOME=/tmp/hf-qwen3 \
VLLM_MODEL=Qwen/Qwen3-0.6B \
VLLM_CONCURRENCY=1 \
VLLM_TOTAL_REQUESTS=1 \
VLLM_STREAM=0 \
python benchmark/gpu/vllm_observability/vllm_openai_server_load_test.py
```
```

For CTA→SMID:

```bash
BPFTIME_CUDA_SASS_SAMPLE_MODE=cta \
BPFTIME_CUDA_SASS_SAMPLE_MAX_RECORDS=65536 \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_PATH=/tmp/bpftime-sass-ctaid-smid.jsonl \
```

## pc-marker / hotspot (SM120)

This mode inserts a tiny marker stub at one or more offsets within a matched `.text.*`
section and emits a record per hit:

`{seq, marker_off, tag, smid_raw, ctaid_x, tid_x}`.

Bring-up example (KV kernel, clamp to CTA0, lane0-only, two points at `0x0` and `0x10`):

```bash
source /tmp/vllm-venv/bin/activate

VLLM_ENABLE_V1_MULTIPROCESSING=0 \
BPFTIME_ALLOW_NO_SHM=1 \
BPFTIME_LOG_OUTPUT=console \
BPFTIME_CUDA_SASS_DETOUR=1 \
BPFTIME_CUDA_SASS_SAMPLE=1 \
BPFTIME_CUDA_SASS_SAMPLE_MODE=marker \
BPFTIME_CUDA_SASS_DETOUR_FILTER=reshape_and_cache_flash_kernel \
BPFTIME_CUDA_SASS_SAMPLE_FILTER=reshape_and_cache_flash_kernel \
BPFTIME_CUDA_SASS_SAMPLE_MAX_RECORDS=1 \
BPFTIME_CUDA_SASS_MARKER_RING_ENTRIES=2048 \
BPFTIME_CUDA_SASS_MARKER_OFFSETS=0,0x10 \
BPFTIME_CUDA_SASS_SAMPLE_SYNC_AFTER_LAUNCH=1 \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_ON_SYNC=1 \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_PATH=/tmp/bpftime-vllm-kv-marker.jsonl \
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
HF_HOME=/tmp/hf \
python benchmark/gpu/vllm_observability/vllm_inprocess_generate.py --model Qwen/Qwen3-0.6B --max-tokens 1
```

Analyze hotspot counts by `marker_off`:

```bash
python benchmark/gpu/vllm_observability/bpftime_sass_marker_analyze.py \
  --dump /tmp/bpftime-vllm-kv-marker.jsonl --top 20
```

For SM/Warp mapping:

```bash
BPFTIME_CUDA_SASS_SAMPLE_MODE=warp \
BPFTIME_CUDA_SASS_SAMPLE_MAX_RECORDS=4096 \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_PATH=/tmp/bpftime-sass-warp.jsonl \
```

For SM/Thread mapping (host-expanded from warp slots):

```bash
BPFTIME_CUDA_SASS_SAMPLE_MODE=thread \
BPFTIME_CUDA_SASS_SAMPLE_MAX_RECORDS=4096 \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_PATH=/tmp/bpftime-sass-thread.jsonl \
```

### Experimental: device-per-thread writes (SM120)

By default, `thread` mode is **host-expanded** from per-warp slots for stability.
To force **device-side per-thread** writes (one slot per `tid.x`, no host expansion):

```bash
BPFTIME_CUDA_SASS_SAMPLE_MODE=thread \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE=1 \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_STRIDE4=1 \
BPFTIME_CUDA_SASS_SAMPLE_MAX_RECORDS=64 \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_PATH=/tmp/bpftime-sass-thread-device-%p.jsonl \
```

Notes:
- This path is still experimental; some vLLM/cutlass cubin-only kernels may require gating:
  `BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_LANE0_ONLY=1`.
- To reduce write density, you can also set `BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_WARP0_ONLY=1`.
  If both `*_LANE0_ONLY` and `*_WARP0_ONLY` are set, it becomes `tid.x==0` only.
- Sampling buffer allocation currently reuses CUDA driver trampolines; keep
  `BPFTIME_CUDA_TRACE_PATH` enabled when testing SASS sampling.

### With bpftime + SASS detour + load test

```bash
source /tmp/vllm-venv/bin/activate

BPFTIME_ALLOW_NO_SHM=1 \
BPFTIME_LOG_OUTPUT=console \
BPFTIME_CUDA_TRACE_PATH=/tmp/bpftime-vllm-trace.jsonl \
BPFTIME_CUDA_SASS_DETOUR=1 \
BPFTIME_CUDA_SASS_DETOUR_DEBUG=1 \
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
HF_HOME=/tmp/hf \
VLLM_CONCURRENCY=4 \
VLLM_TOTAL_REQUESTS=20 \
VLLM_STREAM=1 \
python benchmark/gpu/vllm_observability/vllm_openai_server_load_test.py
```

## Notes

- Default is `VLLM_ENFORCE_EAGER=1` for stability and faster startup. To test
  cudagraph/compile-heavy paths, run with `VLLM_ENFORCE_EAGER=0`.
- Override model/prompt via env vars: `VLLM_MODEL`, `VLLM_PROMPT`,
  `VLLM_MAX_TOKENS`, `VLLM_GPU_MEM_UTIL`.
- For SM/Warp/Lane mapping on CUDA kernels, see `benchmark/gpu/vllm_threadscheduling/README.md:1` (requires PTX-available kernels; SASS-only kernels need SASS-level support).

## Troubleshooting

### `cuInit()` returns `304` / `nvidia-smi` fails

`304` is `CUDA_ERROR_OPERATING_SYSTEM`, usually meaning the process cannot talk to the kernel driver (often container security/device-cgroup related).

Quick checks:

```bash
python3 -c "import os; os.open('/dev/nvidiactl', os.O_RDWR); print('nvidiactl rw ok')"
python3 -c "import ctypes; print(ctypes.CDLL('libcuda.so.1').cuInit(0))"
```

If `/dev/nvidiactl` cannot be opened with `O_RDWR` (permission denied), CUDA will not work in this container.

Typical fixes (Docker/K8s-level, outside bpftime):
- Run with GPU access enabled (e.g. Docker `--gpus all`) and allow RW access to `/dev/nvidia*`.
- Use a permissive seccomp profile if needed (e.g. `--security-opt seccomp=unconfined`).

### bpftime fails early due to shared memory (`/dev/shm`) permissions

Some environments deny creating/writing POSIX shared memory. bpftime’s CUDA tracing/detour can run without shared memory (no eBPF execution) when CUDA features are enabled and shared memory setup fails.

- Set `BPFTIME_ALLOW_NO_SHM=1` as shown in the commands above.
- Optional explicit enable: `BPFTIME_CUDA_STANDALONE=1`.

### SASS detour reports `no_cubin_dumps` / sampler dumps 0 records

If you enabled:
- `BPFTIME_CUDA_SASS_DETOUR=1`
- `BPFTIME_CUDA_SASS_DETOUR_DUMP_DIR=/tmp/...`
- (optional) `BPFTIME_CUDA_SASS_SAMPLE=1`

but the load test output shows:
- `sass_detour_cubin_verify.error: "no_cubin_dumps"` **or**
- your `BPFTIME_CUDA_SASS_SAMPLE_DUMP_PATH` contains only `bpftime_sass_sample_meta` with no thread/warp/cta records,

then the selected kernel likely did not come from a patchable **SM120 ELF cubin** with `.text.*` sections (common cases: fatbin/PTX/JIT payloads, stripped/sectionless images, SM mismatch).

To isolate which kernel you *can* patch and (if needed) dump the failing one:

0) Optional: dump “unpatchable” SM120 ELFs for offline inspection:

- Set `BPFTIME_CUDA_SASS_DETOUR_DUMP_UNPATCHED=1` to write `*.cubin.raw` into the same dump dir.
  These are **not** counted as patched detours (so the JSON summary still reflects `entry_bra` correctly),
  but you can `cuobjdump --dump-sass` them to see what format you’re dealing with.

1) Find recent kernel names from the bpftime trace:

```bash
python3 - <<'PY'
import json
from collections import Counter
p="/tmp/bpftime-vllm-trace.jsonl"
last=[]
counts=Counter()
bad=0
with open(p) as f:
  for line in f:
    line=line.strip()
    if not line: 
      continue
    try:
      o=json.loads(line)
    except Exception:
      bad+=1
      continue
    if o.get("type")=="launch" and o.get("name"):
      last.append(o["name"])
      counts[o["name"]]+=1
print("bad_lines",bad)
print("last_10_launches:")
for n in last[-10:]:
  print("  ", n)
print("top_10_kernels:")
for n,c in counts.most_common(10):
  print(f"  {c:6d}  {n}")
PY
```

2) Pick a stable substring from one kernel name (e.g. `cutlass_80_tensorop_...`) and re-run with:
- `BPFTIME_CUDA_SASS_DETOUR_FILTER=<substring>`
- `BPFTIME_CUDA_SASS_DETOUR_DUMP_DIR=/tmp/bpftime-sass-...` (so `cuobjdump` verification can work)

## Observability coverage (today)

Below is a checklist for a “normal vLLM inference service” and what bpftime can
observe *today* without SASS-level instrumentation.

### Already observable

- **Request experience (metrics)**: `e2e latency` / `TTFT` / `inter-token latency` / `queue time` / `prefill` vs `decode` time via vLLM `/metrics` scraping.
- **Throughput (metrics)**: request/token counters and token distribution buckets via vLLM `/metrics`.
- **GPU execution boundary (bpftime trace)**: kernel launches (`type=launch`) with kernel name + grid/block/shared_mem/stream.
- **GPU memory & sync boundary (bpftime trace)**: `type=memcpy`/`memset`/`alloc`/`free`/`sync` (sync includes a measured blocking duration).
- **Streams/events/library API boundary (bpftime trace)**: `type=stream_create`, `type=event_create`/`event_record`, and `type=library_load`/`library_get_kernel`/`kernel_get_function`.

### Not yet observable (gaps)

- **Request↔GPU correlation**: per-request / per-step segmentation (e.g. injecting `request_id` and writing markers) so GPU events can be attributed to a specific request/iteration.
- **Per-kernel duration**: kernel begin/end timing on the GPU timeline (launch-only is insufficient; SASS/driver callbacks or event-based measurement needed).
- **KV cache semantics**: vLLM-level cache block allocate/free/evict/swap events (beyond aggregate `kv_cache_usage_perc`).
- **Scheduler internals**: batch composition per iteration, preemption reasons, and engine state transitions as structured events (beyond aggregated metrics).
- **GPU hardware counters**: SM occupancy/utilization, DRAM traffic, cache hit rates, etc.
- **Multi-GPU / comms**: NCCL semantic events and per-collective timing/bytes (if using tensor/pipeline parallelism).

#### Hardware counters (not in scope)

We do not implement CUPTI/Nsight hardware counters in this repo at the moment.
