# vLLM SM/Warp/Lane mapping (CUDA eBPF probe)

This probe reuses bpftime’s CUDA eBPF infrastructure to sample **which SM and
warp slots** are executing a selected CUDA kernel while vLLM is running.

Compared to `example/gpu/threadscheduling`, this version uses a GPU kernel-shared
array map (device-visible) and only records **lane0 per warp**, so it is much
more suitable for real workloads like vLLM.

## Build

```bash
make -C benchmark/gpu/vllm_threadscheduling
```

## Run (two terminals)

### Terminal 1: start the probe loader (syscall-server)

Pick a kernel symbol from bpftime CUDA tracing (`type=launch` `name=...`) or from
`benchmark/gpu/vllm_observability/vllm_openai_server_test.py`’s `top_kernels`.

```bash
BPFTIME_LOG_OUTPUT=console \
LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so \
benchmark/gpu/vllm_threadscheduling/vllm_threadscheduling \
  --func '<CUDA_KERNEL_SYMBOL_NAME>'
```

### Terminal 2: run vLLM under bpftime agent

```bash
source /tmp/vllm-venv/bin/activate

BPFTIME_LOG_OUTPUT=console \
BPFTIME_CUDA_TRACE_PATH=/tmp/bpftime-vllm-trace.jsonl \
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
HF_HOME=/tmp/hf \
python benchmark/gpu/vllm_observability/vllm_openai_server_test.py
```

## Notes / limitations

- This only works for kernels bpftime can instrument (typically PTX-available).
  SASS-only kernels require SASS-level instrumentation support (binary-level
  probe insertion), beyond the current PTX rewriting pipeline.
- In the current common `torch`/`vllm` binary wheels, many CUDA kernels are
  shipped as cubin-only fatbins (no embedded PTX). In that case bpftime’s
  PTX-level kprobe can’t be inserted and the probe will keep showing
  “No data collected yet.” This is one of the main motivations for SASS-level
  support.
- `BPFTIME_CUDA_SASS_DETOUR=1` exists as an experimental “do-nothing detour”
  for validating that CUBINs are patchable, but it does not yet run this probe
  inside SASS-only kernels.
- The probe reports **sampled warps** (lane0 only), not exact executed thread
  counts.
