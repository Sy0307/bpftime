#!/usr/bin/env python3
import argparse
import os
import time


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run real vLLM inference in-process (no OpenAI server)."
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("VLLM_MODEL", "Qwen/Qwen3-0.6B"),
        help="HuggingFace model id (or local path). Default: env VLLM_MODEL or Qwen/Qwen3-0.6B",
    )
    parser.add_argument(
        "--prompt",
        default=os.environ.get("VLLM_PROMPT", "Write a short haiku about GPUs."),
        help="Prompt text (or env VLLM_PROMPT).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("VLLM_MAX_TOKENS", "64")),
        help="Max new tokens. Default: env VLLM_MAX_TOKENS or 64",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("VLLM_TEMPERATURE", "0.0")),
        help="Sampling temperature. Default: env VLLM_TEMPERATURE or 0.0",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=os.environ.get("VLLM_TRUST_REMOTE_CODE", "0").lower()
        in ("1", "true", "yes", "y", "on"),
        help="Pass trust_remote_code=True to vLLM.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=os.environ.get("VLLM_ENFORCE_EAGER", "0").lower()
        in ("1", "true", "yes", "y", "on"),
        help="Disable torch.compile and CUDA graphs (enforce eager execution).",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.9")),
        help="vLLM gpu_memory_utilization (0..1). Default: env VLLM_GPU_MEMORY_UTILIZATION or 0.9",
    )
    parser.add_argument(
        "--swap-space",
        type=float,
        default=float(os.environ.get("VLLM_SWAP_SPACE", "4")),
        help="vLLM swap_space (GiB). Default: env VLLM_SWAP_SPACE or 4",
    )
    parser.add_argument(
        "--cpu-offload-gb",
        type=float,
        default=float(os.environ.get("VLLM_CPU_OFFLOAD_GB", "0")),
        help="vLLM cpu_offload_gb (GiB). Default: env VLLM_CPU_OFFLOAD_GB or 0",
    )
    parser.add_argument(
        "--kv-cache-memory-bytes",
        type=int,
        default=int(os.environ.get("VLLM_KV_CACHE_MEMORY_BYTES", "0")),
        help="vLLM kv_cache_memory_bytes (bytes). 0 disables override. Default: env VLLM_KV_CACHE_MEMORY_BYTES or 0",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=int(os.environ.get("VLLM_MAX_MODEL_LEN", "0")),
        help="vLLM max_model_len override. 0 keeps default. Default: env VLLM_MAX_MODEL_LEN or 0",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("VLLM_BATCH_SIZE", "1")),
        help="Number of prompts in a single generate() call. Default: env VLLM_BATCH_SIZE or 1",
    )
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    llm_kwargs = dict(
        model=args.model,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        swap_space=args.swap_space,
        cpu_offload_gb=args.cpu_offload_gb,
        kv_cache_memory_bytes=(
            None if args.kv_cache_memory_bytes <= 0 else args.kv_cache_memory_bytes
        ),
    )
    if args.max_model_len > 0:
        llm_kwargs["max_model_len"] = args.max_model_len
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        max_tokens=args.max_tokens, temperature=args.temperature
    )

    t0 = time.time()
    prompts = [args.prompt] * max(1, args.batch_size)
    outputs = llm.generate(prompts, sampling)
    # Ensure all device-side work is completed so bpftime can reliably dump
    # SASS sampling buffers on sync.
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    t1 = time.time()

    out0 = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
    print(
        f"model={args.model} max_tokens={args.max_tokens} batch_size={len(prompts)} "
        f"elapsed_s={t1 - t0:.3f} output_chars={len(out0)}"
    )
    print(out0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
