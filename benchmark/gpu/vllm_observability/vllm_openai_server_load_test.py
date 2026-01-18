import json
import os
import random
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from queue import Queue
from typing import Dict, List, Optional, Tuple


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _http_get(url: str, timeout_s: float) -> Tuple[int, str]:
    import requests

    r = requests.get(url, timeout=timeout_s)
    return r.status_code, r.text


def _parse_prometheus_text(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        metric, value_s = parts
        base = metric.split("{", 1)[0]
        try:
            value = float(value_s)
        except ValueError:
            continue
        out[base] = out.get(base, 0.0) + value
    return out


def _diff_metrics(after: Dict[str, float], before: Dict[str, float]) -> Dict[str, float]:
    keys = set(before) | set(after)
    out: Dict[str, float] = {}
    for k in keys:
        out[k] = after.get(k, 0.0) - before.get(k, 0.0)
    return out


def _summarize_trace(
    trace_path: str, max_kernels: int = 40
) -> Tuple[Dict[str, int], int, Tuple[Tuple[str, int], ...]]:
    if not trace_path or not os.path.exists(trace_path):
        return {}, 0, tuple()
    counter: Counter[str] = Counter()
    type_counter: Counter[str] = Counter()
    lines = 0
    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue
            ev_type = obj.get("type")
            if not isinstance(ev_type, str) or not ev_type:
                if isinstance(obj.get("name"), str):
                    ev_type = "launch"
            if isinstance(ev_type, str) and ev_type:
                type_counter[ev_type] += 1
            name = obj.get("name")
            if isinstance(name, str) and name:
                counter[name] += 1
    return dict(type_counter), lines, tuple(counter.most_common(max_kernels))


def _log_contains(path: str, substr: str) -> bool:
    try:
        with open(path, "rb") as f:
            return substr.encode("utf-8") in f.read()
    except FileNotFoundError:
        return False


def _find_cuobjdump() -> Optional[str]:
    # Prefer explicit env override, then common CUDA install path, then PATH.
    for cand in [
        os.environ.get("CUOBJDUMP"),
        "/usr/local/cuda/bin/cuobjdump",
        "cuobjdump",
    ]:
        if not cand:
            continue
        if os.path.isabs(cand) and os.path.exists(cand):
            return cand
        # If it's not an absolute path, rely on PATH resolution.
        if not os.path.isabs(cand):
            return cand
    return None


def _verify_cubin_only_detour(dump_dir: str) -> Dict[str, object]:
    out: Dict[str, object] = {
        "dump_dir": dump_dir,
        "sample_cubin": None,
        "no_ptx": None,
        "entry_bra": None,
        "cuobjdump": None,
        "error": None,
    }
    try:
        if not dump_dir or not os.path.isdir(dump_dir):
            out["error"] = "dump_dir_missing"
            return out
        cubins = [os.path.join(dump_dir, f) for f in os.listdir(dump_dir) if f.endswith(".cubin")]
        # Pick the newest dump; older runs may leave stale cubins in the directory.
        cubins.sort(key=lambda p: os.path.getmtime(p))
        if not cubins:
            out["error"] = "no_cubin_dumps"
            return out
        sample = cubins[-1]
        out["sample_cubin"] = sample

        cuobjdump = _find_cuobjdump()
        out["cuobjdump"] = cuobjdump
        if not cuobjdump:
            out["error"] = "cuobjdump_not_found"
            return out

        # 1) verify no PTX
        ptx_out = subprocess.check_output(
            [cuobjdump, "--extract-ptx", "all", sample],
            stderr=subprocess.STDOUT,
            text=True,
        )
        out["no_ptx"] = "No PTX file found" in ptx_out or "does not contain PTX" in ptx_out

        # 2) verify at least one function has a detour BRA at /*0000*/
        sass_out = subprocess.check_output(
            [cuobjdump, "--dump-sass", sample],
            stderr=subprocess.STDOUT,
            text=True,
        )
        entry_bra = False
        for line in sass_out.splitlines():
            s = line.strip()
            if s.startswith("/*0000*/"):
                if " BRA " in s or s.startswith("/*0000*/BRA") or s.startswith("/*0000*/ BRA"):
                    entry_bra = True
                    break
        out["entry_bra"] = entry_bra
        return out
    except subprocess.CalledProcessError as e:
        out["error"] = f"cuobjdump_failed_rc={e.returncode}"
        out["output"] = e.output[:4000] if isinstance(e.output, str) else None
        return out
    except Exception as e:
        out["error"] = str(e)
        return out


@dataclass(frozen=True)
class RequestResult:
    ok: bool
    latency_s: float
    ttft_s: Optional[float]
    error: Optional[str]


def _post_completion_streaming(
    url: str, payload: dict, timeout_s: float
) -> RequestResult:
    import requests

    start = time.time()
    ttft: Optional[float] = None
    try:
        with requests.post(url, json=payload, stream=True, timeout=timeout_s) as r:
            if r.status_code // 100 != 2:
                return RequestResult(
                    ok=False,
                    latency_s=time.time() - start,
                    ttft_s=None,
                    error=f"status={r.status_code} body={r.text[:2000]}",
                )
            for raw in r.iter_lines(decode_unicode=True):
                if raw is None:
                    continue
                line = raw.strip()
                if not line:
                    continue
                # OpenAI-style SSE: "data: {...}" or "data: [DONE]"
                if line.startswith("data:"):
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    if ttft is None:
                        ttft = time.time() - start
    except Exception as e:
        return RequestResult(
            ok=False,
            latency_s=time.time() - start,
            ttft_s=ttft,
            error=str(e),
        )
    return RequestResult(ok=True, latency_s=time.time() - start, ttft_s=ttft, error=None)


def _post_completion_non_streaming(
    url: str, payload: dict, timeout_s: float
) -> RequestResult:
    import requests

    start = time.time()
    try:
        r = requests.post(url, json=payload, timeout=timeout_s)
        if r.status_code // 100 != 2:
            return RequestResult(
                ok=False,
                latency_s=time.time() - start,
                ttft_s=None,
                error=f"status={r.status_code} body={r.text[:2000]}",
            )
        return RequestResult(ok=True, latency_s=time.time() - start, ttft_s=None, error=None)
    except Exception as e:
        return RequestResult(ok=False, latency_s=time.time() - start, ttft_s=None, error=str(e))


def main() -> int:
    model = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-0.6B")
    host = "127.0.0.1"
    port = int(os.environ.get("VLLM_PORT", "0")) or _find_free_port()
    base = f"http://{host}:{port}"

    # More "realistic" than a single request:
    # - concurrency (multiple in-flight requests)
    # - optional streaming
    # - mixed output lengths
    concurrency = int(os.environ.get("VLLM_CONCURRENCY", "4"))
    total_requests = int(os.environ.get("VLLM_TOTAL_REQUESTS", "20"))
    streaming = os.environ.get("VLLM_STREAM", "1") not in ("0", "false", "False")
    timeout_s = float(os.environ.get("VLLM_REQUEST_TIMEOUT_S", "120"))

    # Keep defaults small to reduce OOM risk; override as needed.
    gpu_mem_util = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.2"))
    enforce_eager = os.environ.get("VLLM_ENFORCE_EAGER", "1") not in ("0", "false", "False")

    def _repeat_to_len(s: str, target_chars: int) -> str:
        if target_chars <= 0:
            return s
        if not s:
            s = "x"
        # Repeat with a separator to avoid pathological tokenizer behavior on a single char.
        sep = " "
        out = s
        while len(out) < target_chars:
            out = out + sep + s
        return out[:target_chars]

    prompt = os.environ.get("VLLM_PROMPT", "Write a short story about a GPU kernel launch:")
    prompt2 = os.environ.get("VLLM_PROMPT2", "Explain what KV cache eviction means in LLM serving:")
    prompt_len = int(os.environ.get("VLLM_PROMPT_LEN", "0"))
    prompt2_len = int(os.environ.get("VLLM_PROMPT2_LEN", "0"))
    if prompt_len > 0:
        prompt = _repeat_to_len(prompt, prompt_len)
    if prompt2_len > 0:
        prompt2 = _repeat_to_len(prompt2, prompt2_len)

    max_tokens_min = int(os.environ.get("VLLM_MAX_TOKENS_MIN", "16"))
    max_tokens_max = int(os.environ.get("VLLM_MAX_TOKENS_MAX", "128"))
    temperature = float(os.environ.get("VLLM_TEMPERATURE", "0.0"))
    max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "2048"))
    swap_space = os.environ.get("VLLM_SWAP_SPACE", "")
    cpu_offload_gb = os.environ.get("VLLM_CPU_OFFLOAD_GB", "")
    kv_cache_memory_bytes = os.environ.get("VLLM_KV_CACHE_MEMORY_BYTES", "")
    kv_offloading_size = os.environ.get("VLLM_KV_OFFLOADING_SIZE", "")
    kv_offloading_backend = os.environ.get("VLLM_KV_OFFLOADING_BACKEND", "")
    attention_backend = os.environ.get("VLLM_ATTENTION_BACKEND", "")
    disable_hybrid_kv_cache_mgr = os.environ.get("VLLM_DISABLE_HYBRID_KV_CACHE_MANAGER", "") not in (
        "",
        "0",
        "false",
        "False",
    )

    log_path = os.environ.get("VLLM_SERVER_LOG", "/tmp/vllm-openai-server.log")
    trace_path = os.environ.get("BPFTIME_CUDA_TRACE_PATH") or os.environ.get("BPFTIME_CUDA_LAUNCH_TRACE_PATH", "")
    sass_dump_dir = os.environ.get("BPFTIME_CUDA_SASS_DETOUR_DUMP_DIR", "")

    env = os.environ.copy()
    env.setdefault("HF_HOME", "/tmp/hf")
    # Avoid vLLM reading the deprecated VLLM_ATTENTION_BACKEND env var; we only
    # want the explicit CLI flag (if provided), otherwise vLLM may treat it as
    # attention_config.backend and crash due to conflicts.
    env.pop("VLLM_ATTENTION_BACKEND", None)

    def build_args(util: float) -> list[str]:
        a = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--host",
            host,
            "--port",
            str(port),
            "--model",
            model,
            "--dtype",
            "float16",
            "--gpu-memory-utilization",
            str(util),
            "--max-model-len",
            str(max_model_len),
        ]
        if enforce_eager:
            a.append("--enforce-eager")
        if attention_backend:
            a += ["--attention-backend", attention_backend]
        if swap_space:
            a += ["--swap-space", swap_space]
        if cpu_offload_gb:
            a += ["--cpu-offload-gb", cpu_offload_gb]
        if kv_cache_memory_bytes:
            a += ["--kv-cache-memory-bytes", kv_cache_memory_bytes]
        if kv_offloading_size:
            a += ["--kv-offloading-size", kv_offloading_size]
        if kv_offloading_backend:
            a += ["--kv-offloading-backend", kv_offloading_backend]
        if disable_hybrid_kv_cache_mgr:
            a.append("--disable-hybrid-kv-cache-manager")
        return a

    def log_tail(path: str, max_lines: int = 120) -> str:
        try:
            with open(path, "rb") as f:
                lines = f.read().splitlines()
                return b"\n".join(lines[-max_lines:]).decode("utf-8", errors="replace")
        except FileNotFoundError:
            return ""

    def start_server(util: float) -> subprocess.Popen[bytes]:
        args = build_args(util)
        with open(log_path, "ab") as logf:
            logf.write(b"\n=== bpftime test: starting vLLM server (load test) ===\n")
            logf.write(("args: " + " ".join(args) + "\n").encode("utf-8"))
            logf.flush()
        logf = open(log_path, "ab")
        return subprocess.Popen(
            args,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )

    util_attempts = [gpu_mem_util, 0.1, 0.05, 0.02, 0.01]
    util_attempts = [u for u in util_attempts if 0 < u <= 1.0]
    proc: Optional[subprocess.Popen[bytes]] = None
    for util in util_attempts:
        if os.path.exists(log_path):
            os.remove(log_path)
        proc = start_server(util)
        deadline = time.time() + float(os.environ.get("VLLM_STARTUP_TIMEOUT_S", "300"))
        last_err: Optional[Exception] = None
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            try:
                code, _ = _http_get(f"{base}/health", timeout_s=2.0)
                if 200 <= code < 300:
                    last_err = None
                    break
            except Exception as e:
                last_err = e
            time.sleep(0.25)
        if last_err is None:
            break
        # OOM retry heuristic (similar to the smoke test script)
        if proc.poll() is not None and _log_contains(log_path, "Free memory on device") and _log_contains(
            log_path, "less than desired GPU memory utilization"
        ):
            continue
        tail = log_tail(log_path)
        raise RuntimeError(
            f"vLLM server failed to become healthy (rc={proc.poll()}) last_err={last_err}; log={log_path}\n--- log tail ---\n{tail}"
        )

    if proc is None:
        raise RuntimeError(f"vLLM server failed to start; log={log_path}")

    try:
        deadline = time.time() + float(os.environ.get("VLLM_STARTUP_TIMEOUT_S", "300"))
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            try:
                code, _ = _http_get(f"{base}/health", timeout_s=2.0)
                if 200 <= code < 300:
                    break
            except Exception:
                pass
            time.sleep(0.25)

        if proc.poll() is not None:
            tail = log_tail(log_path)
            raise RuntimeError(
                f"vLLM server died early (rc={proc.poll()}) log={log_path}\n--- log tail ---\n{tail}"
            )

        # Metrics before
        _, metrics_before_txt = _http_get(f"{base}/metrics", timeout_s=10.0)
        metrics_before = _parse_prometheus_text(metrics_before_txt)

        # Generate N requests under concurrency.
        endpoint = f"{base}/v1/completions"
        q: Queue[int] = Queue()
        for i in range(total_requests):
            q.put(i)

        results: List[RequestResult] = []
        results_lock = threading.Lock()

        def worker(worker_id: int) -> None:
            while True:
                try:
                    _ = q.get_nowait()
                except Exception:
                    return
                local_prompt = prompt if (worker_id % 2 == 0) else prompt2
                max_tokens = random.randint(max_tokens_min, max_tokens_max)
                payload = {
                    "model": model,
                    "prompt": local_prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": bool(streaming),
                }
                if streaming:
                    rr = _post_completion_streaming(endpoint, payload, timeout_s=timeout_s)
                else:
                    rr = _post_completion_non_streaming(endpoint, payload, timeout_s=timeout_s)
                with results_lock:
                    results.append(rr)
                q.task_done()

        start_load = time.time()
        threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(concurrency)]
        for t in threads:
            t.start()
        q.join()
        load_duration_s = time.time() - start_load

        # Metrics after
        _, metrics_after_txt = _http_get(f"{base}/metrics", timeout_s=10.0)
        metrics_after = _parse_prometheus_text(metrics_after_txt)

        ok = [r for r in results if r.ok]
        failed = [r for r in results if not r.ok]

        def pct(values: List[float], p: float) -> float:
            if not values:
                return 0.0
            values = sorted(values)
            idx = int(round((len(values) - 1) * p))
            return float(values[max(0, min(len(values) - 1, idx))])

        latencies = [r.latency_s for r in ok]
        ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]

        # Trace summary (bpftime)
        trace_type_counts, trace_lines, top_kernels = _summarize_trace(trace_path)

        sass_detour_patched = _log_contains(log_path, "SASS detour: patched")
        sass_cubin_verify = (
            _verify_cubin_only_detour(sass_dump_dir)
            if sass_dump_dir
            else {"dump_dir": None, "sample_cubin": None, "no_ptx": None, "entry_bra": None, "error": None}
        )
        if not sass_detour_patched and isinstance(sass_cubin_verify, dict):
            if sass_cubin_verify.get("sample_cubin") and not sass_cubin_verify.get("error"):
                sass_detour_patched = True

        out = {
            "model": model,
            "port": port,
            "server_log": log_path,
            "sass_detour_patched": sass_detour_patched,
            "sass_detour_cubin_verify": sass_cubin_verify,
            "load": {
                "total_requests": total_requests,
                "concurrency": concurrency,
                "streaming": streaming,
                "duration_s": load_duration_s,
                "ok": len(ok),
                "failed": len(failed),
                "p50_s": pct(latencies, 0.50),
                "p90_s": pct(latencies, 0.90),
                "p99_s": pct(latencies, 0.99),
                "ttft_p50_s": pct(ttfts, 0.50) if ttfts else None,
                "ttft_p90_s": pct(ttfts, 0.90) if ttfts else None,
            },
            "errors_top": Counter([r.error for r in failed if r.error]).most_common(5),
            "metrics_delta": _diff_metrics(metrics_after, metrics_before),
            "trace_path": trace_path,
            "trace_lines": trace_lines,
            "trace_type_counts": trace_type_counts,
            "top_kernels": top_kernels,
        }

        print(json.dumps(out, indent=2, sort_keys=False))
        return 0 if len(failed) == 0 else 2
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            pass
        try:
            proc.wait(timeout=15)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
