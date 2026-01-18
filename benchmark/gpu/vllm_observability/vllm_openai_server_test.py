import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Tuple


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _http_get(url: str, timeout_s: float) -> Tuple[int, str]:
    import requests

    r = requests.get(url, timeout=timeout_s)
    return r.status_code, r.text


def _http_post_json(url: str, payload: dict, timeout_s: float) -> Tuple[int, dict]:
    import requests

    r = requests.post(url, json=payload, timeout=timeout_s)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"_raw_text": r.text}


def _parse_prometheus_text(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # name{...} value  OR  name value
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


@dataclass(frozen=True)
class RunSummary:
    port: int
    completion_text: str
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    trace_type_counts: Dict[str, int]
    trace_lines: int
    top_kernels: Tuple[Tuple[str, int], ...]


def _summarize_trace(
    trace_path: str, max_kernels: int = 20
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
                # Back-compat: older kernel launch events don't have a "type".
                if isinstance(obj.get("name"), str):
                    ev_type = "launch"
            if isinstance(ev_type, str) and ev_type:
                type_counter[ev_type] += 1
            name = obj.get("name")
            if isinstance(name, str) and name:
                counter[name] += 1
    return dict(type_counter), lines, tuple(counter.most_common(max_kernels))


def main() -> int:
    model = os.environ.get("VLLM_MODEL", "facebook/opt-125m")
    prompt = os.environ.get("VLLM_PROMPT", "Hello, my name is")
    max_tokens = int(os.environ.get("VLLM_MAX_TOKENS", "32"))
    gpu_mem_util = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.2"))

    # Fast/stable defaults for CI-like validation. Set VLLM_ENFORCE_EAGER=0 to
    # exercise cudagraph/compile-heavy paths.
    enforce_eager = os.environ.get("VLLM_ENFORCE_EAGER", "1") not in ("0", "false", "False")

    port = int(os.environ.get("VLLM_PORT", "0")) or _find_free_port()
    host = "127.0.0.1"
    base = f"http://{host}:{port}"

    log_path = os.environ.get("VLLM_SERVER_LOG", "/tmp/vllm-openai-server.log")
    trace_path = os.environ.get("BPFTIME_CUDA_TRACE_PATH") or os.environ.get("BPFTIME_CUDA_LAUNCH_TRACE_PATH", "")

    env = os.environ.copy()
    env.setdefault("HF_HOME", "/tmp/hf")

    def build_args(util: float) -> list[str]:
        args = [
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
            "2048",
        ]
        if enforce_eager:
            args.append("--enforce-eager")
        return args

    def server_log_contains(substr: str) -> bool:
        try:
            with open(log_path, "rb") as f:
                return substr.encode("utf-8") in f.read()
        except FileNotFoundError:
            return False

    def start_server(util: float) -> subprocess.Popen[bytes]:
        args = build_args(util)
        with open(log_path, "ab") as logf:
            logf.write(b"\n=== bpftime test: starting vLLM server ===\n")
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
    proc: subprocess.Popen[bytes] | None = None
    last_start_err: Exception | None = None
    for util in util_attempts:
        if proc is not None and proc.poll() is None:
            break
        if os.path.exists(log_path):
            os.remove(log_path)
        proc = start_server(util)
        try:
            deadline = time.time() + float(os.environ.get("VLLM_STARTUP_TIMEOUT_S", "240"))
            last_err = None
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
            # If the server died because of insufficient free GPU memory, retry with smaller util.
            if proc.poll() is not None and server_log_contains("Free memory on device") and server_log_contains(
                "less than desired GPU memory utilization"
            ):
                continue
            last_start_err = RuntimeError(f"vLLM server not healthy (proc={proc.poll()}) last_err={last_err}; log={log_path}")
            break
        except Exception as e:
            last_start_err = e
            break

    if proc is None or proc.poll() is not None:
        raise RuntimeError(f"vLLM server failed to start; log={log_path} err={last_start_err}")

    try:
        _, metrics_before_text = _http_get(f"{base}/metrics", timeout_s=5.0)
        metrics_before = _parse_prometheus_text(metrics_before_text)

        code, resp = _http_post_json(
            f"{base}/v1/completions",
            {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0},
            timeout_s=float(os.environ.get("VLLM_REQUEST_TIMEOUT_S", "240")),
        )
        if not (200 <= code < 300):
            raise RuntimeError(f"completion failed: status={code} resp={resp} log={log_path}")
        try:
            completion_text = resp["choices"][0]["text"]
        except Exception:
            raise RuntimeError(f"unexpected completion response: {resp}")

        _, metrics_after_text = _http_get(f"{base}/metrics", timeout_s=5.0)
        metrics_after = _parse_prometheus_text(metrics_after_text)

        trace_type_counts, trace_lines, top_kernels = _summarize_trace(trace_path)
        if trace_path and trace_lines == 0:
            raise RuntimeError(
                f"BPFTIME_CUDA_TRACE_PATH/BPFTIME_CUDA_LAUNCH_TRACE_PATH was set but trace is empty: {trace_path} (log={log_path})"
            )

        metrics_delta = {
            k: metrics_after.get(k, 0.0) - metrics_before.get(k, 0.0)
            for k in sorted(set(metrics_before) | set(metrics_after))
        }
        if "vllm:request_success_total" in metrics_delta and metrics_delta["vllm:request_success_total"] < 1:
            raise RuntimeError(
                f"vLLM metrics did not record a successful request: delta(vllm:request_success_total)={metrics_delta['vllm:request_success_total']}"
            )
        if "vllm:request_generation_tokens_sum" in metrics_delta:
            gen = metrics_delta["vllm:request_generation_tokens_sum"]
            if not (0 < gen <= max_tokens):
                raise RuntimeError(
                    f"unexpected generation tokens from metrics: delta(vllm:request_generation_tokens_sum)={gen} (max_tokens={max_tokens})"
                )

        summary = RunSummary(
            port=port,
            completion_text=completion_text,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            trace_type_counts=trace_type_counts,
            trace_lines=trace_lines,
            top_kernels=top_kernels,
        )

        print(
            json.dumps(
                {
                    "port": summary.port,
                    "completion_text": summary.completion_text,
                    "trace_path": trace_path,
                    "trace_lines": summary.trace_lines,
                    "trace_type_counts": summary.trace_type_counts,
                    "top_kernels": summary.top_kernels,
                    "metrics_delta": {
                        k: metrics_delta[k]
                        for k in metrics_delta
                        if any(tok in k for tok in ("vllm", "request", "token", "prompt", "decode"))
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return 0
    finally:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    proc.kill()
                proc.wait(timeout=30)


if __name__ == "__main__":
    raise SystemExit(main())
