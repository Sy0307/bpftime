#!/usr/bin/env python3
import ctypes
import ctypes.util
import os
import subprocess
import sys
import json
from ctypes import c_int, c_void_p, c_ulonglong, c_size_t


CUDA_SUCCESS = 0


def _check(res: int, name: str) -> None:
    if res != CUDA_SUCCESS:
        raise RuntimeError(f"{name} failed: {res}")


def _nvcc() -> str:
    return os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc")


def _compile_cubin(cu_path: str, cubin_path: str) -> None:
    nvcc = _nvcc()
    arch = os.environ.get("CUDA_ARCH", "sm_120")
    cmd = [
        nvcc,
        f"-arch={arch}",
        "-O0",
        "--cubin",
        cu_path,
        "-o",
        cubin_path,
    ]
    subprocess.check_call(cmd)


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    cu_path = os.path.join(here, "cubin_call_entry.cu")
    cubin_path = os.environ.get("BPFTIME_TEST_CUBIN", "/tmp/bpftime_call_entry.cubin")

    if not os.path.exists(cubin_path) or os.path.getmtime(cubin_path) < os.path.getmtime(cu_path):
        _compile_cubin(cu_path, cubin_path)

    # Prefer the real driver library. `find_library("cuda")` can resolve to the
    # CUDA Toolkit stub (`.../libcuda.so`) which returns CUDA_ERROR_SYSTEM_DRIVER_MISMATCH.
    libname = os.environ.get("BPFTIME_CUDA_LIB") or "libcuda.so.1"
    cu = ctypes.CDLL(libname)

    CUdevice = c_int
    CUcontext = c_void_p
    CUmodule = c_void_p
    CUfunction = c_void_p
    CUstream = c_void_p
    CUdeviceptr = c_ulonglong

    _check(cu.cuInit(0), "cuInit")
    dev = CUdevice()
    _check(cu.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    ctx = CUcontext()
    _check(cu.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "cuCtxCreate")

    with open(cubin_path, "rb") as f:
        data = f.read()
    buf = ctypes.create_string_buffer(data)
    mod = CUmodule()
    _check(cu.cuModuleLoadData(ctypes.byref(mod), ctypes.cast(buf, c_void_p)), "cuModuleLoadData")

    func = CUfunction()
    _check(cu.cuModuleGetFunction(ctypes.byref(func), mod, b"call_entry"), "cuModuleGetFunction")

    out_dev = CUdeviceptr()
    _check(cu.cuMemAlloc_v2(ctypes.byref(out_dev), c_size_t(4)), "cuMemAlloc_v2")
    _check(cu.cuMemsetD32_v2(out_dev, 0, c_size_t(1)), "cuMemsetD32_v2")

    param0 = CUdeviceptr(out_dev.value)
    params = (c_void_p * 1)()
    params[0] = ctypes.cast(ctypes.pointer(param0), c_void_p)

    block = int(os.environ.get("BPFTIME_TEST_BLOCK", "128"))
    grid = int(os.environ.get("BPFTIME_TEST_GRID", "1"))
    launches = int(os.environ.get("BPFTIME_TEST_LAUNCHES", "1"))
    for _ in range(max(1, launches)):
        _check(
            cu.cuLaunchKernel(
                func,
                grid,
                1,
                1,
                block,
                1,
                1,
                0,
                CUstream(),
                ctypes.cast(params, c_void_p),
                None,
            ),
            "cuLaunchKernel",
        )
        _check(cu.cuCtxSynchronize(), "cuCtxSynchronize")

    if os.environ.get("BPFTIME_TEST_READ_SASS_SAMPLE_BUFFER"):
        dump_path = os.environ.get("BPFTIME_CUDA_SASS_SAMPLE_DUMP_PATH", "/tmp/bpftime-sample.jsonl")
        try:
            with open(dump_path, "r", encoding="utf-8") as f:
                meta = json.loads(f.readline())
            dev_ptr_s = meta.get("device_ptr") or "0x0"
            data_offset = int(meta.get("data_offset") or 0)
            dev_ptr = int(dev_ptr_s, 16) + data_offset
            buf_u32 = (ctypes.c_uint32 * 16)()
            _check(cu.cuMemcpyDtoH_v2(buf_u32, CUdeviceptr(dev_ptr), c_size_t(ctypes.sizeof(buf_u32))), "cuMemcpyDtoH_v2(sample_buf)")
            print("sass_sample_buf_u32[0:16]=", [hex(x) for x in buf_u32])
        except Exception as e:
            print("failed to read sass sample buffer:", repr(e), file=sys.stderr)

    if os.environ.get("BPFTIME_TEST_READ_SASS_CONTROL"):
        dump_path = os.environ.get("BPFTIME_CUDA_SASS_SAMPLE_DUMP_PATH", "/tmp/bpftime-sample.jsonl")
        try:
            with open(dump_path, "r", encoding="utf-8") as f:
                meta = json.loads(f.readline())
            dev_ptr_s = meta.get("device_ptr") or "0x0"
            dev_ptr = int(dev_ptr_s, 16)
            hdr_u32 = (ctypes.c_uint32 * 16)()
            _check(
                cu.cuMemcpyDtoH_v2(hdr_u32, CUdeviceptr(dev_ptr), c_size_t(ctypes.sizeof(hdr_u32))),
                "cuMemcpyDtoH_v2(control_hdr)",
            )
            print("sass_control_u32[0:16]=", [hex(x) for x in hdr_u32])
        except Exception as e:
            print("failed to read sass control header:", repr(e), file=sys.stderr)

    out_host = (c_int * 1)()
    _check(cu.cuMemcpyDtoH_v2(out_host, out_dev, c_size_t(4)), "cuMemcpyDtoH_v2")
    print(f"call_entry out[0]={out_host[0]}")

    cu.cuMemFree_v2(out_dev)
    cu.cuModuleUnload(mod)
    cu.cuCtxDestroy_v2(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
