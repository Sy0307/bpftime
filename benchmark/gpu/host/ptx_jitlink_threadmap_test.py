#!/usr/bin/env python3

import ctypes
import json
import os
import subprocess
import sys
from ctypes import c_char_p, c_int, c_size_t, c_uint, c_void_p


CUDA_SUCCESS = 0


def ck(res, name):
    if res != CUDA_SUCCESS:
        raise RuntimeError(f"{name} failed: {res}")


class CUdevice(ctypes.c_int):
    pass


class CUcontext(ctypes.c_void_p):
    pass


class CUlibrary(ctypes.c_void_p):
    pass


class CUkernel(ctypes.c_void_p):
    pass


class CUfunction(ctypes.c_void_p):
    pass


class CUmodule(ctypes.c_void_p):
    pass


class CUdeviceptr(ctypes.c_ulonglong):
    pass


class FatbincWrapper(ctypes.Structure):
    _fields_ = [
        ("magic", c_uint),
        ("version", c_uint),
        ("data", c_void_p),
        ("filename_or_fatbins", c_void_p),
    ]


def main():
    nvcc = os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc")
    if not os.path.exists(nvcc):
        nvcc = "/usr/local/cuda-12.9/bin/nvcc"

    dump_path = os.environ.get(
        "BPFTIME_CUDA_SASS_SAMPLE_DUMP_PATH",
        "/tmp/bpftime-ptx-jitlink-threadmap.jsonl",
    )

    workdir = "/tmp/bpftime-ptx-jitlink-threadmap"
    os.makedirs(workdir, exist_ok=True)
    cu_path = os.path.join(workdir, "k.cu")
    fatbin_path = os.path.join(workdir, "k.fatbin")

    # PTX-only fatbin.
    #
    # With bpftime:
    # - `BPFTIME_CUDA_SASS_DETOUR_JITLINK_PTX_THREADMAP=1` injects PTX code that writes
    #   `%smid` into bpftime's sampler buffer (u32 slots), controlled by module globals.
    # - bpftime binds those globals at `cuLaunchKernel` time, and then dump-on-sync
    #   emits standard `bpftime_sass_thread` JSONL records.
    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(
            r"""
extern "C" __global__ void bpftime_ptx_threadmap_kernel() {
  // bpftime will inject threadmap stores before JIT-link
}
"""
        )

    nvcc_env = dict(os.environ)
    nvcc_env.pop("LD_PRELOAD", None)
    subprocess.check_call(
        [
            nvcc,
            "--fatbin",
            "-O0",
            "-lineinfo",
            "-g",
            "-arch=compute_80",
            "-code=compute_80",
            cu_path,
            "-o",
            fatbin_path,
        ],
        env=nvcc_env,
    )

    with open(fatbin_path, "rb") as f:
        fatbin = f.read()
    fatbin_buf = ctypes.create_string_buffer(fatbin)

    FATBINC_MAGIC = 0x466243B1
    wrapper = FatbincWrapper()
    wrapper.magic = FATBINC_MAGIC
    wrapper.version = 1
    wrapper.data = ctypes.cast(fatbin_buf, c_void_p)
    wrapper.filename_or_fatbins = c_void_p(0)

    cu = ctypes.CDLL("libcuda.so.1")
    ck(cu.cuInit(0), "cuInit")

    dev = CUdevice()
    ck(cu.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    ctx = CUcontext()
    ck(cu.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "cuCtxCreate_v2")

    cu.cuLibraryLoadData.restype = c_int
    cu.cuLibraryLoadData.argtypes = [
        ctypes.POINTER(CUlibrary),
        c_void_p,
        c_void_p,
        c_void_p,
        c_uint,
        c_void_p,
        c_void_p,
        c_uint,
    ]
    lib = CUlibrary()
    ck(
        cu.cuLibraryLoadData(
            ctypes.byref(lib),
            ctypes.byref(wrapper),
            None,
            None,
            0,
            None,
            None,
            0,
        ),
        "cuLibraryLoadData",
    )

    cu.cuLibraryGetKernel.restype = c_int
    cu.cuLibraryGetKernel.argtypes = [ctypes.POINTER(CUkernel), CUlibrary, c_char_p]
    kernel = CUkernel()
    ck(
        cu.cuLibraryGetKernel(
            ctypes.byref(kernel), lib, b"bpftime_ptx_threadmap_kernel"
        ),
        "cuLibraryGetKernel",
    )

    cu.cuKernelGetFunction.restype = c_int
    cu.cuKernelGetFunction.argtypes = [ctypes.POINTER(CUfunction), CUkernel]
    func = CUfunction()
    ck(cu.cuKernelGetFunction(ctypes.byref(func), kernel), "cuKernelGetFunction")

    # Launch with blockDim.x=256 so lane0 threads are tid.x=0,32,...,224.
    cu.cuLaunchKernel.restype = c_int
    cu.cuLaunchKernel.argtypes = [
        CUfunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    ck(
        cu.cuLaunchKernel(
            func,
            1,
            1,
            1,
            256,
            1,
            1,
            0,
            None,
            None,
            None,
        ),
        "cuLaunchKernel",
    )
    ck(cu.cuCtxSynchronize(), "cuCtxSynchronize")

    expected = [0, 32, 64, 96, 128, 160, 192, 224]
    if not os.path.exists(dump_path):
        raise RuntimeError(
            f"missing JSONL dump at {dump_path}; did you enable BPFTIME_CUDA_SASS_SAMPLE_DUMP_ON_SYNC=1?"
        )

    have_tids = set()
    have_meta = False
    with open(dump_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "bpftime_sass_sample_meta":
                have_meta = True
            if obj.get("type") != "bpftime_sass_thread":
                continue
            if obj.get("ctaid_x") != 0:
                continue
            tid_x = obj.get("tid_x")
            if isinstance(tid_x, int):
                have_tids.add(tid_x)

    missing = [tid for tid in expected if tid not in have_tids]
    if missing:
        raise RuntimeError(f"missing expected tid_x in JSONL dump: {missing}")
    if not have_meta:
        raise RuntimeError("missing bpftime_sass_sample_meta in JSONL dump")

    print("ok: lane0-only bpftime_sass_thread records for", expected)

    cu.cuCtxDestroy_v2.restype = c_int
    cu.cuCtxDestroy_v2.argtypes = [CUcontext]
    ck(cu.cuCtxDestroy_v2(ctx), "cuCtxDestroy_v2")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
