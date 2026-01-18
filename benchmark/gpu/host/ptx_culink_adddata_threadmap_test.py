#!/usr/bin/env python3

import json
import os
import subprocess
import sys
from ctypes import (
    CDLL,
    POINTER,
    byref,
    c_char_p,
    c_int,
    c_size_t,
    c_uint,
    c_void_p,
    create_string_buffer,
)
import ctypes


CUDA_SUCCESS = 0
CU_JIT_INPUT_PTX = 1


def ck(res, name):
    if res != CUDA_SUCCESS:
        raise RuntimeError(f"{name} failed: {res}")


class CUdevice(ctypes.c_int):
    pass


class CUcontext(ctypes.c_void_p):
    pass


class CUmodule(ctypes.c_void_p):
    pass


class CUfunction(ctypes.c_void_p):
    pass


class CUlinkState(ctypes.c_void_p):
    pass


def _find_link_api(cu, base: str):
    v2 = f"{base}_v2"
    if hasattr(cu, v2):
        return getattr(cu, v2)
    return getattr(cu, base)


def main():
    nvcc = os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc")
    if not os.path.exists(nvcc):
        nvcc = "/usr/local/cuda-12.9/bin/nvcc"

    dump_path = os.environ.get(
        "BPFTIME_CUDA_SASS_SAMPLE_DUMP_PATH",
        "/tmp/bpftime-ptx-culink-adddata-threadmap.jsonl",
    )

    workdir = "/tmp/bpftime-ptx-culink-adddata-threadmap"
    os.makedirs(workdir, exist_ok=True)
    cu_path = os.path.join(workdir, "k.cu")
    ptx_path = os.path.join(workdir, "k.ptx")

    kernel_name = "bpftime_culink_adddata_kernel"
    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(
            rf"""
extern "C" __global__ void {kernel_name}() {{
  // empty; bpftime injects PTX thread-map via cuLinkAddData hook
}}
"""
        )

    nvcc_env = dict(os.environ)
    nvcc_env.pop("LD_PRELOAD", None)
    subprocess.check_call(
        [
            nvcc,
            "-ptx",
            "-O0",
            "-lineinfo",
            "-g",
            "-arch=compute_80",
            cu_path,
            "-o",
            ptx_path,
        ],
        env=nvcc_env,
    )

    ptx = open(ptx_path, "rb").read()
    if not ptx.endswith(b"\0"):
        ptx += b"\0"
    ptx_buf = create_string_buffer(ptx)

    cu = CDLL("libcuda.so.1")
    ck(cu.cuInit(0), "cuInit")
    dev = CUdevice()
    ck(cu.cuDeviceGet(byref(dev), 0), "cuDeviceGet")
    ctx = CUcontext()
    ck(cu.cuCtxCreate_v2(byref(ctx), 0, dev), "cuCtxCreate_v2")

    cuLinkCreate = _find_link_api(cu, "cuLinkCreate")
    cuLinkAddData = _find_link_api(cu, "cuLinkAddData")
    cuLinkComplete = _find_link_api(cu, "cuLinkComplete")
    cuLinkDestroy = _find_link_api(cu, "cuLinkDestroy")

    cuLinkCreate.restype = c_int
    cuLinkCreate.argtypes = [c_uint, c_void_p, c_void_p, POINTER(CUlinkState)]
    cuLinkAddData.restype = c_int
    cuLinkAddData.argtypes = [
        CUlinkState,
        c_int,  # CUjitInputType
        c_void_p,
        c_size_t,
        c_char_p,
        c_uint,
        c_void_p,
        c_void_p,
    ]
    cuLinkComplete.restype = c_int
    cuLinkComplete.argtypes = [CUlinkState, POINTER(c_void_p), POINTER(c_size_t)]
    cuLinkDestroy.restype = c_int
    cuLinkDestroy.argtypes = [CUlinkState]

    state = CUlinkState()
    ck(cuLinkCreate(0, None, None, byref(state)), "cuLinkCreate")
    ck(
        cuLinkAddData(
            state,
            CU_JIT_INPUT_PTX,
            ctypes.cast(ptx_buf, c_void_p),
            c_size_t(len(ptx)),
            b"k.ptx",
            0,
            None,
            None,
        ),
        "cuLinkAddData(PTX)",
    )
    cubin_ptr = c_void_p()
    cubin_sz = c_size_t()
    ck(cuLinkComplete(state, byref(cubin_ptr), byref(cubin_sz)), "cuLinkComplete")
    if not cubin_ptr.value or cubin_sz.value == 0:
        raise RuntimeError("cuLinkComplete produced empty cubin")

    cu.cuModuleLoadData.restype = c_int
    cu.cuModuleLoadData.argtypes = [POINTER(CUmodule), c_void_p]
    mod = CUmodule()
    ck(cu.cuModuleLoadData(byref(mod), cubin_ptr), "cuModuleLoadData")

    cuModuleGetFunction = getattr(cu, "cuModuleGetFunction_v2", None)
    if cuModuleGetFunction is None:
        cuModuleGetFunction = cu.cuModuleGetFunction
    cuModuleGetFunction.restype = c_int
    cuModuleGetFunction.argtypes = [POINTER(CUfunction), CUmodule, c_char_p]
    func = CUfunction()
    ck(
        cuModuleGetFunction(byref(func), mod, kernel_name.encode("utf-8")),
        "cuModuleGetFunction",
    )

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

    # dump-on-sync must have produced JSONL
    if not os.path.exists(dump_path):
        raise RuntimeError(
            f"missing JSONL dump at {dump_path}; did you enable BPFTIME_CUDA_SASS_SAMPLE_DUMP_ON_SYNC=1?"
        )

    expected = [0, 32, 64, 96, 128, 160, 192, 224]
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

    print("ok: cuLinkAddData PTX injection produced bpftime_sass_thread for", expected)

    # cleanup
    cu.cuModuleUnload.restype = c_int
    cu.cuModuleUnload.argtypes = [CUmodule]
    cu.cuCtxDestroy_v2.restype = c_int
    cu.cuCtxDestroy_v2.argtypes = [CUcontext]
    ck(cu.cuModuleUnload(mod), "cuModuleUnload")
    ck(cuLinkDestroy(state), "cuLinkDestroy")
    ck(cu.cuCtxDestroy_v2(ctx), "cuCtxDestroy_v2")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
