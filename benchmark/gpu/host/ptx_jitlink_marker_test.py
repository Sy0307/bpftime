#!/usr/bin/env python3

import ctypes
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


class Dim3(ctypes.Structure):
    _fields_ = [("x", c_uint), ("y", c_uint), ("z", c_uint)]


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

    workdir = "/tmp/bpftime-ptx-jitlink-marker"
    os.makedirs(workdir, exist_ok=True)
    cu_path = os.path.join(workdir, "k.cu")
    fatbin_path = os.path.join(workdir, "k.fatbin")

    # PTX-only fatbin (no SASS), so bpftime's JIT-link PTX path is exercised.
    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(
            r"""
extern "C" __global__ void bpftime_ptx_marker_kernel() {
  // do nothing; bpftime will inject a store into PTX before JIT-link
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

    # cuLibraryLoadData expects a pointer to code; vLLM commonly passes a FATBINC wrapper.
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
            ctypes.byref(kernel), lib, b"bpftime_ptx_marker_kernel"
        ),
        "cuLibraryGetKernel",
    )

    cu.cuKernelGetFunction.restype = c_int
    cu.cuKernelGetFunction.argtypes = [ctypes.POINTER(CUfunction), CUkernel]
    func = CUfunction()
    ck(cu.cuKernelGetFunction(ctypes.byref(func), kernel), "cuKernelGetFunction")

    # Launch
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
            1,
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

    # Read the injected marker symbol from the module.
    cu.cuLibraryGetModule.restype = c_int
    cu.cuLibraryGetModule.argtypes = [ctypes.POINTER(CUmodule), CUlibrary]
    mod = CUmodule()
    ck(cu.cuLibraryGetModule(ctypes.byref(mod), lib), "cuLibraryGetModule")

    cu.cuModuleGetGlobal_v2.restype = c_int
    cu.cuModuleGetGlobal_v2.argtypes = [
        ctypes.POINTER(CUdeviceptr),
        ctypes.POINTER(c_size_t),
        CUmodule,
        c_char_p,
    ]
    gptr = CUdeviceptr()
    gsz = c_size_t()
    ck(
        cu.cuModuleGetGlobal_v2(
            ctypes.byref(gptr),
            ctypes.byref(gsz),
            mod,
            b"__bpftime_ptx_marker",
        ),
        "cuModuleGetGlobal_v2(__bpftime_ptx_marker)",
    )
    if gsz.value < 4:
        raise RuntimeError(f"marker size too small: {gsz.value}")

    cu.cuMemcpyDtoH_v2.restype = c_int
    cu.cuMemcpyDtoH_v2.argtypes = [c_void_p, CUdeviceptr, c_size_t]
    out = (ctypes.c_uint32 * 1)()
    ck(cu.cuMemcpyDtoH_v2(out, gptr, c_size_t(4)), "cuMemcpyDtoH_v2")
    print(f"__bpftime_ptx_marker={out[0]}")
    if out[0] != 1:
        raise RuntimeError("marker mismatch (expected 1)")

    # Cleanup
    cu.cuCtxDestroy_v2.restype = c_int
    cu.cuCtxDestroy_v2.argtypes = [CUcontext]
    ck(cu.cuCtxDestroy_v2(ctx), "cuCtxDestroy_v2")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
