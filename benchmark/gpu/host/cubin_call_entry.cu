#include <cuda.h>

__device__ __noinline__ int foo(int x)
{
	return x + 1;
}

extern "C" __global__ void call_entry(int *out)
{
	// Intentionally call a non-inlined device function early in the kernel.
	// With `BPFTIME_CUDA_SASS_DETOUR_REPLAY_N` > 1, the kernel entry replay
	// window can include the CALL instruction, exercising bpftime's SASS
	// control-flow relocation.
	int x = foo((int)threadIdx.x);
	if (threadIdx.x == 0)
		out[0] = x;
}

