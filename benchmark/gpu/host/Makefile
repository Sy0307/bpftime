# Convenience wrapper; uses the repo's CMake build to resolve link deps.

BPFTIME_REPO_ROOT := $(abspath ../../..)

.PHONY: all clean

all:
	cmake --build $(BPFTIME_REPO_ROOT)/build -j --target gpu_array_map_host_perf
	ln -sf $(BPFTIME_REPO_ROOT)/build/benchmark/gpu/host/gpu_array_map_host_perf ./gpu_array_map_host_perf

clean:
	rm -f gpu_array_map_host_perf
