#ifndef BPFTIME_CUDA_CONTEXT_HELPERS_HPP
#define BPFTIME_CUDA_CONTEXT_HELPERS_HPP

#include "cuda.h"
#include "spdlog/spdlog.h"
#include <mutex>
#include <stdexcept>

namespace bpftime::cuda_utils
{
inline void cuda_throw_on_error(CUresult err, const char *message)
{
	if (err != CUDA_SUCCESS) {
		SPDLOG_ERROR("{}: {}", message, (int)err);
		throw std::runtime_error(message);
	}
}

inline CUcontext get_or_create_primary_context()
{
	static std::once_flag init_flag;
	static CUcontext primary_ctx = nullptr;
	static CUdevice primary_device = 0;
	std::call_once(init_flag, []() {
		if (auto err = cuInit(0); err != CUDA_SUCCESS) {
			SPDLOG_ERROR("cuInit failed while creating primary context: {}",
				     (int)err);
			throw std::runtime_error("cuInit failed");
		}
		cuda_throw_on_error(cuDeviceGet(&primary_device, 0),
				    "cuDeviceGet failed");
		cuda_throw_on_error(
			cuDevicePrimaryCtxRetain(&primary_ctx, primary_device),
			"cuDevicePrimaryCtxRetain failed");
		SPDLOG_INFO(
			"Retained CUDA primary context {} for GPU map management",
			(void *)primary_ctx);
	});
	return primary_ctx;
}

class scoped_primary_ctx
{
	bool pushed = false;

    public:
	explicit scoped_primary_ctx(CUcontext ctx)
	{
		if (!ctx)
			return;
		cuda_throw_on_error(cuCtxPushCurrent(ctx),
				    "cuCtxPushCurrent failed");
		pushed = true;
	}
	~scoped_primary_ctx()
	{
		if (pushed) {
			CUcontext popped = nullptr;
			if (auto err = cuCtxPopCurrent(&popped);
			    err != CUDA_SUCCESS) {
				SPDLOG_WARN("cuCtxPopCurrent failed: {}",
					    (int)err);
			}
		}
	}
};
} // namespace bpftime::cuda_utils

#endif
