#include "nv_attach_fatbin_record.hpp"
#include "cuda.h"
#include "spdlog/spdlog.h"
#include "nv_attach_impl.hpp"
#include <algorithm>
#define CUDA_DRIVER_CHECK_NO_EXCEPTION(expr, message)                          \
	do {                                                                   \
		if (auto err = expr; err != CUDA_SUCCESS) {                    \
			SPDLOG_ERROR("{}: {}", message, (int)err);             \
		}                                                              \
	} while (false)
#define CUDA_DRIVER_CHECK_EXCEPTION(expr, message)                             \
	do {                                                                   \
		if (auto err = expr; err != CUDA_SUCCESS) {                    \
			SPDLOG_ERROR("{}: {}", message, (int)err);             \
			throw std::runtime_error(message);                     \
		}                                                              \
	} while (false)

namespace bpftime::attach
{
fatbin_record::~fatbin_record()
{
}
fatbin_record::ptx_in_module::~ptx_in_module()
{
	CUDA_DRIVER_CHECK_NO_EXCEPTION(cuModuleUnload(this->module_ptr),
				       "Unable to unload module");
}

bool fatbin_record::find_and_fill_variable_info(void *ptr,
						const char *symbol_name)
{
	for (const auto &ptx : ptxs) {
		CUdeviceptr dptr;
		size_t size;
		auto err = cuModuleGetGlobal(&dptr, &size, ptx->module_ptr,
					     symbol_name);
		if (err == CUDA_SUCCESS) {
			variable_addr_to_symbol[ptr] =
				variable_info{ .symbol_name =
						       std::string(symbol_name),
					       .ptr = dptr,
					       .size = size,
					       .ptx = ptx.get() };
			return true;
		} else if (err == CUDA_ERROR_NOT_FOUND) {
			continue;
		} else {
			SPDLOG_ERROR("Unable to lookup symbol: {}", (int)err);
			return false;
		}
	}
	return false;
}
bool fatbin_record::find_and_fill_function_info(void *ptr,
						const char *symbol_name)
{
	for (const auto &ptx : ptxs) {
		CUfunction func;
		auto err = cuModuleGetFunction(&func, ptx->module_ptr,
					       symbol_name);
		if (err == CUDA_SUCCESS) {
			function_addr_to_symbol[ptr] =
				kernel_info{ .symbol_name =
						     std::string(symbol_name),
					     .func = func,
					     .ptx = ptx.get() };
			return true;
		} else if (err == CUDA_ERROR_NOT_FOUND) {
			continue;
		} else {
			SPDLOG_ERROR("Unable to lookup function: {}", (int)err);
			return false;
		}
	}
	return false;
}

void fatbin_record::try_loading_ptxs(class nv_attach_impl &impl)
{
	if (ptx_loaded)
		return;
	SPDLOG_INFO("Loading & patching current fatbin..");
	auto patched_ptx = *impl.hack_fatbin(original_ptx);

	for (const auto &[name, ptx] : patched_ptx) {
		CUmodule module;
		SPDLOG_INFO("Loading module: {}", name);
		std::vector<char> compiled_binary;
		try {
			compiled_binary = impl.compile_ptx_to_cubin(ptx);
		} catch (const std::exception &e) {
			SPDLOG_ERROR(
				"Failed to compile patched PTX {}: {}",
				name, e.what());
			throw;
		}
		char error_buf[8192], info_buf[8192];
		CUjit_option options[] = { CU_JIT_INFO_LOG_BUFFER,
					   CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
					   CU_JIT_ERROR_LOG_BUFFER,
					   CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES };
		void *option_values[] = { (void *)info_buf,
					  (void *)std::size(info_buf),
					  (void *)error_buf,
					  (void *)std::size(error_buf) };
		if (auto err = cuModuleLoadDataEx(&module, compiled_binary.data(),
						  std::size(options), options,
						  option_values);
		    err != CUDA_SUCCESS) {
			SPDLOG_ERROR("Unable to load module {}: {}", name,
				     (int)err);
			SPDLOG_ERROR("Info: {}", info_buf);
			SPDLOG_ERROR("Error: {}", error_buf);
			throw std::runtime_error("Unable to load module");
		}
		SPDLOG_INFO("Copying helper constants to device");
		auto copy_symbol_variants =
			[&](std::initializer_list<const char *> symbols,
			    const void *host_src, size_t host_size,
			    const char *label) {
				for (const auto *symbol : symbols) {
					CUdeviceptr device_ptr;
					size_t device_size;
					auto status =
						cuModuleGetGlobal(&device_ptr,
								  &device_size,
								  module,
								  symbol);
					if (status == CUDA_ERROR_NOT_FOUND)
						continue;
					if (status != CUDA_SUCCESS) {
						auto msg = std::string(
								   "Unable to get pointer of ") +
							   symbol;
						SPDLOG_ERROR(
							"{} (while copying {}): {}",
							msg, label, (int)status);
						throw std::runtime_error(msg);
					}
					size_t copy_size =
						std::min(device_size,
							 host_size);
					auto copy_status = cuMemcpyHtoD(
						device_ptr, host_src,
						copy_size);
					if (copy_status != CUDA_SUCCESS) {
						auto msg = std::string(
								   "Unable to copy symbol ") +
							   symbol;
						SPDLOG_ERROR(
							"{} (while copying {}): {}",
							msg, label,
							(int)copy_status);
						throw std::runtime_error(msg);
					}
					SPDLOG_INFO(
						"Copied {} bytes to symbol {} for {}",
						copy_size, symbol, label);
					return;
				}
				SPDLOG_INFO(
					"No matching symbols found for {} in module {}, skipping copy",
					label, name);
			};
		copy_symbol_variants(
			{ INLINE_CONST_PTR_SYMBOL, INLINE_CONST_PTR_COMPAT_SYMBOL,
			  LEGACY_CONST_PTR_SYMBOL },
			&impl.shared_mem_ptr, sizeof(impl.shared_mem_ptr),
			"shared memory pointer");
		if (impl.map_basic_info.has_value() &&
		    !impl.map_basic_info->empty()) {
			copy_symbol_variants(
				{ INLINE_MAP_INFO_SYMBOL, LEGACY_MAP_INFO_SYMBOL },
				impl.map_basic_info->data(),
				impl.map_basic_info->size() *
					sizeof(MapBasicInfo),
				"map info");
		} else {
			SPDLOG_INFO(
				"No map_basic_info available, skipping map info copy");
		}
		SPDLOG_INFO("Helper constant copy finished");
		ptxs.emplace_back(
			std::make_unique<fatbin_record::ptx_in_module>(module));
		SPDLOG_INFO("Loaded module: {}", name);
	}
	ptx_loaded = true;
}

} // namespace bpftime::attach
