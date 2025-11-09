#include "nv_attach_fatbin_record.hpp"
#include "cuda.h"
#include "spdlog/spdlog.h"
#include "nv_attach_impl.hpp"
#include <array>
#include <cctype>
#include <cstdint>
#include <optional>
#include <string_view>
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
namespace
{
struct sm_target
{
	int value;
	bool accelerated;
};

std::optional<sm_target> parse_sm_number(std::string_view token)
{
	if (!token.starts_with("sm_"))
		return std::nullopt;
	token.remove_prefix(3);
	if (token.empty())
		return std::nullopt;
	bool accelerated = false;
	if (auto last = token.back(); last == 'a' || last == 'A') {
		accelerated = true;
		token.remove_suffix(1);
	}
	if (token.empty())
		return std::nullopt;
	int value = 0;
	for (char c : token) {
		if (!std::isdigit(static_cast<unsigned char>(c)))
			return std::nullopt;
		value = value * 10 + (c - '0');
	}
	if (accelerated && value != 90)
		return std::nullopt;
	return sm_target{ value, accelerated };
}

std::optional<CUjit_target> to_jit_target(int value, bool accelerated)
{
	if (accelerated) {
		if (value == 90)
			return CU_TARGET_COMPUTE_90A;
		return std::nullopt;
	}
	switch (value) {
	case 30:
		return CU_TARGET_COMPUTE_30;
	case 32:
		return CU_TARGET_COMPUTE_32;
	case 35:
		return CU_TARGET_COMPUTE_35;
	case 37:
		return CU_TARGET_COMPUTE_37;
	case 50:
		return CU_TARGET_COMPUTE_50;
	case 52:
		return CU_TARGET_COMPUTE_52;
	case 53:
		return CU_TARGET_COMPUTE_53;
	case 60:
		return CU_TARGET_COMPUTE_60;
	case 61:
		return CU_TARGET_COMPUTE_61;
	case 62:
		return CU_TARGET_COMPUTE_62;
	case 70:
		return CU_TARGET_COMPUTE_70;
	case 72:
		return CU_TARGET_COMPUTE_72;
	case 75:
		return CU_TARGET_COMPUTE_75;
	case 80:
		return CU_TARGET_COMPUTE_80;
	case 86:
		return CU_TARGET_COMPUTE_86;
	case 87:
		return CU_TARGET_COMPUTE_87;
	case 89:
		return CU_TARGET_COMPUTE_89;
	case 90:
		return CU_TARGET_COMPUTE_90;
	default:
		return std::nullopt;
	}
}

std::optional<CUjit_target> find_sm_target(std::string_view text)
{
	const std::string_view marker = "sm_";
	auto pos = text.find(marker);
	while (pos != std::string_view::npos) {
		if (pos > 0 &&
		    std::isalnum(
			    static_cast<unsigned char>(text[pos - 1]))) {
			pos = text.find(marker, pos + marker.size());
			continue;
		}
		auto current = pos + marker.size();
		while (current < text.size() &&
		       std::isdigit(static_cast<unsigned char>(text[current]))) {
			current++;
		}
		bool accelerated = false;
		if (current < text.size() &&
		    (text[current] == 'a' || text[current] == 'A')) {
			accelerated = true;
			current++;
		}
		if (auto token = text.substr(pos, current - pos);
		    token.size() > marker.size()) {
			if (auto parsed = parse_sm_number(token)) {
				if (auto jt = to_jit_target(parsed->value,
							    parsed->accelerated))
					return jt;
			}
		}
		pos = text.find(marker, current);
	}
	return std::nullopt;
}

std::optional<CUjit_target> deduce_jit_target(const std::string &module_name,
					      const std::string &ptx_text)
{
	if (auto from_name = find_sm_target(module_name))
		return from_name;
	return find_sm_target(ptx_text);
}
} // namespace
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
		char error_buf[8192]{}, info_buf[8192]{};
			std::array<CUjit_option, 6> options;
			std::array<void *, 6> option_values;
			size_t option_count = 0;
			options[option_count] = CU_JIT_INFO_LOG_BUFFER;
			option_values[option_count++] = (void *)info_buf;
			options[option_count] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
			option_values[option_count++] =
				reinterpret_cast<void *>(
					static_cast<uintptr_t>(
						sizeof(info_buf)));
			options[option_count] = CU_JIT_ERROR_LOG_BUFFER;
			option_values[option_count++] = (void *)error_buf;
			options[option_count] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
			option_values[option_count++] =
				reinterpret_cast<void *>(
					static_cast<uintptr_t>(
						sizeof(error_buf)));
		unsigned int target_value = 0;
		if (auto target = deduce_jit_target(name, ptx)) {
			target_value = static_cast<unsigned int>(*target);
				options[option_count] = CU_JIT_TARGET;
				option_values[option_count++] =
					reinterpret_cast<void *>(
						static_cast<uintptr_t>(
							target_value));
			SPDLOG_DEBUG("Using CU_JIT_TARGET={} for {}", target_value,
				     name);
		} else {
			options[option_count] = CU_JIT_TARGET_FROM_CUCONTEXT;
			option_values[option_count++] = nullptr;
		}
		if (auto err = cuModuleLoadDataEx(&module, ptx.data(),
						  option_count, options.data(),
						  option_values.data());
		    err != CUDA_SUCCESS) {
			SPDLOG_ERROR("Unable to compile module {}: {}", name,
				     (int)err);
			SPDLOG_ERROR("Info: {}", info_buf);
			SPDLOG_ERROR("Error: {}", error_buf);
			throw std::runtime_error("Unable to compile module");
		}
		CUdeviceptr const_data_ptr, map_basic_info_ptr;
		size_t const_data_size, map_basic_info_size;
		SPDLOG_INFO("Copying trampoline data to device");
		CUDA_DRIVER_CHECK_EXCEPTION(
			cuModuleGetGlobal(&const_data_ptr, &const_data_size,
					  module, "constData"),
			"Unable to get pointer of constData");
		CUDA_DRIVER_CHECK_EXCEPTION(
			cuModuleGetGlobal(&map_basic_info_ptr,
					  &map_basic_info_size, module,
					  "map_info"),
			"Unable to get pointer of map_info");
		CUDA_DRIVER_CHECK_EXCEPTION(
			cuMemcpyHtoD(const_data_ptr, &impl.shared_mem_ptr,
				     const_data_size),
			"Unable to copy constData pointer to device");
		CUDA_DRIVER_CHECK_EXCEPTION(
			cuMemcpyHtoD(map_basic_info_ptr,
				     impl.map_basic_info->data(),
				     map_basic_info_size),
			"Unable to copy constData pointer to device");
		SPDLOG_INFO("Trampoline data copied");
		ptxs.emplace_back(
			std::make_unique<fatbin_record::ptx_in_module>(module));
		SPDLOG_INFO("Loaded module: {}", name);
	}
	ptx_loaded = true;
}

} // namespace bpftime::attach
