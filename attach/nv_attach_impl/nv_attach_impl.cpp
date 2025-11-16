#include "nv_attach_impl.hpp"
#include "driver_types.h"
#include "ebpf_inst.h"
#include "frida-gum.h"

#include "nvPTXCompiler.h"
#include "nv_attach_private_data.hpp"
#include "nv_attach_utils.hpp"
#include "spdlog/spdlog.h"
#include <asm/unistd.h> // For architecture-specific syscall numbers
#include <boost/asio/io_context.hpp>
#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <boost/process.hpp>
#include <boost/process/detail/child_decl.hpp>
#include <boost/process/env.hpp>
#include <boost/process/io.hpp>
#include <boost/process/pipe.hpp>
#include <boost/process/start_dir.hpp>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <cassert>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <filesystem>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <cctype>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <sys/user.h>
#include <sys/uio.h>
#include <link.h>
#include <unistd.h>
#include <variant>
#include <vector>
#include <atomic>
#include <boost/asio.hpp>
#include "ptxpass/core.hpp"
#include "ptx_pass_config.h"
using namespace bpftime;
using namespace attach;
namespace
{
struct RegisterOffsets {
	int reg32 = 0;
	int reg64 = 0;
	int reg16 = 0;
	int pred = 0;
};

std::string trim_string(const std::string &input)
{
	size_t start = input.find_first_not_of(" \t\r\n");
	if (start == std::string::npos)
		return "";
	size_t end = input.find_last_not_of(" \t\r\n");
	return input.substr(start, end - start + 1);
}

struct RegisterDeclLocation {
	bool found = false;
	int count = 0;
	size_t digits_pos = 0;
	size_t digits_len = 0;
};

RegisterDeclLocation find_register_decl_in_header(const std::string &ptx,
						  size_t header_begin,
						  size_t header_end,
						  const std::string &reg_prefix)
{
	RegisterDeclLocation info;
	if (header_begin == std::string::npos ||
	    header_end == std::string::npos || header_end <= header_begin)
		return info;
	size_t search_pos = header_begin;
	while (search_pos < header_end) {
		size_t reg_pos = ptx.find(".reg", search_pos);
		if (reg_pos == std::string::npos || reg_pos >= header_end)
			break;
		size_t prefix_pos = ptx.find(reg_prefix, reg_pos);
		if (prefix_pos == std::string::npos || prefix_pos >= header_end) {
			search_pos = reg_pos + 4;
			continue;
		}
		size_t lt_pos = prefix_pos + reg_prefix.size();
		if (lt_pos >= header_end || ptx[lt_pos] != '<') {
			search_pos = prefix_pos + reg_prefix.size();
			continue;
		}
		size_t digits_start = lt_pos + 1;
		size_t digits_end = digits_start;
		while (digits_end < header_end &&
		       std::isdigit(
			       static_cast<unsigned char>(ptx[digits_end]))) {
			digits_end++;
		}
		if (digits_end == digits_start) {
			search_pos = digits_end;
			continue;
		}
		info.found = true;
		info.count = std::stoi(ptx.substr(digits_start,
						  digits_end - digits_start));
		info.digits_pos = digits_start;
		info.digits_len = digits_end - digits_start;
		return info;
	}
	return info;
}

bool is_identifier_char(char c)
{
	return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') ||
	       (c >= 'a' && c <= 'z') || c == '_' || c == '.' || c == '$';
}

RegisterOffsets grow_kernel_register_usage(
	std::string &ptx, const std::string &kernel,
	const ptxpass::runtime_response::InlineRegisterUsage &usage)
{
	RegisterOffsets offsets{};
	if (usage.reg32 == 0 && usage.reg64 == 0 && usage.pred == 0)
		return offsets;
	auto kernel_range = ptxpass::find_kernel_body(ptx, kernel);
	if (kernel_range.first == std::string::npos) {
		SPDLOG_WARN(
			"Unable to grow registers for kernel {} (body not found)",
			kernel);
		return offsets;
	}
	auto process_decl = [&](const std::string &reg_prefix, int usage_count,
				int &offset_store, const char *type_name) {
		if (usage_count <= 0)
			return;
		size_t header_end = kernel_range.second;
		if (header_end == std::string::npos)
			header_end = ptx.size();
		auto decl = find_register_decl_in_header(ptx, kernel_range.first,
							 header_end, reg_prefix);
		if (!decl.found) {
			if (reg_prefix == "%rs") {
				size_t brace =
					ptx.find('{', kernel_range.first);
				if (brace == std::string::npos) {
					SPDLOG_WARN(
						"Kernel {} missing {} register declaration required for inline block and unable to insert",
						kernel, type_name);
					return;
				}
				std::string insertion = "    .reg .b16 \t%rs<" +
							std::to_string(
								usage_count) +
							">;\n";
				ptx.insert(brace + 1, insertion);
				offset_store = 0;
				return;
			}
			SPDLOG_WARN(
				"Kernel {} missing {} register declaration required for inline block",
				kernel, type_name);
			return;
		}
		offset_store = decl.count;
		int new_value = decl.count + usage_count;
		std::string digits = std::to_string(new_value);
		ptx.replace(decl.digits_pos, decl.digits_len, digits);
	};

	process_decl("%r", usage.reg32, offsets.reg32, "%r");
	process_decl("%rd", usage.reg64, offsets.reg64, "%rd");
	process_decl("%rs", usage.reg16, offsets.reg16, "%rs");
	process_decl("%p", usage.pred, offsets.pred, "%p");
	return offsets;
}

std::string append_inline_identifier_suffix(const std::string &input,
					    const std::string &suffix)
{
	if (suffix.empty())
		return input;
	std::string output = input;
	auto append_suffix_for_prefix = [&](const std::string &prefix) {
		size_t search_pos = 0;
		const std::string suffix_marker = "__i" + suffix;
		while (true) {
			auto pos = output.find(prefix, search_pos);
			if (pos == std::string::npos)
				break;
			size_t name_end = pos + prefix.size();
			while (name_end < output.size() &&
			       is_identifier_char(output[name_end])) {
				name_end++;
			}
			if (name_end >= 3 &&
			    output.compare(name_end - 3, 3, "__i") == 0) {
				search_pos = name_end;
				continue;
			}
			output.insert(name_end, suffix_marker);
			search_pos = name_end + suffix_marker.size();
		}
	};
	append_suffix_for_prefix("__bpftime_inline_local_");
	append_suffix_for_prefix("$bpftime_inline_");
	return output;
}

std::vector<std::string> append_inline_identifier_suffix(
	const std::vector<std::string> &inputs, const std::string &suffix)
{
	if (suffix.empty())
		return inputs;
	std::vector<std::string> result;
	result.reserve(inputs.size());
	for (const auto &line : inputs) {
		result.push_back(
			append_inline_identifier_suffix(line, suffix));
	}
	return result;
}

void insert_kernel_local_decls(std::string &ptx, const std::string &kernel,
			       const std::vector<std::string> &decls)
{
	if (decls.empty())
		return;
	auto kernel_range = ptxpass::find_kernel_body(ptx, kernel);
	if (kernel_range.first == std::string::npos) {
		SPDLOG_WARN(
			"Unable to insert local declarations for kernel {}",
			kernel);
		return;
	}
	size_t brace = ptx.find('{', kernel_range.first);
	if (brace == std::string::npos)
		return;
	size_t insert_pos = brace + 1;
	if (insert_pos < ptx.size() && ptx[insert_pos] == '\n')
		insert_pos++;
	auto starts_with = [](const std::string &value, const char *prefix) {
		return value.rfind(prefix, 0) == 0;
	};
	auto ends_with = [](const std::string &value, char c) {
		return !value.empty() && value.back() == c;
	};
	while (insert_pos < ptx.size()) {
		size_t line_end = ptx.find('\n', insert_pos);
		if (line_end == std::string::npos)
			break;
		auto line = ptx.substr(insert_pos, line_end - insert_pos);
		auto trimmed = trim_string(line);
		if (trimmed.empty() || starts_with(trimmed, ".reg") ||
		    starts_with(trimmed, ".local") ||
		    starts_with(trimmed, ".shared") ||
		    starts_with(trimmed, ".param")) {
			insert_pos = line_end + 1;
			continue;
		}
		break;
	}
	std::ostringstream oss;
	for (const auto &decl : decls) {
		if (decl.empty())
			continue;
		oss << "    " << decl;
		if (!ends_with(decl, ';'))
			oss << ";";
		oss << "\n";
	}
	ptx.insert(insert_pos, oss.str());
}

std::string apply_register_offsets_to_text(const std::string &text,
					   const RegisterOffsets &offsets)
{
	auto apply_single = [](const std::string &input, const std::regex &pattern,
			       int offset, const char *prefix) {
		if (offset == 0)
			return input;
		std::string output;
		output.reserve(input.size());
		size_t last = 0;
		for (std::sregex_iterator it(input.begin(), input.end(), pattern),
		     end;
		     it != end; ++it) {
			output.append(input, last, it->position() - last);
			int original = std::stoi((*it)[1]);
			output.append(prefix);
			output.append(std::to_string(original + offset));
			last = it->position() + it->length();
		}
		output.append(input, last, std::string::npos);
		return output;
	};
	std::string result = text;
	static const std::regex rd_pattern("%rd(\\d+)");
	static const std::regex r_pattern("%r(?!d)(\\d+)");
	static const std::regex rs_pattern("%rs(\\d+)");
	static const std::regex p_pattern("%p(\\d+)");
	result = apply_single(result, rd_pattern, offsets.reg64, "%rd");
	result = apply_single(result, r_pattern, offsets.reg32, "%r");
	result = apply_single(result, rs_pattern, offsets.reg16, "%rs");
	result = apply_single(result, p_pattern, offsets.pred, "%p");
	return result;
}

std::string indent_block(const std::string &text, const std::string &indent)
{
	if (text.empty())
		return text;
	std::ostringstream oss;
	std::istringstream iss(text);
	std::string line;
	bool first_line = true;
	while (std::getline(iss, line)) {
		if (!first_line)
			oss << "\n";
		first_line = false;
		if (!line.empty())
			oss << indent << line;
		else
			oss << indent;
	}
	return oss.str();
}

} // namespace

static std::vector<std::filesystem::path> split_by_colon(const std::string &str)
{
	std::vector<std::filesystem::path> result;

	char *buffer = new char[str.length() + 1];
	strcpy(buffer, str.c_str());

	char *token = strtok(buffer, ":");
	while (token != nullptr) {
		result.push_back(token);
		token = strtok(nullptr, ":");
	}

	delete[] buffer;
	return result;
}
#define CUDA_DRIVER_CHECK_NO_EXCEPTION(expr, message)                          \
	do {                                                                   \
		if (auto err = expr; err != CUDA_SUCCESS) {                    \
			SPDLOG_ERROR("{}: {}", message, (int)err);             \
		}                                                              \
	} while (false)

extern GType cuda_runtime_function_hooker_get_type();

int nv_attach_impl::detach_by_id(int id)
{
	SPDLOG_INFO("Detaching is not supported by nv_attach_impl");
	return 0;
}

void nv_attach_impl::register_custom_helpers(
	ebpf_helper_register_callback register_callback)
{
}

int nv_attach_impl::create_attach_with_ebpf_callback(
	ebpf_run_callback &&cb, const attach_private_data &private_data,
	int attach_type)
{
	auto data = dynamic_cast<const nv_attach_private_data &>(private_data);

	// Safely access the variant
	if (!std::holds_alternative<std::string>(data.code_addr_or_func_name)) {
		SPDLOG_ERROR(
			"code_addr_or_func_name does not hold a string value");
		return -1;
	}
	const auto &func_name =
		std::get<std::string>(data.code_addr_or_func_name);
	std::string attach_point_name;
	if (attach_type == ATTACH_CUDA_PROBE) {
		attach_point_name = "kprobe/" + func_name;
	} else if (attach_type == ATTACH_CUDA_RETPROBE) {
		attach_point_name = "kretprobe/" + func_name;
	} else {
		attach_point_name = func_name;
	}
	struct pass_cfg_with_exec_path *matched = nullptr;
	for (const auto &pd : this->pass_configurations) {
		if (pd->pass_config.attach_type != attach_type)
			continue;
		ptxpass::AttachPointMatcher matcher(
			pd->pass_config.attach_points);

		if (matcher.matches(attach_point_name)) {
			matched = pd.get();
			break; // pass_definitions is sorted deterministically
			       // by executable
		}
	}
	if (matched) {
		auto id = this->allocate_id();
		nv_attach_entry entry;
		entry.instuctions = data.instructions;
		entry.kernels = data.func_names;
		entry.program_name = data.program_name;
		entry.config = matched;
		entry.inline_enabled = data.inline_enabled;
		entry.inline_metadata = data.inline_metadata;
		entry.inline_fallback_enabled = data.inline_fallback_enabled;

		hook_entries[id] = std::move(entry);
		this->map_basic_info = data.map_basic_info;
		this->shared_mem_ptr = data.comm_shared_mem;
		SPDLOG_INFO("Recorded pass {} for func {}",
			    matched->executable_path.c_str(), func_name);
		return id;
	}
	// No matched definition: do not create generic entry; require explicit
	// pass definition to avoid ambiguous instrumentation.
	SPDLOG_WARN(
		"No pass definition matched for function {}, attach_type {}. Skipping.",
		attach_point_name, attach_type);
	return -1;
}

extern "C" {
cudaError_t cuda_runtime_function__cudaLaunchKernel(const void *func,
						    dim3 gridDim, dim3 blockDim,
						    void **args,
						    size_t sharedMem,
						    cudaStream_t stream);
}

nv_attach_impl::nv_attach_impl()
{
	SPDLOG_INFO("Starting nv_attach_impl");
	const char *sm_arch_env = std::getenv("BPFTIME_SM_ARCH");
	target_sm_arch = sm_arch_env ? sm_arch_env : "sm_60";
	SPDLOG_INFO("NVPTX target SM architecture: {}", target_sm_arch);
	gum_init_embedded();
	auto interceptor = gum_interceptor_obtain();
	if (interceptor == nullptr) {
		SPDLOG_ERROR("Failed to obtain Frida interceptor");
		throw std::runtime_error(
			"Failed to initialize Frida interceptor");
	}
	auto listener =
		g_object_new(cuda_runtime_function_hooker_get_type(), nullptr);
	if (listener == nullptr) {
		SPDLOG_ERROR("Failed to create Frida listener");
		throw std::runtime_error("Failed to initialize Frida listener");
	}
	this->frida_interceptor = interceptor;
	this->frida_listener = listener;
	gum_interceptor_begin_transaction(interceptor);

	auto register_hook = [&](AttachedToFunction func, void *addr) {
		if (addr == nullptr) {
			SPDLOG_WARN(
				"Skipping hook registration for function {} - symbol not found",
				(int)func);
			return;
		}
		auto ctx = std::make_unique<CUDARuntimeFunctionHookerContext>();
		ctx->to_function = func;
		ctx->impl = this;
		auto ctx_ptr = ctx.get();
		this->hooker_contexts.push_back(std::move(ctx));
		if (auto result = gum_interceptor_attach(
			    interceptor, (gpointer)addr,
			    (GumInvocationListener *)listener, ctx_ptr);
		    result != GUM_ATTACH_OK) {
			SPDLOG_ERROR(
				"Unable to attach to CUDA functions: func={}, err={}",
				(int)func, (int)result);
			throw std::runtime_error(
				"Failed to attach to CUDA function");
		}
	};

	{
		void *register_fatbin_addr =
			dlsym(RTLD_NEXT, "__cudaRegisterFatBinary");
		register_hook(AttachedToFunction::RegisterFatbin,
			      register_fatbin_addr);
	}
	{
		void *register_function_addr =
			GSIZE_TO_POINTER(gum_module_find_export_by_name(
				nullptr, "__cudaRegisterFunction"));
		register_hook(AttachedToFunction::RegisterFunction,
			      register_function_addr);
	}
	{
		void *register_variable_addr =
			GSIZE_TO_POINTER(gum_module_find_export_by_name(
				nullptr, "__cudaRegisterVar"));
		register_hook(AttachedToFunction::RegisterVariable,
			      register_variable_addr);
	}
	{
		void *register_fatbin_end_addr =
			GSIZE_TO_POINTER(gum_module_find_export_by_name(
				nullptr, "__cudaRegisterFatBinaryEnd"));
		register_hook(AttachedToFunction::RegisterFatbinEnd,
			      register_fatbin_end_addr);
	}

	{
		void *cuda_malloc_addr = GSIZE_TO_POINTER(
			gum_module_find_export_by_name(nullptr, "cudaMalloc"));
		register_hook(AttachedToFunction::CudaMalloc, cuda_malloc_addr);
	}
	{
		void *cuda_malloc_managed_addr =
			GSIZE_TO_POINTER(gum_module_find_export_by_name(
				nullptr, "cudaMallocManaged"));
		register_hook(AttachedToFunction::CudaMallocManaged,
			      cuda_malloc_managed_addr);
	}
	{
		void *cuda_memcpy_to_symbol_addr =
			GSIZE_TO_POINTER(gum_module_find_export_by_name(
				nullptr, "cudaMemcpyToSymbol"));
		register_hook(AttachedToFunction::CudaMemcpyToSymbol,
			      cuda_memcpy_to_symbol_addr);
	}
	{
		void *cuda_memcpy_to_symbol_async_addr =
			GSIZE_TO_POINTER(gum_module_find_export_by_name(
				nullptr, "cudaMemcpyToSymbolAsync"));
		register_hook(AttachedToFunction::CudaMemcpyToSymbolAsync,
			      cuda_memcpy_to_symbol_async_addr);
	}
	{
		void *cuda_launch_kernel_addr =
			GSIZE_TO_POINTER(gum_module_find_export_by_name(
				nullptr, "cudaLaunchKernel"));

		if (auto err = gum_interceptor_replace(
			    interceptor, cuda_launch_kernel_addr,
			    (gpointer)&cuda_runtime_function__cudaLaunchKernel,
			    this, nullptr);
		    err != GUM_REPLACE_OK) {
			SPDLOG_ERROR("Unable to replace cudaLaunchKernel: {}",
				     (int)err);
			assert(false);
		}
	}
	gum_interceptor_end_transaction(interceptor);

	static const char *ptx_pass_libraries = DEFAULT_PTX_PASS_EXECUTABLE;
	std::vector<std::filesystem::path> pass_libraries;
	{
		const char *provided_libraries =
			getenv("BPFTIME_PTXPASS_LIBRARIES");
		if (provided_libraries && strlen(provided_libraries) > 0) {
			ptx_pass_libraries = provided_libraries;
			SPDLOG_INFO(
				"Parsing user provided (by BPFTIME_PTXPASS_LIBRARIES) libraries: {}",
				provided_libraries);
		} else {
			SPDLOG_INFO("Parsing bundled libraries: {}",
				    ptx_pass_libraries);
		}
	}
	auto paths = split_by_colon(ptx_pass_libraries);
	for (const auto &path : paths) {
		SPDLOG_INFO("Found path: {}, executing..", path.c_str());
		void *handle = dlmopen(LM_ID_NEWLM, path.c_str(),
				       RTLD_NOW | RTLD_LOCAL);
		if (!handle) {
			SPDLOG_ERROR(
				"Unable to load dynamic library of pass {}: {}",
				path.c_str(), dlerror());
			continue;
		}
		auto print_config =
			(print_config_fn)dlsym(handle, "print_config");
		if (!print_config) {
			SPDLOG_ERROR("Symbol print_config not found in {}",
				     path.c_str());
			continue;
		}
		auto process_input =
			(process_input_fn)dlsym(handle, "process_input");
		if (!process_input) {
			SPDLOG_ERROR("Symbol process_input not found in {}",
				     path.c_str());
			continue;
		}
		ptxpass::pass_config::PassConfig config;
		std::vector<char> buf(10 << 20);
		print_config(buf.size(), buf.data());

		auto json = nlohmann::json::parse(buf.data());
		ptxpass::pass_config::from_json(json, config);
		SPDLOG_INFO("Retrived config of {}", path.c_str());
		SPDLOG_DEBUG("Config {}", json.dump(4));
		this->pass_configurations.emplace_back(
			std::make_unique<pass_cfg_with_exec_path>(
				path, config, print_config, process_input,
				handle));
	}
}

nv_attach_impl::~nv_attach_impl()
{
	if (frida_listener)
		g_object_unref(frida_listener);
}
std::map<std::string, std::string>
nv_attach_impl::extract_ptxs(std::vector<uint8_t> &&data_vec)
{
	std::map<std::string, std::string> all_ptx;
	char tmp_dir[] = "/tmp/bpftime-fatbin-work.XXXXXX";
	mkdtemp(tmp_dir);
	auto working_dir = std::filesystem::path(tmp_dir);
	auto fatbin_path = working_dir / "temp.fatbin";
	{
		std::ofstream ofs(fatbin_path, std::ios::binary);
		ofs.write((const char *)data_vec.data(), data_vec.size());
		SPDLOG_INFO("Temporary fatbin written to {}",
			    fatbin_path.c_str());
	}
	SPDLOG_INFO("Extracting PTX in the fatbin...");
	boost::asio::io_context ctx;
	boost::process::ipstream stream;
	boost::process::environment env = boost::this_process::environment();
	env["LD_PRELOAD"] = "";
	
	// Build command line - use shell to properly search PATH
	auto cuobjdump_cmd_line = std::string("cuobjdump --extract-ptx all ") +
				  fatbin_path.string();
	SPDLOG_INFO("Calling cuobjdump: {}", cuobjdump_cmd_line);
	
	// Execute through shell to properly use PATH
	boost::process::child child("/bin/sh",
				    boost::process::args({"-c", cuobjdump_cmd_line}),
				    boost::process::std_out > stream,
				    boost::process::env(env),
				    boost::process::start_dir = tmp_dir);

	std::string line;
	while (std::getline(stream, line)) {
		SPDLOG_DEBUG("cuobjdump output: {}", line);
	}
	for (const auto &entry :
	     std::filesystem::directory_iterator(working_dir)) {
		if (entry.is_regular_file() &&
		    entry.path().string().ends_with(".ptx")) {
			// Read the PTX into memory
			std::ifstream ifs(entry.path());
			std::stringstream buffer;
			buffer << ifs.rdbuf();
			all_ptx[entry.path().filename()] = buffer.str();
		}
	}
	SPDLOG_INFO("Got {} PTX files", all_ptx.size());
	return all_ptx;
}
std::optional<std::map<std::string, std::string>>
nv_attach_impl::hack_fatbin(std::map<std::string, std::string> all_ptx)
{
	char tmp_dir[] = "/tmp/bpftime-fatbin-work.XXXXXX";
	mkdtemp(tmp_dir);
	auto working_dir = std::filesystem::path(tmp_dir);

	/**
	Here we can patch the PTX.
	*/
	boost::asio::thread_pool pool(std::thread::hardware_concurrency());
	std::map<std::string, std::string> ptx_out;
	std::mutex map_mutex;
	for (auto &[file_name, original_ptx] : all_ptx) {
		boost::asio::post(
			pool,
			[this, original_ptx, file_name, &map_mutex,
			 &ptx_out]() -> void {
				auto current_ptx = original_ptx;
				bool inline_helpers_required = false;
				bool trampoline_required = false;
				SPDLOG_INFO("Patching PTX: {}", file_name);

				static std::atomic<uint64_t>
					inline_instance_counter{ 0 };
				auto apply_inline_block =
					[&](std::string &ptx,
					    const ptxpass::runtime_response::
						    InlineBlock &block) -> bool {
						uint64_t instance_id =
							inline_instance_counter
								.fetch_add(
									1,
									std::memory_order_relaxed);
						std::string suffix =
							std::to_string(
								instance_id);
						auto offsets =
							grow_kernel_register_usage(
								ptx, block.kernel,
								block.registers);
						auto insert_locals_with_suffix =
							[&](const std::string
								    &local_suffix) {
								if (block.local_decls
									    .empty())
									return;
								auto renamed_locals =
									append_inline_identifier_suffix(
										block.local_decls,
										local_suffix);
								insert_kernel_local_decls(
									ptx,
									block.kernel,
									renamed_locals);
							};
						auto kernel_range =
							ptxpass::find_kernel_body(
								ptx, block.kernel);
						if (kernel_range.first ==
						    std::string::npos) {
							SPDLOG_WARN(
								"Inline kernel {} not found",
								block.kernel);
							return false;
						}
						if (block.insertion_point ==
						    "entry") {
							auto emit_entry_block =
								[&]() -> bool {
								insert_locals_with_suffix(
									suffix);
								auto renamed_text =
									append_inline_identifier_suffix(
										block.text,
										suffix);
								auto adjusted_text =
									apply_register_offsets_to_text(
										renamed_text,
										offsets);
								if (adjusted_text
									    .empty())
									return false;
								size_t brace =
									ptx.find(
										'{',
										kernel_range
											.first);
								if (brace ==
								    std::string::npos) {
									SPDLOG_WARN(
										"Entry insertion failed for kernel {}",
										block.kernel);
									return false;
								}
								size_t insert_pos =
									brace + 1;
								if (insert_pos <
									    ptx.size() &&
								    ptx[insert_pos] ==
									    '\n')
									insert_pos++;
								auto starts_with =
									[](const std::
										   string &value,
									   const char
										   *prefix) {
										return value
											       .rfind(
												       prefix,
												       0) ==
											       0;
									};
								while (insert_pos <
								       ptx.size()) {
									size_t line_end =
										ptx.find(
											'\n',
											insert_pos);
									if (line_end ==
									    std::string::
										    npos)
										break;
									auto line =
										ptx.substr(
											insert_pos,
											line_end -
												insert_pos);
									auto trimmed =
										trim_string(
											line);
									if (trimmed.empty() ||
									    starts_with(
										    trimmed,
										    ".reg") ||
									    starts_with(
										    trimmed,
										    ".local") ||
									    starts_with(
										    trimmed,
										    ".shared") ||
									    starts_with(
										    trimmed,
										    ".param")) {
										insert_pos =
											line_end +
											1;
										continue;
									}
									break;
								}
								std::string indent;
								size_t indent_probe =
									insert_pos;
								while (indent_probe <
									       ptx.size() &&
								       (ptx[indent_probe] ==
										' ' ||
									ptx[indent_probe] ==
										'\t')) {
									indent.push_back(
										ptx[indent_probe]);
									indent_probe++;
								}
								if (indent.empty())
									indent = "    ";
								auto indented_block =
									indent_block(
										adjusted_text,
										indent);
								std::string block_text =
									std::string(
										"\n") +
									indented_block;
								if (!block_text
									     .empty() &&
								    block_text.back() !=
									    '\n')
									block_text.push_back(
										'\n');
								ptx.insert(
									insert_pos,
									block_text);
								return true;
							};
							return emit_entry_block();
						}
						if (block.insertion_point ==
						    "ret") {
							auto range_len =
								kernel_range
									.second -
								kernel_range
									.first;
							std::string section =
								ptx.substr(
									kernel_range
										.first,
									range_len);
							static const std::regex
								ret_regex(
									R"((\s*)(@%p\d+\s+)?ret;)");
							std::string rebuilt;
							rebuilt.reserve(
								section.size() +
								block.text.size());
							size_t last = 0;
							bool matched = false;
							std::vector<std::string>
								ret_local_suffixes;
							size_t ret_index = 0;
							for (std::sregex_iterator it(
								     section.begin(),
								     section.end(),
								     ret_regex),
							     end;
							     it != end; ++it) {
								matched = true;
								std::string ret_suffix =
									suffix +
									"r" +
									std::to_string(
										ret_index++);
								ret_local_suffixes.
									push_back(
										ret_suffix);
								auto ret_renamed_text =
									append_inline_identifier_suffix(
										block.text,
										ret_suffix);
								auto ret_adjusted_text =
									apply_register_offsets_to_text(
										ret_renamed_text,
										offsets);
								if (ret_adjusted_text
									     .empty())
									continue;
								rebuilt.append(
									section,
									last,
									it->position() -
										last);
								std::string indent =
									(*it)[1].str();
								auto indented_block =
									indent_block(
										ret_adjusted_text,
										indent);
								if (!indented_block
									     .empty()) {
									rebuilt.append(
										indent);
									rebuilt.append(
										indented_block);
									if (rebuilt.back() !=
									    '\n')
										rebuilt.push_back(
											'\n');
								}
								rebuilt.append(
									it->str());
								last = it->position() +
								       it->length();
							}
							if (!matched) {
								SPDLOG_WARN(
									"ret insertion failed for kernel {}",
									block.kernel);
								return false;
							}
							rebuilt.append(section,
								       last,
								       std::string::npos);
							ptx.replace(
								kernel_range
									.first,
								range_len,
								rebuilt);
							for (const auto &local_suffix :
							     ret_local_suffixes) {
								insert_locals_with_suffix(
									local_suffix);
							}
							return true;
						}
						SPDLOG_WARN(
							"Unsupported inline insertion point {}",
							block.insertion_point);
						return false;
					};
				for (auto &[_, hook_entry] :
				     this->hook_entries) {
					const auto &kernels =
						hook_entry.kernels;
					for (const auto &kernel : kernels) {
						std::vector<uint64_t>
							ebpf_inst_words;
						ebpf_inst_words.assign(
							(uint64_t *)(uintptr_t)
								hook_entry
									.instuctions
									.data(),
							(uint64_t *)(uintptr_t)hook_entry
									.instuctions
									.data() +
								hook_entry
									.instuctions
									.size()

						);
						ptxpass::runtime_request::
							RuntimeRequest req;
						auto &ri = req.input;
						ri.full_ptx = current_ptx;
						ri.to_patch_kernel = kernel;
						if (hook_entry.inline_enabled) {
							ri.global_ebpf_map_info_symbol =
								INLINE_MAP_INFO_SYMBOL;
							ri.ebpf_communication_data_symbol =
								INLINE_CONST_PTR_SYMBOL;
						} else {
							ri.global_ebpf_map_info_symbol =
								LEGACY_MAP_INFO_SYMBOL;
							ri.ebpf_communication_data_symbol =
								LEGACY_CONST_PTR_SYMBOL;
						}

						req.set_ebpf_instructions(
							ebpf_inst_words);
						req.inline_mode =
							hook_entry.inline_enabled;
						req.inline_metadata =
							hook_entry.inline_metadata;
						nlohmann::json in;
						ptxpass::runtime_request::to_json(
							in, req);
						SPDLOG_DEBUG("Input: {}",
							     in.dump());
						std::vector<char> buf(200
								      << 20);
						int err =
							hook_entry.config->process_input(
								in.dump().c_str(),
								buf.size(),
								buf.data());
						if (err ==
						    ptxpass::ExitCode::Success) {
							auto json = nlohmann::json::
								parse(buf.data());
							using namespace ptxpass::
								runtime_response;
							RuntimeResponse resp;
							from_json(json, resp);
							current_ptx =
								resp.output_ptx;
							if (hook_entry.inline_enabled) {
								if (resp.inline_supported) {
									inline_helpers_required =
										true;
									for (const auto &block :
									     resp.inline_blocks) {
										apply_inline_block(
											current_ptx,
											block);
									}
								} else if (hook_entry.inline_fallback_enabled) {
									SPDLOG_INFO(
										"Inline patch unsupported for kernel {}, falling back to trampoline",
										kernel);
									trampoline_required = true;
								} else {
									SPDLOG_WARN(
										"Inline requested for kernel {} but pass output is not inline-compatible and fallback disabled",
										kernel);
								}
							} else {
								trampoline_required = true;
							}
						} else {
							SPDLOG_ERROR(
								"Unable to run pass on kernel {}: {}",
								kernel,
								(int)err);
							if (hook_entry.inline_enabled &&
							    hook_entry.inline_fallback_enabled) {
								trampoline_required = true;
							}
						}
					}
				}
				if (inline_helpers_required && trampoline_required) {
					SPDLOG_WARN(
						"Both inline helpers and trampoline requested for PTX {}. Prioritizing inline helpers.",
						file_name);
				}
				if (inline_helpers_required) {
					current_ptx =
						ptxpass::filter_out_version_headers_ptx(
							wrap_ptx_with_inline_helpers(
								current_ptx));
				} else if (trampoline_required) {
					current_ptx =
						ptxpass::filter_out_version_headers_ptx(
							wrap_ptx_with_trampoline(
								current_ptx));
				} else {
					current_ptx =
						ptxpass::filter_out_version_headers_ptx(
							current_ptx);
				}
				std::lock_guard<std::mutex> _guard(map_mutex);
				ptx_out["patched." + file_name] = current_ptx;
			});
	}
	pool.join();
	SPDLOG_INFO("Writting patched PTX to {}", working_dir.c_str());
	for (const auto &[file_name, ptx] : ptx_out) {
		auto path = working_dir / (file_name);
		std::ofstream ofs(path);
		ofs << ptx;
	}
	return ptx_out;
}

namespace bpftime::attach
{

int nv_attach_impl::find_attach_entry_by_program_name(const char *name) const
{
	for (const auto &entry : this->hook_entries) {
		if (entry.second.program_name == name)
			return entry.first;
	}
	return -1;
}
#define NVPTXCOMPILER_SAFE_CALL(x)                                             \
	do {                                                                   \
		nvPTXCompileResult result = x;                                 \
		if (result != NVPTXCOMPILE_SUCCESS) {                          \
			SPDLOG_ERROR("{} failed with error code {}", #x,       \
				     (int)result);                             \
			return -1;                                             \
		}                                                              \
	} while (0)
#define CUDA_SAFE_CALL(x)                                                      \
	do {                                                                   \
		CUresult result = x;                                           \
		if (result != CUDA_SUCCESS) {                                  \
			const char *msg;                                       \
			cuGetErrorName(result, &msg);                          \
			SPDLOG_ERROR("{} failed with error {}", #x, msg);      \
			return -1;                                             \
		}                                                              \
	} while (0)

std::vector<char>
nv_attach_impl::compile_ptx_to_cubin(const std::string &ptx) const
{
	auto destroy_compiler = [](nvPTXCompilerHandle &compiler) {
		if (compiler != nullptr) {
			nvPTXCompilerDestroy(&compiler);
			compiler = nullptr;
		}
	};
	auto throw_on_error = [&](nvPTXCompileResult result,
				  const char *label, nvPTXCompilerHandle &compiler) {
		if (result != NVPTXCOMPILE_SUCCESS) {
			SPDLOG_ERROR("{} failed with error code {}", label,
				     (int)result);
			destroy_compiler(compiler);
			throw std::runtime_error("nvPTX compiler error");
		}
	};
	unsigned int major_ver = 0, minor_ver = 0;
	auto version_status =
		nvPTXCompilerGetVersion(&major_ver, &minor_ver);
	if (version_status == NVPTXCOMPILE_SUCCESS) {
		SPDLOG_INFO("PTX compiler version {}.{}", major_ver, minor_ver);
	} else {
		SPDLOG_WARN(
			"nvPTXCompilerGetVersion failed with error code {}",
			(int)version_status);
	}
	nvPTXCompilerHandle compiler = nullptr;
	throw_on_error(
		nvPTXCompilerCreate(&compiler, (size_t)ptx.size(), ptx.c_str()),
		"nvPTXCompilerCreate", compiler);
	std::string gpu_name_opt = "--gpu-name=" + target_sm_arch;
	const char *compile_options[] = { gpu_name_opt.c_str(),
					  "--fmad=false", };
					//   "--relocatable-device-code=false" };
	auto status = nvPTXCompilerCompile(
		compiler, std::size(compile_options), compile_options);
	if (status != NVPTXCOMPILE_SUCCESS) {
		size_t error_size = 0;
		nvPTXCompilerGetErrorLogSize(compiler, &error_size);
		if (error_size != 0) {
			std::string error_log(error_size + 1, '\0');
			nvPTXCompilerGetErrorLog(compiler, error_log.data());
			SPDLOG_ERROR("nvPTX compile error: {}", error_log);
		}
		destroy_compiler(compiler);
		throw std::runtime_error("nvPTXCompilerCompile failed");
	}
	size_t compiled_size = 0;
	throw_on_error(
		nvPTXCompilerGetCompiledProgramSize(compiler, &compiled_size),
		"nvPTXCompilerGetCompiledProgramSize", compiler);
	std::vector<char> output_elf(compiled_size);
	throw_on_error(
		nvPTXCompilerGetCompiledProgram(compiler, output_elf.data()),
		"nvPTXCompilerGetCompiledProgram", compiler);
	size_t info_size = 0;
	if (nvPTXCompilerGetInfoLogSize(compiler, &info_size) ==
	    NVPTXCOMPILE_SUCCESS &&
	    info_size != 0) {
		std::string info_log(info_size + 1, '\0');
		throw_on_error(
			nvPTXCompilerGetInfoLog(compiler, info_log.data()),
			"nvPTXCompilerGetInfoLog", compiler);
		SPDLOG_INFO("nvPTX compile info: {}", info_log);
	}
	destroy_compiler(compiler);
	return output_elf;
}

int nv_attach_impl::run_attach_entry_on_gpu(int attach_id, int run_count,
					    int grid_dim_x, int grid_dim_y,
					    int grid_dim_z, int block_dim_x,
					    int block_dim_y, int block_dim_z)
{
	if (run_count < 1) {
		SPDLOG_ERROR("run_count must be greater than 0");
		return -1;
	}
	std::vector<ebpf_inst> insts;
	if (auto itr = hook_entries.find(attach_id);
	    itr != hook_entries.end()) {
		// In new flow, directly_run is not supported and should be
		// represented by a dedicated pass
		insts = itr->second.instuctions;
	} else {
		SPDLOG_ERROR("Invalid attach id {}", attach_id);
		return -1;
	}
	SPDLOG_INFO("Running program on GPU");

	SPDLOG_INFO("Using SM architecture: {}", target_sm_arch);

	std::vector<uint64_t> ebpf_words;
	for (const auto &insts : insts) {
		ebpf_words.push_back(*(uint64_t *)(uintptr_t)&insts);
	}
	auto ptx = ptxpass::filter_out_version_headers_ptx(
		wrap_ptx_with_trampoline(filter_compiled_ptx_for_ebpf_program(
			ptxpass::compile_ebpf_to_ptx_from_words(
				ebpf_words, target_sm_arch.c_str(), "bpf_main", false,
				false),
			"bpf_main")));
	{
		const std::string to_replace = ".func bpf_main";

		// Replace ".func bpf_main" to ".visible .entry bpf_main" so it
		// can be executed
		auto bpf_main_pos = ptx.find(to_replace);
		if (bpf_main_pos == ptx.npos) {
			SPDLOG_ERROR("Cannot find '{}' in generated PTX code",
				     to_replace);
			return -1;
		}
		ptx = ptx.replace(bpf_main_pos, to_replace.size(),
				  ".visible .entry bpf_main");
	}
	if (spdlog::get_level() <= SPDLOG_LEVEL_DEBUG) {
		auto path = "/tmp/directly-run.ptx";

		std::ofstream ofs(path);
		ofs << ptx << std::endl;
		SPDLOG_DEBUG("Dumped directly run ptx to {}", path);
	}
	// Compile to ELF
	std::vector<char> output_elf;
	try {
		output_elf = compile_ptx_to_cubin(ptx);
	} catch (const std::exception &e) {
		SPDLOG_ERROR("Unable to compile PTX: {}", e.what());
		return -1;
	}
	SPDLOG_INFO("Compiled program size: {}", output_elf.size());
	// Load and run the program
	{
		CUdevice cuDevice;
		CUcontext context;
		CUmodule module;
		CUfunction kernel;
		CUDA_SAFE_CALL(cuInit(0));
		CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));

		CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
		CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, output_elf.data(), 0,
						  0, 0));
		// fill data into it
		{
			CUdeviceptr ptr;
			size_t bytes;
			CUDA_SAFE_CALL(cuModuleGetGlobal(&ptr, &bytes, module,
							 "constData"));
			CUDA_SAFE_CALL(
				cuMemcpyHtoD(ptr, &this->shared_mem_ptr,
					     sizeof(this->shared_mem_ptr)));
			SPDLOG_INFO(
				"shared_mem_ptr copied: device ptr {:x}, device size {}",
				(uintptr_t)ptr, bytes);
		}
		{
			CUdeviceptr ptr;
			size_t bytes;
			CUDA_SAFE_CALL(cuModuleGetGlobal(&ptr, &bytes, module,
							 "map_info"));
			if (!this->map_basic_info.has_value()) {
				SPDLOG_ERROR(
					"map_basic_info is not set, cannot copy to device");
				return -1;
			}
			CUDA_SAFE_CALL(cuMemcpyHtoD(
				ptr, this->map_basic_info->data(),
				sizeof(this->map_basic_info->at(0)) *
					this->map_basic_info->size()));
			SPDLOG_INFO(
				"map_info copied: device ptr {:x}, device size {}",
				(uintptr_t)ptr, bytes);
		}
		CUDA_SAFE_CALL(
			cuModuleGetFunction(&kernel, module, "bpf_main"));
		for (int i = 1; i <= run_count; i++) {
			SPDLOG_INFO("Run {}", i);
			CUDA_SAFE_CALL(cuLaunchKernel(
				kernel, grid_dim_x, grid_dim_y, grid_dim_z,
				block_dim_x, block_dim_y, block_dim_z, 0,
				nullptr, nullptr, 0));
			CUDA_SAFE_CALL(cuCtxSynchronize());
		}
	}
	return 0;
}

void nv_attach_impl::mirror_cuda_memcpy_to_symbol(
	const void *symbol, const void *src, size_t count, size_t offset,
	cudaMemcpyKind kind, cudaStream_t stream, bool async)
{
	auto record_itr = symbol_address_to_fatbin.find((void *)symbol);
	if (record_itr == symbol_address_to_fatbin.end())
		return;
	auto &record = *record_itr->second;
	auto var_itr = record.variable_addr_to_symbol.find((void *)symbol);
	if (var_itr == record.variable_addr_to_symbol.end()) {
		SPDLOG_DEBUG(
			"mirror_cuda_memcpy_to_symbol: no variable info for symbol pointer {:x}",
			(uintptr_t)symbol);
		return;
	}
	auto &var_info = var_itr->second;
	if (offset >= var_info.size) {
		SPDLOG_WARN(
			"mirror_cuda_memcpy_to_symbol: offset {} exceeds size {} for symbol {}",
			offset, var_info.size, var_info.symbol_name);
		return;
	}
	size_t writable = var_info.size - offset;
	size_t bytes_to_copy = std::min(count, writable);
	if (bytes_to_copy == 0)
		return;
	if (bytes_to_copy != count) {
		SPDLOG_WARN(
			"mirror_cuda_memcpy_to_symbol: truncating copy for symbol {} (requested={}, allowed={})",
			var_info.symbol_name, count, bytes_to_copy);
	}
	CUdeviceptr dst = var_info.ptr + offset;
	CUstream cu_stream = reinterpret_cast<CUstream>(stream);
	CUresult status = CUDA_SUCCESS;

	auto copy_device_ptr = [](const void *ptr) -> CUdeviceptr {
		return static_cast<CUdeviceptr>(
			reinterpret_cast<uintptr_t>(ptr));
	};

	switch (kind) {
	case cudaMemcpyHostToDevice:
	case cudaMemcpyDefault:
		status = async ? cuMemcpyHtoDAsync(dst, src, bytes_to_copy,
						   cu_stream) :
				 cuMemcpyHtoD(dst, src, bytes_to_copy);
		break;
	case cudaMemcpyDeviceToDevice:
		status = async ? cuMemcpyDtoDAsync(dst, copy_device_ptr(src),
						   bytes_to_copy, cu_stream) :
				 cuMemcpyDtoD(dst, copy_device_ptr(src),
					      bytes_to_copy);
		break;
	default:
		SPDLOG_DEBUG(
			"mirror_cuda_memcpy_to_symbol: unsupported memcpy kind {} for symbol {}",
			(int)kind, var_info.symbol_name);
		return;
	}
	if (status != CUDA_SUCCESS) {
		SPDLOG_WARN(
			"mirror_cuda_memcpy_to_symbol: failed to copy symbol {} (err={})",
			var_info.symbol_name, (int)status);
	}
}

} // namespace bpftime::attach
