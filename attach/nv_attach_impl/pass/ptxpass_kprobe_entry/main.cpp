#include "json.hpp"
#include "ptxpass/core.hpp"
#include <cstdio>
#include <cstring>
#include <vector>
#include <exception>
#include <iostream>
#include <string>
#include <optional>
namespace entry_params
{
struct EntryParams {
	std::string save_strategy = "minimal"; // "minimal" or "full"
	bool emit_nops_for_alignment = false;
	int pad_nops = 0;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(EntryParams, save_strategy,
						emit_nops_for_alignment,
						pad_nops);

} // namespace entry_params

static ptxpass::pass_config::PassConfig get_default_config()
{
	ptxpass::pass_config::PassConfig cfg;
	cfg.name = "kprobe_entry";
	cfg.description =
		"Instrument PTX at kprobe entry points, excluding __memcapture";
	cfg.attach_points.includes = { "^kprobe/.*$" };
	cfg.attach_points.excludes = { "^kprobe/__memcapture$" };
	cfg.parameters = nlohmann::json{ { "insert_globaltimer", true } };
	cfg.attach_type = 8; // kprobe
	return cfg;
}

static ptxpass::runtime_response::InlineBlock
build_inline_entry_block(const std::string &kernel,
			 const std::string &func_ptx)
{
	ptxpass::runtime_response::InlineBlock block;
	block.kernel = kernel;
	block.insertion_point = "entry";
	block.text = func_ptx;
	return block;
}

static std::optional<std::string>
extract_function_body(const std::string &func_ptx)
{
	auto brace_begin = func_ptx.find('{');
	auto brace_end = func_ptx.rfind('}');
	if (brace_begin == std::string::npos || brace_end == std::string::npos ||
	    brace_end <= brace_begin)
		return std::nullopt;
	auto body =
		func_ptx.substr(brace_begin + 1, brace_end - brace_begin - 1);
	return body;
}

static bool try_generate_inline_response(
	const ptxpass::runtime_request::RuntimeRequest &runtime_request,
	const std::vector<uint64_t> &ebpf_words,
	ptxpass::runtime_response::RuntimeResponse &response)
{
	if (!runtime_request.inline_mode || ebpf_words.empty())
		return false;
	const auto &kernel = runtime_request.input.to_patch_kernel;
	std::string fname = std::string("__probe_func__") + kernel;
	auto func_ptx = ptxpass::compile_ebpf_to_ptx_from_words(
		ebpf_words, "sm_60", fname, false, false);
	auto body = extract_function_body(func_ptx);
	if (!body.has_value())
		return false;
	auto sanitized = ptxpass::sanitize_inline_ptx_body(
		kernel, "entry", body.value());
	response.inline_supported = true;
	auto block =
		build_inline_entry_block(kernel, sanitized.text);
	block.registers = sanitized.registers;
	block.local_decls = sanitized.local_decls;
	response.inline_blocks.push_back(block);
	response.required_helpers.clear();
	return true;
}

static std::pair<std::string, bool>
inject_call_trampoline(const std::string &ptx, const std::string &kernel,
		       const std::vector<uint64_t> &ebpf_words)
{
	if (ebpf_words.empty()) {
		return { ptx, false };
	}
	std::string fname = std::string("__probe_func__") + kernel;
	auto generated_func_ptx = ptxpass::compile_ebpf_to_ptx_from_words(
		ebpf_words, "sm_60", fname, true, false);
	auto body = ptxpass::find_kernel_body(ptx, kernel);
	if (body.first == std::string::npos)
		return { ptx, false };
	std::string out = ptx;
	size_t brace = out.find('{', body.first);
	if (brace == std::string::npos)
		return { ptx, false };
	size_t insert_pos = brace + 1;
	if (insert_pos < out.size() && out[insert_pos] == '\n')
		insert_pos++;

	out.insert(insert_pos, std::string("\n    call ") + fname + ";\n");
	out = generated_func_ptx + "\n" + out;
	ptxpass::log_transform_stats("kprobe_entry", 1, ptx.size(), out.size());
	return { out, true };
}

extern "C" void print_config(int length, char *out)
{
	auto cfg = get_default_config();
	nlohmann::json output_json;
	ptxpass::pass_config::to_json(output_json, cfg);
	snprintf(out, length, "%s", output_json.dump().c_str());
}

extern "C" int process_input(const char *input, int length, char *output)
{
	using namespace ptxpass;
	auto cfg = get_default_config();
	try {
		auto runtime_request = pass_runtime_request_from_string(input);
		if (!validate_input(runtime_request.input.full_ptx,
				    cfg.validation)) {
			return ExitCode::TransformFailed;
		}
		auto ebpf_words =
			runtime_request.get_uint64_ebpf_instructions();
		ptxpass::runtime_response::RuntimeResponse response;
		response.output_ptx = runtime_request.input.full_ptx;
		response.inline_supported = false;
		response.inline_blocks.clear();
		response.required_helpers.clear();

		if (try_generate_inline_response(
			    runtime_request, ebpf_words, response)) {
			snprintf(output, length, "%s",
				 emit_runtime_response_and_return(response)
					 .c_str());
			return ExitCode::Success;
		}

		auto [patched_ptx, modified] = inject_call_trampoline(
			runtime_request.input.full_ptx,
			runtime_request.input.to_patch_kernel, ebpf_words);
		response.output_ptx = patched_ptx;
		response.inline_supported = false;
		response.inline_blocks.clear();
		response.required_helpers.clear();
		snprintf(output, length, "%s",
			 emit_runtime_response_and_return(response).c_str());
		return modified ? ExitCode::Success :
				  ExitCode::TransformFailed;
	} catch (const std::runtime_error &e) {
		std::cerr << e.what() << "\n";
		return ExitCode::ConfigError;
	} catch (const std::exception &e) {
		std::cerr << e.what() << "\n";
		return ExitCode::UnknownError;
	} catch (...) {
		std::cerr << "Unknown error\n";
		return ExitCode::UnknownError;
	}
}
