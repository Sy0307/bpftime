#include "nv_attach_utils.hpp"
#include "trampoline_ptx.h"
#include <spdlog/spdlog.h>
namespace bpftime
{
namespace attach
{
namespace
{
void replace_all(std::string &target, const std::string &from,
		 const std::string &to)
{
	if (from.empty())
		return;
	size_t pos = 0;
	while ((pos = target.find(from, pos)) != std::string::npos) {
		target.replace(pos, from.size(), to);
		pos += to.size();
	}
}

void rewrite_inline_helper_symbols(std::string &helpers)
{
	replace_all(helpers, LEGACY_CONST_PTR_SYMBOL,
		    INLINE_CONST_PTR_SYMBOL);
	replace_all(helpers, LEGACY_MAP_INFO_SYMBOL,
		    INLINE_MAP_INFO_SYMBOL);
}
} // namespace

std::string get_default_trampoline_ptx()
{
	return TRAMPOLINE_PTX;
}
std::string wrap_ptx_with_trampoline(std::string input)
{
	return get_default_trampoline_ptx() + input;
}
static std::string build_inline_helper_ptx()
{
	static std::string inline_ptx = []() {
		std::string helpers = get_default_trampoline_ptx();
		const std::string entry_marker = ".visible .entry bpf_main";
		if (auto pos = helpers.find(entry_marker); pos != helpers.npos) {
			auto next_func = helpers.find(".visible .func", pos + entry_marker.size());
			if (next_func == helpers.npos)
				helpers.erase(pos);
			else
				helpers.erase(pos, next_func - pos);
		}
		rewrite_inline_helper_symbols(helpers);
		return helpers;
	}();
	return inline_ptx;
}
std::string wrap_ptx_with_inline_helpers(std::string input)
{
	return build_inline_helper_ptx() + input;
}
std::string patch_helper_names_and_header(std::string result)
{
	const std::string to_replace_names[][2] = {
		{ "_bpf_helper_ext_0001", "_bpf_helper_ext_0001_dup" },
		{ "_bpf_helper_ext_0002", "_bpf_helper_ext_0002_dup" },
		{ "_bpf_helper_ext_0003", "_bpf_helper_ext_0003_dup" },
		{ "_bpf_helper_ext_0006", "_bpf_helper_ext_0006_dup" },

	};
	const std::string version_headers[] = {
		".version 3.2\n.target sm_60\n.address_size 64\n",
		".version 5.0\n.target sm_60\n.address_size 64\n"
	};
	for (const auto &entry : to_replace_names) {
		auto idx = result.find(entry[0]);
		if (idx != result.npos) {
			result = result.replace(idx, entry[0].size(), entry[1]);
		}
	}
	for (const auto &header : version_headers) {
		auto idx = result.find(header);
		SPDLOG_INFO("Version header ({}) index: {}", header, idx);
		if (idx != result.npos) {
			result = result.replace(idx, header.size(), "");
		}
	}
	return result;
}
std::string patch_main_from_func_to_entry(std::string result)
{
	const std::string entry_func = ".visible .func bpf_main";

	auto idx = result.find(entry_func);
	SPDLOG_INFO("entry_func ({}) index {}", entry_func, idx);

	if (idx != result.npos) {
		result = result.replace(idx, entry_func.size(),
					".visible .entry bpf_main");
	}
	return result;
}

} // namespace attach
} // namespace bpftime
