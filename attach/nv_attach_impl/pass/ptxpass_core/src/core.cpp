#include "ptxpass/core.hpp"
#include "spdlog/spdlog.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <set>
#include <optional>
#include <cctype>
#include <unordered_map>
#include <atomic>
#include <llvmbpf.hpp>
#include <llvm_jit_context.hpp>

#include "json.hpp"

using nlohmann::json;

namespace ptxpass
{

static std::vector<std::regex>
compile_regex_list(const std::vector<std::string> &patterns)
{
	std::vector<std::regex> out;
	out.reserve(patterns.size());
	for (const auto &p : patterns) {
		out.emplace_back(p, std::regex::ECMAScript);
	}
	return out;
}

AttachPointMatcher::AttachPointMatcher(const attach_points::AttachPoints &points)
	: includeRegexes(compile_regex_list(points.includes)),
	  excludeRegexes(compile_regex_list(points.excludes))
{
}

bool AttachPointMatcher::matches(const std::string &attachPoint) const
{
	bool included = false;
	for (const auto &r : includeRegexes) {
		if (std::regex_search(attachPoint, r)) {
			included = true;
			break;
		}
	}
	if (!included)
		return false;
	for (const auto &r : excludeRegexes) {
		if (std::regex_search(attachPoint, r)) {
			return false;
		}
	}
	return true;
}

std::string read_all_from_stdin()
{
	std::ostringstream oss;
	oss << std::cin.rdbuf();
	if (!std::cin.good() && !std::cin.eof()) {
		throw std::runtime_error("Failed to read from stdin");
	}
	return oss.str();
}

bool is_whitespace_only(const std::string &s)
{
	for (char c : s) {
		if (!(c == ' ' || c == '\n' || c == '\r' || c == '\t' ||
		      c == '\v' || c == '\f')) {
			return false;
		}
	}
	return true;
}

std::string get_env(const char *key)
{
	const char *v = std::getenv(key);
	return v ? std::string(v) : std::string();
}

// Validation: basic checks as placeholders
bool validate_input(const std::string &input, const json &validation)
{
	if (validation.is_null())
		return true;
	if (validation.contains("require_entry") &&
	    validation["require_entry"].get<bool>()) {
		if (!contains_entry_function(input))
			return false;
	}
	if (validation.contains("require_ret") &&
	    validation["require_ret"].get<bool>()) {
		if (!contains_ret_instruction(input))
			return false;
	}
	if (validation.contains("ptx_version_min")) {
		if (!validate_ptx_version(
			    input,
			    validation["ptx_version_min"].get<std::string>()))
			return false;
	}
	return true;
}

bool contains_entry_function(const std::string &input)
{
	return input.find(".visible .entry") != std::string::npos;
}

bool contains_ret_instruction(const std::string &input)
{
	return input.find("\n    ret;") != std::string::npos ||
	       input.find("\n\tret;") != std::string::npos;
}

bool validate_ptx_version(const std::string &input,
			  const std::string &minVersion)
{
	// Expect line like: .version 7.0
	std::istringstream iss(input);
	std::string line;
	double want = 0.0;
	try {
		want = std::stod(minVersion);
	} catch (...) {
		return true;
	}
	while (std::getline(iss, line)) {
		if (line.rfind(".version", 0) == 0) {
			// parse number
			std::string v = line.substr(8);
			try {
				double have = std::stod(v);
				return have >= want;
			} catch (...) {
				return true;
			}
		}
	}
	return true;
}

} // namespace ptxpass

namespace ptxpass
{
std::string filter_out_version_headers_ptx(const std::string &input)
{
	static const std::string FILTERED_OUT_PREFIXES[] = {
		".version", ".target", ".address_size", "//"
	};
	std::istringstream iss(input);
	std::ostringstream oss;
	std::string line;
	std::set<std::string> seen;
	while (std::getline(iss, line)) {
		bool skip = false;
		for (const auto &p : FILTERED_OUT_PREFIXES) {
			if (line.rfind(p, 0) == 0) {
				if (seen.contains(p))
					skip = true;
				else
					seen.insert(p);
				break;
			}
		}
		if (!skip)
			oss << line << '\n';
	}
	return oss.str();
}
static uint64_t test_func(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t)
{
	return 0;
}
std::string compile_ebpf_to_ptx_from_words(
	const std::vector<uint64_t> &words, const std::string &target_sm,
	const std::string &func_name,
	bool add_register_guard_and_filter_version_headers, bool with_arguments)
{
	const ebpf_inst *insts =
		reinterpret_cast<const ebpf_inst *>(words.data());
	size_t insts_count = words.size();
	bpftime::llvmbpf_vm vm;
	vm.register_external_function(1, "map_lookup", (void *)test_func);
	vm.register_external_function(2, "map_update", (void *)test_func);
	vm.register_external_function(3, "map_delete", (void *)test_func);
	vm.register_external_function(6, "print", (void *)test_func);
	vm.register_external_function(14, "get_pid_tgid", (void *)test_func);
	vm.register_external_function(25, "perf_event_output",
				      (void *)test_func);

	vm.register_external_function(501, "puts", (void *)test_func);
	vm.register_external_function(502, "get_global_timer",
				      (void *)test_func);
	vm.register_external_function(503, "get_block_idx", (void *)test_func);
	vm.register_external_function(504, "get_block_dim", (void *)test_func);
	vm.register_external_function(505, "get_thread_idx", (void *)test_func);
	vm.register_external_function(507, "cuda_exit", (void *)test_func);
	vm.register_external_function(508, "get_grid_dim", (void *)test_func);

	vm.load_code(insts, insts_count * sizeof(ebpf_inst));
	bpftime::llvm_bpf_jit_context ctx(vm);
	std::string original_ptx;
	if (auto optional_ptx = ctx.generate_ptx(with_arguments, func_name,
						 target_sm.c_str());
	    optional_ptx) {
		original_ptx = *optional_ptx;
	} else {
		SPDLOG_ERROR("Unable to produce PTX from eBPF");
		throw std::runtime_error("Unable to produce PTX from eBPF");
	}
	std::string filtered_ptx;
	if (add_register_guard_and_filter_version_headers) {
		filtered_ptx =
			bpftime::attach::add_register_guard_for_ebpf_ptx_func(
				filter_compiled_ptx_for_ebpf_program(
					original_ptx));
	} else {
		filtered_ptx = original_ptx;
	}
	return filtered_ptx;
}

namespace
{
std::string trim(const std::string &input)
{
	size_t start = input.find_first_not_of(" \t\r\n");
	if (start == std::string::npos)
		return "";
	size_t end = input.find_last_not_of(" \t\r\n");
	return input.substr(start, end - start + 1);
}

struct RegisterDecl {
	std::string prefix;
	int count = 0;
};

enum class RegisterCategory { Reg32, Reg64, Reg16, Pred };

std::optional<RegisterDecl> parse_register_decl(const std::string &line)
{
	static const std::regex reg_regex(
		R"(^\s*\.reg\s+\.[a-zA-Z0-9_]+\s+(%[a-zA-Z]+)<(\d+)>\s*;)");
	std::smatch match;
	if (std::regex_search(line, match, reg_regex)) {
		RegisterDecl decl;
		decl.prefix = match[1];
		decl.count = std::stoi(match[2]);
		return decl;
	}
	return std::nullopt;
}

struct LocalDecl {
	std::string original_name;
	std::string renamed_line;
	std::string renamed_name;
};

std::string escape_regex(const std::string &input);

std::optional<LocalDecl> parse_local_decl(const std::string &line,
					  uint64_t block_hash, size_t ordinal)
{
	static const std::regex local_regex(
		R"(^\s*\.local\s+.*?([A-Za-z_][A-Za-z0-9_$.]*)\s*(\[.*\])\s*;)");
	std::smatch match;
	if (std::regex_search(line, match, local_regex)) {
		const std::string name = match[1];
		const auto &suffix = match[2];
		std::ostringstream oss;
		oss << "__bpftime_inline_local_" << std::hex << block_hash
		    << "_" << std::dec << ordinal;
		std::string new_name = oss.str();
		LocalDecl decl;
		decl.original_name = name;
		decl.renamed_name = new_name;
		std::regex name_regex(escape_regex(name),
				      std::regex_constants::ECMAScript);
		decl.renamed_line =
			std::regex_replace(line, name_regex, new_name,
					   std::regex_constants::
						   format_first_only);
		return decl;
	}
	return std::nullopt;
}

RegisterCategory register_category_from_type(const std::string &type_token)
{
	if (type_token == ".pred")
		return RegisterCategory::Pred;
	if ((type_token.find("16") != std::string::npos ||
	     type_token.find("b16") != std::string::npos) &&
	    type_token.find("64") == std::string::npos &&
	    type_token.find("32") == std::string::npos)
		return RegisterCategory::Reg16;
	if (type_token.find("64") != std::string::npos ||
	    type_token.find("b64") != std::string::npos ||
	    type_token.find("s64") != std::string::npos ||
	    type_token.find("u64") != std::string::npos ||
	    type_token.find("f64") != std::string::npos)
		return RegisterCategory::Reg64;
	return RegisterCategory::Reg32;
}

struct NamedRegisterDecl {
	RegisterCategory category;
	std::string name;
};

std::optional<NamedRegisterDecl>
parse_named_register_decl(const std::string &line)
{
	if (line.find('<') != std::string::npos)
		return std::nullopt;
	static const std::regex named_regex(
		R"(^\s*\.reg\s+(\.[A-Za-z0-9_]+)\s+(%[A-Za-z_.$][A-Za-z0-9_.$]*)\s*;)");
	std::smatch match;
	if (std::regex_search(line, match, named_regex)) {
		NamedRegisterDecl decl;
		decl.category = register_category_from_type(match[1]);
		decl.name = match[2];
		return decl;
	}
	return std::nullopt;
}

std::optional<std::string> parse_label_name(const std::string &line)
{
	static const std::regex label_regex(
		R"(^\s*([$A-Za-z_][A-Za-z0-9_.$]*)\s*:\s*$)");
	std::smatch match;
	if (std::regex_match(line, match, label_regex))
		return match[1];
	return std::nullopt;
}

std::string escape_regex(const std::string &input)
{
	std::string escaped;
	escaped.reserve(input.size() * 2);
	for (char c : input) {
		unsigned char uc = static_cast<unsigned char>(c);
		auto is_alnum = [](unsigned char ch) -> bool {
			return (ch >= '0' && ch <= '9') ||
			       (ch >= 'A' && ch <= 'Z') ||
			       (ch >= 'a' && ch <= 'z');
		};
		if (is_alnum(uc) || c == '_' || c == '$' || c == '.')
			escaped.push_back(c);
		else {
			escaped.push_back('\\');
			escaped.push_back(c);
		}
	}
	return escaped;
}

bool is_identifier_char(char c)
{
	return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') ||
	       (c >= 'a' && c <= 'z') || c == '_' || c == '.' || c == '$';
}

void replace_identifier(std::string &text, const std::string &old_name,
			const std::string &new_name)
{
	if (old_name.empty())
		return;
	size_t pos = 0;
	while (true) {
		pos = text.find(old_name, pos);
		if (pos == std::string::npos)
			break;
		bool left_ok =
			(pos == 0) || !is_identifier_char(text[pos - 1]);
		bool right_ok = true;
		size_t right_index = pos + old_name.size();
		if (right_index < text.size())
			right_ok = !is_identifier_char(text[right_index]);
		if (left_ok && right_ok) {
			text.replace(pos, old_name.size(), new_name);
			pos += new_name.size();
		} else {
			pos += 1;
		}
	}
}

} // namespace

SanitizedInlinePTX
sanitize_inline_ptx_body(const std::string &kernel,
			 const std::string &insertion_point,
			 const std::string &body)
{
	SanitizedInlinePTX result;
	std::stringstream reader(body);
	std::ostringstream sanitized;
	std::string line;
	std::vector<LocalDecl> locals;
	std::unordered_map<std::string, std::string> identifier_rewrites;
	std::unordered_map<std::string, std::string> label_rewrites;
	int reg32_needed = 0;
	int reg64_needed = 0;
	int reg16_needed = 0;
	int pred_needed = 0;
	int extra_reg32 = 0;
	int extra_reg64 = 0;
	int extra_reg16 = 0;
	int extra_pred = 0;
	size_t label_counter = 0;
	static std::atomic<uint64_t> inline_block_counter{ 0 };
	uint64_t block_seq =
		inline_block_counter.fetch_add(1, std::memory_order_relaxed);
	std::hash<std::string> hasher;
	uint64_t block_hash = static_cast<uint64_t>(hasher(
		kernel + "|" + insertion_point + "|" + body));
	block_hash ^= block_seq + 0x9e3779b97f4a7c15ULL +
		      (block_hash << 6) + (block_hash >> 2);
	block_hash ^= reinterpret_cast<uintptr_t>(body.data());
	block_hash ^= reinterpret_cast<uintptr_t>(&sanitize_inline_ptx_body);
	auto allocate_register =
		[&](RegisterCategory category) -> std::string {
		switch (category) {
		case RegisterCategory::Pred:
			return "%p" + std::to_string(pred_needed + extra_pred++);
		case RegisterCategory::Reg64:
			return "%rd" + std::to_string(reg64_needed +
						      extra_reg64++);
		case RegisterCategory::Reg16:
			return "%rs" + std::to_string(reg16_needed +
						      extra_reg16++);
		case RegisterCategory::Reg32:
		default:
			return "%r" + std::to_string(reg32_needed +
						     extra_reg32++);
		}
	};

	while (std::getline(reader, line)) {
		auto trimmed = trim(line);
		if (trimmed.empty()) {
			sanitized << line << "\n";
			continue;
		}
		if (trimmed == "ret;") {
			continue;
		}
		if (auto decl = parse_register_decl(trimmed); decl.has_value()) {
			if (decl->prefix == "%r")
				reg32_needed =
					std::max(reg32_needed, decl->count);
			else if (decl->prefix == "%rs")
				reg16_needed =
					std::max(reg16_needed, decl->count);
			else if (decl->prefix == "%rd")
				reg64_needed =
					std::max(reg64_needed, decl->count);
			else if (decl->prefix == "%p")
				pred_needed =
					std::max(pred_needed, decl->count);
			continue;
		}
		if (auto named = parse_named_register_decl(trimmed);
		    named.has_value()) {
			auto assigned = allocate_register(named->category);
			identifier_rewrites[named->name] = assigned;
			continue;
		}
		if (auto local =
			    parse_local_decl(line, block_hash, locals.size());
		    local.has_value()) {
			locals.push_back(local.value());
			result.local_decls.push_back(local->renamed_line);
			continue;
		}
		if (auto label_name = parse_label_name(trimmed);
		    label_name.has_value()) {
			std::ostringstream oss;
			oss << "$bpftime_inline_" << std::hex << block_hash
			    << "_" << std::dec << label_counter++;
			auto new_label = oss.str();
			label_rewrites[label_name.value()] = new_label;
			sanitized << new_label << ":\n";
			continue;
		}
		sanitized << line << "\n";
	}
	result.text = sanitized.str();
	for (const auto &decl : locals) {
		replace_identifier(result.text, decl.original_name,
				   decl.renamed_name);
	}
	for (const auto &[orig, repl] : identifier_rewrites) {
		replace_identifier(result.text, orig, repl);
	}
	for (const auto &[orig, repl] : label_rewrites) {
		replace_identifier(result.text, orig, repl);
	}
	result.registers.reg32 = reg32_needed + extra_reg32;
	result.registers.reg64 = reg64_needed + extra_reg64;
	result.registers.reg16 = reg16_needed + extra_reg16;
	result.registers.pred = pred_needed + extra_pred;
	return result;
}
std::string filter_compiled_ptx_for_ebpf_program(std::string input)
{
	std::istringstream iss(input);
	std::ostringstream oss;
	std::string line;
	static const std::string FILTERED_OUT_PREFIXES[] = {
		".version", ".target", ".address_size", "//"
	};
	static const std::regex FILTERED_OUT_REGEXS[] = {
		std::regex(
			R"(\.extern\s+\.func\s+\(\s*\.param\s+\.b64\s+func_retval0\s*\)\s+_bpf_helper_ext_\d{4}\s*\(\s*(?:\.param\s+\.b64\s+_bpf_helper_ext_\d{4}_param_\d+\s*,\s*)*\.param\s+\.b64\s+_bpf_helper_ext_\d{4}_param_\d+\s*\)\s*;)"),

	};
	static const std::string FILTERED_OUT_SECTION[] = {
		R"(.visible .func bpf_main(
	.param .b64 bpf_main_param_0,
	.param .b64 bpf_main_param_1
))",
		R"(.visible .func bpf_main())"
	};
	while (std::getline(iss, line)) {
		// if(line.starts_with)
		bool skip = false;
		for (const auto &prefix : FILTERED_OUT_PREFIXES) {
			if (line.starts_with(prefix)) {
				skip = true;
				break;
			}
		}
		if (!skip)
			oss << line << std::endl;
	}
	auto result = oss.str();
	for (const auto &sec : FILTERED_OUT_SECTION) {
		if (auto pos = result.find(sec); pos != result.npos) {
			result = result.replace(pos, sec.size(), "");
		}
	}
	for (const auto &regex : FILTERED_OUT_REGEXS) {
		result = std::regex_replace(result, regex, "");
	}

	return result;
}
std::pair<size_t, size_t> find_kernel_body(const std::string &ptx,
					   const std::string &kernel)
{
	static std::regex kernel_entry(
		R"((\.visible\s+)?\.entry\s+(\w+)\s*\(([^)]*)\))");
	std::smatch m;
	std::string::const_iterator search_start(ptx.cbegin());
	while (std::regex_search(search_start, ptx.cend(), m, kernel_entry)) {
		if (m[2] == kernel) {
			size_t begin = (size_t)(m[0].first - ptx.cbegin());
			size_t end = begin;
			std::vector<char> st;
			do {
				while (end < ptx.size() && ptx[end] != '{' &&
				       ptx[end] != '}')
					end++;
				if (end >= ptx.size())
					break;
				if (ptx[end] == '{')
					st.push_back('{');
				else
					st.pop_back();
				end++;
			} while (!st.empty());
			return { begin, end };
		}
		search_start = m.suffix().first;
	}
	return { std::string::npos, std::string::npos };
}

void log_transform_stats(const char *pass_name, int matched, size_t bytes_in,
			 size_t bytes_out)
{
	std::cerr << "[ptxpass] " << pass_name << ": matched=" << matched
		  << ", in=" << bytes_in << ", out=" << bytes_out << "\n";
}
} // namespace ptxpass
