#include "nv_attach_fatbin_record.hpp"
#include "cuda.h"
#include "nvPTXCompiler.h"
#include "nv_attach_utils.hpp"
#include "spdlog/spdlog.h"
#include "nv_attach_impl.hpp"
#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <ptx_pass_config.h>
#include "ptx_compiler/ptx_compiler.hpp"
#include "sass_map/sass_map.hpp"
#include "json.hpp"
#include <utility>
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
#define NVPTXCOMPILER_CHECK_EXCEPTION(x, message)                              \
	do {                                                                   \
		nvPTXCompileResult result = x;                                 \
		if (result != NVPTXCOMPILE_SUCCESS) {                          \
			SPDLOG_ERROR("error: {} failed with error code {}\n",  \
				     #x, (int)result);                         \
			throw std::runtime_error(message);                     \
		}                                                              \
	} while (0)
namespace bpftime::attach
{
namespace {
static bool env_truthy(const char *key)
{
	const char *v = std::getenv(key);
	if (!v)
		return false;
	std::string s(v);
	std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
		return (char)std::tolower(c);
	});
	return s == "1" || s == "true" || s == "yes" || s == "y" || s == "on";
}

static std::optional<std::filesystem::path> env_dir(const char *key)
{
	const char *v = std::getenv(key);
	if (!v || !*v)
		return std::nullopt;
	return std::filesystem::path(v);
}

static void maybe_dump_file(const std::optional<std::filesystem::path> &dir,
			    const std::string &filename,
			    std::span<const uint8_t> data)
{
	if (!dir)
		return;
	std::filesystem::create_directories(*dir);
	auto path = *dir / filename;
	std::ofstream ofs(path, std::ios::binary);
	ofs.write((const char *)data.data(), (std::streamsize)data.size());
}

static std::optional<std::vector<std::string>> build_nvptx_extra_options()
{
	std::vector<std::string> extra;
	if (env_truthy("BPFTIME_NVPTX_LINEINFO")) {
		// NVPTXCompiler option spelling varies by CUDA versions; we probe.
		extra.emplace_back("--generate-line-info");
		extra.emplace_back("-lineinfo");
	}
	if (env_truthy("BPFTIME_NVPTX_DEBUG")) {
		extra.emplace_back("--device-debug");
		extra.emplace_back("-g");
	}
	if (extra.empty())
		return std::nullopt;
	return extra;
}

static std::string rewrite_ptx_target(std::string ptx, const std::string &sm_arch)
{
	if (sm_arch.empty())
		return ptx;

	// Newer architectures (e.g. sm_120) require a sufficiently new PTX ISA
	// version in the `.version` directive. Some fatbins ship older PTX (e.g.
	// `.version 8.1`) targeting older SMs; rewriting only `.target` would make
	// ptxas reject it. Bump the PTX ISA version conservatively when needed.
	if (sm_arch.rfind("sm_12", 0) == 0) {
		auto vpos = ptx.find(".version");
		if (vpos != std::string::npos) {
			auto p = vpos + strlen(".version");
			while (p < ptx.size() && (ptx[p] == ' ' || ptx[p] == '\t'))
				p++;
			auto start = p;
			while (p < ptx.size() && ((ptx[p] >= '0' && ptx[p] <= '9') || ptx[p] == '.'))
				p++;
			if (p > start) {
				double ver = 0.0;
				try {
					ver = std::stod(ptx.substr(start, p - start));
				} catch (...) {
					ver = 0.0;
				}
				if (ver > 0.0 && ver < 8.7) {
					ptx.replace(start, p - start, "8.7");
				}
			}
		}
	}

	auto pos = ptx.find(".target");
	if (pos == std::string::npos)
		return ptx;
	pos += strlen(".target");
	while (pos < ptx.size() && (ptx[pos] == ' ' || ptx[pos] == '\t'))
		pos++;
	auto start = pos;
	while (pos < ptx.size() && ptx[pos] != ' ' && ptx[pos] != '\t' &&
	       ptx[pos] != '\n' && ptx[pos] != '\r' && ptx[pos] != ',')
		pos++;
	if (pos > start)
		ptx.replace(start, pos - start, sm_arch);
	return ptx;
}
} // namespace

fatbin_record::~fatbin_record()
{
}
ptx_in_module::~ptx_in_module()
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

std::map<std::string, std::vector<uint8_t>> fatbin_record::compile_ptxs(
	class nv_attach_impl &impl,
	std::map<std::string, std::tuple<std::string, bool>> patched_ptx)
{
	std::string sm_arch = get_gpu_sm_arch();
	SPDLOG_INFO("Compiling PTXs with sm_arch {}", sm_arch);

	unsigned major, minor;
	NVPTXCOMPILER_CHECK_EXCEPTION(nvPTXCompilerGetVersion(&major, &minor),
				      "Unable to get compiler version");
	SPDLOG_INFO("Compiler version: {}.{}", major, minor);

	std::map<std::string, std::vector<uint8_t>> compiled_ptx;
	const auto &handler = impl.ptx_compiler;
	const auto extra_compile_options = build_nvptx_extra_options();
	const auto dump_cubin_dir = env_dir("BPFTIME_NV_ATTACH_DUMP_CUBIN_DIR");
	const auto dump_sassmap_dir = env_dir("BPFTIME_NV_ATTACH_DUMP_SASSMAP_DIR");

	boost::asio::thread_pool pool(std::thread::hardware_concurrency());
	std::mutex map_lock;
	for (const auto &[name, ptx_and_trampoline_flag] : patched_ptx) {
		const auto &ptx = std::get<0>(ptx_and_trampoline_flag);

			boost::asio::post(
				pool,
					[&handler, ptx, name, &compiled_ptx, &map_lock, this,
					 sm_arch, extra_compile_options, dump_cubin_dir,
					 dump_sassmap_dir]() -> void {
						const auto ptx_fixed =
							rewrite_ptx_target(ptx, sm_arch);
						const auto ptx_sha256 =
							sha256(ptx_fixed.data(), ptx_fixed.size());
						std::vector<std::string> extra =
							extra_compile_options.value_or(
								std::vector<std::string> {});
						auto make_compile_key = [&](size_t keep) {
							std::string key_material =
								ptx_fixed;
							key_material.append("\n#sm=");
							key_material.append(sm_arch);
							for (size_t i = 0; i < keep; i++) {
								key_material.append("\n#opt=");
								key_material.append(extra[i]);
							}
							return sha256(key_material.data(),
								      key_material.size());
						};

						std::optional<std::string> cache_key;
						for (size_t attempt = 0;
						     attempt <= extra.size(); attempt++) {
							const size_t keep =
								extra.size() - attempt;
							const auto key =
								make_compile_key(keep);
							if (this->ptx_pool->find(key) !=
							    this->ptx_pool->end()) {
								cache_key = key;
								break;
							}
						}

						if (cache_key) {
							SPDLOG_INFO(
							"PTX {} ({}) found in cache",
							name, *cache_key);
							std::lock_guard<std::mutex> _guard(
								map_lock);
							const auto &cached =
								this->ptx_pool->at(*cache_key);
							compiled_ptx[name] = cached;

							const auto elf_sha =
								sha256(cached.data(),
								       cached.size());
							maybe_dump_file(
								dump_cubin_dir,
								elf_sha + ".cubin",
								std::span<const uint8_t>(
									cached.data(),
									cached.size()));
							if (dump_sassmap_dir) {
								auto table = sass_map::
									parse_elf_debug_line(
										std::span<const uint8_t>(
											cached.data(),
											cached.size()));
								nlohmann::json j;
								j["elf_sha256"] = elf_sha;
								j["ptx_sha256"] = ptx_sha256;
								j["compile_key"] = *cache_key;
								j["sm_arch"] = sm_arch;
								j["ptx_name"] = name;
								j["has_debug_line"] =
									table.has_value();
								j["entry_count"] =
									table ? table->entries.size()
									      : 0;
								if (table) {
									auto &arr = j["entries"] =
										nlohmann::json::array();
									for (const auto &e :
									     table->entries) {
										nlohmann::json item;
										item["address"] =
											e.address;
										item["file"] =
											e.loc.file;
										item["line"] =
											e.loc.line;
										item["column"] =
											e.loc.column;
										item["is_stmt"] =
											e.loc.is_stmt;
										arr.push_back(std::move(item));
									}
								}
								const auto dump_dir =
									*dump_sassmap_dir;
								std::filesystem::create_directories(
									dump_dir);
								std::ofstream ofs(
									dump_dir /
									(elf_sha + ".json"));
								ofs << j.dump(2);
							}
						} else {
						SPDLOG_INFO(
							"Start compiling {}, not found in cache",
							name);
						uint8_t *data;
						size_t size;
					std::vector<uint8_t> compiled_program;
						std::string error_log;
						std::string info_log;

						std::string gpu_name = "--gpu-name=" + sm_arch;
						bool compiled = false;
						std::vector<std::string> used_extra;

						// Probe: start with all extra options, then drop from the tail.
					for (size_t attempt = 0;
					     attempt <= extra.size(); attempt++) {
						auto compiler = handler.create();
						if (!compiler) {
							throw std::runtime_error(
								"Unable to create nv_attach_impl_ptx_compiler");
						}

						std::vector<std::string> opts;
						opts.reserve(3 + extra.size());
						opts.push_back(gpu_name);
						opts.push_back("--verbose");
						opts.push_back("-O3");
						const size_t keep = extra.size() - attempt;
						for (size_t i = 0; i < keep; i++)
							opts.push_back(extra[i]);

						std::vector<const char *> c_opts;
						c_opts.reserve(opts.size());
						for (const auto &s : opts)
							c_opts.push_back(s.c_str());

						int err = handler.compile(
							compiler, ptx_fixed.c_str(),
							c_opts.data(), (int)c_opts.size());
						if (err == 0) {
							used_extra.assign(opts.begin() + 3,
									  opts.end());
							info_log = handler.get_info_log(
								compiler);
							handler.get_compiled_program(
								compiler, &data, &size);
							compiled_program.assign(data,
										data + size);
							handler.destroy(compiler);
							compiled = true;
							break;
						}

						error_log = handler.get_error_log(compiler);
						handler.destroy(compiler);
					}

					if (!compiled) {
						SPDLOG_ERROR(
							"Unable to compile {} (ptx_sha256={}): {}",
							name, ptx_sha256, error_log);
						throw std::runtime_error("Unable to compile");
					}
					if (!used_extra.empty()) {
						std::string joined;
						for (const auto &opt : used_extra) {
							if (!joined.empty())
								joined.push_back(' ');
							joined.append(opt);
						}
						SPDLOG_INFO(
							"nvptxcompiler extra options for {}: {}",
							name, joined);
					}
						if (!info_log.empty())
							SPDLOG_DEBUG("Info: {}", info_log);
						const auto compile_key =
							make_compile_key(used_extra.size());

						const auto elf_sha =
							sha256(compiled_program.data(),
							       compiled_program.size());
					maybe_dump_file(dump_cubin_dir,
							elf_sha + ".cubin",
							std::span<const uint8_t>(
								compiled_program.data(),
								compiled_program.size()));
					if (dump_sassmap_dir) {
						auto compiled_span =
							std::span<const uint8_t>(
								compiled_program.data(),
								compiled_program.size());
						auto table =
							sass_map::parse_elf_debug_line(
								compiled_span);
						nlohmann::json j;
						j["elf_sha256"] = elf_sha;
						j["ptx_sha256"] = ptx_sha256;
						j["compile_key"] = compile_key;
						j["sm_arch"] = sm_arch;
						j["ptx_name"] = name;
						j["has_debug_line"] =
							table.has_value();
						j["entry_count"] =
							table ? table->entries.size()
							      : 0;
						if (table) {
							auto &arr = j["entries"] =
								nlohmann::json::array();
							for (const auto &e :
							     table->entries) {
								nlohmann::json item;
								item["address"] = e.address;
								item["file"] = e.loc.file;
								item["line"] = e.loc.line;
								item["column"] =
									e.loc.column;
								item["is_stmt"] =
									e.loc.is_stmt;
								arr.push_back(
									std::move(item));
							}
						}
						const auto dump_dir = *dump_sassmap_dir;
						std::filesystem::create_directories(
							dump_dir);
						std::ofstream ofs(
							dump_dir / (elf_sha + ".json"));
						ofs << j.dump(2);
					}

						std::lock_guard<std::mutex> _guard(
							map_lock);
						compiled_ptx[name] = compiled_program;
						this->ptx_pool->insert(std::make_pair(
							compile_key,
							compiled_program));
					SPDLOG_INFO("Compile of {} done", name);
				}
			});
	}
	pool.join();
	return compiled_ptx;
}
void fatbin_record::try_loading_ptxs(class nv_attach_impl &impl)
{
	if (ptx_loaded)
		return;
	if (impl.shared_mem_ptr == 0) {
		throw std::runtime_error(
			"shared_mem_ptr is not initialized before loading PTX");
	}
	SPDLOG_INFO("Loading & patching current fatbin..");

	auto patched_ptx = *impl.hack_fatbin(original_ptx);

	auto compiled_ptx = compile_ptxs(impl, patched_ptx);

	for (const auto &[name, ptx_and_trampoline_flag] : patched_ptx) {
		const auto &ptx = std::get<0>(ptx_and_trampoline_flag);
		bool added_trampoline = std::get<1>(ptx_and_trampoline_flag);
		const auto &compiled_elf = compiled_ptx.at(name);
		auto sha256_string =
			sha256(compiled_elf.data(), compiled_elf.size());
		if (auto itr = module_pool->find(sha256_string);
		    itr != module_pool->end()) {
			SPDLOG_INFO("Module {} found in cache", name);
			ptxs.push_back(itr->second);
		} else {
			CUmodule module;
			SPDLOG_INFO("Loading module: {}, not found in cache",
				    name);
			char error_buf[8192] = { 0 }, info_buf[8192] = { 0 };
			CUjit_option options[] = {
				CU_JIT_INFO_LOG_BUFFER,
				CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
				CU_JIT_ERROR_LOG_BUFFER,
				CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
			};
			void *option_values[] = {
				(void *)info_buf, (void *)std::size(info_buf),
				(void *)error_buf, (void *)std::size(error_buf)
			};
			if (auto err = cuModuleLoadDataEx(
				    &module, compiled_elf.data(),
				    std::size(options), options, option_values);
			    err != CUDA_SUCCESS) {
				SPDLOG_ERROR("Unable to compile module {}: {}",
					     name, (int)err);
				SPDLOG_ERROR("Info: {}", info_buf);
				SPDLOG_ERROR("Error: {}", error_buf);
				throw std::runtime_error(
					"Unable to compile module");
			}
			if (added_trampoline) {
				CUdeviceptr const_data_ptr, map_basic_info_ptr;
				size_t const_data_size, map_basic_info_size;
				SPDLOG_INFO(
					"Copying trampoline data to device");
				CUDA_DRIVER_CHECK_EXCEPTION(
					cuModuleGetGlobal(&const_data_ptr,
							  &const_data_size,
							  module, "constData"),
					"Unable to get pointer of constData");
				SPDLOG_INFO(
			"constData symbol device_ptr={:x} size={} shared_mem_ptr={:x}",
			(uintptr_t)const_data_ptr, const_data_size,
			(uintptr_t)impl.shared_mem_ptr);
		CUDA_DRIVER_CHECK_EXCEPTION(
					cuModuleGetGlobal(&map_basic_info_ptr,
							  &map_basic_info_size,
							  module, "map_info"),
					"Unable to get pointer of map_info");
				SPDLOG_INFO("map_info symbol device_ptr={:x} size={}",
			    (uintptr_t)map_basic_info_ptr, map_basic_info_size);
		CUDA_DRIVER_CHECK_EXCEPTION(
					cuMemcpyHtoD(const_data_ptr,
						     &impl.shared_mem_ptr,
						     const_data_size),
					"Unable to copy constData pointer to device");
				CUDA_DRIVER_CHECK_EXCEPTION(
					cuMemcpyHtoD(map_basic_info_ptr,
						     impl.map_basic_info->data(),
						     map_basic_info_size),
					"Unable to copy constData pointer to device");
				SPDLOG_INFO("Trampoline data copied");
			}
			auto ptr = std::make_shared<ptx_in_module>(module);
			module_pool->insert(std::make_pair(sha256_string, ptr));
			ptxs.push_back(ptr);
			SPDLOG_INFO("Loaded module: {}", name);
		}
	}
	ptx_loaded = true;
}

} // namespace bpftime::attach
