// #include "pos/cuda_impl/utils/fatbin.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "spdlog/spdlog.h"
#include "vector_types.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <frida-gum.h>
#include <iterator>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <elf.h>
#include <string>
#include <string_view>
#include <time.h>
#include <unordered_set>
#include <vector>
#include "nv_attach_impl.hpp"
#include "sass_detour/sass_detour.hpp"
#include "json.hpp"
#include <stdexcept>
using namespace bpftime;
using namespace attach;

namespace {
// Internal per-thread overrides for the SASS detour patcher.
// Used to implement "patch-all-on-demand" around a specific launch without
// perturbing the process-wide filter state (learned func_id cache, etc.).
thread_local std::optional<std::string_view> tls_sass_detour_filter_override;
thread_local std::optional<std::string_view> tls_sass_sample_filter_override;
// Communicate patcher outcome to wrapper recorders:
// - true  => returned image has SM120 SASS detours applied
// - false => returned image may still be “rewritten” (e.g., jitlink PTX->cubin)
//          but is not detoured yet (so on-demand patching should be allowed).
thread_local bool tls_sass_detour_last_output_is_detoured = false;
// Best-effort base image for upgrade workflows (same format as the returned image,
// but without SM120 SASS detours). Only populated when we actually detour an ELF.
thread_local const void *tls_sass_detour_last_base_image = nullptr;
thread_local size_t tls_sass_detour_last_base_size = 0;
// Best-effort SM120 ELF image_id (matches `sass_detour::DetourResult::image_id`).
// Used to scope identify/tag closure and func_id caches to the actual SM120 code
// object, even when the input container is fatbinc/fatbin.
thread_local uint32_t tls_sass_detour_last_sm120_image_id = 0;

struct ScopedSassDetourFilterOverride {
	std::optional<std::string_view> prev_filter;
	std::optional<std::string_view> prev_sample;

	ScopedSassDetourFilterOverride(std::optional<std::string_view> filter,
				       std::optional<std::string_view> sample)
		: prev_filter(tls_sass_detour_filter_override),
		  prev_sample(tls_sass_sample_filter_override)
	{
		tls_sass_detour_filter_override = filter;
		tls_sass_sample_filter_override = sample;
	}

	~ScopedSassDetourFilterOverride()
	{
		tls_sass_detour_filter_override = prev_filter;
		tls_sass_sample_filter_override = prev_sample;
	}
};
} // namespace

#define CUDA_DRIVER_CHECK_EXCEPTION(expr, message)                             \
	do {                                                                   \
			if (auto err = expr; err != CUDA_SUCCESS) {                    \
				SPDLOG_ERROR("{}: {}", message, (int)err);             \
			throw std::runtime_error(message);                     \
		}                                                              \
	} while (false)

extern "C" {

typedef struct __attribute__((__packed__)) fat_elf_header {
	uint32_t magic;
	uint16_t version;
	uint16_t header_size;
	uint64_t size;
} fat_elf_header_t;
}

typedef struct _CUDARuntimeFunctionHooker {
	GObject parent;
} CUDARuntimeFunctionHooker;

static void cuda_runtime_function_hooker_iface_init(gpointer g_iface,
						    gpointer iface_data);

// #define EXAMPLE_TYPE_LISTENER (cuda_runtime_function_hooker_iface_init())
G_DECLARE_FINAL_TYPE(CUDARuntimeFunctionHooker, cuda_runtime_function_hooker,
		     BPFTIME, NV_ATTACH_IMPL, GObject)
G_DEFINE_TYPE_EXTENDED(
	CUDARuntimeFunctionHooker, cuda_runtime_function_hooker, G_TYPE_OBJECT,
	0,
	G_IMPLEMENT_INTERFACE(GUM_TYPE_INVOCATION_LISTENER,
			      cuda_runtime_function_hooker_iface_init))

using cu_graph_add_kernel_node_v1_fn_t =
	CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *, size_t,
		     const CUDA_KERNEL_NODE_PARAMS_v1 *);
using cu_graph_add_kernel_node_v2_fn_t = decltype(&cuGraphAddKernelNode_v2);
using cu_graph_exec_kernel_node_set_params_v1_fn_t = CUresult (*)(
	CUgraphExec, CUgraphNode, const CUDA_KERNEL_NODE_PARAMS_v1 *);
using cu_graph_exec_kernel_node_set_params_v2_fn_t =
	decltype(&cuGraphExecKernelNodeSetParams_v2);
using cu_graph_kernel_node_set_params_v1_fn_t =
	CUresult (*)(CUgraphNode, const CUDA_KERNEL_NODE_PARAMS_v1 *);
using cu_graph_kernel_node_set_params_v2_fn_t =
	decltype(&cuGraphKernelNodeSetParams_v2);
using cuda_memcpy_from_symbol_async_fn_t = decltype(&cudaMemcpyFromSymbolAsync);
using cuda_memcpy_from_symbol_fn_t = decltype(&cudaMemcpyFromSymbol);

using cuda_launch_kernel_fn_t = cudaError_t (*)(const void *, dim3, dim3,
						void **, size_t, cudaStream_t);

static bool cuda_graph_stream_is_capturing(cudaStream_t stream)
{
	cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
	auto err = cudaStreamIsCapturing(stream, &status);
	if (err != cudaSuccess)
		return true;
	return status != cudaStreamCaptureStatusNone;
}

static bool env_truthy_global(const char *key)
{
	const char *v = std::getenv(key);
	if (!v)
		return false;
	std::string s(v);
	std::transform(s.begin(), s.end(), s.begin(),
		       [](unsigned char c) { return (char)std::tolower(c); });
	return s == "1" || s == "true" || s == "yes" || s == "y" || s == "on";
}

static std::optional<uint32_t> env_u32_global(const char *key)
{
	const char *v = std::getenv(key);
	if (!v || !*v)
		return std::nullopt;
	char *end = nullptr;
	unsigned long x = std::strtoul(v, &end, 0);
	if (end == v)
		return std::nullopt;
	if (x > 0xfffffffful)
		return std::nullopt;
	return static_cast<uint32_t>(x);
}

static bool sass_identify_closure_enabled()
{
	return env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE");
}

static bool sass_detour_debug_enabled()
{
	return env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DEBUG");
}

static std::optional<std::filesystem::path> env_dir_global(const char *key)
{
	const char *v = std::getenv(key);
	if (!v || !*v)
		return std::nullopt;
	return std::filesystem::path(v);
}

static std::filesystem::path sass_detour_func_id_cache_path()
{
	if (const char *v =
		    std::getenv("BPFTIME_CUDA_SASS_DETOUR_FUNC_ID_CACHE_PATH");
	    v && *v) {
		return std::filesystem::path(v);
	}
	return std::filesystem::path("/tmp/bpftime-sass-funcid-cache.json");
}

static void maybe_load_sass_detour_func_id_cache(nv_attach_impl &impl,
						 std::string_view filter_sv)
{
	if (impl.sass_detour_filter_state.func_id_cache_loaded)
		return;
	impl.sass_detour_filter_state.func_id_cache_loaded = true;
	if (filter_sv.empty())
		return;

	const auto path = sass_detour_func_id_cache_path();
	std::error_code ec;
	if (!std::filesystem::exists(path, ec))
		return;

	try {
		std::ifstream ifs(path);
		if (!ifs.is_open())
			return;
		nlohmann::json j = nlohmann::json::parse(ifs, nullptr, false);
		if (j.is_discarded())
			return;
		auto filters_it = j.find("filters");
		if (filters_it == j.end() || !filters_it->is_object())
			return;
		auto entry_it = filters_it->find(std::string(filter_sv));
		if (entry_it == filters_it->end())
			return;
		auto &fs = impl.sass_detour_filter_state;
		size_t added = 0;
		auto load_ids = [&](const nlohmann::json &arr) {
			if (!arr.is_array())
				return;
			for (const auto &v : arr) {
				if (!v.is_number_unsigned())
					continue;
				uint64_t x = v.get<uint64_t>();
				if (x > 0xffffffffull)
					continue;
				if (fs.learned_func_ids.insert(static_cast<uint32_t>(x)).second)
					added++;
			}
		};
		// v1: filters[filter] = [func_id...]
		if (entry_it->is_array()) {
			load_ids(*entry_it);
		} else if (entry_it->is_object()) {
			// v2: filters[filter] = { func_ids:[...], by_image:{ "0x1234":[...], ... } }
			if (auto ids_it = entry_it->find("func_ids");
			    ids_it != entry_it->end()) {
				load_ids(*ids_it);
			}
			if (auto by_it = entry_it->find("by_image");
			    by_it != entry_it->end() && by_it->is_object()) {
				for (auto it = by_it->begin(); it != by_it->end(); ++it) {
					const std::string key = it.key();
					uint32_t image_id = 0;
					try {
						size_t idx = 0;
						unsigned long long v = std::stoull(key, &idx, 0);
						if (idx == 0 || v > 0xffffffffull)
							continue;
						image_id = static_cast<uint32_t>(v);
					} catch (...) {
						continue;
					}
					if (!it.value().is_array())
						continue;
					auto &set = fs.learned_func_ids_by_image_id[image_id];
					for (const auto &x : it.value()) {
						if (!x.is_number_unsigned())
							continue;
						uint64_t v = x.get<uint64_t>();
						if (v > 0xffffffffull)
							continue;
						const uint32_t id = static_cast<uint32_t>(v);
						set.insert(id);
						fs.learned_func_ids.insert(id);
					}
				}
			}
		} else {
			return;
		}
		if (added > 0 && env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DEBUG")) {
			SPDLOG_INFO(
				"SASS detour: loaded {} cached func_id(s) for filter='{}' from {}",
				added, std::string(filter_sv), path.string());
		}
	} catch (...) {
		return;
	}
}

static void maybe_store_sass_detour_func_id_cache(
	std::string_view filter_sv, const std::unordered_set<uint32_t> &ids,
	std::optional<uint32_t> image_id = std::nullopt,
	std::optional<uint32_t> image_target_func_id = std::nullopt)
{
	if (filter_sv.empty() || ids.empty())
		return;
	const auto path = sass_detour_func_id_cache_path();
	try {
		nlohmann::json j;
		{
			std::ifstream ifs(path);
			if (ifs.is_open()) {
				j = nlohmann::json::parse(ifs, nullptr, false);
			}
		}
		if (j.is_discarded() || !j.is_object())
			j = nlohmann::json::object();
		j["version"] = 1;
		if (!j.contains("filters") || !j["filters"].is_object())
			j["filters"] = nlohmann::json::object();
		// Upgrade entry format to v2 (object) while keeping backward compatibility
		// on load.
		j["version"] = 2;

		nlohmann::json entry = nlohmann::json::object();
		if (j["filters"].contains(std::string(filter_sv))) {
			entry = j["filters"][std::string(filter_sv)];
			if (entry.is_array()) {
				// v1 -> v2 conversion
				nlohmann::json old = entry;
				entry = nlohmann::json::object();
				entry["func_ids"] = std::move(old);
			} else if (!entry.is_object()) {
				entry = nlohmann::json::object();
			}
		}
		if (!entry.contains("func_ids") || !entry["func_ids"].is_array())
			entry["func_ids"] = nlohmann::json::array();
		if (!entry.contains("by_image") || !entry["by_image"].is_object())
			entry["by_image"] = nlohmann::json::object();

		std::vector<uint32_t> sorted(ids.begin(), ids.end());
		std::sort(sorted.begin(), sorted.end());
		nlohmann::json arr = nlohmann::json::array();
		for (uint32_t x : sorted)
			arr.push_back(x);
		entry["func_ids"] = std::move(arr);

		if (image_id && *image_id != 0 && image_target_func_id &&
		    *image_target_func_id != 0) {
			char key[16] = {};
			std::snprintf(key, sizeof(key), "0x%08x", *image_id);
			nlohmann::json one = nlohmann::json::array();
			one.push_back(*image_target_func_id);
			entry["by_image"][std::string(key)] = std::move(one);
		}

		j["filters"][std::string(filter_sv)] = std::move(entry);

		std::error_code ec;
		std::filesystem::create_directories(path.parent_path(), ec);
		std::ofstream ofs(path, std::ios::trunc);
		if (!ofs.is_open())
			return;
		ofs << j.dump(2) << "\n";
	} catch (...) {
		return;
	}
}

static void sass_control_arm_identify(nv_attach_impl &impl, CUstream stream,
				      std::string_view kernel_name,
				      uint32_t image_id)
{
	if (!sass_identify_closure_enabled())
		return;
	if (!impl.sass_sampling.enabled || !impl.sass_sampling.initialized ||
	    !impl.sass_sampling.control_enabled ||
	    impl.sass_sampling.buffer_data_offset !=
		    sass_detour::kSm120SassControlDataOffset ||
	    impl.sass_sampling.device_buffer == 0)
		return;

	auto cuMemcpyHtoD_v2 =
		reinterpret_cast<CUresult (*)(CUdeviceptr, const void *, size_t)>(
			impl.original_cu_memcpy_htod);
	auto cuMemcpyDtoH_v2 =
		reinterpret_cast<CUresult (*)(void *, CUdeviceptr, size_t)>(
			impl.original_cu_memcpy_dtoh);
	if (!cuMemcpyHtoD_v2)
		return;

	std::lock_guard<std::mutex> guard(impl.owned_cuda_images_lock);
	auto &fs = impl.sass_detour_filter_state;
	if (fs.identify_pending.load(std::memory_order_acquire))
		return;

	fs.identify_epoch++;
	sass_detour::Sm120SassControlHeader hdr {};
	hdr.magic = sass_detour::kSm120SassControlMagic;
	hdr.version = sass_detour::kSm120SassControlVersion;
	hdr.enable = 1u;
	hdr.mode = (uint32_t)sass_detour::Sm120SassControlMode::Identify;
	hdr.epoch = fs.identify_epoch;
	hdr.target_func_id = 0u;
	// Best-effort code-object identity: used to scope cached func_id propagation
	// for stripped raw-ELF SM120 modules.
	hdr.reserved0 = image_id;
	// Use synchronous copies for correctness: stream 0 can be legacy or per-thread
	// default depending on context flags, and mixing async ops can make the kernel
	// observe stale header values. This is a tiny fixed-size write (0x20 bytes).
	{
		const auto r0 = cuMemcpyHtoD_v2(impl.sass_sampling.device_buffer, &hdr,
					       sizeof(hdr));
		if (r0 != CUDA_SUCCESS) {
			SPDLOG_WARN("SASS control: cuMemcpyHtoD_v2(header) failed: {}",
				    int(r0));
			return;
		}
		if (!std::getenv("BPFTIME_CUDA_SASS_DETOUR_CONTROL_SKIP_CLEAR")) {
			std::array<uint32_t, sass_detour::kSm120SassControlSlotsCount>
				zeros {};
			const auto r1 = cuMemcpyHtoD_v2(
				impl.sass_sampling.device_buffer +
					(CUdeviceptr)sass_detour::
						kSm120SassControlSlotsOffset,
				zeros.data(), zeros.size() * sizeof(uint32_t));
			if (r1 != CUDA_SUCCESS) {
				SPDLOG_WARN("SASS control: cuMemcpyHtoD_v2(slots) failed: {}",
					    int(r1));
				return;
			}
		}
	}

	fs.identify_pending.store(true, std::memory_order_release);
	fs.identify_pending_kernel_name = std::string(kernel_name);
	fs.identify_target_func_id = 0;
	if (sass_detour_debug_enabled()) {
		SPDLOG_INFO("SASS control: armed identify epoch={} stream=0x{:x}",
			    fs.identify_epoch, (uint64_t)stream);
		if (cuMemcpyDtoH_v2) {
			sass_detour::Sm120SassControlHeader check {};
			std::array<uint32_t, sass_detour::kSm120SassControlSlotsCount>
				check_slots {};
			if (cuMemcpyDtoH_v2(&check, impl.sass_sampling.device_buffer,
					    sizeof(check)) == CUDA_SUCCESS &&
			    cuMemcpyDtoH_v2(
				    check_slots.data(),
				    impl.sass_sampling.device_buffer +
					    (CUdeviceptr)sass_detour::
						    kSm120SassControlSlotsOffset,
				    check_slots.size() * sizeof(uint32_t)) ==
				    CUDA_SUCCESS) {
				SPDLOG_INFO(
					"SASS control: pre-launch header enable={} mode={} epoch={} slots0={} slots1={}",
					check.enable, check.mode, check.epoch,
					check_slots[0], check_slots[1]);
			}
		}
	}
}

static void sass_control_maybe_dump_identify(nv_attach_impl &impl)
{
	if (!sass_identify_closure_enabled())
		return;
	if (!impl.sass_sampling.enabled || !impl.sass_sampling.initialized ||
	    !impl.sass_sampling.control_enabled ||
	    impl.sass_sampling.buffer_data_offset !=
		    sass_detour::kSm120SassControlDataOffset ||
	    impl.sass_sampling.device_buffer == 0)
		return;

	auto cuMemcpyDtoH_v2 =
		reinterpret_cast<CUresult (*)(void *, CUdeviceptr, size_t)>(
			impl.original_cu_memcpy_dtoh);
	auto cuMemcpyHtoD_v2 =
		reinterpret_cast<CUresult (*)(CUdeviceptr, const void *, size_t)>(
			impl.original_cu_memcpy_htod);
	if (!cuMemcpyDtoH_v2 || !cuMemcpyHtoD_v2)
		return;

	std::lock_guard<std::mutex> guard(impl.owned_cuda_images_lock);
	auto &fs = impl.sass_detour_filter_state;
	if (!fs.identify_pending.load(std::memory_order_acquire))
		return;

	sass_detour::Sm120SassControlHeader hdr {};
	std::array<uint32_t, sass_detour::kSm120SassControlSlotsCount> slots {};
	if (cuMemcpyDtoH_v2(&hdr, impl.sass_sampling.device_buffer, sizeof(hdr)) !=
	    CUDA_SUCCESS) {
		fs.identify_pending.store(false, std::memory_order_release);
		return;
	}
	if (cuMemcpyDtoH_v2(slots.data(),
			    impl.sass_sampling.device_buffer +
				    (CUdeviceptr)sass_detour::
					    kSm120SassControlSlotsOffset,
			    slots.size() * sizeof(uint32_t)) != CUDA_SUCCESS) {
		fs.identify_pending.store(false, std::memory_order_release);
		return;
	}

	std::unordered_set<uint32_t> ids;
	ids.reserve(slots.size());
	for (uint32_t x : slots) {
		if (x != 0u)
			ids.insert(x);
	}
	// Pick a stable "winner" func_id for this identify window:
	// - slots are indexed by (smid & 7), so the launched kernel should dominate
	//   the non-zero values even if other kernels happen to execute concurrently.
	uint32_t chosen = 0;
	if (!ids.empty()) {
		std::unordered_map<uint32_t, uint32_t> counts;
		counts.reserve(ids.size());
		for (uint32_t x : slots) {
			if (x != 0u)
				counts[x]++;
		}
		uint32_t best_cnt = 0;
		for (const auto &kv : counts) {
			const uint32_t id = kv.first;
			const uint32_t cnt = kv.second;
			if (cnt > best_cnt || (cnt == best_cnt && id < chosen) ||
			    (best_cnt == 0 && chosen == 0)) {
				best_cnt = cnt;
				chosen = id;
			}
		}
		if (chosen == 0)
			chosen = *ids.begin();
	}
		if (sass_detour_debug_enabled()) {
			std::string slots_s;
			slots_s.reserve(slots.size() * 12);
			for (size_t i = 0; i < slots.size(); i++) {
				if (i)
					slots_s.push_back(',');
				slots_s += std::to_string(slots[i]);
			}
			SPDLOG_INFO(
				"SASS control: identify dump epoch={} hdr(enable={},mode={},epoch={},target={},image_id=0x{:08x}) slots=[{}] unique_ids={} filter='{}'",
				fs.identify_epoch, hdr.enable, hdr.mode, hdr.epoch,
				hdr.target_func_id, hdr.reserved0, slots_s, ids.size(),
				fs.active_filter);
		}
	if (!ids.empty()) {
		if (fs.identify_target_func_id == 0)
			fs.identify_target_func_id = chosen;
		if (fs.identify_target_func_id != 0 &&
		    !fs.identify_pending_kernel_name.empty()) {
			fs.resolved_func_id_by_kernel_name[fs.identify_pending_kernel_name] =
				fs.identify_target_func_id;
		}

		// IMPORTANT: do NOT persist every non-zero slot value into `learned_func_ids`.
		// Identify windows can observe unrelated kernels (initialization/concurrency),
		// and func_id numbers are not globally unique across modules. Persist only the
		// chosen target func_id, and (when available) scope it by image_id.
		const uint32_t image_id = hdr.reserved0;
		const size_t before = fs.learned_func_ids.size();
		if (fs.identify_target_func_id != 0)
			fs.learned_func_ids.insert(fs.identify_target_func_id);
		if (image_id != 0 && fs.identify_target_func_id != 0) {
			fs.learned_func_ids_by_image_id[image_id].insert(
				fs.identify_target_func_id);
		}
		if (fs.learned_func_ids.size() != before || image_id != 0) {
			maybe_store_sass_detour_func_id_cache(
				fs.active_filter, fs.learned_func_ids, image_id,
				fs.identify_target_func_id);
		}

		sass_detour::Sm120SassControlHeader out = hdr;
		out.magic = sass_detour::kSm120SassControlMagic;
		out.version = sass_detour::kSm120SassControlVersion;
		out.enable = 1u;
		out.mode = (uint32_t)sass_detour::Sm120SassControlMode::Target;
		out.target_func_id = fs.identify_target_func_id;
		// Debug marker: host-side target selection.
		out.reserved0 = 0x54415247u; // "TARG"
		out.reserved1 = fs.identify_target_func_id;
		(void)cuMemcpyHtoD_v2(impl.sass_sampling.device_buffer, &out,
				      sizeof(out));
		// Keep host-side dedupe state in sync: subsequent `cuLaunchKernel` calls
		// will invoke `sass_control_set_target()` for every matching launch; avoid
		// repeating the same header write/log when we already transitioned to
		// Target mode here.
		{
			std::lock_guard<std::mutex> guard2(impl.sass_sampling.lock);
			auto &st = impl.sass_sampling;
			st.last_control_valid = true;
			st.last_control_epoch = fs.identify_epoch;
			st.last_control_target_func_id = fs.identify_target_func_id;
		}
	}

			fs.identify_pending.store(false, std::memory_order_release);
		fs.identify_pending_kernel_name.clear();
	}

static void sass_control_set_target(nv_attach_impl &impl, CUstream stream,
				    uint32_t func_id)
{
	if (!sass_identify_closure_enabled())
		return;
	if (!impl.sass_sampling.enabled || !impl.sass_sampling.initialized ||
	    !impl.sass_sampling.control_enabled ||
	    impl.sass_sampling.buffer_data_offset !=
		    sass_detour::kSm120SassControlDataOffset ||
	    impl.sass_sampling.device_buffer == 0)
		return;

	const uint32_t epoch = impl.sass_detour_filter_state.identify_epoch;
	{
		std::lock_guard<std::mutex> guard(impl.sass_sampling.lock);
		auto &st = impl.sass_sampling;
		if (st.last_control_valid && st.last_control_target_func_id == func_id &&
		    st.last_control_epoch == epoch) {
			return;
		}
	}

	auto cuMemcpyHtoD_v2 =
		reinterpret_cast<CUresult (*)(CUdeviceptr, const void *, size_t)>(
			impl.original_cu_memcpy_htod);
	if (!cuMemcpyHtoD_v2)
		return;
	sass_detour::Sm120SassControlHeader hdr {};
	hdr.magic = sass_detour::kSm120SassControlMagic;
	hdr.version = sass_detour::kSm120SassControlVersion;
	hdr.enable = 1u;
	hdr.mode = (uint32_t)sass_detour::Sm120SassControlMode::Target;
	hdr.epoch = epoch;
	hdr.target_func_id = func_id;
	// Debug marker: host-side target selection.
	hdr.reserved0 = 0x54415247u; // "TARG"
	hdr.reserved1 = func_id;
	{
		const auto r0 = cuMemcpyHtoD_v2(impl.sass_sampling.device_buffer, &hdr,
					       sizeof(hdr));
		if (r0 != CUDA_SUCCESS) {
			SPDLOG_WARN("SASS control: cuMemcpyHtoD_v2(set_target) failed: {}",
				    int(r0));
			return;
		}
		{
			std::lock_guard<std::mutex> guard(impl.sass_sampling.lock);
			auto &st = impl.sass_sampling;
			st.last_control_valid = true;
			st.last_control_epoch = epoch;
			st.last_control_target_func_id = func_id;
		}
		if (sass_detour_debug_enabled()) {
			SPDLOG_INFO(
				"SASS control: set target func_id={} stream=0x{:x}",
				func_id, (uint64_t)stream);
		}
	}
}

static void maybe_bind_jitlink_ptx_threadmap_globals(nv_attach_impl &impl,
						     CUfunction f,
						     std::string_view kernel_name,
						     CUstream hStream)
{
	// This path is only relevant when we injected PTX-level threadmap globals
	// during JIT-link. If we don't bind pointers/caps, the PTX snippet will
	// early-exit and bpftime's JSONL dump will stay empty.
	if (!env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_JITLINK_PTX") ||
	    !env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_JITLINK_PTX_THREADMAP")) {
		return;
	}

	// Avoid doing extra CUDA API calls under graph capture.
	if (cuda_graph_stream_is_capturing((cudaStream_t)hStream))
		return;

	auto sampling_cfg_opt = impl.get_sm120_sampling_cfg();
	if (!sampling_cfg_opt || !sampling_cfg_opt->enabled)
		return;
	const auto &cfg = *sampling_cfg_opt;
	if (cfg.mode != sass_detour::Sm120SamplingConfig::Mode::ThreadMap)
		return;
	if (!cfg.thread_map_device || !cfg.thread_map_device_stride4)
		return;

	// Resolve owning module for CUfunction so we can locate injected globals.
	auto mod_opt = impl.resolve_cumodule_for_cufunction(f);
	if (!mod_opt || *mod_opt == nullptr)
		return;
	const CUmodule mod = *mod_opt;

	// Cache per-module lookup to keep overhead low on real workloads.
	enum : uint8_t { kUnknown = 0, kBound = 1, kMissing = 2 };
	static std::mutex mu;
	static std::unordered_map<uint64_t, uint8_t> state;
	{
		std::lock_guard<std::mutex> g(mu);
		if (auto it = state.find((uint64_t)mod); it != state.end()) {
			if (it->second == kBound || it->second == kMissing)
				return;
		}
	}

	using cu_module_get_global_v2_fn_t =
		CUresult (*)(CUdeviceptr *, size_t *, CUmodule, const char *);
	static cu_module_get_global_v2_fn_t cuModuleGetGlobal_v2 =
		(cu_module_get_global_v2_fn_t)dlsym(RTLD_DEFAULT,
						    "cuModuleGetGlobal_v2");
	if (!cuModuleGetGlobal_v2) {
		// Some driver versions export only cuModuleGetGlobal.
		cuModuleGetGlobal_v2 = (cu_module_get_global_v2_fn_t)dlsym(
			RTLD_DEFAULT, "cuModuleGetGlobal");
	}
	if (!cuModuleGetGlobal_v2)
		return;

	CUdeviceptr g_out_ptr = 0;
	CUdeviceptr g_out_cap = 0;
	CUdeviceptr g_out_gate = 0;
	size_t out_ptr_sz = 0;
	size_t out_cap_sz = 0;
	size_t out_gate_sz = 0;
	const auto r0 = cuModuleGetGlobal_v2(&g_out_ptr, &out_ptr_sz, mod,
					     "__bpftime_ptx_out_ptr");
	const auto r1 = cuModuleGetGlobal_v2(&g_out_cap, &out_cap_sz, mod,
					     "__bpftime_ptx_out_cap");
	const auto r2 = cuModuleGetGlobal_v2(&g_out_gate, &out_gate_sz, mod,
					     "__bpftime_ptx_out_gate");
	if (r0 != CUDA_SUCCESS || r1 != CUDA_SUCCESS || r2 != CUDA_SUCCESS ||
	    g_out_ptr == 0 || g_out_cap == 0 || g_out_gate == 0 ||
	    out_ptr_sz < sizeof(uint64_t) || out_cap_sz < sizeof(uint32_t) ||
	    out_gate_sz < sizeof(uint32_t)) {
		std::lock_guard<std::mutex> g(mu);
		state[(uint64_t)mod] = kMissing;
		if (env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DEBUG")) {
			SPDLOG_INFO(
				"SASS detour: jitlink PTX threadmap globals not found (mod=0x{:x} r=[{}, {}, {}])",
				(uint64_t)mod, (int)r0, (int)r1, (int)r2);
		}
		return;
	}

	// Bind to bpftime sampler buffer data region (u32 slots).
	const uint64_t out_ptr =
		cfg.sample_buffer_device_ptr + uint64_t(cfg.buffer_data_offset);
	const uint32_t out_cap = 1024u;
	enum : uint32_t {
		kGateLane0Only = 1u,
		kGateWarp0Only = 2u,
		kGateCtaClamp = 4u,
	};
	uint32_t gate = 0;
	if (cfg.thread_map_device_lane0_only)
		gate |= kGateLane0Only;
	if (cfg.thread_map_device_warp0_only)
		gate |= kGateWarp0Only;
	if (cfg.thread_map_device_cta_clamp)
		gate |= kGateCtaClamp;

	auto cuMemcpyHtoDAsync_v2 =
		reinterpret_cast<CUresult (*)(CUdeviceptr, const void *, size_t,
					      CUstream)>(
			impl.original_cu_memcpy_htod_async);
	auto cuMemcpyHtoD_v2 =
		reinterpret_cast<CUresult (*)(CUdeviceptr, const void *, size_t)>(
			impl.original_cu_memcpy_htod);
	auto cuMemsetD32Async =
		reinterpret_cast<CUresult (*)(CUdeviceptr, unsigned int, size_t,
					      CUstream)>(
			impl.original_cu_memset_d32_async);
	if (!cuMemcpyHtoD_v2)
		return;

	// Best-effort: clear first record region (1 CTA, 1024 slots) so JSONL dump
	// isn't polluted by stale data from previous runs in long-lived processes.
	if (cuMemsetD32Async) {
		(void)cuMemsetD32Async((CUdeviceptr)out_ptr, 0xffffffffu,
				       size_t(out_cap), hStream);
	}

	const bool use_async = (cuMemcpyHtoDAsync_v2 != nullptr);
	if (use_async) {
		(void)cuMemcpyHtoDAsync_v2(g_out_ptr, &out_ptr, sizeof(out_ptr),
					  hStream);
		(void)cuMemcpyHtoDAsync_v2(g_out_cap, &out_cap, sizeof(out_cap),
					  hStream);
		(void)cuMemcpyHtoDAsync_v2(g_out_gate, &gate, sizeof(gate),
					  hStream);
	} else {
		(void)cuMemcpyHtoD_v2(g_out_ptr, &out_ptr, sizeof(out_ptr));
		(void)cuMemcpyHtoD_v2(g_out_cap, &out_cap, sizeof(out_cap));
		(void)cuMemcpyHtoD_v2(g_out_gate, &gate, sizeof(gate));
	}

	{
		std::lock_guard<std::mutex> g(mu);
		state[(uint64_t)mod] = kBound;
	}
	if (env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DEBUG")) {
		SPDLOG_INFO(
			"SASS detour: jitlink PTX threadmap bound (name='{}' mod=0x{:x} out_ptr=0x{:x} cap={} gate=0x{:x})",
			std::string(kernel_name), (uint64_t)mod, out_ptr, out_cap,
			gate);
	}
}

static std::string sanitize_filename_component(std::string_view s, size_t max_len)
{
	std::string out;
	out.reserve(std::min(max_len, s.size()));
	for (char c : s) {
		const bool ok =
			(c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
			(c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
		out.push_back(ok ? c : '_');
		if (out.size() >= max_len)
			break;
	}
	while (!out.empty() && out.back() == '_')
		out.pop_back();
	if (out.empty())
		out = "unknown";
	return out;
}

static std::string json_escape_string(std::string_view s)
{
	std::string out;
	out.reserve(s.size() + 8);
	for (char c : s) {
		switch (c) {
		case '\\':
			out += "\\\\";
			break;
		case '"':
			out += "\\\"";
			break;
		case '\n':
			out += "\\n";
			break;
		case '\r':
			out += "\\r";
			break;
		case '\t':
			out += "\\t";
			break;
		default:
			out.push_back(c);
			break;
		}
	}
	return out;
}

static uint64_t fnv1a64(const uint8_t *data, size_t size)
{
	uint64_t h = 1469598103934665603ull;
	for (size_t i = 0; i < size; i++) {
		h ^= data[i];
		h *= 1099511628211ull;
	}
	return h;
}

static uint32_t sass_image_id32(std::span<const uint8_t> bytes)
{
	const uint64_t h = fnv1a64(bytes.data(), bytes.size());
	return uint32_t((h & 0xffffffffull) ^ (h >> 32));
}

static uint32_t sass_elf_image_id32(std::span<const uint8_t> elf_bytes)
{
	// Keep this consistent with `sass_detour::DetourResult::image_id` computation:
	// hash only the meaningful ELF extent (section payloads + tables), not the
	// entire host-provided buffer (raw-ELF loads can include extra bytes/padding).
	if (elf_bytes.size() < sizeof(Elf64_Ehdr))
		return sass_image_id32(elf_bytes);

	Elf64_Ehdr ehdr {};
	std::memcpy(&ehdr, elf_bytes.data(), sizeof(ehdr));
	if (std::memcmp(ehdr.e_ident, "\x7f"
					"ELF",
			4) != 0)
		return sass_image_id32(elf_bytes);

	if (ehdr.e_shoff == 0 || ehdr.e_shentsize == 0 || ehdr.e_shnum == 0 ||
	    ehdr.e_shentsize != sizeof(Elf64_Shdr))
		return sass_image_id32(elf_bytes);

	const size_t shoff = static_cast<size_t>(ehdr.e_shoff);
	const size_t shnum = static_cast<size_t>(ehdr.e_shnum);
	if (shoff + shnum * sizeof(Elf64_Shdr) > elf_bytes.size())
		return sass_image_id32(elf_bytes);

	size_t file_end = 0;
	for (size_t i = 0; i < shnum; i++) {
		Elf64_Shdr sh {};
		std::memcpy(&sh, elf_bytes.data() + shoff + i * sizeof(Elf64_Shdr),
			    sizeof(sh));
		const size_t end =
			static_cast<size_t>(sh.sh_offset) + static_cast<size_t>(sh.sh_size);
		if (end > file_end)
			file_end = end;
	}
	file_end = std::max(file_end, shoff + shnum * sizeof(Elf64_Shdr));

	if (ehdr.e_phoff != 0 && ehdr.e_phnum != 0 &&
	    ehdr.e_phentsize == sizeof(Elf64_Phdr)) {
		const size_t phoff = static_cast<size_t>(ehdr.e_phoff);
		const size_t phnum = static_cast<size_t>(ehdr.e_phnum);
		if (phoff + phnum * sizeof(Elf64_Phdr) <= elf_bytes.size()) {
			file_end = std::max(file_end, phoff + phnum * sizeof(Elf64_Phdr));
		}
	}

	if (file_end == 0 || file_end > elf_bytes.size())
		file_end = elf_bytes.size();

	return sass_image_id32(elf_bytes.subspan(0, file_end));
}

static void maybe_clone_launch_attributes(CUfunction src, CUfunction dst)
{
	if (src == nullptr || dst == nullptr || src == dst)
		return;
	using cu_func_get_attribute_fn_t =
		CUresult (*)(int *, CUfunction_attribute, CUfunction);
	using cu_func_set_attribute_fn_t =
		CUresult (*)(CUfunction, CUfunction_attribute, int);
	static cu_func_get_attribute_fn_t cu_func_get_attribute =
		(cu_func_get_attribute_fn_t)dlsym(RTLD_DEFAULT, "cuFuncGetAttribute");
	static cu_func_set_attribute_fn_t cu_func_set_attribute =
		(cu_func_set_attribute_fn_t)dlsym(RTLD_DEFAULT, "cuFuncSetAttribute");
	using cu_func_get_cache_config_fn_t =
		CUresult (*)(CUfunc_cache *, CUfunction);
	using cu_func_set_cache_config_fn_t =
		CUresult (*)(CUfunction, CUfunc_cache);
	using cu_func_get_shared_mem_config_fn_t =
		CUresult (*)(CUsharedconfig *, CUfunction);
	using cu_func_set_shared_mem_config_fn_t =
		CUresult (*)(CUfunction, CUsharedconfig);
	static cu_func_get_cache_config_fn_t cu_func_get_cache_config =
		(cu_func_get_cache_config_fn_t)dlsym(RTLD_DEFAULT, "cuFuncGetCacheConfig");
	static cu_func_set_cache_config_fn_t cu_func_set_cache_config =
		(cu_func_set_cache_config_fn_t)dlsym(RTLD_DEFAULT, "cuFuncSetCacheConfig");
	static cu_func_get_shared_mem_config_fn_t cu_func_get_shared_mem_config =
		(cu_func_get_shared_mem_config_fn_t)dlsym(RTLD_DEFAULT, "cuFuncGetSharedMemConfig");
	static cu_func_set_shared_mem_config_fn_t cu_func_set_shared_mem_config =
		(cu_func_set_shared_mem_config_fn_t)dlsym(RTLD_DEFAULT, "cuFuncSetSharedMemConfig");
	if (!cu_func_get_attribute || !cu_func_set_attribute)
		return;

	auto clone_attr = [&](CUfunction_attribute attr) {
		int v = 0;
		if (auto r = cu_func_get_attribute(&v, attr, src); r != CUDA_SUCCESS)
			return;
		(void)cu_func_set_attribute(dst, attr, v);
	};

	// Flashattention/other vendor kernels may rely on these attributes being
	// configured on the CUfunction handle. If we load a replacement module for
	// on-demand upgrades, the new CUfunction starts with defaults and the next
	// launch can fail with CUDA_ERROR_INVALID_VALUE.
	clone_attr(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES);
	clone_attr(CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT);

	if (cu_func_get_cache_config && cu_func_set_cache_config) {
		CUfunc_cache cfg = CU_FUNC_CACHE_PREFER_NONE;
		if (auto r = cu_func_get_cache_config(&cfg, src); r == CUDA_SUCCESS)
			(void)cu_func_set_cache_config(dst, cfg);
	}
	if (cu_func_get_shared_mem_config && cu_func_set_shared_mem_config) {
		CUsharedconfig cfg = CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE;
		if (auto r = cu_func_get_shared_mem_config(&cfg, src); r == CUDA_SUCCESS)
			(void)cu_func_set_shared_mem_config(dst, cfg);
	}
}

static void maybe_dump_cuda_image_bytes(
	const std::optional<std::filesystem::path> &dir, const char *api_name,
	uint32_t magic, std::string_view kind, std::string_view reason,
	const uint8_t *bytes, size_t size)
{
	if (!dir || !bytes || size == 0)
		return;
	std::filesystem::create_directories(*dir);
	const uint64_t h = fnv1a64(bytes, size);
	const std::string api = sanitize_filename_component(
		api_name ? std::string_view(api_name) : std::string_view("api"),
		32);
	const std::string why = sanitize_filename_component(reason, 80);
	const std::string k = sanitize_filename_component(kind, 24);
	char magic_hex[11] = {};
	std::snprintf(magic_hex, sizeof(magic_hex), "0x%08x", magic);
	std::string fn = api + "_" + k + "_" + magic_hex + "_" + why + "_" +
			 std::to_string(h) + ".bin";
	std::ofstream ofs(*dir / fn, std::ios::binary);
	ofs.write(reinterpret_cast<const char *>(bytes),
		  static_cast<std::streamsize>(size));
}

static const void *maybe_patch_cuda_image_sass_detour(
	nv_attach_impl *impl, const void *code, size_t code_size,
	const char *api_name, size_t *out_size)
{
	if (!impl || code == nullptr || !env_truthy_global("BPFTIME_CUDA_SASS_DETOUR"))
		return code;
	// Default: treat the output as "not detoured" unless we explicitly apply
	// SM120 SASS detours below. This is used by the wrapper recorders to decide
	// whether on-demand patching is still allowed.
	tls_sass_detour_last_output_is_detoured = false;
	tls_sass_detour_last_base_image = nullptr;
	tls_sass_detour_last_base_size = 0;
	tls_sass_detour_last_sm120_image_id = 0;
	if (out_size != nullptr)
		*out_size = code_size;
	if (code_size != 0 && code_size < sizeof(uint32_t))
		return code;

	std::unique_lock<std::mutex> guard(impl->owned_cuda_images_lock);

	auto sampling_cfg_opt = impl->get_sm120_sampling_cfg();
	const auto *sampling_cfg = sampling_cfg_opt ? &*sampling_cfg_opt : nullptr;
	uint32_t out_sm120_image_id = 0;

	if (auto it = impl->sass_detour_code_cache.find(code);
	    it != impl->sass_detour_code_cache.end()) {
		tls_sass_detour_last_output_is_detoured = it->second.detoured;
		tls_sass_detour_last_base_image = it->second.base;
		tls_sass_detour_last_base_size = it->second.base_size;
		tls_sass_detour_last_sm120_image_id = it->second.sm120_image_id;
		if (out_size != nullptr && it->second.size != 0)
			*out_size = it->second.size;
		return it->second.patched;
	}

	uint32_t magic = 0;
	std::memcpy(&magic, code, sizeof(magic));
	if (env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DEBUG")) {
		SPDLOG_INFO("SASS detour: {} code={} magic=0x{:08x}", api_name, code,
			    magic);
	}

	const char *filter = std::getenv("BPFTIME_CUDA_SASS_DETOUR_FILTER");
	const bool filter_overridden = tls_sass_detour_filter_override.has_value();
	std::string_view filter_sv =
		filter_overridden ? *tls_sass_detour_filter_override
				  : (filter ? std::string_view(filter) : "");
	// 1.2 (tag/func_id closure): keep a process-wide learned func_id cache keyed by
	// the current filter string. This is required because stripped SM120 raw-ELF
	// cubins often lack `.text.<kernel>` section names, but other code objects in
	// the same fatbin may still carry `.nv.info.<kernel>` section names whose
	// `sh_info` (func_id) matches the stripped SM120 entry.
	if (!filter_overridden) {
		if (std::string_view(impl->sass_detour_filter_state.active_filter) !=
		    filter_sv) {
			impl->sass_detour_filter_state.active_filter =
				std::string(filter_sv);
			impl->sass_detour_filter_state.learned_func_ids.clear();
			impl->sass_detour_filter_state.learned_func_ids_by_image_id.clear();
			impl->sass_detour_filter_state.resolved_func_id_by_kernel_name.clear();
			impl->sass_detour_filter_state.recent_fatbinc_filter_budget = 0;
			impl->sass_detour_filter_state.func_id_cache_loaded = false;
		}
		maybe_load_sass_detour_func_id_cache(*impl, filter_sv);
	}
	std::unordered_set<uint32_t> extra_filter_func_ids;
	if (const char *v =
		    std::getenv("BPFTIME_CUDA_SASS_DETOUR_FILTER_FUNC_IDS");
	    v && *v) {
		const char *p = v;
		while (*p) {
			while (*p == ' ' || *p == '\t' || *p == '\n' ||
			       *p == '\r' || *p == ',')
				p++;
			if (!*p)
				break;
			char *end = nullptr;
			unsigned long x = std::strtoul(p, &end, 0);
			if (end == p)
				break;
			if (x <= 0xfffffffful)
				extra_filter_func_ids.insert(
					static_cast<uint32_t>(x));
			p = end ? end : p;
		}
	}
	const std::unordered_set<uint32_t> *extra_filter_func_ids_ptr =
		extra_filter_func_ids.empty() ? nullptr : &extra_filter_func_ids;
	const char *sample_filter = std::getenv("BPFTIME_CUDA_SASS_SAMPLE_FILTER");
	// Sampling filter:
	// - If user explicitly sets `BPFTIME_CUDA_SASS_SAMPLE_FILTER`, we use it to
	//   decide which `.text.*` sections should get the sampling stub.
	// - Otherwise, keep it empty so that sampling "follows selection": any
	//   section that passes the detour filter (including func_id propagation for
	//   stripped SM120 code objects) will be eligible for sampling.
	const bool sample_filter_overridden =
		tls_sass_sample_filter_override.has_value();
	std::string_view sample_filter_sv =
		sample_filter_overridden
			? *tls_sass_sample_filter_override
			: ((sample_filter && *sample_filter)
				   ? std::string_view(sample_filter)
				   : std::string_view {});
		const auto dump_dir = env_dir_global("BPFTIME_CUDA_SASS_DETOUR_DUMP_DIR");
		const bool dump_unpatched =
			env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DUMP_UNPATCHED");
		const bool dump_patched =
			env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DUMP_PATCHED");

		auto merged_filter_func_ids_for_image =
			[&](bool include_learned,
			    std::optional<uint32_t> image_id = std::nullopt)
				-> std::optional<std::unordered_set<uint32_t>> {
			// When the caller already knows the precise target func_id for this
			// process (e.g., a bring-up harness that ran an identify pass and now
			// wants a stable, minimal-impact thread-map/device run), force the
			// detour filter set to exactly {target}. This avoids detouring many
			// template instantiations that share the same kernel substring (common
			// for flashattention), which can otherwise destabilize real workloads.
			if (auto forced = env_u32_global(
				    "BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE_FORCE_TARGET_FUNC_ID");
			    forced && *forced != 0) {
				std::unordered_set<uint32_t> merged;
				merged.insert(*forced);
				return merged;
			}
			std::unordered_set<uint32_t> merged;
		const bool has_any_image_scoped =
			!impl->sass_detour_filter_state.learned_func_ids_by_image_id.empty();
		// Once we have an image-scoped mapping for this filter, prefer it and avoid
		// applying the global `learned_func_ids` set to unrelated code objects (func_id
		// values are not globally unique). Allow overriding for debugging.
		const bool allow_global_fallback =
			env_truthy_global(
				"BPFTIME_CUDA_SASS_DETOUR_FUNC_ID_CACHE_ALLOW_GLOBAL_FALLBACK");
		const bool strict_image_scoped =
			(has_any_image_scoped && !allow_global_fallback);
		if (image_id) {
			if (auto it =
				    impl->sass_detour_filter_state.learned_func_ids_by_image_id.find(
					    *image_id);
			    it != impl->sass_detour_filter_state.learned_func_ids_by_image_id.end()) {
				merged = it->second;
			} else if (strict_image_scoped) {
				include_learned = false;
			}
		}
		if (merged.empty() && include_learned) {
			merged = impl->sass_detour_filter_state.learned_func_ids;
		}
		if (extra_filter_func_ids_ptr) {
			merged.insert(extra_filter_func_ids_ptr->begin(),
				      extra_filter_func_ids_ptr->end());
		}
		if (merged.empty())
			return std::nullopt;
		return merged;
	};

	constexpr uint32_t FATBIN_TEXT_MAGIC = 0xBA55ED50;
	constexpr uint32_t FATBINC_MAGIC = 0x466243B1;
	constexpr size_t kMaxFatbinBytes = 256u << 20; // best-effort cap per fatbin
	constexpr size_t kMaxFatPrefixScanBytes = 256u << 10;
	constexpr size_t kMaxFatbinsV2 = 64;
	constexpr size_t kMaxModuleDecompressBound = 64u << 20;

	struct alignas(8) FatbinHeaderLocal {
		uint32_t magic;
		uint16_t version;
		uint16_t header_size;
		uint64_t files_size;
	};
	static_assert(sizeof(FatbinHeaderLocal) == 16);

	struct FatbinFileHeaderLocal {
		uint16_t kind;
		uint16_t version;
		uint32_t header_size;
		uint32_t padded_payload_size;
		uint32_t unknown0;
		uint32_t payload_size;
		uint32_t unknown1;
		uint32_t unknown2;
		uint32_t sm_version;
		uint32_t bit_width;
		uint32_t unknown3;
		uint64_t unknown4;
		uint64_t unknown5;
		uint64_t uncompressed_payload;
	};
	static_assert(sizeof(FatbinFileHeaderLocal) == 0x40);

	auto align_up = [](size_t v, size_t a) -> size_t {
		if (a == 0)
			return v;
		return (v + a - 1) & ~(a - 1);
	};

	struct Lz4Api {
		using decompress_fn_t = int (*)(const char *, char *, int, int);
		using compress_fn_t = int (*)(const char *, char *, int, int);
		using compress_bound_fn_t = int (*)(int);

		void *handle = nullptr;
		decompress_fn_t decompress = nullptr;
		compress_fn_t compress = nullptr;
		compress_bound_fn_t compress_bound = nullptr;

		static const Lz4Api &instance()
		{
			static Lz4Api api = [] {
				Lz4Api out;
				out.handle = dlopen("liblz4.so.1", RTLD_LAZY | RTLD_LOCAL);
				if (!out.handle)
					out.handle = dlopen("liblz4.so", RTLD_LAZY | RTLD_LOCAL);
				if (!out.handle)
					return out;
				out.decompress = (decompress_fn_t)dlsym(
					out.handle, "LZ4_decompress_safe");
				out.compress = (compress_fn_t)dlsym(
					out.handle, "LZ4_compress_default");
				out.compress_bound = (compress_bound_fn_t)dlsym(
					out.handle, "LZ4_compressBound");
				if (!out.decompress || !out.compress || !out.compress_bound) {
					dlclose(out.handle);
					out.handle = nullptr;
					out.decompress = nullptr;
					out.compress = nullptr;
					out.compress_bound = nullptr;
				}
				return out;
			}();
			return api;
		}
	};

	auto find_fatbin_text_magic_offset =
		[](const uint8_t *data, size_t max_scan) -> std::optional<size_t> {
		if (!data || max_scan < 4)
			return std::nullopt;
		for (size_t off = 0; off + 4 <= max_scan; off++) {
			uint32_t v = 0;
			std::memcpy(&v, data + off, sizeof(v));
			if (v == FATBIN_TEXT_MAGIC)
				return off;
		}
		return std::nullopt;
	};

	auto parse_fatbin_total_size = [&](const uint8_t *data,
					  size_t cap) -> std::optional<size_t> {
		if (!data || cap < sizeof(FatbinHeaderLocal))
			return std::nullopt;
		FatbinHeaderLocal h {};
		std::memcpy(&h, data, sizeof(h));
		if (h.magic != FATBIN_TEXT_MAGIC)
			return std::nullopt;
		if (h.version != 1)
			return std::nullopt;
		if (h.header_size < sizeof(FatbinHeaderLocal))
			return std::nullopt;
		const size_t header_size = static_cast<size_t>(h.header_size);
		const size_t files_size = static_cast<size_t>(h.files_size);
		if (header_size > cap)
			return std::nullopt;
		if (files_size > cap - header_size)
			return std::nullopt;
		const size_t total = header_size + files_size;
		if (total == 0 || total > cap)
			return std::nullopt;
		return total;
	};

	struct PatchedFatbin {
		std::vector<uint8_t> bytes;
		bool any_patched = false;
		size_t patched_text_sections = 0;
		size_t sampled_text_sections = 0;
		std::vector<sass_detour::InstrumentedKernelInfo> sampled_kernels;
		bool saw_filter_name = false;
		std::string reason;
	};

	auto decompress_lz4 = [&](const uint8_t *src, size_t src_size,
				  size_t expected_size) -> std::optional<std::vector<uint8_t>> {
		const auto &lz4 = Lz4Api::instance();
		if (!lz4.decompress)
			return std::nullopt;
		size_t out_cap = std::max<size_t>(1024, expected_size);
		out_cap = std::min(out_cap, kMaxModuleDecompressBound);
		std::vector<uint8_t> out(out_cap);
		while (true) {
			int r = lz4.decompress(reinterpret_cast<const char *>(src),
					       reinterpret_cast<char *>(out.data()),
					       (int)src_size, (int)out.size());
			if (r >= 0) {
				out.resize((size_t)r);
				return out;
			}
			const size_t new_cap = out.size() * 2;
			if (new_cap <= out.size() || new_cap > kMaxModuleDecompressBound)
				return std::nullopt;
			out.resize(new_cap);
		}
	};

	auto compress_lz4 = [&](std::span<const uint8_t> src)
		-> std::optional<std::vector<uint8_t>> {
		const auto &lz4 = Lz4Api::instance();
		if (!lz4.compress || !lz4.compress_bound)
			return std::nullopt;
		const int bound = lz4.compress_bound((int)src.size());
		if (bound <= 0 || (size_t)bound > (1u << 31))
			return std::nullopt;
		std::vector<uint8_t> out((size_t)bound);
		const int r = lz4.compress(reinterpret_cast<const char *>(src.data()),
					   reinterpret_cast<char *>(out.data()),
					   (int)src.size(), bound);
		if (r <= 0)
			return std::nullopt;
		out.resize((size_t)r);
		return out;
	};

	auto blob_contains = [](std::span<const uint8_t> blob,
				std::string_view needle) -> bool {
		if (needle.empty() || blob.empty())
			return false;
		const char *p = reinterpret_cast<const char *>(blob.data());
		const size_t n = blob.size();
		// Best-effort substring scan; avoids bringing in heavier search.
		for (size_t i = 0; i + needle.size() <= n; i++) {
			if (std::memcmp(p + i, needle.data(), needle.size()) == 0)
				return true;
		}
		return false;
	};

	auto patch_fatbin_rebuild =
		[&](std::span<const uint8_t> fatbin_bytes,
		    bool force_decompress_elf,
		    bool assume_contains_filter_name) -> std::optional<PatchedFatbin> {
		if (fatbin_bytes.size() < sizeof(FatbinHeaderLocal))
			return std::nullopt;
		FatbinHeaderLocal hdr {};
		std::memcpy(&hdr, fatbin_bytes.data(), sizeof(hdr));
		if (hdr.magic != FATBIN_TEXT_MAGIC || hdr.version != 1 ||
		    hdr.header_size < sizeof(FatbinHeaderLocal))
			return std::nullopt;
		const size_t header_size = static_cast<size_t>(hdr.header_size);
		if (header_size > fatbin_bytes.size())
			return std::nullopt;
		const size_t files_size = static_cast<size_t>(hdr.files_size);
		if (files_size > fatbin_bytes.size() - header_size)
			return std::nullopt;

		const auto file_region =
			fatbin_bytes.subspan(header_size, files_size);

		auto collect_filter_func_ids_from_elf =
			[&](std::span<const uint8_t> elf,
			    std::unordered_set<uint32_t> &out_ids) {
				if (filter_sv.empty() || elf.size() < 64)
					return;
				struct Elf64_EhdrLocal {
					uint8_t e_ident[16];
					uint16_t e_type;
					uint16_t e_machine;
					uint32_t e_version;
					uint64_t e_entry;
					uint64_t e_phoff;
					uint64_t e_shoff;
					uint32_t e_flags;
					uint16_t e_ehsize;
					uint16_t e_phentsize;
					uint16_t e_phnum;
					uint16_t e_shentsize;
					uint16_t e_shnum;
					uint16_t e_shstrndx;
				} eh {};
				std::memcpy(&eh, elf.data(), sizeof(eh));
				if (eh.e_ident[0] != 0x7f || eh.e_ident[1] != 'E' ||
				    eh.e_ident[2] != 'L' || eh.e_ident[3] != 'F')
					return;
				if (eh.e_ident[4] != 2 /* ELFCLASS64 */ ||
				    eh.e_ident[5] != 1 /* little */)
					return;
				const size_t shoff = static_cast<size_t>(eh.e_shoff);
				const size_t shentsize = static_cast<size_t>(eh.e_shentsize);
				const size_t shnum = static_cast<size_t>(eh.e_shnum);
				const size_t shstrndx = static_cast<size_t>(eh.e_shstrndx);
				if (shoff == 0 || shentsize < 64 || shnum == 0)
					return;
				if (shstrndx == 0 || shstrndx >= shnum)
					return;
				if (shoff > elf.size() || shoff + shentsize * shnum > elf.size())
					return;

				struct Elf64_ShdrLocal {
					uint32_t sh_name;
					uint32_t sh_type;
					uint64_t sh_flags;
					uint64_t sh_addr;
					uint64_t sh_offset;
					uint64_t sh_size;
					uint32_t sh_link;
					uint32_t sh_info;
					uint64_t sh_addralign;
					uint64_t sh_entsize;
				};

				Elf64_ShdrLocal shstr {};
				std::memcpy(&shstr, elf.data() + shoff + shentsize * shstrndx,
					    sizeof(shstr));
				const size_t str_off = static_cast<size_t>(shstr.sh_offset);
				const size_t str_sz = static_cast<size_t>(shstr.sh_size);
				if (str_off > elf.size() || str_off + str_sz > elf.size())
					return;
				const char *str_base =
					reinterpret_cast<const char *>(elf.data() + str_off);

				auto safe_section_name = [&](uint32_t name_off) -> std::string_view {
					if (name_off >= str_sz)
						return {};
					const char *p = str_base + name_off;
					const size_t cap = str_sz - name_off;
					const size_t n = strnlen(p, cap);
					if (n == cap)
						return {};
					return std::string_view(p, n);
				};

				for (size_t i = 0; i < shnum; i++) {
					Elf64_ShdrLocal sh {};
					std::memcpy(&sh, elf.data() + shoff + shentsize * i,
						    sizeof(sh));
					auto name = safe_section_name(sh.sh_name);
					if (name.empty())
						continue;
					const bool is_nvinfo =
						name.starts_with(".nv.info.") ||
						name.starts_with(".nv.merc.nv.info.");
					if (!is_nvinfo)
						continue;
					if (name.find(filter_sv) == std::string_view::npos)
						continue;
					out_ids.insert(sh.sh_info);
				}
			};

		// Collect stable per-function IDs to match stripped/anonymized SM120 sections.
		std::unordered_set<uint32_t> fatbin_filter_func_ids;
		if (!filter_sv.empty()) {
			size_t scan_off = 0;
			while (scan_off < file_region.size()) {
				if (file_region.size() - scan_off <
				    sizeof(FatbinFileHeaderLocal))
					break;
				FatbinFileHeaderLocal fh {};
				std::memcpy(&fh, file_region.data() + scan_off, sizeof(fh));
				const size_t fh_size = static_cast<size_t>(fh.header_size);
				const size_t padded =
					static_cast<size_t>(fh.padded_payload_size);
				const size_t payload_size =
					static_cast<size_t>(fh.payload_size);
				if (fh_size < sizeof(FatbinFileHeaderLocal) ||
				    fh_size > (1u << 20))
					break;
				if (padded > file_region.size() - scan_off - fh_size)
					break;
				const auto payload_padded =
					file_region.subspan(scan_off + fh_size, padded);
				const auto payload_comp = payload_padded.subspan(
					0, std::min(padded, payload_size));

				constexpr uint16_t kKindElf = 0x02;
				if (fh.kind == kKindElf && !payload_comp.empty()) {
					const bool likely_compressed =
						force_decompress_elf ||
						(fh.uncompressed_payload >
							 static_cast<uint64_t>(payload_size) &&
						 payload_size != 0 &&
						 !(payload_comp.size() >= 4 &&
						   payload_comp[0] == 0x7f &&
						   payload_comp[1] == 'E' &&
						   payload_comp[2] == 'L' &&
						   payload_comp[3] == 'F'));
					if (!likely_compressed) {
						collect_filter_func_ids_from_elf(
							payload_comp, fatbin_filter_func_ids);
					} else if (fh.uncompressed_payload != 0 &&
						   fh.uncompressed_payload <=
							   kMaxModuleDecompressBound) {
						auto elf = decompress_lz4(
							payload_comp.data(),
							payload_comp.size(),
							static_cast<size_t>(fh.uncompressed_payload));
						if (elf && !elf->empty())
							collect_filter_func_ids_from_elf(
								std::span<const uint8_t>(elf->data(),
											elf->size()),
								fatbin_filter_func_ids);
					}
				}
				scan_off += fh_size + padded;
			}
		}

		// Promote newly discovered func_ids into the process-wide cache so raw-ELF
		// loads (which may happen on other threads) can still be detoured by func_id.
		if (!fatbin_filter_func_ids.empty()) {
			const size_t before =
				impl->sass_detour_filter_state.learned_func_ids.size();
			impl->sass_detour_filter_state.learned_func_ids.insert(
				fatbin_filter_func_ids.begin(), fatbin_filter_func_ids.end());
			const size_t after =
				impl->sass_detour_filter_state.learned_func_ids.size();
			const bool changed = after != before;
			if (changed && env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DEBUG")) {
					SPDLOG_INFO(
						"SASS detour: learned {} new func_id(s) from fatbin (total={})",
						(after - before), after);
			}
			// Persist learned ids so early raw-ELF loads can be detoured on the
			// next run without manual func_id lists.
			if (changed) {
				maybe_store_sass_detour_func_id_cache(
					filter_sv,
					impl->sass_detour_filter_state.learned_func_ids);
			}
		}

		std::unordered_set<uint32_t> merged_filter_func_ids =
			impl->sass_detour_filter_state.learned_func_ids;
		if (extra_filter_func_ids_ptr) {
			merged_filter_func_ids.insert(extra_filter_func_ids_ptr->begin(),
						      extra_filter_func_ids_ptr->end());
		}
		const std::unordered_set<uint32_t> *merged_filter_func_ids_ptr =
			merged_filter_func_ids.empty() ? nullptr : &merged_filter_func_ids;

		bool saw_filter_name_anywhere = assume_contains_filter_name;
		if (!saw_filter_name_anywhere && !filter_sv.empty()) {
			// Best-effort: uncompressed fatbins may carry readable names.
			saw_filter_name_anywhere =
				blob_contains(fatbin_bytes, filter_sv);
		}
				const bool allow_patch_all_fallback =
					env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_PATCH_ALL_FALLBACK") ||
					(sass_identify_closure_enabled() &&
					 !env_truthy_global(
						 "BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE_NO_PATCH_ALL_FALLBACK_DEFAULT"));

			// Pre-scan: establish whether *any* code object in this fatbin carries the
			// filter name. This avoids relying on entry ordering (sm120 may appear
			// before older SMs in the record list).
			if (!saw_filter_name_anywhere && !filter_sv.empty()) {
				size_t scan_off = 0;
				while (scan_off < file_region.size()) {
					if (file_region.size() - scan_off <
					    sizeof(FatbinFileHeaderLocal))
						break;
					FatbinFileHeaderLocal fh {};
					std::memcpy(&fh, file_region.data() + scan_off,
						    sizeof(fh));
					const size_t fh_size = static_cast<size_t>(fh.header_size);
					const size_t padded =
						static_cast<size_t>(fh.padded_payload_size);
					const size_t payload_size =
						static_cast<size_t>(fh.payload_size);
					if (fh_size < sizeof(FatbinFileHeaderLocal) ||
					    fh_size > (1u << 20))
						break;
					if (padded > file_region.size() - scan_off - fh_size)
						break;
					const auto payload_padded =
						file_region.subspan(scan_off + fh_size, padded);
					const auto payload_comp = payload_padded.subspan(
						0, std::min(padded, payload_size));

					constexpr uint16_t kKindElf = 0x02;
					if (fh.kind == kKindElf) {
						const bool likely_compressed =
							force_decompress_elf ||
							(fh.uncompressed_payload >
								 static_cast<uint64_t>(payload_size) &&
							 payload_size != 0 &&
							 !(payload_comp.size() >= 4 &&
							   payload_comp[0] == 0x7f &&
							   payload_comp[1] == 'E' &&
							   payload_comp[2] == 'L' &&
							   payload_comp[3] == 'F'));
						if (!likely_compressed) {
							if (blob_contains(payload_padded, filter_sv)) {
								saw_filter_name_anywhere = true;
								break;
							}
						} else {
							auto elf = decompress_lz4(
								payload_comp.data(),
								payload_comp.size(),
								static_cast<size_t>(fh.uncompressed_payload));
							if (elf &&
							    blob_contains(std::span<const uint8_t>(
										 elf->data(), elf->size()),
									  filter_sv)) {
								saw_filter_name_anywhere = true;
								break;
							}
						}
					}
					scan_off += fh_size + padded;
				}
				if (saw_filter_name_anywhere &&
				    env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DEBUG")) {
					SPDLOG_INFO("SASS detour: {} fatbin pre-scan saw filter name",
						    api_name);
				}
			}

			PatchedFatbin out;
			out.bytes.reserve(fatbin_bytes.size());
		out.bytes.insert(out.bytes.end(), fatbin_bytes.begin(),
				 fatbin_bytes.begin() + header_size);

		size_t off = 0;
		while (off < file_region.size()) {
			if (file_region.size() - off < sizeof(FatbinFileHeaderLocal)) {
				out.reason = "truncated_fatbin_file_header";
				break;
			}
			FatbinFileHeaderLocal fh {};
			std::memcpy(&fh, file_region.data() + off, sizeof(fh));
			const size_t fh_size = static_cast<size_t>(fh.header_size);
			const size_t padded = static_cast<size_t>(fh.padded_payload_size);
			const size_t payload_size = static_cast<size_t>(fh.payload_size);
			if (fh_size < sizeof(FatbinFileHeaderLocal) ||
			    fh_size > (1u << 20)) {
				out.reason = "bad_fatbin_file_header_size";
				break;
			}
			if (padded > file_region.size() - off - fh_size) {
				out.reason = "fatbin_file_payload_out_of_range";
				break;
			}
			const auto rec_span =
				file_region.subspan(off, fh_size + padded);
			const auto payload_padded =
				file_region.subspan(off + fh_size, padded);
			const auto payload_comp =
				payload_padded.subspan(0, std::min(padded, payload_size));

			constexpr uint16_t kKindPtx = 0x01;
			constexpr uint16_t kKindElf = 0x02;

			bool rec_patched = false;
			std::vector<uint8_t> new_payload_bytes;
			FatbinFileHeaderLocal fh_out = fh;

			if (fh.kind == kKindElf) {
				const bool likely_compressed =
					force_decompress_elf ||
					(fh.uncompressed_payload >
						 static_cast<uint64_t>(payload_size) &&
					 payload_size != 0 &&
					 !(payload_comp.size() >= 4 &&
					   payload_comp[0] == 0x7f &&
					   payload_comp[1] == 'E' &&
					   payload_comp[2] == 'L' &&
					   payload_comp[3] == 'F'));

				std::optional<std::vector<uint8_t>> elf_opt;
				if (likely_compressed) {
					elf_opt = decompress_lz4(
						payload_comp.data(), payload_comp.size(),
						static_cast<size_t>(fh.uncompressed_payload));
				} else {
					elf_opt = std::vector<uint8_t>(payload_padded.begin(),
								       payload_padded.end());
				}

				if (elf_opt) {
					auto &elf_bytes = *elf_opt;
					if (!saw_filter_name_anywhere && !filter_sv.empty()) {
						const bool hit = blob_contains(
							std::span<const uint8_t>(elf_bytes.data(),
										 elf_bytes.size()),
							filter_sv);
						if (hit && env_truthy_global(
								   "BPFTIME_CUDA_SASS_DETOUR_DEBUG")) {
							SPDLOG_INFO(
								"SASS detour: {} fatbin saw filter name inside ELF (sm_version_field={})",
								api_name, fh.sm_version);
						}
						saw_filter_name_anywhere = hit;
					}
					// Prevent cross-module `func_id(sh_info)` collisions: only enable
					// learned/cached func_id filtering for this fatbin when we have
					// evidence that the fatbin carries the filter name somewhere.
					//
					// Otherwise, a filter like `flash_fwd_splitkv_kernel` could
						// accidentally match unrelated SM120 `.text.*` sections in other
						// libraries that happen to reuse the same sh_info numbers.
						const uint32_t elf_image_id = sass_elf_image_id32(
							std::span<const uint8_t>(elf_bytes.data(),
										 elf_bytes.size()));
					// If we have an image-scoped func_id mapping for this filter,
					// prefer it even when the SM120 code object no longer carries the
					// kernel name string (strip scenario). Otherwise, fall back to the
					// conservative "fatbin contains filter name" gating.
					auto merged_opt = merged_filter_func_ids_for_image(
						/*include_learned=*/saw_filter_name_anywhere,
						elf_image_id);
					const std::unordered_set<uint32_t> *func_id_filter_ptr =
						merged_opt ? &*merged_opt : extra_filter_func_ids_ptr;
					// When we have a single resolved/image-scoped func_id, prefer
					// patching by func_id only (ignore the text-name substring) to avoid
					// detouring many template instantiations and unrelated kernels.
					std::string_view apply_filter_sv = filter_sv;
					if (sass_identify_closure_enabled() && !filter_sv.empty() &&
					    func_id_filter_ptr != nullptr &&
					    func_id_filter_ptr->size() == 1) {
						apply_filter_sv = std::string_view {};
					}

					auto det =
						sass_detour::apply_elf_text_detours_sm120(
							elf_bytes, apply_filter_sv,
							sample_filter_sv, sampling_cfg,
							func_id_filter_ptr);
					if (det && det->image_id != 0)
						out_sm120_image_id = det->image_id;
						if (det) {
						out.patched_text_sections += det->patched_text_sections;
						out.sampled_text_sections += det->sampled_text_sections;
						if (!det->sampled_kernels.empty()) {
							out.sampled_kernels.insert(
								out.sampled_kernels.end(),
								det->sampled_kernels.begin(),
								det->sampled_kernels.end());
						}
						if (det->patched_text_sections > 0 ||
						    det->sampled_text_sections > 0) {
							rec_patched = true;
								} else if (allow_patch_all_fallback &&
									   !filter_sv.empty() &&
									   det->reason.find(
										   "no .text.* sections matched filter") !=
										   std::string::npos) {
									// Single-run identify closure (1.2):
									// For stripped SM120 code objects inside fatbins, we may have no
									// `.text.<name>`/nvinfo/symtab strings to match at load time.
									//
									// Patch-all fallback can be invasive on real workloads; keep it
									// opt-in to avoid instrumenting large unrelated libraries.
									// (The launch-time on-demand patch-all path remains available.)
									const bool patch_all_on_miss =
										sass_identify_closure_enabled() &&
										env_truthy_global(
											"BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE_PATCH_ALL_ON_MISS") &&
										!env_truthy_global(
											"BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE_NO_PATCH_ALL_ON_MISS");
									const bool allow_on_miss =
										saw_filter_name_anywhere || patch_all_on_miss;
									if (allow_on_miss) {
										if (env_truthy_global(
											    "BPFTIME_CUDA_SASS_DETOUR_DEBUG")) {
											const char *why =
												saw_filter_name_anywhere
													? "saw_filter_name"
													: "on_miss_optin";
											SPDLOG_INFO(
												"SASS detour: {} fatbin SM120 patch-all fallback enabled ({})",
												api_name ? api_name : "api",
												why);
									}
										auto det_all =
											sass_detour::apply_elf_text_detours_sm120(
												elf_bytes, std::string_view{},
												std::string_view{}, sampling_cfg,
												nullptr,
												/*patch_all_is_fallback=*/true);
										if (det_all && det_all->image_id != 0)
											out_sm120_image_id = det_all->image_id;
									if (det_all &&
									    det_all->patched_text_sections > 0) {
										rec_patched = true;
										out.patched_text_sections +=
											det_all->patched_text_sections;
										out.sampled_text_sections +=
											det_all->sampled_text_sections;
										if (!det_all->sampled_kernels.empty())
											out.sampled_kernels.insert(
												out.sampled_kernels.end(),
												det_all->sampled_kernels.begin(),
												det_all->sampled_kernels.end());
									}
								}
							}

								if (!rec_patched && dump_unpatched && dump_dir) {
									maybe_dump_cuda_image_bytes(
										dump_dir, api_name, magic, "elf_unpatched",
									det->reason.empty() ? "no_patch" : det->reason,
									elf_bytes.data(), elf_bytes.size());
							}
							if (rec_patched && dump_patched && dump_dir) {
								maybe_dump_cuda_image_bytes(
									dump_dir, api_name, magic, "elf_patched", "patched",
									elf_bytes.data(), elf_bytes.size());
							}
						}

					if (rec_patched) {
						out.any_patched = true;
						if (likely_compressed) {
							auto comp = compress_lz4(elf_bytes);
							if (!comp) {
								out.reason = "lz4_compress_failed";
								return std::nullopt;
							}
							new_payload_bytes = std::move(*comp);
							fh_out.payload_size =
								(uint32_t)new_payload_bytes.size();
							fh_out.padded_payload_size =
								(uint32_t)align_up(new_payload_bytes.size(), 8);
							fh_out.uncompressed_payload =
								(uint64_t)elf_bytes.size();
						} else {
							new_payload_bytes = std::move(elf_bytes);
							fh_out.payload_size =
								(uint32_t)new_payload_bytes.size();
							fh_out.padded_payload_size =
								(uint32_t)align_up(new_payload_bytes.size(), 8);
							fh_out.uncompressed_payload =
								(uint64_t)new_payload_bytes.size();
						}
					}
				}
			}

			if (rec_patched) {
				// Keep the original record header bytes up to fh.header_size, but
				// update the size fields in the fixed prefix we understand.
				std::vector<uint8_t> rec_hdr_bytes(rec_span.begin(),
								   rec_span.begin() + fh_size);
				if (rec_hdr_bytes.size() >= sizeof(FatbinFileHeaderLocal)) {
					std::memcpy(rec_hdr_bytes.data(), &fh_out,
						    sizeof(FatbinFileHeaderLocal));
				}
				out.bytes.insert(out.bytes.end(), rec_hdr_bytes.begin(),
						 rec_hdr_bytes.end());
				out.bytes.insert(out.bytes.end(), new_payload_bytes.begin(),
						 new_payload_bytes.end());
				const size_t pad = align_up(new_payload_bytes.size(), 8) -
						   new_payload_bytes.size();
				out.bytes.insert(out.bytes.end(), pad, 0);
			} else {
				// Unchanged record: copy verbatim.
				out.bytes.insert(out.bytes.end(), rec_span.begin(),
						 rec_span.end());
			}

			off += fh_size + padded;
		}

		// Fix up files_size in the fatbin header.
		const size_t new_files_size = out.bytes.size() - header_size;
		if (out.bytes.size() >= sizeof(FatbinHeaderLocal)) {
			std::memcpy(out.bytes.data() + offsetof(FatbinHeaderLocal, files_size),
				    &new_files_size, sizeof(uint64_t));
		}
		out.saw_filter_name = saw_filter_name_anywhere;

		if (sampling_cfg && sampling_cfg->enabled &&
		    out.sampled_text_sections > 0) {
			SPDLOG_INFO(
				"SASS sample: instrumented {} .text.* sections (mode={})",
				out.sampled_text_sections, (unsigned)sampling_cfg->mode);
		}

		if (out.bytes.empty())
			out.reason = "empty_fatbin";
		return out;
	};

	std::optional<nv_attach_impl::owned_cuda_image> patched;
	std::optional<nv_attach_impl::owned_cuda_image> base_image;
	bool output_is_detoured = false;

	// Common case: fatbin wrapper (FATBINC_MAGIC).
	if (magic == FATBINC_MAGIC) {
		auto *wrapper = (const __fatBinC_Wrapper_t *)code;
		const uint8_t *data0 =
			reinterpret_cast<const uint8_t *>(wrapper->data);

		auto locate_fatbin_span = [&](const uint8_t *p)
			-> std::optional<std::span<const uint8_t>> {
			if (!p)
				return std::nullopt;
			if (auto sz = parse_fatbin_total_size(p, kMaxFatbinBytes))
				return std::span<const uint8_t>(p, *sz);
			if (auto off =
				    find_fatbin_text_magic_offset(p, kMaxFatPrefixScanBytes)) {
				if (auto sz = parse_fatbin_total_size(
					    p + *off, kMaxFatbinBytes > *off
							     ? (kMaxFatbinBytes - *off)
							     : 0)) {
					return std::span<const uint8_t>(p + *off, *sz);
				}
			}
			return std::nullopt;
		};

		auto primary_span_opt = locate_fatbin_span(data0);
		if (!primary_span_opt) {
			if (dump_unpatched && dump_dir) {
				maybe_dump_cuda_image_bytes(
					dump_dir, api_name, magic, "fatbinc_prefix",
					"cannot_parse_primary_fatbin", data0,
					std::min(kMaxFatPrefixScanBytes, size_t(64u << 10)));
			}
		} else {
			auto primary_patch =
				patch_fatbin_rebuild(*primary_span_opt, false, false);
			if (!primary_patch) {
				if (dump_unpatched && dump_dir) {
					maybe_dump_cuda_image_bytes(
							dump_dir, api_name, magic, "fatbinc_primary",
						"cannot_patch_primary_fatbin",
						primary_span_opt->data(),
						std::min(primary_span_opt->size(),
								 size_t(64u << 10)));
					}
				} else {
					const bool jitlink_ptx_enabled =
						env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_JITLINK_PTX") ||
						(sass_identify_closure_enabled() &&
						 !env_truthy_global(
							 "BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE_NO_JITLINK_PTX_DEFAULT"));
					if (!primary_patch->any_patched && jitlink_ptx_enabled &&
					    !filter_sv.empty() &&
					    impl->original_cu_link_complete != nullptr) {
					// No sm120 patchable code object was found in the fatbin.
					// For modern GPUs (e.g. sm_120) some libraries ship only older
					// SASS (e.g. sm_80) + PTX, and rely on driver JIT.
					//
					// As a minimal coverage boost, JIT-link PTX to a sm_120 cubin
					// ourselves and then apply sm_120 detours on that cubin.
					struct JitInput {
						CUjitInputType type = CU_JIT_INPUT_PTX;
						std::vector<uint8_t> bytes;
						bool has_filter = false;
					};
					std::vector<JitInput> jit_inputs;
					bool ptx_has_filter = false;
					{
						// Extract PTX records (best-effort; supports optional LZ4).
						auto fat = *primary_span_opt;
						if (fat.size() >= sizeof(FatbinHeaderLocal)) {
							FatbinHeaderLocal hdr {};
							std::memcpy(&hdr, fat.data(), sizeof(hdr));
							const size_t header_size =
								static_cast<size_t>(hdr.header_size);
							const size_t files_size =
								static_cast<size_t>(hdr.files_size);
							if (hdr.magic == FATBIN_TEXT_MAGIC && hdr.version == 1 &&
							    header_size <= fat.size() &&
							    files_size <= fat.size() - header_size) {
								auto file_region =
									fat.subspan(header_size, files_size);
								size_t off = 0;
								while (off < file_region.size()) {
									if (file_region.size() - off <
									    sizeof(FatbinFileHeaderLocal))
										break;
									FatbinFileHeaderLocal fh {};
									std::memcpy(&fh, file_region.data() + off,
										    sizeof(fh));
									const size_t fh_size =
										static_cast<size_t>(fh.header_size);
									const size_t padded = static_cast<size_t>(
										fh.padded_payload_size);
									const size_t payload_size =
										static_cast<size_t>(fh.payload_size);
									if (fh_size < sizeof(FatbinFileHeaderLocal) ||
									    fh_size > (1u << 20))
										break;
									if (padded >
									    file_region.size() - off - fh_size)
										break;
									const auto payload_padded =
										file_region.subspan(off + fh_size, padded);
									const auto payload_comp = payload_padded.subspan(
										0, std::min(padded, payload_size));
									constexpr uint16_t kKindPtx = 0x01;
									if (fh.kind == kKindPtx &&
									    !payload_comp.empty()) {
										std::vector<uint8_t> blob;
										const bool likely_compressed =
											(fh.uncompressed_payload >
												 static_cast<uint64_t>(payload_size) &&
											 payload_size != 0);
										if (!likely_compressed) {
											blob.assign(payload_comp.begin(),
												    payload_comp.end());
										} else {
											auto dec = decompress_lz4(
												payload_comp.data(),
												payload_comp.size(),
												static_cast<size_t>(
													fh.uncompressed_payload));
											if (dec)
												blob = std::move(*dec);
										}
										if (!blob.empty()) {
											if (blob.size() > (128u << 20))
												goto next_fatbin_record;

											JitInput in;
											// Heuristic: LLVM bitcode starts with "BC\xc0\xde".
											const bool looks_bitcode =
												(blob.size() >= 4 && blob[0] == 'B' &&
												 blob[1] == 'C' && blob[2] == 0xc0 &&
												 blob[3] == 0xde);
											if (looks_bitcode) {
												in.type = CU_JIT_INPUT_NVVM;
												in.bytes = std::move(blob);
												in.has_filter =
													blob_contains(std::span<const uint8_t>(
															      in.bytes.data(),
															      in.bytes.size()),
														      filter_sv);
											} else {
												// Treat as PTX text. Be lenient about leading
												// whitespace/comments: look for `.version`/`.target`
												// in the first few hundred bytes.
												if (blob.back() != 0)
													blob.push_back(0);
												const char *c =
													reinterpret_cast<const char *>(
														blob.data());
												const size_t c_cap = blob.size();
												size_t ptx_len = strnlen(c, c_cap);
												if (ptx_len < 8 || ptx_len + 1 > c_cap)
													goto next_fatbin_record;

												auto skip_ws = [](std::string_view s) {
													size_t i = 0;
													while (i < s.size()) {
														const char ch = s[i];
														if (ch != ' ' && ch != '\t' &&
														    ch != '\r' && ch != '\n')
															break;
														i++;
													}
													return s.substr(i);
												};
												std::string_view full(c, ptx_len);
												const std::string_view head =
													skip_ws(full.substr(
														0, std::min<size_t>(full.size(), 512)));
												const bool looks_ptx =
													(head.starts_with(".version") ||
													 head.starts_with("//") ||
													 head.find(".version") !=
														 std::string_view::npos ||
													 head.find(".target") !=
														 std::string_view::npos);
												if (!looks_ptx) {
													if (env_truthy_global(
														    "BPFTIME_CUDA_SASS_DETOUR_DEBUG")) {
														SPDLOG_INFO(
															"SASS detour: {} jitlink candidate rejected (kind=PTX, sm_version_field={}, head='{}')",
															api_name,
															(unsigned)fh.sm_version,
															std::string_view(
																head.data(),
																std::min<size_t>(
																	head.size(), 64)));
													}
													goto next_fatbin_record;
												}

												// Some vendors ship PTX with an explicit older `.target`
												// (e.g. `sm_80`). When JIT-linking for newer GPUs (e.g.
												// sm_120), some driver versions fail or behave badly if the
												// PTX `.target` doesn't match the requested JIT target.
												//
												// Best-effort: rewrite the first `.target sm_*` token in the
												// header to `sm_120`.
												{
													std::string_view full0(c, ptx_len);
													const size_t scan_cap =
														std::min<size_t>(full0.size(), 4096);
													const size_t tgt =
														full0.substr(0, scan_cap)
															.find(".target");
													if (tgt != std::string_view::npos) {
														const size_t line_end =
															full0.find('\n', tgt);
														const size_t scan_end =
															(line_end == std::string_view::npos)
																? std::min<size_t>(full0.size(),
																				   tgt + 256)
																: line_end;
														const size_t sm_rel =
															full0.substr(tgt, scan_end - tgt)
																.find("sm_");
														if (sm_rel != std::string_view::npos) {
															const size_t sm_abs =
																tgt + sm_rel;
															size_t tok_end = sm_abs + 3;
															while (tok_end < scan_end) {
																const unsigned char ch =
																	static_cast<unsigned char>(
																		full0[tok_end]);
																if (!(std::isalnum(ch) ||
																      ch == '_' || ch == 'a' ||
																      ch == 'A' || ch == 'f' ||
																      ch == 'F'))
																	break;
																tok_end++;
															}
																const std::string_view old_tok =
																	full0.substr(sm_abs,
																			     tok_end - sm_abs);
																if (old_tok != "sm_120") {
																	// `old_tok` points into `blob`; preserve it
																	// before mutating `blob` (which can reallocate).
																	const std::string old_tok_s(old_tok);
																	static constexpr char kNewTok[] =
																		"sm_120";
																	blob.erase(
																		blob.begin() + sm_abs,
																		blob.begin() + tok_end);
																blob.insert(
																	blob.begin() + sm_abs,
																	kNewTok,
																	kNewTok + sizeof(kNewTok) - 1);
																c = reinterpret_cast<const char *>(
																	blob.data());
																ptx_len = strnlen(c, blob.size());
																	full = std::string_view(c, ptx_len);
																	if (env_truthy_global(
																		    "BPFTIME_CUDA_SASS_DETOUR_DEBUG")) {
																		SPDLOG_INFO(
																			"SASS detour: {} jitlink rewrote PTX .target {} -> sm_120",
																			api_name,
																			old_tok_s);
																	}
																}
															}
														}
													}

													in.type = CU_JIT_INPUT_PTX;
													// Keep only the PTX string itself (+NUL). Passing extra
													// bytes after the first NUL to `cuLinkAddData` can crash
													// some driver versions.
													blob.resize(ptx_len + 1);
													in.bytes = std::move(blob);
												const auto ptx_span =
													std::span<const uint8_t>(
														reinterpret_cast<const uint8_t *>(
															full.data()),
														full.size());
													in.has_filter =
														blob_contains(ptx_span, filter_sv);
												}

												// hetGPU-style step (minimal): when JIT-linking PTX, we can
												// inject instrumentation at the PTX/IR level so register
												// allocation/spills/layout are handled by the toolchain
												// rather than SASS "dead-reg" hunting.
												//
												// This is gated behind env knobs and only applies to PTX
												// inputs that already match the current filter.
												const bool inject_ptx_marker =
													env_truthy_global(
														"BPFTIME_CUDA_SASS_DETOUR_JITLINK_PTX_MARKER");
												const bool inject_ptx_threadmap =
													env_truthy_global(
														"BPFTIME_CUDA_SASS_DETOUR_JITLINK_PTX_THREADMAP");
												if (in.type == CU_JIT_INPUT_PTX && in.has_filter &&
												    (inject_ptx_marker || inject_ptx_threadmap)) {
													auto inject_ptx_instrumentation = [&]() -> bool {
														if (in.bytes.empty())
															return false;
														const char *p =
															reinterpret_cast<const char *>(
																in.bytes.data());
														const size_t cap = in.bytes.size();
														const size_t len = strnlen(p, cap);
														if (len < 8 || len + 1 > cap)
															return false;
														std::string s(p, len);
														const bool have_marker =
															(s.find("__bpftime_ptx_marker") !=
															 std::string::npos);
														const bool have_threadmap =
															(s.find("__bpftime_ptx_out_ptr") !=
															 std::string::npos);
														if ((inject_ptx_marker && have_marker) &&
														    (!inject_ptx_threadmap || have_threadmap))
															return false;
														if ((inject_ptx_threadmap && have_threadmap) &&
														    (!inject_ptx_marker || have_marker))
															return false;

														// 1) Add a global marker symbol once.
														{
															std::string decl;
															if (inject_ptx_marker && !have_marker) {
																decl +=
																	"\n// bpftime jitlink PTX marker\n"
																	".visible .global .align 4 .u32 __bpftime_ptx_marker;\n";
															}
															if (inject_ptx_threadmap && !have_threadmap) {
																// Bind-at-launch interface:
																// - __bpftime_ptx_out_ptr points to bpftime's sampler buffer
																//   (data region, u32 slots)
																// - __bpftime_ptx_out_cap is capacity in u32 slots
																// - __bpftime_ptx_out_gate controls density (bitmask)
																//   bit0: lane0-only, bit1: warp0-only, bit2: CTA0-only
																decl +=
																	"\n// bpftime jitlink PTX threadmap (bpftime sampler-bound)\n"
																	".visible .global .align 8 .u64 __bpftime_ptx_out_ptr;\n"
																	".visible .global .align 4 .u32 __bpftime_ptx_out_cap;\n"
																	".visible .global .align 4 .u32 __bpftime_ptx_out_gate;\n";
															}
															auto insert_after_directive =
																[&](const char *needle)
																-> std::optional<size_t> {
																const size_t scan_cap =
																	std::min<size_t>(s.size(), 8192);
																const size_t pos =
																	s.substr(0, scan_cap)
																		.find(needle);
																if (pos ==
																    std::string::npos)
																	return std::nullopt;
																const size_t end =
																	s.find('\n', pos);
																if (end ==
																    std::string::npos)
																	return std::nullopt;
																return end + 1;
															};
															std::optional<size_t> ins =
																insert_after_directive(
																	".address_size");
															if (!ins)
																ins = insert_after_directive(
																	".target");
															if (ins && !decl.empty())
																s.insert(*ins, decl);
														}

														// 2) For each matching `.entry`, insert a minimal store
														//    before the first instruction (after `.reg`/directives).
														{
															const std::string_view filt =
																filter_sv;
															const std::string_view key =
																".entry";
															size_t pos = 0;
															while (true) {
																pos = s.find(
																	std::string(key),
																	pos);
																if (pos ==
																    std::string::npos)
																	break;
																const size_t name_start =
																	pos + key.size();
																size_t i = name_start;
																while (i < s.size() &&
																       (s[i] == ' ' ||
																	s[i] == '\t'))
																	i++;
																if (i >= s.size())
																	break;
																size_t j = i;
																while (j < s.size()) {
																	const char ch = s[j];
																	if (ch == '(' ||
																	    ch == ' ' ||
																	    ch == '\t' ||
																	    ch == '\r' ||
																	    ch == '\n')
																		break;
																	j++;
																}
																if (j <= i) {
																	pos = pos +
																	      key.size();
																	continue;
																}
																const std::string_view entry_name =
																	std::string_view(
																		s.data() + i,
																		j - i);
																if (!filt.empty() &&
																    entry_name.find(
																	    filt) ==
																	    std::string_view::
																		npos) {
																	pos = j;
																	continue;
																}
																const size_t brace =
																	s.find('{', j);
																if (brace ==
																    std::string::npos) {
																	pos = j;
																	continue;
																}
																size_t scan = brace + 1;
																while (scan < s.size()) {
																	size_t line_end =
																		s.find('\n',
																		       scan);
																	if (line_end ==
																	    std::string::npos)
																		line_end =
																			s.size();
																	std::string_view line(
																		s.data() + scan,
																		line_end - scan);
																	auto trim_left =
																		[](std::string_view x) {
																		size_t k =
																			0;
																		while (k <
																		       x.size()) {
																			const char ch =
																				x[k];
																			if (ch !=
																			    ' ' &&
																			    ch !=
																			    '\t' &&
																			    ch !=
																			    '\r')
																				break;
																			k++;
																		}
																		return x.substr(
																			k);
																	};
																	const auto t =
																		trim_left(
																			line);
																	const bool is_empty =
																		t.empty() ||
																		t == "}";
																	const bool is_comment =
																		t.starts_with(
																			"//") ||
																		t.starts_with(
																			"/*");
																	const bool is_directive =
																		(!t.empty() &&
																		 t[0] == '.');
																	if (!is_empty &&
																	    !is_comment &&
																	    !is_directive) {
																		std::string snippet;
																		if (inject_ptx_marker && !have_marker) {
																			snippet +=
																				"\n\t// bpftime jitlink marker (PTX-level)\n"
																				"\t.reg .b32 %bpftime_r<1>;\n"
																				"\tmov.u32 %bpftime_r0, 1;\n"
																				"\tst.global.u32 [__bpftime_ptx_marker], %bpftime_r0;\n";
																		}
																		if (inject_ptx_threadmap && !have_threadmap) {
																			// Thread-map integration: write SMID into bpftime sampler buffer
																			// (u32 slots), with runtime gate/cap configured by bpftime at launch.
																			snippet +=
																				"\n\t// bpftime jitlink threadmap (PTX-level, sampler-bound)\n"
																				"\t.reg .pred %bpftime_p<6>;\n"
																				"\t.reg .b32 %bpftime_t<6>;\n"
																				"\t.reg .b64 %bpftime_rd<4>;\n"
																				"\tld.global.u64 %bpftime_rd0, [__bpftime_ptx_out_ptr];\n"
																				"\tsetp.eq.u64 %bpftime_p0, %bpftime_rd0, 0;\n"
																				"\t@%bpftime_p0 bra $__bpftime_tm_skip;\n"
																				"\tld.global.u32 %bpftime_t0, [__bpftime_ptx_out_cap];\n"
																				"\tld.global.u32 %bpftime_t1, [__bpftime_ptx_out_gate];\n"
																				"\t// gate bit2: CTA0-only (clamp)\n"
																				"\tand.b32 %bpftime_t2, %bpftime_t1, 4;\n"
																				"\tsetp.ne.u32 %bpftime_p1, %bpftime_t2, 0;\n"
																				"\t@%bpftime_p1 bra $__bpftime_tm_check_cta;\n"
																				"\tbra $__bpftime_tm_after_cta;\n"
																				"$__bpftime_tm_check_cta:\n"
																				"\tmov.u32 %bpftime_t3, %ctaid.x;\n"
																				"\tsetp.ne.u32 %bpftime_p2, %bpftime_t3, 0;\n"
																				"\t@%bpftime_p2 bra $__bpftime_tm_skip;\n"
																				"\tmov.u32 %bpftime_t3, %ctaid.y;\n"
																				"\tsetp.ne.u32 %bpftime_p2, %bpftime_t3, 0;\n"
																				"\t@%bpftime_p2 bra $__bpftime_tm_skip;\n"
																				"\tmov.u32 %bpftime_t3, %ctaid.z;\n"
																				"\tsetp.ne.u32 %bpftime_p2, %bpftime_t3, 0;\n"
																				"\t@%bpftime_p2 bra $__bpftime_tm_skip;\n"
																				"$__bpftime_tm_after_cta:\n"
																				"\tmov.u32 %bpftime_t2, %tid.x;\n"
																				"\tsetp.ge.u32 %bpftime_p3, %bpftime_t2, %bpftime_t0;\n"
																				"\t@%bpftime_p3 bra $__bpftime_tm_skip;\n"
																				"\t// gate bit0: lane0-only\n"
																				"\tand.b32 %bpftime_t3, %bpftime_t1, 1;\n"
																				"\tsetp.ne.u32 %bpftime_p4, %bpftime_t3, 0;\n"
																				"\t@%bpftime_p4 bra $__bpftime_tm_check_lane;\n"
																				"\tbra $__bpftime_tm_after_lane;\n"
																				"$__bpftime_tm_check_lane:\n"
																				"\tmov.u32 %bpftime_t4, %laneid;\n"
																				"\tsetp.ne.u32 %bpftime_p5, %bpftime_t4, 0;\n"
																				"\t@%bpftime_p5 bra $__bpftime_tm_skip;\n"
																				"$__bpftime_tm_after_lane:\n"
																				"\t// gate bit1: warp0-only\n"
																				"\tand.b32 %bpftime_t3, %bpftime_t1, 2;\n"
																				"\tsetp.ne.u32 %bpftime_p4, %bpftime_t3, 0;\n"
																				"\t@%bpftime_p4 bra $__bpftime_tm_check_warp;\n"
																				"\tbra $__bpftime_tm_after_warp;\n"
																				"$__bpftime_tm_check_warp:\n"
																				"\tshr.u32 %bpftime_t4, %bpftime_t2, 5;\n"
																				"\tsetp.ne.u32 %bpftime_p5, %bpftime_t4, 0;\n"
																				"\t@%bpftime_p5 bra $__bpftime_tm_skip;\n"
																				"$__bpftime_tm_after_warp:\n"
																				"\tmul.wide.u32 %bpftime_rd1, %bpftime_t2, 4;\n"
																				"\tadd.u64 %bpftime_rd2, %bpftime_rd0, %bpftime_rd1;\n"
																				"\tcvta.to.global.u64 %bpftime_rd3, %bpftime_rd2;\n"
																				"\tmov.u32 %bpftime_t5, %smid;\n"
																				"\tst.global.u32 [%bpftime_rd3], %bpftime_t5;\n"
																				"$__bpftime_tm_skip:\n";
																		}
																		if (!snippet.empty())
																			s.insert(scan,
																				 snippet);
																		break;
																	}
																	if (line_end >=
																	    s.size())
																		break;
																	scan = line_end +
																	       1;
																}
																pos = brace + 1;
															}
														}

														// Re-materialize NUL-terminated bytes.
														in.bytes.assign(s.begin(), s.end());
														in.bytes.push_back(0);
														return true;
													};
													const bool injected = inject_ptx_instrumentation();
													if (injected &&
													    env_truthy_global(
														    "BPFTIME_CUDA_SASS_DETOUR_DEBUG")) {
														SPDLOG_INFO(
															"SASS detour: {} jitlink injected PTX marker/threadmap for filter='{}' (bytes={})",
															api_name, filter_sv,
															in.bytes.size());
													}
												}

												if (in.has_filter && !ptx_has_filter)
													ptx_has_filter = true;
												if (in.has_filter)
													jit_inputs.emplace_back(std::move(in));
											}
									}
								next_fatbin_record:
									off += fh_size + padded;
								}
							}
						}
					}

						if (!jit_inputs.empty() && ptx_has_filter) {
							// Avoid holding the cache/owned-image lock across CUDA JIT.
							guard.unlock();

							const size_t max_ptx_bytes = []() -> size_t {
								// Large PTX payloads can trigger driver instability in the
								// cuLinkAddData path on some setups. Keep a conservative
								// default and allow override.
								if (auto v = env_u32_global(
									    "BPFTIME_CUDA_SASS_DETOUR_JITLINK_MAX_PTX_BYTES"))
									return size_t(*v);
								return size_t(20u) << 20; // 20 MiB
							}();
							const size_t max_inputs = []() -> size_t {
								if (auto v = env_u32_global(
									    "BPFTIME_CUDA_SASS_DETOUR_JITLINK_MAX_INPUTS"))
									return size_t(std::min<uint32_t>(*v, 8u));
								return size_t(4);
							}();

							CUlinkState state = nullptr;
							std::array<char, 8192> info_log {};
							std::array<char, 8192> err_log {};
							unsigned int info_log_sz = (unsigned int)info_log.size();
							unsigned int err_log_sz = (unsigned int)err_log.size();
						unsigned int log_verbose = 1;
						unsigned int jit_target = (unsigned)CU_TARGET_COMPUTE_120;
						std::array<CUjit_option, 6> opts = {
							// Use an explicit target to avoid relying on the current
							// context state during `cuLibraryLoadData` (some drivers can
							// crash when using TARGET_FROM_CUCONTEXT in this hook path).
							CU_JIT_TARGET,
							CU_JIT_INFO_LOG_BUFFER,
							CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
							CU_JIT_ERROR_LOG_BUFFER,
							CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
							CU_JIT_LOG_VERBOSE,
						};
						// NOTE: CUDA driver JIT APIs take option values "cast to void*"
						// (i.e. numeric options are passed by value via pointer-cast), while
						// pointer options (log buffers) are passed as pointers.
						std::array<void *, 6> opt_vals = {
							(void *)(uintptr_t)jit_target,
							info_log.data(),
							(void *)(uintptr_t)info_log_sz,
							err_log.data(),
							(void *)(uintptr_t)err_log_sz,
							(void *)(uintptr_t)log_verbose,
							};

							CUresult link_res =
								cuLinkCreate((unsigned int)opts.size(), opts.data(),
									     opt_vals.data(), &state);
						if (env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DEBUG")) {
							SPDLOG_INFO(
								"SASS detour: {} jitlink inputs={} (filtered) ptx_has_filter={}",
								api_name, jit_inputs.size(), ptx_has_filter);
						}
							if (link_res == CUDA_SUCCESS && state != nullptr) {
								// Prefer the smallest PTX input(s) that contain the filter
								// string; this reduces compilation time and avoids feeding
								// extremely large PTX blobs into the driver JIT path.
								std::vector<size_t> candidates;
								candidates.reserve(jit_inputs.size());
								for (size_t i = 0; i < jit_inputs.size(); i++) {
									const auto &in = jit_inputs[i];
									if (!in.has_filter || in.bytes.empty())
										continue;
									if (in.bytes.size() > max_ptx_bytes)
										continue;
									candidates.push_back(i);
								}
								std::sort(candidates.begin(), candidates.end(),
									  [&](size_t a, size_t b) {
										  return jit_inputs[a].bytes.size() <
											 jit_inputs[b].bytes.size();
									  });

								size_t added = 0;
								for (size_t ci = 0; ci < candidates.size(); ci++) {
									const size_t i = candidates[ci];
									auto &in = jit_inputs[i];
									char name[32] = {};
									std::snprintf(name, sizeof(name), "ptx_%zu", i);
									if (env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DEBUG")) {
										const size_t n = std::min<size_t>(in.bytes.size(), 96);
										std::string head;
										head.reserve(n);
									for (size_t j = 0; j < n; j++) {
										const unsigned char ch =
											static_cast<unsigned char>(in.bytes[j]);
										if (ch >= 0x20 && ch <= 0x7e)
											head.push_back((char)ch);
										else if (ch == '\n' || ch == '\r' ||
											 ch == '\t')
											head.push_back(' ');
										else
											head.push_back('.');
									}
									SPDLOG_INFO(
										"SASS detour: {} jitlink add input idx={} type={} bytes={} head='{}'",
										api_name, i, (unsigned)in.type,
										in.bytes.size(), head);
								}
									link_res = cuLinkAddData(
										state, in.type,
										(void *)in.bytes.data(),
										in.bytes.size(),
										name, 0, nullptr, nullptr);
									if (link_res != CUDA_SUCCESS)
										break;
									added++;
									if (added >= max_inputs)
										break;
								}
								if (added == 0)
									link_res = CUDA_ERROR_INVALID_VALUE;
							if (env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DEBUG") &&
							    link_res != CUDA_SUCCESS) {
								SPDLOG_INFO(
									"SASS detour: {} jitlink add failed link_res={} added={} err_log='{}' info_log='{}'",
									api_name, (int)link_res, added,
									std::string_view(err_log.data()),
									std::string_view(info_log.data()));
							}
						}

						void *cubin_out = nullptr;
						size_t cubin_sz = 0;
						if (link_res == CUDA_SUCCESS && state != nullptr) {
							using complete_fn_t =
								CUresult (*)(CUlinkState, void **, size_t *);
							auto complete =
								reinterpret_cast<complete_fn_t>(
									impl->original_cu_link_complete);
							if (complete != nullptr) {
								link_res = complete(state, &cubin_out, &cubin_sz);
							} else {
								link_res = CUDA_ERROR_UNKNOWN;
							}
						}
						if (env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DEBUG") &&
						    link_res != CUDA_SUCCESS) {
							SPDLOG_INFO(
								"SASS detour: {} jitlink complete failed link_res={} cubin_sz={} err_log='{}' info_log='{}'",
								api_name, (int)link_res, cubin_sz,
								std::string_view(err_log.data()),
								std::string_view(info_log.data()));
						}

						std::vector<uint8_t> cubin_bytes;
						if (link_res == CUDA_SUCCESS && cubin_out != nullptr &&
						    cubin_sz >= 64) {
							cubin_bytes.assign(
								reinterpret_cast<const uint8_t *>(cubin_out),
								reinterpret_cast<const uint8_t *>(cubin_out) +
									cubin_sz);
						}

						if (state != nullptr)
							cuLinkDestroy(state);

						guard.lock();

						if (!cubin_bytes.empty()) {
							auto merged_opt =
								merged_filter_func_ids_for_image(
									/*include_learned=*/true);
							const std::unordered_set<uint32_t> *merged_ptr =
								merged_opt ? &*merged_opt : extra_filter_func_ids_ptr;
							// When we have a single resolved/cached func_id for this filter,
							// prefer patching by func_id only (ignore text-name substring) to
							// avoid detouring many template instantiations (flashattention).
							std::string_view apply_filter_sv = filter_sv;
							if (sass_identify_closure_enabled() && !filter_sv.empty() &&
							    merged_ptr != nullptr && merged_ptr->size() == 1) {
								apply_filter_sv = std::string_view {};
							}
							std::vector<uint8_t> base_cubin_bytes;
							base_cubin_bytes.assign(cubin_bytes.begin(),
									       cubin_bytes.end());
							auto det = sass_detour::apply_elf_text_detours_sm120(
								cubin_bytes, apply_filter_sv, sample_filter_sv,
								sampling_cfg, merged_ptr);
							if (det && det->image_id != 0)
								out_sm120_image_id = det->image_id;
							const bool changed =
								(det && (det->patched_text_sections > 0 ||
									 det->sampled_text_sections > 0));
							if (changed && det) {
								if (!det->sampled_kernels.empty())
									impl->record_sass_sampled_kernels(
										det->sampled_kernels);
								if (sampling_cfg && sampling_cfg->enabled &&
								    det->sampled_text_sections > 0) {
									SPDLOG_INFO(
										"SASS sample: instrumented {} .text.* sections (mode={})",
										det->sampled_text_sections,
										(unsigned)sampling_cfg->mode);
								}
								nv_attach_impl::owned_cuda_image base_img;
								base_img.size = base_cubin_bytes.size();
								base_img.data =
									std::make_unique<uint8_t[]>(base_img.size);
								std::memcpy(base_img.data.get(),
									    base_cubin_bytes.data(),
									    base_img.size);
								base_image = std::move(base_img);
								output_is_detoured = true;
								if (env_truthy_global(
									    "BPFTIME_CUDA_SASS_DETOUR_DEBUG")) {
									SPDLOG_INFO(
										"SASS detour: {} jitlink PTX->cubin bytes={} patched_text_sections={}",
										api_name, cubin_bytes.size(),
										det->patched_text_sections);
								}
							} else if (env_truthy_global(
									   "BPFTIME_CUDA_SASS_DETOUR_DEBUG")) {
								const std::string why =
									(det && !det->reason.empty())
										? det->reason
										: "no_patch";
								SPDLOG_INFO(
									"SASS detour: {} jitlink PTX->cubin no patch (reason='{}')",
									api_name, why);
							}

							// JITlink path already produced a raw cubin image; pass it as-is
							// (detoured or not).
							nv_attach_impl::owned_cuda_image img;
							img.size = cubin_bytes.size();
							img.data =
								std::make_unique<uint8_t[]>(img.size);
							std::memcpy(img.data.get(),
								    cubin_bytes.data(), img.size);
							if (dump_patched && dump_dir) {
								maybe_dump_cuda_image_bytes(
									dump_dir, api_name, 0x464c457f /*ELF*/,
									output_is_detoured
										? "jitlink_elf_patched"
										: "jitlink_elf_unpatched",
									output_is_detoured ? "patched" : "no_patch",
									img.data.get(), img.size);
							}
							patched = std::move(img);
						} else if (env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DEBUG")) {
							SPDLOG_INFO(
								"SASS detour: {} jitlink PTX->cubin failed (err_log='{}')",
								api_name, std::string_view(err_log.data()));
						}
					}
				}

				if (patched) {
					// JITlink path already produced a raw cubin image; pass it as-is.
					goto done_detour_patch;
				}

				if (primary_patch->saw_filter_name) {
					if (env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DEBUG")) {
						SPDLOG_INFO(
							"SASS detour: {} fatbinc contains filter name; enabling raw-ELF fallback budget",
							api_name);
					}
					impl->sass_detour_filter_state.recent_fatbinc_filter_budget =
						64;
				}

				const bool is_wrapper_v2 = (wrapper->version == 2);
				std::vector<PatchedFatbin> extra_patches;
				bool extra_list_complete = true;
				if (is_wrapper_v2) {
						const void *const *list =
							reinterpret_cast<const void *const *>(
								wrapper->filename_or_fatbins);
						if (list) {
							for (size_t i = 0; i < kMaxFatbinsV2; i++) {
								const void *entry = list[i];
								if (!entry)
									break;
								auto span_opt = locate_fatbin_span(
									reinterpret_cast<const uint8_t *>(entry));
								if (!span_opt) {
									extra_list_complete = false;
									break;
								}
								auto patch =
									patch_fatbin_rebuild(
										*span_opt, true,
										primary_patch->saw_filter_name);
								if (!patch) {
									extra_list_complete = false;
									break;
								}
								extra_patches.push_back(std::move(*patch));
							}
						} else {
							extra_list_complete = false;
						}
						}

						// Cross-fatbin propagation (strip scenario):
						// A fatbinc wrapper can reference multiple fatbins. In practice, some
						// kernels keep readable `.nv.info.<kernel>` names only in one fatbin
						// (e.g., older SM), while the SM120 fatbin is fully stripped and
						// can't "prove" that it contains the filter name.
						//
						// For stable kernel-name detouring, allow a second pass: if any
						// referenced fatbin saw the filter name, re-run the primary fatbin
						// patch with `assume_contains_filter_name=true` so it can apply the
						// learned func_id set to stripped SM120 code objects.
						const bool saw_in_any_fatbin =
							primary_patch->saw_filter_name ||
							std::any_of(extra_patches.begin(), extra_patches.end(),
								    [](const PatchedFatbin &p) {
									    return p.saw_filter_name;
								    });
						if (!primary_patch->saw_filter_name && saw_in_any_fatbin) {
							auto retry =
								patch_fatbin_rebuild(*primary_span_opt,
										     /*force_decompress_elf=*/true,
										     /*assume_contains_filter_name=*/true);
							if (retry)
								primary_patch = std::move(retry);
						}

						if (!primary_patch->saw_filter_name) {
							for (const auto &p : extra_patches) {
								if (p.saw_filter_name) {
									if (env_truthy_global(
										    "BPFTIME_CUDA_SASS_DETOUR_DEBUG")) {
										SPDLOG_INFO(
											"SASS detour: {} fatbinc(extra) contains filter name; enabling raw-ELF fallback budget",
											api_name);
									}
									impl->sass_detour_filter_state
										.recent_fatbinc_filter_budget = 64;
									break;
								}
							}
						}

						const bool can_rewrite_extra_list =
							is_wrapper_v2 && extra_list_complete;
					const bool any_extra_changes =
						can_rewrite_extra_list &&
						std::any_of(extra_patches.begin(),
							    extra_patches.end(),
							    [](const PatchedFatbin &p) {
								    return p.any_patched;
							    });
						const bool any_changes =
							primary_patch->any_patched || any_extra_changes;

						// When `code_size==0` (common for cuLibraryLoadData), callers can't
						// later re-read this image for on-demand patching unless we preserve
						// a concrete (owned) byte span. Even if we don't detour at load time,
						// keep a best-effort owned copy of the fatbinc wrapper + referenced
						// fatbins so identify-closure can patch on-demand.
						//
						// This is intentionally limited to the cases where we can build a
						// self-contained wrapper image (wrapper v1, or v2 with a fully
						// parsable fatbin pointer list).
						if (!any_changes && code_size == 0) {
							const bool can_copy_wrapper_v2 =
								(wrapper->version == 2) && extra_list_complete;
							if (wrapper->version != 2 || can_copy_wrapper_v2) {
								const size_t wrapper_size =
									sizeof(__fatBinC_Wrapper_t);
								size_t cur = align_up(wrapper_size, 8);

								struct CopyFatbin {
									std::vector<uint8_t> bytes;
								};
								std::vector<CopyFatbin> fatbins_to_copy;
								fatbins_to_copy.reserve(
									1 + (can_copy_wrapper_v2 ? extra_patches.size() : 0));
								// Primary fatbin.
								{
									CopyFatbin fb;
									fb.bytes.assign(primary_span_opt->begin(),
											primary_span_opt->end());
									fatbins_to_copy.push_back(std::move(fb));
								}
								// Extra fatbins (wrapper v2).
								if (can_copy_wrapper_v2) {
									const void *const *list =
										reinterpret_cast<const void *const *>(
											wrapper->filename_or_fatbins);
									bool ok = (list != nullptr);
									for (size_t i = 0; i < extra_patches.size(); i++) {
										const void *entry = ok ? list[i] : nullptr;
										if (!entry) {
											ok = false;
											break;
										}
										auto span_opt = locate_fatbin_span(
											reinterpret_cast<const uint8_t *>(entry));
										if (!span_opt) {
											ok = false;
											break;
										}
										CopyFatbin fb;
										fb.bytes.assign(span_opt->begin(),
												span_opt->end());
										fatbins_to_copy.push_back(std::move(fb));
									}
									if (!ok ||
									    fatbins_to_copy.size() !=
										    1 + extra_patches.size()) {
										fatbins_to_copy.clear();
									}
								}
								if (!fatbins_to_copy.empty()) {
									std::vector<size_t> fatbin_offsets;
									fatbin_offsets.reserve(fatbins_to_copy.size());
									for (const auto &fb : fatbins_to_copy) {
										cur = align_up(cur, 8);
										fatbin_offsets.push_back(cur);
										cur += fb.bytes.size();
									}

									size_t list_off = 0;
									if (can_copy_wrapper_v2 && fatbins_to_copy.size() > 1) {
										cur = align_up(cur, alignof(void *));
										list_off = cur;
										const size_t n_ptrs =
											(fatbins_to_copy.size() - 1) + 1;
										cur += n_ptrs * sizeof(void *);
									}

									nv_attach_impl::owned_cuda_image img;
									img.size = cur;
									img.data =
										std::make_unique<uint8_t[]>(img.size);
									std::memset(img.data.get(), 0, img.size);
									auto *wcopy =
										reinterpret_cast<__fatBinC_Wrapper_t *>(
											img.data.get());
									*wcopy = *wrapper;

									for (size_t i = 0; i < fatbins_to_copy.size(); i++) {
										std::memcpy(img.data.get() + fatbin_offsets[i],
											    fatbins_to_copy[i].bytes.data(),
											    fatbins_to_copy[i].bytes.size());
									}

									wcopy->data =
										reinterpret_cast<const unsigned long long *>(
											img.data.get() +
											fatbin_offsets[0]);
									if (list_off != 0) {
										void **out_list =
											reinterpret_cast<void **>(
												img.data.get() + list_off);
										for (size_t i = 1; i < fatbins_to_copy.size(); i++)
											out_list[i - 1] =
												img.data.get() + fatbin_offsets[i];
										out_list[fatbins_to_copy.size() - 1] =
											nullptr;
										wcopy->filename_or_fatbins =
											reinterpret_cast<void *>(out_list);
									}

									patched = std::move(img);
									output_is_detoured = false;
									if (env_truthy_global(
										    "BPFTIME_CUDA_SASS_DETOUR_DEBUG")) {
										SPDLOG_INFO(
											"SASS detour: {} fatbinc preserved owned image bytes={} (no_patch)",
											api_name ? api_name : "api",
											patched->size);
									}
								}
							}
						}

						if (any_changes) {
							if (!primary_patch->sampled_kernels.empty())
								impl->record_sass_sampled_kernels(
									primary_patch->sampled_kernels);
						if (any_extra_changes) {
							for (const auto &p : extra_patches) {
								if (!p.sampled_kernels.empty())
									impl->record_sass_sampled_kernels(
										p.sampled_kernels);
							}
						}

						const size_t wrapper_size =
							sizeof(__fatBinC_Wrapper_t);
						size_t cur = align_up(wrapper_size, 8);

						std::vector<std::vector<uint8_t>> fatbins_to_copy;
						fatbins_to_copy.reserve(1 + extra_patches.size());
						fatbins_to_copy.push_back(std::move(primary_patch->bytes));
						if (any_extra_changes) {
							for (auto &p : extra_patches)
								fatbins_to_copy.push_back(
									std::move(p.bytes));
						}

						std::vector<size_t> fatbin_offsets;
						fatbin_offsets.reserve(fatbins_to_copy.size());
					for (const auto &fb : fatbins_to_copy) {
						cur = align_up(cur, 8);
						fatbin_offsets.push_back(cur);
						cur += fb.size();
					}

						size_t list_off = 0;
						if (any_extra_changes) {
							cur = align_up(cur, alignof(void *));
							list_off = cur;
							const size_t n_ptrs =
								(fatbins_to_copy.size() - 1) + 1;
							cur += n_ptrs * sizeof(void *);
						}

					nv_attach_impl::owned_cuda_image img;
					img.size = cur;
					img.data = std::make_unique<uint8_t[]>(img.size);
					std::memset(img.data.get(), 0, img.size);
					auto *wcopy =
						reinterpret_cast<__fatBinC_Wrapper_t *>(
							img.data.get());
					*wcopy = *wrapper;

					for (size_t i = 0; i < fatbins_to_copy.size(); i++) {
						std::memcpy(img.data.get() + fatbin_offsets[i],
							    fatbins_to_copy[i].data(),
							    fatbins_to_copy[i].size());
					}

					wcopy->data =
						reinterpret_cast<const unsigned long long *>(
							img.data.get() + fatbin_offsets[0]);

						if (any_extra_changes) {
							void **out_list =
								reinterpret_cast<void **>(img.data.get() +
											  list_off);
							for (size_t i = 1; i < fatbins_to_copy.size(); i++) {
								out_list[i - 1] = img.data.get() + fatbin_offsets[i];
						}
						out_list[fatbins_to_copy.size() - 1] = nullptr;
						wcopy->filename_or_fatbins =
							reinterpret_cast<void *>(out_list);
					}

					patched = std::move(img);
					output_is_detoured = true;
				} else if (dump_unpatched && dump_dir) {
					maybe_dump_cuda_image_bytes(
						dump_dir, api_name, magic, "fatbinc_unpatched",
						"no_matching_text_sections",
						primary_span_opt->data(),
						std::min(primary_span_opt->size(),
							 size_t(64u << 10)));
				}
			}
		}
	} else if (magic == FATBIN_TEXT_MAGIC) {
		// Sometimes module/library load receives the payload directly.
		const uint8_t *data = reinterpret_cast<const uint8_t *>(code);
		size_t cap = kMaxFatbinBytes;
		if (code_size != 0)
			cap = std::min(cap, code_size);
		if (auto fatbin_size = parse_fatbin_total_size(data, cap)) {
			// Even if no patch happens, preserve a best-effort size so callers
			// (e.g., cuLibraryLoadData -> cuLibraryGetModule -> cuLaunchKernel)
			// can later re-read/patch this image on-demand.
			if (out_size != nullptr && *out_size == 0)
				*out_size = *fatbin_size;
			std::span<const uint8_t> fatbin(data, *fatbin_size);
			auto fat_patch = patch_fatbin_rebuild(fatbin, false, false);
			if (fat_patch && fat_patch->any_patched) {
				if (!fat_patch->sampled_kernels.empty())
					impl->record_sass_sampled_kernels(
						fat_patch->sampled_kernels);
				nv_attach_impl::owned_cuda_image img;
				img.size = fat_patch->bytes.size();
				img.data = std::make_unique<uint8_t[]>(img.size);
				std::memcpy(img.data.get(), fat_patch->bytes.data(),
					    fat_patch->bytes.size());
				patched = std::move(img);
				output_is_detoured = true;
			}
		}
		} else if (magic == 0x464c457f /* ELF */) {
				// Raw CUBIN passed directly (common for JITed code).
					const uint8_t *data = reinterpret_cast<const uint8_t *>(code);
					// When `code_size==0` (e.g., cuModuleLoadData/cuLibraryLoadData), we
					// must infer the ELF extent. Be conservative here: an invalid size
					// inference can read far beyond the actual buffer and hang or OOM.
					constexpr size_t kMaxElfProbeKnown = 256u << 20;
					constexpr size_t kMaxElfProbeUnknown = 16u << 20;
					const size_t elf_cap =
						(code_size != 0)
							? std::min(code_size, kMaxElfProbeKnown)
							: kMaxElfProbeUnknown;
			const bool allow_patch_all_fallback =
				env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_PATCH_ALL_FALLBACK") ||
				(sass_identify_closure_enabled() &&
				 !env_truthy_global(
					 "BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE_NO_PATCH_ALL_FALLBACK_DEFAULT"));
				auto infer_elf_size = [&](size_t cap) -> size_t {
					if (cap < 64)
						return 0;
					struct Elf64_EhdrLocal {
						uint8_t e_ident[16];
						uint16_t e_type;
						uint16_t e_machine;
						uint32_t e_version;
						uint64_t e_entry;
						uint64_t e_phoff;
						uint64_t e_shoff;
						uint32_t e_flags;
						uint16_t e_ehsize;
						uint16_t e_phentsize;
						uint16_t e_phnum;
						uint16_t e_shentsize;
						uint16_t e_shnum;
						uint16_t e_shstrndx;
					} eh {};
					std::memcpy(&eh, data, sizeof(eh));
					if (eh.e_ident[0] != 0x7f || eh.e_ident[1] != 'E' ||
					    eh.e_ident[2] != 'L' || eh.e_ident[3] != 'F')
						return 0;
					if (eh.e_ident[4] != 2 /* ELFCLASS64 */ ||
					    eh.e_ident[5] != 1 /* little */)
						return 0;

					size_t max_end = sizeof(eh);

					// Include program header table and PT_LOAD segment extents.
					if (eh.e_phoff != 0 && eh.e_phentsize != 0 && eh.e_phnum != 0) {
						const size_t phoff = static_cast<size_t>(eh.e_phoff);
						const size_t phentsize = static_cast<size_t>(eh.e_phentsize);
						const size_t phnum = static_cast<size_t>(eh.e_phnum);
						if (phoff < cap && phoff + phentsize * phnum <= cap) {
							max_end = std::max(max_end, phoff + phentsize * phnum);
							struct Elf64_PhdrLocal {
								uint32_t p_type;
								uint32_t p_flags;
								uint64_t p_offset;
								uint64_t p_vaddr;
								uint64_t p_paddr;
								uint64_t p_filesz;
								uint64_t p_memsz;
								uint64_t p_align;
							} ph {};
							if (phentsize >= sizeof(ph)) {
								for (size_t i = 0; i < phnum; i++) {
									const size_t off = phoff + i * phentsize;
									if (off + sizeof(ph) > cap)
										break;
									std::memcpy(&ph, data + off, sizeof(ph));
									const size_t end = static_cast<size_t>(
										ph.p_offset + ph.p_filesz);
									if (end > max_end && end <= cap)
										max_end = end;
								}
							}
						}
					}

					// Include section header table and section extents when present.
					if (eh.e_shoff != 0 && eh.e_shentsize != 0 && eh.e_shnum != 0) {
						const size_t shoff = static_cast<size_t>(eh.e_shoff);
						const size_t shentsize = static_cast<size_t>(eh.e_shentsize);
						const size_t shnum = static_cast<size_t>(eh.e_shnum);
						if (shoff < cap && shoff + shentsize * shnum <= cap) {
							max_end = std::max(max_end, shoff + shentsize * shnum);
							struct Elf64_ShdrLocal {
								uint32_t sh_name;
								uint32_t sh_type;
								uint64_t sh_flags;
								uint64_t sh_addr;
								uint64_t sh_offset;
								uint64_t sh_size;
								uint32_t sh_link;
								uint32_t sh_info;
								uint64_t sh_addralign;
								uint64_t sh_entsize;
							} sh {};
							if (shentsize >= sizeof(sh)) {
								for (size_t i = 0; i < shnum; i++) {
									const size_t off = shoff + i * shentsize;
									if (off + sizeof(sh) > cap)
										break;
									std::memcpy(&sh, data + off, sizeof(sh));
									const size_t end = static_cast<size_t>(
										sh.sh_offset + sh.sh_size);
									if (end > max_end && end <= cap)
										max_end = end;
								}
							}
						}
					}

					return max_end;
				};

					const size_t elf_size =
						(code_size != 0) ? elf_cap : infer_elf_size(elf_cap);
					// Preserve inferred extent even if we don't end up patching. This
					// is critical for cuLibraryLoadData-style loads where `code_size==0`
					// and we still want to record a usable module image for later
					// launch-time detours.
					if (out_size != nullptr && *out_size == 0 && elf_size != 0)
						*out_size = elf_size;
					if (env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DEBUG")) {
						SPDLOG_INFO(
							"SASS detour: {} raw-ELF probe cap={} inferred_size={} code_size={}",
							api_name ? api_name : "api", elf_cap, elf_size,
							code_size);
					}
						if (elf_size >= 64 && elf_size <= elf_cap) {
							std::vector<uint8_t> elf_vec(data, data + elf_size);
							const int prior_budget =
								impl->sass_detour_filter_state.recent_fatbinc_filter_budget;
							// Single-run identify closure: allow patching raw-ELF code objects that
							// are loaded lazily during a filtered `cuLaunchKernel` call.
							//
							// This addresses the "raw-ELF first" ordering where the SM120 code
							// object appears before we can learn func_id(s) from a later fatbin.
							// When `sass_control_arm_identify()` is active, we treat any SM120
							// raw-ELF loaded in that window as a closure candidate and allow a
							// gated patch-all fallback.
							// Best-effort: avoid taking `owned_cuda_images_lock` here because raw
							// CUDA image loads can be re-entrant and we only need a hint.
								const bool identify_pending_now =
									impl->sass_detour_filter_state.identify_pending.load(
										std::memory_order_acquire);
							// By default, only apply learned/cached func_id filters to raw-ELF
							// cubins after we've seen a matching fatbinc in this run (to reduce
							// the chance of cross-module func_id collisions). For "raw-ELF first"
							// workloads (vLLM flashattention is a common case), allow an explicit
							// override to apply cached func_ids immediately.
							const uint32_t image_id = sass_elf_image_id32(
								std::span<const uint8_t>(elf_vec.data(),
											 elf_vec.size()));
						const bool has_image_scoped_cached =
							(impl->sass_detour_filter_state
								 .learned_func_ids_by_image_id.count(image_id) !=
							 0);
						const bool include_learned =
							(prior_budget > 0) ||
							has_image_scoped_cached ||
							env_truthy_global(
								"BPFTIME_CUDA_SASS_DETOUR_USE_CACHED_FUNC_IDS_FOR_RAW_ELF");
					auto merged_opt = merged_filter_func_ids_for_image(
						include_learned, image_id);
					const std::unordered_set<uint32_t> *merged_ptr =
						merged_opt ? &*merged_opt
							   : extra_filter_func_ids_ptr;
					std::string_view apply_filter_sv = filter_sv;
					if (sass_identify_closure_enabled() && !filter_sv.empty() &&
					    merged_ptr != nullptr && merged_ptr->size() == 1) {
						apply_filter_sv = std::string_view {};
					}
					auto det = sass_detour::apply_elf_text_detours_sm120(
						elf_vec, apply_filter_sv, sample_filter_sv, sampling_cfg,
						merged_ptr);
					if (det && det->image_id != 0)
						out_sm120_image_id = det->image_id;
					if (impl->sass_detour_filter_state.recent_fatbinc_filter_budget > 0)
						impl->sass_detour_filter_state.recent_fatbinc_filter_budget--;

							bool changed = (det && (det->patched_text_sections > 0 ||
										det->sampled_text_sections > 0));
							const bool raw_elf_contains_filter =
								(!filter_sv.empty() &&
								 blob_contains(std::span<const uint8_t>(
										      elf_vec.data(),
										      elf_vec.size()),
									       filter_sv));
					if (!changed && det && allow_patch_all_fallback &&
					    prior_budget > 0 &&
					    env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DEBUG")) {
						SPDLOG_INFO(
							"SASS detour: raw-ELF fallback candidate (reason='{}')",
							det->reason);
					}
							// Single-run identify closure: for stripped SM120 raw-ELF cubins,
							// we may not be able to match by `.text.<name>` or nvinfo/symtab.
							// Patch-all with a gated "controlled" stub (enable=0 by default),
							// and let the host arm Identify/Target around the actual launch.
							// For identify-closure to work on real workloads, we need the
							// candidate SM120 code objects to be detoured *before* the first
							// matching launch. In practice, many SM120 raw-ELF modules can be
							// loaded ahead of the first filtered `cuLaunchKernel` call, and
							// often don't preserve readable names after strip/JIT.
							//
								// Single-run identify closure often needs a "patch-all" fallback for
								// stripped SM120 raw-ELF cubins (no `.text.<name>` match). However,
								// patch-all can be invasive on real workloads because regcount=255
								// kernels cannot reserve scratch registers and must use a more
								// fragile fallback stub.
								//
								// Therefore, keep patch-all-on-miss opt-in. The safer defaults are:
								// - patch-all only while an identify window is armed, or
								// - patch-all only if the raw-ELF blob itself contains the filter
								//   name (e.g., `.nv.info.<kernel>` still present).
								const bool patch_all_on_miss =
									sass_identify_closure_enabled() &&
									env_truthy_global(
										"BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE_PATCH_ALL_ON_MISS") &&
									!env_truthy_global(
										"BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE_NO_PATCH_ALL_ON_MISS");
							if (!changed && det && allow_patch_all_fallback &&
							    sass_identify_closure_enabled() &&
							    !filter_sv.empty() &&
								    (identify_pending_now || patch_all_on_miss ||
								     raw_elf_contains_filter) &&
								    det->reason.find(
									    "no .text.* sections matched filter") !=
									    std::string::npos) {
										const char *why =
											identify_pending_now
												? "identify_pending"
												: (raw_elf_contains_filter
													   ? "contains_filter"
													   : "on_miss_optin");
									SPDLOG_INFO(
										"SASS detour: raw-ELF identify-closure patch-all enabled ({})",
										why);
									auto det_all =
										sass_detour::apply_elf_text_detours_sm120(
											elf_vec, std::string_view{},
											sample_filter_sv, sampling_cfg,
											nullptr,
											/*patch_all_is_fallback=*/true);
									if (det_all && det_all->image_id != 0)
										out_sm120_image_id = det_all->image_id;
						changed = (det_all &&
							   (det_all->patched_text_sections > 0 ||
							    det_all->sampled_text_sections > 0));
						if (changed)
							det = std::move(det_all);
					}
					if (!changed && det && allow_patch_all_fallback &&
					    prior_budget > 0 &&
					    !filter_sv.empty() &&
					    (det->reason.find("no .text.") !=
					     std::string::npos)) {
						SPDLOG_INFO(
							"SASS detour: raw-ELF fallback patch-all enabled (recent fatbinc hit)");
							auto det_all =
								sass_detour::apply_elf_text_detours_sm120(
									elf_vec, std::string_view{},
									sample_filter_sv, sampling_cfg,
									nullptr,
									/*patch_all_is_fallback=*/true);
							if (det_all && det_all->image_id != 0)
								out_sm120_image_id = det_all->image_id;
						changed = (det_all &&
							   (det_all->patched_text_sections > 0 ||
							    det_all->sampled_text_sections > 0));
						if (changed)
							det = std::move(det_all);
						impl->sass_detour_filter_state.recent_fatbinc_filter_budget = 0;
					}

					if (changed && det) {
						if (!det->sampled_kernels.empty())
							impl->record_sass_sampled_kernels(
								det->sampled_kernels);
						if (sampling_cfg && sampling_cfg->enabled &&
						    det->sampled_text_sections > 0) {
							SPDLOG_INFO(
								"SASS sample: instrumented {} .text.* sections (mode={})",
								det->sampled_text_sections,
								(unsigned)sampling_cfg->mode);
						}
						nv_attach_impl::owned_cuda_image base_img;
						base_img.size = elf_vec.size();
						base_img.data =
							std::make_unique<uint8_t[]>(base_img.size);
						std::memcpy(base_img.data.get(), data,
							    base_img.size);
						base_image = std::move(base_img);

						nv_attach_impl::owned_cuda_image img;
						img.size = elf_vec.size();
						img.data = std::make_unique<uint8_t[]>(img.size);
						std::memcpy(img.data.get(), elf_vec.data(), img.size);
						patched = std::move(img);
						output_is_detoured = true;
						if (dump_patched && dump_dir) {
							maybe_dump_cuda_image_bytes(
								dump_dir, api_name, magic, "elf_patched",
								"patched", elf_vec.data(), elf_vec.size());
						}
						} else if (dump_unpatched && dump_dir && det) {
							maybe_dump_cuda_image_bytes(
								dump_dir, api_name, magic, "elf_unpatched",
								det->reason.empty() ? "no_patch" : det->reason,
								elf_vec.data(), elf_vec.size());
					}
				}
			}

	if (!patched)
		return code;

done_detour_patch:
	const void *effective_code = patched->data.get();
	const void *effective_base = base_image ? base_image->data.get() : nullptr;
	const size_t effective_base_size = base_image ? base_image->size : 0;
	// Another thread may have patched the same code while we were compiling/JITing.
	if (auto it = impl->sass_detour_code_cache.find(code);
	    it != impl->sass_detour_code_cache.end()) {
		tls_sass_detour_last_output_is_detoured = it->second.detoured;
		tls_sass_detour_last_base_image = it->second.base;
		tls_sass_detour_last_base_size = it->second.base_size;
		tls_sass_detour_last_sm120_image_id = it->second.sm120_image_id;
		if (out_size != nullptr && it->second.size != 0)
			*out_size = it->second.size;
		return it->second.patched;
	}
	if (base_image)
		impl->owned_cuda_images.emplace_back(std::move(*base_image));
	impl->owned_cuda_images.emplace_back(std::move(*patched));
	const size_t cached_size = impl->owned_cuda_images.back().size;
	impl->sass_detour_code_cache[code] = nv_attach_impl::sass_detour_cache_entry{
		.patched = effective_code,
		.size = cached_size,
		.detoured = output_is_detoured,
		.sm120_image_id = out_sm120_image_id,
		.base = effective_base,
		.base_size = effective_base_size,
	};
	tls_sass_detour_last_output_is_detoured = output_is_detoured;
	tls_sass_detour_last_base_image = effective_base;
	tls_sass_detour_last_base_size = effective_base_size;
	tls_sass_detour_last_sm120_image_id = out_sm120_image_id;
	if (out_size != nullptr)
		*out_size = cached_size;
	if (env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DEBUG")) {
		SPDLOG_INFO("SASS detour: patched {} code={} -> {}", api_name, code,
			    effective_code);
	}
	return effective_code;
}

static cudaError_t
cuda_launch_kernel_common(nv_attach_impl *impl, void *original_fn_ptr,
			  const void *func, dim3 grid_dim, dim3 block_dim,
			  void **args, size_t shared_mem, cudaStream_t stream)
{
	if (impl == nullptr)
		return cudaErrorUnknown;
	auto original =
		reinterpret_cast<cuda_launch_kernel_fn_t>(original_fn_ptr);
	if (!original) {
		SPDLOG_ERROR("Original cudaLaunchKernel function is null");
		return cudaErrorUnknown;
	}
	if (cuda_graph_stream_is_capturing(stream))
		return original(func, grid_dim, block_dim, args, shared_mem,
				stream);
	if (auto itr1 = impl->symbol_address_to_fatbin.find((void *)func);
	    itr1 != impl->symbol_address_to_fatbin.end()) {
		const auto &fatbin = *itr1->second;
		const auto &handle =
			fatbin.function_addr_to_symbol.at((void *)func);
		if (auto err = cuLaunchKernel(
			    handle.func, grid_dim.x, grid_dim.y, grid_dim.z,
			    block_dim.x, block_dim.y, block_dim.z, shared_mem,
			    stream, args, nullptr);
		    err != CUDA_SUCCESS) {
			const char *error_name = nullptr;
			const char *error_string = nullptr;
			cuGetErrorName(err, &error_name);
			cuGetErrorString(err, &error_string);
			SPDLOG_ERROR("Unable to launch kernel: {} ({})",
				     error_name ? error_name : "UNKNOWN",
				     error_string ? error_string :
						    "No description");
			SPDLOG_ERROR("Error code: {}", (int)err);
			return cudaErrorLaunchFailure;
		}
		return cudaSuccess;
	}
	return original(func, grid_dim, block_dim, args, shared_mem, stream);
}

static void example_listener_on_enter(GumInvocationListener *listener,
				      GumInvocationContext *ic)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto context =
		GUM_IC_GET_FUNC_DATA(ic, CUDARuntimeFunctionHookerContext *);
	if (context == nullptr || context->impl == nullptr)
		return;
	if (context->to_function == AttachedToFunction::RegisterFatbin) {
		if (!context->impl->can_patch_ptx())
			return;
		SPDLOG_DEBUG("Entering __cudaRegisterFatBinary..");

		auto header = (__fatBinC_Wrapper_t *)
			gum_invocation_context_get_nth_argument(gum_ctx, 0);
		auto data = (const char *)header->data;
		fat_elf_header_t *curr_header = (fat_elf_header_t *)data;
		const char *tail = (const char *)curr_header;
		while (true) {
			// #define FATBIN_TEXT_MAGIC 0xBA55ED50
			if (curr_header->magic == 0xBA55ED50) {
				SPDLOG_DEBUG(
					"Got CUBIN section header size = {}, size = {}",
					static_cast<int>(
						curr_header->header_size),
					static_cast<int>(curr_header->size));
				tail = ((const char *)curr_header) +
				       curr_header->header_size +
				       curr_header->size;
				curr_header = (fat_elf_header_t *)tail;
			} else {
				break;
			}
		};
		std::vector<uint8_t> data_vec((uint8_t *)data, (uint8_t *)tail);
		SPDLOG_INFO("Finally size = {}", data_vec.size());
		auto extracted_ptx =
			context->impl->extract_ptxs(std::move(data_vec));
		SPDLOG_INFO("Patching PTXs");
		auto fatbin_record = std::make_unique<struct fatbin_record>();
		fatbin_record->original_ptx = extracted_ptx;
		fatbin_record->module_pool = context->impl->module_pool;
		fatbin_record->ptx_pool = context->impl->ptx_pool;

		context->impl->current_fatbin = fatbin_record.get();
		context->impl->fatbin_records.emplace_back(
			std::move(fatbin_record));

	} else if (context->to_function ==
		   AttachedToFunction::RegisterFunction) {
		if (!context->impl->can_patch_ptx())
			return;
		SPDLOG_DEBUG("Entering __cudaRegisterFunction..");
		auto &impl = *context->impl;
		auto current_fatbin = context->impl->current_fatbin;
		current_fatbin->try_loading_ptxs(*context->impl);

		auto func_addr =
			gum_invocation_context_get_nth_argument(gum_ctx, 1);
		auto symbol_name =
			(const char *)gum_invocation_context_get_nth_argument(
				gum_ctx, 3);
		if (auto ok = current_fatbin->find_and_fill_function_info(
			    func_addr, symbol_name);
		    !ok) {
			SPDLOG_WARN(
				"Unable to find_and_fill function info of symbol named {}, the PTX may not be compiled due to not modifying by nv_attach_impl",
				symbol_name);
		} else {
			context->impl->symbol_address_to_fatbin[func_addr] =
				current_fatbin;
				if (auto itr = current_fatbin->function_addr_to_symbol
					       .find(func_addr);
				    itr !=
				    current_fatbin->function_addr_to_symbol.end())
				impl.record_patched_kernel_function_ex(
					std::string(symbol_name), itr->second.func,
					nv_attach_impl::PatchedKernelKind::PtxRewrite, 0);
			SPDLOG_DEBUG(
				"Registered kernel function name {} addr {:x}",
				symbol_name, (uintptr_t)func_addr);
		}

	} else if (context->to_function ==
		   AttachedToFunction::RegisterVariable) {
		if (!context->impl->can_patch_ptx())
			return;
		SPDLOG_DEBUG("Entering __cudaRegisterVar");
		auto current_fatbin = context->impl->current_fatbin;
		current_fatbin->try_loading_ptxs(*context->impl);
		auto fatbin_handle =
			gum_invocation_context_get_nth_argument(gum_ctx, 0);
		auto var_addr =
			gum_invocation_context_get_nth_argument(gum_ctx, 1);
		auto symbol_name =
			(const char *)gum_invocation_context_get_nth_argument(
				gum_ctx, 3);
		SPDLOG_DEBUG("Registering variable named {}", symbol_name);

		if (bool ok = current_fatbin->find_and_fill_variable_info(
			    var_addr, symbol_name);
		    !ok) {
			SPDLOG_WARN(
				"Unable to find_and_fill variable info of symbol names {}, the PTX may not be compiled due to not modifying by nv_attach_impl",
				symbol_name);
		} else {
			context->impl->symbol_address_to_fatbin[var_addr] =
				current_fatbin;
			SPDLOG_DEBUG("Registered variable name {} addr {:x}",
				     symbol_name, (uintptr_t)var_addr);
		}

	} else if (context->to_function ==
		   AttachedToFunction::RegisterFatbinEnd) {
		SPDLOG_DEBUG("Entering __cudaRegisterFatBinaryEnd..");
		auto &current_fatbin = context->impl->current_fatbin;

		current_fatbin = nullptr;
	} else if (context->to_function == AttachedToFunction::CudaMalloc) {
		SPDLOG_DEBUG("Entering cudaMalloc..");
	} else if (context->to_function ==
			   AttachedToFunction::CudaMemcpyToSymbol ||
		   context->to_function ==
			   AttachedToFunction::CudaMemcpyToSymbolAsync) {
		auto symbol =
			(const void *)gum_invocation_context_get_nth_argument(
				gum_ctx, 0);
		auto src =
			(const void *)gum_invocation_context_get_nth_argument(
				gum_ctx, 1);
		auto count = static_cast<size_t>(reinterpret_cast<uintptr_t>(
			gum_invocation_context_get_nth_argument(gum_ctx, 2)));
		auto offset = static_cast<size_t>(reinterpret_cast<uintptr_t>(
			gum_invocation_context_get_nth_argument(gum_ctx, 3)));
		auto kind =
			static_cast<cudaMemcpyKind>(reinterpret_cast<uintptr_t>(
				gum_invocation_context_get_nth_argument(gum_ctx,
									4)));
		cudaStream_t stream = nullptr;
		bool async = context->to_function ==
			     AttachedToFunction::CudaMemcpyToSymbolAsync;
		if (async) {
			stream = (cudaStream_t)
				gum_invocation_context_get_nth_argument(gum_ctx,
									5);
		}
		context->impl->mirror_cuda_memcpy_to_symbol(
			symbol, src, count, offset, kind, stream, async);
	}
}

static void example_listener_on_leave(GumInvocationListener *listener,
				      GumInvocationContext *ic)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto context =
		GUM_IC_GET_FUNC_DATA(ic, CUDARuntimeFunctionHookerContext *);
	if (context->to_function == AttachedToFunction::RegisterFatbin) {
		SPDLOG_DEBUG("Leaving RegisterFatbin");
	} else if (context->to_function ==
		   AttachedToFunction::RegisterFunction) {
		SPDLOG_DEBUG("Leaving RegisterFunction");
	} else if (context->to_function ==
		   AttachedToFunction::RegisterVariable) {
		SPDLOG_DEBUG("Leaving __cudaRegisterVar");
	} else if (context->to_function ==
		   AttachedToFunction::RegisterFatbinEnd) {
		SPDLOG_DEBUG("Leaving __cudaRegisterFatBinaryEnd..");
	}
}

static void
cuda_runtime_function_hooker_class_init(CUDARuntimeFunctionHookerClass *klass)
{
}

static void cuda_runtime_function_hooker_iface_init(gpointer g_iface,
						    gpointer iface_data)
{
	auto iface = (GumInvocationListenerInterface *)g_iface;

	iface->on_enter = example_listener_on_enter;
	iface->on_leave = example_listener_on_leave;
}

static void cuda_runtime_function_hooker_init(CUDARuntimeFunctionHooker *self)
{
}

extern "C" cudaError_t
cuda_runtime_function__cudaLaunchKernel(const void *func, dim3 grid_dim,
					dim3 block_dim, void **args,
					size_t shared_mem, cudaStream_t stream)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	if (impl != nullptr) {
		SPDLOG_DEBUG("grid_dim: {}, {}, {}", grid_dim.x, grid_dim.y,
			     grid_dim.z);
		SPDLOG_DEBUG("block_dim: {}, {}, {}", block_dim.x, block_dim.y,
			     block_dim.z);
	}
	return cuda_launch_kernel_common(impl,
					 impl->original_cuda_launch_kernel,
					 func, grid_dim, block_dim, args,
					 shared_mem, stream);
}

static std::optional<std::string>
cuda_graph_maybe_get_kernel_name_from_cufunction(nv_attach_impl &impl,
						 CUfunction function)
{
	if (auto cached = impl.find_original_kernel_name(function); cached)
		return cached;
	using cu_func_get_name_fn_t = CUresult (*)(const char **, CUfunction);
	static cu_func_get_name_fn_t cu_func_get_name =
		(cu_func_get_name_fn_t)dlsym(RTLD_DEFAULT, "cuFuncGetName");
	if (!cu_func_get_name)
		return std::nullopt;
	const char *name = nullptr;
	if (auto err = cu_func_get_name(&name, function); err != CUDA_SUCCESS)
		return std::nullopt;
	if (name == nullptr || name[0] == '\0')
		return std::nullopt;
	impl.record_original_cufunction_name(function, std::string(name));
	return std::string(name);
}

static std::optional<std::string>
cuda_graph_maybe_get_kernel_name_from_cukernel(CUkernel kernel)
{
	using cu_kernel_get_name_fn_t = CUresult (*)(const char **, CUkernel);
	static cu_kernel_get_name_fn_t cu_kernel_get_name =
		(cu_kernel_get_name_fn_t)dlsym(RTLD_DEFAULT, "cuKernelGetName");
	if (!cu_kernel_get_name)
		return std::nullopt;
	const char *name = nullptr;
	if (auto err = cu_kernel_get_name(&name, kernel); err != CUDA_SUCCESS)
		return std::nullopt;
	if (name == nullptr)
		return std::nullopt;
	return std::string(name);
}

namespace {
constexpr uint32_t kFatbinTextMagic = 0xBA55ED50;
constexpr uint32_t kFatbincMagic = 0x466243B1;

struct alignas(8) FatbinHeaderLocal {
	uint32_t magic;
	uint16_t version;
	uint16_t header_size;
	uint64_t files_size;
};
static_assert(sizeof(FatbinHeaderLocal) == 16);

struct FatbinFileHeaderLocal {
	uint16_t kind;
	uint16_t version;
	uint32_t header_size;
	uint32_t padded_payload_size;
	uint32_t unknown0;
	uint32_t payload_size;
	uint32_t unknown1;
	uint32_t unknown2;
	uint32_t sm_version;
	uint32_t bit_width;
	uint32_t unknown3;
	uint64_t unknown4;
	uint64_t unknown5;
	uint64_t uncompressed_payload;
};
static_assert(sizeof(FatbinFileHeaderLocal) == 0x40);

static size_t align_up_size(size_t v, size_t a)
{
	if (a == 0)
		return v;
	return (v + a - 1) & ~(a - 1);
}

struct Lz4ApiLocal {
	using decompress_fn_t = int (*)(const char *, char *, int, int);
	using compress_fn_t = int (*)(const char *, char *, int, int);
	using compress_bound_fn_t = int (*)(int);

	void *handle = nullptr;
	decompress_fn_t decompress = nullptr;
	compress_fn_t compress = nullptr;
	compress_bound_fn_t compress_bound = nullptr;

	static const Lz4ApiLocal &instance()
	{
		static Lz4ApiLocal api = [] {
			Lz4ApiLocal out;
			out.handle = dlopen("liblz4.so.1", RTLD_LAZY | RTLD_LOCAL);
			if (!out.handle)
				out.handle = dlopen("liblz4.so", RTLD_LAZY | RTLD_LOCAL);
			if (!out.handle)
				return out;
			out.decompress = (decompress_fn_t)dlsym(out.handle,
							       "LZ4_decompress_safe");
			out.compress =
				(compress_fn_t)dlsym(out.handle, "LZ4_compress_default");
			out.compress_bound =
				(compress_bound_fn_t)dlsym(out.handle, "LZ4_compressBound");
			if (!out.decompress || !out.compress || !out.compress_bound) {
				dlclose(out.handle);
				out.handle = nullptr;
				out.decompress = nullptr;
				out.compress = nullptr;
				out.compress_bound = nullptr;
			}
			return out;
		}();
		return api;
	}
};

static std::optional<std::vector<uint8_t>>
decompress_lz4_local(const uint8_t *src, size_t src_len, size_t dst_len)
{
	const auto &lz4 = Lz4ApiLocal::instance();
	if (!lz4.decompress)
		return std::nullopt;
	if (dst_len == 0 || dst_len > (256u << 20))
		return std::nullopt;
	std::vector<uint8_t> out(dst_len);
	const int r = lz4.decompress(reinterpret_cast<const char *>(src),
				     reinterpret_cast<char *>(out.data()),
				     (int)src_len, (int)dst_len);
	if (r < 0)
		return std::nullopt;
	return out;
}

static std::optional<std::vector<uint8_t>>
compress_lz4_local(const std::vector<uint8_t> &src)
{
	const auto &lz4 = Lz4ApiLocal::instance();
	if (!lz4.compress || !lz4.compress_bound)
		return std::nullopt;
	const int bound = lz4.compress_bound((int)src.size());
	if (bound <= 0 || bound > (256u << 20))
		return std::nullopt;
	std::vector<uint8_t> out((size_t)bound);
	const int n = lz4.compress(reinterpret_cast<const char *>(src.data()),
				   reinterpret_cast<char *>(out.data()), (int)src.size(),
				   bound);
	if (n <= 0)
		return std::nullopt;
	out.resize((size_t)n);
	return out;
}

static std::optional<size_t>
parse_fatbin_total_size_local(const uint8_t *data, size_t cap)
{
	if (!data || cap < sizeof(FatbinHeaderLocal))
		return std::nullopt;
	FatbinHeaderLocal h {};
	std::memcpy(&h, data, sizeof(h));
	if (h.magic != kFatbinTextMagic)
		return std::nullopt;
	if (h.version != 1)
		return std::nullopt;
	if (h.header_size < sizeof(FatbinHeaderLocal))
		return std::nullopt;
	const size_t header_size = (size_t)h.header_size;
	const size_t files_size = (size_t)h.files_size;
	if (header_size > cap)
		return std::nullopt;
	if (files_size > cap - header_size)
		return std::nullopt;
	return header_size + files_size;
}

static std::optional<size_t>
find_fatbin_text_magic_offset_local(const uint8_t *data, size_t max_scan)
{
	if (!data || max_scan < 4)
		return std::nullopt;
	for (size_t off = 0; off + 4 <= max_scan; off++) {
		uint32_t v = 0;
		std::memcpy(&v, data + off, sizeof(v));
		if (v == kFatbinTextMagic)
			return off;
	}
	return std::nullopt;
}

static std::optional<std::span<const uint8_t>>
locate_fatbin_span_local(const uint8_t *p)
{
	constexpr size_t kMaxFatbinBytes = 256u << 20;
	constexpr size_t kMaxFatPrefixScanBytes = 256u << 10;
	if (!p)
		return std::nullopt;
	if (auto sz = parse_fatbin_total_size_local(p, kMaxFatbinBytes))
		return std::span<const uint8_t>(p, *sz);
	if (auto off = find_fatbin_text_magic_offset_local(p, kMaxFatPrefixScanBytes)) {
		if (auto sz = parse_fatbin_total_size_local(
			    p + *off, kMaxFatbinBytes > *off ? (kMaxFatbinBytes - *off) : 0)) {
			return std::span<const uint8_t>(p + *off, *sz);
		}
	}
	return std::nullopt;
}

static std::optional<std::vector<uint8_t>>
sass_detour_patch_fatbin_patch_all_sm120(std::span<const uint8_t> fatbin,
					const sass_detour::Sm120SamplingConfig *sampling_cfg)
{
	if (fatbin.size() < sizeof(FatbinHeaderLocal))
		return std::nullopt;
	FatbinHeaderLocal h {};
	std::memcpy(&h, fatbin.data(), sizeof(h));
	if (h.magic != kFatbinTextMagic || h.version != 1)
		return std::nullopt;
	const size_t header_size = (size_t)h.header_size;
	if (header_size < sizeof(FatbinHeaderLocal) || header_size > fatbin.size())
		return std::nullopt;
	if (fatbin.size() < header_size + (size_t)h.files_size)
		return std::nullopt;

	const auto file_region =
		fatbin.subspan(header_size, (size_t)h.files_size);
	std::vector<uint8_t> out;
	out.assign(fatbin.begin(), fatbin.begin() + header_size);

	bool any_patched = false;
	size_t off = 0;
	while (off < file_region.size()) {
		if (file_region.size() - off < sizeof(FatbinFileHeaderLocal))
			break;
		FatbinFileHeaderLocal fh {};
		std::memcpy(&fh, file_region.data() + off, sizeof(fh));
		const size_t fh_size = (size_t)fh.header_size;
		const size_t padded = (size_t)fh.padded_payload_size;
		const size_t payload_size = (size_t)fh.payload_size;
		if (fh_size < sizeof(FatbinFileHeaderLocal) || fh_size > (1u << 20))
			break;
		if (padded > file_region.size() - off - fh_size)
			break;
		const auto rec_span = file_region.subspan(off, fh_size + padded);

		const auto payload_padded = file_region.subspan(off + fh_size, padded);
		const auto payload_comp =
			payload_padded.subspan(0, std::min(padded, payload_size));

		constexpr uint16_t kKindElf = 0x02;
		if (fh.kind != kKindElf) {
			out.insert(out.end(), rec_span.begin(), rec_span.end());
			off += fh_size + padded;
			continue;
		}

		const bool likely_compressed =
			(fh.uncompressed_payload > (uint64_t)payload_size &&
			 payload_size != 0 &&
			 !(payload_comp.size() >= 4 && payload_comp[0] == 0x7f &&
			   payload_comp[1] == 'E' && payload_comp[2] == 'L' &&
			   payload_comp[3] == 'F'));

		std::optional<std::vector<uint8_t>> elf_opt;
		if (likely_compressed) {
			elf_opt = decompress_lz4_local(
				payload_comp.data(), payload_comp.size(),
				(size_t)fh.uncompressed_payload);
		} else {
			elf_opt = std::vector<uint8_t>(payload_padded.begin(),
						       payload_padded.end());
		}
		if (!elf_opt) {
			out.insert(out.end(), rec_span.begin(), rec_span.end());
			off += fh_size + padded;
			continue;
		}

		auto &elf_bytes = *elf_opt;
		auto det = sass_detour::apply_elf_text_detours_sm120(
			elf_bytes, std::string_view {}, std::string_view {}, sampling_cfg,
			/*extra_filter_func_ids=*/nullptr,
			/*patch_all_is_fallback=*/true);
		const bool rec_patched =
			(det && (det->patched_text_sections > 0 ||
				 det->sampled_text_sections > 0));
		if (!rec_patched) {
			out.insert(out.end(), rec_span.begin(), rec_span.end());
			off += fh_size + padded;
			continue;
		}

		any_patched = true;

		std::vector<uint8_t> new_payload_bytes;
		FatbinFileHeaderLocal fh_out = fh;
		if (likely_compressed) {
			auto comp = compress_lz4_local(elf_bytes);
			if (!comp)
				return std::nullopt;
			new_payload_bytes = std::move(*comp);
			fh_out.payload_size = (uint32_t)new_payload_bytes.size();
			fh_out.padded_payload_size =
				(uint32_t)align_up_size(new_payload_bytes.size(), 8);
			fh_out.uncompressed_payload = (uint64_t)elf_bytes.size();
		} else {
			new_payload_bytes = std::move(elf_bytes);
			fh_out.payload_size = (uint32_t)new_payload_bytes.size();
			fh_out.padded_payload_size =
				(uint32_t)align_up_size(new_payload_bytes.size(), 8);
			fh_out.uncompressed_payload = (uint64_t)new_payload_bytes.size();
		}

		std::vector<uint8_t> rec_hdr_bytes(rec_span.begin(),
						   rec_span.begin() + fh_size);
		if (rec_hdr_bytes.size() >= sizeof(FatbinFileHeaderLocal))
			std::memcpy(rec_hdr_bytes.data(), &fh_out,
				    sizeof(FatbinFileHeaderLocal));

		out.insert(out.end(), rec_hdr_bytes.begin(), rec_hdr_bytes.end());
		out.insert(out.end(), new_payload_bytes.begin(), new_payload_bytes.end());
		const size_t pad =
			align_up_size(new_payload_bytes.size(), 8) - new_payload_bytes.size();
		out.insert(out.end(), pad, 0);

		off += fh_size + padded;
	}

	if (!any_patched)
		return std::nullopt;

	// Fix up files_size in the fatbin header.
	const size_t new_files_size = out.size() - header_size;
	if (out.size() >= sizeof(FatbinHeaderLocal)) {
		std::memcpy(out.data() + offsetof(FatbinHeaderLocal, files_size),
			    &new_files_size, sizeof(uint64_t));
	}
	return out;
}

static std::optional<nv_attach_impl::owned_cuda_image>
sass_detour_patch_fatbinc_patch_all_sm120(
	const __fatBinC_Wrapper_t &wrapper,
	const sass_detour::Sm120SamplingConfig *sampling_cfg)
{
	const uint8_t *data0 =
		reinterpret_cast<const uint8_t *>(wrapper.data);
	auto primary_span_opt = locate_fatbin_span_local(data0);
	if (!primary_span_opt)
		return std::nullopt;

	const bool is_wrapper_v2 = (wrapper.version == 2);
	std::vector<std::span<const uint8_t>> spans;
	spans.reserve(1);
	spans.push_back(*primary_span_opt);

	bool extra_list_complete = true;
	if (is_wrapper_v2) {
		const void *const *list =
			reinterpret_cast<const void *const *>(wrapper.filename_or_fatbins);
		if (!list) {
			extra_list_complete = false;
		} else {
			for (size_t i = 0; i < 64; i++) {
				const void *entry = list[i];
				if (!entry)
					break;
				auto span_opt = locate_fatbin_span_local(
					reinterpret_cast<const uint8_t *>(entry));
				if (!span_opt) {
					extra_list_complete = false;
					break;
				}
				spans.push_back(*span_opt);
			}
		}
	}

	struct FbBytes {
		std::vector<uint8_t> bytes;
	};
	std::vector<FbBytes> fatbins_to_copy;
	fatbins_to_copy.reserve(spans.size());
	bool any_changes = false;
	for (const auto &s : spans) {
		FbBytes fb;
		if (auto patched = sass_detour_patch_fatbin_patch_all_sm120(
			    s, sampling_cfg)) {
			fb.bytes = std::move(*patched);
			any_changes = true;
		} else {
			fb.bytes.assign(s.begin(), s.end());
		}
		fatbins_to_copy.push_back(std::move(fb));
	}

	if (!any_changes)
		return std::nullopt;

	const size_t wrapper_size = sizeof(__fatBinC_Wrapper_t);
	size_t cur = align_up_size(wrapper_size, 8);

	std::vector<size_t> fatbin_offsets;
	fatbin_offsets.reserve(fatbins_to_copy.size());
	for (const auto &fb : fatbins_to_copy) {
		cur = align_up_size(cur, 8);
		fatbin_offsets.push_back(cur);
		cur += fb.bytes.size();
	}

	size_t list_off = 0;
	const bool can_rewrite_extra_list =
		is_wrapper_v2 && extra_list_complete && fatbins_to_copy.size() > 1;
	if (can_rewrite_extra_list) {
		cur = align_up_size(cur, alignof(void *));
		list_off = cur;
		const size_t n_ptrs = (fatbins_to_copy.size() - 1) + 1;
		cur += n_ptrs * sizeof(void *);
	}

	nv_attach_impl::owned_cuda_image img;
	img.size = cur;
	img.data = std::make_unique<uint8_t[]>(img.size);
	std::memset(img.data.get(), 0, img.size);
	auto *wcopy =
		reinterpret_cast<__fatBinC_Wrapper_t *>(img.data.get());
	*wcopy = wrapper;

	for (size_t i = 0; i < fatbins_to_copy.size(); i++) {
		std::memcpy(img.data.get() + fatbin_offsets[i],
			    fatbins_to_copy[i].bytes.data(),
			    fatbins_to_copy[i].bytes.size());
	}
	wcopy->data = reinterpret_cast<const unsigned long long *>(
		img.data.get() + fatbin_offsets[0]);

	if (list_off != 0) {
		void **out_list =
			reinterpret_cast<void **>(img.data.get() + list_off);
		for (size_t i = 1; i < fatbins_to_copy.size(); i++)
			out_list[i - 1] = img.data.get() + fatbin_offsets[i];
		out_list[fatbins_to_copy.size() - 1] = nullptr;
		wcopy->filename_or_fatbins = reinterpret_cast<void *>(out_list);
	}

	return img;
}
} // namespace

static std::optional<CUfunction>
sass_detour_on_demand_patch_all_for_kernel(nv_attach_impl &impl, CUfunction f,
					   const std::string &kernel_name)
{
	if (f == nullptr || kernel_name.empty())
		return std::nullopt;
	if (!sass_identify_closure_enabled())
		return std::nullopt;
	if (!env_truthy_global("BPFTIME_CUDA_SASS_DETOUR"))
		return std::nullopt;

	// If we already have a patched replacement for this kernel, use it.
	//
	// Note: patch-all fallback is an intermediate state for identify closure.
	// Once a target func_id is resolved we will upgrade to a func-id-targeted
	// patch; keep returning whatever we have for the moment.
	if (auto cached = impl.find_patched_kernel_function(kernel_name); cached)
		return cached;

	auto sampling_cfg_opt = impl.get_sm120_sampling_cfg();
	const auto *sampling_cfg = sampling_cfg_opt ? &*sampling_cfg_opt : nullptr;
	if (!sampling_cfg || !sampling_cfg->enabled || !sampling_cfg->control_enabled)
		return std::nullopt;

	const auto mi = impl.find_cuda_module_image_by_function(f);
	if (!mi || mi->image_ptr == 0)
		return std::nullopt;

	// If the module image was already patched at load time, don't try to
	// re-load another module here.
	if (mi->patched)
		return std::nullopt;

	auto cuModuleLoadData_v2 =
		reinterpret_cast<CUresult (*)(CUmodule *, const void *)>(
			impl.original_cu_module_load_data);
	auto cuModuleGetFunction_v2 =
		reinterpret_cast<CUresult (*)(CUfunction *, CUmodule, const char *)>(
			impl.original_cu_module_get_function);
	const bool prefer_library_load =
		(mi->api.find("cuLibrary") != std::string::npos);
	auto cuLibraryLoadData_v2 =
		reinterpret_cast<CUresult (*)(
			CUlibrary *, const void *, CUjit_option *, void **,
			unsigned int, CUlibraryOption *, void **, unsigned int)>(
			impl.original_cu_library_load_data);
	auto cuLibraryGetKernel_v2 =
		reinterpret_cast<CUresult (*)(CUkernel *, CUlibrary, const char *)>(
			impl.original_cu_library_get_kernel);
	auto cuKernelGetFunction_v2 =
		reinterpret_cast<CUresult (*)(CUfunction *, CUkernel)>(
			impl.original_cu_kernel_get_function);

	if ((!prefer_library_load &&
	     (!cuModuleLoadData_v2 || !cuModuleGetFunction_v2)) ||
	    (prefer_library_load &&
	     (!cuLibraryLoadData_v2 || !cuLibraryGetKernel_v2 || !cuKernelGetFunction_v2))) {
		return std::nullopt;
	}

	const void *patched_code = nullptr;
	size_t patched_size = 0;
	uint32_t magic = 0;
	std::memcpy(&magic, reinterpret_cast<const void *>(mi->image_ptr), sizeof(magic));

	// Fast path: raw ELF module image with a known span.
	if (magic == 0x464c457f /* ELF */) {
		if (mi->image_size == 0) {
			if (env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DEBUG_LAUNCH")) {
				SPDLOG_WARN(
					"SASS detour: on-demand patch-all image_size=0 for ELF kernel='{}' (api='{}')",
					kernel_name, mi->api);
			}
			return std::nullopt;
		}
		std::vector<uint8_t> elf_vec(
			reinterpret_cast<const uint8_t *>(mi->image_ptr),
			reinterpret_cast<const uint8_t *>(mi->image_ptr) + mi->image_size);
		auto det = sass_detour::apply_elf_text_detours_sm120(
			elf_vec, std::string_view {}, std::string_view {}, sampling_cfg,
			/*extra_filter_func_ids=*/nullptr,
			/*patch_all_is_fallback=*/true);
		if (!det || det->patched_text_sections == 0) {
			if (env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DEBUG_LAUNCH")) {
				SPDLOG_INFO(
					"SASS detour: on-demand patch-all(fallback) did not produce a patched image for kernel='{}' (magic=0x{:08x})",
					kernel_name, magic);
			}
			return std::nullopt;
		}
		{
			std::lock_guard<std::mutex> guard(impl.owned_cuda_images_lock);
			nv_attach_impl::owned_cuda_image img;
			img.size = elf_vec.size();
			img.data = std::make_unique<uint8_t[]>(img.size);
			std::memcpy(img.data.get(), elf_vec.data(), img.size);
			impl.owned_cuda_images.push_back(std::move(img));
			patched_code = impl.owned_cuda_images.back().data.get();
			patched_size = impl.owned_cuda_images.back().size;
		}
	} else {
		// General path: let the load-time patcher handle fatbinc/fatbin/unknown
		// size (including PTX->cubin JIT-link fallback when enabled).
		size_t out_sz = mi->image_size;
		const void *patched = maybe_patch_cuda_image_sass_detour(
			&impl, reinterpret_cast<const void *>(mi->image_ptr),
			mi->image_size, "cuLaunchKernel_on_demand_patch_all", &out_sz);
		if (!tls_sass_detour_last_output_is_detoured || patched == nullptr ||
		    out_sz == 0) {
			if (env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DEBUG_LAUNCH")) {
				SPDLOG_INFO(
					"SASS detour: on-demand patch-all(fallback) did not produce a patched image for kernel='{}' (magic=0x{:08x})",
					kernel_name, magic);
			}
			return std::nullopt;
		}
		patched_code = patched;
		patched_size = out_sz;
	}

	CUmodule new_mod = nullptr;
	CUlibrary new_lib = nullptr;
	CUkernel new_kernel = nullptr;
	CUfunction new_func = nullptr;
	if (prefer_library_load) {
		if (auto r = cuLibraryLoadData_v2(
			    &new_lib, patched_code,
			    /*jitOptions=*/nullptr, /*jitOptionValues=*/nullptr,
			    /*numJitOptions=*/0, /*libraryOptions=*/nullptr,
			    /*libraryOptionValues=*/nullptr,
			    /*numLibraryOptions=*/0);
		    r != CUDA_SUCCESS || new_lib == nullptr) {
			if (env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DEBUG_LAUNCH")) {
				SPDLOG_WARN(
					"SASS detour: on-demand patch-all cuLibraryLoadData failed for kernel='{}': {}",
					kernel_name, (int)r);
			}
			return std::nullopt;
		}
		if (auto r = cuLibraryGetKernel_v2(&new_kernel, new_lib,
						   kernel_name.c_str());
		    r != CUDA_SUCCESS || new_kernel == nullptr) {
			if (env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DEBUG_LAUNCH")) {
				SPDLOG_WARN(
					"SASS detour: on-demand patch-all cuLibraryGetKernel failed for kernel='{}': {}",
					kernel_name, (int)r);
			}
			return std::nullopt;
		}
		if (auto r = cuKernelGetFunction_v2(&new_func, new_kernel);
		    r != CUDA_SUCCESS || new_func == nullptr) {
			if (env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DEBUG_LAUNCH")) {
				SPDLOG_WARN(
					"SASS detour: on-demand patch-all cuKernelGetFunction failed for kernel='{}': {}",
					kernel_name, (int)r);
			}
			return std::nullopt;
		}
	} else {
		if (auto r = cuModuleLoadData_v2(&new_mod, patched_code);
		    r != CUDA_SUCCESS || new_mod == nullptr) {
			if (env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DEBUG_LAUNCH")) {
				SPDLOG_WARN(
					"SASS detour: on-demand cuModuleLoadData failed for kernel='{}': {}",
					kernel_name, (int)r);
			}
			return std::nullopt;
		}
		if (auto r = cuModuleGetFunction_v2(&new_func, new_mod,
						    kernel_name.c_str());
		    r != CUDA_SUCCESS || new_func == nullptr) {
			if (env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DEBUG_LAUNCH")) {
				SPDLOG_WARN(
					"SASS detour: on-demand cuModuleGetFunction failed for kernel='{}': {}",
					kernel_name, (int)r);
			}
			return std::nullopt;
		}
	}

	// Ensure the replacement CUfunction inherits launch-related attributes that
	// the framework may have configured on the original CUfunction (e.g., max
	// dynamic shared memory for flashattention kernels).
	maybe_clone_launch_attributes(f, new_func);

	impl.record_patched_kernel_function_ex(
		kernel_name, new_func, nv_attach_impl::PatchedKernelKind::PatchAllFallback, 0);
	impl.record_original_cufunction_name(new_func, kernel_name);
	if (prefer_library_load) {
		impl.record_original_cufunction_cukernel(new_func, new_kernel);
		impl.record_original_cukernel_name(new_kernel, kernel_name);
		impl.record_original_cukernel_library(new_kernel, new_lib);
		impl.record_cuda_library_image(
			new_lib, patched_code, patched_size,
			fnv1a64((const uint8_t *)patched_code, patched_size),
			/*patched=*/true, "cuLaunchKernel_on_demand_patch_all");
	} else {
		impl.record_original_cufunction_module(new_func, new_mod);
		impl.record_cuda_module_image(
			new_mod, patched_code, patched_size,
			fnv1a64((const uint8_t *)patched_code, patched_size),
			/*patched=*/true, "cuLaunchKernel_on_demand_patch_all");
	}
	if (env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DEBUG_LAUNCH")) {
		SPDLOG_INFO(
			"SASS detour: on-demand patched launch kernel='{}' old_func=0x{:x} -> new_func=0x{:x} (new_mod=0x{:x} new_lib=0x{:x} size={})",
			kernel_name, (uint64_t)f, (uint64_t)new_func, (uint64_t)new_mod,
			(uint64_t)new_lib, patched_size);
	}
	return new_func;
}

static std::optional<CUfunction>
sass_detour_on_demand_patch_func_id_for_kernel(nv_attach_impl &impl, CUfunction f,
					       const std::string &kernel_name,
					       uint32_t func_id)
{
	if (f == nullptr || kernel_name.empty() || func_id == 0)
		return std::nullopt;
	if (!sass_identify_closure_enabled())
		return std::nullopt;
	if (!env_truthy_global("BPFTIME_CUDA_SASS_DETOUR"))
		return std::nullopt;

	// If we already have a func-id-targeted replacement for this kernel and it
	// matches the requested func_id, use it. Otherwise, allow upgrading from a
	// patch-all fallback into a func-id-targeted patch.
	if (auto cached = impl.find_patched_kernel_entry(kernel_name); cached) {
		if (cached->kind == nv_attach_impl::PatchedKernelKind::FuncIdTargeted &&
		    cached->target_func_id == func_id && cached->function != nullptr) {
			return cached->function;
		}
		// If we have *any* cached function that is not the patch-all fallback,
		// do not try to reload yet another module here.
		if (cached->kind != nv_attach_impl::PatchedKernelKind::PatchAllFallback &&
		    cached->function != nullptr) {
			return cached->function;
		}
	}

	auto sampling_cfg_opt = impl.get_sm120_sampling_cfg();
	const auto *sampling_cfg = sampling_cfg_opt ? &*sampling_cfg_opt : nullptr;
	if (!sampling_cfg || !sampling_cfg->enabled)
		return std::nullopt;

	const auto mi = impl.find_cuda_module_image_by_function(f);
	if (!mi || mi->image_ptr == 0)
		return std::nullopt;

	// For load-time detoured images, prefer the stored base image for upgrades
	// to avoid double-detouring.
	const uint8_t *base_ptr =
		reinterpret_cast<const uint8_t *>(mi->image_ptr);
	size_t base_size = mi->image_size;
	if (mi->patched) {
		if (mi->base_image_ptr == 0 || mi->base_image_size < 64)
			return std::nullopt;
		base_ptr = reinterpret_cast<const uint8_t *>(mi->base_image_ptr);
		base_size = mi->base_image_size;
	}
	if (base_size < 64)
		return std::nullopt;

	const bool prefer_library_load =
		(mi->api.find("cuLibrary") != std::string::npos);
	auto cuModuleLoadData_v2 =
		reinterpret_cast<CUresult (*)(CUmodule *, const void *)>(
			impl.original_cu_module_load_data);
	auto cuModuleGetFunction_v2 =
		reinterpret_cast<CUresult (*)(CUfunction *, CUmodule, const char *)>(
			impl.original_cu_module_get_function);
	auto cuLibraryLoadData_v2 =
		reinterpret_cast<CUresult (*)(
			CUlibrary *, const void *, CUjit_option *, void **,
			unsigned int, CUlibraryOption *, void **, unsigned int)>(
			impl.original_cu_library_load_data);
	auto cuLibraryGetKernel_v2 =
		reinterpret_cast<CUresult (*)(CUkernel *, CUlibrary, const char *)>(
			impl.original_cu_library_get_kernel);
	auto cuKernelGetFunction_v2 =
		reinterpret_cast<CUresult (*)(CUfunction *, CUkernel)>(
			impl.original_cu_kernel_get_function);
	if ((!prefer_library_load &&
	     (!cuModuleLoadData_v2 || !cuModuleGetFunction_v2)) ||
	    (prefer_library_load &&
	     (!cuLibraryLoadData_v2 || !cuLibraryGetKernel_v2 || !cuKernelGetFunction_v2))) {
		return std::nullopt;
	}

	std::vector<uint8_t> elf_vec(base_ptr, base_ptr + base_size);
	std::unordered_set<uint32_t> only { func_id };
	auto det = sass_detour::apply_elf_text_detours_sm120(
		elf_vec, std::string_view {}, std::string_view {}, sampling_cfg,
		&only);
	const bool changed =
		(det && (det->patched_text_sections > 0 || det->sampled_text_sections > 0));
	if (!changed)
		return std::nullopt;

	const void *patched_code = nullptr;
	size_t patched_size = 0;
	{
		std::lock_guard<std::mutex> guard(impl.owned_cuda_images_lock);
		nv_attach_impl::owned_cuda_image img;
		img.size = elf_vec.size();
		img.data = std::make_unique<uint8_t[]>(img.size);
		std::memcpy(img.data.get(), elf_vec.data(), img.size);
		impl.owned_cuda_images.push_back(std::move(img));
		patched_code = impl.owned_cuda_images.back().data.get();
		patched_size = impl.owned_cuda_images.back().size;
	}

	CUmodule new_mod = nullptr;
	CUlibrary new_lib = nullptr;
	CUkernel new_kernel = nullptr;
	CUfunction new_func = nullptr;
	if (prefer_library_load) {
		if (auto r = cuLibraryLoadData_v2(
			    &new_lib, patched_code,
			    /*jitOptions=*/nullptr, /*jitOptionValues=*/nullptr,
			    /*numJitOptions=*/0, /*libraryOptions=*/nullptr,
			    /*libraryOptionValues=*/nullptr,
			    /*numLibraryOptions=*/0);
		    r != CUDA_SUCCESS || new_lib == nullptr) {
			if (env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DEBUG_LAUNCH")) {
				SPDLOG_WARN(
					"SASS detour: on-demand func_id={} cuLibraryLoadData failed for kernel='{}': {}",
					func_id, kernel_name, (int)r);
			}
			return std::nullopt;
		}
		if (auto r = cuLibraryGetKernel_v2(&new_kernel, new_lib,
						   kernel_name.c_str());
		    r != CUDA_SUCCESS || new_kernel == nullptr) {
			if (env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DEBUG_LAUNCH")) {
				SPDLOG_WARN(
					"SASS detour: on-demand func_id={} cuLibraryGetKernel failed for kernel='{}': {}",
					func_id, kernel_name, (int)r);
			}
			return std::nullopt;
		}
		if (auto r = cuKernelGetFunction_v2(&new_func, new_kernel);
		    r != CUDA_SUCCESS || new_func == nullptr) {
			if (env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DEBUG_LAUNCH")) {
				SPDLOG_WARN(
					"SASS detour: on-demand func_id={} cuKernelGetFunction failed for kernel='{}': {}",
					func_id, kernel_name, (int)r);
			}
			return std::nullopt;
		}
	} else {
		if (auto r = cuModuleLoadData_v2(&new_mod, patched_code);
		    r != CUDA_SUCCESS || new_mod == nullptr) {
			if (env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DEBUG_LAUNCH")) {
				SPDLOG_WARN(
					"SASS detour: on-demand func_id={} cuModuleLoadData failed for kernel='{}': {}",
					func_id, kernel_name, (int)r);
			}
			return std::nullopt;
		}
		if (auto r = cuModuleGetFunction_v2(&new_func, new_mod,
						    kernel_name.c_str());
		    r != CUDA_SUCCESS || new_func == nullptr) {
			if (env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DEBUG_LAUNCH")) {
				SPDLOG_WARN(
					"SASS detour: on-demand func_id={} cuModuleGetFunction failed for kernel='{}': {}",
					func_id, kernel_name, (int)r);
			}
			return std::nullopt;
		}
	}

	// Ensure the replacement CUfunction inherits launch-related attributes that
	// the framework may have configured on the original CUfunction (e.g., max
	// dynamic shared memory for flashattention kernels).
	maybe_clone_launch_attributes(f, new_func);

	impl.record_patched_kernel_function_ex(
		kernel_name, new_func, nv_attach_impl::PatchedKernelKind::FuncIdTargeted, func_id);
	impl.record_original_cufunction_name(new_func, kernel_name);
	if (prefer_library_load) {
		impl.record_original_cufunction_cukernel(new_func, new_kernel);
		impl.record_original_cukernel_name(new_kernel, kernel_name);
		impl.record_original_cukernel_library(new_kernel, new_lib);
		impl.record_cuda_library_image(
			new_lib, patched_code, patched_size,
			fnv1a64((const uint8_t *)patched_code, patched_size),
			/*patched=*/true, "cuLaunchKernel_on_demand_func_id");
	} else {
		impl.record_original_cufunction_module(new_func, new_mod);
		impl.record_cuda_module_image(
			new_mod, patched_code, patched_size,
			fnv1a64((const uint8_t *)patched_code, patched_size),
			/*patched=*/true, "cuLaunchKernel_on_demand_func_id");
	}
	if (env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DEBUG_LAUNCH")) {
		SPDLOG_INFO(
			"SASS detour: on-demand func_id={} patched launch kernel='{}' old_func=0x{:x} -> new_func=0x{:x} (new_mod=0x{:x} new_lib=0x{:x} size={})",
			func_id, kernel_name, (uint64_t)f, (uint64_t)new_func,
			(uint64_t)new_mod, (uint64_t)new_lib, patched_size);
	}
	return new_func;
}

extern "C" cudaError_t cuda_runtime_function__cudaLaunchKernel_ptsz(
	const void *func, dim3 grid_dim, dim3 block_dim, void **args,
	size_t shared_mem, cudaStream_t stream)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	if (impl != nullptr) {
		SPDLOG_DEBUG("grid_dim: {}, {}, {}", grid_dim.x, grid_dim.y,
			     grid_dim.z);
		SPDLOG_DEBUG("block_dim: {}, {}, {}", block_dim.x, block_dim.y,
			     block_dim.z);
	}
	return cuda_launch_kernel_common(impl,
					 impl->original_cuda_launch_kernel_ptsz,
					 func, grid_dim, block_dim, args,
					 shared_mem, stream);
}

extern "C" CUresult cuda_driver_function__cuModuleGetFunction(CUfunction *hfunc,
							      CUmodule hmod,
							      const char *name)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(CUfunction *, CUmodule, const char *);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_module_get_function : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	auto res = original(hfunc, hmod, name);
	if (impl && res == CUDA_SUCCESS && hfunc != nullptr && *hfunc != nullptr &&
	    name != nullptr && name[0] != '\0') {
		impl->record_original_cufunction_name(*hfunc, std::string(name));
		impl->record_original_cufunction_module(*hfunc, hmod);
		if (env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DEBUG_LAUNCH")) {
			const char *filter = std::getenv("BPFTIME_CUDA_SASS_DETOUR_FILTER");
			const std::string_view filter_sv =
				(filter && *filter) ? std::string_view(filter)
						    : std::string_view {};
			const std::string_view name_sv(name);
			if ((filter_sv.empty() && name_sv.find("flash") != std::string_view::npos) ||
			    (!filter_sv.empty() &&
			     name_sv.find(filter_sv) != std::string_view::npos)) {
				SPDLOG_INFO(
					"SASS detour: cuModuleGetFunction mod=0x{:x} -> func=0x{:x} name='{}'",
					(uint64_t)hmod, (uint64_t)*hfunc, name);
			}
		}
	}
	return res;
}

extern "C" CUresult cuda_driver_function__cuLaunchKernel(
	CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
	unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
	unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
	void **kernelParams, void **extra)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = decltype(&cuLaunchKernel);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_launch_kernel : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	std::string kernel_name = "unknown";
	CUfunction original_f = f;
	if (impl) {
		if (auto n = cuda_graph_maybe_get_kernel_name_from_cufunction(*impl, f);
		    n) {
			kernel_name = *n;
		}
		// Prefer a cached patched CUfunction (PTX rewrite or previous on-demand
		// patch). For identify-closure, patch-all fallback is an intermediate
		// state and should only be applied within the closure path for the
		// matching kernel filter.
		if (auto pe = impl->find_patched_kernel_entry(kernel_name); pe &&
		    pe->function != nullptr) {
			const bool identify_enabled = sass_identify_closure_enabled();
			const char *filter = std::getenv("BPFTIME_CUDA_SASS_DETOUR_FILTER");
			const std::string_view filter_sv =
				(filter && *filter) ? std::string_view(filter) : std::string_view {};
			const bool filter_hit =
				(!filter_sv.empty() &&
				 kernel_name.find(std::string(filter_sv)) != std::string::npos);
			const bool can_apply_cache =
				(pe->kind != nv_attach_impl::PatchedKernelKind::PatchAllFallback) ||
				(!identify_enabled) || (!filter_hit);
			if (can_apply_cache)
				f = pe->function;
		}
		// JIT-link PTX instrumentation needs a launch-time binding step:
		// bind injected globals (out_ptr/cap/gate) to bpftime's sampler buffer,
		// so PTX writes show up in the normal JSONL dump pipeline.
		maybe_bind_jitlink_ptx_threadmap_globals(*impl, f, kernel_name,
							hStream);
	}

	// Optional: collect candidate kernels for ThreadMap(device) bring-up on real
	// workloads (e.g., vLLM). This is intentionally independent of SASS detour
	// enablement, so we can first discover patchable SM120 kernels without
	// rewriting anything.
	//
	// Output: JSONL at BPFTIME_CUDA_KERNEL_CANDIDATES_PATH, one line per unique
	// kernel name.
	if (impl) {
		static std::mutex cand_mu;
		static std::unordered_set<std::string> seen_names;
		const char *cand_path = std::getenv("BPFTIME_CUDA_KERNEL_CANDIDATES_PATH");
		if (cand_path && *cand_path) {
			bool need_write = false;
			{
				std::lock_guard<std::mutex> g(cand_mu);
				if (seen_names.insert(kernel_name).second)
					need_write = true;
			}
			if (need_write) {
				using cu_func_get_attribute_fn_t =
					CUresult (*)(int *, CUfunction_attribute, CUfunction);
				static cu_func_get_attribute_fn_t cu_func_get_attribute =
					(cu_func_get_attribute_fn_t)dlsym(RTLD_DEFAULT,
									  "cuFuncGetAttribute");
				int regs = -1;
				int binver = -1;
				int ptxver = -1;
				if (cu_func_get_attribute) {
					(void)cu_func_get_attribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS,
								    original_f);
					(void)cu_func_get_attribute(
						&binver, CU_FUNC_ATTRIBUTE_BINARY_VERSION, original_f);
					(void)cu_func_get_attribute(
						&ptxver, CU_FUNC_ATTRIBUTE_PTX_VERSION, original_f);
				}
				const bool capturing =
					cuda_graph_stream_is_capturing((cudaStream_t)hStream);
				bool have_img = false;
				bool img_patched = false;
				uint64_t img_hash = 0;
				if (auto mi = impl->find_cuda_module_image_by_function(original_f);
				    mi) {
					have_img = true;
					img_patched = mi->patched;
					img_hash = mi->image_hash;
				}
				std::ofstream ofs(cand_path, std::ios::out | std::ios::app);
				if (ofs.good()) {
					ofs << "{\"type\":\"kernel_candidate\","
					    << "\"name\":\"" << json_escape_string(kernel_name) << "\","
					    << "\"regs\":" << regs << ","
					    << "\"binary_version\":" << binver << ","
					    << "\"ptx_version\":" << ptxver << ","
					    << "\"grid\":[" << gridDimX << "," << gridDimY << ","
					    << gridDimZ << "],"
					    << "\"block\":[" << blockDimX << "," << blockDimY << ","
					    << blockDimZ << "],"
					    << "\"shared_mem\":" << sharedMemBytes << ","
					    << "\"capturing\":" << (capturing ? "true" : "false") << ","
					    << "\"have_module_image\":"
					    << (have_img ? "true" : "false") << ","
					    << "\"module_patched\":" << (img_patched ? "true" : "false")
					    << ","
					    << "\"module_hash\":\"0x" << std::hex << img_hash << std::dec
					    << "\""
					    << "}\n";
				}
			}
		}
	}

	const bool launch_debug =
		(impl && sass_detour_debug_enabled() &&
		 env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DEBUG_LAUNCH"));
	if (launch_debug) {
		if (auto mi = impl->find_cuda_module_image_by_function(f); mi) {
					SPDLOG_INFO(
						"SASS detour: cuLaunchKernel enter name='{}' module_patched={} module_hash=0x{:x} module_image=0x{:x} size={} api='{}'",
						kernel_name, mi->patched, mi->image_hash, mi->image_ptr,
						mi->image_size, mi->api);
				} else {
					SPDLOG_INFO(
						"SASS detour: cuLaunchKernel enter name='{}' module=<unknown> trace={} good={} path='{}'",
						kernel_name, impl->cuda_launch_trace_is_enabled(),
						impl->cuda_launch_trace_stream_good(),
						impl->cuda_launch_trace_path_copy());
				}
			}
	if (impl && sass_identify_closure_enabled()) {
		const char *filter = std::getenv("BPFTIME_CUDA_SASS_DETOUR_FILTER");
		const std::string_view filter_sv =
			(filter && *filter) ? std::string_view(filter) : std::string_view {};
		if (!filter_sv.empty() &&
		    kernel_name.find(std::string(filter_sv)) != std::string::npos) {
				// CUDA Graph capture: do not arm/dump identify or touch the control
				// header via host memcpys/syncs during capture, otherwise vLLM's
				// capture can be invalidated (cudaErrorStreamCaptureInvalidated).
				const bool capturing =
					cuda_graph_stream_is_capturing((cudaStream_t)hStream);
					bool armed_identify = false;
			if (!capturing &&
			    !std::getenv("BPFTIME_CUDA_SASS_DETOUR_CONTROL_SKIP_ARM")) {
				// If we have a per-kernel resolved func_id, target it directly.
				// Otherwise:
				// - if `learned_func_ids` is empty: arm Identify (single-run closure)
				// - if `learned_func_ids` has exactly one id: target it
					// - if `learned_func_ids` has multiple ids (common for flashattention):
					//   arm Identify to resolve the exact func_id for this kernel_name
					uint32_t target = 0;
						bool need_arm = false;
						{
							std::lock_guard<std::mutex> guard(
								impl->owned_cuda_images_lock);
							auto &fs = impl->sass_detour_filter_state;
						if (auto it =
							    fs.resolved_func_id_by_kernel_name.find(
								    kernel_name);
						    it != fs.resolved_func_id_by_kernel_name.end()) {
							target = it->second;
						} else if (!fs.identify_pending.load(
								   std::memory_order_acquire)) {
							if (fs.learned_func_ids.empty()) {
								need_arm = true;
							} else if (fs.learned_func_ids.size() == 1) {
								target = *fs.learned_func_ids.begin();
								fs.resolved_func_id_by_kernel_name[kernel_name] =
									target;
							} else {
								need_arm = true;
								}
							}
							// Optional: caller already knows the precise func_id (e.g. from a
							// preceding identify run in the same test harness). Allow forcing
							// the closure state into "targeted" mode so we can skip the
							// patch-all + identify window entirely (more stable for reg255).
							if (auto forced = env_u32_global(
								    "BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE_FORCE_TARGET_FUNC_ID");
							    forced && *forced != 0) {
								target = *forced;
								need_arm = false;
								fs.resolved_func_id_by_kernel_name[kernel_name] = target;
								fs.learned_func_ids.insert(target);
								fs.identify_target_func_id = target;
							}
						if (target != 0 && fs.identify_target_func_id == 0) {
							fs.identify_target_func_id = target;
						}
					}
				// Choose which CUfunction to launch:
				// - Default: ensure we at least have a patch-all fallback so the
				//   identify-only stub exists for this launch (single-run closure).
				// - Optional: if we have a resolved func_id, upgrade to a
				//   func-id-targeted patched function (the “precise detour” steady
				//   state) when explicitly enabled.
				const bool enable_on_demand_upgrade =
					env_truthy_global(
						"BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE_ON_DEMAND_UPGRADE");
				const bool disable_on_demand_patch_all =
					env_truthy_global(
						"BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE_NO_ON_DEMAND_PATCH_ALL");
				const bool enable_on_demand_patch_all =
					!disable_on_demand_patch_all;
				// Best-effort: always prefer an already cached patched function.
					if (auto cached = impl->find_patched_kernel_function(kernel_name);
					    cached) {
						f = *cached;
					} else if (!capturing && enable_on_demand_patch_all) {
						// Single-run closure: if this kernel launch matches the filter but
						// the SM120 code object was loaded earlier in a stripped form (no
						// `.text.<name>` match), load a patch-all fallback module so the
						// identify-only stub exists for this launch.
					if (auto nf = sass_detour_on_demand_patch_all_for_kernel(
						    *impl, original_f, kernel_name);
					    nf) {
						f = *nf;
					}
				}
				// Optional: after identify resolves, allow upgrading from patch-all
				// fallback to a func-id-targeted patch (can be riskier on some real
				// workloads, so keep it opt-in).
				if (!capturing && enable_on_demand_upgrade && target != 0) {
					if (auto nf = sass_detour_on_demand_patch_func_id_for_kernel(
						    *impl, original_f, kernel_name, target);
					    nf) {
						f = *nf;
					}
				}
				if (target != 0) {
					sass_control_set_target(*impl, hStream, target);
				} else if (need_arm) {
					uint32_t image_id = 0;
					if (auto mi0 =
						    impl->find_cuda_module_image_by_function(original_f);
					    mi0) {
						// Prefer the SM120 ELF image_id computed by the load-time
						// patcher (matches BPFTIME_CUDA_SASS_DETOUR_FILTER_IMAGE_ID).
						image_id = mi0->patched ? mi0->base_sm120_image_id
									: mi0->sm120_image_id;
					}
					sass_control_arm_identify(*impl, hStream, kernel_name,
								  image_id);
						armed_identify = true;
					}
				if (std::getenv(
					    "BPFTIME_CUDA_SASS_CONTROL_DEBUG_SYNC_BEFORE_LAUNCH")) {
					auto cuCtxSynchronize =
						reinterpret_cast<CUresult (*)()>(
							impl->original_cu_ctx_synchronize);
					if (cuCtxSynchronize) {
						const auto r0 = cuCtxSynchronize();
						if (r0 != CUDA_SUCCESS) {
							SPDLOG_WARN(
								"SASS control: pre-launch cuCtxSynchronize failed: {}",
								int(r0));
						}
					}
				}
				}
				// Optional "single-launch" identify: force a sync right after the
				// first matching launch so we can dump slots and learn func_id(s)
				// within the same run (useful when stripped SM120 raw-ELF appears
				// before the fatbin that contains names).
				bool sync_after =
					std::getenv(
						"BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_SYNC_AFTER_LAUNCH") !=
					nullptr;
				// Default closure behavior: when we just armed identify, force a sync
				// right after the matching launch so we can learn func_id(s) within
				// the same run even if the SM120 raw-ELF appears before any fatbin
				// that still carries names.
				if (!sync_after && armed_identify)
					sync_after = true;
				// Optional: non-intrusive kernel GPU duration timing (CUDA events).
				// Only enabled when tracing is enabled (we emit timing records into
				// the same JSONL file), and never during CUDA graph capture.
				CUevent timing_start = nullptr;
				CUevent timing_end = nullptr;
				bool timing_armed = false;
				{
					const bool timing_enabled =
						env_truthy_global("BPFTIME_CUDA_KERNEL_TIMING");
					const bool capturing2 =
						cuda_graph_stream_is_capturing((cudaStream_t)hStream);
					if (timing_enabled && impl &&
					    impl->cuda_launch_trace_is_enabled() && !capturing2) {
						const char *fs = std::getenv(
							"BPFTIME_CUDA_KERNEL_TIMING_FILTERS");
						const char *f0 = std::getenv(
							"BPFTIME_CUDA_KERNEL_TIMING_FILTER");
						const char *f1 = std::getenv(
							"BPFTIME_CUDA_SASS_DETOUR_FILTER");
						const char *f2 = std::getenv(
							"BPFTIME_CUDA_SASS_SAMPLE_FILTER");
						const std::string_view fsv =
							(f0 && *f0) ? std::string_view(f0)
							: (f1 && *f1) ? std::string_view(f1)
							: (f2 && *f2) ? std::string_view(f2)
								      : std::string_view {};
						bool hit = false;
						if (fs && *fs) {
							// Comma/space separated list of substrings.
							std::string s(fs);
							for (char &c : s) {
								if (c == ',' || c == ';')
									c = ' ';
							}
							std::istringstream iss(s);
							std::string tok;
							while (iss >> tok) {
								if (tok.empty())
									continue;
								if (kernel_name.find(tok) !=
								    std::string::npos) {
									hit = true;
									break;
								}
							}
						} else if (!fsv.empty()) {
							hit = (kernel_name.find(std::string(fsv)) !=
							       std::string::npos);
						}
						const uint32_t sample_every =
							env_u32_global(
								"BPFTIME_CUDA_KERNEL_TIMING_SAMPLE_EVERY")
								.value_or(1u);
						static std::atomic<uint64_t> sample_ctr { 0 };
						const uint64_t c =
							sample_ctr.fetch_add(1, std::memory_order_relaxed);
						const bool do_sample =
							(sample_every <= 1u) || ((c % sample_every) == 0u);
						if (hit && do_sample) {
							auto cuEventCreate =
								reinterpret_cast<CUresult (*)(CUevent *,
											      unsigned int)>(
									impl->original_cu_event_create);
							auto cuEventRecord =
								reinterpret_cast<CUresult (*)(CUevent,
											      CUstream)>(
									impl->original_cu_event_record);
							if (cuEventCreate && cuEventRecord) {
								if (cuEventCreate(&timing_start,
										  /*flags=*/0) ==
									    CUDA_SUCCESS &&
								    cuEventCreate(&timing_end,
										  /*flags=*/0) ==
									    CUDA_SUCCESS &&
								    cuEventRecord(timing_start,
										  hStream) ==
									    CUDA_SUCCESS) {
									timing_armed = true;
								}
							}
						}
					}
				}
				auto res = original(f, gridDimX, gridDimY, gridDimZ, blockDimX,
						    blockDimY, blockDimZ, sharedMemBytes, hStream,
						    kernelParams, extra);
				if (impl) {
					const uint64_t launch_seq =
						impl->trace_cuda_kernel_launch(
						kernel_name, (int)gridDimX, (int)gridDimY,
						(int)gridDimZ, (int)blockDimX, (int)blockDimY,
						(int)blockDimZ, (size_t)sharedMemBytes,
						(void *)hStream, kernelParams, extra);
					if (timing_armed && res == CUDA_SUCCESS) {
						auto cuEventRecord =
							reinterpret_cast<CUresult (*)(CUevent, CUstream)>(
								impl->original_cu_event_record);
						auto cuEventDestroy =
							reinterpret_cast<CUresult (*)(CUevent)>(
								impl->original_cu_event_destroy_v2);
						const bool end_ok =
							(cuEventRecord &&
							 cuEventRecord(timing_end, hStream) ==
								 CUDA_SUCCESS);
						const bool enq_ok =
							end_ok &&
							impl->enqueue_cuda_kernel_timing(
								launch_seq, kernel_name, (void *)hStream,
								timing_start, timing_end);
						if (!enq_ok) {
							if (cuEventDestroy && timing_start)
								(void)cuEventDestroy(timing_start);
							if (cuEventDestroy && timing_end)
								(void)cuEventDestroy(timing_end);
						}
					} else if (timing_armed) {
						auto cuEventDestroy =
							reinterpret_cast<CUresult (*)(CUevent)>(
								impl->original_cu_event_destroy_v2);
						if (cuEventDestroy && timing_start)
							(void)cuEventDestroy(timing_start);
						if (cuEventDestroy && timing_end)
							(void)cuEventDestroy(timing_end);
					}
				}
					if (impl && !capturing && sync_after &&
					    res == CUDA_SUCCESS) {
						// Call the original syncs directly to avoid recursion into our
						// wrappers; then manually dump identify + samples.
						//
						// Note: keep this enabled even after identify resolves into
						// "target-only" launches; otherwise we can't validate that the
						// target-mode sampler is actually producing records on real vLLM.
						if (hStream != CUstream() &&
						    impl->original_cu_stream_synchronize) {
							auto cuStreamSynchronize =
								reinterpret_cast<CUresult (*)(CUstream)>(
									impl->original_cu_stream_synchronize);
							(void)cuStreamSynchronize(hStream);
						} else if (impl->original_cu_ctx_synchronize) {
							auto cuCtxSynchronize =
								reinterpret_cast<CUresult (*)()>(
									impl->original_cu_ctx_synchronize);
							(void)cuCtxSynchronize();
						}
						sass_control_maybe_dump_identify(*impl);
						impl->dump_sass_samples_force();
					}
					if (launch_debug) {
						SPDLOG_INFO("SASS detour: cuLaunchKernel exit res={}",
							    (int)res);
					}
					return res;
				}
			}
			// Optional: non-intrusive kernel GPU duration timing (CUDA events).
			CUevent timing_start = nullptr;
			CUevent timing_end = nullptr;
			bool timing_armed = false;
			{
				const bool timing_enabled =
					env_truthy_global("BPFTIME_CUDA_KERNEL_TIMING");
				const bool capturing2 =
					cuda_graph_stream_is_capturing((cudaStream_t)hStream);
				if (timing_enabled && impl && impl->cuda_launch_trace_is_enabled() &&
				    !capturing2) {
					const char *fs = std::getenv(
						"BPFTIME_CUDA_KERNEL_TIMING_FILTERS");
					const char *f0 = std::getenv(
						"BPFTIME_CUDA_KERNEL_TIMING_FILTER");
					const char *f1 =
						std::getenv("BPFTIME_CUDA_SASS_DETOUR_FILTER");
					const char *f2 =
						std::getenv("BPFTIME_CUDA_SASS_SAMPLE_FILTER");
					const std::string_view fsv =
						(f0 && *f0) ? std::string_view(f0)
						: (f1 && *f1) ? std::string_view(f1)
						: (f2 && *f2) ? std::string_view(f2)
							      : std::string_view {};
					bool hit = false;
					if (fs && *fs) {
						std::string s(fs);
						for (char &c : s) {
							if (c == ',' || c == ';')
								c = ' ';
						}
						std::istringstream iss(s);
						std::string tok;
						while (iss >> tok) {
							if (tok.empty())
								continue;
							if (kernel_name.find(tok) !=
							    std::string::npos) {
								hit = true;
								break;
							}
						}
					} else if (!fsv.empty()) {
						hit = (kernel_name.find(std::string(fsv)) !=
						       std::string::npos);
					}
					const uint32_t sample_every =
						env_u32_global(
							"BPFTIME_CUDA_KERNEL_TIMING_SAMPLE_EVERY")
							.value_or(1u);
					static std::atomic<uint64_t> sample_ctr { 0 };
					const uint64_t c =
						sample_ctr.fetch_add(1, std::memory_order_relaxed);
					const bool do_sample =
						(sample_every <= 1u) || ((c % sample_every) == 0u);
					if (hit && do_sample) {
						auto cuEventCreate =
							reinterpret_cast<CUresult (*)(CUevent *,
										      unsigned int)>(
								impl->original_cu_event_create);
						auto cuEventRecord =
							reinterpret_cast<CUresult (*)(CUevent, CUstream)>(
								impl->original_cu_event_record);
						if (cuEventCreate && cuEventRecord) {
							if (cuEventCreate(&timing_start, /*flags=*/0) ==
								    CUDA_SUCCESS &&
							    cuEventCreate(&timing_end, /*flags=*/0) ==
								    CUDA_SUCCESS &&
							    cuEventRecord(timing_start, hStream) ==
								    CUDA_SUCCESS) {
								timing_armed = true;
							}
						}
					}
				}
			}
			auto res = original(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
					    blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
			if (impl) {
				const uint64_t launch_seq = impl->trace_cuda_kernel_launch(
					kernel_name, (int)gridDimX, (int)gridDimY, (int)gridDimZ,
					(int)blockDimX, (int)blockDimY, (int)blockDimZ,
					(size_t)sharedMemBytes, (void *)hStream, kernelParams, extra);
				if (timing_armed && res == CUDA_SUCCESS) {
					auto cuEventRecord =
						reinterpret_cast<CUresult (*)(CUevent, CUstream)>(
							impl->original_cu_event_record);
					auto cuEventDestroy =
						reinterpret_cast<CUresult (*)(CUevent)>(
							impl->original_cu_event_destroy_v2);
					const bool end_ok =
						(cuEventRecord &&
						 cuEventRecord(timing_end, hStream) == CUDA_SUCCESS);
					const bool enq_ok =
						end_ok &&
						impl->enqueue_cuda_kernel_timing(
							launch_seq, kernel_name, (void *)hStream,
							timing_start, timing_end);
					if (!enq_ok) {
						if (cuEventDestroy && timing_start)
							(void)cuEventDestroy(timing_start);
						if (cuEventDestroy && timing_end)
							(void)cuEventDestroy(timing_end);
					}
				} else if (timing_armed) {
					auto cuEventDestroy =
						reinterpret_cast<CUresult (*)(CUevent)>(
							impl->original_cu_event_destroy_v2);
					if (cuEventDestroy && timing_start)
						(void)cuEventDestroy(timing_start);
					if (cuEventDestroy && timing_end)
						(void)cuEventDestroy(timing_end);
				}
			}
			// Sampling bring-up helper: in many real workloads (including vLLM eager),
			// kernels are launched asynchronously and there may be no immediate sync
			// point to trigger dump-on-sync. When enabled, force a one-time sync
			// after the first matching sampled launch, then dump samples.
			//
			// This is opt-in and intended for debugging / bring-up; do NOT enable it
			// in production as it introduces a global synchronization.
			if (impl && res == CUDA_SUCCESS &&
			    env_truthy_global("BPFTIME_CUDA_SASS_SAMPLE_SYNC_AFTER_LAUNCH")) {
				const bool capturing =
					cuda_graph_stream_is_capturing((cudaStream_t)hStream);
				bool do_sync = false;
					{
						std::lock_guard<std::mutex> guard(impl->sass_sampling.lock);
						if (impl->sass_sampling.enabled &&
						    impl->sass_sampling.initialized &&
						    !impl->sass_sampling.dumped_on_first_launch) {
							// Prefer matching against kernels we have actually instrumented
							// (tag_to_kernel), to avoid triggering the one-shot sync+dump too
							// early during vLLM/PyTorch initialization (before code objects are
							// loaded/patched).
							bool match = false;
							if (!impl->sass_sampling.tag_to_kernel.empty()) {
								for (const auto &kv :
								     impl->sass_sampling.tag_to_kernel) {
									const auto &k = kv.second;
									if (k.empty())
										continue;
									if (kernel_name.find(k) != std::string::npos ||
									    k.find(kernel_name) != std::string::npos) {
										match = true;
										break;
									}
								}
							}

							const bool require_sampled =
								env_truthy_global(
									"BPFTIME_CUDA_SASS_SAMPLE_SYNC_AFTER_LAUNCH_REQUIRE_SAMPLED");
							if (!match && !require_sampled) {
							const char *sample_filter =
								std::getenv("BPFTIME_CUDA_SASS_SAMPLE_FILTER");
							const char *detour_filter =
								std::getenv("BPFTIME_CUDA_SASS_DETOUR_FILTER");
							const std::string_view filter_sv =
							(sample_filter && *sample_filter)
								? std::string_view(sample_filter)
							: (detour_filter && *detour_filter)
									? std::string_view(detour_filter)
									: std::string_view {};
							if (!filter_sv.empty() &&
							    kernel_name.find(std::string(filter_sv)) !=
								    std::string::npos) {
									match = true;
							}
							}
							if (match) {
								impl->sass_sampling.dumped_on_first_launch = true;
								do_sync = !capturing;
							}
						}
							}
						if (do_sync) {
							if (hStream != CUstream() &&
							    impl->original_cu_stream_synchronize) {
						auto cuStreamSynchronize =
							reinterpret_cast<CUresult (*)(CUstream)>(
								impl->original_cu_stream_synchronize);
						(void)cuStreamSynchronize(hStream);
					} else if (impl->original_cu_ctx_synchronize) {
						auto cuCtxSynchronize =
							reinterpret_cast<CUresult (*)()>(
								impl->original_cu_ctx_synchronize);
								(void)cuCtxSynchronize();
							}
						// This path is explicitly synchronized after a matching sampled
						// launch, so we should dump regardless of the generic
						// dump-on-sync attempt budgeting (which can be consumed by early
						// syncs during vLLM/PyTorch initialization).
						impl->dump_sass_samples_force();
					}
				}
				if (launch_debug) {
					SPDLOG_INFO("SASS detour: cuLaunchKernel exit res={}", (int)res);
			}
			return res;
		}

extern "C" CUresult cuda_driver_function__cuMemcpyHtoD_v2(CUdeviceptr dstDevice,
							  const void *srcHost,
							  size_t ByteCount)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(CUdeviceptr, const void *, size_t);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_memcpy_htod : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	auto res = original(dstDevice, srcHost, ByteCount);
	if (impl) {
		impl->trace_cuda_memcpy("HtoD", ByteCount, (void *)dstDevice,
					(void *)srcHost, nullptr, (int)res,
					false);
	}
	return res;
}

extern "C" CUresult
cuda_driver_function__cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice,
					  const void *srcHost,
					  size_t ByteCount, CUstream hStream)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(CUdeviceptr, const void *, size_t, CUstream);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_memcpy_htod_async : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	auto res = original(dstDevice, srcHost, ByteCount, hStream);
	if (impl) {
		impl->trace_cuda_memcpy("HtoDAsync", ByteCount,
					(void *)dstDevice, (void *)srcHost,
					(void *)hStream, (int)res, true);
	}
	return res;
}

extern "C" CUresult cuda_driver_function__cuMemcpyDtoH_v2(void *dstHost,
							  CUdeviceptr srcDevice,
							  size_t ByteCount)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(void *, CUdeviceptr, size_t);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_memcpy_dtoh : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	auto res = original(dstHost, srcDevice, ByteCount);
	if (impl) {
		impl->trace_cuda_memcpy("DtoH", ByteCount, (void *)dstHost,
					(void *)srcDevice, nullptr, (int)res,
					false);
	}
	return res;
}

extern "C" CUresult
cuda_driver_function__cuMemcpyDtoHAsync_v2(void *dstHost,
					  CUdeviceptr srcDevice,
					  size_t ByteCount, CUstream hStream)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(void *, CUdeviceptr, size_t, CUstream);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_memcpy_dtoh_async : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	auto res = original(dstHost, srcDevice, ByteCount, hStream);
	if (impl) {
		impl->trace_cuda_memcpy("DtoHAsync", ByteCount, (void *)dstHost,
					(void *)srcDevice, (void *)hStream,
					(int)res, true);
	}
	return res;
}

extern "C" CUresult cuda_driver_function__cuMemcpyDtoD_v2(CUdeviceptr dstDevice,
							  CUdeviceptr srcDevice,
							  size_t ByteCount)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(CUdeviceptr, CUdeviceptr, size_t);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_memcpy_dtod : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	auto res = original(dstDevice, srcDevice, ByteCount);
	if (impl) {
		impl->trace_cuda_memcpy("DtoD", ByteCount, (void *)dstDevice,
					(void *)srcDevice, nullptr, (int)res,
					false);
	}
	return res;
}

extern "C" CUresult
cuda_driver_function__cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice,
					  CUdeviceptr srcDevice,
					  size_t ByteCount, CUstream hStream)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_memcpy_dtod_async : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	auto res = original(dstDevice, srcDevice, ByteCount, hStream);
	if (impl) {
		impl->trace_cuda_memcpy("DtoDAsync", ByteCount, (void *)dstDevice,
					(void *)srcDevice, (void *)hStream,
					(int)res, true);
	}
	return res;
}

extern "C" CUresult cuda_driver_function__cuMemsetD8Async(CUdeviceptr dstDevice,
							  unsigned char uc,
							  size_t N,
							  CUstream hStream)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(CUdeviceptr, unsigned char, size_t, CUstream);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_memset_d8_async : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	auto res = original(dstDevice, uc, N, hStream);
	if (impl) {
		impl->trace_cuda_memset("D8Async", N, (void *)dstDevice, uc,
					(void *)hStream, (int)res, true);
	}
	return res;
}

extern "C" CUresult
cuda_driver_function__cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui,
				       size_t N, CUstream hStream)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(CUdeviceptr, unsigned int, size_t, CUstream);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_memset_d32_async : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	auto res = original(dstDevice, ui, N, hStream);
	if (impl) {
		impl->trace_cuda_memset("D32Async", N * 4ULL,
					(void *)dstDevice, ui, (void *)hStream,
					(int)res, true);
	}
	return res;
}

extern "C" CUresult cuda_driver_function__cuMemAlloc_v2(CUdeviceptr *dptr,
							size_t bytesize)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(CUdeviceptr *, size_t);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_mem_alloc : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	auto res = original(dptr, bytesize);
	if (impl) {
		void *ptr = (dptr && res == CUDA_SUCCESS) ? (void *)(*dptr)
							  : nullptr;
		impl->trace_cuda_alloc("cuMemAlloc_v2", bytesize, ptr, nullptr,
				       (int)res);
	}
	return res;
}

extern "C" CUresult cuda_driver_function__cuMemFree_v2(CUdeviceptr dptr)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(CUdeviceptr);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_mem_free : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	auto res = original(dptr);
	if (impl) {
		impl->trace_cuda_free("cuMemFree_v2", (void *)dptr, nullptr,
				      (int)res);
	}
	return res;
}

extern "C" CUresult cuda_driver_function__cuMemAllocAsync(CUdeviceptr *dptr,
							  size_t bytesize,
							  CUstream hStream)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(CUdeviceptr *, size_t, CUstream);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_mem_alloc_async : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	auto res = original(dptr, bytesize, hStream);
	if (impl) {
		void *ptr = (dptr && res == CUDA_SUCCESS) ? (void *)(*dptr)
							  : nullptr;
		impl->trace_cuda_alloc("cuMemAllocAsync", bytesize, ptr,
				       (void *)hStream, (int)res);
	}
	return res;
}

extern "C" CUresult cuda_driver_function__cuMemFreeAsync(CUdeviceptr dptr,
							 CUstream hStream)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(CUdeviceptr, CUstream);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_mem_free_async : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	auto res = original(dptr, hStream);
	if (impl) {
		impl->trace_cuda_free("cuMemFreeAsync", (void *)dptr,
				      (void *)hStream, (int)res);
	}
	return res;
}

static uint64_t now_ns()
{
	timespec ts {};
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return static_cast<uint64_t>(ts.tv_sec) * 1000ULL * 1000ULL * 1000ULL +
	       static_cast<uint64_t>(ts.tv_nsec);
}

extern "C" CUresult
cuda_driver_function__cuStreamSynchronize(CUstream hStream)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(CUstream);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_stream_synchronize : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	const uint64_t start = now_ns();
	auto res = original(hStream);
	const uint64_t dur = now_ns() - start;
	if (impl) {
		impl->trace_cuda_sync("cuStreamSynchronize", (void *)hStream,
				      dur, (int)res);
		sass_control_maybe_dump_identify(*impl);
		impl->maybe_dump_sass_samples_on_sync();
	}
	return res;
}

extern "C" CUresult cuda_driver_function__cuCtxSynchronize()
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)();
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_ctx_synchronize : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	const uint64_t start = now_ns();
	auto res = original();
	const uint64_t dur = now_ns() - start;
	if (impl) {
		impl->trace_cuda_sync("cuCtxSynchronize", nullptr, dur,
				      (int)res);
		sass_control_maybe_dump_identify(*impl);
		impl->maybe_dump_sass_samples_on_sync();
	}
	return res;
}

extern "C" CUresult cuda_driver_function__cuCtxDestroy(CUcontext ctx)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(CUcontext);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_ctx_destroy : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;

	// Avoid nested CUDA API calls during context teardown by default: some
	// drivers/frameworks can misbehave if we call `cuCtxSynchronize/cuMemcpy*`
	// from inside `cuCtxDestroy*`. Prefer dump-on-sync in normal execution.
	if (impl && env_truthy_global("BPFTIME_CUDA_SASS_SAMPLE_DUMP_ON_CTX_DESTROY")) {
		sass_control_maybe_dump_identify(*impl);
		impl->dump_sass_samples_force();
	}
	return original(ctx);
}

extern "C" CUresult cuda_driver_function__cuCtxDestroy_v2(CUcontext ctx)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(CUcontext);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_ctx_destroy_v2 : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;

	if (impl && env_truthy_global("BPFTIME_CUDA_SASS_SAMPLE_DUMP_ON_CTX_DESTROY")) {
		sass_control_maybe_dump_identify(*impl);
		impl->dump_sass_samples_force();
	}

	return original(ctx);
}

extern "C" CUresult cuda_driver_function__cuDevicePrimaryCtxRelease(CUdevice dev)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(CUdevice);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_device_primary_ctx_release : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;

	if (impl && env_truthy_global("BPFTIME_CUDA_SASS_SAMPLE_DUMP_ON_CTX_DESTROY")) {
		sass_control_maybe_dump_identify(*impl);
		impl->dump_sass_samples_force();
	}
	return original(dev);
}

extern "C" CUresult
cuda_driver_function__cuDevicePrimaryCtxRelease_v2(CUdevice dev)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(CUdevice);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_device_primary_ctx_release_v2 : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;

	if (impl && env_truthy_global("BPFTIME_CUDA_SASS_SAMPLE_DUMP_ON_CTX_DESTROY")) {
		sass_control_maybe_dump_identify(*impl);
		impl->dump_sass_samples_force();
	}
	return original(dev);
}

extern "C" CUresult cuda_driver_function__cuEventRecord(CUevent hEvent,
							CUstream hStream)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(CUevent, CUstream);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_event_record : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	auto res = original(hEvent, hStream);
	if (impl) {
		impl->trace_cuda_event_record((void *)hEvent, (void *)hStream,
					      (int)res);
	}
	return res;
}

extern "C" CUresult cuda_driver_function__cuEventSynchronize(CUevent hEvent)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(CUevent);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_event_synchronize : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	const uint64_t start = now_ns();
	auto res = original(hEvent);
	const uint64_t dur = now_ns() - start;
	if (impl) {
		impl->trace_cuda_sync("cuEventSynchronize", (void *)hEvent, dur,
				      (int)res);
	}
	return res;
}

extern "C" CUresult cuda_driver_function__cuGraphLaunch(CUgraphExec hGraphExec,
							CUstream hStream)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(CUgraphExec, CUstream);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_graph_launch : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	auto res = original(hGraphExec, hStream);
	if (impl) {
		impl->trace_cuda_graph_launch((void *)hGraphExec,
					      (void *)hStream, (int)res);
	}
	return res;
}

static CUresult culink_add_data_common(
	nv_attach_impl *impl, void *original_fn, CUlinkState state,
	CUjitInputType type, void *data, size_t size, const char *name,
	unsigned int numOptions, CUjit_option *options, void **optionValues)
{
	using fn_t = CUresult (*)(CUlinkState, CUjitInputType, void *, size_t,
				  const char *, unsigned int, CUjit_option *,
				  void **);
	auto original = reinterpret_cast<fn_t>(original_fn);
	if (!original)
		return CUDA_ERROR_UNKNOWN;

	// Inject at the real driver JIT-link entry to cover vLLM/Triton/NVRTC paths:
	//   cuLinkCreate -> cuLinkAddData(PTX/...) -> cuLinkComplete -> cuModuleLoad*
	//
	// This complements the "fatbin parse -> bpftime fallback cuLink" path.
	const bool jitlink_ptx_enabled =
		env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_JITLINK_PTX") ||
		(sass_identify_closure_enabled() &&
		 !env_truthy_global(
			 "BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE_NO_JITLINK_PTX_DEFAULT"));
	const bool inject_ptx_marker =
		env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_JITLINK_PTX_MARKER");
	const bool inject_ptx_threadmap =
		env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_JITLINK_PTX_THREADMAP");
	if (!jitlink_ptx_enabled || (!inject_ptx_marker && !inject_ptx_threadmap) ||
	    state == nullptr || data == nullptr || size == 0 ||
	    type != CU_JIT_INPUT_PTX || impl == nullptr) {
		return original(state, type, data, size, name, numOptions, options,
				optionValues);
	}

	const char *filter0 = std::getenv("BPFTIME_CUDA_SASS_DETOUR_FILTER");
	const char *filter1 = std::getenv("BPFTIME_CUDA_SASS_SAMPLE_FILTER");
	const std::string_view filter_sv =
		(filter0 && *filter0) ? std::string_view(filter0)
		: (filter1 && *filter1) ? std::string_view(filter1)
					: std::string_view {};
	if (filter_sv.empty())
		return original(state, type, data, size, name, numOptions, options,
				optionValues);

	const size_t max_ptx_bytes = []() -> size_t {
		if (auto v = env_u32_global(
			    "BPFTIME_CUDA_SASS_DETOUR_JITLINK_MAX_PTX_BYTES"))
			return size_t(*v);
		return size_t(20u) << 20; // 20 MiB
	}();

	const char *p = reinterpret_cast<const char *>(data);
	const size_t cap = size;
	const size_t len = strnlen(p, cap);
	if (len < 8 || len + 1 > cap || (len + 1) > max_ptx_bytes) {
		return original(state, type, data, size, name, numOptions, options,
				optionValues);
	}
	const std::string_view ptx_sv(p, len);
	if (ptx_sv.find(filter_sv) == std::string_view::npos) {
		return original(state, type, data, size, name, numOptions, options,
				optionValues);
	}

	std::string s(p, len);
	// Reuse the same minimal PTX-level injection logic as the fallback path:
	// - add globals
	// - insert a snippet at the first real instruction in the matching `.entry`
	{
		const bool have_marker =
			(s.find("__bpftime_ptx_marker") != std::string::npos);
		const bool have_threadmap =
			(s.find("__bpftime_ptx_out_ptr") != std::string::npos);
		if (!((inject_ptx_marker && !have_marker) ||
		      (inject_ptx_threadmap && !have_threadmap))) {
			return original(state, type, data, size, name, numOptions,
					options, optionValues);
		}

		// 1) Decls
		{
			std::string decl;
			if (inject_ptx_marker && !have_marker) {
				decl +=
					"\n// bpftime jitlink PTX marker\n"
					".visible .global .align 4 .u32 __bpftime_ptx_marker;\n";
			}
			if (inject_ptx_threadmap && !have_threadmap) {
				decl +=
					"\n// bpftime jitlink PTX threadmap (bpftime sampler-bound)\n"
					".visible .global .align 8 .u64 __bpftime_ptx_out_ptr;\n"
					".visible .global .align 4 .u32 __bpftime_ptx_out_cap;\n"
					".visible .global .align 4 .u32 __bpftime_ptx_out_gate;\n";
			}
			auto insert_after_directive =
				[&](const char *needle) -> std::optional<size_t> {
				const size_t scan_cap = std::min<size_t>(s.size(), 8192);
				const size_t pos = s.substr(0, scan_cap).find(needle);
				if (pos == std::string::npos)
					return std::nullopt;
				const size_t end = s.find('\n', pos);
				if (end == std::string::npos)
					return std::nullopt;
				return end + 1;
			};
			std::optional<size_t> ins =
				insert_after_directive(".address_size");
			if (!ins)
				ins = insert_after_directive(".target");
			if (ins && !decl.empty())
				s.insert(*ins, decl);
		}

		// 2) Insert snippet into matching `.entry`
		{
			const std::string_view key = ".entry";
			size_t pos = 0;
			while (true) {
				pos = s.find(std::string(key), pos);
				if (pos == std::string::npos)
					break;
				const size_t name_start = pos + key.size();
				size_t i = name_start;
				while (i < s.size() && (s[i] == ' ' || s[i] == '\t'))
					i++;
				if (i >= s.size())
					break;
				size_t j = i;
				while (j < s.size()) {
					const char ch = s[j];
					if (ch == '(' || ch == ' ' || ch == '\t' ||
					    ch == '\r' || ch == '\n')
						break;
					j++;
				}
				if (j <= i) {
					pos = pos + key.size();
					continue;
				}
				const std::string_view entry_name(
					s.data() + i, j - i);
				if (entry_name.find(filter_sv) ==
				    std::string_view::npos) {
					pos = j;
					continue;
				}
				const size_t brace = s.find('{', j);
				if (brace == std::string::npos)
					break;
				// find first real instruction line
				size_t scan = brace + 1;
				while (scan < s.size()) {
					const size_t line_end = s.find('\n', scan);
					const size_t end = (line_end == std::string::npos)
								   ? s.size()
								   : line_end;
					std::string_view line(s.data() + scan, end - scan);
					auto trim_left = [](std::string_view x) {
						size_t k = 0;
						while (k < x.size()) {
							const char ch = x[k];
							if (ch != ' ' && ch != '\t' &&
							    ch != '\r')
								break;
							k++;
						}
						return x.substr(k);
					};
					const auto t = trim_left(line);
					const bool is_empty = t.empty() || t == "}";
					const bool is_comment =
						t.starts_with("//") || t.starts_with("/*");
					const bool is_directive =
						(!t.empty() && t[0] == '.');
					if (!is_empty && !is_comment && !is_directive) {
						std::string snippet;
						if (inject_ptx_marker && !have_marker) {
							snippet +=
								"\n\t// bpftime jitlink marker (PTX-level)\n"
								"\t.reg .b32 %bpftime_r<1>;\n"
								"\tmov.u32 %bpftime_r0, 1;\n"
								"\tst.global.u32 [__bpftime_ptx_marker], %bpftime_r0;\n";
						}
						if (inject_ptx_threadmap && !have_threadmap) {
							snippet +=
								"\n\t// bpftime jitlink threadmap (PTX-level, sampler-bound)\n"
								"\t.reg .pred %bpftime_p<6>;\n"
								"\t.reg .b32 %bpftime_t<6>;\n"
								"\t.reg .b64 %bpftime_rd<4>;\n"
								"\tld.global.u64 %bpftime_rd0, [__bpftime_ptx_out_ptr];\n"
								"\tsetp.eq.u64 %bpftime_p0, %bpftime_rd0, 0;\n"
								"\t@%bpftime_p0 bra $__bpftime_tm_skip;\n"
								"\tld.global.u32 %bpftime_t0, [__bpftime_ptx_out_cap];\n"
								"\tld.global.u32 %bpftime_t1, [__bpftime_ptx_out_gate];\n"
								"\t// gate bit2: CTA0-only (clamp)\n"
								"\tand.b32 %bpftime_t2, %bpftime_t1, 4;\n"
								"\tsetp.ne.u32 %bpftime_p1, %bpftime_t2, 0;\n"
								"\t@%bpftime_p1 bra $__bpftime_tm_check_cta;\n"
								"\tbra $__bpftime_tm_after_cta;\n"
								"$__bpftime_tm_check_cta:\n"
								"\tmov.u32 %bpftime_t3, %ctaid.x;\n"
								"\tsetp.ne.u32 %bpftime_p2, %bpftime_t3, 0;\n"
								"\t@%bpftime_p2 bra $__bpftime_tm_skip;\n"
								"\tmov.u32 %bpftime_t3, %ctaid.y;\n"
								"\tsetp.ne.u32 %bpftime_p2, %bpftime_t3, 0;\n"
								"\t@%bpftime_p2 bra $__bpftime_tm_skip;\n"
								"\tmov.u32 %bpftime_t3, %ctaid.z;\n"
								"\tsetp.ne.u32 %bpftime_p2, %bpftime_t3, 0;\n"
								"\t@%bpftime_p2 bra $__bpftime_tm_skip;\n"
								"$__bpftime_tm_after_cta:\n"
								"\tmov.u32 %bpftime_t2, %tid.x;\n"
								"\tsetp.ge.u32 %bpftime_p3, %bpftime_t2, %bpftime_t0;\n"
								"\t@%bpftime_p3 bra $__bpftime_tm_skip;\n"
								"\t// gate bit0: lane0-only\n"
								"\tand.b32 %bpftime_t3, %bpftime_t1, 1;\n"
								"\tsetp.ne.u32 %bpftime_p4, %bpftime_t3, 0;\n"
								"\t@%bpftime_p4 bra $__bpftime_tm_check_lane;\n"
								"\tbra $__bpftime_tm_after_lane;\n"
								"$__bpftime_tm_check_lane:\n"
								"\tmov.u32 %bpftime_t4, %laneid;\n"
								"\tsetp.ne.u32 %bpftime_p5, %bpftime_t4, 0;\n"
								"\t@%bpftime_p5 bra $__bpftime_tm_skip;\n"
								"$__bpftime_tm_after_lane:\n"
								"\t// gate bit1: warp0-only\n"
								"\tand.b32 %bpftime_t3, %bpftime_t1, 2;\n"
								"\tsetp.ne.u32 %bpftime_p4, %bpftime_t3, 0;\n"
								"\t@%bpftime_p4 bra $__bpftime_tm_check_warp;\n"
								"\tbra $__bpftime_tm_after_warp;\n"
								"$__bpftime_tm_check_warp:\n"
								"\tshr.u32 %bpftime_t4, %bpftime_t2, 5;\n"
								"\tsetp.ne.u32 %bpftime_p5, %bpftime_t4, 0;\n"
								"\t@%bpftime_p5 bra $__bpftime_tm_skip;\n"
								"$__bpftime_tm_after_warp:\n"
								"\tmul.wide.u32 %bpftime_rd1, %bpftime_t2, 4;\n"
								"\tadd.u64 %bpftime_rd2, %bpftime_rd0, %bpftime_rd1;\n"
								"\tcvta.to.global.u64 %bpftime_rd3, %bpftime_rd2;\n"
								"\tmov.u32 %bpftime_t5, %smid;\n"
								"\tst.global.u32 [%bpftime_rd3], %bpftime_t5;\n"
								"$__bpftime_tm_skip:\n";
						}
						if (!snippet.empty())
							s.insert(scan, snippet);
						break;
					}
					if (line_end == std::string::npos)
						break;
					scan = line_end + 1;
				}
				break;
			}
		}
	}

	std::vector<uint8_t> owned;
	owned.assign(s.begin(), s.end());
	owned.push_back(0);
	const void *data_to_pass = nullptr;
	size_t size_to_pass = 0;
	impl->culink_track_owned_input(state, std::move(owned), &data_to_pass,
				       &size_to_pass);
	if (data_to_pass != nullptr && size_to_pass != 0) {
		return original(state, type, const_cast<void *>(data_to_pass),
				size_to_pass, name, numOptions, options,
				optionValues);
	}

	return original(state, type, data, size, name, numOptions, options,
			optionValues);
}

extern "C" CUresult cuda_driver_function__cuLinkAddData(CUlinkState state,
							CUjitInputType type,
							void *data, size_t size,
							const char *name,
							unsigned int numOptions,
							CUjit_option *options,
							void **optionValues)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	return culink_add_data_common(impl, impl ? impl->original_cu_link_add_data : nullptr,
				      state, type, data, size, name, numOptions,
				      options, optionValues);
}

extern "C" CUresult cuda_driver_function__cuLinkAddData_v2(CUlinkState state,
							   CUjitInputType type,
							   void *data, size_t size,
							   const char *name,
							   unsigned int numOptions,
							   CUjit_option *options,
							   void **optionValues)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	return culink_add_data_common(
		impl, impl ? impl->original_cu_link_add_data_v2 : nullptr, state, type,
		data, size, name, numOptions, options, optionValues);
}

extern "C" CUresult cuda_driver_function__cuLinkDestroy(CUlinkState state)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(CUlinkState);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_link_destroy : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	auto res = original(state);
	if (impl) {
		impl->culink_release_state(state);
	}
	return res;
}

extern "C" CUresult cuda_driver_function__cuLinkComplete(CUlinkState state,
							 void **cubinOut,
							 size_t *sizeOut)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(CUlinkState, void **, size_t *);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_link_complete : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	auto res = original(state, cubinOut, sizeOut);

	if (impl && (int)res == 0 && cubinOut != nullptr && sizeOut != nullptr &&
	    *cubinOut != nullptr && *sizeOut != 0 &&
	    env_truthy_global("BPFTIME_CUDA_SASS_DETOUR")) {
		size_t patched_size = *sizeOut;
		const void *patched = maybe_patch_cuda_image_sass_detour(
			impl, *cubinOut, *sizeOut, "cuLinkComplete", &patched_size);
		if (patched != *cubinOut) {
			*cubinOut = const_cast<void *>(patched);
			*sizeOut = patched_size;
		}
	}

	return res;
}

extern "C" CUresult cuda_driver_function__cuStreamCreate(CUstream *phStream,
							 unsigned int Flags)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(CUstream *, unsigned int);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_stream_create : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	auto res = original(phStream, Flags);
	if (impl) {
		void *stream = nullptr;
		if (phStream != nullptr)
			stream = (void *)(*phStream);
		impl->trace_cuda_stream_create("cuStreamCreate", stream, Flags,
					       0, (int)res);
	}
	return res;
}

extern "C" CUresult
cuda_driver_function__cuStreamCreateWithPriority(CUstream *phStream,
						 unsigned int Flags,
						 int priority)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(CUstream *, unsigned int, int);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_stream_create_with_priority : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	auto res = original(phStream, Flags, priority);
	if (impl) {
		void *stream = nullptr;
		if (phStream != nullptr)
			stream = (void *)(*phStream);
		impl->trace_cuda_stream_create("cuStreamCreateWithPriority",
					       stream, Flags, priority,
					       (int)res);
	}
	return res;
}

extern "C" CUresult cuda_driver_function__cuStreamDestroy_v2(CUstream hStream)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(CUstream);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_stream_destroy_v2 : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	auto res = original(hStream);
	if (impl) {
		impl->trace_cuda_stream_destroy((void *)hStream, (int)res);
	}
	return res;
}

extern "C" CUresult cuda_driver_function__cuStreamWaitEvent(CUstream hStream,
							   CUevent hEvent,
							   unsigned int Flags)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(CUstream, CUevent, unsigned int);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_stream_wait_event : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	auto res = original(hStream, hEvent, Flags);
	if (impl) {
		impl->trace_cuda_stream_wait_event((void *)hStream,
						   (void *)hEvent, Flags,
						   (int)res);
	}
	return res;
}

extern "C" CUresult cuda_driver_function__cuEventCreate(CUevent *phEvent,
						       unsigned int Flags)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(CUevent *, unsigned int);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_event_create : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	auto res = original(phEvent, Flags);
	if (impl) {
		void *event = nullptr;
		if (phEvent != nullptr)
			event = (void *)(*phEvent);
		impl->trace_cuda_event_create(event, Flags, (int)res);
	}
	return res;
}

extern "C" CUresult cuda_driver_function__cuEventDestroy_v2(CUevent hEvent)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(CUevent);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_event_destroy_v2 : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	auto res = original(hEvent);
	if (impl) {
		impl->trace_cuda_event_destroy((void *)hEvent, (int)res);
	}
	return res;
}

extern "C" CUresult cuda_driver_function__cuModuleLoadData(CUmodule *module,
								  const void *image)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(CUmodule *, const void *);
		auto original = reinterpret_cast<fn_t>(
			impl ? impl->original_cu_module_load_data : nullptr);
		if (!original)
			return CUDA_ERROR_UNKNOWN;
		const bool detour_debug = env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DEBUG");
		if (impl && detour_debug) {
			SPDLOG_INFO("SASS detour: cuModuleLoadData enter image=0x{:x}",
				    (uint64_t)image);
		}
			size_t effective_size = 0;
			const void *effective_image = image;
			bool effective_detoured = false;
			if (impl && image != nullptr &&
			    env_truthy_global("BPFTIME_CUDA_SASS_DETOUR")) {
				effective_image = maybe_patch_cuda_image_sass_detour(
					impl, image, 0, "cuModuleLoadData", &effective_size);
				effective_detoured = tls_sass_detour_last_output_is_detoured;
				if (impl && detour_debug) {
					SPDLOG_INFO(
						"SASS detour: cuModuleLoadData after patch effective_image=0x{:x}",
						(uint64_t)effective_image);
				}
		}
		if (impl && detour_debug) {
			SPDLOG_INFO("SASS detour: cuModuleLoadData calling original");
		}
			auto res = original(module, effective_image);
			if (impl && detour_debug) {
				SPDLOG_INFO("SASS detour: cuModuleLoadData original returned {}",
					    (int)res);
			}
			if (impl && res == CUDA_SUCCESS && module != nullptr &&
			    *module != nullptr && effective_image != nullptr) {
				const uint64_t h =
					(effective_size == 0)
						? 0
						: fnv1a64(reinterpret_cast<const uint8_t *>(
								  effective_image),
							  effective_size);
				const void *base_img = tls_sass_detour_last_base_image;
				const size_t base_sz = tls_sass_detour_last_base_size;
				const uint64_t hb =
					(base_img != nullptr && base_sz != 0)
						? fnv1a64(reinterpret_cast<const uint8_t *>(
								  base_img),
							  base_sz)
						: 0;
				impl->record_cuda_module_image(
					*module, effective_image, effective_size, h,
					effective_detoured, "cuModuleLoadData", base_img,
					base_sz, hb, tls_sass_detour_last_sm120_image_id,
					tls_sass_detour_last_sm120_image_id);
			}
			if (impl) {
				void *mod = nullptr;
				if (module != nullptr)
					mod = (void *)(*module);
			impl->trace_cuda_module_load("cuModuleLoadData", mod,
					     (void *)image, 0, (int)res);
	}
	return res;
}

extern "C" CUresult cuda_driver_function__cuModuleLoadDataEx(
	CUmodule *module, const void *image, unsigned int numOptions,
	CUjit_option *options, void **optionValues)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(CUmodule *, const void *, unsigned int,
				  CUjit_option *, void **);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_module_load_data_ex : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	size_t effective_size = 0;
	const void *effective_image = image;
	bool effective_detoured = false;
	if (impl && image != nullptr &&
	    env_truthy_global("BPFTIME_CUDA_SASS_DETOUR")) {
		effective_image = maybe_patch_cuda_image_sass_detour(
			impl, image, 0, "cuModuleLoadDataEx", &effective_size);
		effective_detoured = tls_sass_detour_last_output_is_detoured;
	}
	auto res = original(module, effective_image, numOptions, options, optionValues);
	if (impl && res == CUDA_SUCCESS && module != nullptr && *module != nullptr &&
	    effective_image != nullptr) {
		const uint64_t h =
			(effective_size == 0)
				? 0
				: fnv1a64(reinterpret_cast<const uint8_t *>(effective_image),
					  effective_size);
		const void *base_img = tls_sass_detour_last_base_image;
		const size_t base_sz = tls_sass_detour_last_base_size;
		const uint64_t hb =
			(base_img != nullptr && base_sz != 0)
				? fnv1a64(reinterpret_cast<const uint8_t *>(base_img),
					  base_sz)
				: 0;
		impl->record_cuda_module_image(*module, effective_image, effective_size,
					       h, effective_detoured,
					       "cuModuleLoadDataEx", base_img,
					       base_sz, hb,
					       tls_sass_detour_last_sm120_image_id,
					       tls_sass_detour_last_sm120_image_id);
	}
	if (impl) {
		void *mod = nullptr;
		if (module != nullptr)
			mod = (void *)(*module);
		impl->trace_cuda_module_load("cuModuleLoadDataEx", mod,
					     (void *)image, numOptions,
					     (int)res);
	}
	return res;
}

extern "C" CUresult
cuda_driver_function__cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(CUmodule *, const void *);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_module_load_fatbinary : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;

	size_t effective_size = 0;
	const void *effective_code = fatCubin;
	bool effective_detoured = false;
	if (impl && fatCubin != nullptr &&
	    env_truthy_global("BPFTIME_CUDA_SASS_DETOUR")) {
		effective_code = maybe_patch_cuda_image_sass_detour(
			impl, fatCubin, 0, "cuModuleLoadFatBinary", &effective_size);
		effective_detoured = tls_sass_detour_last_output_is_detoured;
	}
	auto res = original(module, effective_code);
	if (impl && res == CUDA_SUCCESS && module != nullptr && *module != nullptr &&
	    effective_code != nullptr) {
		const uint64_t h =
			(effective_size == 0)
				? 0
				: fnv1a64(reinterpret_cast<const uint8_t *>(effective_code),
					  effective_size);
		const void *base_img = tls_sass_detour_last_base_image;
		const size_t base_sz = tls_sass_detour_last_base_size;
		const uint64_t hb =
			(base_img != nullptr && base_sz != 0)
				? fnv1a64(reinterpret_cast<const uint8_t *>(base_img),
					  base_sz)
				: 0;
		impl->record_cuda_module_image(*module, effective_code, effective_size,
					       h, effective_detoured,
					       "cuModuleLoadFatBinary", base_img,
					       base_sz, hb,
					       tls_sass_detour_last_sm120_image_id,
					       tls_sass_detour_last_sm120_image_id);
	}
	if (impl) {
		void *mod = nullptr;
		if (module != nullptr)
			mod = (void *)(*module);
		impl->trace_cuda_module_load("cuModuleLoadFatBinary", mod,
					     (void *)fatCubin, 0, (int)res);
	}
	return res;
}

extern "C" CUresult cuda_driver_function__cuModuleLoad(CUmodule *module,
						       const char *fname)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(gum_ctx);
	using fn_t = CUresult (*)(CUmodule *, const char *);
	auto original =
		reinterpret_cast<fn_t>(impl ? impl->original_cu_module_load : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;

	if (!impl || fname == nullptr || fname[0] == '\0' ||
	    !env_truthy_global("BPFTIME_CUDA_SASS_DETOUR")) {
		return original(module, fname);
	}

	// Prefer a load-from-memory path so we can patch cubin-only SM120 images
	// even when the application uses `cuModuleLoad(<file>)`.
	using load_data_fn_t = CUresult (*)(CUmodule *, const void *);
	auto load_data = reinterpret_cast<load_data_fn_t>(impl->original_cu_module_load_data);
	if (!load_data) {
		return original(module, fname);
	}

	std::vector<uint8_t> file_bytes;
	try {
		std::ifstream ifs(fname, std::ios::binary);
		if (!ifs.is_open())
			return original(module, fname);
		ifs.seekg(0, std::ios::end);
		const std::streamoff end = ifs.tellg();
		if (end <= 0)
			return original(module, fname);
		if (end > std::streamoff(256ull << 20)) // best-effort cap
			return original(module, fname);
		file_bytes.resize(static_cast<size_t>(end));
		ifs.seekg(0, std::ios::beg);
		if (!ifs.read(reinterpret_cast<char *>(file_bytes.data()),
			      static_cast<std::streamsize>(file_bytes.size()))) {
			return original(module, fname);
		}
	} catch (...) {
		return original(module, fname);
	}

	// Keep the file bytes alive in `owned_cuda_images`, then feed them to the
	// normal detour patcher.
	const void *code_ptr = nullptr;
	size_t code_size = file_bytes.size();
	{
		std::lock_guard<std::mutex> guard(impl->owned_cuda_images_lock);
		nv_attach_impl::owned_cuda_image img;
		img.size = code_size;
		img.data = std::make_unique<uint8_t[]>(img.size);
		std::memcpy(img.data.get(), file_bytes.data(), img.size);
		code_ptr = img.data.get();
		impl->owned_cuda_images.emplace_back(std::move(img));
	}

	size_t effective_size = code_size;
	bool effective_detoured = false;
	const void *effective_code = maybe_patch_cuda_image_sass_detour(
		impl, code_ptr, code_size, "cuModuleLoad(file)", &effective_size);
	effective_detoured = tls_sass_detour_last_output_is_detoured;

	if (env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DEBUG_LAUNCH")) {
		SPDLOG_INFO("SASS detour: cuModuleLoad(file) fname='{}' code=0x{:x} -> 0x{:x}",
			    fname, (uint64_t)code_ptr, (uint64_t)effective_code);
	}

	auto res = load_data(module, effective_code);
	if (impl && res == CUDA_SUCCESS && module != nullptr && *module != nullptr &&
	    effective_code != nullptr) {
		const uint64_t h =
			(effective_size == 0)
				? 0
				: fnv1a64(reinterpret_cast<const uint8_t *>(effective_code),
					  effective_size);
		const void *base_img = tls_sass_detour_last_base_image;
		const size_t base_sz = tls_sass_detour_last_base_size;
		const uint64_t hb =
			(base_img != nullptr && base_sz != 0)
				? fnv1a64(reinterpret_cast<const uint8_t *>(base_img),
					  base_sz)
				: 0;
		impl->record_cuda_module_image(*module, effective_code, effective_size,
					       h, effective_detoured,
					       "cuModuleLoad(file)", base_img,
					       base_sz, hb,
					       tls_sass_detour_last_sm120_image_id,
					       tls_sass_detour_last_sm120_image_id);
	}
	if (impl) {
		void *mod = nullptr;
		if (module != nullptr)
			mod = (void *)(*module);
		impl->trace_cuda_module_load("cuModuleLoad(file)", mod,
					     (void *)code_ptr, 0, (int)res);
	}
	return res;
}

extern "C" CUresult cuda_driver_function__cuModuleUnload(CUmodule hmod)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(CUmodule);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_module_unload : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	auto res = original(hmod);
	if (impl) {
		impl->trace_cuda_module_unload((void *)hmod, (int)res);
	}
	return res;
}

extern "C" CUresult cuda_driver_function__cuLibraryLoadData(
	CUlibrary *library, const void *code, CUjit_option *jitOptions,
	void **jitOptionsValues, unsigned int numJitOptions,
	CUlibraryOption *libraryOptions, void **libraryOptionValues,
	unsigned int numLibraryOptions)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(CUlibrary *, const void *, CUjit_option *,
				  void **, unsigned int, CUlibraryOption *,
				  void **, unsigned int);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_library_load_data : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;

	const void *effective_code = code;
	size_t effective_size = 0;
	bool effective_detoured = false;
	if (impl && code != nullptr &&
	    env_truthy_global("BPFTIME_CUDA_SASS_DETOUR")) {
		effective_code = maybe_patch_cuda_image_sass_detour(
			impl, code, 0, "cuLibraryLoadData", &effective_size);
		effective_detoured = tls_sass_detour_last_output_is_detoured;
	}

	auto res = original(library, effective_code, jitOptions, jitOptionsValues,
			    numJitOptions, libraryOptions, libraryOptionValues,
			    numLibraryOptions);
	if (impl && res == CUDA_SUCCESS && library != nullptr && *library != nullptr &&
	    effective_code != nullptr) {
		const uint64_t h =
			(effective_size == 0)
				? 0
				: fnv1a64(reinterpret_cast<const uint8_t *>(effective_code),
					  effective_size);
		const void *base_img = tls_sass_detour_last_base_image;
		const size_t base_sz = tls_sass_detour_last_base_size;
		const uint64_t hb =
			(base_img != nullptr && base_sz != 0)
				? fnv1a64(reinterpret_cast<const uint8_t *>(base_img),
					  base_sz)
				: 0;
		impl->record_cuda_library_image(
			*library, effective_code, effective_size, h,
			effective_detoured, "cuLibraryLoadData", base_img, base_sz,
			hb, tls_sass_detour_last_sm120_image_id,
			tls_sass_detour_last_sm120_image_id);
	}
	if (impl) {
		void *lib = nullptr;
		if (library != nullptr)
			lib = (void *)(*library);
		impl->trace_cuda_library_load("cuLibraryLoadData", lib,
					      (void *)code, numJitOptions,
					      numLibraryOptions, (int)res);
	}
	return res;
}

extern "C" CUresult cuda_driver_function__cuLibraryLoadFromFile(
	CUlibrary *library, const char *fileName, CUjit_option *jitOptions,
	void **jitOptionsValues, unsigned int numJitOptions,
	CUlibraryOption *libraryOptions, void **libraryOptionValues,
	unsigned int numLibraryOptions)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(CUlibrary *, const char *, CUjit_option *, void **,
				  unsigned int, CUlibraryOption *, void **,
				  unsigned int);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_library_load_from_file : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;

	// Prefer an in-memory load so we can feed the image through the same detour
	// path as `cuLibraryLoadData` (no temp files).
	using load_data_fn_t = CUresult (*)(
		CUlibrary *, const void *, CUjit_option *, void **, unsigned int,
		CUlibraryOption *, void **, unsigned int);
	auto original_load_data = reinterpret_cast<load_data_fn_t>(
		impl ? impl->original_cu_library_load_data : nullptr);

	std::vector<uint8_t> file_bytes;
	const void *code_ptr = nullptr;
	size_t code_size = 0;
	if (impl && fileName != nullptr && *fileName && original_load_data) {
		try {
			std::ifstream ifs(fileName, std::ios::binary);
			if (ifs.is_open()) {
				ifs.seekg(0, std::ios::end);
				const std::streamoff end = ifs.tellg();
				if (end > 0 && end <= std::streamoff(256ull << 20)) {
					file_bytes.resize(static_cast<size_t>(end));
					ifs.seekg(0, std::ios::beg);
					if (ifs.read(reinterpret_cast<char *>(file_bytes.data()),
						     static_cast<std::streamsize>(
							     file_bytes.size()))) {
						code_size = file_bytes.size();
						std::lock_guard<std::mutex> guard(
							impl->owned_cuda_images_lock);
						nv_attach_impl::owned_cuda_image img;
						img.size = code_size;
						img.data = std::make_unique<uint8_t[]>(
							img.size);
						std::memcpy(img.data.get(), file_bytes.data(),
							    img.size);
						code_ptr = img.data.get();
						impl->owned_cuda_images.emplace_back(
							std::move(img));
					}
				}
			}
		} catch (...) {
			code_ptr = nullptr;
			code_size = 0;
		}
	}

	if (impl && code_ptr != nullptr && code_size != 0 && original_load_data &&
	    env_truthy_global("BPFTIME_CUDA_SASS_DETOUR")) {
		size_t effective_size = code_size;
		const void *effective_code = code_ptr;
		bool effective_detoured = false;
		effective_code = maybe_patch_cuda_image_sass_detour(
			impl, code_ptr, code_size, "cuLibraryLoadFromFile",
			&effective_size);
		effective_detoured = tls_sass_detour_last_output_is_detoured;

		auto res = original_load_data(library, effective_code, jitOptions,
					      jitOptionsValues, numJitOptions,
					      libraryOptions, libraryOptionValues,
					      numLibraryOptions);
		if (impl && res == CUDA_SUCCESS && library != nullptr &&
		    *library != nullptr && effective_code != nullptr) {
			const uint64_t h = (effective_size == 0)
						   ? 0
						   : fnv1a64(
							 reinterpret_cast<const uint8_t *>(
								 effective_code),
							 effective_size);
			const void *base_img = tls_sass_detour_last_base_image;
			const size_t base_sz = tls_sass_detour_last_base_size;
			const uint64_t hb =
				(base_img != nullptr && base_sz != 0)
					? fnv1a64(reinterpret_cast<const uint8_t *>(base_img),
						  base_sz)
					: 0;
			impl->record_cuda_library_image(
				*library, effective_code, effective_size, h,
				effective_detoured, "cuLibraryLoadFromFile", base_img,
				base_sz, hb, tls_sass_detour_last_sm120_image_id,
				tls_sass_detour_last_sm120_image_id);
		}
		if (impl) {
			void *lib = nullptr;
			if (library != nullptr)
				lib = (void *)(*library);
			impl->trace_cuda_library_load("cuLibraryLoadFromFile", lib,
						      (void *)code_ptr, numJitOptions,
						      numLibraryOptions, (int)res);
		}
		return res;
	}

	return original(library, fileName, jitOptions, jitOptionsValues, numJitOptions,
			libraryOptions, libraryOptionValues, numLibraryOptions);
}

extern "C" CUresult cuda_driver_function__cuLibraryUnload(CUlibrary library)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(CUlibrary);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_library_unload : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	auto res = original(library);
	if (impl) {
		impl->trace_cuda_library_unload((void *)library, (int)res);
	}
	return res;
}

extern "C" CUresult cuda_driver_function__cuLibraryGetModule(CUmodule *pMod,
							    CUlibrary library)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(CUmodule *, CUlibrary);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_library_get_module : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;

	auto res = original(pMod, library);
	if (impl && res == CUDA_SUCCESS && pMod != nullptr && *pMod != nullptr &&
	    library != nullptr) {
		// vLLM/flashattention commonly goes through:
		//   cuLibraryLoadData -> cuLibraryGetModule -> cuModuleGetFunction -> cuLaunchKernel
		//
		// Map CUmodule back to the code image we already tracked on the CUlibrary,
		// so cuLaunchKernel can attribute the function to a patchable image.
		impl->record_cuda_module_image_from_library(*pMod, library,
							    "cuLibraryGetModule");
		if (env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DEBUG_LAUNCH")) {
			SPDLOG_INFO(
				"SASS detour: cuLibraryGetModule lib=0x{:x} -> mod=0x{:x}",
				(uint64_t)library, (uint64_t)*pMod);
		}
	}
	return res;
}

extern "C" CUresult cuda_driver_function__cuLibraryGetKernel(CUkernel *pKernel,
							    CUlibrary library,
							    const char *name)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(CUkernel *, CUlibrary, const char *);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_library_get_kernel : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	auto res = original(pKernel, library, name);
	if (impl) {
		void *kernel = nullptr;
		if (pKernel != nullptr)
			kernel = (void *)(*pKernel);
		impl->trace_cuda_library_get_kernel((void *)library, kernel,
						    name, (int)res);
		if ((int)res == 0 && kernel != nullptr &&
		    name != nullptr && name[0] != '\0') {
			impl->record_original_cukernel_name(*pKernel, name);
			impl->record_original_cukernel_library(*pKernel, library);
			if (env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DEBUG_LAUNCH")) {
				SPDLOG_INFO(
					"SASS detour: cuLibraryGetKernel lib=0x{:x} kernel=0x{:x} name='{}'",
					(uint64_t)library, (uint64_t)*pKernel, name);
			}
		}
	}
	return res;
}

extern "C" CUresult cuda_driver_function__cuKernelGetName(const char **name,
							 CUkernel hfunc)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(const char **, CUkernel);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_kernel_get_name : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	auto res = original(name, hfunc);
	if (impl) {
		const char *resolved = nullptr;
		if (name != nullptr)
			resolved = *name;
		impl->trace_cuda_kernel_get_name((void *)hfunc, resolved,
						 (int)res);
		if ((int)res == 0 && resolved != nullptr && resolved[0] != '\0') {
			impl->record_original_cukernel_name(hfunc, resolved);
		}
	}
	return res;
}

extern "C" CUresult
cuda_driver_function__cuKernelGetFunction(CUfunction *pFunc, CUkernel kernel)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	using fn_t = CUresult (*)(CUfunction *, CUkernel);
	auto original = reinterpret_cast<fn_t>(
		impl ? impl->original_cu_kernel_get_function : nullptr);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	auto res = original(pFunc, kernel);
	if (impl) {
		void *func = nullptr;
		if (pFunc != nullptr)
			func = (void *)(*pFunc);
		impl->trace_cuda_kernel_get_function((void *)kernel, func,
						     (int)res);
		if ((int)res == 0 && pFunc != nullptr && *pFunc != nullptr) {
			impl->record_original_cufunction_cukernel(*pFunc, kernel);
			auto name = impl->find_original_cukernel_name(kernel);
			if (name) {
				impl->record_original_cufunction_name(*pFunc,
								      *name);
			}
			if (env_truthy_global("BPFTIME_CUDA_SASS_DETOUR_DEBUG_LAUNCH")) {
				SPDLOG_INFO(
					"SASS detour: cuKernelGetFunction kernel=0x{:x} -> func=0x{:x} name='{}'",
					(uint64_t)kernel, (uint64_t)*pFunc,
					name ? *name : std::string("unknown"));
			}
		}
	}
	return res;
}

static const CUDA_KERNEL_NODE_PARAMS_v1 *
cuda_graph_maybe_patch_kernel_node_params_v1(
	nv_attach_impl &impl, const CUDA_KERNEL_NODE_PARAMS_v1 *params,
	CUDA_KERNEL_NODE_PARAMS_v1 &patched_params)
{
	if (params == nullptr || params->func == nullptr) {
		return params;
	}
	auto kernel_name = cuda_graph_maybe_get_kernel_name_from_cufunction(
		impl, params->func);
	if (!kernel_name) {
		return params;
	}
	auto patched_func = impl.find_patched_kernel_function(*kernel_name);
	if (!patched_func) {
		return params;
	}
	patched_params = *params;
	patched_params.func = *patched_func;
	return &patched_params;
}

static const CUDA_KERNEL_NODE_PARAMS_v2 *
cuda_graph_maybe_patch_kernel_node_params_v2(
	nv_attach_impl &impl, const CUDA_KERNEL_NODE_PARAMS_v2 *params,
	CUDA_KERNEL_NODE_PARAMS_v2 &patched_params)
{
	if (params == nullptr) {
		return params;
	}
	std::optional<std::string> kernel_name;
	if (params->func != nullptr) {
		kernel_name = cuda_graph_maybe_get_kernel_name_from_cufunction(
			impl, params->func);
	} else if (params->kern != nullptr) {
		kernel_name = cuda_graph_maybe_get_kernel_name_from_cukernel(
			params->kern);
	} else {
	}
	if (!kernel_name) {
		return params;
	}
	auto patched_func = impl.find_patched_kernel_function(*kernel_name);
	if (!patched_func) {
		return params;
	}
	patched_params = *params;
	patched_params.func = *patched_func;
	patched_params.kern = nullptr;
	return &patched_params;
}

extern "C" CUresult cuda_driver_function__cuGraphAddKernelNode_v1(
	CUgraphNode *phGraphNode, CUgraph hGraph,
	const CUgraphNode *dependencies, size_t numDependencies,
	const CUDA_KERNEL_NODE_PARAMS_v1 *nodeParams)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	if (impl == nullptr)
		return CUDA_ERROR_UNKNOWN;
	auto original = reinterpret_cast<cu_graph_add_kernel_node_v1_fn_t>(
		impl->original_cu_graph_add_kernel_node_v1);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	CUDA_KERNEL_NODE_PARAMS_v1 patched_params;
	auto params_to_use = cuda_graph_maybe_patch_kernel_node_params_v1(
		*impl, nodeParams, patched_params);
	return original(phGraphNode, hGraph, dependencies, numDependencies,
			params_to_use);
}

extern "C" CUresult cuda_driver_function__cuGraphAddKernelNode_v2(
	CUgraphNode *phGraphNode, CUgraph hGraph,
	const CUgraphNode *dependencies, size_t numDependencies,
	const CUDA_KERNEL_NODE_PARAMS_v2 *nodeParams)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	if (impl == nullptr)
		return CUDA_ERROR_UNKNOWN;
	auto original = reinterpret_cast<cu_graph_add_kernel_node_v2_fn_t>(
		impl->original_cu_graph_add_kernel_node_v2);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	CUDA_KERNEL_NODE_PARAMS_v2 patched_params;
	auto params_to_use = cuda_graph_maybe_patch_kernel_node_params_v2(
		*impl, nodeParams, patched_params);
	return original(phGraphNode, hGraph, dependencies, numDependencies,
			params_to_use);
}

extern "C" CUresult cuda_driver_function__cuGraphExecKernelNodeSetParams_v1(
	CUgraphExec hGraphExec, CUgraphNode hNode,
	const CUDA_KERNEL_NODE_PARAMS_v1 *nodeParams)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	if (impl == nullptr)
		return CUDA_ERROR_UNKNOWN;
	auto original =
		reinterpret_cast<cu_graph_exec_kernel_node_set_params_v1_fn_t>(
			impl->original_cu_graph_exec_kernel_node_set_params_v1);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	CUDA_KERNEL_NODE_PARAMS_v1 patched_params;
	auto params_to_use = cuda_graph_maybe_patch_kernel_node_params_v1(
		*impl, nodeParams, patched_params);
	return original(hGraphExec, hNode, params_to_use);
}

extern "C" CUresult cuda_driver_function__cuGraphExecKernelNodeSetParams_v2(
	CUgraphExec hGraphExec, CUgraphNode hNode,
	const CUDA_KERNEL_NODE_PARAMS_v2 *nodeParams)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	if (impl == nullptr)
		return CUDA_ERROR_UNKNOWN;
	auto original =
		reinterpret_cast<cu_graph_exec_kernel_node_set_params_v2_fn_t>(
			impl->original_cu_graph_exec_kernel_node_set_params_v2);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	CUDA_KERNEL_NODE_PARAMS_v2 patched_params;
	auto params_to_use = cuda_graph_maybe_patch_kernel_node_params_v2(
		*impl, nodeParams, patched_params);
	return original(hGraphExec, hNode, params_to_use);
}

extern "C" CUresult cuda_driver_function__cuGraphKernelNodeSetParams_v1(
	CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS_v1 *nodeParams)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	if (impl == nullptr)
		return CUDA_ERROR_UNKNOWN;
	auto original =
		reinterpret_cast<cu_graph_kernel_node_set_params_v1_fn_t>(
			impl->original_cu_graph_kernel_node_set_params_v1);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	CUDA_KERNEL_NODE_PARAMS_v1 patched_params;
	auto params_to_use = cuda_graph_maybe_patch_kernel_node_params_v1(
		*impl, nodeParams, patched_params);
	return original(hNode, params_to_use);
}

extern "C" CUresult cuda_driver_function__cuGraphKernelNodeSetParams_v2(
	CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS_v2 *nodeParams)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	if (impl == nullptr)
		return CUDA_ERROR_UNKNOWN;
	auto original =
		reinterpret_cast<cu_graph_kernel_node_set_params_v2_fn_t>(
			impl->original_cu_graph_kernel_node_set_params_v2);
	if (!original)
		return CUDA_ERROR_UNKNOWN;
	CUDA_KERNEL_NODE_PARAMS_v2 patched_params;
	auto params_to_use = cuda_graph_maybe_patch_kernel_node_params_v2(
		*impl, nodeParams, patched_params);
	return original(hNode, params_to_use);
}

static cudaError_t
mirror_cuda_memcpy_from_symbol(nv_attach_impl *impl, bool async, void *dst,
			       const void *symbol, size_t count, size_t offset,
			       cudaMemcpyKind kind, cudaStream_t stream = 0)
{
	auto record_itr = impl->symbol_address_to_fatbin.find((void *)symbol);
	if (record_itr == impl->symbol_address_to_fatbin.end()) {
		SPDLOG_DEBUG(
			"In mirror_cuda_memcpy_from_symbol: calling original cudaMemcpyFromSymbol");
		if (async) {
			auto original = reinterpret_cast<
				cuda_memcpy_from_symbol_async_fn_t>(
				impl->original_cuda_memcpy_from_symbol_async);
			if (!original) {
				return cudaErrorUnknown;
			}
			return original(dst, symbol, count, offset, kind,
					stream);
		} else {
			auto original =
				reinterpret_cast<cuda_memcpy_from_symbol_fn_t>(
					impl->original_cuda_memcpy_from_symbol);
			if (!original) {
				return cudaErrorUnknown;
			}
			return original(dst, symbol, count, offset, kind);
		}
		return cudaErrorUnknown;
	}
	auto &record = *record_itr->second;
	auto var_itr = record.variable_addr_to_symbol.find((void *)symbol);
	if (var_itr == record.variable_addr_to_symbol.end()) {
		SPDLOG_DEBUG(
			"mirror_cuda_memcpy_from_symbol: no variable info for symbol pointer {:x}",
			(uintptr_t)symbol);
		return cudaErrorUnknown;
	}
	auto &var_info = var_itr->second;
	if (offset >= var_info.size) {
		SPDLOG_WARN(
			"mirror_cuda_memcpy_from_symbol: offset {} exceeds size {} for symbol {}",
			offset, var_info.size, var_info.symbol_name);
		return cudaErrorUnknown;
	}
	size_t writable = var_info.size - offset;
	size_t bytes_to_copy = std::min(count, writable);
	if (bytes_to_copy == 0)
		return cudaErrorUnknown;
	if (bytes_to_copy != count) {
		SPDLOG_WARN(
			"mirror_cuda_memcpy_from_symbol: truncating copy for symbol {} (requested={}, allowed={})",
			var_info.symbol_name, count, bytes_to_copy);
	}
	CUdeviceptr src = var_info.ptr + offset;
	CUstream cu_stream = reinterpret_cast<CUstream>(stream);
	CUresult status = CUDA_SUCCESS;

	auto copy_device_ptr = [](const void *ptr) -> CUdeviceptr {
		return static_cast<CUdeviceptr>(
			reinterpret_cast<uintptr_t>(ptr));
	};

	switch (kind) {
	case cudaMemcpyDeviceToHost:
	case cudaMemcpyDefault:
		status = async ? cuMemcpyDtoHAsync(dst, src, bytes_to_copy,
						   cu_stream) :
				 cuMemcpyDtoH(dst, src, bytes_to_copy);
		break;
	case cudaMemcpyDeviceToDevice:
		status = async ? cuMemcpyDtoDAsync(copy_device_ptr(dst), src,
						   bytes_to_copy, cu_stream) :
				 cuMemcpyDtoD(copy_device_ptr(dst), src,
					      bytes_to_copy);
		break;
	default:
		SPDLOG_DEBUG(
			"mirror_cuda_memcpy_from_symbol: unsupported memcpy kind {} for symbol {}",
			(int)kind, var_info.symbol_name);
		return cudaErrorUnknown;
	}
	if (status != CUDA_SUCCESS) {
		SPDLOG_WARN(
			"mirror_cuda_memcpy_from_symbol: failed to copy symbol {} (err={})",
			var_info.symbol_name, (int)status);
		return cudaErrorUnknown;
	}
	return cudaSuccess;
}

extern "C" cudaError_t cuda_runtime_function__cudaMemcpyFromSymbol(
	void *dst, const void *symbol, size_t count, size_t offset = 0,
	cudaMemcpyKind kind = cudaMemcpyDeviceToHost)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	if (impl == nullptr) {
		SPDLOG_ERROR(
			"cudaMemcpyFromSymbol called without nv_attach_impl");
		return cudaErrorUnknown;
	} else {
		SPDLOG_DEBUG(
			"call cudaMemcpyFromSymbol with args: {:x}, {:x}, {}, {}, {}",
			(uintptr_t)dst, (uintptr_t)symbol, count, offset,
			(int)kind);
	}
	return mirror_cuda_memcpy_from_symbol(impl, false, dst, symbol, count,
					      offset, kind);
}

extern "C" cudaError_t cuda_runtime_function__cudaMemcpyFromSymbolAsync(
	void *dst, const void *symbol, size_t count, size_t offset,
	cudaMemcpyKind kind, cudaStream_t stream = 0)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto impl =
		(nv_attach_impl *)gum_invocation_context_get_replacement_data(
			gum_ctx);
	if (impl == nullptr) {
		SPDLOG_ERROR(
			"cudaMemcpyFromSymbol called without nv_attach_impl");
		return cudaErrorUnknown;
	} else {
		SPDLOG_DEBUG(
			"call cudaMemcpyFromSymbol with args: {:x}, {:x}, {}, {}, {}, {:x}",
			(uintptr_t)dst, (uintptr_t)symbol, count, offset,
			(int)kind, (uintptr_t)stream);
	}
	return mirror_cuda_memcpy_from_symbol(impl, true, dst, symbol, count,
					      offset, kind, stream);
}
