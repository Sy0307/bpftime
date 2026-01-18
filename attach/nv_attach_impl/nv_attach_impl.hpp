#ifndef _BPFTIME_NV_ATTACH_IMPL_HPP
#define _BPFTIME_NV_ATTACH_IMPL_HPP
#include "ebpf_inst.h"
#include "nv_attach_utils.hpp"
#include "ptx_compiler/ptx_compiler.hpp"
#include "ptxpass/core.hpp"
#include <base_attach_impl.hpp>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <nvml.h>
#include <cuda.h>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <atomic>
#include <fstream>
#include <sys/ptrace.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include "nv_attach_fatbin_record.hpp"
#include <tuple>
#include <variant>
#include <vector>

#include "sass_detour/sass_detour.hpp"

namespace bpftime
{
namespace attach
{

using print_config_fn = void (*)(int length, char *out);
using process_input_fn = int (*)(const char *input, int length, char *output);

std::string filter_compiled_ptx_for_ebpf_program(std::string input,
						 std::string);

constexpr int ATTACH_CUDA_PROBE = 8;
constexpr int ATTACH_CUDA_RETPROBE = 9;
struct MapBasicInfo {
	bool enabled;
	int key_size;
	int value_size;
	int max_entries;
	int map_type;
	void *extra_buffer;
	uint64_t max_thread_count;
};
struct nv_hooker_func_t {
	void *func;
};

enum class AttachedToFunction {
	RegisterFatbin,
	RegisterFunction,
	RegisterVariable,
	RegisterFatbinEnd,
	CudaMalloc,
	CudaMallocManaged,
	CudaMemcpyToSymbol,
	CudaMemcpyToSymbolAsync
};
struct CUDARuntimeFunctionHookerContext {
	class nv_attach_impl *impl;
	AttachedToFunction to_function;
};

struct nv_attach_entry {
	std::vector<ebpf_inst> instuctions;
	// Kernels to be patched for this attach entry
	std::vector<std::string> kernels;
	// program name for this attach entry
	std::string program_name;
	// pass-based execution fields
	std::map<std::string, std::string> parameters; // arbitrary parameters
						       // for pass
	// Extra serialized parameters (JSON string) reserved for future use
	std::optional<std::string> extras;
	struct pass_cfg_with_exec_path *config;
};

struct pass_cfg_with_exec_path {
	std::filesystem::path executable_path;
	ptxpass::pass_config::PassConfig pass_config;
	print_config_fn print_config;
	process_input_fn process_input;

	void *handle;

	pass_cfg_with_exec_path(std::filesystem::path path,
				ptxpass::pass_config::PassConfig config,
				print_config_fn print_config,
				process_input_fn process_input, void *handle)
		: executable_path(path), pass_config(config),
		  print_config(print_config), process_input(process_input),
		  handle(handle)
	{
	}

	pass_cfg_with_exec_path(const pass_cfg_with_exec_path &) = delete;
	pass_cfg_with_exec_path &
	operator=(const pass_cfg_with_exec_path &) = delete;
	pass_cfg_with_exec_path(pass_cfg_with_exec_path &&) = default;
	pass_cfg_with_exec_path &
	operator=(pass_cfg_with_exec_path &&) = default;

	~pass_cfg_with_exec_path()
	{
		dlclose(handle);
	}
};

// Attach implementation of syscall trace
// It provides a callback to receive original syscall calls, and dispatch the
// concrete stuff to individual callbacks
class nv_attach_impl final : public base_attach_impl {
    public:
	int detach_by_id(int id) override;
	int create_attach_with_ebpf_callback(
		ebpf_run_callback &&cb, const attach_private_data &private_data,
		int attach_type) override;
	// Register CUDA-specific ext helpers required by LLVM-JIT to resolve
	// symbols like _bpf_helper_ext_0502/_0503 when compiling programs
	void register_custom_helpers(
		ebpf_helper_register_callback register_callback) override;
	nv_attach_impl(const nv_attach_impl &) = delete;
	nv_attach_impl &operator=(const nv_attach_impl &) = delete;
	nv_attach_impl();
	virtual ~nv_attach_impl();
	bool can_patch_ptx() const;
	std::optional<std::map<std::string, std::tuple<std::string, bool>>>
		hack_fatbin(std::map<std::string, std::string>);
	std::map<std::string, std::string>
	extract_ptxs(std::vector<uint8_t> &&);
	void mirror_cuda_memcpy_to_symbol(const void *symbol, const void *src,
					  size_t count, size_t offset,
					  cudaMemcpyKind kind,
					  cudaStream_t stream, bool async);
	void mirror_cuda_memcpy_from_symbol(void *dst, const void *symbol,
					    size_t count, size_t offset,
					    cudaMemcpyKind kind,
					    cudaStream_t stream, bool async);

	int find_attach_entry_by_program_name(const char *name) const;
	int run_attach_entry_on_gpu(int attach_id, int run_count = 1,
				    int grid_dim_x = 1, int grid_dim_y = 1,
				    int grid_dim_z = 1, int block_dim_x = 1,
				    int block_dim_y = 1, int block_dim_z = 1);
	void record_patched_kernel_function(const std::string &kernel_name,
					    CUfunction function);
	enum class PatchedKernelKind : uint8_t {
		Unknown = 0,
		// On-demand patch-all fallback used to inject identify-only stubs before
		// func_id is known.
		PatchAllFallback = 1,
		// On-demand patch targeted by a resolved func_id.
		FuncIdTargeted = 2,
		// Eager patch at module load time (e.g. by text name / func_id propagation).
		LoadTime = 3,
		// PTX/LLVM rewrite produced a replacement function.
		PtxRewrite = 4,
	};
	void record_patched_kernel_function_ex(const std::string &kernel_name,
					       CUfunction function,
					       PatchedKernelKind kind,
					       uint32_t target_func_id = 0);
	std::optional<CUfunction>
	find_patched_kernel_function(const std::string &kernel_name) const;
	struct PatchedKernelEntry {
		CUfunction function = nullptr;
		PatchedKernelKind kind = PatchedKernelKind::Unknown;
		uint32_t target_func_id = 0;
	};
	std::optional<PatchedKernelEntry>
	find_patched_kernel_entry(const std::string &kernel_name) const;
	void record_original_cufunction_name(CUfunction function,
					     const std::string &kernel_name);
	void record_original_cufunction_module(CUfunction function, CUmodule module);
	void record_original_cufunction_cukernel(CUfunction function, CUkernel kernel);
	std::optional<std::string>
	find_original_kernel_name(CUfunction function) const;
	void record_original_cukernel_name(CUkernel kernel,
					   const std::string &kernel_name);
	void record_original_cukernel_library(CUkernel kernel, CUlibrary library);
	std::optional<std::string>
	find_original_cukernel_name(CUkernel kernel) const;

	struct cuda_module_image_info {
		uint64_t image_ptr = 0;
		size_t image_size = 0;
		uint64_t image_hash = 0;
		bool patched = false;
		// Best-effort SM120 ELF image_id for this code object (0 when unknown).
		uint32_t sm120_image_id = 0;
		// Best-effort base image (same format as `image_ptr`, but without SM120
		// SASS detours). Used to support on-demand func_id-targeted upgrades even
		// when the load-time path already detoured a candidate image.
		uint64_t base_image_ptr = 0;
		size_t base_image_size = 0;
		uint64_t base_image_hash = 0;
		uint32_t base_sm120_image_id = 0;
		std::string api;
	};
	void record_cuda_module_image(CUmodule module, const void *image, size_t size,
				      uint64_t hash, bool patched,
				      std::string_view api,
				      const void *base_image = nullptr,
				      size_t base_size = 0,
				      uint64_t base_hash = 0,
				      uint32_t sm120_image_id = 0,
				      uint32_t base_sm120_image_id = 0);
	void record_cuda_library_image(CUlibrary library, const void *code, size_t size,
				       uint64_t hash, bool patched,
				       std::string_view api,
				       const void *base_image = nullptr,
				       size_t base_size = 0,
				       uint64_t base_hash = 0,
				       uint32_t sm120_image_id = 0,
				       uint32_t base_sm120_image_id = 0);
	void record_cuda_module_image_from_library(CUmodule module, CUlibrary library,
						   std::string_view api);
	std::optional<cuda_module_image_info>
	find_cuda_module_image_by_function(CUfunction function) const;
	// Best-effort: resolve the CUmodule that owns a CUfunction.
	//
	// Used by JIT-link PTX instrumentation to bind module globals at launch time
	// (e.g., writing an output pointer/cap/gates into injected globals).
	std::optional<CUmodule> resolve_cumodule_for_cufunction(CUfunction function);
	std::vector<std::unique_ptr<fatbin_record>> fatbin_records;
	fatbin_record *current_fatbin = nullptr;
	std::map<void *, fatbin_record *> symbol_address_to_fatbin;
	uintptr_t shared_mem_ptr;
	std::optional<std::vector<MapBasicInfo>> map_basic_info;
	void *ptx_compiler_dl_handle = nullptr;
	nv_attach_impl_ptx_compiler_handler ptx_compiler;
	/// SHA256 of ELF -> PTX module
	std::shared_ptr<std::map<std::string, std::shared_ptr<ptx_in_module>>>
		module_pool;
	/// SHA256 of PTX -> ELF
	std::shared_ptr<std::map<std::string, std::vector<uint8_t>>> ptx_pool;

	// Original function pointers for Frida replace hooks (trampolines)
	// They are set by gum_interceptor_replace(...) and must be used to call
	// the original implementation (calling the symbol directly will
	// recurse). Which is used for cudagraph hook.
	void *original_cuda_launch_kernel = nullptr;
	void *original_cuda_launch_kernel_ptsz = nullptr;
	void *original_cu_graph_add_kernel_node_v1 = nullptr;
	void *original_cu_graph_add_kernel_node_v2 = nullptr;
	void *original_cu_graph_exec_kernel_node_set_params_v1 = nullptr;
	void *original_cu_graph_exec_kernel_node_set_params_v2 = nullptr;
	void *original_cu_graph_kernel_node_set_params_v1 = nullptr;
	void *original_cu_graph_kernel_node_set_params_v2 = nullptr;
	void *original_cuda_memcpy_from_symbol = nullptr;
	void *original_cuda_memcpy_from_symbol_async = nullptr;
	void *original_cu_module_get_function = nullptr;
	void *original_cu_launch_kernel = nullptr;
	void *original_cu_memcpy_htod = nullptr;
	void *original_cu_memcpy_htod_async = nullptr;
	void *original_cu_memcpy_dtoh = nullptr;
	void *original_cu_memcpy_dtoh_async = nullptr;
	void *original_cu_memcpy_dtod = nullptr;
	void *original_cu_memcpy_dtod_async = nullptr;
	void *original_cu_memset_d8_async = nullptr;
	void *original_cu_memset_d32_async = nullptr;
	void *original_cu_mem_alloc = nullptr;
	void *original_cu_mem_free = nullptr;
	void *original_cu_mem_alloc_async = nullptr;
	void *original_cu_mem_free_async = nullptr;
	void *original_cu_stream_synchronize = nullptr;
	void *original_cu_ctx_synchronize = nullptr;
	void *original_cu_ctx_destroy = nullptr;
	void *original_cu_ctx_destroy_v2 = nullptr;
	void *original_cu_device_primary_ctx_release = nullptr;
	void *original_cu_device_primary_ctx_release_v2 = nullptr;
	void *original_cu_event_record = nullptr;
	void *original_cu_event_synchronize = nullptr;
	void *original_cu_graph_launch = nullptr;
	void *original_cu_link_add_data = nullptr;
	void *original_cu_link_add_data_v2 = nullptr;
	void *original_cu_link_complete = nullptr;
	void *original_cu_link_destroy = nullptr;
	void *original_cu_stream_create = nullptr;
	void *original_cu_stream_create_with_priority = nullptr;
	void *original_cu_stream_destroy_v2 = nullptr;
	void *original_cu_stream_wait_event = nullptr;
	void *original_cu_event_create = nullptr;
	void *original_cu_event_destroy_v2 = nullptr;
	void *original_cu_module_load_data = nullptr;
	void *original_cu_module_load_data_ex = nullptr;
	void *original_cu_module_load_fatbinary = nullptr;
	void *original_cu_module_load = nullptr;
	void *original_cu_module_unload = nullptr;
	void *original_cu_library_load_data = nullptr;
	void *original_cu_library_load_from_file = nullptr;
	void *original_cu_library_unload = nullptr;
	void *original_cu_library_get_module = nullptr;
	void *original_cu_library_get_kernel = nullptr;
	void *original_cu_kernel_get_name = nullptr;
	void *original_cu_kernel_get_function = nullptr;

	struct owned_cuda_image {
		std::unique_ptr<uint8_t[]> data;
		size_t size = 0;
	};
	std::mutex owned_cuda_images_lock;
	std::vector<owned_cuda_image> owned_cuda_images;
	// Keep injected PTX buffers alive across cuLinkAddData -> cuLinkComplete.
	std::mutex culink_owned_inputs_lock;
	std::unordered_map<CUlinkState, std::vector<std::vector<uint8_t>>>
		culink_owned_inputs;
	struct sass_detour_cache_entry {
		const void *patched = nullptr;
		size_t size = 0;
		// Whether the returned image has SM120 SASS detours applied.
		// (Distinguish from “rewritten” images like jitlink PTX->cubin.)
		bool detoured = false;
		// SM120 ELF image_id for the detoured code object (0 when unknown/non-SM120).
		uint32_t sm120_image_id = 0;
		// Optional base image pointer/size (same format as `patched`, but without
		// SM120 SASS detours).
		const void *base = nullptr;
		size_t base_size = 0;
	};
	std::unordered_map<const void *, sass_detour_cache_entry> sass_detour_code_cache;
	// SASS detour filter state (1.2 tag/func_id mapping closure):
	// - Tracks func_ids learned from fatbins that still have `.nv.info.<name>`
	//   section names, so we can detour stripped SM120 raw-ELF cubins by func_id.
	// - Protected by `owned_cuda_images_lock` (same as patcher cache).
	struct sass_detour_filter_state_t {
		std::string active_filter;
		std::unordered_set<uint32_t> learned_func_ids;
		// Optional stronger scoping for func_id propagation:
		// image_id (32-bit) -> func_id set.
		// When available (e.g. from identify closure cache), we can safely detour
		// stripped raw-ELF SM120 code objects even if the kernel name string is
		// missing from the blob, without risking cross-module func_id collisions.
		std::unordered_map<uint32_t, std::unordered_set<uint32_t>>
			learned_func_ids_by_image_id;
		// In-process (single-run) resolution: kernel_name -> func_id.
		// Needed when `learned_func_ids` is a superset (e.g. many flashattention
		// template instantiations match the same substring filter).
		std::unordered_map<std::string, uint32_t> resolved_func_id_by_kernel_name;
		int recent_fatbinc_filter_budget = 0;
			bool func_id_cache_loaded = false;
			// Single-run identify closure (1.2): when enabled, we can arm an identify
			// window on a matching kernel launch and collect func_id(s) at the next
			// sync point.
			std::atomic<bool> identify_pending { false };
			std::string identify_pending_kernel_name;
			uint32_t identify_epoch = 0;
			uint32_t identify_target_func_id = 0;
		};
	sass_detour_filter_state_t sass_detour_filter_state;

	// Minimal SASS sampler state (SM120-only for now).
	struct sass_sampling_state_t {
		std::mutex lock;
		bool enabled = false;
		bool initialized = false;
		bool dumped_once = false;
		// Best-effort host-side dedupe for control header writes.
		// Avoid spamming cuMemcpyHtoD/logs on every launch when the target
		// (func_id, epoch) hasn't changed.
		bool last_control_valid = false;
		uint32_t last_control_epoch = 0;
		uint32_t last_control_target_func_id = 0;
		uint32_t sync_dump_attempts = 0;
		bool dumped_on_first_launch = false;
		bool dump_on_exit = true;
		std::string dump_path = "/tmp/bpftime-sass-samples.jsonl";
		sass_detour::Sm120SamplingConfig::Mode mode =
			sass_detour::Sm120SamplingConfig::Mode::SmidBitmap;
		// PcMarker settings:
		// - ring buffer entries (power-of-two)
		// - optional extra patch offsets within the `.text.*` section
		// - density gates
		uint32_t marker_ring_entries = 4096;
		std::vector<uint32_t> marker_offsets;
		bool marker_lane0_only = true;
		bool marker_warp0_only = false;
		bool marker_cta_clamp = true;
			// ThreadMap storage layout:
			// - false: device writes per-warp slots, host expands to per-thread.
			// - true:  device writes per-thread bytes directly.
			bool thread_map_device = false;
				// ThreadMap device-side bring-up knobs:
				// - lane0_only: only lane0 writes (debug).
				// - warp0_only: only warp0 writes (density control).
				// - stride4: use aligned offsets (idx*4) for each thread.
					bool thread_map_device_lane0_only = false;
					bool thread_map_device_warp0_only = false;
					bool thread_map_device_stride4 = false;
					// When true, skip device stores for CTAs with ctaid.x>=max_records
					// (reduces contention vs mask-based indexing).
					bool thread_map_device_cta_clamp = false;
					bool thread_map_device_no_store = false;
					// reg255: reserve a small spill area after the sampler data so
					// experimental stubs can save/restore a few GPRs/predicates.
					bool reg255_thread_map_spill_enable = false;
					bool reg255_thread_map_spill_per_thread = false;
					uint32_t reg255_thread_map_spill_stride_bytes = 0;
					uint32_t reg255_thread_map_spill_bytes = 0;
		// Capacity for non-bitmap modes. Note:
		// - `CtaSmid`:    max_records * 4B
			// - `WarpMap`:    max_records * 32 * 4B
			// - `ThreadMap`:  (device=false) max_records * 32 * 4B (expanded on host)
			//               (device=true)  max_records * 1024 * (1B or 4B stride)
			//
		// Keep a conservative default to avoid huge allocations when switching
		// modes.
		uint32_t max_records = 4096;
		uint8_t desc_ur = 4;
		CUdeviceptr device_buffer = 0;
		size_t device_bytes = 0;
		uint32_t buffer_data_offset = 0;
		bool control_enabled = false;
		std::unordered_map<uint32_t, std::string> tag_to_kernel;
	};
	sass_sampling_state_t sass_sampling;

	// Returns a sampling config if enabled and successfully initialized.
	std::optional<sass_detour::Sm120SamplingConfig> get_sm120_sampling_cfg();
		void record_sass_sampled_kernels(
			const std::vector<sass_detour::InstrumentedKernelInfo> &kernels);
		void maybe_dump_sass_samples();
		void maybe_dump_sass_samples_on_sync();
		// Force a dump regardless of dump-on-sync attempt budgeting.
		// Intended for bring-up paths that explicitly synchronize after a matching
		// sampled launch (e.g. `BPFTIME_CUDA_SASS_SAMPLE_SYNC_AFTER_LAUNCH`).
		void dump_sass_samples_force();

		uint64_t trace_cuda_kernel_launch(const std::string &kernel_name,
						  int grid_x, int grid_y,
						  int grid_z, int block_x,
						  int block_y, int block_z,
						  size_t shared_mem, void *stream,
						  void **kernel_params,
						  void **extra);
		void trace_cuda_kernel_launch(const std::string &kernel_name,
					      int grid_x, int grid_y, int grid_z,
					      int block_x, int block_y,
					      int block_z, size_t shared_mem,
					      void *stream)
		{
			(void)trace_cuda_kernel_launch(kernel_name, grid_x, grid_y,
						       grid_z, block_x, block_y,
						       block_z, shared_mem, stream,
						       nullptr, nullptr);
		}
	void trace_cuda_memcpy(const char *kind, uint64_t bytes, void *dst,
			       void *src, void *stream, int result,
			       bool async);
	void trace_cuda_memset(const char *kind, uint64_t bytes, void *dst,
			       uint64_t value, void *stream, int result,
			       bool async);
	void trace_cuda_alloc(const char *kind, uint64_t bytes, void *ptr,
			      void *stream, int result);
	void trace_cuda_free(const char *kind, void *ptr, void *stream,
			     int result);
	void trace_cuda_sync(const char *api, void *obj, uint64_t duration_ns,
			     int result);
	void trace_cuda_graph_launch(void *graph_exec, void *stream, int result);
	void trace_cuda_event_record(void *event, void *stream, int result);
	void trace_cuda_stream_create(const char *api, void *stream,
				      unsigned int flags, int priority,
				      int result);
	void trace_cuda_stream_destroy(void *stream, int result);
	void trace_cuda_stream_wait_event(void *stream, void *event,
					  unsigned int flags, int result);
	void trace_cuda_event_create(void *event, unsigned int flags,
				     int result);
	void trace_cuda_event_destroy(void *event, int result);
	void trace_cuda_module_load(const char *api, void *module, void *image,
				    unsigned int num_options, int result);
	void trace_cuda_module_unload(void *module, int result);
	void trace_cuda_library_load(const char *api, void *library, void *code,
				     unsigned int num_jit_options,
				     unsigned int num_library_options,
				     int result);
	void trace_cuda_library_unload(void *library, int result);
	void trace_cuda_library_get_kernel(void *library, void *kernel,
					   const char *name, int result);
	void trace_cuda_kernel_get_name(void *kernel, const char *name,
					int result);
	void trace_cuda_kernel_get_function(void *kernel, void *function,
					    int result);

	// Non-intrusive kernel timing: record (start,end) events around a launch,
	// then flush durations at sync points.
	bool enqueue_cuda_kernel_timing(uint64_t launch_seq,
					const std::string &kernel_name,
					void *stream, CUevent start,
					CUevent end);

	// Track owned JIT-link input buffers for the CUDA driver linker APIs.
	//
	// When we inject PTX at `cuLinkAddData`, we must keep the patched bytes alive
	// until `cuLinkComplete` finishes and the linker state is destroyed.
	void culink_track_owned_input(CUlinkState state,
				      std::vector<uint8_t> &&bytes,
				      const void **out_data,
				      size_t *out_size);
	void culink_release_state(CUlinkState state);

	// Debug/diagnostic helpers for wrappers (keep minimal surface).
	bool cuda_launch_trace_is_enabled() const;
	bool cuda_launch_trace_stream_good() const;
	std::string cuda_launch_trace_path_copy() const;

    private:
	void *frida_interceptor;
	void *frida_listener;
	std::vector<std::unique_ptr<CUDARuntimeFunctionHookerContext>>
		hooker_contexts;
	std::map<int, nv_attach_entry> hook_entries;
	// discovered pass definitions
	std::vector<std::unique_ptr<pass_cfg_with_exec_path>>
		pass_configurations;
	std::map<std::string, ptxpass::runtime_response::RuntimeResponse>
		patch_cache;
	mutable std::mutex cuda_symbol_map_mutex;
	std::unordered_map<std::string, PatchedKernelEntry> patched_kernel_by_name;
	std::unordered_map<CUfunction, std::string> kernel_name_by_cufunction;
	std::unordered_map<CUfunction, CUmodule> module_by_cufunction;
	std::unordered_map<CUmodule, cuda_module_image_info> module_image_by_module;
	std::unordered_map<CUfunction, CUkernel> cukernel_by_cufunction;
	std::unordered_map<CUkernel, CUlibrary> library_by_cukernel;
	std::unordered_map<CUlibrary, cuda_module_image_info> library_image_by_library;
	std::unordered_map<CUkernel, std::string> kernel_name_by_cukernel;

	bool cuda_launch_trace_enabled = false;
	std::string cuda_launch_trace_path;
	std::mutex cuda_launch_trace_mutex;
	std::ofstream cuda_launch_trace_ofs;
	std::atomic<uint64_t> cuda_launch_seq { 0 };

	struct pending_kernel_timing {
		uint64_t launch_seq = 0;
		std::string kernel_name;
		void *stream = nullptr;
		CUevent ev_start = nullptr;
		CUevent ev_end = nullptr;
	};
	std::mutex kernel_timing_mutex;
	std::vector<pending_kernel_timing> pending_kernel_timings;
};

std::string add_semicolon_for_variable_lines(std::string input);
} // namespace attach
} // namespace bpftime
#endif /* _BPFTIME_NV_ATTACH_IMPL_HPP */
