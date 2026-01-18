#include "nv_attach_impl.hpp"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "ebpf_inst.h"
#include "frida-gum.h"

#include "nvPTXCompiler.h"
#include "nv_attach_private_data.hpp"
#include "nv_attach_utils.hpp"
#include "ptx_compiler/ptx_compiler.hpp"
// #include "spdlog/common.h"
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
#include <chrono>
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
#include <sstream>
#include <span>
#include <sys/uio.h>
#include <set>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <sys/user.h>
#include <sys/uio.h>
#include <link.h>
#include <tuple>
#include <time.h>
#include <unistd.h>
#include <variant>
#include <vector>
#include <boost/asio.hpp>
#include "ptxpass/core.hpp"
#include "ptx_pass_config.h"
using namespace bpftime;
using namespace attach;
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

		hook_entries[id] = std::move(entry);
		this->map_basic_info = data.map_basic_info;
		if (data.comm_shared_mem == 0) {
			SPDLOG_ERROR(
				"comm_shared_mem is null when creating CUDA attach for {}",
				func_name);
			return -1;
		}
		if (this->shared_mem_ptr == 0) {
			this->shared_mem_ptr = data.comm_shared_mem;
			SPDLOG_INFO("Cached shared_mem_ptr at {:x}",
				    (uintptr_t)this->shared_mem_ptr);
		} else if (this->shared_mem_ptr != data.comm_shared_mem) {
			SPDLOG_WARN(
				"Ignoring new comm_shared_mem {:x}; already using {:x}",
				(uintptr_t)data.comm_shared_mem,
				(uintptr_t)this->shared_mem_ptr);
		}
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
cudaError_t cuda_runtime_function__cudaLaunchKernel_ptsz(
	const void *func, dim3 gridDim, dim3 blockDim, void **args,
	size_t sharedMem, cudaStream_t stream);
CUresult cuda_driver_function__cuGraphAddKernelNode_v1(
	CUgraphNode *phGraphNode, CUgraph hGraph,
	const CUgraphNode *dependencies, size_t numDependencies,
	const CUDA_KERNEL_NODE_PARAMS_v1 *nodeParams);
CUresult cuda_driver_function__cuGraphAddKernelNode_v2(
	CUgraphNode *phGraphNode, CUgraph hGraph,
	const CUgraphNode *dependencies, size_t numDependencies,
	const CUDA_KERNEL_NODE_PARAMS_v2 *nodeParams);
CUresult cuda_driver_function__cuGraphExecKernelNodeSetParams_v1(
	CUgraphExec hGraphExec, CUgraphNode hNode,
	const CUDA_KERNEL_NODE_PARAMS_v1 *nodeParams);
CUresult cuda_driver_function__cuGraphExecKernelNodeSetParams_v2(
	CUgraphExec hGraphExec, CUgraphNode hNode,
	const CUDA_KERNEL_NODE_PARAMS_v2 *nodeParams);
CUresult cuda_driver_function__cuGraphKernelNodeSetParams_v1(
	CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS_v1 *nodeParams);
CUresult cuda_driver_function__cuGraphKernelNodeSetParams_v2(
	CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS_v2 *nodeParams);
cudaError_t cuda_runtime_function__cudaMemcpyFromSymbol(
	void *dst, const void *symbol, size_t count, size_t offset = 0,
	cudaMemcpyKind kind = cudaMemcpyDeviceToHost);
cudaError_t cuda_runtime_function__cudaMemcpyFromSymbolAsync(
	void *dst, const void *symbol, size_t count, size_t offset,
	cudaMemcpyKind kind, cudaStream_t stream = 0);

CUresult cuda_driver_function__cuModuleGetFunction(CUfunction *hfunc,
						   CUmodule hmod,
						   const char *name);
CUresult cuda_driver_function__cuLaunchKernel(
	CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
	unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
	unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
	void **kernelParams, void **extra);

CUresult cuda_driver_function__cuMemcpyHtoD_v2(CUdeviceptr dstDevice,
					       const void *srcHost,
					       size_t ByteCount);
CUresult cuda_driver_function__cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice,
						    const void *srcHost,
						    size_t ByteCount,
						    CUstream hStream);
CUresult cuda_driver_function__cuMemcpyDtoH_v2(void *dstHost,
					       CUdeviceptr srcDevice,
					       size_t ByteCount);
CUresult cuda_driver_function__cuMemcpyDtoHAsync_v2(void *dstHost,
						    CUdeviceptr srcDevice,
						    size_t ByteCount,
						    CUstream hStream);
CUresult cuda_driver_function__cuMemcpyDtoD_v2(CUdeviceptr dstDevice,
					       CUdeviceptr srcDevice,
					       size_t ByteCount);
CUresult cuda_driver_function__cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice,
						    CUdeviceptr srcDevice,
						    size_t ByteCount,
						    CUstream hStream);
CUresult cuda_driver_function__cuMemsetD8Async(CUdeviceptr dstDevice,
					       unsigned char uc,
					       size_t N, CUstream hStream);
CUresult cuda_driver_function__cuMemsetD32Async(CUdeviceptr dstDevice,
						unsigned int ui, size_t N,
						CUstream hStream);
CUresult cuda_driver_function__cuMemAlloc_v2(CUdeviceptr *dptr,
					     size_t bytesize);
CUresult cuda_driver_function__cuMemFree_v2(CUdeviceptr dptr);
CUresult cuda_driver_function__cuMemAllocAsync(CUdeviceptr *dptr,
					       size_t bytesize,
					       CUstream hStream);
CUresult cuda_driver_function__cuMemFreeAsync(CUdeviceptr dptr,
					      CUstream hStream);
CUresult cuda_driver_function__cuStreamSynchronize(CUstream hStream);
CUresult cuda_driver_function__cuCtxSynchronize();
CUresult cuda_driver_function__cuCtxDestroy(CUcontext ctx);
CUresult cuda_driver_function__cuCtxDestroy_v2(CUcontext ctx);
CUresult cuda_driver_function__cuDevicePrimaryCtxRelease(CUdevice dev);
CUresult cuda_driver_function__cuDevicePrimaryCtxRelease_v2(CUdevice dev);
CUresult cuda_driver_function__cuEventRecord(CUevent hEvent, CUstream hStream);
CUresult cuda_driver_function__cuEventSynchronize(CUevent hEvent);
	CUresult cuda_driver_function__cuGraphLaunch(CUgraphExec hGraphExec,
						     CUstream hStream);
	CUresult cuda_driver_function__cuLinkAddData(CUlinkState state,
						     CUjitInputType type,
						     void *data, size_t size,
						     const char *name,
						     unsigned int numOptions,
						     CUjit_option *options,
						     void **optionValues);
	CUresult cuda_driver_function__cuLinkAddData_v2(CUlinkState state,
							CUjitInputType type,
							void *data, size_t size,
							const char *name,
							unsigned int numOptions,
							CUjit_option *options,
							void **optionValues);
	CUresult cuda_driver_function__cuLinkComplete(CUlinkState state,
						      void **cubinOut,
						      size_t *sizeOut);
	CUresult cuda_driver_function__cuLinkDestroy(CUlinkState state);
	CUresult cuda_driver_function__cuStreamCreate(CUstream *phStream,
						      unsigned int Flags);
	CUresult cuda_driver_function__cuStreamCreateWithPriority(CUstream *phStream,
								  unsigned int Flags,
								  int priority);
CUresult cuda_driver_function__cuStreamDestroy_v2(CUstream hStream);
CUresult cuda_driver_function__cuStreamWaitEvent(CUstream hStream,
						 CUevent hEvent,
						 unsigned int Flags);
CUresult cuda_driver_function__cuEventCreate(CUevent *phEvent,
					     unsigned int Flags);
CUresult cuda_driver_function__cuEventDestroy_v2(CUevent hEvent);
CUresult cuda_driver_function__cuModuleLoadData(CUmodule *module,
						const void *image);
	CUresult cuda_driver_function__cuModuleLoadDataEx(CUmodule *module,
							  const void *image,
							  unsigned int numOptions,
							  CUjit_option *options,
							  void **optionValues);
CUresult cuda_driver_function__cuModuleLoadFatBinary(CUmodule *module,
						     const void *fatCubin);
	CUresult cuda_driver_function__cuModuleLoad(CUmodule *module,
						    const char *fname);
	CUresult cuda_driver_function__cuModuleUnload(CUmodule hmod);
CUresult cuda_driver_function__cuLibraryLoadData(
	CUlibrary *library, const void *code, CUjit_option *jitOptions,
	void **jitOptionsValues, unsigned int numJitOptions,
	CUlibraryOption *libraryOptions, void **libraryOptionValues,
	unsigned int numLibraryOptions);
CUresult cuda_driver_function__cuLibraryLoadFromFile(
	CUlibrary *library, const char *fileName, CUjit_option *jitOptions,
	void **jitOptionsValues, unsigned int numJitOptions,
	CUlibraryOption *libraryOptions, void **libraryOptionValues,
	unsigned int numLibraryOptions);
CUresult cuda_driver_function__cuLibraryUnload(CUlibrary library);
CUresult cuda_driver_function__cuLibraryGetModule(CUmodule *pMod,
						  CUlibrary library);
CUresult cuda_driver_function__cuLibraryGetKernel(CUkernel *pKernel,
						  CUlibrary library,
						  const char *name);
CUresult cuda_driver_function__cuKernelGetName(const char **name,
					       CUkernel hfunc);
CUresult cuda_driver_function__cuKernelGetFunction(CUfunction *pFunc,
						   CUkernel kernel);
}

nv_attach_impl::nv_attach_impl()
{
	SPDLOG_INFO("Starting nv_attach_impl");
	// Ensure CUDA driver library is loaded before we attempt to hook driver
	// APIs such as cuGraph*.
	{
		void *handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
		if (handle == nullptr) {
			SPDLOG_DEBUG("dlopen(libcuda.so.1) failed: {}",
				     dlerror());
		}
	}
	this->module_pool = std::make_shared<
		std::map<std::string, std::shared_ptr<ptx_in_module>>>();
	this->ptx_pool =
		std::make_shared<std::map<std::string, std::vector<uint8_t>>>();

	this->shared_mem_ptr = 0;
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
	auto replace_hook = [&](const char *symbol_name, gpointer replacement,
				void **original_storage) {
		void *addr = GSIZE_TO_POINTER(
			gum_module_find_export_by_name(nullptr, symbol_name));
		if (addr == nullptr) {
			SPDLOG_DEBUG(
				"Skipping replace hook for {} - symbol not found",
				symbol_name);
			return;
		}
		if (auto err = gum_interceptor_replace(
			    interceptor, addr, replacement, this,
			    (gpointer *)original_storage);
		    err != GUM_REPLACE_OK) {
			SPDLOG_ERROR("Unable to replace {}: {}", symbol_name,
				     (int)err);
			assert(false);
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
		replace_hook("cudaLaunchKernel",
			     (gpointer)&cuda_runtime_function__cudaLaunchKernel,
			     &this->original_cuda_launch_kernel);
	}
	replace_hook("cudaLaunchKernel_ptsz",
		     (gpointer)&cuda_runtime_function__cudaLaunchKernel_ptsz,
		     &this->original_cuda_launch_kernel_ptsz);
	replace_hook("cuGraphAddKernelNode",
		     (gpointer)&cuda_driver_function__cuGraphAddKernelNode_v1,
		     &this->original_cu_graph_add_kernel_node_v1);
	replace_hook("cuGraphAddKernelNode_v2",
		     (gpointer)&cuda_driver_function__cuGraphAddKernelNode_v2,
		     &this->original_cu_graph_add_kernel_node_v2);
	replace_hook(
		"cuGraphExecKernelNodeSetParams",
		(gpointer)&cuda_driver_function__cuGraphExecKernelNodeSetParams_v1,
		&this->original_cu_graph_exec_kernel_node_set_params_v1);
	replace_hook(
		"cuGraphExecKernelNodeSetParams_v2",
		(gpointer)&cuda_driver_function__cuGraphExecKernelNodeSetParams_v2,
		&this->original_cu_graph_exec_kernel_node_set_params_v2);
	replace_hook(
		"cuGraphKernelNodeSetParams",
		(gpointer)&cuda_driver_function__cuGraphKernelNodeSetParams_v1,
		&this->original_cu_graph_kernel_node_set_params_v1);
	replace_hook(
		"cuGraphKernelNodeSetParams_v2",
		(gpointer)&cuda_driver_function__cuGraphKernelNodeSetParams_v2,
		&this->original_cu_graph_kernel_node_set_params_v2);
	replace_hook("cudaMemcpyFromSymbol",
		     (gpointer)&cuda_runtime_function__cudaMemcpyFromSymbol,
		     &this->original_cuda_memcpy_from_symbol);
	replace_hook(
		"cudaMemcpyFromSymbolAsync",
		(gpointer)&cuda_runtime_function__cudaMemcpyFromSymbolAsync,
		&this->original_cuda_memcpy_from_symbol_async);
	replace_hook("cuModuleGetFunction",
		     (gpointer)&cuda_driver_function__cuModuleGetFunction,
		     &this->original_cu_module_get_function);
	replace_hook("cuLaunchKernel",
		     (gpointer)&cuda_driver_function__cuLaunchKernel,
		     &this->original_cu_launch_kernel);
	replace_hook("cuMemcpyHtoD_v2",
		     (gpointer)&cuda_driver_function__cuMemcpyHtoD_v2,
		     &this->original_cu_memcpy_htod);
	replace_hook("cuMemcpyHtoDAsync_v2",
		     (gpointer)&cuda_driver_function__cuMemcpyHtoDAsync_v2,
		     &this->original_cu_memcpy_htod_async);
	replace_hook("cuMemcpyDtoH_v2",
		     (gpointer)&cuda_driver_function__cuMemcpyDtoH_v2,
		     &this->original_cu_memcpy_dtoh);
	replace_hook("cuMemcpyDtoHAsync_v2",
		     (gpointer)&cuda_driver_function__cuMemcpyDtoHAsync_v2,
		     &this->original_cu_memcpy_dtoh_async);
	replace_hook("cuMemcpyDtoD_v2",
		     (gpointer)&cuda_driver_function__cuMemcpyDtoD_v2,
		     &this->original_cu_memcpy_dtod);
	replace_hook("cuMemcpyDtoDAsync_v2",
		     (gpointer)&cuda_driver_function__cuMemcpyDtoDAsync_v2,
		     &this->original_cu_memcpy_dtod_async);
	replace_hook("cuMemsetD8Async",
		     (gpointer)&cuda_driver_function__cuMemsetD8Async,
		     &this->original_cu_memset_d8_async);
	replace_hook("cuMemsetD32Async",
		     (gpointer)&cuda_driver_function__cuMemsetD32Async,
		     &this->original_cu_memset_d32_async);
	replace_hook("cuMemAlloc_v2",
		     (gpointer)&cuda_driver_function__cuMemAlloc_v2,
		     &this->original_cu_mem_alloc);
	replace_hook("cuMemFree_v2",
		     (gpointer)&cuda_driver_function__cuMemFree_v2,
		     &this->original_cu_mem_free);
	replace_hook("cuMemAllocAsync",
		     (gpointer)&cuda_driver_function__cuMemAllocAsync,
		     &this->original_cu_mem_alloc_async);
	replace_hook("cuMemFreeAsync",
		     (gpointer)&cuda_driver_function__cuMemFreeAsync,
		     &this->original_cu_mem_free_async);
	replace_hook("cuStreamSynchronize",
		     (gpointer)&cuda_driver_function__cuStreamSynchronize,
		     &this->original_cu_stream_synchronize);
	replace_hook("cuCtxSynchronize",
		     (gpointer)&cuda_driver_function__cuCtxSynchronize,
		     &this->original_cu_ctx_synchronize);
	replace_hook("cuCtxDestroy",
		     (gpointer)&cuda_driver_function__cuCtxDestroy,
		     &this->original_cu_ctx_destroy);
	replace_hook("cuCtxDestroy_v2",
		     (gpointer)&cuda_driver_function__cuCtxDestroy_v2,
		     &this->original_cu_ctx_destroy_v2);
	replace_hook("cuDevicePrimaryCtxRelease",
		     (gpointer)&cuda_driver_function__cuDevicePrimaryCtxRelease,
		     &this->original_cu_device_primary_ctx_release);
	replace_hook("cuDevicePrimaryCtxRelease_v2",
		     (gpointer)&cuda_driver_function__cuDevicePrimaryCtxRelease_v2,
		     &this->original_cu_device_primary_ctx_release_v2);
	replace_hook("cuEventRecord",
		     (gpointer)&cuda_driver_function__cuEventRecord,
		     &this->original_cu_event_record);
	replace_hook("cuEventSynchronize",
		     (gpointer)&cuda_driver_function__cuEventSynchronize,
		     &this->original_cu_event_synchronize);
		replace_hook("cuGraphLaunch",
			     (gpointer)&cuda_driver_function__cuGraphLaunch,
			     &this->original_cu_graph_launch);
		replace_hook("cuLinkAddData",
			     (gpointer)&cuda_driver_function__cuLinkAddData,
			     &this->original_cu_link_add_data);
		replace_hook("cuLinkAddData_v2",
			     (gpointer)&cuda_driver_function__cuLinkAddData_v2,
			     &this->original_cu_link_add_data_v2);
		replace_hook("cuLinkComplete",
			     (gpointer)&cuda_driver_function__cuLinkComplete,
			     &this->original_cu_link_complete);
		replace_hook("cuLinkDestroy",
			     (gpointer)&cuda_driver_function__cuLinkDestroy,
			     &this->original_cu_link_destroy);
		replace_hook("cuStreamCreate",
			     (gpointer)&cuda_driver_function__cuStreamCreate,
			     &this->original_cu_stream_create);
	replace_hook("cuStreamCreateWithPriority",
		     (gpointer)&cuda_driver_function__cuStreamCreateWithPriority,
		     &this->original_cu_stream_create_with_priority);
	replace_hook("cuStreamDestroy_v2",
		     (gpointer)&cuda_driver_function__cuStreamDestroy_v2,
		     &this->original_cu_stream_destroy_v2);
	replace_hook("cuStreamWaitEvent",
		     (gpointer)&cuda_driver_function__cuStreamWaitEvent,
		     &this->original_cu_stream_wait_event);
	replace_hook("cuEventCreate",
		     (gpointer)&cuda_driver_function__cuEventCreate,
		     &this->original_cu_event_create);
	replace_hook("cuEventDestroy_v2",
		     (gpointer)&cuda_driver_function__cuEventDestroy_v2,
		     &this->original_cu_event_destroy_v2);
	replace_hook("cuModuleLoadData",
		     (gpointer)&cuda_driver_function__cuModuleLoadData,
		     &this->original_cu_module_load_data);
		replace_hook("cuModuleLoadDataEx",
			     (gpointer)&cuda_driver_function__cuModuleLoadDataEx,
			     &this->original_cu_module_load_data_ex);
	replace_hook("cuModuleLoadFatBinary",
		     (gpointer)&cuda_driver_function__cuModuleLoadFatBinary,
		     &this->original_cu_module_load_fatbinary);
		replace_hook("cuModuleLoad",
			     (gpointer)&cuda_driver_function__cuModuleLoad,
			     &this->original_cu_module_load);
		replace_hook("cuModuleUnload",
			     (gpointer)&cuda_driver_function__cuModuleUnload,
			     &this->original_cu_module_unload);
	replace_hook("cuLibraryLoadData",
		     (gpointer)&cuda_driver_function__cuLibraryLoadData,
		     &this->original_cu_library_load_data);
	replace_hook("cuLibraryLoadFromFile",
		     (gpointer)&cuda_driver_function__cuLibraryLoadFromFile,
		     &this->original_cu_library_load_from_file);
	replace_hook("cuLibraryUnload",
		     (gpointer)&cuda_driver_function__cuLibraryUnload,
		     &this->original_cu_library_unload);
	replace_hook("cuLibraryGetModule",
		     (gpointer)&cuda_driver_function__cuLibraryGetModule,
		     &this->original_cu_library_get_module);
	replace_hook("cuLibraryGetKernel",
		     (gpointer)&cuda_driver_function__cuLibraryGetKernel,
		     &this->original_cu_library_get_kernel);
	replace_hook("cuKernelGetName",
		     (gpointer)&cuda_driver_function__cuKernelGetName,
		     &this->original_cu_kernel_get_name);
	replace_hook("cuKernelGetFunction",
		     (gpointer)&cuda_driver_function__cuKernelGetFunction,
		     &this->original_cu_kernel_get_function);
	gum_interceptor_end_transaction(interceptor);

	const char *path = std::getenv("BPFTIME_CUDA_TRACE_PATH");
	if (path == nullptr || path[0] == '\0')
		path = std::getenv("BPFTIME_CUDA_LAUNCH_TRACE_PATH");
	if (path != nullptr && path[0] != '\0') {
		this->cuda_launch_trace_path = path;
		this->cuda_launch_trace_ofs.open(
			this->cuda_launch_trace_path,
			std::ios::out | std::ios::app);
		if (this->cuda_launch_trace_ofs.good()) {
			this->cuda_launch_trace_enabled = true;
			SPDLOG_INFO("CUDA tracing enabled at {}",
				    this->cuda_launch_trace_path);
		} else {
			SPDLOG_ERROR(
				"Failed to open BPFTIME_CUDA_TRACE_PATH/BPFTIME_CUDA_LAUNCH_TRACE_PATH={}",
				this->cuda_launch_trace_path);
		}
	}

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
	{
		this->ptx_compiler = *load_nv_attach_impl_ptx_compiler(
			DEFAULT_PTX_COMPILER_SHARED_LIB,
			this->ptx_compiler_dl_handle);
	}
}

static void
revert_frida_replaced_exports_if_present(GumInterceptor *interceptor,
					 const char *const *symbols,
					 size_t symbol_count)
{
	if (interceptor == nullptr || symbols == nullptr || symbol_count == 0)
		return;
	for (size_t i = 0; i < symbol_count; i++) {
		void *addr = GSIZE_TO_POINTER(
			gum_module_find_export_by_name(nullptr, symbols[i]));
		if (addr != nullptr)
			gum_interceptor_revert(interceptor, addr);
	}
}

nv_attach_impl::~nv_attach_impl()
{
	// Dump SASS sampler output before reverting any CUDA hooks.
	maybe_dump_sass_samples();

	if (frida_interceptor != nullptr) {
		auto interceptor = (GumInterceptor *)frida_interceptor;
		gum_interceptor_begin_transaction(interceptor);

		if (frida_listener != nullptr) {
			gum_interceptor_detach(
				interceptor,
				(GumInvocationListener *)frida_listener);
		}

		static const char *const replaced_symbols[] = {
			"cudaLaunchKernel",
			"cudaLaunchKernel_ptsz",
			"cuModuleGetFunction",
			"cuLaunchKernel",
			"cuMemcpyHtoD_v2",
			"cuMemcpyHtoDAsync_v2",
			"cuMemcpyDtoH_v2",
			"cuMemcpyDtoHAsync_v2",
			"cuMemcpyDtoD_v2",
			"cuMemcpyDtoDAsync_v2",
			"cuMemsetD8Async",
			"cuMemsetD32Async",
			"cuMemAlloc_v2",
			"cuMemFree_v2",
			"cuMemAllocAsync",
			"cuMemFreeAsync",
			"cuStreamSynchronize",
			"cuCtxSynchronize",
			"cuEventRecord",
			"cuEventSynchronize",
			"cuGraphLaunch",
			"cuGraphAddKernelNode",
			"cuGraphAddKernelNode_v2",
			"cuGraphExecKernelNodeSetParams",
			"cuGraphExecKernelNodeSetParams_v2",
			"cuGraphKernelNodeSetParams",
			"cuGraphKernelNodeSetParams_v2",
		};
		revert_frida_replaced_exports_if_present(
			interceptor, replaced_symbols,
			std::size(replaced_symbols));

		gum_interceptor_end_transaction(interceptor);
	}

	if (frida_listener) {
		g_object_unref(frida_listener);
		frida_listener = nullptr;
	}
	if (ptx_compiler_dl_handle) {
		dlclose(ptx_compiler_dl_handle);
	}
}

namespace
{
static bool env_truthy_local(const char *key)
{
	const char *v = std::getenv(key);
	if (!v || !*v)
		return false;
	std::string s(v);
	std::transform(s.begin(), s.end(), s.begin(),
		       [](unsigned char c) { return (char)std::tolower(c); });
	return s == "1" || s == "true" || s == "yes" || s == "y" || s == "on";
}

static std::optional<uint32_t> env_u32_local(const char *key)
{
	const char *v = std::getenv(key);
	if (!v || !*v)
		return std::nullopt;
	char *end = nullptr;
	errno = 0;
	unsigned long x = std::strtoul(v, &end, 10);
	if (errno != 0 || end == v)
		return std::nullopt;
	if (x > std::numeric_limits<uint32_t>::max())
		return std::nullopt;
	return static_cast<uint32_t>(x);
}

static uint32_t read_u32_le(const uint8_t *p)
{
	return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) |
	       (uint32_t(p[3]) << 24);
}

static std::string json_escape_local(std::string_view s)
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
			if (static_cast<unsigned char>(c) < 0x20) {
				char buf[7];
				std::snprintf(buf, sizeof(buf), "\\u%04x",
					      (unsigned)c);
				out += buf;
			} else {
				out += c;
			}
		}
	}
	return out;
}

static std::string expand_dump_path_local(std::string_view path)
{
	// Expand a minimal set of placeholders to avoid multi-process clobbering:
	// - "%p" => pid
	// - "%%" => "%"
	std::string out;
	out.reserve(path.size() + 16);
	const int pid = static_cast<int>(getpid());
	for (size_t i = 0; i < path.size(); i++) {
		const char c = path[i];
		if (c != '%' || (i + 1) >= path.size()) {
			out.push_back(c);
			continue;
		}
		const char n = path[i + 1];
		if (n == '%') {
			out.push_back('%');
			i++;
			continue;
		}
		if (n == 'p') {
			out += std::to_string(pid);
			i++;
			continue;
		}
		out.push_back('%');
	}
	return out;
}

static void dump_sass_samples_locked(bpftime::attach::nv_attach_impl &impl,
				     bool free_after)
{
	auto &st = impl.sass_sampling;
	if (!st.enabled || !st.initialized || st.device_buffer == 0 ||
	    st.device_bytes == 0)
		return;
	if (st.dump_path.empty())
		return;
	if (!impl.original_cu_memcpy_dtoh || !impl.original_cu_ctx_synchronize) {
		SPDLOG_WARN(
			"SASS sample: missing CUDA trampolines (memcpy_dtoh/ctx_sync), skip dump");
		return;
	}

	auto cuCtxSynchronize = reinterpret_cast<CUresult (*)()>(
		impl.original_cu_ctx_synchronize);
	(void)cuCtxSynchronize();

	std::vector<uint8_t> host(st.device_bytes);
	auto cuMemcpyDtoH_v2 =
		reinterpret_cast<CUresult (*)(void *, CUdeviceptr, size_t)>(
			impl.original_cu_memcpy_dtoh);
	const auto dtoh_res =
		cuMemcpyDtoH_v2(host.data(), st.device_buffer, st.device_bytes);
	if (dtoh_res != CUDA_SUCCESS) {
		// Best-effort: during process teardown the CUDA context may already be
		// destroyed. Treat deinit-related errors as non-fatal and avoid
		// spurious warnings.
		if (dtoh_res == CUDA_ERROR_NOT_INITIALIZED ||
		    dtoh_res == CUDA_ERROR_DEINITIALIZED ||
		    dtoh_res == CUDA_ERROR_INVALID_CONTEXT) {
			if (env_truthy_local("BPFTIME_CUDA_SASS_DETOUR_DEBUG")) {
				SPDLOG_INFO(
					"SASS sample: cuMemcpyDtoH_v2 failed ({}), likely context teardown; skip dump",
					int(dtoh_res));
			}
			return;
		}
		SPDLOG_WARN("SASS sample: cuMemcpyDtoH_v2 failed ({}), skip dump",
			    int(dtoh_res));
		return;
	}

	const std::string dump_path = expand_dump_path_local(st.dump_path);
	std::ofstream ofs(dump_path, std::ios::out | std::ios::trunc);
	if (!ofs) {
		SPDLOG_WARN("SASS sample: failed to open dump path {}", dump_path);
		return;
	}

	// Optional: include control header + slots in meta for identify-closure bring-up.
	std::optional<sass_detour::Sm120SassControlHeader> ctrl_hdr;
	std::optional<std::array<uint32_t, sass_detour::kSm120SassControlSlotsCount>>
		ctrl_slots;
	if (st.control_enabled &&
	    st.buffer_data_offset == sass_detour::kSm120SassControlDataOffset &&
	    host.size() >= sass_detour::kSm120SassControlHeaderBytes) {
		sass_detour::Sm120SassControlHeader hdr {};
		std::memcpy(&hdr, host.data(),
			    std::min(host.size(), sizeof(hdr)));
		ctrl_hdr = hdr;
		std::array<uint32_t, sass_detour::kSm120SassControlSlotsCount> slots {};
		const size_t slots_off = sass_detour::kSm120SassControlSlotsOffset;
		if (slots_off + slots.size() * sizeof(uint32_t) <= host.size()) {
			std::memcpy(slots.data(), host.data() + slots_off,
				    slots.size() * sizeof(uint32_t));
			ctrl_slots = slots;
		}
	}

	if (st.mode == sass_detour::Sm120SamplingConfig::Mode::SmidBitmap) {
		const uint8_t *data = host.data() + std::min<size_t>(host.size(), st.buffer_data_offset);
		const size_t data_size = host.size() - std::min<size_t>(host.size(), st.buffer_data_offset);
		ofs << "{\"type\":\"bpftime_sass_sample_meta\","
		    << "\"mode\":\"smid_bitmap\","
		    << "\"data_offset\":" << st.buffer_data_offset << ","
		    << "\"device_ptr\":\"0x" << std::hex
		    << (uint64_t)st.device_buffer << std::dec << "\","
		    << "\"device_bytes\":" << st.device_bytes << ","
		    << "\"desc_ur\":" << unsigned(st.desc_ur);
		if (ctrl_hdr) {
			ofs << ",\"control\":{"
			    << "\"enable\":" << ctrl_hdr->enable << ","
			    << "\"mode\":" << ctrl_hdr->mode << ","
			    << "\"epoch\":" << ctrl_hdr->epoch << ","
			    << "\"target_func_id\":" << ctrl_hdr->target_func_id << ","
			    << "\"reserved0\":" << ctrl_hdr->reserved0 << ","
			    << "\"reserved1\":" << ctrl_hdr->reserved1;
			if (ctrl_slots) {
				ofs << ",\"slots\":[";
				for (size_t i = 0; i < ctrl_slots->size(); i++) {
					if (i)
						ofs << ",";
					ofs << (*ctrl_slots)[i];
				}
				ofs << "]";
			}
			ofs << "}";
		}
		ofs << "}\n";
		const size_t dwords =
			std::min<size_t>(256, data_size / sizeof(uint32_t));
		size_t set_cnt = 0;
		for (size_t smid = 0; smid < dwords; smid++) {
			const uint32_t v =
				read_u32_le(data + smid * 4);
			if (v == 0u)
				continue;
			set_cnt++;
			ofs << "{\"type\":\"bpftime_sass_smid\","
			    << "\"smid\":" << smid << ","
			    << "\"value\":" << v << "}\n";
		}
		st.dumped_once = set_cnt > 0;
		SPDLOG_INFO("SASS sample: dumped {} smids to {}", set_cnt, dump_path);
	} else if (st.mode == sass_detour::Sm120SamplingConfig::Mode::CtaSmid) {
		const uint8_t *data = host.data() + std::min<size_t>(host.size(), st.buffer_data_offset);
		const size_t data_size = host.size() - std::min<size_t>(host.size(), st.buffer_data_offset);
		const uint32_t dump_max =
			env_u32_local("BPFTIME_CUDA_SASS_SAMPLE_DUMP_MAX")
				.value_or(10000u);
		const uint32_t n = std::min<uint32_t>(
			{ st.max_records, dump_max, (uint32_t)(data_size / 4) });
		ofs << "{\"type\":\"bpftime_sass_sample_meta\","
		    << "\"mode\":\"ctaid_smid\","
		    << "\"data_offset\":" << st.buffer_data_offset << ","
		    << "\"device_ptr\":\"0x" << std::hex
		    << (uint64_t)st.device_buffer << std::dec << "\","
		    << "\"device_bytes\":" << st.device_bytes << ","
		    << "\"max_ctas\":" << st.max_records << ","
		    << "\"dump_max\":" << dump_max << ","
		    << "\"desc_ur\":" << unsigned(st.desc_ur);
		if (ctrl_hdr) {
			ofs << ",\"control\":{"
			    << "\"enable\":" << ctrl_hdr->enable << ","
			    << "\"mode\":" << ctrl_hdr->mode << ","
			    << "\"epoch\":" << ctrl_hdr->epoch << ","
			    << "\"target_func_id\":" << ctrl_hdr->target_func_id << ","
			    << "\"reserved0\":" << ctrl_hdr->reserved0 << ","
			    << "\"reserved1\":" << ctrl_hdr->reserved1;
			if (ctrl_slots) {
				ofs << ",\"slots\":[";
				for (size_t i = 0; i < ctrl_slots->size(); i++) {
					if (i)
						ofs << ",";
					ofs << (*ctrl_slots)[i];
				}
				ofs << "]";
			}
			ofs << "}";
		}
		ofs << "}\n";
		uint32_t dumped = 0;
		for (uint32_t ctaid_x = 0; ctaid_x < n; ctaid_x++) {
			const size_t off = size_t(ctaid_x) * 4u;
			const uint32_t raw = read_u32_le(data + off);
			if (raw == 0xffffffffu)
				continue;
			const uint32_t smid_hi = (raw >> 8) & 0xffu;
			const uint32_t smid_lo = raw & 0xffu;
			ofs << "{\"type\":\"bpftime_sass_cta\","
			    << "\"ctaid_x\":" << ctaid_x << ","
			    << "\"smid_raw\":" << raw << ","
			    << "\"smid_hi8\":" << smid_hi << ","
			    << "\"smid_lo8\":" << smid_lo << "}\n";
			dumped++;
		}
		st.dumped_once = dumped > 0;
		SPDLOG_INFO("SASS sample: dumped {} ctAs to {}", dumped, dump_path);
	} else if (st.mode == sass_detour::Sm120SamplingConfig::Mode::PcMarker) {
		const size_t data_off =
			std::min<size_t>(host.size(), st.buffer_data_offset);
		const uint8_t *data = host.data() + data_off;
		const size_t data_size = host.size() - data_off;
		const uint32_t dump_max =
			env_u32_local("BPFTIME_CUDA_SASS_SAMPLE_DUMP_MAX")
				.value_or(2000u);
		const uint32_t dump_cap = std::min<uint32_t>(dump_max, 20000u);
		const uint32_t ring_entries = std::max<uint32_t>(1u, st.marker_ring_entries);
		const uint32_t record_words = 6;
		const uint32_t record_bytes = record_words * 4u; // 24
		const uint32_t header_bytes = 0x10u;
		if (data_size < header_bytes) {
			SPDLOG_WARN("SASS marker: buffer too small ({} bytes)", data_size);
			return;
		}
		const uint32_t write_idx = read_u32_le(data + 0x0);
		ofs << "{\"type\":\"bpftime_sass_sample_meta\","
		    << "\"mode\":\"pc_marker\","
		    << "\"data_offset\":" << st.buffer_data_offset << ","
		    << "\"device_ptr\":\"0x" << std::hex
		    << (uint64_t)st.device_buffer << std::dec << "\","
		    << "\"device_bytes\":" << st.device_bytes << ","
		    << "\"write_idx\":" << write_idx << ","
		    << "\"ring_entries\":" << ring_entries << ","
		    << "\"record_bytes\":" << record_bytes << ","
		    << "\"max_ctas\":" << st.max_records << ","
		    << "\"cta_clamp\":" << (st.marker_cta_clamp ? "true" : "false") << ","
		    << "\"lane0_only\":" << (st.marker_lane0_only ? "true" : "false") << ","
		    << "\"warp0_only\":" << (st.marker_warp0_only ? "true" : "false") << ","
		    << "\"dump_max\":" << dump_cap << ","
		    << "\"desc_ur\":" << unsigned(st.desc_ur);
		if (ctrl_hdr) {
			ofs << ",\"control\":{"
			    << "\"enable\":" << ctrl_hdr->enable << ","
			    << "\"mode\":" << ctrl_hdr->mode << ","
			    << "\"epoch\":" << ctrl_hdr->epoch << ","
			    << "\"target_func_id\":" << ctrl_hdr->target_func_id << ","
			    << "\"reserved0\":" << ctrl_hdr->reserved0 << ","
			    << "\"reserved1\":" << ctrl_hdr->reserved1;
			if (ctrl_slots) {
				ofs << ",\"slots\":[";
				for (size_t i = 0; i < ctrl_slots->size(); i++) {
					if (i)
						ofs << ",";
					ofs << (*ctrl_slots)[i];
				}
				ofs << "]";
			}
			ofs << "}";
		}
		ofs << "}\n";

		const uint8_t *rec_base = data + header_bytes;
		const size_t rec_bytes_total = data_size - header_bytes;
		const uint32_t max_records_in_buf =
			std::min<uint32_t>(ring_entries,
					   (uint32_t)(rec_bytes_total / record_bytes));
		uint32_t dumped = 0;
		for (uint32_t i = 0; i < max_records_in_buf && dumped < dump_cap; i++) {
			const size_t off = size_t(i) * size_t(record_bytes);
			const uint32_t seq = read_u32_le(rec_base + off + 0);
			const uint32_t marker_off = read_u32_le(rec_base + off + 4);
			if (marker_off == 0xffffffffu)
				continue;
			const uint32_t tag = read_u32_le(rec_base + off + 8);
			const uint32_t smid_raw = read_u32_le(rec_base + off + 12);
			const uint32_t ctaid_x = read_u32_le(rec_base + off + 16);
			const uint32_t tid_x = read_u32_le(rec_base + off + 20);
			const uint32_t lane_id = tid_x & 31u;
			const uint32_t warp_id = (tid_x >> 5) & 31u;
			ofs << "{\"type\":\"bpftime_sass_marker\","
			    << "\"slot\":" << i << ","
			    << "\"seq\":" << seq << ","
			    << "\"marker_off\":" << marker_off << ","
			    << "\"tag\":" << tag << ","
			    << "\"smid_raw\":" << smid_raw << ","
			    << "\"ctaid_x\":" << ctaid_x << ","
			    << "\"tid_x\":" << tid_x << ","
			    << "\"warp_id\":" << warp_id << ","
			    << "\"lane_id\":" << lane_id << "}\n";
			dumped++;
		}
		st.dumped_once = dumped > 0;
		SPDLOG_INFO("SASS marker: dumped {} records to {}", dumped, dump_path);
	} else {
		const size_t data_off =
			std::min<size_t>(host.size(), st.buffer_data_offset);
		const uint8_t *data = host.data() + data_off;
		const size_t data_size = host.size() - data_off;
		const bool thread_mode =
			(st.mode ==
			 sass_detour::Sm120SamplingConfig::Mode::ThreadMap);
		const uint32_t warps_per_cta = 32;
		const uint32_t lanes_per_warp = 32;
		const bool thread_device = thread_mode && st.thread_map_device;
		const uint32_t per_cta_slots = thread_device ? 1024u : warps_per_cta;
		const uint32_t record_bytes =
			thread_device ? (st.thread_map_device_stride4 ? 4u : 1u)
				      : 4u;
		const uint32_t max_entries = std::min<uint32_t>(
			st.max_records,
			(uint32_t)std::max<size_t>(
				1, data_size / (size_t)per_cta_slots /
					   (size_t)record_bytes));
		const uint32_t dump_max =
			env_u32_local("BPFTIME_CUDA_SASS_SAMPLE_DUMP_MAX")
				.value_or(2000u);
		const uint32_t dump_cap = std::min<uint32_t>(dump_max, 20000u);
			ofs << "{\"type\":\"bpftime_sass_sample_meta\","
				    << "\"mode\":\"" << (thread_mode ? "thread_map" : "warp_map")
				    << "\","
				    << "\"data_offset\":" << st.buffer_data_offset << ","
				    << "\"thread_device\":" << (thread_device ? "true" : "false")
				    << ","
				    << "\"thread_stride_bytes\":"
				    << (thread_device ? record_bytes : 0u) << ","
				    << "\"cta_clamp\":"
				    << ((thread_device && st.thread_map_device_cta_clamp) ? "true"
										  : "false")
				    << ","
				    << "\"device_ptr\":\"0x" << std::hex
				    << (uint64_t)st.device_buffer << std::dec << "\","
				    << "\"device_bytes\":" << st.device_bytes << ","
			    << "\"max_ctas\":" << max_entries << ","
		    << "\"per_cta\":" << per_cta_slots << ","
		    << "\"expanded_per_cta\":"
		    << (thread_mode ? (warps_per_cta * lanes_per_warp) : per_cta_slots)
		    << ","
		    << "\"record_bytes\":" << record_bytes << ","
		    << "\"dump_max\":" << dump_cap << ","
		    << "\"desc_ur\":" << unsigned(st.desc_ur);
		if (ctrl_hdr) {
			ofs << ",\"control\":{"
			    << "\"enable\":" << ctrl_hdr->enable << ","
			    << "\"mode\":" << ctrl_hdr->mode << ","
			    << "\"epoch\":" << ctrl_hdr->epoch << ","
			    << "\"target_func_id\":" << ctrl_hdr->target_func_id << ","
			    << "\"reserved0\":" << ctrl_hdr->reserved0 << ","
			    << "\"reserved1\":" << ctrl_hdr->reserved1;
			if (ctrl_slots) {
				ofs << ",\"slots\":[";
				for (size_t i = 0; i < ctrl_slots->size(); i++) {
					if (i)
						ofs << ",";
					ofs << (*ctrl_slots)[i];
				}
				ofs << "]";
			}
			ofs << "}";
		}
		ofs << "}\n";
			uint32_t dumped = 0;
			const bool dbg_raw_u32 =
				env_truthy_local("BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_DEBUG_STORE_TID") ||
				env_truthy_local("BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_DEBUG_STORE_IDX");
			for (uint32_t cta = 0; cta < max_entries && dumped < dump_cap; cta++) {
				for (uint32_t j = 0; j < per_cta_slots && dumped < dump_cap; j++) {
				const uint32_t idx = cta * per_cta_slots + j;
				const size_t off = size_t(idx) * (size_t)record_bytes;
				if (off + record_bytes > data_size)
					break;
				if (!thread_mode) {
					const uint32_t a =
						read_u32_le(data + off + 0);
					if (a == 0xffffffffu)
						continue;
					const uint32_t smid_lo8 = a & 0xffu;
					ofs << "{\"type\":\"bpftime_sass_warp\","
					    << "\"slot\":" << idx << ","
					    << "\"smid_raw\":" << a << ","
					    << "\"smid_lo8\":" << smid_lo8 << ","
					    << "\"ctaid_x\":" << cta << ","
					    << "\"warp_id\":" << j << "}\n";
					dumped++;
					continue;
				}

						if (thread_device) {
							uint32_t smid_lo8 = 0;
							uint32_t raw_u32 = 0;
							if (record_bytes == 1) {
								const uint8_t a = data[off];
								if (a == 0xffu)
									continue;
								smid_lo8 = unsigned(a);
							} else {
								const uint32_t a =
									read_u32_le(data + off);
								if (a == 0xffffffffu)
									continue;
								raw_u32 = a;
								smid_lo8 = a & 0xffu;
							}
							const uint32_t tid_x = j & 1023u;
							const uint32_t lane_id = tid_x & 31u;
							const uint32_t warp_id =
								(tid_x >> 5) & 31u;
							ofs << "{\"type\":\"bpftime_sass_thread\","
							    << "\"slot\":" << idx << ","
							    << (dbg_raw_u32 && record_bytes == 4
									    ? "\"raw_u32\":" +
										      std::to_string(raw_u32) + ","
									    : "")
							    << "\"smid_lo8\":" << smid_lo8
							    << ","
							    << "\"ctaid_x\":" << cta << ","
						    << "\"tid_x\":" << tid_x << ","
						    << "\"warp_id\":" << warp_id << ","
					    << "\"lane_id\":" << lane_id << "}\n";
					dumped++;
					continue;
				}

				// Host-expanded (per-warp slots => 32 lanes).
				const uint32_t a =
					read_u32_le(data + off + 0);
				if (a == 0xffffffffu)
					continue;
				const uint32_t smid_lo8 = a & 0xffu;
				for (uint32_t lane = 0; lane < lanes_per_warp &&
						     dumped < dump_cap;
				     lane++) {
					const uint32_t tid_x =
						j * lanes_per_warp + lane;
					const uint32_t thread_slot =
						cta * (warps_per_cta * lanes_per_warp) +
						tid_x;
					ofs << "{\"type\":\"bpftime_sass_thread\","
					    << "\"slot\":" << thread_slot << ","
					    << "\"smid_lo8\":" << smid_lo8 << ","
					    << "\"ctaid_x\":" << cta << ","
					    << "\"tid_x\":" << tid_x << ","
					    << "\"warp_id\":" << j << ","
					    << "\"lane_id\":" << lane << "}\n";
					dumped++;
				}
			}
		}
		st.dumped_once = dumped > 0;
		SPDLOG_INFO("SASS sample: dumped {} records to {}", dumped, dump_path);
	}
	ofs.flush();

	if (free_after) {
		if (impl.original_cu_mem_free) {
			auto cuMemFree_v2 =
				reinterpret_cast<CUresult (*)(CUdeviceptr)>(
					impl.original_cu_mem_free);
			(void)cuMemFree_v2(st.device_buffer);
		}
		st.device_buffer = 0;
		st.device_bytes = 0;
		st.initialized = false;
	}
}
} // namespace

std::optional<sass_detour::Sm120SamplingConfig>
nv_attach_impl::get_sm120_sampling_cfg()
{
	std::lock_guard<std::mutex> guard(sass_sampling.lock);

	sass_sampling.enabled = env_truthy_local("BPFTIME_CUDA_SASS_SAMPLE");
	if (!sass_sampling.enabled)
		return std::nullopt;

	if (auto v = std::getenv("BPFTIME_CUDA_SASS_SAMPLE_DUMP_PATH");
	    v && *v) {
		sass_sampling.dump_path = v;
	}
	if (auto v = std::getenv("BPFTIME_CUDA_SASS_SAMPLE_MODE"); v && *v) {
		std::string s(v);
		std::transform(s.begin(), s.end(), s.begin(),
			       [](unsigned char c) {
				       return (char)std::tolower(c);
			       });
		if (s.find("thread") != std::string::npos)
			sass_sampling.mode =
				sass_detour::Sm120SamplingConfig::Mode::ThreadMap;
		else if (s.find("marker") != std::string::npos ||
			 s.find("pc") != std::string::npos)
			sass_sampling.mode =
				sass_detour::Sm120SamplingConfig::Mode::PcMarker;
		else if (s.find("warp") != std::string::npos)
			sass_sampling.mode =
				sass_detour::Sm120SamplingConfig::Mode::WarpMap;
		else if (s.find("cta") != std::string::npos ||
			 s.find("record") != std::string::npos)
			sass_sampling.mode =
				sass_detour::Sm120SamplingConfig::Mode::CtaSmid;
		else
			sass_sampling.mode = sass_detour::Sm120SamplingConfig::Mode::
				SmidBitmap;
	}
	if (auto v = env_u32_local("BPFTIME_CUDA_SASS_SAMPLE_MAX_RECORDS"))
		sass_sampling.max_records = *v;
	// `max_records` must be power-of-two for mask-based indexing in CTA/Warp/Thread
	// modes. PcMarker uses a separate ring buffer and can use clamp-based gating,
	// so allow any value (including 0) there.
	if (sass_sampling.mode != sass_detour::Sm120SamplingConfig::Mode::SmidBitmap &&
	    sass_sampling.mode != sass_detour::Sm120SamplingConfig::Mode::PcMarker) {
		uint32_t v = std::max(1u, sass_sampling.max_records);
		if ((v & (v - 1u)) != 0) {
			uint32_t p = 1u;
			while ((p << 1u) != 0 && (p << 1u) <= v)
				p <<= 1u;
			SPDLOG_WARN(
				"SASS sample: max_records={} is not power-of-two; round down to {} (mask-based indexing)",
				v, p);
			sass_sampling.max_records = p;
		} else {
			sass_sampling.max_records = v;
		}
	}
	// PcMarker config (ring buffer + optional extra offsets + density gates).
	if (sass_sampling.mode == sass_detour::Sm120SamplingConfig::Mode::PcMarker) {
		if (auto v = env_u32_local("BPFTIME_CUDA_SASS_MARKER_RING_ENTRIES"))
			sass_sampling.marker_ring_entries = *v;
		// Keep ring entries as a power-of-two for mask-based indexing.
		{
			uint32_t v = std::max(1u, sass_sampling.marker_ring_entries);
			if ((v & (v - 1u)) != 0) {
				uint32_t p = 1u;
				while ((p << 1u) != 0 && (p << 1u) <= v)
					p <<= 1u;
				SPDLOG_WARN(
					"SASS marker: ring_entries={} is not power-of-two; round down to {}",
					v, p);
				sass_sampling.marker_ring_entries = p;
			} else {
				sass_sampling.marker_ring_entries = v;
			}
		}
		sass_sampling.marker_lane0_only =
			!env_truthy_local("BPFTIME_CUDA_SASS_MARKER_LANE0_ONLY_DISABLE");
		sass_sampling.marker_warp0_only =
			env_truthy_local("BPFTIME_CUDA_SASS_MARKER_WARP0_ONLY");
		sass_sampling.marker_cta_clamp =
			!env_truthy_local("BPFTIME_CUDA_SASS_MARKER_CTA_CLAMP_DISABLE");
		sass_sampling.marker_offsets.clear();
		if (auto v = std::getenv("BPFTIME_CUDA_SASS_MARKER_OFFSETS"); v && *v) {
			// Comma/space separated list of byte offsets (hex or dec) relative to
			// the `.text.*` section start.
			std::string s(v);
			for (char &c : s) {
				if (c == ',' || c == ';')
					c = ' ';
			}
			std::istringstream iss(s);
			std::string tok;
			while (iss >> tok) {
				char *end = nullptr;
				unsigned long long off =
					std::strtoull(tok.c_str(), &end, 0);
				if (!end || *end != '\0')
					continue;
				sass_sampling.marker_offsets.push_back(
					static_cast<uint32_t>(off));
				if (sass_sampling.marker_offsets.size() >= 16)
					break;
			}
		}
	}
	// Configure thread-map layout before allocating the buffer.
		sass_sampling.thread_map_device = false;
		sass_sampling.thread_map_device_lane0_only = false;
		sass_sampling.thread_map_device_warp0_only = false;
		sass_sampling.thread_map_device_stride4 = false;
		sass_sampling.thread_map_device_cta_clamp = false;
		sass_sampling.thread_map_device_no_store = false;
		if (sass_sampling.mode == sass_detour::Sm120SamplingConfig::Mode::ThreadMap) {
			sass_sampling.thread_map_device =
				env_truthy_local("BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE");
			if (sass_sampling.thread_map_device) {
				sass_sampling.thread_map_device_lane0_only = env_truthy_local(
					"BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_LANE0_ONLY");
				sass_sampling.thread_map_device_warp0_only = env_truthy_local(
					"BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_WARP0_ONLY");
				sass_sampling.thread_map_device_stride4 = env_truthy_local(
					"BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_STRIDE4");
				sass_sampling.thread_map_device_cta_clamp = env_truthy_local(
					"BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_CTA_CLAMP");
				sass_sampling.thread_map_device_no_store = env_truthy_local(
					"BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_NO_STORE");
				sass_sampling.reg255_thread_map_spill_enable =
					env_truthy_local(
						"BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_REG255_SPILL");
				sass_sampling.reg255_thread_map_spill_per_thread =
					env_truthy_local(
						"BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_REG255_SPILL_PER_THREAD");
				sass_sampling.reg255_thread_map_spill_bytes = 0;
				sass_sampling.reg255_thread_map_spill_stride_bytes = 0;
				if (sass_sampling.reg255_thread_map_spill_enable) {
					if (sass_sampling.reg255_thread_map_spill_per_thread) {
						uint32_t stride = 0x10u;
						if (auto v = env_u32_local(
							    "BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_REG255_SPILL_STRIDE_BYTES"))
							stride = std::min<uint32_t>(*v, 0x100u);
						// Keep imm8-friendly offsets; align to 16.
						stride = (stride + 0x0fu) & ~0x0fu;
						sass_sampling.reg255_thread_map_spill_stride_bytes =
							stride;
						const size_t total =
							(size_t)std::max(1u, sass_sampling.max_records) *
							size_t(1024) * size_t(stride);
						// Safety: avoid huge allocations by default.
						const size_t cap = size_t(64) * 1024 * 1024;
						if (total > cap) {
							SPDLOG_WARN(
								"SASS sample: reg255 per-thread spill would allocate {} bytes (>64MiB); disable spill (reduce max_records or stride)",
								total);
							sass_sampling.reg255_thread_map_spill_enable =
								false;
							sass_sampling.reg255_thread_map_spill_per_thread =
								false;
							sass_sampling.reg255_thread_map_spill_stride_bytes =
								0;
							sass_sampling.reg255_thread_map_spill_bytes = 0;
						} else {
							sass_sampling.reg255_thread_map_spill_bytes =
								uint32_t(total);
						}
					} else {
						if (auto v = env_u32_local(
							    "BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_REG255_SPILL_BYTES")) {
							sass_sampling.reg255_thread_map_spill_bytes =
								std::min<uint32_t>(*v, 4096u);
						} else {
							sass_sampling.reg255_thread_map_spill_bytes =
								256u;
						}
						// Keep imm8-friendly offsets; align to 16.
						sass_sampling.reg255_thread_map_spill_bytes =
							(sass_sampling.reg255_thread_map_spill_bytes +
							 0x0fu) &
							~0x0fu;
					}
				}
			}
		}
	if (auto v = env_u32_local("BPFTIME_CUDA_SASS_SAMPLE_DESC_UR"))
		sass_sampling.desc_ur = static_cast<uint8_t>(*v & 0xffu);
		// Use a high UR pair by default to minimize clobbering of compiler-selected
		// UR registers in kernel prologues (e.g., UR4/UR5 are commonly used).
		// Override via `BPFTIME_CUDA_SASS_SAMPLE_DESC_UR` if needed.
		if (!std::getenv("BPFTIME_CUDA_SASS_SAMPLE_DESC_UR"))
			sass_sampling.desc_ur = 62;
	// Descriptor uses a UR pair (URx/URx+1), so require an even base within
	// the architectural range.
	if ((sass_sampling.desc_ur & 1u) != 0 || sass_sampling.desc_ur > 62) {
		SPDLOG_WARN(
			"SASS sample: invalid desc_ur={}, forcing to 62",
			(unsigned)sass_sampling.desc_ur);
		sass_sampling.desc_ur = 62;
	}

	if (!sass_sampling.initialized) {
		if (!original_cu_mem_alloc || !original_cu_memset_d32_async ||
		    !original_cu_ctx_synchronize) {
			SPDLOG_WARN(
				"SASS sample: missing CUDA trampolines (mem_alloc/memset/ctx_sync), disable");
			return std::nullopt;
		}

		sass_sampling.control_enabled =
			env_truthy_local("BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE");
		sass_sampling.buffer_data_offset =
			sass_sampling.control_enabled
				? sass_detour::kSm120SassControlDataOffset
				: 0u;

		const size_t thread_map_stride_bytes =
			sass_sampling.thread_map_device
				? (sass_sampling.thread_map_device_stride4 ? 4u : 1u)
				: 4u;
		const size_t marker_record_bytes = size_t(6) * sizeof(uint32_t);
		const size_t marker_header_bytes = 0x10u;
		const size_t data_bytes =
			(sass_sampling.mode ==
			 sass_detour::Sm120SamplingConfig::Mode::SmidBitmap)
				? (size_t(256) * sizeof(uint32_t))
			: (sass_sampling.mode ==
				   sass_detour::Sm120SamplingConfig::Mode::CtaSmid)
					  ? (size_t(std::max(1u, sass_sampling.max_records)) *
					     sizeof(uint32_t))
					  : (sass_sampling.mode ==
						     sass_detour::Sm120SamplingConfig::Mode::
							     PcMarker)
						    ? (marker_header_bytes +
						       size_t(std::max(
							      1u,
							      sass_sampling.marker_ring_entries)) *
						       marker_record_bytes)
					  : (sass_sampling.mode ==
						     sass_detour::Sm120SamplingConfig::Mode::
							     WarpMap)
						    ? (size_t(std::max(1u, sass_sampling.max_records)) *
						       size_t(32) * sizeof(uint32_t))
						    : (sass_sampling.thread_map_device
							       ? (size_t(std::max(
									      1u,
									      sass_sampling.max_records)) *
								  size_t(1024) *
								  thread_map_stride_bytes)
							       : (size_t(std::max(
									      1u,
									      sass_sampling.max_records)) *
								  size_t(32) *
								  sizeof(uint32_t)));
		const size_t bytes = (size_t)sass_sampling.buffer_data_offset +
				     data_bytes +
				     (size_t)sass_sampling.reg255_thread_map_spill_bytes;
		CUdeviceptr dptr = 0;
		auto cuMemAlloc_v2 =
			reinterpret_cast<CUresult (*)(CUdeviceptr *, size_t)>(
				original_cu_mem_alloc);
		if (cuMemAlloc_v2(&dptr, bytes) != CUDA_SUCCESS) {
			SPDLOG_WARN("SASS sample: cuMemAlloc_v2 failed");
			return std::nullopt;
		}

		auto cuMemsetD32Async =
			reinterpret_cast<CUresult (*)(CUdeviceptr, unsigned int,
						      size_t, CUstream)>(
				original_cu_memset_d32_async);
		const size_t dwords = bytes / 4;
		(void)cuMemsetD32Async(dptr, 0u, dwords, nullptr);
		const size_t data_dwords = data_bytes / 4;
		const unsigned int init =
			(sass_sampling.mode ==
			 sass_detour::Sm120SamplingConfig::Mode::SmidBitmap)
				? 0u
				: 0xffffffffu;
		if (data_dwords != 0) {
			(void)cuMemsetD32Async(
				dptr + (CUdeviceptr)sass_sampling.buffer_data_offset, init,
				data_dwords, nullptr);
		}
		// PcMarker: keep write_idx at 0 while leaving the record region at 0xffffffff.
		if (sass_sampling.mode == sass_detour::Sm120SamplingConfig::Mode::PcMarker) {
			(void)cuMemsetD32Async(
				dptr + (CUdeviceptr)sass_sampling.buffer_data_offset, 0u,
				1, nullptr);
		}

		auto cuCtxSynchronize =
			reinterpret_cast<CUresult (*)()>(original_cu_ctx_synchronize);
		(void)cuCtxSynchronize();

		sass_sampling.device_buffer = dptr;
		sass_sampling.device_bytes = bytes;
		sass_sampling.initialized = true;
		sass_sampling.dumped_once = false;
		sass_sampling.sync_dump_attempts = 0;
		sass_sampling.dumped_on_first_launch = false;
			SPDLOG_INFO(
				"SASS sample: initialized buffer dev_ptr=0x{:x} bytes={} data_offset={} max_records={} desc_ur={} marker_ring_entries={} marker_offsets={} marker_lane0_only={} marker_warp0_only={} marker_cta_clamp={} thread_device={} lane0_only={} warp0_only={} stride4={} cta_clamp={} control={} reg255_spill={} reg255_spill_per_thread={} reg255_spill_stride={} reg255_spill_bytes={}",
				(uint64_t)dptr, bytes, sass_sampling.buffer_data_offset,
				sass_sampling.max_records, (unsigned)sass_sampling.desc_ur,
				sass_sampling.marker_ring_entries,
				sass_sampling.marker_offsets.size(),
				sass_sampling.marker_lane0_only ? "true" : "false",
				sass_sampling.marker_warp0_only ? "true" : "false",
				sass_sampling.marker_cta_clamp ? "true" : "false",
				sass_sampling.thread_map_device,
				sass_sampling.thread_map_device_lane0_only,
				sass_sampling.thread_map_device_warp0_only,
				sass_sampling.thread_map_device_stride4,
				sass_sampling.thread_map_device_cta_clamp,
				sass_sampling.control_enabled ? "true" : "false",
				sass_sampling.reg255_thread_map_spill_enable ? "true"
									     : "false",
				sass_sampling.reg255_thread_map_spill_per_thread ? "true"
										 : "false",
				sass_sampling.reg255_thread_map_spill_stride_bytes,
				sass_sampling.reg255_thread_map_spill_bytes);
		}

	return sass_detour::Sm120SamplingConfig {
		.enabled = true,
		.mode = sass_sampling.mode,
		.sample_buffer_device_ptr = (uint64_t)sass_sampling.device_buffer,
		.buffer_data_offset = sass_sampling.buffer_data_offset,
		.control_enabled = sass_sampling.control_enabled,
		.max_records =
			(sass_sampling.mode ==
			 sass_detour::Sm120SamplingConfig::Mode::SmidBitmap)
				? 0u
				: (sass_sampling.mode ==
						   sass_detour::Sm120SamplingConfig::Mode::PcMarker
					   ? sass_sampling.max_records
					   : std::max(1u, sass_sampling.max_records)),
		.thread_map_device = sass_sampling.thread_map_device,
		.thread_map_device_lane0_only =
			sass_sampling.thread_map_device_lane0_only,
		.thread_map_device_warp0_only =
			sass_sampling.thread_map_device_warp0_only,
			.thread_map_device_stride4 =
				sass_sampling.thread_map_device_stride4,
			.thread_map_device_cta_clamp =
				sass_sampling.thread_map_device_cta_clamp,
			.thread_map_device_no_store =
				sass_sampling.thread_map_device_no_store,
		.reg255_thread_map_spill_enable =
			sass_sampling.reg255_thread_map_spill_enable,
		.reg255_thread_map_spill_per_thread =
			sass_sampling.reg255_thread_map_spill_per_thread,
		.reg255_thread_map_spill_stride_bytes =
			sass_sampling.reg255_thread_map_spill_stride_bytes,
		.reg255_thread_map_spill_bytes =
			sass_sampling.reg255_thread_map_spill_bytes,
		.marker_ring_entries =
			(sass_sampling.mode ==
			 sass_detour::Sm120SamplingConfig::Mode::PcMarker)
				? std::max(1u, sass_sampling.marker_ring_entries)
				: 0u,
		.marker_offsets = sass_sampling.marker_offsets,
		.marker_lane0_only = sass_sampling.marker_lane0_only,
		.marker_warp0_only = sass_sampling.marker_warp0_only,
		.marker_cta_clamp = sass_sampling.marker_cta_clamp,
		.desc_ur = sass_sampling.desc_ur,
	};
}

void nv_attach_impl::record_sass_sampled_kernels(
	const std::vector<sass_detour::InstrumentedKernelInfo> &kernels)
{
	if (kernels.empty())
		return;
	std::lock_guard<std::mutex> guard(sass_sampling.lock);
	for (const auto &k : kernels) {
		if (k.tag == 0 || k.kernel_name.empty())
			continue;
		sass_sampling.tag_to_kernel.emplace(k.tag, k.kernel_name);
	}
}

void nv_attach_impl::maybe_dump_sass_samples()
{
	std::lock_guard<std::mutex> guard(sass_sampling.lock);
	if (!sass_sampling.dump_on_exit)
		return;
	// If we've already produced a valid dump (via dump-on-sync or other trigger),
	// avoid issuing CUDA API calls again during process teardown. Some workloads
	// destroy the CUDA context before our destructor runs, and a second dump
	// attempt will fail with deinit/invalid-context errors.
	if (sass_sampling.dumped_once)
		return;
	dump_sass_samples_locked(*this, /*free_after=*/true);
}

void nv_attach_impl::maybe_dump_sass_samples_on_sync()
{
	if (!env_truthy_local("BPFTIME_CUDA_SASS_SAMPLE_DUMP_ON_SYNC"))
		return;
	std::lock_guard<std::mutex> guard(sass_sampling.lock);
	if (sass_sampling.dumped_once)
		return;
	const bool allow_retry =
		sass_sampling.control_enabled &&
		env_truthy_local("BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE");
	// Avoid consuming the limited "dump on sync" attempts before any matching
	// kernel has had a chance to arm identify or set a target func_id. Real
	// vLLM workloads can issue many early syncs during initialization.
	if (allow_retry) {
		const bool pending =
			sass_detour_filter_state.identify_pending.load(
				std::memory_order_acquire);
		const bool has_any_target =
			(sass_detour_filter_state.identify_target_func_id != 0u) ||
			(sass_detour_filter_state.identify_epoch != 0u);
		if (!pending && !has_any_target)
			return;
	}
	const uint32_t max_attempts =
		allow_retry
			? env_u32_local(
				  "BPFTIME_CUDA_SASS_SAMPLE_SYNC_DUMP_MAX_ATTEMPTS")
				  .value_or(4u)
			: 1u;
	if (sass_sampling.sync_dump_attempts >= max_attempts)
		return;
	sass_sampling.sync_dump_attempts++;
	dump_sass_samples_locked(*this, /*free_after=*/false);
}

void nv_attach_impl::dump_sass_samples_force()
{
	std::lock_guard<std::mutex> guard(sass_sampling.lock);
	dump_sass_samples_locked(*this, /*free_after=*/false);
}

bool nv_attach_impl::can_patch_ptx() const
{
	return this->shared_mem_ptr != 0 && !this->hook_entries.empty();
}

namespace
{
static std::string json_escape(std::string_view s)
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
} // namespace

static uint64_t fnv1a64_local(std::span<const uint8_t> data)
{
	uint64_t h = 1469598103934665603ull;
	for (uint8_t b : data) {
		h ^= uint64_t(b);
		h *= 1099511628211ull;
	}
	return h;
}

static bool safe_read_u64_local(const void *remote, uint64_t *out)
{
	if (!remote || !out)
		return false;
	iovec local_iov { out, sizeof(uint64_t) };
	iovec remote_iov { const_cast<void *>(remote), sizeof(uint64_t) };
	const ssize_t n =
		process_vm_readv(getpid(), &local_iov, 1, &remote_iov, 1, 0);
	return n == (ssize_t)sizeof(uint64_t);
}

uint64_t nv_attach_impl::trace_cuda_kernel_launch(
	const std::string &kernel_name, int grid_x, int grid_y, int grid_z,
	int block_x, int block_y, int block_z, size_t shared_mem, void *stream,
	void **kernel_params, void **extra)
{
	if (env_truthy_local(
		    "BPFTIME_CUDA_SASS_SAMPLE_DUMP_ON_FIRST_SAMPLED_LAUNCH")) {
		std::lock_guard<std::mutex> guard(sass_sampling.lock);
		if (sass_sampling.enabled && sass_sampling.initialized &&
		    !sass_sampling.dumped_on_first_launch) {
			bool match = false;
			if (!sass_sampling.tag_to_kernel.empty()) {
				for (const auto &kv :
				     sass_sampling.tag_to_kernel) {
					const auto &k = kv.second;
					if (k.empty())
						continue;
					if (kernel_name.find(k) !=
						    std::string::npos ||
					    k.find(kernel_name) !=
						    std::string::npos) {
						match = true;
						break;
					}
				}
			}

			if (!match) {
				const char *sample_filter = std::getenv(
					"BPFTIME_CUDA_SASS_SAMPLE_FILTER");
				const char *detour_filter = std::getenv(
					"BPFTIME_CUDA_SASS_DETOUR_FILTER");
				std::string_view filter_sv =
					(sample_filter && *sample_filter)
						? std::string_view(
							  sample_filter)
					: (detour_filter && *detour_filter)
						? std::string_view(
							  detour_filter)
						: std::string_view();
				if (!filter_sv.empty() &&
				    kernel_name.find(filter_sv) !=
					    std::string::npos)
					match = true;
			}

			if (match) {
				sass_sampling.dumped_on_first_launch = true;
				dump_sass_samples_locked(*this,
							 /*free_after=*/false);
			}
		}
	} else if (env_truthy_local("BPFTIME_CUDA_SASS_SAMPLE_DUMP_ON_FIRST_LAUNCH")) {
		std::lock_guard<std::mutex> guard(sass_sampling.lock);
		if (sass_sampling.enabled && sass_sampling.initialized &&
		    !sass_sampling.dumped_on_first_launch) {
			sass_sampling.dumped_on_first_launch = true;
			dump_sass_samples_locked(*this, /*free_after=*/false);
		}
	}

	if (!cuda_launch_trace_enabled)
		return 0;
	timespec ts {};
	clock_gettime(CLOCK_MONOTONIC, &ts);
	const uint64_t ts_ns =
		static_cast<uint64_t>(ts.tv_sec) * 1000ULL * 1000ULL * 1000ULL +
		static_cast<uint64_t>(ts.tv_nsec);
	const uint64_t seq =
		cuda_launch_seq.fetch_add(1, std::memory_order_relaxed);
	const int pid = static_cast<int>(getpid());
	const long tid = static_cast<long>(syscall(SYS_gettid));

	const bool want_args = env_truthy_local("BPFTIME_CUDA_TRACE_ARGS");
	const bool want_args_words =
		env_truthy_local("BPFTIME_CUDA_TRACE_ARGS_WORDS");
	const uint32_t max_args =
		env_u32_local("BPFTIME_CUDA_TRACE_ARGS_MAX").value_or(8u);
	std::vector<uint64_t> arg_words;
	uint64_t args_fingerprint = 0;
	if (want_args && kernel_params != nullptr && max_args != 0) {
		const char *f0 = std::getenv("BPFTIME_CUDA_TRACE_ARGS_FILTER");
		const char *f1 = std::getenv("BPFTIME_CUDA_SASS_DETOUR_FILTER");
		const char *f2 = std::getenv("BPFTIME_CUDA_SASS_SAMPLE_FILTER");
		const std::string_view fsv =
			(f0 && *f0) ? std::string_view(f0)
			: (f1 && *f1) ? std::string_view(f1)
			: (f2 && *f2) ? std::string_view(f2)
				      : std::string_view {};
		const bool hit = (!fsv.empty() &&
				  kernel_name.find(fsv) != std::string::npos);
		if (!hit) {
			// Keep it safe by default: do not attempt to read args unless the
			// user provides a filter (or reuses an existing SASS filter).
		} else {
		arg_words.reserve(std::min<uint32_t>(max_args, 32u));
		for (uint32_t i = 0; i < max_args; i++) {
			uint64_t ap_u64 = 0;
			if (!safe_read_u64_local(&kernel_params[i], &ap_u64))
				break;
			const void *ap = reinterpret_cast<const void *>(ap_u64);
			if (!ap) {
				arg_words.push_back(0);
				continue;
			}
			uint64_t w = 0;
			if (!safe_read_u64_local(ap, &w)) {
				arg_words.push_back(0);
				continue;
			}
			arg_words.push_back(w);
		}
		args_fingerprint = fnv1a64_local(std::span<const uint8_t>(
			reinterpret_cast<const uint8_t *>(arg_words.data()),
			arg_words.size() * sizeof(uint64_t)));
		}
	}

	{
		std::lock_guard<std::mutex> guard(cuda_launch_trace_mutex);
		cuda_launch_trace_ofs << "{\"type\":\"launch\",\"seq\":" << seq
				      << ",\"ts_ns\":" << ts_ns << ",\"pid\":" << pid
				      << ",\"tid\":" << tid << ",\"name\":\""
				      << json_escape(kernel_name) << "\",\"grid\":["
				      << grid_x << "," << grid_y << "," << grid_z
				      << "],\"block\":[" << block_x << "," << block_y
				      << "," << block_z << "],\"shared_mem\":"
				      << shared_mem << ",\"stream\":\"0x" << std::hex
				      << reinterpret_cast<uintptr_t>(stream) << "\""
				      << std::dec
				      << ",\"uses_extra\":"
				      << (extra ? "true" : "false");
		if (want_args && args_fingerprint != 0) {
			cuda_launch_trace_ofs << ",\"args_fingerprint\":\"0x"
					      << std::hex << args_fingerprint
					      << std::dec << "\"";
			if (want_args_words) {
				cuda_launch_trace_ofs << ",\"args_words\":[";
				for (size_t i = 0; i < arg_words.size(); i++) {
					if (i)
						cuda_launch_trace_ofs << ",";
					cuda_launch_trace_ofs << "\"0x" << std::hex
							      << arg_words[i]
							      << std::dec << "\"";
				}
				cuda_launch_trace_ofs << "]";
			}
		}
		cuda_launch_trace_ofs << "}\n";
		if ((seq & 0x3ff) == 0)
			cuda_launch_trace_ofs.flush();
	}

	return seq;
}

void nv_attach_impl::trace_cuda_memcpy(const char *kind, uint64_t bytes,
				       void *dst, void *src, void *stream,
				       int result, bool async)
{
	if (!cuda_launch_trace_enabled)
		return;
	timespec ts {};
	clock_gettime(CLOCK_MONOTONIC, &ts);
	const uint64_t ts_ns =
		static_cast<uint64_t>(ts.tv_sec) * 1000ULL * 1000ULL * 1000ULL +
		static_cast<uint64_t>(ts.tv_nsec);
	const uint64_t seq =
		cuda_launch_seq.fetch_add(1, std::memory_order_relaxed);
	const int pid = static_cast<int>(getpid());
	const long tid = static_cast<long>(syscall(SYS_gettid));
	std::lock_guard<std::mutex> guard(cuda_launch_trace_mutex);
	cuda_launch_trace_ofs << "{\"type\":\"memcpy\",\"seq\":" << seq
			      << ",\"ts_ns\":" << ts_ns << ",\"pid\":" << pid
			      << ",\"tid\":" << tid << ",\"kind\":\""
			      << (kind ? json_escape(kind) : "unknown")
			      << "\",\"bytes\":" << bytes << ",\"dst\":\"0x"
			      << std::hex << reinterpret_cast<uintptr_t>(dst)
			      << "\",\"src\":\"0x"
			      << reinterpret_cast<uintptr_t>(src)
			      << "\",\"stream\":\"0x"
			      << reinterpret_cast<uintptr_t>(stream)
			      << std::dec << "\",\"async\":"
			      << (async ? "true" : "false") << ",\"result\":"
			      << result << "}\n";
	if ((seq & 0x3ff) == 0)
		cuda_launch_trace_ofs.flush();
}

void nv_attach_impl::trace_cuda_memset(const char *kind, uint64_t bytes,
				       void *dst, uint64_t value, void *stream,
				       int result, bool async)
{
	if (!cuda_launch_trace_enabled)
		return;
	timespec ts {};
	clock_gettime(CLOCK_MONOTONIC, &ts);
	const uint64_t ts_ns =
		static_cast<uint64_t>(ts.tv_sec) * 1000ULL * 1000ULL * 1000ULL +
		static_cast<uint64_t>(ts.tv_nsec);
	const uint64_t seq =
		cuda_launch_seq.fetch_add(1, std::memory_order_relaxed);
	const int pid = static_cast<int>(getpid());
	const long tid = static_cast<long>(syscall(SYS_gettid));
	std::lock_guard<std::mutex> guard(cuda_launch_trace_mutex);
	cuda_launch_trace_ofs << "{\"type\":\"memset\",\"seq\":" << seq
			      << ",\"ts_ns\":" << ts_ns << ",\"pid\":" << pid
			      << ",\"tid\":" << tid << ",\"kind\":\""
			      << (kind ? json_escape(kind) : "unknown")
			      << "\",\"bytes\":" << bytes << ",\"dst\":\"0x"
			      << std::hex << reinterpret_cast<uintptr_t>(dst)
			      << std::dec << "\",\"value\":" << value
			      << ",\"stream\":\"0x" << std::hex
			      << reinterpret_cast<uintptr_t>(stream)
			      << std::dec << "\",\"async\":"
			      << (async ? "true" : "false") << ",\"result\":"
			      << result << "}\n";
	if ((seq & 0x3ff) == 0)
		cuda_launch_trace_ofs.flush();
}

void nv_attach_impl::trace_cuda_alloc(const char *kind, uint64_t bytes,
				      void *ptr, void *stream, int result)
{
	if (!cuda_launch_trace_enabled)
		return;
	timespec ts {};
	clock_gettime(CLOCK_MONOTONIC, &ts);
	const uint64_t ts_ns =
		static_cast<uint64_t>(ts.tv_sec) * 1000ULL * 1000ULL * 1000ULL +
		static_cast<uint64_t>(ts.tv_nsec);
	const uint64_t seq =
		cuda_launch_seq.fetch_add(1, std::memory_order_relaxed);
	const int pid = static_cast<int>(getpid());
	const long tid = static_cast<long>(syscall(SYS_gettid));
	std::lock_guard<std::mutex> guard(cuda_launch_trace_mutex);
	cuda_launch_trace_ofs << "{\"type\":\"alloc\",\"seq\":" << seq
			      << ",\"ts_ns\":" << ts_ns << ",\"pid\":" << pid
			      << ",\"tid\":" << tid << ",\"kind\":\""
			      << (kind ? json_escape(kind) : "unknown")
			      << "\",\"bytes\":" << bytes << ",\"ptr\":\"0x"
			      << std::hex << reinterpret_cast<uintptr_t>(ptr)
			      << std::dec << "\",\"stream\":\"0x" << std::hex
			      << reinterpret_cast<uintptr_t>(stream)
			      << std::dec << "\",\"result\":" << result
			      << "}\n";
	if ((seq & 0x3ff) == 0)
		cuda_launch_trace_ofs.flush();
}

void nv_attach_impl::trace_cuda_free(const char *kind, void *ptr, void *stream,
				     int result)
{
	if (!cuda_launch_trace_enabled)
		return;
	timespec ts {};
	clock_gettime(CLOCK_MONOTONIC, &ts);
	const uint64_t ts_ns =
		static_cast<uint64_t>(ts.tv_sec) * 1000ULL * 1000ULL * 1000ULL +
		static_cast<uint64_t>(ts.tv_nsec);
	const uint64_t seq =
		cuda_launch_seq.fetch_add(1, std::memory_order_relaxed);
	const int pid = static_cast<int>(getpid());
	const long tid = static_cast<long>(syscall(SYS_gettid));
	std::lock_guard<std::mutex> guard(cuda_launch_trace_mutex);
	cuda_launch_trace_ofs << "{\"type\":\"free\",\"seq\":" << seq
			      << ",\"ts_ns\":" << ts_ns << ",\"pid\":" << pid
			      << ",\"tid\":" << tid << ",\"kind\":\""
			      << (kind ? json_escape(kind) : "unknown")
			      << "\",\"ptr\":\"0x" << std::hex
			      << reinterpret_cast<uintptr_t>(ptr) << std::dec
			      << "\",\"stream\":\"0x" << std::hex
			      << reinterpret_cast<uintptr_t>(stream)
			      << std::dec << "\",\"result\":" << result
			      << "}\n";
	if ((seq & 0x3ff) == 0)
		cuda_launch_trace_ofs.flush();
}

void nv_attach_impl::trace_cuda_sync(const char *api, void *obj,
				     uint64_t duration_ns, int result)
{
	if (!cuda_launch_trace_enabled)
		return;
	timespec ts {};
	clock_gettime(CLOCK_MONOTONIC, &ts);
	const uint64_t ts_ns =
		static_cast<uint64_t>(ts.tv_sec) * 1000ULL * 1000ULL * 1000ULL +
		static_cast<uint64_t>(ts.tv_nsec);
	const uint64_t seq =
		cuda_launch_seq.fetch_add(1, std::memory_order_relaxed);
	const int pid = static_cast<int>(getpid());
	const long tid = static_cast<long>(syscall(SYS_gettid));

	// Flush any pending kernel timings at safe points.
	//
	// We only flush on stream/context synchronization, since those imply
	// the recorded events have completed.
	std::vector<pending_kernel_timing> ready;
	const std::string_view api_sv = api ? std::string_view(api) : std::string_view {};
	const bool flush_all = (api_sv == "cuCtxSynchronize");
	const bool flush_stream = (api_sv == "cuStreamSynchronize");
	if (flush_all || (flush_stream && obj != nullptr)) {
		std::lock_guard<std::mutex> g(kernel_timing_mutex);
		for (auto it = pending_kernel_timings.begin();
		     it != pending_kernel_timings.end();) {
			const bool match =
				flush_all || (flush_stream && it->stream == obj);
			if (match) {
				ready.push_back(std::move(*it));
				it = pending_kernel_timings.erase(it);
			} else {
				++it;
			}
		}
	}

	using cu_event_elapsed_time_fn_t =
		CUresult (*)(float *, CUevent, CUevent);
	static cu_event_elapsed_time_fn_t cu_event_elapsed_time =
		(cu_event_elapsed_time_fn_t)dlsym(RTLD_DEFAULT,
						  "cuEventElapsedTime");
	using cu_event_destroy_fn_t = CUresult (*)(CUevent);
	auto cu_event_destroy =
		reinterpret_cast<cu_event_destroy_fn_t>(original_cu_event_destroy_v2);

	struct timing_out {
		uint64_t launch_seq = 0;
		std::string kernel_name;
		void *stream = nullptr;
		float gpu_ms = 0.0f;
		int elapsed_res = 0;
	};
	std::vector<timing_out> timings;
	timings.reserve(ready.size());
	for (auto &p : ready) {
		float ms = 0.0f;
		int elapsed_res = CUDA_ERROR_UNKNOWN;
		if (cu_event_elapsed_time && p.ev_start && p.ev_end) {
			elapsed_res = cu_event_elapsed_time(&ms, p.ev_start, p.ev_end);
		}
		if (cu_event_destroy && p.ev_start)
			(void)cu_event_destroy(p.ev_start);
		if (cu_event_destroy && p.ev_end)
			(void)cu_event_destroy(p.ev_end);
		timings.push_back(timing_out {
			p.launch_seq,
			std::move(p.kernel_name),
			p.stream,
			ms,
			elapsed_res,
		});
	}

	{
		std::lock_guard<std::mutex> guard(cuda_launch_trace_mutex);
		// Emit kernel timing records first, then the sync record.
		for (const auto &t : timings) {
			const uint64_t tseq = cuda_launch_seq.fetch_add(
				1, std::memory_order_relaxed);
			cuda_launch_trace_ofs
				<< "{\"type\":\"kernel_timing\",\"seq\":" << tseq
				<< ",\"ts_ns\":" << ts_ns << ",\"pid\":" << pid
				<< ",\"tid\":" << tid << ",\"launch_seq\":" << t.launch_seq
				<< ",\"name\":\"" << json_escape(t.kernel_name)
				<< "\",\"stream\":\"0x" << std::hex
				<< reinterpret_cast<uintptr_t>(t.stream) << std::dec
				<< "\",\"gpu_ms\":" << t.gpu_ms
				<< ",\"elapsed_result\":" << t.elapsed_res
				<< ",\"flushed_by_api\":\""
				<< (api ? json_escape(api) : "unknown")
				<< "\",\"flushed_by_seq\":" << seq << "}\n";
		}
		cuda_launch_trace_ofs << "{\"type\":\"sync\",\"seq\":" << seq
				      << ",\"ts_ns\":" << ts_ns << ",\"pid\":" << pid
				      << ",\"tid\":" << tid << ",\"api\":\""
				      << (api ? json_escape(api) : "unknown")
				      << "\",\"obj\":\"0x" << std::hex
				      << reinterpret_cast<uintptr_t>(obj) << std::dec
				      << "\",\"duration_ns\":" << duration_ns
				      << ",\"result\":" << result << "}\n";
		if ((seq & 0x3ff) == 0)
			cuda_launch_trace_ofs.flush();
	}
}

void nv_attach_impl::trace_cuda_graph_launch(void *graph_exec, void *stream,
					     int result)
{
	if (!cuda_launch_trace_enabled)
		return;
	timespec ts {};
	clock_gettime(CLOCK_MONOTONIC, &ts);
	const uint64_t ts_ns =
		static_cast<uint64_t>(ts.tv_sec) * 1000ULL * 1000ULL * 1000ULL +
		static_cast<uint64_t>(ts.tv_nsec);
	const uint64_t seq =
		cuda_launch_seq.fetch_add(1, std::memory_order_relaxed);
	const int pid = static_cast<int>(getpid());
	const long tid = static_cast<long>(syscall(SYS_gettid));
	std::lock_guard<std::mutex> guard(cuda_launch_trace_mutex);
	cuda_launch_trace_ofs
		<< "{\"type\":\"graph_launch\",\"seq\":" << seq
		<< ",\"ts_ns\":" << ts_ns << ",\"pid\":" << pid
		<< ",\"tid\":" << tid << ",\"graph_exec\":\"0x" << std::hex
		<< reinterpret_cast<uintptr_t>(graph_exec) << std::dec
		<< "\",\"stream\":\"0x" << std::hex
		<< reinterpret_cast<uintptr_t>(stream) << std::dec
		<< "\",\"result\":" << result << "}\n";
	if ((seq & 0x3ff) == 0)
		cuda_launch_trace_ofs.flush();
}

void nv_attach_impl::trace_cuda_event_record(void *event, void *stream,
					     int result)
{
	if (!cuda_launch_trace_enabled)
		return;
	timespec ts {};
	clock_gettime(CLOCK_MONOTONIC, &ts);
	const uint64_t ts_ns =
		static_cast<uint64_t>(ts.tv_sec) * 1000ULL * 1000ULL * 1000ULL +
		static_cast<uint64_t>(ts.tv_nsec);
	const uint64_t seq =
		cuda_launch_seq.fetch_add(1, std::memory_order_relaxed);
	const int pid = static_cast<int>(getpid());
	const long tid = static_cast<long>(syscall(SYS_gettid));
	std::lock_guard<std::mutex> guard(cuda_launch_trace_mutex);
	cuda_launch_trace_ofs << "{\"type\":\"event_record\",\"seq\":" << seq
			      << ",\"ts_ns\":" << ts_ns << ",\"pid\":" << pid
			      << ",\"tid\":" << tid << ",\"event\":\"0x"
			      << std::hex << reinterpret_cast<uintptr_t>(event)
			      << std::dec << "\",\"stream\":\"0x" << std::hex
			      << reinterpret_cast<uintptr_t>(stream)
			      << std::dec << "\",\"result\":" << result
			      << "}\n";
	if ((seq & 0x3ff) == 0)
		cuda_launch_trace_ofs.flush();
}

void nv_attach_impl::trace_cuda_stream_create(const char *api, void *stream,
					      unsigned int flags, int priority,
					      int result)
{
	if (!cuda_launch_trace_enabled)
		return;
	timespec ts {};
	clock_gettime(CLOCK_MONOTONIC, &ts);
	const uint64_t ts_ns =
		static_cast<uint64_t>(ts.tv_sec) * 1000ULL * 1000ULL * 1000ULL +
		static_cast<uint64_t>(ts.tv_nsec);
	const uint64_t seq =
		cuda_launch_seq.fetch_add(1, std::memory_order_relaxed);
	const int pid = static_cast<int>(getpid());
	const long tid = static_cast<long>(syscall(SYS_gettid));
	std::lock_guard<std::mutex> guard(cuda_launch_trace_mutex);
	cuda_launch_trace_ofs
		<< "{\"type\":\"stream_create\",\"seq\":" << seq
		<< ",\"ts_ns\":" << ts_ns << ",\"pid\":" << pid
		<< ",\"tid\":" << tid << ",\"api\":\""
		<< (api ? json_escape(api) : "unknown") << "\",\"stream\":\"0x"
		<< std::hex << reinterpret_cast<uintptr_t>(stream) << std::dec
		<< "\",\"flags\":" << flags << ",\"priority\":" << priority
		<< ",\"result\":" << result << "}\n";
	if ((seq & 0x3ff) == 0)
		cuda_launch_trace_ofs.flush();
}

void nv_attach_impl::trace_cuda_stream_destroy(void *stream, int result)
{
	if (!cuda_launch_trace_enabled)
		return;
	timespec ts {};
	clock_gettime(CLOCK_MONOTONIC, &ts);
	const uint64_t ts_ns =
		static_cast<uint64_t>(ts.tv_sec) * 1000ULL * 1000ULL * 1000ULL +
		static_cast<uint64_t>(ts.tv_nsec);
	const uint64_t seq =
		cuda_launch_seq.fetch_add(1, std::memory_order_relaxed);
	const int pid = static_cast<int>(getpid());
	const long tid = static_cast<long>(syscall(SYS_gettid));
	std::lock_guard<std::mutex> guard(cuda_launch_trace_mutex);
	cuda_launch_trace_ofs
		<< "{\"type\":\"stream_destroy\",\"seq\":" << seq
		<< ",\"ts_ns\":" << ts_ns << ",\"pid\":" << pid
		<< ",\"tid\":" << tid << ",\"stream\":\"0x" << std::hex
		<< reinterpret_cast<uintptr_t>(stream) << std::dec
		<< "\",\"result\":" << result << "}\n";
	if ((seq & 0x3ff) == 0)
		cuda_launch_trace_ofs.flush();
}

void nv_attach_impl::trace_cuda_stream_wait_event(void *stream, void *event,
						  unsigned int flags,
						  int result)
{
	if (!cuda_launch_trace_enabled)
		return;
	timespec ts {};
	clock_gettime(CLOCK_MONOTONIC, &ts);
	const uint64_t ts_ns =
		static_cast<uint64_t>(ts.tv_sec) * 1000ULL * 1000ULL * 1000ULL +
		static_cast<uint64_t>(ts.tv_nsec);
	const uint64_t seq =
		cuda_launch_seq.fetch_add(1, std::memory_order_relaxed);
	const int pid = static_cast<int>(getpid());
	const long tid = static_cast<long>(syscall(SYS_gettid));
	std::lock_guard<std::mutex> guard(cuda_launch_trace_mutex);
	cuda_launch_trace_ofs
		<< "{\"type\":\"stream_wait_event\",\"seq\":" << seq
		<< ",\"ts_ns\":" << ts_ns << ",\"pid\":" << pid
		<< ",\"tid\":" << tid << ",\"stream\":\"0x" << std::hex
		<< reinterpret_cast<uintptr_t>(stream) << std::dec
		<< "\",\"event\":\"0x" << std::hex
		<< reinterpret_cast<uintptr_t>(event) << std::dec
		<< "\",\"flags\":" << flags << ",\"result\":" << result
		<< "}\n";
	if ((seq & 0x3ff) == 0)
		cuda_launch_trace_ofs.flush();
}

void nv_attach_impl::trace_cuda_event_create(void *event, unsigned int flags,
					     int result)
{
	if (!cuda_launch_trace_enabled)
		return;
	timespec ts {};
	clock_gettime(CLOCK_MONOTONIC, &ts);
	const uint64_t ts_ns =
		static_cast<uint64_t>(ts.tv_sec) * 1000ULL * 1000ULL * 1000ULL +
		static_cast<uint64_t>(ts.tv_nsec);
	const uint64_t seq =
		cuda_launch_seq.fetch_add(1, std::memory_order_relaxed);
	const int pid = static_cast<int>(getpid());
	const long tid = static_cast<long>(syscall(SYS_gettid));
	std::lock_guard<std::mutex> guard(cuda_launch_trace_mutex);
	cuda_launch_trace_ofs
		<< "{\"type\":\"event_create\",\"seq\":" << seq
		<< ",\"ts_ns\":" << ts_ns << ",\"pid\":" << pid
		<< ",\"tid\":" << tid << ",\"event\":\"0x" << std::hex
		<< reinterpret_cast<uintptr_t>(event) << std::dec
		<< "\",\"flags\":" << flags << ",\"result\":" << result
		<< "}\n";
	if ((seq & 0x3ff) == 0)
		cuda_launch_trace_ofs.flush();
}

void nv_attach_impl::trace_cuda_event_destroy(void *event, int result)
{
	if (!cuda_launch_trace_enabled)
		return;
	timespec ts {};
	clock_gettime(CLOCK_MONOTONIC, &ts);
	const uint64_t ts_ns =
		static_cast<uint64_t>(ts.tv_sec) * 1000ULL * 1000ULL * 1000ULL +
		static_cast<uint64_t>(ts.tv_nsec);
	const uint64_t seq =
		cuda_launch_seq.fetch_add(1, std::memory_order_relaxed);
	const int pid = static_cast<int>(getpid());
	const long tid = static_cast<long>(syscall(SYS_gettid));
	std::lock_guard<std::mutex> guard(cuda_launch_trace_mutex);
	cuda_launch_trace_ofs
		<< "{\"type\":\"event_destroy\",\"seq\":" << seq
		<< ",\"ts_ns\":" << ts_ns << ",\"pid\":" << pid
		<< ",\"tid\":" << tid << ",\"event\":\"0x" << std::hex
		<< reinterpret_cast<uintptr_t>(event) << std::dec
		<< "\",\"result\":" << result << "}\n";
	if ((seq & 0x3ff) == 0)
		cuda_launch_trace_ofs.flush();
}

void nv_attach_impl::trace_cuda_module_load(const char *api, void *module,
					    void *image,
					    unsigned int num_options,
					    int result)
{
	if (!cuda_launch_trace_enabled)
		return;
	timespec ts {};
	clock_gettime(CLOCK_MONOTONIC, &ts);
	const uint64_t ts_ns =
		static_cast<uint64_t>(ts.tv_sec) * 1000ULL * 1000ULL * 1000ULL +
		static_cast<uint64_t>(ts.tv_nsec);
	const uint64_t seq =
		cuda_launch_seq.fetch_add(1, std::memory_order_relaxed);
	const int pid = static_cast<int>(getpid());
	const long tid = static_cast<long>(syscall(SYS_gettid));
	std::lock_guard<std::mutex> guard(cuda_launch_trace_mutex);
	cuda_launch_trace_ofs
		<< "{\"type\":\"module_load\",\"seq\":" << seq
		<< ",\"ts_ns\":" << ts_ns << ",\"pid\":" << pid
		<< ",\"tid\":" << tid << ",\"api\":\""
		<< (api ? json_escape(api) : "unknown") << "\",\"module\":\"0x"
		<< std::hex << reinterpret_cast<uintptr_t>(module) << std::dec
		<< "\",\"image\":\"0x" << std::hex
		<< reinterpret_cast<uintptr_t>(image) << std::dec
		<< "\",\"num_options\":" << num_options
		<< ",\"result\":" << result << "}\n";
	if ((seq & 0x3ff) == 0)
		cuda_launch_trace_ofs.flush();
}

void nv_attach_impl::trace_cuda_module_unload(void *module, int result)
{
	if (!cuda_launch_trace_enabled)
		return;
	timespec ts {};
	clock_gettime(CLOCK_MONOTONIC, &ts);
	const uint64_t ts_ns =
		static_cast<uint64_t>(ts.tv_sec) * 1000ULL * 1000ULL * 1000ULL +
		static_cast<uint64_t>(ts.tv_nsec);
	const uint64_t seq =
		cuda_launch_seq.fetch_add(1, std::memory_order_relaxed);
	const int pid = static_cast<int>(getpid());
	const long tid = static_cast<long>(syscall(SYS_gettid));
	std::lock_guard<std::mutex> guard(cuda_launch_trace_mutex);
	cuda_launch_trace_ofs
		<< "{\"type\":\"module_unload\",\"seq\":" << seq
		<< ",\"ts_ns\":" << ts_ns << ",\"pid\":" << pid
		<< ",\"tid\":" << tid << ",\"module\":\"0x" << std::hex
		<< reinterpret_cast<uintptr_t>(module) << std::dec
		<< "\",\"result\":" << result << "}\n";
	if ((seq & 0x3ff) == 0)
		cuda_launch_trace_ofs.flush();
}

void nv_attach_impl::trace_cuda_library_load(const char *api, void *library,
					     void *code,
					     unsigned int num_jit_options,
					     unsigned int num_library_options,
					     int result)
{
	if (!cuda_launch_trace_enabled)
		return;
	timespec ts {};
	clock_gettime(CLOCK_MONOTONIC, &ts);
	const uint64_t ts_ns =
		static_cast<uint64_t>(ts.tv_sec) * 1000ULL * 1000ULL * 1000ULL +
		static_cast<uint64_t>(ts.tv_nsec);
	const uint64_t seq =
		cuda_launch_seq.fetch_add(1, std::memory_order_relaxed);
	const int pid = static_cast<int>(getpid());
	const long tid = static_cast<long>(syscall(SYS_gettid));
	std::lock_guard<std::mutex> guard(cuda_launch_trace_mutex);
	cuda_launch_trace_ofs
		<< "{\"type\":\"library_load\",\"seq\":" << seq
		<< ",\"ts_ns\":" << ts_ns << ",\"pid\":" << pid
		<< ",\"tid\":" << tid << ",\"api\":\""
		<< (api ? json_escape(api) : "unknown") << "\",\"library\":\"0x"
		<< std::hex << reinterpret_cast<uintptr_t>(library) << std::dec
		<< "\",\"code\":\"0x" << std::hex
		<< reinterpret_cast<uintptr_t>(code) << std::dec
		<< "\",\"num_jit_options\":" << num_jit_options
		<< ",\"num_library_options\":" << num_library_options
		<< ",\"result\":" << result << "}\n";
	if ((seq & 0x3ff) == 0)
		cuda_launch_trace_ofs.flush();
}

void nv_attach_impl::trace_cuda_library_unload(void *library, int result)
{
	if (!cuda_launch_trace_enabled)
		return;
	timespec ts {};
	clock_gettime(CLOCK_MONOTONIC, &ts);
	const uint64_t ts_ns =
		static_cast<uint64_t>(ts.tv_sec) * 1000ULL * 1000ULL * 1000ULL +
		static_cast<uint64_t>(ts.tv_nsec);
	const uint64_t seq =
		cuda_launch_seq.fetch_add(1, std::memory_order_relaxed);
	const int pid = static_cast<int>(getpid());
	const long tid = static_cast<long>(syscall(SYS_gettid));
	std::lock_guard<std::mutex> guard(cuda_launch_trace_mutex);
	cuda_launch_trace_ofs
		<< "{\"type\":\"library_unload\",\"seq\":" << seq
		<< ",\"ts_ns\":" << ts_ns << ",\"pid\":" << pid
		<< ",\"tid\":" << tid << ",\"library\":\"0x" << std::hex
		<< reinterpret_cast<uintptr_t>(library) << std::dec
		<< "\",\"result\":" << result << "}\n";
	if ((seq & 0x3ff) == 0)
		cuda_launch_trace_ofs.flush();
}

void nv_attach_impl::trace_cuda_library_get_kernel(void *library, void *kernel,
						   const char *name,
						   int result)
{
	if (!cuda_launch_trace_enabled)
		return;
	timespec ts {};
	clock_gettime(CLOCK_MONOTONIC, &ts);
	const uint64_t ts_ns =
		static_cast<uint64_t>(ts.tv_sec) * 1000ULL * 1000ULL * 1000ULL +
		static_cast<uint64_t>(ts.tv_nsec);
	const uint64_t seq =
		cuda_launch_seq.fetch_add(1, std::memory_order_relaxed);
	const int pid = static_cast<int>(getpid());
	const long tid = static_cast<long>(syscall(SYS_gettid));
	std::lock_guard<std::mutex> guard(cuda_launch_trace_mutex);
	cuda_launch_trace_ofs
		<< "{\"type\":\"library_get_kernel\",\"seq\":" << seq
		<< ",\"ts_ns\":" << ts_ns << ",\"pid\":" << pid
		<< ",\"tid\":" << tid << ",\"library\":\"0x" << std::hex
		<< reinterpret_cast<uintptr_t>(library) << std::dec
		<< "\",\"kernel\":\"0x" << std::hex
		<< reinterpret_cast<uintptr_t>(kernel) << std::dec
		<< "\",\"name\":\"" << (name ? json_escape(name) : "")
		<< "\",\"result\":" << result << "}\n";
	if ((seq & 0x3ff) == 0)
		cuda_launch_trace_ofs.flush();
}

void nv_attach_impl::trace_cuda_kernel_get_name(void *kernel, const char *name,
						int result)
{
	if (!cuda_launch_trace_enabled)
		return;
	timespec ts {};
	clock_gettime(CLOCK_MONOTONIC, &ts);
	const uint64_t ts_ns =
		static_cast<uint64_t>(ts.tv_sec) * 1000ULL * 1000ULL * 1000ULL +
		static_cast<uint64_t>(ts.tv_nsec);
	const uint64_t seq =
		cuda_launch_seq.fetch_add(1, std::memory_order_relaxed);
	const int pid = static_cast<int>(getpid());
	const long tid = static_cast<long>(syscall(SYS_gettid));
	std::lock_guard<std::mutex> guard(cuda_launch_trace_mutex);
	cuda_launch_trace_ofs
		<< "{\"type\":\"kernel_get_name\",\"seq\":" << seq
		<< ",\"ts_ns\":" << ts_ns << ",\"pid\":" << pid
		<< ",\"tid\":" << tid << ",\"kernel\":\"0x" << std::hex
		<< reinterpret_cast<uintptr_t>(kernel) << std::dec
		<< "\",\"name\":\"" << (name ? json_escape(name) : "")
		<< "\",\"result\":" << result << "}\n";
	if ((seq & 0x3ff) == 0)
		cuda_launch_trace_ofs.flush();
}

void nv_attach_impl::trace_cuda_kernel_get_function(void *kernel, void *function,
					    int result)
{
	if (!cuda_launch_trace_enabled)
		return;
	timespec ts {};
	clock_gettime(CLOCK_MONOTONIC, &ts);
	const uint64_t ts_ns =
		static_cast<uint64_t>(ts.tv_sec) * 1000ULL * 1000ULL * 1000ULL +
		static_cast<uint64_t>(ts.tv_nsec);
	const uint64_t seq =
		cuda_launch_seq.fetch_add(1, std::memory_order_relaxed);
	const int pid = static_cast<int>(getpid());
	const long tid = static_cast<long>(syscall(SYS_gettid));
	std::lock_guard<std::mutex> guard(cuda_launch_trace_mutex);
	cuda_launch_trace_ofs
		<< "{\"type\":\"kernel_get_function\",\"seq\":" << seq
		<< ",\"ts_ns\":" << ts_ns << ",\"pid\":" << pid
		<< ",\"tid\":" << tid << ",\"kernel\":\"0x" << std::hex
		<< reinterpret_cast<uintptr_t>(kernel) << std::dec
		<< "\",\"function\":\"0x" << std::hex
		<< reinterpret_cast<uintptr_t>(function) << std::dec
		<< "\",\"result\":" << result << "}\n";
	if ((seq & 0x3ff) == 0)
		cuda_launch_trace_ofs.flush();
}

bool nv_attach_impl::enqueue_cuda_kernel_timing(uint64_t launch_seq,
						const std::string &kernel_name,
						void *stream, CUevent start,
						CUevent end)
{
	if (!cuda_launch_trace_enabled)
		return false;
	if (start == nullptr || end == nullptr)
		return false;
	// Avoid unbounded growth if the workload never synchronizes.
	const uint32_t max_pending =
		env_u32_local("BPFTIME_CUDA_KERNEL_TIMING_MAX_PENDING").value_or(1024u);
	std::lock_guard<std::mutex> g(kernel_timing_mutex);
	if (max_pending != 0 && pending_kernel_timings.size() >= max_pending) {
		// Drop the newly created events. They will be destroyed by the caller.
		return false;
	}
	pending_kernel_timings.push_back(pending_kernel_timing {
		launch_seq,
		kernel_name,
		stream,
		start,
		end,
	});
	return true;
}

void nv_attach_impl::culink_track_owned_input(CUlinkState state,
					      std::vector<uint8_t> &&bytes,
					      const void **out_data,
					      size_t *out_size)
{
	if (out_data == nullptr || out_size == nullptr)
		return;
	*out_data = nullptr;
	*out_size = 0;
	if (state == nullptr || bytes.empty())
		return;
	std::lock_guard<std::mutex> g(culink_owned_inputs_lock);
	auto &v = culink_owned_inputs[state];
	v.emplace_back(std::move(bytes));
	const auto &back = v.back();
	*out_data = back.data();
	*out_size = back.size();
}

void nv_attach_impl::culink_release_state(CUlinkState state)
{
	if (state == nullptr)
		return;
	std::lock_guard<std::mutex> g(culink_owned_inputs_lock);
	culink_owned_inputs.erase(state);
}

bool nv_attach_impl::cuda_launch_trace_is_enabled() const
{
	return cuda_launch_trace_enabled;
}

bool nv_attach_impl::cuda_launch_trace_stream_good() const
{
	return cuda_launch_trace_ofs.good();
}

std::string nv_attach_impl::cuda_launch_trace_path_copy() const
{
	return cuda_launch_trace_path;
}

void nv_attach_impl::record_patched_kernel_function(
	const std::string &kernel_name, CUfunction function)
{
	record_patched_kernel_function_ex(kernel_name, function,
					  PatchedKernelKind::Unknown, 0);
}

void nv_attach_impl::record_patched_kernel_function_ex(
	const std::string &kernel_name, CUfunction function,
	PatchedKernelKind kind, uint32_t target_func_id)
{
	if (kernel_name.empty() || function == nullptr)
		return;
	std::lock_guard<std::mutex> guard(cuda_symbol_map_mutex);
	auto itr = patched_kernel_by_name.find(kernel_name);
	if (itr == patched_kernel_by_name.end()) {
		patched_kernel_by_name.emplace(kernel_name,
					       PatchedKernelEntry {
						       .function = function,
						       .kind = kind,
						       .target_func_id =
							       target_func_id,
					       });
		return;
	}
	if (itr->second.function != function)
		itr->second.function = function;
	itr->second.kind = kind;
	itr->second.target_func_id = target_func_id;
}

std::optional<CUfunction> nv_attach_impl::find_patched_kernel_function(
	const std::string &kernel_name) const
{
	if (kernel_name.empty())
		return std::nullopt;
	std::lock_guard<std::mutex> guard(cuda_symbol_map_mutex);
	auto itr = patched_kernel_by_name.find(kernel_name);
	if (itr == patched_kernel_by_name.end())
		return std::nullopt;
	return itr->second.function;
}

std::optional<nv_attach_impl::PatchedKernelEntry>
nv_attach_impl::find_patched_kernel_entry(const std::string &kernel_name) const
{
	if (kernel_name.empty())
		return std::nullopt;
	std::lock_guard<std::mutex> guard(cuda_symbol_map_mutex);
	auto itr = patched_kernel_by_name.find(kernel_name);
	if (itr == patched_kernel_by_name.end())
		return std::nullopt;
	return itr->second;
}

void nv_attach_impl::record_original_cufunction_name(
	CUfunction function, const std::string &kernel_name)
{
	if (function == nullptr || kernel_name.empty())
		return;
	std::lock_guard<std::mutex> guard(cuda_symbol_map_mutex);
	auto itr = kernel_name_by_cufunction.find(function);
	if (itr == kernel_name_by_cufunction.end()) {
		kernel_name_by_cufunction.emplace(function, kernel_name);
		return;
	}
	if (itr->second != kernel_name)
		itr->second = kernel_name;
}

void nv_attach_impl::record_original_cufunction_module(CUfunction function,
						       CUmodule module)
{
	if (function == nullptr || module == nullptr)
		return;
	std::lock_guard<std::mutex> guard(cuda_symbol_map_mutex);
	auto itr = module_by_cufunction.find(function);
	if (itr == module_by_cufunction.end()) {
		module_by_cufunction.emplace(function, module);
		return;
	}
	if (itr->second != module)
		itr->second = module;
}

void nv_attach_impl::record_original_cufunction_cukernel(CUfunction function,
							 CUkernel kernel)
{
	if (function == nullptr || kernel == nullptr)
		return;
	std::lock_guard<std::mutex> guard(cuda_symbol_map_mutex);
	auto itr = cukernel_by_cufunction.find(function);
	if (itr == cukernel_by_cufunction.end()) {
		cukernel_by_cufunction.emplace(function, kernel);
		return;
	}
	if (itr->second != kernel)
		itr->second = kernel;
}

std::optional<std::string>
nv_attach_impl::find_original_kernel_name(CUfunction function) const
{
	if (function == nullptr)
		return std::nullopt;
	std::lock_guard<std::mutex> guard(cuda_symbol_map_mutex);
	auto itr = kernel_name_by_cufunction.find(function);
	if (itr == kernel_name_by_cufunction.end())
		return std::nullopt;
	return itr->second;
}

void nv_attach_impl::record_cuda_module_image(CUmodule module, const void *image,
					      size_t size, uint64_t hash,
					      bool patched, std::string_view api,
					      const void *base_image,
					      size_t base_size, uint64_t base_hash,
					      uint32_t sm120_image_id,
					      uint32_t base_sm120_image_id)
{
	if (module == nullptr || image == nullptr)
		return;
	cuda_module_image_info info;
	info.image_ptr = reinterpret_cast<uint64_t>(image);
	info.image_size = size;
	info.image_hash = hash;
	info.patched = patched;
	info.sm120_image_id = sm120_image_id;
	if (base_image != nullptr && base_size != 0) {
		info.base_image_ptr = reinterpret_cast<uint64_t>(base_image);
		info.base_image_size = base_size;
		info.base_image_hash = base_hash;
		info.base_sm120_image_id = base_sm120_image_id;
	}
	info.api = std::string(api);
	std::lock_guard<std::mutex> guard(cuda_symbol_map_mutex);
	module_image_by_module[module] = std::move(info);
}

void nv_attach_impl::record_cuda_library_image(CUlibrary library, const void *code,
					       size_t size, uint64_t hash,
					       bool patched, std::string_view api,
					       const void *base_image,
					       size_t base_size, uint64_t base_hash,
					       uint32_t sm120_image_id,
					       uint32_t base_sm120_image_id)
{
	if (library == nullptr || code == nullptr)
		return;
	cuda_module_image_info info;
	info.image_ptr = reinterpret_cast<uint64_t>(code);
	info.image_size = size;
	info.image_hash = hash;
	info.patched = patched;
	info.sm120_image_id = sm120_image_id;
	if (base_image != nullptr && base_size != 0) {
		info.base_image_ptr = reinterpret_cast<uint64_t>(base_image);
		info.base_image_size = base_size;
		info.base_image_hash = base_hash;
		info.base_sm120_image_id = base_sm120_image_id;
	}
	info.api = std::string(api);
	std::lock_guard<std::mutex> guard(cuda_symbol_map_mutex);
	library_image_by_library[library] = std::move(info);
}

void nv_attach_impl::record_cuda_module_image_from_library(CUmodule module,
							   CUlibrary library,
							   std::string_view api)
{
	if (module == nullptr || library == nullptr)
		return;
	std::lock_guard<std::mutex> guard(cuda_symbol_map_mutex);
	auto it = library_image_by_library.find(library);
	if (it == library_image_by_library.end())
		return;
	cuda_module_image_info info = it->second;
	info.api = std::string(api);
	module_image_by_module[module] = std::move(info);
}

std::optional<nv_attach_impl::cuda_module_image_info>
nv_attach_impl::find_cuda_module_image_by_function(CUfunction function) const
{
	if (function == nullptr)
		return std::nullopt;
	std::lock_guard<std::mutex> guard(cuda_symbol_map_mutex);
	if (auto it_mod = module_by_cufunction.find(function);
	    it_mod != module_by_cufunction.end()) {
		if (auto it_img = module_image_by_module.find(it_mod->second);
		    it_img != module_image_by_module.end()) {
			return it_img->second;
		}
	}
	// cuLibraryLoadData -> cuLibraryGetKernel -> cuKernelGetFunction path
	if (auto it_k = cukernel_by_cufunction.find(function);
	    it_k != cukernel_by_cufunction.end()) {
		if (auto it_lib = library_by_cukernel.find(it_k->second);
		    it_lib != library_by_cukernel.end()) {
			if (auto it_img = library_image_by_library.find(it_lib->second);
			    it_img != library_image_by_library.end()) {
				return it_img->second;
			}
		}
	}
	return std::nullopt;
}

std::optional<CUmodule>
nv_attach_impl::resolve_cumodule_for_cufunction(CUfunction function)
{
	if (function == nullptr)
		return std::nullopt;

	// Fast path: already known CUmodule.
	{
		std::lock_guard<std::mutex> guard(cuda_symbol_map_mutex);
		if (auto it_mod = module_by_cufunction.find(function);
		    it_mod != module_by_cufunction.end()) {
			return it_mod->second;
		}
	}

	// cuLibraryLoadData -> cuLibraryGetKernel -> cuKernelGetFunction path:
	// resolve CUmodule via owning CUlibrary.
	CUlibrary lib = nullptr;
	{
		std::lock_guard<std::mutex> guard(cuda_symbol_map_mutex);
		if (auto it_k = cukernel_by_cufunction.find(function);
		    it_k != cukernel_by_cufunction.end()) {
			if (auto it_lib = library_by_cukernel.find(it_k->second);
			    it_lib != library_by_cukernel.end()) {
				lib = it_lib->second;
			}
		}
	}
	if (lib == nullptr)
		return std::nullopt;

	if (original_cu_library_get_module == nullptr)
		return std::nullopt;
	auto cuLibraryGetModule = reinterpret_cast<CUresult (*)(CUmodule *, CUlibrary)>(
		original_cu_library_get_module);

	CUmodule mod = nullptr;
	if (cuLibraryGetModule(&mod, lib) != CUDA_SUCCESS || mod == nullptr)
		return std::nullopt;

	record_original_cufunction_module(function, mod);
	record_cuda_module_image_from_library(mod, lib,
					      "resolve_cumodule_for_cufunction");
	return mod;
}

void nv_attach_impl::record_original_cukernel_name(CUkernel kernel,
						   const std::string &kernel_name)
{
	if (kernel == nullptr || kernel_name.empty())
		return;
	std::lock_guard<std::mutex> guard(cuda_symbol_map_mutex);
	auto itr = kernel_name_by_cukernel.find(kernel);
	if (itr == kernel_name_by_cukernel.end()) {
		kernel_name_by_cukernel.emplace(kernel, kernel_name);
		return;
	}
	if (itr->second != kernel_name)
		itr->second = kernel_name;
}

void nv_attach_impl::record_original_cukernel_library(CUkernel kernel,
						      CUlibrary library)
{
	if (kernel == nullptr || library == nullptr)
		return;
	std::lock_guard<std::mutex> guard(cuda_symbol_map_mutex);
	auto it = library_by_cukernel.find(kernel);
	if (it == library_by_cukernel.end()) {
		library_by_cukernel.emplace(kernel, library);
		return;
	}
	if (it->second != library)
		it->second = library;
}

std::optional<std::string>
nv_attach_impl::find_original_cukernel_name(CUkernel kernel) const
{
	if (kernel == nullptr)
		return std::nullopt;
	std::lock_guard<std::mutex> guard(cuda_symbol_map_mutex);
	auto itr = kernel_name_by_cukernel.find(kernel);
	if (itr == kernel_name_by_cukernel.end())
		return std::nullopt;
	return itr->second;
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

	auto find_cuobjdump = []() -> std::string {
		if (const char *p = std::getenv("CUOBJDUMP");
		    p != nullptr && p[0] != '\0') {
			return p;
		}
		if (const char *cuda_root = std::getenv("BPFTIME_CUDA_ROOT");
		    cuda_root != nullptr && cuda_root[0] != '\0') {
			auto candidate = std::filesystem::path(cuda_root) /
					 "bin" / "cuobjdump";
			if (std::filesystem::exists(candidate))
				return candidate.string();
		}
		if (const char *cuda_home = std::getenv("CUDA_HOME");
		    cuda_home != nullptr && cuda_home[0] != '\0') {
			auto candidate = std::filesystem::path(cuda_home) /
					 "bin" / "cuobjdump";
			if (std::filesystem::exists(candidate))
				return candidate.string();
		}
		if (const char *cuda_path = std::getenv("CUDA_PATH");
		    cuda_path != nullptr && cuda_path[0] != '\0') {
			auto candidate = std::filesystem::path(cuda_path) /
					 "bin" / "cuobjdump";
			if (std::filesystem::exists(candidate))
				return candidate.string();
		}
		{
			auto candidate =
				std::filesystem::path("/usr/local/cuda/bin") /
				"cuobjdump";
			if (std::filesystem::exists(candidate))
				return candidate.string();
		}
		return "cuobjdump";
	};

	const auto cuobjdump_bin = find_cuobjdump();

	// Build command line - use shell to properly search PATH
	auto cuobjdump_cmd_line =
		cuobjdump_bin + " --extract-ptx all " + fatbin_path.string();
	SPDLOG_INFO("Calling cuobjdump: {}", cuobjdump_cmd_line);

	// Execute through shell to properly use PATH
	boost::process::child child(
		"/bin/sh", boost::process::args({ "-c", cuobjdump_cmd_line }),
		boost::process::std_out > stream, boost::process::env(env),
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
	if (!spdlog::should_log(spdlog::level::debug)) {
		SPDLOG_INFO("Remove extracted files..");
		std::filesystem::remove_all(working_dir);
	}
	SPDLOG_INFO("Got {} PTX files", all_ptx.size());
	return all_ptx;
}
std::optional<std::map<std::string, std::tuple<std::string, bool>>>
nv_attach_impl::hack_fatbin(std::map<std::string, std::string> all_ptx)
{
	/**
	Here we can patch the PTX.
	*/
	boost::asio::thread_pool pool(std::thread::hardware_concurrency());
	std::map<std::string, std::tuple<std::string, bool>> ptx_out;
	std::mutex map_mutex;
	std::mutex cache_mutex;
	for (auto &[file_name, original_ptx] : all_ptx) {
		boost::asio::post(pool, [this, original_ptx, file_name, &map_mutex, &ptx_out, &cache_mutex]() -> void {
			auto current_ptx = original_ptx;
			SPDLOG_INFO("Patching PTX: {}", file_name);
			bool should_add_trampoline = false;
			for (const auto &[_, hook_entry] : this->hook_entries) {
				const auto &kernels = hook_entry.kernels;
				for (const auto &kernel : kernels) {
					std::vector<uint64_t> ebpf_inst_words;
					ebpf_inst_words.assign(
						(uint64_t *)(uintptr_t)hook_entry
							.instuctions.data(),
						(uint64_t *)(uintptr_t)hook_entry
								.instuctions
								.data() +
							hook_entry.instuctions
								.size()

					);
					ptxpass::runtime_request::RuntimeRequest
						req;
					auto &ri = req.input;
					ri.full_ptx = current_ptx;
					ri.to_patch_kernel = kernel;
					ri.global_ebpf_map_info_symbol =
						"map_info";
					ri.ebpf_communication_data_symbol =
						"constData";

					req.set_ebpf_instructions(
						ebpf_inst_words);
					nlohmann::json in;
					ptxpass::runtime_request::to_json(in,
									  req);
					auto input_json = in.dump();
					SPDLOG_DEBUG("Input: {}", input_json);
					auto sha256_string =
						sha256(input_json.data(),
						       input_json.size());

					ptxpass::runtime_response::RuntimeResponse
						resp;

					cache_mutex.lock();
					if (auto itr = this->patch_cache.find(
						    sha256_string);
					    itr != this->patch_cache.end()) {
						SPDLOG_INFO(
							"Patching request {} found in cache",
							sha256_string);
						resp = itr->second;
						cache_mutex.unlock();
					} else {
						cache_mutex.unlock();
						SPDLOG_INFO(
							"Patching request {} not found in cache, patching..",
							sha256_string);
						std::vector<char> buf(1 << 30);
						int err =
							hook_entry.config->process_input(
								input_json
									.c_str(),
								buf.size(),
								buf.data());

						if (err ==
						    ptxpass::ExitCode::Success) {
							auto json = nlohmann::json::
								parse(buf.data());
							using namespace ptxpass::
								runtime_response;

							from_json(json, resp);

						} else {
							SPDLOG_ERROR(
								"Unable to run pass on kernel {}: {}",
								kernel,
								(int)err);
							return;
						}
						std::lock_guard<std::mutex>
							_cache_guard(
								cache_mutex);
						patch_cache[sha256_string] =
							resp;
					}
					current_ptx = resp.output_ptx;
					should_add_trampoline =
						should_add_trampoline ||
						resp.modified;
				}
			}
			if (should_add_trampoline) {
				current_ptx =
					ptxpass::filter_out_version_headers_ptx(
						wrap_ptx_with_trampoline(
							current_ptx));
			}
			std::lock_guard<std::mutex> _guard(map_mutex);
			ptx_out["patched." + file_name] = std::make_tuple(
				current_ptx, should_add_trampoline);
		});
	}
	pool.join();
	if (spdlog::should_log(spdlog::level::debug)) {
		char tmp_dir[] = "/tmp/bpftime-fatbin-work.XXXXXX";
		mkdtemp(tmp_dir);
		auto working_dir = std::filesystem::path(tmp_dir);

		SPDLOG_DEBUG("Writing patched PTX to {}", working_dir.c_str());
		for (const auto &[file_name, ptx] : ptx_out) {
			auto path = working_dir / (file_name);
			std::ofstream ofs(path);
			ofs << std::get<0>(ptx);
		}
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

int nv_attach_impl::run_attach_entry_on_gpu(int attach_id, int run_count,
					    int grid_dim_x, int grid_dim_y,
					    int grid_dim_z, int block_dim_x,
					    int block_dim_y, int block_dim_z)
{
	if (this->shared_mem_ptr == 0) {
		SPDLOG_ERROR(
			"shared_mem_ptr is not initialized; cannot run attach {} on GPU",
			attach_id);
		return -1;
	}
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

	// Get SM architecture (auto-detect or from BPFTIME_SM_ARCH env)
	std::string sm_arch = get_gpu_sm_arch();
	SPDLOG_INFO("Using SM architecture: {}", sm_arch);

	std::vector<uint64_t> ebpf_words;
	for (const auto &insts : insts) {
		ebpf_words.push_back(*(uint64_t *)(uintptr_t)&insts);
	}
		auto ptx = ptxpass::filter_out_version_headers_ptx(
			wrap_ptx_with_trampoline_for_sm(
				filter_compiled_ptx_for_ebpf_program(
					ptxpass::compile_ebpf_to_ptx_from_words(
						ebpf_words, sm_arch.c_str(),
						"bpf_main", false, false),
					"bpf_main"),
				sm_arch));
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
	{
		unsigned int major_ver, minor_ver;
		NVPTXCOMPILER_SAFE_CALL(
			nvPTXCompilerGetVersion(&major_ver, &minor_ver));
		SPDLOG_INFO("PTX compiler version {}.{}", major_ver, minor_ver);
		nvPTXCompilerHandle compiler = nullptr;
		NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerCreate(
			&compiler, (size_t)ptx.size(), ptx.c_str()));
		std::string gpu_name_opt = "--gpu-name=" + sm_arch;
		const char *compile_options[] = { gpu_name_opt.c_str(),
						  "--verbose" };
		auto status = nvPTXCompilerCompile(
			compiler, std::size(compile_options), compile_options);
		if (status != NVPTXCOMPILE_SUCCESS) {
			size_t error_size;

			NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetErrorLogSize(
				compiler, &error_size));

			if (error_size != 0) {
				std::string error_log(error_size + 1, '\0');
				NVPTXCOMPILER_SAFE_CALL(
					nvPTXCompilerGetErrorLog(
						compiler, error_log.data()));
				SPDLOG_ERROR("Unable to compile: {}",
					     error_log);
			}
			return -1;
		}
		size_t compiled_size;
		NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetCompiledProgramSize(
			compiler, &compiled_size));
		output_elf.resize(compiled_size);
		NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetCompiledProgram(
			compiler, (void *)output_elf.data()));
		size_t info_size;
		NVPTXCOMPILER_SAFE_CALL(
			nvPTXCompilerGetInfoLogSize(compiler, &info_size));
		std::string info_log(info_size + 1, '\0');
		NVPTXCOMPILER_SAFE_CALL(
			nvPTXCompilerGetInfoLog(compiler, info_log.data()));
		SPDLOG_INFO("{}", info_log);
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
	if (record_itr == symbol_address_to_fatbin.end()) {
		SPDLOG_DEBUG(
			"In mirror_cuda_memcpy_to_symbol: calling original cudaMemcpyToSymbol");
		if (async) {
			cudaMemcpyToSymbolAsync(symbol, src, count, offset,
						kind, stream);
		} else {
			cudaMemcpyToSymbol(symbol, src, count, offset, kind);
		}
		return;
	}
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
