#include <atomic>
#include <cstdint>

namespace bpftime::cuda
{
namespace
{
std::atomic<uintptr_t> g_cuda_comm_shared_mem_device_ptr{ 0 };
}

uintptr_t get_cuda_shared_mem_device_pointer()
{
	return g_cuda_comm_shared_mem_device_ptr.load(std::memory_order_acquire);
}

void set_cuda_shared_mem_device_pointer(uintptr_t ptr)
{
	g_cuda_comm_shared_mem_device_ptr.store(ptr, std::memory_order_release);
}
} // namespace bpftime::cuda

