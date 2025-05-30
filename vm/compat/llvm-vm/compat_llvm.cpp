#include "llvmbpf.hpp"
#include "spdlog/spdlog.h"
#include <bpftime_vm_compat.hpp>
#include <compat_llvm.hpp>
#include <optional>
#include <memory>

namespace bpftime::vm::compat
{

std::unique_ptr<bpftime_vm_impl> create_llvm_vm_instance()
{
	SPDLOG_DEBUG("create llvm vm instance");
	return std::make_unique<llvm::bpftime_llvm_vm>();
}

} // namespace bpftime::vm::compat

namespace bpftime::vm::llvm
{

int bpftime_llvm_vm::register_external_function(size_t index,
						const std::string &name,
						void *fn)
{
	return bpftime::llvmbpf_vm::register_external_function(index, name, fn);
}

void bpftime_llvm_vm::unload_code()
{
	bpftime::llvmbpf_vm::unload_code();
}

int bpftime_llvm_vm::exec(void *mem, size_t mem_len, uint64_t &bpf_return_value)
{
	return bpftime::llvmbpf_vm::exec(mem, mem_len, bpf_return_value);
}

std::vector<uint8_t> bpftime_llvm_vm::do_aot_compile(bool print_ir)
{
	std::optional<std::vector<uint8_t> > bytecode_optional =
		bpftime::llvmbpf_vm::do_aot_compile(print_ir);
	if (bytecode_optional.has_value()) {
		return bytecode_optional.value();
	} else {
		throw std::runtime_error(
			"bpftime_llvm_vm::do_aot_compile: AOT compilation failed, bytecode_optional is empty.");
	}
}

std::optional<compat::precompiled_ebpf_function>
bpftime_llvm_vm::load_aot_object(const std::vector<uint8_t> &object)
{
	return bpftime::llvmbpf_vm::load_aot_object(object);
}

std::optional<compat::precompiled_ebpf_function> bpftime_llvm_vm::compile()
{
	return bpftime::llvmbpf_vm::compile();
}

void bpftime_llvm_vm::set_lddw_helpers(uint64_t (*map_by_fd)(uint32_t),
				       uint64_t (*map_by_idx)(uint32_t),
				       uint64_t (*map_val)(uint64_t),
				       uint64_t (*var_addr)(uint32_t),
				       uint64_t (*code_addr)(uint32_t))
{
	bpftime::llvmbpf_vm::set_lddw_helpers(map_by_fd, map_by_idx, map_val,
					      var_addr, code_addr);
}
std::optional<std::string> bpftime_llvm_vm::generate_ptx(const char *target_cpu)
{
	return bpftime::llvmbpf_vm::generate_ptx(target_cpu);
}
} // namespace bpftime::vm::llvm

namespace bpftime::vm::compat
{
namespace llvm
{
__attribute__((constructor(0))) static inline void register_llvm_vm_factory()
{
	register_vm_factory("llvm", create_llvm_vm_instance);
	SPDLOG_DEBUG("llvm register vm factory\n");
}

} // namespace llvm
} // namespace bpftime::vm::compat
