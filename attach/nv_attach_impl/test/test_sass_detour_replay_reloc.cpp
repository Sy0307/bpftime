#include "../sass_detour/sass_detour.hpp"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace
{
constexpr uint8_t ELF_MAGIC[4] = { 0x7f, 'E', 'L', 'F' };
constexpr size_t SASS_INST_BYTES = 16;

struct Elf64_Ehdr {
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
};

struct Elf64_Shdr {
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

struct Elf64_Sym {
	uint32_t st_name;
	uint8_t st_info;
	uint8_t st_other;
	uint16_t st_shndx;
	uint64_t st_value;
	uint64_t st_size;
};

template <typename T>
static bool read_pod(std::span<const uint8_t> bytes, size_t offset, T &out)
{
	if (offset + sizeof(T) > bytes.size())
		return false;
	std::memcpy(&out, bytes.data() + offset, sizeof(T));
	return true;
}

static std::optional<std::string_view>
read_cstr(std::span<const uint8_t> data, size_t offset)
{
	if (offset >= data.size())
		return std::nullopt;
	const char *p = reinterpret_cast<const char *>(data.data() + offset);
	const size_t max_len = data.size() - offset;
	const size_t len = strnlen(p, max_len);
	if (offset + len >= data.size())
		return std::nullopt;
	return std::string_view(p, len);
}

static std::optional<int64_t> decode_rel_imm36_delta_bytes_sm120(uint64_t w0)
{
	const uint64_t low6 = (w0 >> 18) & 0x3fu;
	const uint64_t high30 = (w0 >> 34) & 0x3fffffffu;
	uint64_t imm_u = (high30 << 6) | low6;
	if (imm_u & (1ull << 35))
		imm_u |= ~((1ull << 36) - 1ull);
	const int64_t imm = static_cast<int64_t>(imm_u);
	return (imm + 1) * (int64_t)SASS_INST_BYTES;
}

static std::vector<uint8_t> read_file(const std::filesystem::path &p)
{
	std::ifstream ifs(p, std::ios::binary);
	REQUIRE(ifs.good());
	return std::vector<uint8_t>(std::istreambuf_iterator<char>(ifs),
				    std::istreambuf_iterator<char>());
}

struct SectionView {
	size_t file_off = 0;
	size_t size = 0;
};

static std::optional<SectionView>
find_section(std::span<const uint8_t> elf, std::string_view name)
{
	Elf64_Ehdr ehdr {};
	if (!read_pod(elf, 0, ehdr))
		return std::nullopt;
	if (std::memcmp(ehdr.e_ident, ELF_MAGIC, sizeof(ELF_MAGIC)) != 0)
		return std::nullopt;
	if (ehdr.e_ident[4] != 2 /* ELFCLASS64 */ || ehdr.e_ident[5] != 1 /* LSB */)
		return std::nullopt;
	if (ehdr.e_shoff == 0 || ehdr.e_shentsize != sizeof(Elf64_Shdr) ||
	    ehdr.e_shnum == 0 || ehdr.e_shstrndx >= ehdr.e_shnum)
		return std::nullopt;

	const size_t shoff = static_cast<size_t>(ehdr.e_shoff);
	const size_t shnum = static_cast<size_t>(ehdr.e_shnum);
	const size_t shstrndx = static_cast<size_t>(ehdr.e_shstrndx);

	Elf64_Shdr shstr {};
	if (!read_pod(elf, shoff + shstrndx * sizeof(Elf64_Shdr), shstr))
		return std::nullopt;
	if (shstr.sh_offset + shstr.sh_size > elf.size())
		return std::nullopt;
	auto shstrtab =
		elf.subspan(static_cast<size_t>(shstr.sh_offset),
			    static_cast<size_t>(shstr.sh_size));

	for (size_t i = 0; i < shnum; i++) {
		Elf64_Shdr sh {};
		if (!read_pod(elf, shoff + i * sizeof(Elf64_Shdr), sh))
			return std::nullopt;
		auto s = read_cstr(shstrtab, sh.sh_name);
		if (!s || *s != name)
			continue;
		if (sh.sh_offset + sh.sh_size > elf.size())
			return std::nullopt;
		return SectionView { static_cast<size_t>(sh.sh_offset),
				     static_cast<size_t>(sh.sh_size) };
	}
	return std::nullopt;
}

static std::optional<uint64_t>
find_symbol_value(std::span<const uint8_t> elf, std::string_view sym_name)
{
	auto symtab_sec = find_section(elf, ".symtab");
	auto strtab_sec = find_section(elf, ".strtab");
	if (!symtab_sec || !strtab_sec)
		return std::nullopt;

	auto symtab = elf.subspan(symtab_sec->file_off, symtab_sec->size);
	auto strtab = elf.subspan(strtab_sec->file_off, strtab_sec->size);

	if (symtab.size() < sizeof(Elf64_Sym) || (symtab.size() % sizeof(Elf64_Sym)) != 0)
		return std::nullopt;
	const size_t n = symtab.size() / sizeof(Elf64_Sym);
	for (size_t i = 0; i < n; i++) {
		Elf64_Sym sym {};
		std::memcpy(&sym, symtab.data() + i * sizeof(Elf64_Sym),
			    sizeof(Elf64_Sym));
		auto s = read_cstr(strtab, sym.st_name);
		if (!s || *s != sym_name)
			continue;
		return sym.st_value;
	}
	return std::nullopt;
}

} // namespace

TEST_CASE("SASS detour relocates CALL within replay window", "[sass_detour][sm120]")
{
	const std::filesystem::path repo_root = std::filesystem::current_path();
	const std::filesystem::path cu_path =
		repo_root / "benchmark/gpu/host/cubin_call_entry.cu";
	REQUIRE(std::filesystem::exists(cu_path));

	const std::filesystem::path out_cubin =
		std::filesystem::temp_directory_path() / "bpftime_call_entry_test.cubin";

	// Compile cubin (does not require a GPU, only nvcc/ptxas).
	const char *nvcc = std::getenv("NVCC");
	const std::string nvcc_path =
		(nvcc && *nvcc) ? std::string(nvcc) : std::string("/usr/local/cuda/bin/nvcc");
	const std::string arch =
		(std::getenv("CUDA_ARCH") && *std::getenv("CUDA_ARCH"))
			? std::string(std::getenv("CUDA_ARCH"))
			: std::string("sm_120");
	{
		const std::string cmd = nvcc_path + " -arch=" + arch +
					" -O0 --cubin " + cu_path.string() +
					" -o " + out_cubin.string();
		const int rc = std::system(cmd.c_str());
		REQUIRE(rc == 0);
	}

	auto elf_bytes = read_file(out_cubin);
	REQUIRE(elf_bytes.size() > 4096);

	// Force a replay window that includes the CALL in the kernel prologue.
	// The `call_entry` kernel compiled with -O0 typically emits a CALL at PC=0x60.
	setenv("BPFTIME_CUDA_SASS_DETOUR_REPLAY_N", "8", 1);

	auto det = bpftime::attach::sass_detour::apply_elf_text_detours_sm120(
		elf_bytes, "call_entry", /*sample_section_filter=*/"", /*sampling_cfg=*/nullptr);
	REQUIRE(det.has_value());
	REQUIRE(det->patched_text_sections >= 1);

	// Locate .text.call_entry.
	auto text = find_section(std::span<const uint8_t>(elf_bytes.data(), elf_bytes.size()),
				 ".text.call_entry");
	REQUIRE(text.has_value());
	REQUIRE(text->size >= 0x100);

	// Decode the entry BRA to locate trampoline.
	uint64_t entry_w0 = 0, entry_w1 = 0;
	std::memcpy(&entry_w0, elf_bytes.data() + text->file_off, 8);
	std::memcpy(&entry_w1, elf_bytes.data() + text->file_off + 8, 8);
	REQUIRE((entry_w0 & 0xffffu) == 0x7947u); // BRA
	auto tramp_delta = decode_rel_imm36_delta_bytes_sm120(entry_w0);
	REQUIRE(tramp_delta.has_value());
	const size_t tramp_pc = static_cast<size_t>(*tramp_delta);
	REQUIRE(tramp_pc + 9 * SASS_INST_BYTES <= text->size);

	// The CALL is expected at instruction offset 0x60 within the replay window.
	const size_t call_pc = tramp_pc + 0x60;
	uint64_t call_w0 = 0, call_w1 = 0;
	std::memcpy(&call_w0, elf_bytes.data() + text->file_off + call_pc, 8);
	std::memcpy(&call_w1, elf_bytes.data() + text->file_off + call_pc + 8, 8);
	REQUIRE((call_w0 & 0xffffu) == 0x7944u); // CALL (nvcc -O0 observed)
	auto call_delta = decode_rel_imm36_delta_bytes_sm120(call_w0);
	REQUIRE(call_delta.has_value());
	const size_t call_target_pc = call_pc + static_cast<size_t>(*call_delta);

	// The callee symbol is emitted as `$call_entry$_Z3fooi` in the same .text section.
	auto sym_val = find_symbol_value(
		std::span<const uint8_t>(elf_bytes.data(), elf_bytes.size()),
		"$call_entry$_Z3fooi");
	REQUIRE(sym_val.has_value());
	REQUIRE(call_target_pc == static_cast<size_t>(*sym_val));

	// The final BRA in the trampoline should return to entry + replay_n*16 == 0x80.
	const size_t back_bra_pc = tramp_pc + 8 * SASS_INST_BYTES;
	uint64_t back_w0 = 0, back_w1 = 0;
	std::memcpy(&back_w0, elf_bytes.data() + text->file_off + back_bra_pc, 8);
	std::memcpy(&back_w1, elf_bytes.data() + text->file_off + back_bra_pc + 8, 8);
	REQUIRE((back_w0 & 0xffffu) == 0x7947u);
	auto back_delta = decode_rel_imm36_delta_bytes_sm120(back_w0);
	REQUIRE(back_delta.has_value());
	REQUIRE(back_bra_pc + static_cast<size_t>(*back_delta) == 0x80);
}

TEST_CASE("SASS thread_map lane0 gate uses tid.x&31 + @!P0 store", "[sass_detour][sm120]")
{
	bpftime::attach::sass_detour::Sm120SamplingConfig sampling {};
	sampling.enabled = true;
	sampling.mode = bpftime::attach::sass_detour::Sm120SamplingConfig::Mode::ThreadMap;
	sampling.sample_buffer_device_ptr = 0x1122334455667788ull;
	sampling.max_records = 1;
	sampling.thread_map_device = true;
	sampling.thread_map_device_lane0_only = true;
	sampling.thread_map_device_stride4 = true;
	sampling.desc_ur = 62;

	auto insts = bpftime::attach::sass_detour::sm120_build_thread_map_stub_for_test(
		sampling, /*old_regcount=*/32);
	REQUIRE(!insts.empty());

	// Scan the stub for the lane0 gate sequence:
	// - LOP3 lane!=0 -> P0
	// - @!P0 STG.E (u32)

	bool found_lop3_gate = false;
	bool found_not_p0_stg = false;

	const uint64_t want_stg_w1 =
		0x0009e2000c10090cull & 0xffffffffffffff00ull |
		(uint64_t(sampling.desc_ur) & 0xffu);

	for (const auto &it : insts) {
		const uint64_t w0 = it[0];
		const uint64_t w1 = it[1];
		const uint16_t op = static_cast<uint16_t>(w0 & 0xffffu);
		if (op == 0x7812u && w1 == 0x002fda000780c0ffull)
			found_lop3_gate = true;
		if (op == 0x8986u && w1 == want_stg_w1)
			found_not_p0_stg = true;
	}

	REQUIRE(found_lop3_gate);
	REQUIRE(found_not_p0_stg);
}
