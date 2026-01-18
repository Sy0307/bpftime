#include "sass_detour.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <spdlog/spdlog.h>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <unistd.h>

namespace bpftime::attach::sass_detour
{
namespace
{
constexpr uint8_t ELF_MAGIC[4] = { 0x7f, 'E', 'L', 'F' };
constexpr size_t SASS_INST_BYTES = 16;
constexpr uint16_t EIATTR_REGCOUNT = 0x0401;
constexpr uint8_t NVINFO_KIND_SMALL = 0x03;
// CUDA 12.x SM100+ nvinfo "small" entry IDs vary across toolchains.
// Observed on SM120 cubin-only images:
// - id=0x1b: regcount (value_le16 <= 255)
// - id=0x19: other attribute (often > 255, e.g. 0x01b8), NOT regcount
// Keep a fallback for older variants where regcount may still be 0x19.
constexpr uint8_t NVINFO_SMALL_REGCOUNT_ID_PRIMARY = 0x1b;
constexpr uint8_t NVINFO_SMALL_REGCOUNT_ID_FALLBACK = 0x19;
constexpr uint16_t NVINFO_GLOBAL_REGCOUNT_TYPE = 0x2f04;

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

struct Elf64_Phdr {
	uint32_t p_type;
	uint32_t p_flags;
	uint64_t p_offset;
	uint64_t p_vaddr;
	uint64_t p_paddr;
	uint64_t p_filesz;
	uint64_t p_memsz;
	uint64_t p_align;
};

constexpr uint32_t SHT_NOBITS = 8;
constexpr uint32_t PT_LOAD = 1;

template <typename T>
static bool read_pod(std::span<const uint8_t> bytes, size_t offset, T &out)
{
	if (offset + sizeof(T) > bytes.size())
		return false;
	std::memcpy(&out, bytes.data() + offset, sizeof(T));
	return true;
}

template <typename T>
static bool write_pod(std::span<uint8_t> bytes, size_t offset, const T &in)
{
	if (offset + sizeof(T) > bytes.size())
		return false;
	std::memcpy(bytes.data() + offset, &in, sizeof(T));
	return true;
}

static std::optional<std::span<const uint8_t>>
get_shstrtab(std::span<const uint8_t> elf, const Elf64_Ehdr &ehdr)
{
	if (ehdr.e_shoff == 0 || ehdr.e_shentsize == 0 || ehdr.e_shnum == 0)
		return std::nullopt;
	if (ehdr.e_shentsize != sizeof(Elf64_Shdr))
		return std::nullopt;
	const size_t shoff = static_cast<size_t>(ehdr.e_shoff);
	const size_t shnum = static_cast<size_t>(ehdr.e_shnum);
	const size_t shstrndx = static_cast<size_t>(ehdr.e_shstrndx);
	if (shstrndx >= shnum)
		return std::nullopt;
	if (shoff + shnum * sizeof(Elf64_Shdr) > elf.size())
		return std::nullopt;
	Elf64_Shdr shstr {};
	if (!read_pod(elf, shoff + shstrndx * sizeof(Elf64_Shdr), shstr))
		return std::nullopt;
	if (shstr.sh_offset + shstr.sh_size > elf.size())
		return std::nullopt;
	return elf.subspan(static_cast<size_t>(shstr.sh_offset),
			   static_cast<size_t>(shstr.sh_size));
}

static std::optional<std::span<const uint8_t>>
get_shstrtab_from_shdrs(std::span<const uint8_t> elf, const Elf64_Ehdr &ehdr,
			const std::vector<Elf64_Shdr> &shdrs)
{
	const size_t shnum = static_cast<size_t>(ehdr.e_shnum);
	const size_t shstrndx = static_cast<size_t>(ehdr.e_shstrndx);
	if (shstrndx >= shnum || shstrndx >= shdrs.size())
		return std::nullopt;
	const auto &shstr = shdrs[shstrndx];
	const size_t off = static_cast<size_t>(shstr.sh_offset);
	const size_t sz = static_cast<size_t>(shstr.sh_size);
	if (off + sz > elf.size())
		return std::nullopt;
	return elf.subspan(off, sz);
}

static std::string_view safe_cstr(std::span<const uint8_t> tab, size_t offset)
{
	if (offset >= tab.size())
		return {};
	const char *p = reinterpret_cast<const char *>(tab.data() + offset);
	const size_t max_len = tab.size() - offset;
	const size_t len = strnlen(p, max_len);
	return std::string_view(p, len);
}

struct BraInst {
	uint64_t w0;
	uint64_t w1;
};

static uint64_t set_u64_byte(uint64_t v, int byte_index, uint8_t b)
{
	const uint64_t mask = 0xffull << (byte_index * 8);
	return (v & ~mask) | (uint64_t(b) << (byte_index * 8));
}

static uint32_t fnv1a32(std::string_view s)
{
	uint32_t h = 2166136261u;
	for (unsigned char c : s) {
		h ^= uint32_t(c);
		h *= 16777619u;
	}
	return h;
}

static std::optional<BraInst> encode_bra_sm120(int64_t delta_bytes,
					       uint16_t w0_base_low16)
{
	if ((delta_bytes % (int64_t)SASS_INST_BYTES) != 0)
		return std::nullopt;

	// cuobjdump shows: imm = (target - pc) / 16 - 1
	const int64_t imm = (delta_bytes / (int64_t)SASS_INST_BYTES) - 1;

	// Immediate is a 36-bit signed value split into:
	// - low6 in bits [23:18]
	// - high30 in bits [63:34]
	// Observed on sm_120 via nvptxcompiler+cuobjdump.
	const int64_t imm_min = -(int64_t(1) << 35);
	const int64_t imm_max = (int64_t(1) << 35) - 1;
	if (imm < imm_min || imm > imm_max)
		return std::nullopt;

	const uint64_t low6 = static_cast<uint64_t>(imm) & 0x3fu;
	const uint64_t high30 =
		(static_cast<uint64_t>(imm >> 6)) & 0x3fffffffu;

	const uint64_t BRA_W0_BASE =
		0x0000000000000000ULL | uint64_t(w0_base_low16);
	const uint64_t w0 =
		BRA_W0_BASE | (low6 << 18) | (high30 << 34);

	// w1 (control word) depends on branch direction and whether the BRA is
	// predicated. This is not just performance metadata: incorrect control
	// words can trigger "illegal instruction" on real workloads when the
	// predicated BRA is taken.
	//
	// Verified from ptxas 12.9 sm_120 output (nvdisasm -hex):
	// - forward BRA (pred/unpred):       w1=0x003fde0003800000
	// - backward/self-loop BRA (unpred): w1=0x000fc0000383ffff
	// - backward/self-loop BRA (pred):   w1=0x003fde000383ffff
	//
	// Note: the unpredicated BRA uses opcode 0x7947; predicated uses 0x?947
	// (predicate id/neg in the high nibble).
	const bool is_pred = (w0_base_low16 != 0x7947u);
	const uint64_t w1 =
		(imm >= 0) ? 0x003fde0003800000ULL
			   : (is_pred ? 0x003fde000383ffffULL
				      : 0x000fc0000383ffffULL);
	return BraInst { w0, w1 };
}

static std::optional<BraInst> encode_bra_sm120_unpred(int64_t delta_bytes)
{
	return encode_bra_sm120(delta_bytes, 0x7947);
}

static std::optional<BraInst> encode_bra_sm120_pred(int64_t delta_bytes, uint8_t pred,
						    bool neg)
{
	uint16_t op16 =
		uint16_t((uint16_t(pred & 0x7u) << 12) | uint16_t(0x0947u));
	if (neg)
		op16 |= 0x8000u;
	return encode_bra_sm120(delta_bytes, op16);
}

static std::optional<BraInst> encode_bra_sm120_pred(int64_t delta_bytes, uint8_t pred)
{
	return encode_bra_sm120_pred(delta_bytes, pred, /*neg=*/false);
}

static std::optional<BraInst> encode_bra_sm120_pred_p0(int64_t delta_bytes)
{
	// Predicated BRA differs from the unpredicated form in the low 16-bit
	// opcode field (observed from ptxas output): 0x0947 vs 0x7947.
	auto b = encode_bra_sm120_pred(delta_bytes, /*pred=*/0u);
	if (!b)
		return std::nullopt;
	return b;
}

static uint8_t sm120_pack_pred(uint8_t pred, bool neg)
{
	return uint8_t((pred & 0x7u) | (neg ? 0x80u : 0u));
}

static uint8_t sm120_unpack_pred(uint8_t pred_code)
{
	return uint8_t(pred_code & 0x7u);
}

static bool sm120_unpack_pred_neg(uint8_t pred_code)
{
	return (pred_code & 0x80u) != 0;
}

static std::optional<int64_t> decode_bra_delta_bytes_sm120(uint64_t w0)
{
	// Inverse of encode_bra_sm120():
	// imm = (target - pc) / 16 - 1  (36-bit signed)
	// - low6 in bits [23:18]
	// - high30 in bits [63:34]
	const uint64_t low6 = (w0 >> 18) & 0x3fu;
	const uint64_t high30 = (w0 >> 34) & 0x3fffffffu;
	uint64_t imm_u = (high30 << 6) | low6;
	// Sign-extend 36-bit immediate.
	if (imm_u & (1ull << 35))
		imm_u |= ~((1ull << 36) - 1ull);
	const int64_t imm = static_cast<int64_t>(imm_u);
	const int64_t delta_bytes = (imm + 1) * (int64_t)SASS_INST_BYTES;
	if ((delta_bytes % (int64_t)SASS_INST_BYTES) != 0)
		return std::nullopt;
	return delta_bytes;
}

static std::optional<uint64_t>
infer_bra_w1_template_sm120(std::span<const uint8_t> text_section,
			    bool want_forward)
{
	const size_t ninst = text_section.size() / SASS_INST_BYTES;
	const size_t scan = std::min<size_t>(ninst, 8192);
	for (size_t i = 0; i < scan; i++) {
		const size_t off = i * SASS_INST_BYTES;
		uint64_t w0 = 0, w1 = 0;
		std::memcpy(&w0, text_section.data() + off, sizeof(w0));
		std::memcpy(&w1, text_section.data() + off + 8, sizeof(w1));
		const uint16_t op16 = uint16_t(w0 & 0xffffu);
		// BRA op16 low 12 bits are 0x947 (unpred is 0x7947; pred encodes pred/neg).
		if ((op16 & 0x0fffu) != 0x0947u)
			continue;
		auto d = decode_bra_delta_bytes_sm120(w0);
		if (!d)
			continue;
		if (want_forward && *d > 0)
			return w1;
		if (!want_forward && *d < 0)
			return w1;
	}
	return std::nullopt;
}

static std::optional<uint64_t>
patch_rel_imm36_in_w0_sm120(uint64_t w0, int64_t delta_bytes)
{
	if ((delta_bytes % (int64_t)SASS_INST_BYTES) != 0)
		return std::nullopt;

	// Same immediate encoding as `encode_bra_sm120()`:
	// imm = (target - pc) / 16 - 1  (36-bit signed)
	const int64_t imm = (delta_bytes / (int64_t)SASS_INST_BYTES) - 1;

	const int64_t imm_min = -(int64_t(1) << 35);
	const int64_t imm_max = (int64_t(1) << 35) - 1;
	if (imm < imm_min || imm > imm_max)
		return std::nullopt;

	const uint64_t low6 = static_cast<uint64_t>(imm) & 0x3fu;
	const uint64_t high30 =
		(static_cast<uint64_t>(imm >> 6)) & 0x3fffffffu;

	const uint64_t IMM_MASK =
		(0x3full << 18) | (0x3fffffffull << 34);
	uint64_t out = w0 & ~IMM_MASK;
	out |= (low6 << 18) | (high30 << 34);
	return out;
}

static bool is_rel_imm36_ctrl_sm120(uint16_t op16)
{
	// SM120 uses a 36-bit PC-relative immediate encoding (imm = delta/16 - 1)
	// for several control-flow instructions. We only relocate a minimal set of
	// opcodes that were observed in real cubin-only workloads and in nvcc
	// output. Predicated forms typically differ in the low 16-bit opcode field.
	switch (op16) {
	case 0x7947: // BRA
	case 0x0947: // @P0 BRA (observed)
	case 0x7944: // CALL (observed in nvcc -O0)
	case 0x0944: // predicated CALL (best-effort)
		return true;
	default:
		return false;
	}
}

static bool write_u64_le(std::span<uint8_t> out, size_t off, uint64_t v)
{
	if (off + 8 > out.size())
		return false;
	for (size_t i = 0; i < 8; i++)
		out[off + i] = static_cast<uint8_t>((v >> (i * 8)) & 0xff);
	return true;
}

static bool write_sass_inst(std::span<uint8_t> out, size_t off, uint64_t w0,
			    uint64_t w1)
{
	if (off + SASS_INST_BYTES > out.size())
		return false;
	return write_u64_le(out, off, w0) && write_u64_le(out, off + 8, w1);
}

static bool insert_file_gap(std::vector<uint8_t> &elf_bytes, size_t insert_off,
			    size_t gap_bytes)
{
	if (gap_bytes == 0)
		return true;
	if (insert_off > elf_bytes.size())
		return false;
	const size_t old_size = elf_bytes.size();
	elf_bytes.resize(old_size + gap_bytes);
	std::memmove(elf_bytes.data() + insert_off + gap_bytes,
		     elf_bytes.data() + insert_off, old_size - insert_off);
	// Fill with SM120 NOPs to keep the new region executable-safe by default.
	auto out = std::span<uint8_t>(elf_bytes.data(), elf_bytes.size());
	constexpr uint64_t NOP_W0 = 0x0000000000007918ULL;
	constexpr uint64_t NOP_W1 = 0x000fc00000000000ULL;
	for (size_t off = 0; off + SASS_INST_BYTES <= gap_bytes;
	     off += SASS_INST_BYTES) {
		(void)write_sass_inst(out, insert_off + off, NOP_W0, NOP_W1);
	}
	// If the caller requests a non-instruction-aligned gap, just zero the tail.
	if ((gap_bytes % SASS_INST_BYTES) != 0) {
		const size_t tail = gap_bytes - (gap_bytes % SASS_INST_BYTES);
		std::memset(elf_bytes.data() + insert_off + tail, 0,
			    gap_bytes - tail);
	}
	return true;
}

static bool insert_sm120_text_cave_by_shifting(std::vector<uint8_t> &elf_bytes,
					       Elf64_Ehdr &ehdr,
					       std::vector<Elf64_Shdr> &shdrs,
					       std::vector<Elf64_Phdr> &phdrs,
					       size_t insert_off,
					       size_t cave_bytes)
{
	if (cave_bytes == 0)
		return false;
	if ((cave_bytes % SASS_INST_BYTES) != 0)
		return false;

	if (!insert_file_gap(elf_bytes, insert_off, cave_bytes))
		return false;

	if (ehdr.e_phoff >= insert_off)
		ehdr.e_phoff += cave_bytes;
	if (ehdr.e_shoff >= insert_off)
		ehdr.e_shoff += cave_bytes;

	for (auto &ph : phdrs) {
		if (ph.p_offset == 0)
			continue;
		if (ph.p_offset >= insert_off) {
			ph.p_offset += cave_bytes;
			continue;
		}
		const uint64_t seg_end = ph.p_offset + ph.p_filesz;
		if (ph.p_type == PT_LOAD && ph.p_filesz != 0 &&
		    ph.p_offset < insert_off &&
		    seg_end >= insert_off) {
			ph.p_filesz += cave_bytes;
			ph.p_memsz += cave_bytes;
		}
	}

	for (auto &sh : shdrs) {
		if (sh.sh_offset == 0)
			continue;
		if (sh.sh_offset >= insert_off)
			sh.sh_offset += cave_bytes;
		// Note: SHT_NOBITS has no file payload, but sh_offset is still used by
		// some tooling; keep ordering consistent by shifting it too.
		(void)SHT_NOBITS;
	}
	return true;
}

static bool write_updated_headers(std::vector<uint8_t> &elf_bytes,
				  const Elf64_Ehdr &ehdr,
				  const std::vector<Elf64_Phdr> &phdrs,
				  const std::vector<Elf64_Shdr> &shdrs)
{
	auto out = std::span<uint8_t>(elf_bytes.data(), elf_bytes.size());
	if (!write_pod(out, 0, ehdr))
		return false;

	if (!phdrs.empty()) {
		if (ehdr.e_phentsize != sizeof(Elf64_Phdr))
			return false;
		const size_t phoff = static_cast<size_t>(ehdr.e_phoff);
		const size_t phnum = static_cast<size_t>(ehdr.e_phnum);
		if (phnum != phdrs.size())
			return false;
		for (size_t i = 0; i < phnum; i++) {
			if (!write_pod(out, phoff + i * sizeof(Elf64_Phdr),
				       phdrs[i]))
				return false;
		}
	}

	if (ehdr.e_shoff == 0 || ehdr.e_shnum == 0)
		return false;
	if (ehdr.e_shentsize != sizeof(Elf64_Shdr))
		return false;
	const size_t shoff = static_cast<size_t>(ehdr.e_shoff);
	const size_t shnum = static_cast<size_t>(ehdr.e_shnum);
	if (shnum != shdrs.size())
		return false;
	for (size_t i = 0; i < shnum; i++) {
		if (!write_pod(out, shoff + i * sizeof(Elf64_Shdr), shdrs[i]))
			return false;
	}
	return true;
}

static std::optional<uint64_t> read_u64_le(std::span<const uint8_t> in,
					   size_t off)
{
	if (off + 8 > in.size())
		return std::nullopt;
	uint64_t v = 0;
	for (size_t i = 0; i < 8; i++)
		v |= uint64_t(in[off + i]) << (i * 8);
	return v;
}

static bool is_nop_sm120(std::span<const uint8_t> inst)
{
	if (inst.size() < SASS_INST_BYTES)
		return false;
	auto w0 = read_u64_le(inst, 0);
	auto w1 = read_u64_le(inst, 8);
	if (!w0 || !w1)
		return false;
	return *w0 == 0x0000000000007918ULL &&
	       *w1 == 0x000fc00000000000ULL;
}

enum class Sm120ExitKind : uint8_t {
	NotExit = 0,
	ExitUnpred,
	ExitPred,
};

static Sm120ExitKind exit_kind_from_op16_sm120(uint16_t op16, uint8_t *out_pred,
					       bool *out_neg)
{
	// Observed on sm_120 via ptxas 12.9 + cuobjdump:
	// - EXIT:      op16=0x794d (pred=7)
	// - @P0 EXIT:  op16=0x094d (pred=0)
	// - @!P0 EXIT: op16=0x894d (pred=0, neg=1)
	// - @P1 EXIT:  op16=0x194d (pred=1)
	// - @P2 EXIT:  op16=0x294d (pred=2)
	// ...
	// i.e. op16 = (pred << 12) | 0x094d  (plus an optional negation bit)
	if ((op16 & 0x0fffu) != 0x094du)
		return Sm120ExitKind::NotExit;
	const uint8_t pred = uint8_t((op16 >> 12) & 0x7u);
	const bool neg = (op16 & 0x8000u) != 0;
	if (out_pred)
		*out_pred = pred;
	if (out_neg)
		*out_neg = neg;
	if (pred == 7u)
		return Sm120ExitKind::ExitUnpred;
	return Sm120ExitKind::ExitPred;
}

static std::optional<size_t>
find_tail_exit_sm120(std::span<const uint8_t> section_bytes, bool prefer_unpred,
		     bool allow_pred_p0, bool allow_pred_any, uint8_t *out_pred,
		     bool *out_neg)
{
	if ((section_bytes.size() % SASS_INST_BYTES) != 0)
		return std::nullopt;
	auto pred_allowed = [&](uint8_t pred) -> bool {
		if (pred == 7u)
			return true;
		if (allow_pred_any)
			return true;
		return allow_pred_p0 && pred == 0u;
	};
	std::optional<size_t> best_pred;
	uint8_t best_pred_idx = 0;
	bool best_pred_neg = false;
	for (size_t off = section_bytes.size(); off >= SASS_INST_BYTES;
	     off -= SASS_INST_BYTES) {
		auto w0 = read_u64_le(section_bytes, off - SASS_INST_BYTES);
		if (!w0)
			continue;
		uint8_t pred = 0;
		bool neg = false;
		const auto kind = exit_kind_from_op16_sm120(
			uint16_t(*w0 & 0xffffu), &pred, &neg);
		if (kind == Sm120ExitKind::NotExit)
			continue;
		if (kind == Sm120ExitKind::ExitUnpred)
		{
			if (out_pred)
				*out_pred = 7u;
			if (out_neg)
				*out_neg = false;
			return off - SASS_INST_BYTES;
		}
		if (kind == Sm120ExitKind::ExitPred && pred_allowed(pred)) {
			best_pred = off - SASS_INST_BYTES;
			best_pred_idx = pred;
			best_pred_neg = neg;
		}
		if (!prefer_unpred && best_pred) {
			if (out_pred)
				*out_pred = best_pred_idx;
			if (out_neg)
				*out_neg = best_pred_neg;
			return best_pred;
		}
	}
	if (best_pred) {
		if (out_pred)
			*out_pred = best_pred_idx;
		if (out_neg)
			*out_neg = best_pred_neg;
		return best_pred;
	}
	return std::nullopt;
}

static std::optional<size_t>
find_pre_exit_nop_sm120(std::span<const uint8_t> section_bytes, size_t exit_rel,
			size_t max_scan_insts)
{
	if ((section_bytes.size() % SASS_INST_BYTES) != 0)
		return std::nullopt;
	if ((exit_rel % SASS_INST_BYTES) != 0)
		return std::nullopt;
	if (exit_rel < SASS_INST_BYTES || exit_rel > section_bytes.size())
		return std::nullopt;
	if (max_scan_insts == 0)
		return std::nullopt;

	size_t scanned = 0;
	for (size_t off = exit_rel; off >= SASS_INST_BYTES && scanned < max_scan_insts;
	     off -= SASS_INST_BYTES, scanned++) {
		const size_t cand = off - SASS_INST_BYTES;
		auto inst = section_bytes.subspan(cand, SASS_INST_BYTES);
		if (is_nop_sm120(inst))
			return cand;
	}
	return std::nullopt;
}

static std::optional<size_t>
find_tail_nop_cave_sm120(std::span<const uint8_t> section_bytes,
			 size_t required_bytes)
{
	if (required_bytes == 0 || section_bytes.size() < required_bytes ||
	    (required_bytes % SASS_INST_BYTES) != 0)
		return std::nullopt;

	size_t nop_bytes = 0;
	for (size_t off = section_bytes.size(); off >= SASS_INST_BYTES;
	     off -= SASS_INST_BYTES) {
		auto inst = section_bytes.subspan(off - SASS_INST_BYTES,
						  SASS_INST_BYTES);
		if (!is_nop_sm120(inst))
			break;
		nop_bytes += SASS_INST_BYTES;
		if (nop_bytes >= required_bytes)
			return off - nop_bytes;
	}
	return std::nullopt;
}

static std::optional<std::filesystem::path> env_dir(const char *key)
{
	const char *v = std::getenv(key);
	if (!v || !*v)
		return std::nullopt;
	return std::filesystem::path(v);
}

static bool env_truthy(const char *key)
{
	const char *v = std::getenv(key);
	if (!v || !*v)
		return false;
	std::string s(v);
	std::transform(s.begin(), s.end(), s.begin(),
		       [](unsigned char c) { return (char)std::tolower(c); });
	return s == "1" || s == "true" || s == "yes" || s == "y" || s == "on";
}

static std::optional<uint32_t> env_u32(const char *key)
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

static uint8_t sass_detour_pr_save_mask()
{
	// Preserve only P0 by default. Users can opt into a wider predicate mask
	// (e.g., 0xff) when experimenting with more complex stubs.
	if (auto v = env_u32("BPFTIME_CUDA_SASS_DETOUR_PR_SAVE_MASK"))
		return uint8_t(*v & 0xffu);
	return 0x1;
}

static void maybe_dump_file(const std::optional<std::filesystem::path> &dir,
			    const std::string &filename,
			    std::span<const uint8_t> data)
{
	if (!dir)
		return;
	std::filesystem::create_directories(*dir);
	std::ofstream ofs(*dir / filename, std::ios::binary);
	ofs.write(reinterpret_cast<const char *>(data.data()),
		  static_cast<std::streamsize>(data.size()));
}

static uint64_t fnv1a64(std::span<const uint8_t> data)
{
	uint64_t h = 1469598103934665603ull;
	for (uint8_t b : data) {
		h ^= b;
		h *= 1099511628211ull;
	}
	return h;
}

static uint32_t sass_image_id(std::span<const uint8_t> data)
{
	// Best-effort stable identifier for a CUDA code object (ELF cubin bytes).
	// Used to scope func_id-based detours and avoid cross-module func_id collisions.
	const uint64_t h = fnv1a64(data);
	return uint32_t((h & 0xffffffffull) ^ (h >> 32));
}

static std::optional<std::filesystem::path> find_nvdisasm_path()
{
	if (const char *v = std::getenv("BPFTIME_CUDA_SASS_NVDISASM")) {
		if (*v)
			return std::filesystem::path(v);
	}
	for (const char *cand : {
		     "/usr/local/cuda-12.9/bin/nvdisasm",
		     "/usr/local/cuda/bin/nvdisasm",
		     "/usr/local/cuda-12/bin/nvdisasm",
	     }) {
		std::error_code ec;
		if (std::filesystem::exists(cand, ec))
			return std::filesystem::path(cand);
	}
	return std::nullopt;
}

static std::optional<std::string> run_command_capture_stdout(const std::string &cmd)
{
	FILE *fp = popen(cmd.c_str(), "r");
	if (!fp)
		return std::nullopt;
	std::string out;
	std::array<char, 4096> buf {};
	while (true) {
		const size_t n = std::fread(buf.data(), 1, buf.size(), fp);
		if (n == 0)
			break;
		out.append(buf.data(), n);
	}
	(void)pclose(fp);
	if (out.empty())
		return std::nullopt;
	return out;
}

static void mark_used_gprs_from_line(std::string_view line,
				     std::array<bool, 256> &used)
{
	for (size_t i = 0; i < line.size(); i++) {
		if (line[i] != 'R')
			continue;
		if ((i + 1) >= line.size() ||
		    !std::isdigit(static_cast<unsigned char>(line[i + 1])))
			continue;
		size_t j = i + 1;
		unsigned v = 0;
		while (j < line.size() &&
		       std::isdigit(static_cast<unsigned char>(line[j]))) {
			v = v * 10u + unsigned(line[j] - '0');
			j++;
		}
		if (v < used.size())
			used[v] = true;
		i = j;
	}
}

static std::optional<std::vector<uint8_t>>
compute_dead_gprs_by_nvdisasm(std::span<const uint8_t> cubin_bytes,
			      std::string_view section_name,
			      uint32_t old_regcount_u32,
			      size_t from_pc_bytes_in_section)
{
	const auto nvdisasm = find_nvdisasm_path();
	if (!nvdisasm)
		return std::nullopt;

	const uint64_t h = fnv1a64(cubin_bytes);
	std::error_code ec;
	auto tmp_dir = std::filesystem::temp_directory_path(ec);
	if (ec)
		return std::nullopt;
	auto tmp = tmp_dir / ("bpftime_nvdisasm_" + std::to_string(getpid()) + "_" +
			      std::to_string(h) + ".cubin");
	{
		std::ofstream ofs(tmp, std::ios::binary);
		if (!ofs)
			return std::nullopt;
		ofs.write(reinterpret_cast<const char *>(cubin_bytes.data()),
			  static_cast<std::streamsize>(cubin_bytes.size()));
	}

	const std::string cmd = nvdisasm->string() + " --print-code --print-line-info " +
				tmp.string() + " 2>/dev/null";
	auto out = run_command_capture_stdout(cmd);
	std::filesystem::remove(tmp, ec);
	if (!out)
		return std::nullopt;

	std::array<bool, 256> used {};
	bool in_section = false;
	for (size_t pos = 0; pos < out->size();) {
		const size_t nl = out->find('\n', pos);
		const size_t end = (nl == std::string::npos) ? out->size() : nl;
		std::string_view line(out->data() + pos, end - pos);
		pos = (nl == std::string::npos) ? out->size() : (nl + 1);

		if (line.starts_with("//--------------------- ")) {
			// New section header: reset state until we find `.section <name>`.
			in_section = false;
			continue;
		}

		constexpr std::string_view kSectionPrefix = "\t.section\t";
		if (line.starts_with(kSectionPrefix)) {
			auto rest = line.substr(kSectionPrefix.size());
			const size_t comma = rest.find(',');
			const auto sec = (comma == std::string_view::npos)
						 ? rest
						 : rest.substr(0, comma);
			in_section = (sec == section_name);
			continue;
		}

		if (!in_section)
			continue;

		// Parse instruction lines:
		//   /*00a0*/ <SASS...>
		const size_t l = line.find("/*");
		const size_t r = (l == std::string_view::npos) ? std::string_view::npos
							       : line.find("*/", l + 2);
		if (l == std::string_view::npos || r == std::string_view::npos)
			continue;
		auto hex = line.substr(l + 2, r - (l + 2));
		unsigned pc = 0;
		for (char c : hex) {
			if (c == ' ' || c == '\t')
				continue;
			pc <<= 4;
			if (c >= '0' && c <= '9')
				pc |= unsigned(c - '0');
			else if (c >= 'a' && c <= 'f')
				pc |= unsigned(10 + (c - 'a'));
			else if (c >= 'A' && c <= 'F')
				pc |= unsigned(10 + (c - 'A'));
			else {
				pc = 0;
				break;
			}
		}
		if (pc < from_pc_bytes_in_section)
			continue;

		mark_used_gprs_from_line(line, used);
	}

	const uint32_t nregs =
		std::min<uint32_t>(old_regcount_u32, uint32_t(255u));
	std::vector<uint8_t> dead;
	dead.reserve(nregs);
	for (uint32_t r = 0; r < nregs; r++) {
		if (!used[r])
			dead.push_back(uint8_t(r));
	}
	return dead;
}

static std::optional<std::array<uint8_t, 6>>
pick_exit_thread_map_scratch_regs_from_dead(std::span<const uint8_t> dead_gprs)
{
	std::array<bool, 256> dead {};
	for (uint8_t r : dead_gprs)
		dead[r] = true;

	// Need a `.64` pair for r_ptr: pick the highest even pair within [0..254].
	int r_ptr = -1;
	for (int r = 252; r >= 0; r -= 2) {
		if (dead[uint8_t(r)] && dead[uint8_t(r + 1)]) {
			r_ptr = r;
			break;
		}
	}
	if (r_ptr < 0)
		return std::nullopt;

	std::array<uint8_t, 6> out {};
	out[0] = uint8_t(r_ptr);
	int filled = 1;
	for (int r = 254; r >= 0 && filled < 6; r--) {
		if (!dead[uint8_t(r)])
			continue;
		if (r == r_ptr || r == (r_ptr + 1))
			continue;
		out[filled++] = uint8_t(r);
	}
	if (filled < 6)
		return std::nullopt;
	return out;
}

static std::optional<uint8_t>
infer_existing_desc_ur_base_near_sm120(std::span<const uint8_t> section_span,
				       size_t from_rel,
				       size_t scan_insts)
{
	// Heuristic: scan backward from `from_rel` for a nearby global memory
	// instruction that uses a `desc[URx]` operand, and reuse that descriptor's
	// UR base for our no-UR-write EXIT stub.
	//
	// This matters for predicated EXIT detours: the lanes that take the detour
	// may jump to the trampoline *before* the kernel's own prologue initializes
	// UR4/UR5 via `LDCU.64 UR4, c[...]`, so assuming `desc[UR4]` is always ready
	// can trigger illegal instruction faults.
	//
	// NOTE: For SM120, we only decode the few patterns we need:
	// - STG.* ... desc[URx] ... : `x` is typically encoded in w1[7:0]
	// - LDG.* ... desc[URx] ... : `x` is typically encoded in w0[39:32]
	if (scan_insts == 0)
		return std::nullopt;
	if (from_rel > section_span.size())
		return std::nullopt;

	const size_t max_back_bytes =
		std::min(from_rel, scan_insts * SASS_INST_BYTES);
	for (size_t back = SASS_INST_BYTES; back <= max_back_bytes;
	     back += SASS_INST_BYTES) {
		const size_t rel = from_rel - back;
		auto w0 = read_u64_le(section_span, rel);
		if (!w0)
			continue;
		auto w1 = read_u64_le(section_span, rel + 8);
		if (!w1)
			continue;
		const uint16_t op16 = uint16_t(*w0 & 0xffffu);
		// STG.* (common for global stores): use w1 low byte.
		if (op16 == 0x7986u) {
			const uint8_t urb = uint8_t(*w1 & 0xffu);
			if (urb < 64u)
				return urb;
			continue;
		}
		// LDG.* (common for global loads): use w0 byte[4] (bits 39:32).
		if (op16 == 0x7981u) {
			const uint8_t urb = uint8_t((*w0 >> 32) & 0xffu);
			if (urb < 64u)
				return urb;
			continue;
		}
	}
	return std::nullopt;
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

static std::optional<uint32_t>
read_nvinfo_regcount(std::span<const uint8_t> nvinfo)
{
	// Newer nvinfo format (CUDA 12.x SM100+ observed): a 4-byte entry
	// [kind=0x03][id][value_le16], where id=0x1b encodes regcount.
	auto scan_small = [&](uint8_t id) -> std::optional<uint32_t> {
		for (size_t off = 0; off + 4 <= nvinfo.size(); off++) {
			if (nvinfo[off + 0] != NVINFO_KIND_SMALL || nvinfo[off + 1] != id)
				continue;
			const uint32_t v = uint32_t(nvinfo[off + 2]) |
					   (uint32_t(nvinfo[off + 3]) << 8);
			// Regcount is always <= 255.
			if (v <= 255u)
				return v;
		}
		return std::nullopt;
	};
	if (auto v = scan_small(NVINFO_SMALL_REGCOUNT_ID_PRIMARY))
		return v;
	if (auto v = scan_small(NVINFO_SMALL_REGCOUNT_ID_FALLBACK))
		return v;

	// Older nvinfo format: u16 type + u16 size + payload, 4-byte aligned.
	size_t off = 0;
	while (off + 4 <= nvinfo.size()) {
		const uint16_t attr_type =
			uint16_t(nvinfo[off]) | (uint16_t(nvinfo[off + 1]) << 8);
		const uint16_t attr_size =
			uint16_t(nvinfo[off + 2]) |
			(uint16_t(nvinfo[off + 3]) << 8);
		off += 4;
		if (off + attr_size > nvinfo.size())
			break;
		if (attr_type == EIATTR_REGCOUNT && attr_size >= 4) {
			const uint32_t v = uint32_t(nvinfo[off]) |
					   (uint32_t(nvinfo[off + 1]) << 8) |
					   (uint32_t(nvinfo[off + 2]) << 16) |
					   (uint32_t(nvinfo[off + 3]) << 24);
			return v;
		}
		off += attr_size;
		off = (off + 3) & ~size_t(3);
	}
	return std::nullopt;
}

static std::optional<std::pair<uint32_t, uint32_t>>
patch_nvinfo_regcount(std::span<uint8_t> nvinfo, uint32_t new_regcount)
{
	// Newer nvinfo format: [0x03][0x1b][value_le16].
	auto patch_small = [&](uint8_t id) -> std::optional<std::pair<uint32_t, uint32_t>> {
		for (size_t off = 0; off + 4 <= nvinfo.size(); off++) {
			if (nvinfo[off + 0] != NVINFO_KIND_SMALL || nvinfo[off + 1] != id)
				continue;
			const uint32_t old_regcount = uint32_t(nvinfo[off + 2]) |
						      (uint32_t(nvinfo[off + 3]) << 8);
			if (old_regcount > 255u)
				continue;
			const uint16_t v = uint16_t(new_regcount & 0xffffu);
			nvinfo[off + 2] = uint8_t(v & 0xffu);
			nvinfo[off + 3] = uint8_t((v >> 8) & 0xffu);
			return std::make_pair(old_regcount, new_regcount);
		}
		return std::nullopt;
	};
	if (auto p = patch_small(NVINFO_SMALL_REGCOUNT_ID_PRIMARY))
		return p;
	if (auto p = patch_small(NVINFO_SMALL_REGCOUNT_ID_FALLBACK))
		return p;

	// Older nvinfo format: u16 type + u16 size + payload, 4-byte aligned.
	size_t off = 0;
	while (off + 4 <= nvinfo.size()) {
		const uint16_t attr_type =
			uint16_t(nvinfo[off]) | (uint16_t(nvinfo[off + 1]) << 8);
		const uint16_t attr_size =
			uint16_t(nvinfo[off + 2]) |
			(uint16_t(nvinfo[off + 3]) << 8);
		off += 4;
		if (off + attr_size > nvinfo.size())
			break;
		if (attr_type == EIATTR_REGCOUNT && attr_size >= 4) {
			const uint32_t old_regcount =
				uint32_t(nvinfo[off]) |
				(uint32_t(nvinfo[off + 1]) << 8) |
				(uint32_t(nvinfo[off + 2]) << 16) |
				(uint32_t(nvinfo[off + 3]) << 24);
			nvinfo[off + 0] = uint8_t(new_regcount & 0xffu);
			nvinfo[off + 1] =
				uint8_t((new_regcount >> 8) & 0xffu);
			nvinfo[off + 2] =
				uint8_t((new_regcount >> 16) & 0xffu);
			nvinfo[off + 3] =
				uint8_t((new_regcount >> 24) & 0xffu);
			return std::make_pair(old_regcount, new_regcount);
		}
		off += attr_size;
		off = (off + 3) & ~size_t(3);
	}
	return std::nullopt;
}

struct NvInfoGlobalRegcountIndex {
	size_t section_file_off = 0;
	size_t section_size = 0;
	// func_id (sh_info of .text.*) -> file offset of u32 regcount value
	std::unordered_map<uint32_t, size_t> func_to_regcount_file_off;
};

static std::optional<NvInfoGlobalRegcountIndex>
build_nvinfo_global_regcount_index(std::span<const uint8_t> elf,
				   const std::vector<Elf64_Shdr> &shdrs,
				   std::span<const uint8_t> shstrtab,
				   std::string_view section_name)
{
	NvInfoGlobalRegcountIndex idx;
	bool found = false;
	for (const auto &sh : shdrs) {
		if (sh.sh_offset == 0 || sh.sh_size < 8)
			continue;
		auto name = safe_cstr(shstrtab, sh.sh_name);
		if (name != section_name)
			continue;
		const size_t start = static_cast<size_t>(sh.sh_offset);
		const size_t size = static_cast<size_t>(sh.sh_size);
		if (start + size > elf.size())
			return std::nullopt;
		idx.section_file_off = start;
		idx.section_size = size;

		auto sec = elf.subspan(start, size);
		size_t off = 0;
		while (off + 4 <= sec.size()) {
			uint16_t type = 0, sz = 0;
			std::memcpy(&type, sec.data() + off, 2);
			std::memcpy(&sz, sec.data() + off + 2, 2);
			if (off + 4 + sz > sec.size())
				break;
			if (type == NVINFO_GLOBAL_REGCOUNT_TYPE && sz == 8) {
				uint32_t func_id = 0, reg = 0;
				std::memcpy(&func_id, sec.data() + off + 4, 4);
				std::memcpy(&reg, sec.data() + off + 8, 4);
				const size_t reg_file_off =
					start + off + 8; // u32 regcount
				idx.func_to_regcount_file_off.emplace(func_id,
								      reg_file_off);
			}
			off += 4 + sz;
			off = (off + 3) & ~size_t(3);
		}
		found = true;
		break;
	}
	if (!found)
		return std::nullopt;
	return idx;
}

struct SassInst {
	uint64_t w0 = 0;
	uint64_t w1 = 0;
};

	static SassInst inst_bra_sm120_pred_p0(int64_t delta_bytes,
					       std::optional<uint64_t> w1_override = std::nullopt)
	{
		// @P0 BRA +delta
		auto b = encode_bra_sm120_pred_p0(delta_bytes);
		assert(b.has_value());
		return SassInst { b->w0, w1_override.value_or(b->w1) };
	}

	static SassInst inst_bra_sm120_unpred(int64_t delta_bytes,
					      std::optional<uint64_t> w1_override = std::nullopt)
	{
		// BRA +delta
		auto b = encode_bra_sm120_unpred(delta_bytes);
		assert(b.has_value());
		return SassInst { b->w0, w1_override.value_or(b->w1) };
	}

static SassInst inst_ldcu64_desc_ur(uint8_t ur_base)
{
	// LDCU.64 UR4, c[0x0][0x358]
	//   w0 = 0x00006b00ff0477ac
	//   w1 = 0x000e620008000a00  (observed from nvcc 12.9 sm_120 output)
	SassInst i { 0x00006b00ff0477acULL, 0x000e620008000a00ULL };
	// UR base in w0 byte2 (observed: UR4 => 0x04).
	i.w0 = set_u64_byte(i.w0, 2, ur_base);
	return i;
}

static std::optional<SassInst>
find_ldcu64_desc_template_sm120(std::span<const uint8_t> text_section,
				std::optional<uint8_t> override_ur_base)
{
	// Best-effort extraction of a kernel-specific `LDCU.64` descriptor load
	// template from the prologue.
	//
	// Motivation:
	// - The constant-memory address that holds the "global memory descriptor"
	//   is *not stable* across toolchains/kernels (it is often not always 0x358).
	// - Many kernels contain multiple `LDCU.64` instructions loading unrelated
	//   constants into UR registers.
	//
	// Heuristic:
	// - First collect UR bases that are actually used by descriptor-form
	//   `LDG.*` / `STG.*` instructions in this function.
	// - Then pick an `LDCU.64` whose destination UR base appears in that set.
	//
	// This avoids the common failure mode where we load a non-descriptor value
	// into UR4/UR5 and all control header reads silently return zeros.
	const size_t ninst = text_section.size() / SASS_INST_BYTES;
	// Large vendor kernels (flashattention/cutlass) may not touch global memory
	// (descriptor-form LDG/STG) within the first few hundred instructions.
	// Scan deeper to reliably find both the descriptor usage and its LDCU load.
	const size_t scan = std::min<size_t>(ninst, 8192);

	auto rd_ur_base_from_w0_byte2 = [](uint64_t w0) -> uint8_t {
		return uint8_t((w0 >> 16) & 0xffu);
	};
	auto desc_ur_for_desc_op = [](uint16_t op16, uint64_t w0,
				      uint64_t w1) -> std::optional<uint8_t> {
		// Note: UR base encoding differs across opcodes on SM120.
		// - `STG.E desc[URx][R.64+imm]` encodes UR base in w1 byte0.
		// - `LDG.E.* desc[URx][R.64]` encodes UR base in w0 byte4.
		//
		// Real-world kernels (flashattention/cutlass) often use `LD.E`/`ST.E`
		// rather than `LDG`/`STG` mnemonics, and predication changes the low16
		// opcode bits. Normalize by masking to the low 12 bits.
		const uint16_t op12 = op16 & 0x0fffu;
		if (op12 == 0x986u) // STG (predicated variants share low12)
			return uint8_t(w1 & 0xffu);
		if (op12 == 0x981u || op12 == 0x980u) // LDG / LD.E
			return uint8_t((w0 >> 32) & 0xffu);
		if (op12 == 0x985u) // ST.E
			return uint8_t((w0 >> 32) & 0xffu);
		return std::nullopt;
	};

	std::array<uint32_t, 64> desc_ur_use_counts {};
	for (size_t i = 0; i < scan; i++) {
		const size_t off = i * SASS_INST_BYTES;
		uint64_t w0 = 0, w1 = 0;
		std::memcpy(&w0, text_section.data() + off, sizeof(w0));
		std::memcpy(&w1, text_section.data() + off + 8, sizeof(w1));
		const uint16_t op16 = uint16_t(w0 & 0xffffu);
		const uint16_t op12 = op16 & 0x0fffu;
		const bool is_desc_op = (op12 == 0x980u /*LD.E*/ ||
					 op12 == 0x981u /*LDG*/ ||
					 op12 == 0x985u /*ST.E*/ ||
					 op12 == 0x986u /*STG*/);
		if (!is_desc_op)
			continue;
		const auto ur_opt = desc_ur_for_desc_op(op16, w0, w1);
		if (!ur_opt)
			continue;
		const uint8_t ur = *ur_opt;
		if (ur < desc_ur_use_counts.size())
			desc_ur_use_counts[ur]++;
	}

	struct Candidate {
		size_t inst_index = 0;
		uint32_t use_count = 0;
		uint64_t w0 = 0;
		uint64_t w1 = 0;
		uint8_t dst_ur_base = 0;
	};
	std::optional<Candidate> best;
	for (size_t i = 0; i < scan; i++) {
		const size_t off = i * SASS_INST_BYTES;
		uint64_t w0 = 0, w1 = 0;
		std::memcpy(&w0, text_section.data() + off, sizeof(w0));
		std::memcpy(&w1, text_section.data() + off + 8, sizeof(w1));
		const uint16_t op16 = uint16_t(w0 & 0xffffu);
		if (op16 != 0x77ac) // LDCU.64 (observed)
			continue;
		const uint8_t dst = rd_ur_base_from_w0_byte2(w0);
		if ((dst & 1u) != 0 || dst > 62u)
			continue;
		const uint32_t cnt =
			(dst < desc_ur_use_counts.size()) ? desc_ur_use_counts[dst] : 0u;
		if (cnt == 0)
			continue;
		Candidate c {
			.inst_index = i,
			.use_count = cnt,
			.w0 = w0,
			.w1 = w1,
			.dst_ur_base = dst,
		};
		if (!best || c.use_count > best->use_count ||
		    (c.use_count == best->use_count && c.inst_index < best->inst_index)) {
			best = c;
		}
	}
	if (!best)
		return std::nullopt;

	uint64_t out_w0 = best->w0;
	if (override_ur_base)
		out_w0 = set_u64_byte(out_w0, 2, *override_ur_base);
	return SassInst { out_w0, best->w1 };
}

static std::optional<SassInst>
find_ldg_e64_desc_template_sm120(std::span<const uint8_t> text_section,
				 uint8_t want_ur_base)
{
	// Best-effort extraction of an unpredicated `LDG.E.64` template:
	//   LDG.E.64 R*, desc[UR*][R*.64]
	//
	// Many kernels do not contain a plain `LDG.E` (u32) instruction, but almost
	// all contain some form of `LDG.E.64` for global pointer reads. Using an
	// in-kernel control word (w1) here improves stability compared to a single
	// hardcoded scheduling word.
	const size_t ninst = text_section.size() / SASS_INST_BYTES;
	const size_t scan = std::min<size_t>(ninst, 8192);
	for (size_t i = 0; i < scan; i++) {
		const size_t off = i * SASS_INST_BYTES;
		uint64_t w0 = 0, w1 = 0;
		std::memcpy(&w0, text_section.data() + off, sizeof(w0));
		std::memcpy(&w1, text_section.data() + off + 8, sizeof(w1));
		const uint16_t op16 = uint16_t(w0 & 0xffffu);
		if (op16 != 0x7981) // LDG (unpredicated variants observed)
			continue;
		// Heuristic: `LDG.E.64` observed to use a low-16 control suffix of 0x1b00
		// in w1 (see `inst_ldg_e_u64_desc_ur`).
		if ((w1 & 0xffffu) != 0x1b00u)
			continue;
		// Require imm=0 (we materialize absolute addresses in GPRs and keep the
		// LDG itself in the imm=0 form to avoid misalignment/encoding ambiguity).
		if (((w0 >> 40) & 0x00ffffffull) != 0)
			continue;
		// Patch UR base in w0 byte4 to the requested UR pair.
		w0 = set_u64_byte(w0, 4, want_ur_base);
		return SassInst { w0, w1 };
	}
	return std::nullopt;
}

static std::optional<SassInst>
find_ldg_e32_desc_template_sm120(std::span<const uint8_t> text_section,
				 std::optional<uint8_t> override_ur_base)
{
	// Best-effort extraction of an unpredicated `LDG.E` (u32) template:
	//   LDG.E R*, desc[UR*][R*.64]
	//
	// We prefer using an in-kernel control word (w1) for stability on
	// cubin-only vendor kernels (flashattention/cutlass), rather than a single
	// hardcoded scheduling word.
	const size_t ninst = text_section.size() / SASS_INST_BYTES;
	const size_t scan = std::min<size_t>(ninst, 8192);
	for (size_t i = 0; i < scan; i++) {
		const size_t off = i * SASS_INST_BYTES;
		uint64_t w0 = 0, w1 = 0;
		std::memcpy(&w0, text_section.data() + off, sizeof(w0));
		std::memcpy(&w1, text_section.data() + off + 8, sizeof(w1));
		const uint16_t op16 = uint16_t(w0 & 0xffffu);
		if (op16 != 0x7981) // LDG (unpredicated variants observed)
			continue;
		// Heuristic: `LDG.E` (u32) observed to use a low-16 control suffix of
		// 0x1900 in w1 (see flashattention/cutlass cuobjdump output).
		if ((w1 & 0xffffu) != 0x1900u)
			continue;
		// Require imm=0 (we materialize absolute addresses in GPRs).
		if (((w0 >> 40) & 0x00ffffffull) != 0)
			continue;
		if (override_ur_base)
			w0 = set_u64_byte(w0, 4, *override_ur_base);
		return SassInst { w0, w1 };
	}
	return std::nullopt;
}

static SassInst inst_hfma2_imm_u32(uint8_t rd, uint32_t imm)
{
	// HFMA2 R5, -RZ, RZ, <imm32>, <const>
	// ptxas uses HFMA2 as a 32-bit immediate materialization on sm_120:
	//   w0 = 0x12345678ff057431  (upper 32 bits carry imm32; rd in w0 byte2)
	//   w1 = 0x000fca00000001ff
	SassInst i { 0x00000000ff007431ULL, 0x000fca00000001ffULL };
	i.w0 |= (uint64_t(imm) << 32);
	i.w0 = set_u64_byte(i.w0, 2, rd);
	return i;
}

static SassInst inst_mov_imm_u32(uint8_t rd, uint32_t imm)
{
	// NOTE: For SM120, ptxas frequently materializes 32-bit immediates via
	// HFMA2 (see inst_hfma2_imm_u32) rather than a dedicated MOV-imm encoding.
	//
	// We keep this helper as a semantic "materialize imm32" and currently map
	// it to the HFMA2 pattern for stability across cubin-only vendor kernels
	// (flashattention/cutlass) where some opcode/control-word combinations can
	// trip `cudaErrorIllegalInstruction` when used at EXIT detour points.
	return inst_hfma2_imm_u32(rd, imm);
}

static SassInst inst_s2r(uint8_t rd, uint64_t w1_src)
{
	// S2R R0, <SR_*>
	//   w0 = 0x0000000000007919
	//   w1 depends on SR.
	SassInst i { 0x0000000000007919ULL, w1_src };
	i.w0 = set_u64_byte(i.w0, 2, rd);
	return i;
}

// Observed `S2R` w1 encodings for SM120 (verified via ptxas 12.9, sm_120).
//
// IMPORTANT: The full 64-bit control word matters. Using only the low selector
// bits (e.g., `...2100` for SR_TID.X) can yield incorrect values or trigger
// subtle dependency hazards in cubin-only workloads.
//
// Probe kernels (reproducible):
//   /usr/local/cuda-12.9/bin/ptxas -arch=sm_120 -O0 /tmp/bpftime_probe_cta.ptx -o /tmp/bpftime_probe_cta.cubin
//   /usr/local/cuda-12.9/bin/ptxas -arch=sm_120 -O0 /tmp/bpftime_probe_sr.ptx  -o /tmp/bpftime_probe_sr.cubin
// These w1 control words are taken from `cuobjdump --dump-sass` on sm_120
// probe cubins (nvcc 12.9).
constexpr uint64_t SM120_SR_TID_X = 0x000e640000002100ULL;
constexpr uint64_t SM120_SR_CTAID_X = 0x000e620000002500ULL;
constexpr uint64_t SM120_SR_LANEID = 0x000e620000000000ULL;
constexpr uint64_t SM120_SR_VIRTID = 0x000e620000000300ULL; // unused (kept for future)
constexpr uint64_t SM120_SR_VIRTUALSMID = 0x000ea20000004300ULL;

static SassInst inst_s2ur(uint8_t ur, uint64_t w1_src)
{
	// S2UR UR4, <SR_*>
	//   w0 = 0x00000000000479c3
	//   w1 depends on SR.
	SassInst i { 0x00000000000079c3ULL, w1_src };
	i.w0 = set_u64_byte(i.w0, 2, ur);
	return i;
}

static SassInst inst_ushf_l_u32(uint8_t urd, uint8_t ura, uint8_t imm8)
{
	// USHF.L.U32 UR4, UR4, 0x5, URZ
	//   w0 = 0x0000000504047899
	//   w1 = 0x002fe200080006ff
	// (Observed from nvcc sm_120 output.)
	SassInst i { 0x0000000000007899ULL, 0x002fe200080006ffULL };
	i.w0 = set_u64_byte(i.w0, 2, urd);
	i.w0 = set_u64_byte(i.w0, 3, ura);
	i.w0 = set_u64_byte(i.w0, 4, imm8);
	return i;
}

static SassInst inst_lop3_or_ur(uint8_t rd, uint8_t ra, uint8_t urb)
{
	// LOP3.LUT R5, R5, UR6, RZ, 0xfc, !PT
	//   w0 = 0x0000000605057c12  (urb in w0 byte4)
	//   w1 = 0x001fca000f8efcff
	// (Observed from nvcc sm_120 output.)
	SassInst i { 0x0000000000007c12ULL, 0x001fca000f8efcffULL };
	i.w0 = set_u64_byte(i.w0, 2, rd);
	i.w0 = set_u64_byte(i.w0, 3, ra);
	i.w0 = set_u64_byte(i.w0, 4, urb);
	return i;
}

static SassInst inst_lop3_or_rr(uint8_t rd, uint8_t ra, uint8_t rb)
{
	// LOP3.LUT R156, R165, R156, RZ, 0xfc, !PT
	//   w0 = 0x0000009ca59c7212  (rd=0x9c, ra=0xa5, rb=0x9c)
	//   w1 = 0x000fc400078efcff  (rc=RZ)
	// (Observed from sm_120 cuobjdump output: cutlass kernels.)
	SassInst i { 0x0000000000007212ULL, 0x000fc400078efcffULL };
	i.w0 = set_u64_byte(i.w0, 2, rd);
	i.w0 = set_u64_byte(i.w0, 3, rb);
	i.w0 = set_u64_byte(i.w0, 4, ra);
	return i;
}

static SassInst inst_lop3_lane_ne0_to_p0(uint8_t r_tid)
{
	// LOP3.LUT P0, RZ, R5, 0x1f, RZ, 0xc0, !PT
	//   w0 = 0x0000001f05ff7812
	//   w1 = 0x002fda000780c0ff
	// (Observed from nvcc sm_120 output.)
	SassInst i { 0x0000001f00ff7812ULL, 0x002fda000780c0ffULL };
	// tid register is encoded in w0 byte3 (R5 => 0x05).
	i.w0 = set_u64_byte(i.w0, 3, r_tid);
	return i;
}

static SassInst inst_isetp_ne_u32_and_p0(uint8_t ra)
{
	// ISETP.NE.AND P0, PT, R2, RZ, PT
	//   w0 = 0x000000ff0200720c
	//   w1 = 0x003fde0003f05070  (ptxas 12.9 sm_120 output)
	// NOTE: the full w1 control word matters for correctness on cubin-only kernels.
	SassInst i { 0x000000ff0000720cULL, 0x003fde0003f05070ULL };
	// On sm_120, the compared GPR is encoded in w0 byte3.
	i.w0 = set_u64_byte(i.w0, 3, ra);
	return i;
}

static SassInst inst_isetp_ne_u32_and_p0_imm(uint8_t ra, uint32_t imm32)
{
	// ISETP.NE.AND P0, PT, R2, 0x1, PT
	//   w0 = 0x000000010200780c  (imm32 in w0[63:32], ra in w0 byte3)
	//   w1 = 0x003fde0003f05070  (ptxas 12.9 sm_120 output)
	SassInst i { 0x000000000000780cULL, 0x003fde0003f05070ULL };
	i.w0 |= (uint64_t(imm32) << 32);
	i.w0 = set_u64_byte(i.w0, 3, ra);
	return i;
}

static SassInst inst_isetp_ne_u32_and_p0_rr(uint8_t ra, uint8_t rb)
{
	// ISETP.NE.AND P0, PT, R0, R5, PT
	//   w0 = 0x000000050000720c  (ra in w0 byte3, rb in w0 byte4)
	//   w1 = 0x003fde0003f05070  (ptxas 12.9 sm_120 output)
	SassInst i { 0x000000000000720cULL, 0x003fde0003f05070ULL };
	i.w0 = set_u64_byte(i.w0, 3, ra);
	i.w0 = set_u64_byte(i.w0, 4, rb);
	return i;
}

static SassInst inst_isetp_gt_u32_and_p0_imm(uint8_t ra, uint32_t imm32)
{
	// ISETP.GT.U32.AND P0, PT, R0, <imm32>, PT
	//   w0 = 0x0000001f0700780c  (imm32 in w0[63:32], ra in w0 byte3)
	//   w1 = 0x002fda0003f04070
	// (Observed from nvcc sm_120 output: /tmp/bpftime_probe_laneid.cubin)
	SassInst i { 0x000000000000780cULL, 0x002fda0003f04070ULL };
	i.w0 |= (uint64_t(imm32) << 32);
	i.w0 = set_u64_byte(i.w0, 3, ra);
	return i;
}

static SassInst inst_isetp_ge_u32_and_p0(uint8_t ra, uint8_t rb)
{
	// ISETP.GE.U32.AND P0, PT, R0, R5, PT
	//   w0 = 0x000000050000720c  (ra in w0 byte3, rb in w0 byte4)
	//   w1 = 0x002fda0003f06070
	// (Observed from nvcc 12.9 sm_120 output.)
	SassInst i { 0x000000000000720cULL, 0x002fda0003f06070ULL };
	i.w0 = set_u64_byte(i.w0, 3, ra);
	i.w0 = set_u64_byte(i.w0, 4, rb);
	return i;
}

static SassInst inst_isetp_ge_u32_and_p0_imm(uint8_t ra, uint32_t imm32)
{
	// ISETP.GE.U32.AND P0, PT, R2, 0x20, PT
	//   w0 = 0x000000200200780c  (imm32 in w0[63:32], ra in w0 byte3)
	//   w1 = 0x002fda0003f06070
	// (Observed from nvcc 12.9 sm_120 output.)
	SassInst i { 0x000000000000780cULL, 0x002fda0003f06070ULL };
	i.w0 |= (uint64_t(imm32) << 32);
	i.w0 = set_u64_byte(i.w0, 3, ra);
	return i;
}

static SassInst inst_p2r_pr_mask(uint8_t rd, uint8_t ra, uint8_t mask)
{
	// P2R R3, PR, RZ, 0xf
	//   w0 = 0x0000000fff037803
	//   w1 = 0x000fe20000000000
	// (Observed from sm_120 cuobjdump output: cutlass kernels.)
	SassInst i { 0x0000000f00007803ULL, 0x000fe20000000000ULL };
	i.w0 = set_u64_byte(i.w0, 2, rd);
	i.w0 = set_u64_byte(i.w0, 3, ra);
	i.w0 = set_u64_byte(i.w0, 4, mask);
	return i;
}

static SassInst inst_r2p_pr_mask(uint8_t ra, uint8_t mask)
{
	// R2P PR, R3, 0xf
	//   w0 = 0x0000000f03007804
	//   w1 = 0x000fe20000000000
	// (Observed from sm_120 cuobjdump output: cutlass kernels.)
	SassInst i { 0x0000000f00007804ULL, 0x000fe20000000000ULL };
	i.w0 = set_u64_byte(i.w0, 3, ra);
	i.w0 = set_u64_byte(i.w0, 4, mask);
	return i;
}

static SassInst inst_atomg_add_u32(uint8_t ur_base, uint8_t rd, uint8_t addr,
				   uint8_t rs)
{
	// ATOMG.E.ADD.STRONG.GPU PT, R4, desc[UR4][R2.64], R0
	//   w0 = 0x80000000020479a8
	//   w1 = 0x00321e00081ef104  (UR base in w1 byte0)
	SassInst i { 0x80000000020479a8ULL, 0x00321e00081ef104ULL };
	i.w0 = set_u64_byte(i.w0, 2, rd);
	i.w0 = set_u64_byte(i.w0, 3, addr);
	i.w0 = set_u64_byte(i.w0, 4, rs);
	i.w1 = set_u64_byte(i.w1, 0, ur_base);
	return i;
}

static SassInst inst_imad_wide_u32(uint8_t rd_pair, uint8_t ra, uint8_t imm8,
				   uint8_t rb_pair)
{
	// IMAD.WIDE.U32 R2, R5, 0x4, R2
	//   w0 = 0x0000000405027825
	//   w1 = 0x001fca00078e0002
	// (Observed from nvcc sm_120 output.)
	SassInst i { 0x0000000405027825ULL, 0x001fca00078e0002ULL };
	i.w0 = set_u64_byte(i.w0, 2, rd_pair);
	i.w0 = set_u64_byte(i.w0, 3, ra);
	i.w0 = set_u64_byte(i.w0, 4, imm8);
	i.w1 = set_u64_byte(i.w1, 0, rb_pair);
	return i;
}

static SassInst inst_imad_shl_u32(uint8_t rd, uint8_t ra, uint8_t imm8,
				  uint8_t rb)
{
	// IMAD.SHL.U32 R5, R7, 0x20, RZ
	//   w0 = 0x0000002007057824
	//   w1 = 0x000fe200078e00ff  (rb in w1 byte0; RZ => 0xff)
	// (Observed from sm_120 cuobjdump output: cutlass kernels.)
	SassInst i { 0x0000002000007824ULL, 0x000fe200078e00ffULL };
	i.w0 = set_u64_byte(i.w0, 2, rd);
	i.w0 = set_u64_byte(i.w0, 3, ra);
	i.w0 = set_u64_byte(i.w0, 4, imm8);
	i.w1 = set_u64_byte(i.w1, 0, rb);
	return i;
}

static SassInst inst_imad_shl_u32_imm32(uint8_t rd, uint8_t ra, uint32_t imm32,
					uint8_t rb)
{
	// IMAD.SHL.U32 R0, R0, 0x400, RZ
	//   w0 = 0x0000040000007824  (imm32 in w0[63:32])
	//   w1 = 0x000fc600078e00ff  (rb in w1 byte0; RZ => 0xff)
	// (Observed from nvcc/ptxas 12.9 sm_120 output.)
	SassInst i { 0x0000000000007824ULL, 0x000fc600078e00ffULL };
	i.w0 |= (uint64_t(imm32) << 32);
	i.w0 = set_u64_byte(i.w0, 2, rd);
	i.w0 = set_u64_byte(i.w0, 3, ra);
	i.w1 = set_u64_byte(i.w1, 0, rb);
	return i;
}

static SassInst inst_lop3_and_imm_u32(uint8_t rd, uint8_t ra, uint32_t imm32)
{
	// LOP3.LUT R9, R9, 0x1f, RZ, 0xc0
	//   w0 = 0x0000001f09097812  (imm32 in w0[63:32])
	//   w1 = 0x001fca00078ec0ff  (rb=RZ)
	// (Observed from ptxas sm_120 output: `and.b32` lowering.)
	SassInst i { 0x0000000000007812ULL, 0x001fca00078ec0ffULL };
	i.w0 |= (uint64_t(imm32) << 32);
	i.w0 = set_u64_byte(i.w0, 2, rd);
	i.w0 = set_u64_byte(i.w0, 3, ra);
	return i;
}

static SassInst inst_shf_r_u32_hi_rz(uint8_t rd, uint8_t rb, uint8_t imm8)
{
	// SHF.R.U32.HI R7, RZ, 0x5, R9
	//   w0 = 0x00000005ff077819  (imm8 in w0 byte4; ra fixed to RZ in w0 byte3)
	//   w1 = 0x000fe20000011600  (rb in w1 byte0)
	// (Observed from nvcc sm_120 output.)
	SassInst i { 0x00000000ff007819ULL, 0x000fe20000011600ULL };
	i.w0 = set_u64_byte(i.w0, 2, rd);
	i.w0 = set_u64_byte(i.w0, 4, imm8);
	i.w1 = set_u64_byte(i.w1, 0, rb);
	return i;
}

static SassInst inst_shf_l_u32_rz(uint8_t rd, uint8_t ra, uint8_t imm8)
{
	// SHF.L.U32 R0, R0, 0xa, RZ
	//   w0 = 0x0000000a00007819  (imm8 in w0 byte4; ra in w0 byte3)
	//   w1 = 0x004fca00000006ff  (rb fixed to RZ => 0xff)
	// (Observed from nvcc/ptxas 12.9 sm_120 output: /tmp/bpftime_probe_shfl.cubin)
	SassInst i { 0x0000000000007819ULL, 0x004fca00000006ffULL };
	i.w0 = set_u64_byte(i.w0, 2, rd);
	i.w0 = set_u64_byte(i.w0, 3, ra);
	i.w0 = set_u64_byte(i.w0, 4, imm8);
	return i;
}

static SassInst inst_nop()
{
	// NOP
	//   w0 = 0x0000000000007918
	//   w1 = 0x000fc00000000000
	// (Observed from nvdisasm sm_120 output.)
	return SassInst { 0x0000000000007918ULL, 0x000fc00000000000ULL };
}

static SassInst inst_iadd64_rr(uint8_t rd_pair, uint8_t ra_pair,
			       uint8_t rb_pair)
{
	// IADD.64 R4, R2, R4
	//   w0 = 0x0000000402047235
	//   w1 = 0x003fde00078e0200
	SassInst i { 0x0000000402047235ULL, 0x003fde00078e0200ULL };
	i.w0 = set_u64_byte(i.w0, 2, rd_pair);
	i.w0 = set_u64_byte(i.w0, 3, ra_pair);
	i.w0 = set_u64_byte(i.w0, 4, rb_pair);
	return i;
}

static SassInst inst_iadd64_ri(uint8_t rd_pair, uint8_t ra_pair, uint8_t imm8)
{
	// IADD.64 R2, R2, 0x4
	//   w0 = 0x0000000402027835
	//   w1 = 0x003fde00078e0200
	SassInst i { 0x0000000402027835ULL, 0x003fde00078e0200ULL };
	i.w0 = set_u64_byte(i.w0, 2, rd_pair);
	i.w0 = set_u64_byte(i.w0, 3, ra_pair);
	i.w0 = set_u64_byte(i.w0, 4, imm8);
	return i;
}

static SassInst inst_stg_e_u32(uint8_t ur_base, uint8_t addr_pair, uint8_t rs)
{
	// STG.E desc[UR4][R2.64], R4
	//   w0 = 0x0000000402007986
	//   w1 = 0x001fe2000c101904  (UR base in w1 byte0)
	SassInst i { 0x0000000002007986ULL, 0x001fe2000c101900ULL };
	i.w0 = set_u64_byte(i.w0, 3, addr_pair);
	// For the no-imm form, ptxas encodes the source GPR in w0 byte4.
	// Example:
	//   STG.E desc[UR4][R2.64], R5
	//     w0 = 0x0000000502007986  (byte4=0x05, byte5=0x00)
	//     w1 = 0x001fe2000c101904
	//
	// NOTE: w0 byte5 is used by the "+imm" encoding (inst_stg_e_u32_off). Keep
	// it at 0 here to avoid mis-decoding this form as `[R+imm], R0`.
	i.w0 = set_u64_byte(i.w0, 4, rs);
	i.w1 = set_u64_byte(i.w1, 0, ur_base);
	return i;
}

static SassInst inst_stg_e_u32_not_p0(uint8_t ur_base, uint8_t addr_pair,
				      uint8_t rs)
{
	// @!P0 STG.E desc[UR4][R2.64], R5
	//   w0 = 0x0000000502008986
	//   w1 = 0x0043e2000c101904  (UR base in w1 byte0)
	// (Observed from nvcc sm_120 output.)
	SassInst i { 0x0000000002008986ULL, 0x0043e2000c101900ULL };
	i.w0 = set_u64_byte(i.w0, 3, addr_pair);
	// Source GPR is encoded in w0 byte4 (same as the unpredicated form).
	i.w0 = set_u64_byte(i.w0, 4, rs);
	i.w1 = set_u64_byte(i.w1, 0, ur_base);
	return i;
}

static SassInst inst_stg_e_u32_off(uint8_t ur_base, uint8_t addr_pair,
				   uint8_t rs, uint8_t imm8)
{
	// STG.E desc[UR4][R2.64+0x4], R7
	//   w0 = 0x0000040702007986
	//   w1 = 0x001fe2000c101904
	// (Observed from ptxas 12.9 sm_120 output: /tmp/bpftime_stg_test_off.cubin)
	SassInst i { 0x0000000002007986ULL, 0x001fe2000c101900ULL };
	i.w0 = set_u64_byte(i.w0, 3, addr_pair);
	i.w0 = set_u64_byte(i.w0, 4, rs);
	i.w0 = set_u64_byte(i.w0, 5, imm8);
	i.w1 = set_u64_byte(i.w1, 0, ur_base);
	return i;
}

static SassInst inst_stg_e_u8(uint8_t ur_base, uint8_t addr_pair, uint8_t rs)
{
	// STG.E.U8 desc[UR4][R2.64], R0
	//   w0 = 0x0000000002007986
	//   w1 = 0x001fe2000c101104
	SassInst i { 0x0000000002007986ULL, 0x001fe2000c101100ULL };
	i.w0 = set_u64_byte(i.w0, 3, addr_pair);
	i.w0 = set_u64_byte(i.w0, 4, rs);
	i.w1 = set_u64_byte(i.w1, 0, ur_base);
	return i;
}

static SassInst inst_ldg_e_u32_addr_ur(uint8_t rd, uint8_t addr_pair,
				       uint8_t ur_off, uint8_t imm8)
{
	// LD.E R13, [R8.64+UR62+0x8]
	//
	// IMPORTANT: On SM120 this addressing form encodes a 16-bit immediate
	// (imm16 = w0 byte6:byte5). Empirically, the base UR register is encoded in
	// w0 byte4 (if we accidentally place other fields there, nvdisasm can decode
	// a different UR base and kernels can fault).
	//
	// NOTE: Prefer the generic `LD.E` form (opcode 0x7980) for control/header
	// loads; `LDG` variants can be more restrictive on some kernels.
	SassInst i { 0x0000000000007980ULL, 0x000fe2000c1e0900ULL };
	i.w0 = set_u64_byte(i.w0, 2, rd);
	i.w0 = set_u64_byte(i.w0, 3, addr_pair);
	i.w0 = set_u64_byte(i.w0, 4, ur_off);
	// imm16 = (byte6 << 8) | byte5
	i.w0 = set_u64_byte(i.w0, 5, imm8);
	i.w0 = set_u64_byte(i.w0, 6, 0);
	return i;
}

static SassInst inst_ldg_e_u32_desc_ur(uint8_t rd, uint8_t addr_pair,
				       uint8_t ur_base)
{
	// LDG.E R3, desc[UR4][R2.64]
	//
	// Verified nvcc 12.9 sm_120 encoding (probe):
	//   LDG.E R3, desc[UR4][R2.64]
	//     w0 = 0x0000000402037981  (rd=3, addr_pair=2, ur_base=4)
	//     w1 = 0x002eac000c1e1900
	//
	// Field mapping (empirical):
	// - w0 byte2: rd
	// - w0 byte3: addr_pair (low reg of the .64 pair)
	// - w0 byte4: ur_base
	// NOTE: For stability, we only rely on the imm=0 form here and perform
	// small address adjustments via `IADD.64` in the caller. (We saw nvdisasm
	// interpret nonzero upper bytes as a different immediate mode.)
	SassInst i { 0x0000000000007981ULL, 0x002eac000c1e1900ULL };
	i.w0 = set_u64_byte(i.w0, 2, rd);
	i.w0 = set_u64_byte(i.w0, 3, addr_pair);
	i.w0 = set_u64_byte(i.w0, 4, ur_base);
	return i;
}

static SassInst inst_ldg_e_u64_desc_ur(uint8_t rd_pair, uint8_t addr_pair,
				       uint8_t ur_base)
{
	// LDG.E.64 R2, desc[UR4][R2.64]
	//
	// Verified nvcc 12.9 sm_120 encoding (probe):
	//   LDG.E.64 R2, desc[UR4][R2.64]
	//     w0 = 0x0000000402027981  (rd_pair=2, addr_pair=2, ur_base=4)
	//     w1 = 0x002eac000c1e1b00
	SassInst i { 0x0000000000007981ULL, 0x002eac000c1e1b00ULL };
	i.w0 = set_u64_byte(i.w0, 2, rd_pair);
	i.w0 = set_u64_byte(i.w0, 3, addr_pair);
	i.w0 = set_u64_byte(i.w0, 4, ur_base);
	return i;
}

static SassInst inst_stg_e_u8_addr_ur(uint8_t addr_pair, uint8_t ur_off,
				      uint8_t rs, uint8_t imm8)
{
	// STG.E.U8 [R10.64+UR12+0x10], R3
	//   w0 = 0x000010030a007986
	//   w1 uses the same control word prefix as SM120 cubin-only kernels:
	//     STG.E.U8 [R2.64+UR20], R0
	//       w1 = 0x000fe2000c100114  (UR index in w1 byte0)
	SassInst i { 0x000010030a007986ULL, 0x000fe2000c100100ULL };
	i.w0 = set_u64_byte(i.w0, 3, addr_pair);
	i.w0 = set_u64_byte(i.w0, 4, rs);
	i.w0 = set_u64_byte(i.w0, 5, imm8);
	i.w1 = set_u64_byte(i.w1, 0, ur_off);
	return i;
}

static SassInst inst_stg_e_u8_addr_ur_not_p0(uint8_t addr_pair, uint8_t ur_off,
					     uint8_t rs, uint8_t imm8)
{
	// @!P0 STG.E.U8 [R10.64+UR12+0x10], R3
	//
	// Observed pattern (sm_120 cuobjdump):
	// - unpredicated: 0x7986
	// - @!P0:         0x8986
	// Control word must preserve predicate dependencies:
	//   @!P0 STG.E.U8 [R168.64+UR20], R21
	//     w1 = 0x000fe2000c100114  (UR index in w1 byte0)
	SassInst i { 0x000010030a008986ULL, 0x000fe2000c100100ULL };
	i.w0 = set_u64_byte(i.w0, 3, addr_pair);
	i.w0 = set_u64_byte(i.w0, 4, rs);
	i.w0 = set_u64_byte(i.w0, 5, imm8);
	i.w1 = set_u64_byte(i.w1, 0, ur_off);
	return i;
}

static SassInst inst_stg_e_u32_addr_ur(uint8_t addr_pair, uint8_t ur_off,
				       uint8_t rs, uint8_t imm8)
{
	// STG.E [R10.64+UR12+0x10], R3
	//
	// Same addressing form as `inst_stg_e_u8_addr_ur`, but stores a 32-bit word.
	// The opcode differs only in the type bits in w1 (observed as +0x800 vs U8).
	SassInst i { 0x000010030a007986ULL, 0x000fe2000c10090cULL };
	i.w0 = set_u64_byte(i.w0, 3, addr_pair);
	i.w0 = set_u64_byte(i.w0, 4, rs);
	i.w0 = set_u64_byte(i.w0, 5, imm8);
	i.w1 = set_u64_byte(i.w1, 0, ur_off);
	return i;
}

static SassInst inst_stg_e_u32_addr_ur_not_p0(uint8_t addr_pair, uint8_t ur_off,
					      uint8_t rs, uint8_t imm8)
{
	// @!P0 STG.E [R10.64+UR12+0x10], R3
	//
	// Predication on SM120 is encoded in the low 16-bit opcode field.
	// Observed pattern:
	// - unpredicated: 0x7986
	// - @!P0:         0x8986
	// Keep the control word consistent with the unpredicated `STG.E` form; only
	// the opcode field encodes predication.
	SassInst i { 0x000010030a008986ULL, 0x000fe2000c10090cULL };
	i.w0 = set_u64_byte(i.w0, 3, addr_pair);
	i.w0 = set_u64_byte(i.w0, 4, rs);
	i.w0 = set_u64_byte(i.w0, 5, imm8);
	i.w1 = set_u64_byte(i.w1, 0, ur_off);
	return i;
}

static SassInst inst_umov_ur_imm(uint8_t ur, uint32_t imm)
{
	// UMOV UR4, 0x2
	//   w0 = 0x0000000200047882
	//   w1 = 0x000fe20000000000
	SassInst i { 0x0000000000047882ULL, 0x000fe20000000000ULL };
	i.w0 |= (uint64_t(imm) << 32);
	i.w0 = set_u64_byte(i.w0, 2, ur);
	return i;
}

static SassInst inst_umov_ur_ur(uint8_t ur_dst, uint8_t ur_src)
{
	// UMOV UR13, UR7
	//   w0 = 0x00000007000d7c82  (dst in w0 byte2; src in w0 byte4)
	//   w1 = 0x000fe20008000000
	SassInst i { 0x0000000000007c82ULL, 0x000fe20008000000ULL };
	i.w0 = set_u64_byte(i.w0, 2, ur_dst);
	i.w0 = set_u64_byte(i.w0, 4, ur_src);
	return i;
}

static SassInst inst_umov64_ur_imm(uint8_t ur_base, uint64_t imm)
{
	// UMOV.64 UR4, 0x1122334455667788
	//   w0 = 0x4455667788047482
	//   w1 = 0x000fe20008112233
	//
	// Observed encoding for SM120:
	// - w0[63:32] = imm[39:8]
	// - w0 byte3  = imm[7:0]
	// - w1[23:0]  = imm[63:40]
	// - w1[31:24] = 0x08
	SassInst i { 0x0000000000007482ULL, 0x000fe20008000000ULL };
	i.w0 = set_u64_byte(i.w0, 2, ur_base);
	i.w0 = set_u64_byte(i.w0, 3, uint8_t(imm & 0xffu));
	i.w0 |= (uint64_t((imm >> 8) & 0xffffffffull) << 32);
	i.w1 |= uint64_t((imm >> 40) & 0x00ffffffull);
	return i;
}

	struct Sm120RegLayout {
		uint8_t r_ptr = 0; // pair
		uint8_t r_tmp = 0;
		uint8_t r_pr = 0; // predicate save (P0) via P2R/R2P
		uint8_t r_smid = 0;
		uint8_t r_ctrl = 0; // pair (control/header u64 loads)
		uint8_t r_off = 0; // pair
		uint8_t r_cta = 0;
		uint8_t r_tid = 0;
		uint8_t r_lane = 0;
		uint8_t r_warp = 0;
	uint8_t r_idx = 0;
	uint32_t old_regcount = 0;
	uint32_t new_regcount = 0;
};

static std::optional<Sm120RegLayout>
	compute_sm120_layout(uint32_t old_regcount,
			     Sm120SamplingConfig::Mode mode)
	{
	// Keep the layout compact by allocating scratch registers directly above
	// the kernel's original regcount.
	uint32_t base = (old_regcount + 1u) & ~1u; // align to even for .64 pairs
		// Layout (see design/gpu_sass_observability.md). We always reserve one
		// dedicated predicate-save GPR (`r_pr`) because the controlled stubs
		// must preserve P0 reliably (saving into `r_tmp` is unsafe since `r_tmp`
		// is used as a general scratch register).
		//
		// - SmidBitmap:
		//   - r_ptr:  base/base+1 (address scratch)
		//   - r_tmp:  base+2
		//   - r_pr:   base+3
		//   - r_smid: base+4
		// - CtaSmid:
		//   - r_ptr:  base/base+1 (address scratch)
		//   - r_cta:  base+2
		//   - r_pr:   base+3
		//   - r_smid: base+4
		// - WarpMap:
		//   - r_ptr:  base/base+1 (address scratch)
		//   - r_cta:  base+2
		//   - r_warp: base+3
		//   - r_smid: base+4
		//   - r_idx:  base+5
		//   - r_tmp:  base+6
		//   - r_pr:   base+7
		// - ThreadMap:
		//   - r_ptr:  base/base+1 (address scratch)
		//   - r_cta:  base+2
		//   - r_tid:  base+3
		//   - r_lane: base+4
		//   - r_smid: base+5
		//   - r_idx:  base+6
		//   - r_tmp:  base+7
		//   - r_pr:   base+8
	uint32_t max_reg = base + 7u; // default includes r_ctrl pair (aligned)
	if (mode == Sm120SamplingConfig::Mode::WarpMap)
		max_reg = base + 9u;
	else if (mode == Sm120SamplingConfig::Mode::ThreadMap ||
		 mode == Sm120SamplingConfig::Mode::PcMarker)
		max_reg = base + 11u;
	// Leave slack above the highest scratch register. We observed
	// `cudaErrorIllegalInstruction` when wide ops write to the boundary
	// register pair on some SM120 cubin-only kernels; increasing regcount
	// headroom makes our scratch registers “interior” again.
	auto align_up = [](uint32_t v, uint32_t a) {
		return (v + a - 1u) & ~(a - 1u);
	};
	uint32_t new_regcount = 0;
	for (uint32_t slack : { 8u, 4u, 0u }) {
		new_regcount = align_up(max_reg + 1u + slack, 8u);
		if (new_regcount <= 255u)
			break;
	}
	if (new_regcount > 255u)
		return std::nullopt;
		Sm120RegLayout l;
		if (mode == Sm120SamplingConfig::Mode::SmidBitmap) {
			l.r_ptr = uint8_t(base);
			l.r_tmp = uint8_t(base + 2u);
			l.r_pr = uint8_t(base + 3u);
			l.r_smid = uint8_t(base + 4u);
			// Keep `.64` pairs even-aligned: `LDG.E.64` requires an even rd_pair.
			l.r_ctrl = uint8_t(base + 6u);
			l.r_off = l.r_ptr;
		} else if (mode == Sm120SamplingConfig::Mode::CtaSmid) {
			l.r_ptr = uint8_t(base);
			l.r_cta = uint8_t(base + 2u);
			l.r_pr = uint8_t(base + 3u);
			l.r_smid = uint8_t(base + 4u);
			l.r_ctrl = uint8_t(base + 6u);
			l.r_tmp = l.r_cta;
			l.r_off = l.r_ptr;
		} else if (mode == Sm120SamplingConfig::Mode::WarpMap) {
			l.r_ptr = uint8_t(base);
			l.r_cta = uint8_t(base + 2u);
			l.r_warp = uint8_t(base + 3u);
			l.r_smid = uint8_t(base + 4u);
			l.r_idx = uint8_t(base + 5u);
			l.r_tmp = uint8_t(base + 6u);
			l.r_pr = uint8_t(base + 7u);
			l.r_ctrl = uint8_t(base + 8u);
			l.r_off = l.r_ptr;
		} else if (mode == Sm120SamplingConfig::Mode::ThreadMap) {
			l.r_ptr = uint8_t(base);
			l.r_cta = uint8_t(base + 2u);
			l.r_tid = uint8_t(base + 3u);
			l.r_lane = uint8_t(base + 4u);
			l.r_smid = uint8_t(base + 5u);
			l.r_idx = uint8_t(base + 6u);
			l.r_tmp = uint8_t(base + 7u);
			l.r_pr = uint8_t(base + 8u);
			l.r_ctrl = uint8_t(base + 10u);
			l.r_off = l.r_ptr;
		} else if (mode == Sm120SamplingConfig::Mode::PcMarker) {
			// PcMarker needs the same scratch set as ThreadMap (ctaid, tid, lane gate).
			l.r_ptr = uint8_t(base);
			l.r_cta = uint8_t(base + 2u);
			l.r_tid = uint8_t(base + 3u);
			l.r_lane = uint8_t(base + 4u);
			l.r_smid = uint8_t(base + 5u);
			l.r_idx = uint8_t(base + 6u);
			l.r_tmp = uint8_t(base + 7u);
			l.r_pr = uint8_t(base + 8u);
			l.r_ctrl = uint8_t(base + 10u);
			l.r_off = l.r_ptr;
		}
	l.old_regcount = old_regcount;
	l.new_regcount = new_regcount;
	return l;
}

static std::vector<SassInst>
build_sm120_smid_bitmap_stub(const Sm120SamplingConfig &cfg,
			     const Sm120RegLayout &r)
{
	std::vector<SassInst> insts;
	insts.reserve(12);

	const uint8_t ur = cfg.desc_ur;
	const uint8_t ur_save = (ur >= 2u) ? uint8_t(ur - 2u) : uint8_t(ur + 2u);
	// Do not assume a valid descriptor at kernel entry; use UR-base addressing.
	// Preserve the original UR pair since vendor kernels may depend on it.
	insts.push_back(inst_umov_ur_ur(ur_save, ur));
	insts.push_back(inst_umov_ur_ur(uint8_t(ur_save + 1u), uint8_t(ur + 1u)));
	const uint32_t lo = uint32_t(cfg.sample_buffer_device_ptr & 0xffffffffull);
	const uint32_t hi =
		uint32_t((cfg.sample_buffer_device_ptr >> 32) & 0xffffffffull);
	insts.push_back(inst_umov_ur_imm(ur, lo));
	insts.push_back(inst_umov_ur_imm(uint8_t(ur + 1u), hi));

	// smid -> r_smid
	insts.push_back(inst_s2r(r.r_smid, SM120_SR_VIRTUALSMID));
	// Keep the bitmap bounded (host buffer is 256 entries).
	insts.push_back(inst_lop3_and_imm_u32(r.r_smid, r.r_smid, 0xffu));

	// r_ptr(.64) = smid * 4 (byte offset).
	insts.push_back(inst_hfma2_imm_u32(r.r_ptr, 0));
	insts.push_back(inst_hfma2_imm_u32(uint8_t(r.r_ptr + 1u), 0));
	insts.push_back(inst_imad_wide_u32(r.r_ptr, r.r_smid, 0x4, r.r_ptr));

	// Store u32(1) at buffer[smid] via UR-base address.
	insts.push_back(inst_hfma2_imm_u32(r.r_tmp, 1));
	insts.push_back(inst_stg_e_u32_addr_ur(
		r.r_ptr, ur, r.r_tmp, (uint8_t)(cfg.buffer_data_offset & 0xffu)));

	insts.push_back(inst_umov_ur_ur(ur, ur_save));
	insts.push_back(
		inst_umov_ur_ur(uint8_t(ur + 1u), uint8_t(ur_save + 1u)));

	return insts;
}

static std::vector<SassInst>
build_sm120_cta_smid_stub(const Sm120SamplingConfig &cfg,
			  const Sm120RegLayout &r)
{
	std::vector<SassInst> insts;
	insts.reserve(16);

	const uint8_t ur = cfg.desc_ur;
	const uint8_t ur_save = (ur >= 2u) ? uint8_t(ur - 2u) : uint8_t(ur + 2u);
	insts.push_back(inst_umov_ur_ur(ur_save, ur));
	insts.push_back(inst_umov_ur_ur(uint8_t(ur_save + 1u), uint8_t(ur + 1u)));
	const uint32_t lo = uint32_t(cfg.sample_buffer_device_ptr & 0xffffffffull);
	const uint32_t hi =
		uint32_t((cfg.sample_buffer_device_ptr >> 32) & 0xffffffffull);
	insts.push_back(inst_umov_ur_imm(ur, lo));
	insts.push_back(inst_umov_ur_imm(uint8_t(ur + 1u), hi));

	insts.push_back(inst_s2r(r.r_cta, SM120_SR_CTAID_X));
	if (cfg.max_records != 0) {
		const uint32_t mask = cfg.max_records - 1u;
		insts.push_back(inst_lop3_and_imm_u32(r.r_cta, r.r_cta, mask));
	}
	insts.push_back(inst_hfma2_imm_u32(r.r_ptr, 0));
	insts.push_back(inst_hfma2_imm_u32(uint8_t(r.r_ptr + 1u), 0));
	insts.push_back(inst_imad_wide_u32(r.r_ptr, r.r_cta, 0x4, r.r_ptr));

	insts.push_back(inst_s2r(r.r_smid, SM120_SR_VIRTUALSMID));
	insts.push_back(inst_stg_e_u32_addr_ur(
		r.r_ptr, ur, r.r_smid, (uint8_t)(cfg.buffer_data_offset & 0xffu)));

	insts.push_back(inst_umov_ur_ur(ur, ur_save));
	insts.push_back(
		inst_umov_ur_ur(uint8_t(ur + 1u), uint8_t(ur_save + 1u)));
	return insts;
}

static std::vector<SassInst>
build_sm120_warp_map_stub(const Sm120SamplingConfig &cfg,
			  const Sm120RegLayout &r)
{
	std::vector<SassInst> insts;
	insts.reserve(24);

	const uint8_t ur = cfg.desc_ur;
	const uint8_t ur_save = (ur >= 2u) ? uint8_t(ur - 2u) : uint8_t(ur + 2u);
	// Use a UR-based addressing form (no `desc[URx]`), since many cubin-only
	// kernels in vLLM do not use descriptor-based global stores and may not
	// carry a valid descriptor at c[0x0][0x358].
	//
	// Preserve the original `ur/ur+1` pair since vendor libraries (e.g. cuBLAS)
	// can depend on their initial values at kernel entry.
	insts.push_back(inst_umov_ur_ur(ur_save, ur));
	insts.push_back(inst_umov_ur_ur(uint8_t(ur_save + 1u), uint8_t(ur + 1u)));
	insts.push_back(
		inst_umov_ur_imm(ur, uint32_t(cfg.sample_buffer_device_ptr & 0xffffffffull)));
	insts.push_back(inst_umov_ur_imm(
		uint8_t(ur + 1u),
		uint32_t((cfg.sample_buffer_device_ptr >> 32) & 0xffffffffull)));

	// idx = (ctaid.x * 32) + warpid
	// warpid = tid.x >> 5
	insts.push_back(inst_s2r(r.r_cta, SM120_SR_CTAID_X));
	insts.push_back(inst_s2r(r.r_tmp, SM120_SR_TID_X));
	insts.push_back(inst_shf_r_u32_hi_rz(r.r_warp, r.r_tmp, 0x5));
	insts.push_back(inst_imad_shl_u32(r.r_idx, r.r_cta, 0x20, r.r_warp));
	if (cfg.max_records != 0) {
		const uint32_t mask_slots = (cfg.max_records * 32u) - 1u;
		insts.push_back(inst_lop3_and_imm_u32(r.r_idx, r.r_idx, mask_slots));
	}

	// addr = idx * 4 (u32 slot offset)
	insts.push_back(inst_imad_wide_u32(r.r_ptr, r.r_idx, 0x4, 0xff));

	insts.push_back(inst_s2r(r.r_smid, SM120_SR_VIRTUALSMID));
	// Store only the low byte (enough for SM id on current parts).
	insts.push_back(inst_stg_e_u8_addr_ur(
		r.r_ptr, ur, r.r_smid, (uint8_t)(cfg.buffer_data_offset & 0xffu)));
	insts.push_back(inst_umov_ur_ur(ur, ur_save));
	insts.push_back(inst_umov_ur_ur(uint8_t(ur + 1u), uint8_t(ur_save + 1u)));
	return insts;
}

static std::vector<SassInst>
build_sm120_thread_map_stub(const Sm120SamplingConfig &cfg,
			    const Sm120RegLayout &r,
			    const std::optional<SassInst> &ldcu_desc)
{
	std::vector<SassInst> insts;
	insts.reserve(28);

	if (!cfg.thread_map_device) {
		const uint8_t ur = cfg.desc_ur;
		const uint8_t ur_save =
			(ur >= 2u) ? uint8_t(ur - 2u) : uint8_t(ur + 2u);
		insts.push_back(inst_umov_ur_ur(ur_save, ur));
		insts.push_back(inst_umov_ur_ur(uint8_t(ur_save + 1u), uint8_t(ur + 1u)));
		insts.push_back(inst_umov_ur_imm(
			ur, uint32_t(cfg.sample_buffer_device_ptr & 0xffffffffull)));
		insts.push_back(inst_umov_ur_imm(
			uint8_t(ur + 1u),
			uint32_t((cfg.sample_buffer_device_ptr >> 32) & 0xffffffffull)));

		// Stable default: device writes per-warp slots, host expands to per-thread.
		//   out[(ctaid.x * 32) + warp_id] = smid_lo8
		insts.push_back(inst_s2r(r.r_cta, SM120_SR_CTAID_X));
		insts.push_back(inst_s2r(r.r_tmp, SM120_SR_TID_X));
		// Reuse r_lane as a scratch register to hold warp_id.
		insts.push_back(inst_shf_r_u32_hi_rz(r.r_lane, r.r_tmp, 0x5));
		insts.push_back(inst_imad_shl_u32(r.r_idx, r.r_cta, 0x20, r.r_lane));
		if (cfg.max_records != 0) {
			const uint32_t mask_slots = (cfg.max_records * 32u) - 1u;
			insts.push_back(
				inst_lop3_and_imm_u32(r.r_idx, r.r_idx, mask_slots));
		}
		insts.push_back(inst_imad_wide_u32(r.r_ptr, r.r_idx, 0x4, 0xff));
		insts.push_back(inst_s2r(r.r_smid, SM120_SR_VIRTUALSMID));
		insts.push_back(inst_stg_e_u8_addr_ur(
			r.r_ptr, ur, r.r_smid, (uint8_t)(cfg.buffer_data_offset & 0xffu)));
		insts.push_back(inst_umov_ur_ur(ur, ur_save));
		insts.push_back(inst_umov_ur_ur(uint8_t(ur + 1u), uint8_t(ur_save + 1u)));
		return insts;
	}

		// Device per-thread writes:
		// - when `stride4`: store u32 to an aligned slot (more robust on vLLM/cutlass)
		// - otherwise: store u8 to a packed byte buffer (currently unstable on some
		//   kernels; keep for experimentation)
		const bool stride4_slots = cfg.thread_map_device_stride4;
			const bool lane0_only = cfg.thread_map_device_lane0_only;
			const bool warp0_only = cfg.thread_map_device_warp0_only;
			const bool cta_clamp = cfg.thread_map_device_cta_clamp;
			// Descriptor-based addressing (`STG.E desc[...]`) can be fragile on some
			// vendor/ATen kernels (we have seen CUBLAS_STATUS_INTERNAL_ERROR in real
			// vLLM runs). UR-base addressing is generally more robust.
			//
			// When `PREFER_EXIT` is enabled (high-risk bring-up), default to UR-base
			// unless the user explicitly forces desc mode.
			const bool prefer_exit =
				env_truthy("BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_PREFER_EXIT");
			const bool use_ur_addr =
				env_truthy("BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_USE_UR") ||
				prefer_exit;
			const bool use_desc = !use_ur_addr;

	const uint32_t raw_lo = uint32_t(cfg.sample_buffer_device_ptr & 0xffffffffull);
	const uint32_t raw_hi =
		uint32_t((cfg.sample_buffer_device_ptr >> 32) & 0xffffffffull);
	const uint32_t base_lo =
		raw_lo + uint32_t(cfg.buffer_data_offset & 0xffffffffu);
	const uint32_t base_hi =
		raw_hi + uint32_t(base_lo < raw_lo ? 1u : 0u);
	if (use_desc) {
		const uint8_t desc_ur = cfg.desc_ur;
		// Descriptor-based addressing (matches ptxas patterns):
		//   LDCU.64 URx, c[0][...]; addr = base + idx*stride; STG.E desc[URx][addr]
		//
		// This avoids the UR-base addressing form, which can produce unexpected
		// slot mappings on some kernels.
		// Use MOV for exact 32-bit materialization: HFMA2-based constants can lose
		// integer precision for large address words and trigger misaligned stores.
		insts.push_back(inst_mov_imm_u32(r.r_ptr, base_lo));
		insts.push_back(
			inst_mov_imm_u32(uint8_t(r.r_ptr + 1u), base_hi));
		insts.push_back(ldcu_desc.value_or(inst_ldcu64_desc_ur(desc_ur)));
		for (int i = 0; i < 8; i++)
			insts.push_back(inst_nop());
	} else {
		// UR-base addressing form:
		//   UMOV URx, base; addr = (idx*stride) + URx; STG.E [Roff.64+URx]
		const uint8_t ur = cfg.desc_ur;
		const uint8_t ur_save =
			(ur >= 2u) ? uint8_t(ur - 2u) : uint8_t(ur + 2u);
		insts.push_back(inst_umov_ur_ur(ur_save, ur));
		insts.push_back(
			inst_umov_ur_ur(uint8_t(ur_save + 1u), uint8_t(ur + 1u)));
		insts.push_back(inst_umov_ur_imm(ur, base_lo));
		insts.push_back(inst_umov_ur_imm(uint8_t(ur + 1u), base_hi));
		for (int i = 0; i < 8; i++)
			insts.push_back(inst_nop());
	}

		// Device per-thread write (no host expansion):
		//   out[(ctaid_mod * 1024) + tid_x] = smid_lo8
		const uint8_t stride = stride4_slots ? uint8_t(0x4) : uint8_t(0x1);

		insts.push_back(inst_s2r(r.r_cta, SM120_SR_CTAID_X));
	// Be conservative about S2R latency at function entry; missing stalls can
	// lead to non-deterministic slot computations in minimal stubs.
	const int s2r_wait_nops = [] {
		if (auto v = env_u32("BPFTIME_CUDA_SASS_DETOUR_S2R_WAIT_NOPS"))
			return int(std::min<uint32_t>(*v, 32u));
		return 4;
	}();
		for (int i = 0; i < s2r_wait_nops; i++)
			insts.push_back(inst_nop());

		// This stub may use P0 for density gates and/or CTA clamp gating. Preserve
		// the original P0 state for the kernel.
		const bool needs_p0_save =
			(lane0_only || warp0_only) || (cta_clamp && cfg.max_records != 0);
		if (needs_p0_save)
			insts.push_back(inst_p2r_pr_mask(r.r_pr, 0xff,
							 sass_detour_pr_save_mask()));

		std::optional<size_t> bra_skip_all;
		if (cfg.max_records != 0) {
			if (cta_clamp) {
				// Reduce contention for full device-per-thread writes:
				// skip stores for CTAs outside the configured window.
				//
				//   P0 = (ctaid.x >= max_records); @P0 BRA label_skip_all
				insts.push_back(
					inst_isetp_ge_u32_and_p0_imm(r.r_cta, cfg.max_records));
				bra_skip_all = insts.size();
				insts.push_back(SassInst {}); // @P0 BRA label_skip_all
			} else {
				// Mask-based indexing (requires power-of-two max_records).
				const uint32_t mask = cfg.max_records - 1u;
				insts.push_back(
					inst_lop3_and_imm_u32(r.r_cta, r.r_cta, mask));
			}
		}
		insts.push_back(inst_s2r(r.r_tid, SM120_SR_TID_X));
		for (int i = 0; i < s2r_wait_nops; i++)
			insts.push_back(inst_nop());
	insts.push_back(inst_lop3_and_imm_u32(r.r_tid, r.r_tid, 0x3ffu));
	// idx = (ctaid_mod * 1024) + tid_x
	//
	// Since tid_x < 1024 (10 bits), this is equivalent to:
	//   idx = (ctaid_mod << 10) | tid_x
	//
	// Use r_idx directly to avoid consuming additional scratch registers in
	// the minimal sampling stub.
	insts.push_back(inst_shf_l_u32_rz(r.r_idx, r.r_cta, 0x0a));
	insts.push_back(inst_lop3_or_rr(r.r_idx, r.r_idx, r.r_tid));
	// The compiler schedules a few cycles between idx materialization and the
	// following wide address math; without a conservative gap here, we observed
	// non-deterministic behavior where many threads appear to reuse a stale idx
	// (often 0), collapsing writes into slot0.
	const int idx_wait_nops = [] {
		if (auto v = env_u32("BPFTIME_CUDA_SASS_DETOUR_IDX_WAIT_NOPS"))
			return int(std::min<uint32_t>(*v, 32u));
		return 8;
	}();
	for (int i = 0; i < idx_wait_nops; i++)
		insts.push_back(inst_nop());

		if (!cfg.thread_map_device_no_store) {
			insts.push_back(inst_s2r(r.r_smid, SM120_SR_VIRTUALSMID));
			// SMID is also sourced from S2R and can have non-trivial latency at
			// kernel entry. Without a conservative gap here, we observed cases on
			// real vLLM kernels where the following store uses a stale/uninitialized
			// value (often 0xffffffff), resulting in “meta-only” dumps.
			for (int i = 0; i < s2r_wait_nops; i++)
				insts.push_back(inst_nop());
			const bool dbg_store_tid = env_truthy(
				"BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_DEBUG_STORE_TID");
			const bool dbg_store_idx = env_truthy(
				"BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_DEBUG_STORE_IDX");
		uint8_t store_val_reg = r.r_smid;
		if (dbg_store_tid)
			store_val_reg = r.r_tid;
		if (dbg_store_idx)
			store_val_reg = r.r_idx;
		// addr = base + (idx * stride)
		if (use_desc) {
			insts.push_back(inst_imad_wide_u32(
				r.r_ptr, r.r_idx, stride, r.r_ptr /*rb=*/));
		} else {
			const uint8_t ur = cfg.desc_ur;
			// ptr = idx * stride (rb=RZ)
			insts.push_back(inst_imad_wide_u32(r.r_ptr, r.r_idx, stride, 0xff));
			(void)ur;
		}

		// Optional density gates for real workloads:
		// - lane0_only: only lane0 writes (1/32 per warp)
		// - warp0_only: only warp0 writes (1/32 per CTA when full)
		// - both: only tid_x==0 writes (1/1024)
		//
			// Use predicated BRA (not `@!P0 STG`) for the lane0/warp0 gate. The
			// predicated store form has shown unreliable behavior on some real
			// workloads (vLLM/cutlass), while BRA+unpredicated store is stable.
			if (lane0_only || warp0_only) {
				if (lane0_only && warp0_only) {
					// Only tid_x==0 writes: P0 = (tid_x != 0) => skip.
					insts.push_back(inst_isetp_ne_u32_and_p0(r.r_tid));
				} else if (warp0_only) {
				// Only warp0 writes: P0 = (tid_x >= 32) => skip.
				insts.push_back(inst_isetp_ge_u32_and_p0_imm(r.r_tid, 0x20u));
			} else {
				// Only lane0 writes: P0 = (lane_id != 0) => skip.
				// Derive lane_id from tid_x: lane_id = tid_x & 31.
				// This is cheap and avoids relying on SR_LANEID being valid at
				// every kernel entry point (some cubin-only kernels show
				// surprising SR_LANEID behavior before their own prologues).
				insts.push_back(
					inst_lop3_and_imm_u32(r.r_lane, r.r_tid, 0x1fu));
				insts.push_back(inst_isetp_ne_u32_and_p0(r.r_lane));
				}

				const size_t bra_skip = insts.size();
				insts.push_back(SassInst {}); // @P0 BRA label_skip_store
			insts.push_back(
				stride4_slots
					? (use_desc
						   ? inst_stg_e_u32_off(/*ur_base=*/cfg.desc_ur,
									r.r_ptr,
									store_val_reg, 0)
						   : inst_stg_e_u32_addr_ur(
							     r.r_ptr, cfg.desc_ur,
							     store_val_reg, 0))
					: (use_desc
						   ? inst_stg_e_u8(/*ur_base=*/cfg.desc_ur,
								   r.r_ptr, store_val_reg)
						   : inst_stg_e_u8_addr_ur(
							     r.r_ptr, cfg.desc_ur,
							     store_val_reg, 0)));
			const size_t label_skip_store = insts.size();
				{
					const int64_t delta =
						int64_t(label_skip_store - bra_skip) *
						(int64_t)SASS_INST_BYTES;
					insts[bra_skip] = inst_bra_sm120_pred_p0(delta);
				}
			} else {
				insts.push_back(
					stride4_slots
					? (use_desc
						   ? inst_stg_e_u32_off(/*ur_base=*/cfg.desc_ur,
									r.r_ptr,
									store_val_reg, 0)
						   : inst_stg_e_u32_addr_ur(
							     r.r_ptr, cfg.desc_ur,
							     store_val_reg, 0))
					: (use_desc
						   ? inst_stg_e_u8(/*ur_base=*/cfg.desc_ur,
								   r.r_ptr, store_val_reg)
						   : inst_stg_e_u8_addr_ur(
							     r.r_ptr, cfg.desc_ur,
							     store_val_reg, 0)));
			}
		}
		const size_t label_skip_all = insts.size();
		if (bra_skip_all.has_value()) {
			const int64_t delta =
				int64_t(label_skip_all - *bra_skip_all) *
				(int64_t)SASS_INST_BYTES;
			insts[*bra_skip_all] = inst_bra_sm120_pred_p0(delta);
		}
		if (needs_p0_save)
			insts.push_back(inst_r2p_pr_mask(r.r_pr,
							 sass_detour_pr_save_mask()));
		if (!use_desc) {
			const uint8_t ur = cfg.desc_ur;
			const uint8_t ur_save =
			(ur >= 2u) ? uint8_t(ur - 2u) : uint8_t(ur + 2u);
		insts.push_back(inst_umov_ur_ur(ur, ur_save));
		insts.push_back(
			inst_umov_ur_ur(uint8_t(ur + 1u), uint8_t(ur_save + 1u)));
	}
		return insts;
	}

static std::vector<SassInst>
build_sm120_pc_marker_stub(const Sm120SamplingConfig &cfg,
			   const Sm120RegLayout &r, uint32_t tag,
			   uint32_t marker_off,
			   const std::optional<SassInst> &ldcu_desc)
{
			std::vector<SassInst> insts;
			insts.reserve(96);
			// Debug/bring-up: make the EXIT trampoline as small as possible.
			// If this still crashes, the root cause is likely the EXIT detour itself
			// (BRA/trampoline placement) rather than the body of the stub.
			if (env_truthy("BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_EXIT_STUB_MINIMAL")) {
				insts.push_back(inst_nop());
				return insts;
			}

	const uint32_t ring_entries = std::max<uint32_t>(1u, cfg.marker_ring_entries);
	const uint32_t ring_mask = ring_entries - 1u;
	const bool lane0_only = cfg.marker_lane0_only;
	const bool warp0_only = cfg.marker_warp0_only;
	const bool cta_clamp = cfg.marker_cta_clamp;
	const uint32_t max_ctas = cfg.max_records; // 0 => no CTA gating

	constexpr uint8_t kMarkerHeaderBytes = 0x10u;
	constexpr uint8_t kRecordBytes = 0x18u; // 6*u32

	const uint32_t raw_lo = uint32_t(cfg.sample_buffer_device_ptr & 0xffffffffull);
	const uint32_t raw_hi =
		uint32_t((cfg.sample_buffer_device_ptr >> 32) & 0xffffffffull);
	const uint32_t base_lo =
		raw_lo + uint32_t(cfg.buffer_data_offset & 0xffffffffu);
	const uint32_t base_hi =
		raw_hi + uint32_t(base_lo < raw_lo ? 1u : 0u);

	// Materialize the absolute base address and load a global memory descriptor.
	// Keep it similar to ThreadMap(device) to match ptxas patterns on SM120.
	insts.push_back(inst_mov_imm_u32(r.r_ptr, base_lo));
	insts.push_back(inst_mov_imm_u32(uint8_t(r.r_ptr + 1u), base_hi));
	insts.push_back(ldcu_desc.value_or(inst_ldcu64_desc_ur(cfg.desc_ur)));
	for (int i = 0; i < 8; i++)
		insts.push_back(inst_nop());

	insts.push_back(inst_s2r(r.r_cta, SM120_SR_CTAID_X));
	insts.push_back(inst_s2r(r.r_tid, SM120_SR_TID_X));
	const int s2r_wait_nops = [] {
		if (auto v = env_u32("BPFTIME_CUDA_SASS_DETOUR_S2R_WAIT_NOPS"))
			return int(std::min<uint32_t>(*v, 32u));
		return 4;
	}();
	for (int i = 0; i < s2r_wait_nops; i++)
		insts.push_back(inst_nop());

	const bool needs_p0_save =
		(lane0_only || warp0_only) || (cta_clamp && max_ctas != 0);
	if (needs_p0_save)
		insts.push_back(inst_p2r_pr_mask(r.r_pr, 0xff,
						 sass_detour_pr_save_mask()));

	std::optional<size_t> bra_skip_all;
	if (cta_clamp && max_ctas != 0) {
		insts.push_back(inst_isetp_ge_u32_and_p0_imm(r.r_cta, max_ctas));
		bra_skip_all = insts.size();
		insts.push_back(SassInst {}); // @P0 BRA label_skip_all
	}

	std::optional<size_t> bra_skip_store;
	if (lane0_only || warp0_only) {
		if (lane0_only && warp0_only) {
			// Only tid_x==0 writes.
			insts.push_back(inst_isetp_ne_u32_and_p0(r.r_tid));
		} else if (warp0_only) {
			// Only warp0 writes: skip when tid_x >= 32.
			insts.push_back(inst_isetp_ge_u32_and_p0_imm(r.r_tid, 0x20u));
		} else {
			// Only lane0 writes: skip when lane_id != 0.
			insts.push_back(inst_lop3_and_imm_u32(r.r_lane, r.r_tid, 0x1fu));
			insts.push_back(inst_isetp_ne_u32_and_p0(r.r_lane));
		}
		bra_skip_store = insts.size();
		insts.push_back(SassInst {}); // @P0 BRA label_skip_store
	}

	// seq = atomicAdd(write_idx, 1)
	insts.push_back(inst_mov_imm_u32(r.r_lane, 1u));
	insts.push_back(inst_atomg_add_u32(/*ur_base=*/cfg.desc_ur,
					   /*rd=*/r.r_idx,
					   /*addr_pair=*/r.r_ptr,
					   /*rs=*/r.r_lane));
	// idx = seq & (ring_entries-1)
	insts.push_back(inst_lop3_and_imm_u32(r.r_tmp, r.r_idx, ring_mask));

	// Similar to ThreadMap(device) bring-up: give the wide address math some
	// slack so the masked index is reliably visible (avoids collapsing writes
	// into the header / slot0 on some cubin-only kernels).
	const int idx_wait_nops = [] {
		if (auto v = env_u32("BPFTIME_CUDA_SASS_DETOUR_IDX_WAIT_NOPS"))
			return int(std::min<uint32_t>(*v, 32u));
		return 8;
	}();
	for (int i = 0; i < idx_wait_nops; i++)
		insts.push_back(inst_nop());

	// r_ptr = base + marker_header_bytes + idx * record_bytes
	insts.push_back(inst_iadd64_ri(r.r_ptr, r.r_ptr, kMarkerHeaderBytes));
	insts.push_back(inst_imad_wide_u32(r.r_ptr, r.r_tmp, kRecordBytes, r.r_ptr));

	insts.push_back(inst_s2r(r.r_smid, SM120_SR_VIRTUALSMID));
	for (int i = 0; i < s2r_wait_nops; i++)
		insts.push_back(inst_nop());

	// record: {seq, marker_off, tag, smid_raw, ctaid_x, tid_x}
	insts.push_back(inst_stg_e_u32_off(/*ur_base=*/cfg.desc_ur,
					   /*addr_pair=*/r.r_ptr,
					   /*rs=*/r.r_idx,
					   /*imm8=*/0x00));
	// Use distinct source regs for back-to-back stores. On some cubin-only
	// workloads, reusing the same GPR for `MOV imm` + `STG` without spacing can
	// result in the earlier store observing the later value.
	insts.push_back(inst_mov_imm_u32(r.r_lane, marker_off));
	insts.push_back(inst_stg_e_u32_off(cfg.desc_ur, r.r_ptr, r.r_lane, 0x04));
	insts.push_back(inst_mov_imm_u32(r.r_tmp, tag));
	insts.push_back(inst_stg_e_u32_off(cfg.desc_ur, r.r_ptr, r.r_tmp, 0x08));
	insts.push_back(inst_stg_e_u32_off(cfg.desc_ur, r.r_ptr, r.r_smid, 0x0c));
	insts.push_back(inst_stg_e_u32_off(cfg.desc_ur, r.r_ptr, r.r_cta, 0x10));
	insts.push_back(inst_stg_e_u32_off(cfg.desc_ur, r.r_ptr, r.r_tid, 0x14));

	const size_t label_skip_all = insts.size();
	if (bra_skip_store) {
		const int64_t delta =
			int64_t(label_skip_all - *bra_skip_store) *
			(int64_t)SASS_INST_BYTES;
		insts[*bra_skip_store] = inst_bra_sm120_pred_p0(delta);
	}
	if (bra_skip_all) {
		const int64_t delta =
			int64_t(label_skip_all - *bra_skip_all) *
			(int64_t)SASS_INST_BYTES;
		insts[*bra_skip_all] = inst_bra_sm120_pred_p0(delta);
	}

	if (needs_p0_save)
		insts.push_back(inst_r2p_pr_mask(r.r_pr,
						 sass_detour_pr_save_mask()));
	return insts;
}

				static std::vector<SassInst>
				build_sm120_thread_map_stub_no_regcount_at_exit(
					const Sm120SamplingConfig &cfg, uint32_t func_id,
					const std::optional<std::array<uint8_t, 6>> &scratch_regs,
					std::optional<uint8_t> existing_desc_ur_base = std::nullopt,
					std::optional<uint64_t> bra_w1_fwd_override = std::nullopt)
				{
			// regcount=255 bring-up stub:
		// - detour at a (preferably unpredicated) EXIT instruction
		// - use low GPRs as scratch without save/restore (kernel terminates after EXIT)
		//
		// This avoids the entry/prologue sensitivity of reg-saturated vendor kernels
		// (flashattention) while still enabling device-per-thread observability.
		if (cfg.mode != Sm120SamplingConfig::Mode::ThreadMap ||
		    !cfg.thread_map_device)
			return {};

			std::vector<SassInst> insts;
			insts.reserve(96);

			// Debug/bring-up: minimize the EXIT stub body. The trampoline still
			// replays the original EXIT instruction, so a single NOP here is enough
			// to validate whether crashes come from the detour mechanism (BRA/cave)
			// or from the stub body itself.
			if (env_truthy("BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_EXIT_STUB_MINIMAL")) {
				insts.push_back(inst_nop());
				return insts;
			}

				// Scratch registers:
				// - default: fixed low GPRs (safe because the kernel terminates at EXIT)
				// - optional: user/analysis-chosen dead regs (reduces interference risk)
				const std::array<uint8_t, 6> regs =
				scratch_regs.value_or(std::array<uint8_t, 6> { 0, 2, 3, 4, 5, 6 });
		const uint8_t r_ptr = regs[0]; // .64 pair: r_ptr/r_ptr+1
		const uint8_t r_cta = regs[1];
		const uint8_t r_tid = regs[2];
			const uint8_t r_lane = regs[3];
			const uint8_t r_smid = regs[4];
			const uint8_t r_idx = regs[5];

			// Bisect helper for reg255 flash bring-up: build a tiny EXIT stub and
			// gradually enable opcodes to pinpoint which template triggers
			// `cudaErrorIllegalInstruction` on real workloads.
			//
			// Step meaning:
			//   0: NOP
			//   1: imm32 -> r_ptr (low)
			//   2: imm32 -> r_ptr+1 (high)
			//   3: imm32 -> r_smid (marker)
			//   4: STG.E.32 marker to data[0]
			if (auto step = env_u32(
				    "BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_EXIT_STUB_STEP")) {
				const uint64_t data_base =
					cfg.sample_buffer_device_ptr + cfg.buffer_data_offset;
				const uint32_t data_lo =
					uint32_t(data_base & 0xffffffffull);
				const uint32_t data_hi =
					uint32_t((data_base >> 32) & 0xffffffffull);
				const uint8_t ur = cfg.desc_ur;

				insts.push_back(inst_nop());
				if (*step >= 1) {
					insts.push_back(inst_umov_ur_imm(ur, data_lo));
					insts.push_back(
						inst_umov_ur_imm(uint8_t(ur + 1u), data_hi));
					for (int i = 0; i < 8; i++)
						insts.push_back(inst_nop());
				}
				if (*step >= 2) {
					insts.push_back(inst_mov_imm_u32(r_smid, 1u));
					for (int i = 0; i < 4; i++)
						insts.push_back(inst_nop());
				}
				if (*step >= 3) {
					insts.push_back(
						inst_stg_e_u32_addr_ur(/*addr_pair=*/0xff,
								       /*ur_off=*/ur,
								       /*rs=*/r_smid,
								       /*imm8=*/0));
				}
				return insts;
			}

			// Bring-up: validate the EXIT detour + trampoline path with the smallest
			// possible "writes something" stub, without any S2R/ISETP/branch gating.
			// This helps bisect "illegal instruction" failures down to the first
			// problematic opcode/template.
			if (env_truthy("BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_EXIT_STUB_SIMPLE_STORE")) {
				const uint64_t data_base =
					cfg.sample_buffer_device_ptr + cfg.buffer_data_offset;
				const uint32_t data_lo = uint32_t(data_base & 0xffffffffull);
				const uint32_t data_hi =
					uint32_t((data_base >> 32) & 0xffffffffull);
				const uint8_t desc_ur_base =
					existing_desc_ur_base.value_or(uint8_t(4u));

				insts.push_back(inst_mov_imm_u32(r_ptr, data_lo));
				insts.push_back(
					inst_mov_imm_u32(uint8_t(r_ptr + 1u), data_hi));
				insts.push_back(inst_mov_imm_u32(r_smid, 1u));
				for (int i = 0; i < 4; i++)
					insts.push_back(inst_nop());
				insts.push_back(inst_stg_e_u32(desc_ur_base, r_ptr, r_smid));
				return insts;
			}

			const bool stride4_slots = cfg.thread_map_device_stride4;
			const bool lane0_only = cfg.thread_map_device_lane0_only;
		const bool warp0_only = cfg.thread_map_device_warp0_only;
			const bool cta_clamp = cfg.thread_map_device_cta_clamp;
			const uint8_t stride = stride4_slots ? uint8_t(0x4) : uint8_t(0x1);
		const int s2r_wait_nops = [] {
			if (auto v =
				    env_u32("BPFTIME_CUDA_SASS_DETOUR_S2R_WAIT_NOPS"))
				return int(std::min<uint32_t>(*v, 32u));
			return 8;
		}();

		// Optional reg255 spill area: reserve a small region after sampler data
		// so we can preserve a small subset of GPRs/predicates while still using
		// a minimal no-regcount stub shape.
		const bool tid0_only = lane0_only && warp0_only;
		const bool spill_unsafe = env_truthy(
			"BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_REG255_SPILL_UNSAFE");
		const bool spill_per_thread = cfg.reg255_thread_map_spill_per_thread &&
					      cfg.reg255_thread_map_spill_stride_bytes >=
						      0x10u;
		const bool spill_enable =
			cfg.reg255_thread_map_spill_enable &&
			((spill_per_thread && cfg.reg255_thread_map_spill_bytes != 0) ||
			 (!spill_per_thread && cfg.reg255_thread_map_spill_bytes >=
						      0x40u &&
			  // Shared spill header is unsafe for >1 thread. Keep it as a
			  // debug-only mode unless the user forces it.
			  (spill_unsafe || (tid0_only && cta_clamp &&
					    cfg.max_records == 1u))));
		const uint32_t data_bytes =
			(uint32_t(std::max(1u, cfg.max_records)) * 1024u *
			 uint32_t(stride));
		const uint64_t spill_base =
			cfg.sample_buffer_device_ptr + uint64_t(cfg.buffer_data_offset) +
			uint64_t(data_bytes);
		const uint32_t spill_lo = uint32_t(spill_base & 0xffffffffull);
		const uint32_t spill_hi = uint32_t((spill_base >> 32) & 0xffffffffull);

				// Legacy path uses UR-base addressing:
				//   UMOV URx, base; addr = idx*stride; STG.E [addr + URx], Rs
				//
				// WARNING: UR registers are warp-uniform. For predicated EXIT detours,
				// only a subset of lanes may take the detour and execute the stub. In
				// that case, writing UR (UMOV URx, imm) can corrupt the UR state for
				// lanes that did not take the detour and continue execution.
				//
				// To keep predicated EXIT detours safe, we support a "no-UR-write" mode:
				// - use an existing global descriptor (UR4 by convention)
				// - materialize absolute addresses in GPR pairs
				// - avoid any UMOV/LDCU that writes UR in the stub
				const uint8_t ur = cfg.desc_ur;
				// Default to UR-base addressing for both control-header access and
				// sampler writes. This is the only mode that guarantees our own
				// sampler buffer is reachable on cubin-only vendor kernels; using an
				// in-kernel descriptor (desc[UR*]) can silently drop stores because the
				// descriptor may be specialized to the kernel's own address ranges.
				//
				// For debugging only: allow a descriptor-based no-UR-write path.
				const bool use_existing_desc =
					existing_desc_ur_base.has_value() &&
					env_truthy("BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_EXIT_STUB_USE_EXISTING_DESC");
				const uint8_t desc_ur_base =
					existing_desc_ur_base.value_or(uint8_t(4u));
				const uint64_t buf_base = cfg.sample_buffer_device_ptr;
			const uint64_t data_base =
				cfg.sample_buffer_device_ptr + cfg.buffer_data_offset;
			const uint32_t buf_lo = uint32_t(buf_base & 0xffffffffull);
			const uint32_t buf_hi = uint32_t((buf_base >> 32) & 0xffffffffull);
			const uint32_t data_lo = uint32_t(data_base & 0xffffffffull);
			const uint32_t data_hi =
				uint32_t((data_base >> 32) & 0xffffffffull);
				// For legacy UR-base stubs, start from buffer base so we can consult
				// the control header, then switch UR to the data base before writes.
				if (!use_existing_desc) {
					insts.push_back(inst_umov_ur_imm(ur, buf_lo));
					insts.push_back(inst_umov_ur_imm(uint8_t(ur + 1u), buf_hi));
					for (int i = 0; i < 8; i++)
						insts.push_back(inst_nop());
					// Debug: prove the EXIT stub actually executes by scribbling a marker
					// into the control header (visible in bpftime_sass_sample_meta).
					//
					// WARNING: this overwrites host-side `reserved0` ("TARG").
					if (env_truthy("BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_EXIT_STUB_DEBUG_WRITE_CTRL")) {
						insts.push_back(inst_mov_imm_u32(r_smid, 0x44454247u)); // "DEBG"
						insts.push_back(inst_stg_e_u32_addr_ur(
							/*addr_pair=*/0xff, /*ur_off=*/ur, /*rs=*/r_smid,
							/*imm8=*/0x18u));
					}
				}

			// Optional CTA clamp / CTA mask.
			insts.push_back(inst_s2r(r_cta, SM120_SR_CTAID_X));
			for (int i = 0; i < s2r_wait_nops; i++)
				insts.push_back(inst_nop());
			std::vector<size_t> bra_skip_all_sites;
			if (cfg.max_records != 0) {
				if (cta_clamp) {
					insts.push_back(inst_isetp_ge_u32_and_p0_imm(
						r_cta, cfg.max_records));
					bra_skip_all_sites.push_back(insts.size());
					insts.push_back(SassInst {}); // @P0 BRA label_skip_all
				} else {
					const uint32_t mask = cfg.max_records - 1u;
					insts.push_back(
						inst_lop3_and_imm_u32(r_cta, r_cta, mask));
				}
			}

			// Optional: control header gating (stable at EXIT detour points).
			// If enabled, only write when the host has set Target mode for this
			// specific func_id. This reduces cross-kernel interference (the buffer
			// is shared across all instrumented kernels) and avoids needing the
			// reg255 entry/prologue `_UNSAFE` device-side control read path.
					if (cfg.control_enabled &&
					    !env_truthy(
						    "BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_EXIT_STUB_NO_CONTROL")) {
					// NOTE: r_ptr is a .64 pair. Ensure r_ptr+1 is usable (not RZ).
					if (r_ptr < 254u) {
						if (use_existing_desc) {
							// Safe for predicated EXIT detours: do NOT write UR registers.
							// Use desc[UR4][Rptr.64] loads with absolute addresses in r_ptr.
							insts.push_back(inst_mov_imm_u32(r_ptr, buf_lo));
							insts.push_back(inst_mov_imm_u32(uint8_t(r_ptr + 1u), buf_hi));

							// enable @ +0x08
							insts.push_back(inst_iadd64_ri(r_ptr, r_ptr, 0x08));
							insts.push_back(inst_ldg_e_u32_desc_ur(
								/*rd=*/r_smid, /*addr_pair=*/r_ptr,
								/*ur_base=*/desc_ur_base));
							for (int i = 0; i < 8; i++)
								insts.push_back(inst_nop());
							insts.push_back(inst_isetp_ne_u32_and_p0_imm(r_smid, 1u));
							bra_skip_all_sites.push_back(insts.size());
							insts.push_back(SassInst {}); // @P0 BRA label_skip_all

							// mode @ +0x0c
							insts.push_back(inst_iadd64_ri(r_ptr, r_ptr, 0x04));
							insts.push_back(inst_ldg_e_u32_desc_ur(
								/*rd=*/r_smid, /*addr_pair=*/r_ptr,
								/*ur_base=*/desc_ur_base));
							for (int i = 0; i < 8; i++)
								insts.push_back(inst_nop());
							insts.push_back(inst_isetp_ne_u32_and_p0_imm(r_smid, 2u));
							bra_skip_all_sites.push_back(insts.size());
							insts.push_back(SassInst {}); // @P0 BRA label_skip_all

							// target_func_id @ +0x14
							insts.push_back(inst_iadd64_ri(r_ptr, r_ptr, 0x08));
							insts.push_back(inst_ldg_e_u32_desc_ur(
								/*rd=*/r_smid, /*addr_pair=*/r_ptr,
								/*ur_base=*/desc_ur_base));
							for (int i = 0; i < 8; i++)
								insts.push_back(inst_nop());
							insts.push_back(inst_isetp_ne_u32_and_p0_imm(r_smid, func_id));
							bra_skip_all_sites.push_back(insts.size());
							insts.push_back(SassInst {}); // @P0 BRA label_skip_all
						} else {
							// Legacy: UR-base loads using [Rptr.64 + UR + imm] (requires UR set).
							// r_ptr(.64) = 0 (address base for [UR + imm] loads).
							insts.push_back(inst_mov_imm_u32(r_ptr, 0u));
							insts.push_back(inst_mov_imm_u32(uint8_t(r_ptr + 1u), 0u));

							// enable must be 1.
							insts.push_back(inst_ldg_e_u32_addr_ur(
								/*rd=*/r_smid, /*addr_pair=*/r_ptr,
								/*ur_off=*/ur, /*imm8=*/0x08));
							for (int i = 0; i < 8; i++)
								insts.push_back(inst_nop());
							insts.push_back(inst_isetp_ne_u32_and_p0_imm(r_smid, 1u));
							bra_skip_all_sites.push_back(insts.size());
							insts.push_back(SassInst {}); // @P0 BRA label_skip_all

							// mode must be Target (2).
							insts.push_back(inst_ldg_e_u32_addr_ur(
								/*rd=*/r_smid, /*addr_pair=*/r_ptr,
								/*ur_off=*/ur, /*imm8=*/0x0c));
							for (int i = 0; i < 8; i++)
								insts.push_back(inst_nop());
							insts.push_back(inst_isetp_ne_u32_and_p0_imm(r_smid, 2u));
							bra_skip_all_sites.push_back(insts.size());
							insts.push_back(SassInst {}); // @P0 BRA label_skip_all

							// target_func_id must match this section's func_id.
							insts.push_back(inst_ldg_e_u32_addr_ur(
								/*rd=*/r_smid, /*addr_pair=*/r_ptr,
								/*ur_off=*/ur, /*imm8=*/0x14));
							for (int i = 0; i < 8; i++)
								insts.push_back(inst_nop());
							insts.push_back(inst_isetp_ne_u32_and_p0_imm(r_smid, func_id));
							bra_skip_all_sites.push_back(insts.size());
							insts.push_back(SassInst {}); // @P0 BRA label_skip_all
						}
					}
				}

				// Set up the sampler data base for the main write path.
				if (use_existing_desc) {
					// No-UR-write mode: materialize base address in r_ptr(.64).
					insts.push_back(inst_mov_imm_u32(r_ptr, data_lo));
					insts.push_back(
						inst_mov_imm_u32(uint8_t(r_ptr + 1u), data_hi));
				} else {
					// Legacy: switch UR to sampler data base.
					insts.push_back(inst_umov_ur_imm(ur, data_lo));
					insts.push_back(inst_umov_ur_imm(uint8_t(ur + 1u), data_hi));
					for (int i = 0; i < 8; i++)
						insts.push_back(inst_nop());
				}

			// tid_x and idx = (ctaid_mod*1024) + tid_x.
			insts.push_back(inst_s2r(r_tid, SM120_SR_TID_X));
			for (int i = 0; i < s2r_wait_nops; i++)
				insts.push_back(inst_nop());
		insts.push_back(inst_lop3_and_imm_u32(r_tid, r_tid, 0x3ffu));
		insts.push_back(inst_shf_l_u32_rz(r_idx, r_cta, 0x0a));
		insts.push_back(inst_lop3_or_rr(r_idx, r_idx, r_tid));

		// Preserve predicate state (at least P0):
		// - capture PR to r_lane
		// - if per-thread spill is enabled, write it to a unique slot so r_lane
		//   can be reused for other scratch purposes.
			insts.push_back(inst_p2r_pr_mask(/*rd=*/r_lane, /*ra=*/0xff,
							 /*mask=*/sass_detour_pr_save_mask()));
				if (spill_enable && spill_per_thread && !use_existing_desc) {
			const uint32_t spill_stride =
				std::max<uint32_t>(0x10u, cfg.reg255_thread_map_spill_stride_bytes);
			// r_ptr = idx * spill_stride (rb=RZ)
			insts.push_back(
				inst_imad_wide_u32(r_ptr, r_idx, spill_stride, 0xff));
			// Switch UR to spill base and store PR snapshot at +0x0c.
			insts.push_back(inst_umov_ur_imm(ur, spill_lo));
			insts.push_back(inst_umov_ur_imm(uint8_t(ur + 1u), spill_hi));
			for (int i = 0; i < 8; i++)
				insts.push_back(inst_nop());
			insts.push_back(inst_stg_e_u32_addr_ur(
				/*addr_pair=*/r_ptr, /*ur_off=*/ur, /*rs=*/r_lane,
				/*imm8=*/0x0c));
			// Restore UR to sampler base for the main write path.
			insts.push_back(inst_umov_ur_imm(ur, data_lo));
			insts.push_back(inst_umov_ur_imm(uint8_t(ur + 1u), data_hi));
				for (int i = 0; i < 8; i++)
					insts.push_back(inst_nop());
			}

			// Optional density gates (branch-based).
			std::optional<size_t> bra_skip_store;
			if (lane0_only || warp0_only) {
			if (lane0_only && warp0_only) {
				insts.push_back(inst_isetp_ne_u32_and_p0(r_tid));
			} else if (warp0_only) {
				insts.push_back(
					inst_isetp_ge_u32_and_p0_imm(r_tid, 0x20u));
			} else {
				insts.push_back(
					inst_lop3_and_imm_u32(r_smid, r_tid, 0x1fu));
				insts.push_back(inst_isetp_ne_u32_and_p0(r_smid));
			}
			bra_skip_store = insts.size();
			insts.push_back(SassInst {}); // @P0 BRA label_skip_store
		}

			if (!cfg.thread_map_device_no_store) {
			const bool dbg_store_tid = env_truthy(
				"BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_DEBUG_STORE_TID");
			const bool dbg_store_idx = env_truthy(
				"BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_DEBUG_STORE_IDX");
			const auto dbg_store_const =
				env_u32("BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_DEBUG_STORE_CONST");
			uint8_t store_val_reg = r_smid;
			if (dbg_store_tid)
				store_val_reg = r_tid;
			if (dbg_store_idx)
				store_val_reg = r_idx;

				// addr = base + idx * stride
				// - legacy: base is in UR, so compute offset only (rb=RZ)
				// - desc mode: base is already in r_ptr(.64), so accumulate into it
				insts.push_back(inst_imad_wide_u32(
					r_ptr, r_idx, stride,
					use_existing_desc ? r_ptr : uint8_t(0xff)));
				if (dbg_store_const.has_value()) {
					insts.push_back(inst_mov_imm_u32(
						r_smid, uint32_t(*dbg_store_const)));
					store_val_reg = r_smid;
				} else if (store_val_reg == r_smid) {
					insts.push_back(inst_s2r(r_smid, SM120_SR_VIRTUALSMID));
					for (int i = 0; i < s2r_wait_nops; i++)
						insts.push_back(inst_nop());
				}
				if (use_existing_desc) {
					insts.push_back(stride4_slots
								? inst_stg_e_u32(
									  desc_ur_base, r_ptr,
									  store_val_reg)
								: inst_stg_e_u8(
									  desc_ur_base, r_ptr,
									  store_val_reg));
				} else {
					insts.push_back(stride4_slots
								? inst_stg_e_u32_addr_ur(
									  r_ptr, ur,
									  store_val_reg, 0)
								: inst_stg_e_u8_addr_ur(
									  r_ptr, ur,
									  store_val_reg, 0));
				}
			}

		const size_t label_skip_store = insts.size();
			if (bra_skip_store) {
				const int64_t delta =
					int64_t(label_skip_store - *bra_skip_store) *
					(int64_t)SASS_INST_BYTES;
				insts[*bra_skip_store] =
					inst_bra_sm120_pred_p0(delta, bra_w1_fwd_override);
			}

			const size_t label_skip_all = insts.size();
				for (size_t pos : bra_skip_all_sites) {
					const int64_t delta =
						int64_t(label_skip_all - pos) *
						(int64_t)SASS_INST_BYTES;
					insts[pos] =
						inst_bra_sm120_pred_p0(delta, bra_w1_fwd_override);
				}

		// Restore original P0 state before replaying the EXIT instruction.
			if (spill_enable && spill_per_thread && !use_existing_desc) {
			const uint32_t spill_stride =
				std::max<uint32_t>(0x10u, cfg.reg255_thread_map_spill_stride_bytes);
			// Re-materialize spill address (r_ptr = idx * spill_stride).
			insts.push_back(
				inst_imad_wide_u32(r_ptr, r_idx, spill_stride, 0xff));
			insts.push_back(inst_umov_ur_imm(ur, spill_lo));
			insts.push_back(inst_umov_ur_imm(uint8_t(ur + 1u), spill_hi));
			for (int i = 0; i < 8; i++)
				insts.push_back(inst_nop());
			insts.push_back(inst_ldg_e_u32_addr_ur(
				/*rd=*/r_lane, /*addr_pair=*/r_ptr, /*ur_off=*/ur,
				/*imm8=*/0x0c));
			insts.push_back(inst_r2p_pr_mask(
				/*ra=*/r_lane, /*mask=*/sass_detour_pr_save_mask()));
		} else {
			insts.push_back(inst_r2p_pr_mask(
				/*ra=*/r_lane, /*mask=*/sass_detour_pr_save_mask()));
		}

		return insts;
	}

	static std::vector<SassInst>
	build_sm120_sampling_stub(const Sm120SamplingConfig &cfg,
				  const Sm120RegLayout &r, uint32_t tag,
				  const std::optional<SassInst> &ldcu_desc)
	{
	if (cfg.mode == Sm120SamplingConfig::Mode::SmidBitmap)
		return build_sm120_smid_bitmap_stub(cfg, r);
	(void)tag;
	if (cfg.mode == Sm120SamplingConfig::Mode::CtaSmid)
		return build_sm120_cta_smid_stub(cfg, r);
	if (cfg.mode == Sm120SamplingConfig::Mode::WarpMap)
		return build_sm120_warp_map_stub(cfg, r);
	if (cfg.mode == Sm120SamplingConfig::Mode::ThreadMap)
		return build_sm120_thread_map_stub(cfg, r, ldcu_desc);
	if (cfg.mode == Sm120SamplingConfig::Mode::PcMarker)
		return build_sm120_pc_marker_stub(cfg, r, tag, /*marker_off=*/0,
						  ldcu_desc);
	return {};
}

static std::vector<SassInst>
build_sm120_sampling_body_assuming_ur(const Sm120SamplingConfig &cfg,
				      const Sm120RegLayout &r, uint32_t tag)
{
	auto full = build_sm120_sampling_stub(cfg, r, tag, std::nullopt);
	// All current sampler stubs share the same shape:
	//   [save_ur_pair][load_desc_ur_ptr][...body...][restore_ur_pair]
	// Remove the UR prologue/epilogue so a higher-level stub can manage UR.
	if (full.size() < 6)
		return {};
	full.erase(full.begin(), full.begin() + 4);
	full.erase(full.end() - 2, full.end());
	return full;
}

static std::vector<SassInst>
build_sm120_identify_only_stub_no_regcount(const Sm120SamplingConfig &cfg,
						   uint8_t scratch_rd,
						   uint32_t func_id,
						   uint32_t image_id,
						   const std::optional<SassInst> &ldcu_desc,
						   const std::optional<SassInst> &ldg32_desc,
						   bool detour_selected)
{
	// Fallback for kernels where we cannot increase regcount (e.g. regcount=255):
	// build a tiny "identify" stub that writes `func_id` into `control.slots[0]`
	// (lane0-only), without relying on reserved scratch registers.
	//
	// This keeps the identify/tag closure functional even when we can't reserve
	// scratch GPRs via regcount patching.
	constexpr uint8_t kOffSlots =
		(uint8_t)(kSm120SassControlSlotsOffset & 0xffu);
	constexpr uint8_t kOffEnable = 0x08u;
	constexpr uint8_t kOffMode = 0x0cu;
	constexpr uint8_t kOffTargetFunc = 0x14u;
	constexpr uint8_t kOffReserved0 = 0x18u;
	constexpr uint8_t kOffReserved1 = 0x1cu;

	std::vector<SassInst> insts;
	insts.reserve(128);

	const bool do_control_reads =
		env_truthy("BPFTIME_CUDA_SASS_DETOUR_NO_REGCOUNT_CONTROL") &&
		env_truthy("BPFTIME_CUDA_SASS_DETOUR_NO_REGCOUNT_CONTROL_UNSAFE") &&
		env_truthy("BPFTIME_CUDA_SASS_DETOUR_CONTROL_READ_GLOBAL") &&
		detour_selected;

	const uint32_t lo = uint32_t(cfg.sample_buffer_device_ptr & 0xffffffffull);
	const uint32_t hi =
		uint32_t((cfg.sample_buffer_device_ptr >> 32) & 0xffffffffull);

	// NOTE: regcount=255 kernels can use *all* GPRs and may execute multiple
	// warps/CTAs concurrently. Any save/restore scheme that spills to a single
	// shared scratch address is race-prone and can corrupt registers.
	//
	// The "unsafe" control-read path below is kept opt-in for bring-up only.

	// Optional control reads (Identify/Target) without regcount patching.
	//
	// When enabled:
	// - Identify mode: slots[0] = func_id (lane0-only)
	// - Target mode (target_func_id == func_id): write a minimal record + marker
	//
	// NOTE: Use descriptor-form LDG for the control header. We observed the
	// UR-base `LD.E [R+UR+imm]` form returning zeros/stale values at kernel entry
	// on vendor kernels (flashattention), which breaks Identify/Target gating.
	if (do_control_reads && ldcu_desc) {
		// Preserve P0 for the original kernel. Even at function entry, some
		// cubin-only workloads appear sensitive to predicate state.
		insts.push_back(inst_p2r_pr_mask(/*rd=*/4, /*ra=*/0xff,
						 /*mask=*/sass_detour_pr_save_mask()));

		// Put the minimal sample record away from the control header (still within imm8).
		const uint8_t off_sample_u32 =
			uint8_t((cfg.buffer_data_offset + 0x20u) & 0xffu);

		// Use the kernel's own global-memory descriptor UR base (from its
		// prologue template), rather than loading into an arbitrary UR pair that
		// might be live at entry (regcount=255 kernels often use many UR registers).
		const uint8_t ctrl_desc_ur = uint8_t((ldcu_desc->w0 >> 16) & 0xffu);
		if ((ctrl_desc_ur & 1u) != 0 || ctrl_desc_ur > 62u) {
			// Should not happen for valid `LDCU.64` templates; fall back to the
			// write-only identify path.
			goto write_only_identify;
		}
		insts.push_back(*ldcu_desc);
		for (int i = 0; i < 8; i++)
			insts.push_back(inst_nop());

		auto mov_base_ptr = [&] {
			insts.push_back(inst_mov_imm_u32(/*rd=*/0, lo));
			insts.push_back(inst_mov_imm_u32(/*rd=*/1, hi));
		};
		auto ldg_wait = [&] {
			for (int i = 0; i < 8; i++)
				insts.push_back(inst_nop());
		};
		auto inst_ldg32_desc = [&](uint8_t rd, uint8_t addr_pair,
					   uint8_t ur_base) {
			SassInst i = ldg32_desc.value_or(
				inst_ldg_e_u32_desc_ur(rd, addr_pair, ur_base));
			i.w0 = set_u64_byte(i.w0, 2, rd);
			i.w0 = set_u64_byte(i.w0, 3, addr_pair);
			i.w0 = set_u64_byte(i.w0, 4, ur_base);
			return i;
		};

		mov_base_ptr();
		// addr = base + kOffEnable
		insts.push_back(inst_iadd64_ri(/*rd_pair=*/0, /*ra_pair=*/0, kOffEnable));
		insts.push_back(
			inst_ldg32_desc(/*rd=*/2, /*addr_pair=*/0, ctrl_desc_ur));
		ldg_wait();
		// addr += 4 => kOffMode
		insts.push_back(inst_iadd64_ri(/*rd_pair=*/0, /*ra_pair=*/0, 0x4u));
		insts.push_back(
			inst_ldg32_desc(/*rd=*/3, /*addr_pair=*/0, ctrl_desc_ur));
		ldg_wait();

		// P0 = (enable != 0)
		insts.push_back(inst_isetp_ne_u32_and_p0(/*ra=*/2));
		const size_t bra_if_enabled = insts.size();
		insts.push_back(SassInst {}); // @P0 BRA label_enabled
		const size_t bra_to_exit_disabled = insts.size();
		insts.push_back(SassInst {}); // BRA label_exit

		const size_t label_enabled = insts.size();

		// if (mode != Identify) goto not_identify;
		insts.push_back(inst_isetp_ne_u32_and_p0_imm(
			/*ra=*/3, (uint32_t)Sm120SassControlMode::Identify));
		const size_t bra_not_identify = insts.size();
		insts.push_back(SassInst {}); // @P0 BRA label_not_identify

		// Identify path: lane0-only store slots[0] = func_id
		mov_base_ptr();
		insts.push_back(inst_s2r(/*rd=*/2, SM120_SR_TID_X));
		for (int i = 0; i < 4; i++)
			insts.push_back(inst_nop());
		insts.push_back(inst_lop3_and_imm_u32(/*rd=*/2, /*ra=*/2, 0x1fu));
		insts.push_back(inst_isetp_ne_u32_and_p0(/*ra=*/2)); // P0 = (lane != 0)
		const size_t bra_skip_lane0 = insts.size();
		insts.push_back(SassInst {}); // @P0 BRA label_skip_store
		insts.push_back(inst_mov_imm_u32(/*rd=*/2, func_id));
		insts.push_back(inst_stg_e_u32_off(/*ur_base=*/ctrl_desc_ur,
						   /*addr_pair=*/0, /*rs=*/2,
						   /*imm8=*/kOffSlots));
		const size_t label_skip_store = insts.size();
		// BRA exit
		const size_t bra_after_identify = insts.size();
		insts.push_back(SassInst {}); // BRA label_exit

		const size_t label_not_identify = insts.size();
		// if (mode != Target) goto exit;
		insts.push_back(inst_isetp_ne_u32_and_p0_imm(
			/*ra=*/3, (uint32_t)Sm120SassControlMode::Target));
		const size_t bra_exit_if_not_target = insts.size();
		insts.push_back(SassInst {}); // @P0 BRA label_exit

		// addr += 8 => kOffTargetFunc
		insts.push_back(inst_iadd64_ri(/*rd_pair=*/0, /*ra_pair=*/0, 0x8u));
		insts.push_back(
			inst_ldg32_desc(/*rd=*/2, /*addr_pair=*/0, ctrl_desc_ur));
		ldg_wait();

		// P0 = (target != func_id) => skip
		insts.push_back(inst_isetp_ne_u32_and_p0_imm(/*ra=*/2, func_id));
		const size_t bra_exit_if_not_target_func = insts.size();
		insts.push_back(SassInst {}); // @P0 BRA label_exit

		// Target hit: lane0-only store a minimal sample record.
		mov_base_ptr();
		insts.push_back(inst_s2r(/*rd=*/2, SM120_SR_TID_X));
		for (int i = 0; i < 4; i++)
			insts.push_back(inst_nop());
		insts.push_back(inst_lop3_and_imm_u32(/*rd=*/2, /*ra=*/2, 0x1fu));
		insts.push_back(inst_isetp_ne_u32_and_p0(/*ra=*/2)); // P0 = (lane != 0)
		const size_t bra_skip_lane0_target = insts.size();
		insts.push_back(SassInst {}); // @P0 BRA label_exit
		insts.push_back(inst_s2r(/*rd=*/2, SM120_SR_VIRTUALSMID));
		// reserved0/reserved1: target-hit marker + func_id (debug)
		insts.push_back(inst_mov_imm_u32(/*rd=*/3, 0x54415247u)); // "TARG"
		insts.push_back(inst_stg_e_u32_off(/*ur_base=*/ctrl_desc_ur,
						   /*addr_pair=*/0, /*rs=*/3,
						   /*imm8=*/kOffReserved0));
		insts.push_back(inst_mov_imm_u32(/*rd=*/3, func_id));
		insts.push_back(inst_stg_e_u32_off(/*ur_base=*/ctrl_desc_ur,
						   /*addr_pair=*/0, /*rs=*/3,
						   /*imm8=*/kOffReserved1));
		// data[scratch+0x20] = smid_raw (u32), so host dump prints >=1 record.
		insts.push_back(inst_stg_e_u32_off(/*ur_base=*/ctrl_desc_ur,
						   /*addr_pair=*/0, /*rs=*/2,
						   /*imm8=*/off_sample_u32));

		const size_t label_exit = insts.size();

		auto patch_bra = [&](size_t at, size_t target, bool pred_p0) {
			const int64_t delta =
				int64_t(target) * (int64_t)SASS_INST_BYTES -
				int64_t(at) * (int64_t)SASS_INST_BYTES;
			auto b = pred_p0 ? encode_bra_sm120_pred_p0(delta)
					 : encode_bra_sm120_unpred(delta);
			assert(b.has_value());
			insts[at] = SassInst { b->w0, b->w1 };
		};
		patch_bra(bra_if_enabled, label_enabled, /*pred_p0=*/true);
		patch_bra(bra_to_exit_disabled, label_exit, /*pred_p0=*/false);
		patch_bra(bra_not_identify, label_not_identify, /*pred_p0=*/true);
		patch_bra(bra_skip_lane0, label_skip_store, /*pred_p0=*/true);
		patch_bra(bra_after_identify, label_exit, /*pred_p0=*/false);
		patch_bra(bra_exit_if_not_target, label_exit, /*pred_p0=*/true);
		patch_bra(bra_exit_if_not_target_func, label_exit, /*pred_p0=*/true);
		patch_bra(bra_skip_lane0_target, label_exit, /*pred_p0=*/true);
		insts.push_back(inst_r2p_pr_mask(/*ra=*/4,
						 /*mask=*/sass_detour_pr_save_mask()));
		return insts;
	}

write_only_identify:
	// Write-only Identify fallback:
	// - store slots[0] = func_id (for identify closure)
	// - also write one minimal record into the sampler data region so host dump
	//   produces >= 1 record line even on regcount=255 kernels.
	{
		// Use UR-base addressing `[RZ + UR + imm]` with the configured UR pair
		// (default UR62/UR63) to minimize interference with compiler-selected
		// UR registers (UR4/UR5 are commonly used in prologues).
		//
		// We intentionally avoid:
		// - descriptor-form stores: some vendor kernels appear to use a
		//   load-only global descriptor (LDCU.64), and STG.E via that descriptor
		//   can silently fail in practice (identify slots remain zero).
		// - lane0 predicates: touching P0 and/or relying on predicated STG at
		//   entry is brittle on regcount-saturated kernels; for bring-up we
		//   allow all lanes to store the same value.
		const uint8_t ur = cfg.desc_ur;
		const uint8_t ur_save =
			(ur >= 2u) ? uint8_t(ur - 2u) : uint8_t(ur + 2u);
		// Save/restore the selected UR pair: some vendor kernels can use high UR
		// numbers, and clobbering them at entry can destabilize workloads.
		insts.push_back(inst_umov_ur_ur(ur_save, ur));
		insts.push_back(
			inst_umov_ur_ur(uint8_t(ur_save + 1u), uint8_t(ur + 1u)));
		insts.push_back(inst_umov_ur_imm(ur, lo));
		insts.push_back(inst_umov_ur_imm(uint8_t(ur + 1u), hi));
		const int ur_wait_nops = [] {
			if (auto v = env_u32("BPFTIME_CUDA_SASS_DETOUR_CONTROL_UR_WAIT_NOPS"))
				return int(std::min<uint32_t>(*v, 256u));
			return 8;
		}();
		for (int i = 0; i < ur_wait_nops; i++)
			insts.push_back(inst_nop());

		// Store func_id into control slots, and also write one data record so
		// the JSONL dump emits at least one record line (data slots are
		// initialized to 0xffffffff on the host).
		//
			// NOTE: keep using the HFMA2 immediate materialization pattern observed
			// in ptxas SM120 output. MOV immediate is not consistently emitted by
			// ptxas and may not be stable across cubin-only vendor kernels.
			insts.push_back(inst_hfma2_imm_u32(/*rd=*/scratch_rd, func_id));
			insts.push_back(inst_stg_e_u32_addr_ur(/*addr_pair=*/0xff, /*ur_off=*/ur,
							      /*rs=*/scratch_rd,
							      /*imm8=*/kOffSlots));
			// control.reserved0 = image_id (host-side disambiguation).
			insts.push_back(inst_hfma2_imm_u32(/*rd=*/scratch_rd, image_id));
			insts.push_back(inst_stg_e_u32_addr_ur(/*addr_pair=*/0xff, /*ur_off=*/ur,
							      /*rs=*/scratch_rd,
							      /*imm8=*/kOffReserved0));
			// data[0] = smid_raw (warp_map) so the dumped record carries a real SMID.
			insts.push_back(inst_s2r(/*rd=*/scratch_rd, SM120_SR_VIRTUALSMID));
			for (int i = 0; i < 4; i++)
				insts.push_back(inst_nop());
		insts.push_back(inst_stg_e_u32_addr_ur(
			/*addr_pair=*/0xff, /*ur_off=*/ur, /*rs=*/scratch_rd,
			/*imm8=*/uint8_t(cfg.buffer_data_offset & 0xffu)));
		insts.push_back(inst_umov_ur_ur(ur, ur_save));
		insts.push_back(
			inst_umov_ur_ur(uint8_t(ur + 1u), uint8_t(ur_save + 1u)));
		return insts;
	}
}

static std::vector<SassInst>
	build_sm120_controlled_sampling_stub(const Sm120SamplingConfig &cfg,
					     const Sm120RegLayout &r, uint32_t tag,
					     uint32_t func_id, uint32_t image_id,
					     bool detour_selected,
					     const std::optional<SassInst> &ldcu_desc,
					     const std::optional<SassInst> &ldg64_desc)
			{
	// Control/header offsets are fixed (imm8 in STG).
	constexpr uint8_t kOffSlots =
		(uint8_t)(kSm120SassControlSlotsOffset & 0xffu);
	constexpr uint8_t kOffEnable = 0x08u;
	constexpr uint8_t kOffMode = 0x0cu;
	constexpr uint8_t kOffTargetFunc = 0x14u;
	constexpr uint8_t kOffReserved0 = 0x18u;
		const bool dbg_store_ctrl =
			env_truthy("BPFTIME_CUDA_SASS_DETOUR_CONTROL_DEBUG_STORE_CTRL");
		const bool dbg_device_reads =
			env_truthy("BPFTIME_CUDA_SASS_DETOUR_CONTROL_DEBUG_DEVICE_READS");
		const bool dbg_target_hit =
			env_truthy("BPFTIME_CUDA_SASS_DETOUR_CONTROL_DEBUG_TARGET_HIT");
		const bool control_use_desc =
			!env_truthy("BPFTIME_CUDA_SASS_DETOUR_CONTROL_USE_UR");
	const bool control_read_global =
		env_truthy("BPFTIME_CUDA_SASS_DETOUR_CONTROL_READ_GLOBAL");
	// Our UR-base `LD.E [R+UR+imm]` encoding is still bring-up-only; avoid using
	// it for read-control in real workloads. If the user forces UR mode, fall
	// back to write-only identify (no reads) to keep kernels stable.
	//
	// IMPORTANT: even if read-control works on a specific kernel, it is not
	// safe to enable it globally when we are in “patch-all” mode (or a
	// patch-all fallback). Only the user-selected detour targets should perform
	// global loads at function entry.
	const bool do_control_reads =
		detour_selected && control_use_desc && control_read_global;
	const int addr_wait_nops = [] {
		if (auto v = env_u32("BPFTIME_CUDA_SASS_DETOUR_CONTROL_ADDR_WAIT_NOPS"))
			return int(std::min<uint32_t>(*v, 256u));
		return 8;
	}();
	const int ldg_wait_nops = [] {
		if (auto v = env_u32("BPFTIME_CUDA_SASS_DETOUR_CONTROL_LDG_WAIT_NOPS"))
			return int(std::min<uint32_t>(*v, 256u));
		// On some cubin-only kernels, consuming an LDG result immediately can
		// observe stale values (scheduler/control-word mismatch). Use a small
		// fixed gap to keep control decisions deterministic.
		return 8;
	}();

	std::vector<SassInst> insts;
	insts.reserve(64);
	auto addr_wait = [&] {
		for (int i = 0; i < addr_wait_nops; i++)
			insts.push_back(inst_nop());
	};
	auto ldg_wait = [&] {
		for (int i = 0; i < ldg_wait_nops; i++)
			insts.push_back(inst_nop());
	};
	std::optional<std::pair<size_t, size_t>> bra_lane0_skip_store;

	const uint8_t ur = cfg.desc_ur;
	const uint8_t ur_save = (ur >= 2u) ? uint8_t(ur - 2u) : uint8_t(ur + 2u);

	const uint32_t lo = uint32_t(cfg.sample_buffer_device_ptr & 0xffffffffull);
	const uint32_t hi =
		uint32_t((cfg.sample_buffer_device_ptr >> 32) & 0xffffffffull);

	// Prologue: preserve UR pair, then load the base pointer.
	insts.push_back(inst_umov_ur_ur(ur_save, ur));
	insts.push_back(inst_umov_ur_ur(uint8_t(ur_save + 1u), uint8_t(ur + 1u)));
		insts.push_back(inst_umov_ur_imm(ur, lo));
		insts.push_back(inst_umov_ur_imm(uint8_t(ur + 1u), hi));
		// UR writes can take a few cycles to become visible to memory ops at the
		// function entrypoint. If we issue `LD/ST [R+UR+imm]` immediately after
		// `UMOV URx, imm`, we can observe stale UR values and trigger illegal
		// addresses on real workloads.
		const int ur_wait_nops = [] {
			if (auto v = env_u32("BPFTIME_CUDA_SASS_DETOUR_CONTROL_UR_WAIT_NOPS"))
				return int(std::min<uint32_t>(*v, 256u));
			return 8;
		}();
			for (int i = 0; i < ur_wait_nops; i++)
				insts.push_back(inst_nop());

				// Debug bring-up: prove the controlled stub is actually executing by
				// writing a recognizable marker into `control.reserved0` (offset 0x18)
				// unconditionally.
				//
				// Do NOT write into `control.slots[]` here: slots are used by Identify
				// mode for func_id collection, and polluting them makes the closure
				// selection ambiguous.
				if (env_truthy("BPFTIME_CUDA_SASS_DETOUR_CONTROL_DEBUG_MARKER")) {
					insts.push_back(inst_mov_imm_u32(r.r_ptr, 0u));
					insts.push_back(inst_mov_imm_u32(uint8_t(r.r_ptr + 1u), 0u));
					insts.push_back(inst_hfma2_imm_u32(r.r_tmp, 0xfeedc0deu));
					insts.push_back(inst_stg_e_u32_addr_ur(
						r.r_ptr, ur, r.r_tmp,
						/*imm8=*/0x18u));
				}

			// Save P0 for the original kernel.
			insts.push_back(inst_p2r_pr_mask(r.r_pr, 0xff,
							 sass_detour_pr_save_mask()));
		// Debug bring-up: skip all control header loads/stores.
		if (env_truthy("BPFTIME_CUDA_SASS_DETOUR_CONTROL_NO_LOADS")) {
			insts.push_back(inst_r2p_pr_mask(r.r_pr,
							 sass_detour_pr_save_mask()));
			insts.push_back(inst_umov_ur_ur(ur, ur_save));
			insts.push_back(
				inst_umov_ur_ur(uint8_t(ur + 1u), uint8_t(ur_save + 1u)));
			return insts;
		}

	// Default path (stable on current SM120 bring-up): write-only identify.
	// We avoid any global-memory loads from inside the detoured stub (we've
	// observed `CUDA_ERROR_ILLEGAL_ADDRESS` on `LDG/LD.*` even when the address
	// is valid and host-side memcpy can read it). This enables the name->func_id
	// mapping to be learned and persisted across runs, at the cost of requiring
	// a second run for "target-only" sampling.
			if (!do_control_reads) {
				const bool store_all_lanes = env_truthy(
					"BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_STORE_ALL_LANES");
			// Optional bring-up mode: ignore lane0-only gating and store from
			// all lanes. This helps debug predicate/entry-state assumptions.
				if (!store_all_lanes) {
					// P0 = (lane_id != 0)
					insts.push_back(inst_s2r(r.r_smid, SM120_SR_LANEID));
					// S2R latency at entry can be non-trivial on some cubin-only
					// kernels; add a tiny gap to avoid consuming stale lane_id.
					const int lane_wait_nops = [] {
						if (auto v = env_u32(
							    "BPFTIME_CUDA_SASS_DETOUR_LANEID_WAIT_NOPS"))
							return int(std::min<uint32_t>(*v, 32u));
						return 4;
					}();
					for (int i = 0; i < lane_wait_nops; i++)
						insts.push_back(inst_nop());
					// Prefer the LOP3-based predicate materialization observed in SM120
					// kernels; `ISETP` predicate setup has shown entry sensitivity in
					// some stripped/cubin-only workloads.
					insts.push_back(inst_lop3_lane_ne0_to_p0(r.r_smid));
				}
			// r_ptr(.64) = (smid & 7) * 4
			insts.push_back(inst_s2r(r.r_smid, SM120_SR_VIRTUALSMID));
			insts.push_back(inst_lop3_and_imm_u32(
				r.r_smid, r.r_smid, kSm120SassControlSlotsCount - 1u));
			insts.push_back(inst_imad_wide_u32(r.r_ptr, r.r_smid, 0x4, 0xff));
				// Store func_id into slots[smid].
				insts.push_back(inst_hfma2_imm_u32(r.r_tmp, func_id));
				if (store_all_lanes) {
					insts.push_back(inst_stg_e_u32_addr_ur(
						r.r_ptr, ur, r.r_tmp, kOffSlots));
				} else {
				// Avoid predicated STG for lane0 gating: on some kernels the
				// `@!P0 STG` form can be unreliable at entry. Use a predicated BRA
				// to skip the store for non-lane0 threads.
				const size_t bra_skip = insts.size();
				insts.push_back(SassInst {}); // @P0 BRA label_skip_store
				const size_t store_at = insts.size();
				insts.push_back(inst_stg_e_u32_addr_ur(
					r.r_ptr, ur, r.r_tmp, kOffSlots));
				const size_t label_skip_store = insts.size();
				(void)store_at;
				// Patch branch delta now (local to this early-return stub).
				const int64_t delta =
					int64_t(label_skip_store) * (int64_t)SASS_INST_BYTES -
					int64_t(bra_skip) * (int64_t)SASS_INST_BYTES;
				insts[bra_skip] = inst_bra_sm120_pred_p0(delta);
			}
			// Restore P0.
			insts.push_back(inst_r2p_pr_mask(r.r_pr,
							 sass_detour_pr_save_mask()));
			// Restore UR pair and exit.
			insts.push_back(inst_umov_ur_ur(ur, ur_save));
			insts.push_back(inst_umov_ur_ur(uint8_t(ur + 1u),
							uint8_t(ur_save + 1u)));
			return insts;
		}

	// r_ptr(.64) is used as an address scratch for control header access.
	// - `control_use_desc`: materialize an absolute address in r_ptr and use
	//   `LDG.E desc[UR4][r_ptr.64]` (matches nvcc's SM120 global load pattern).
	// - else: use `[r_ptr.64 + UR + imm16]` addressing with `LD.E`/`STG.E`.
	insts.push_back(inst_mov_imm_u32(r.r_ptr, 0u));
	insts.push_back(inst_mov_imm_u32(uint8_t(r.r_ptr + 1u), 0u));

		if (control_use_desc) {
			// Load a global-memory descriptor into UR4/UR5. Prefer a template copied
			// from the target kernel's own prologue (hetGPU-style) to avoid depending
			// on a single hardcoded scheduling/control word.
			{
				insts.push_back(
					ldcu_desc.value_or(inst_ldcu64_desc_ur(4)));
			}
			// On SM120, UR write hazards can be subtle. Add explicit UR-to-UR moves to
			// create a dependency chain before the first `LDG/STG desc[...]`.
			insts.push_back(inst_umov_ur_ur(4, 4));
			insts.push_back(inst_umov_ur_ur(5, 5));
			// Be conservative about UR4/UR5 readiness at the entry point: give the
			// descriptor load a few cycles before issuing LDG/STG.
			const int desc_wait_nops = [] {
				if (auto v = env_u32(
					    "BPFTIME_CUDA_SASS_DETOUR_CONTROL_DESC_WAIT_NOPS"))
					return int(std::min<uint32_t>(*v, 256u));
				return 8;
			}();
			for (int i = 0; i < desc_wait_nops; i++)
				insts.push_back(inst_nop());
		}
		if (env_truthy("BPFTIME_CUDA_SASS_DETOUR_CONTROL_NO_LDG")) {
			insts.push_back(inst_r2p_pr_mask(r.r_pr,
							 sass_detour_pr_save_mask()));
			insts.push_back(inst_umov_ur_ur(ur, ur_save));
			insts.push_back(
				inst_umov_ur_ur(uint8_t(ur + 1u), uint8_t(ur_save + 1u)));
			return insts;
		}

	auto inst_ldg64_desc = [&](uint8_t rd_pair, uint8_t addr_pair,
				   uint8_t ur_base) {
		SassInst i = ldg64_desc.value_or(
			inst_ldg_e_u64_desc_ur(rd_pair, addr_pair, ur_base));
		i.w0 = set_u64_byte(i.w0, 2, rd_pair);
		i.w0 = set_u64_byte(i.w0, 3, addr_pair);
		i.w0 = set_u64_byte(i.w0, 4, ur_base);
		return i;
	};
	auto load_ctrl_u64_pair = [&](uint8_t off_imm8) {
		// Use `LDG.E.64` for control reads:
		// - off=0x08 loads {enable, mode}
		// - off=0x10 loads {epoch, target_func_id}
		if (control_use_desc) {
			const uint32_t alo = lo + uint32_t(off_imm8);
			const uint32_t ahi = hi + uint32_t(alo < lo ? 1u : 0u);
			insts.push_back(inst_mov_imm_u32(r.r_ptr, alo));
			insts.push_back(inst_mov_imm_u32(uint8_t(r.r_ptr + 1u), ahi));
			addr_wait();
			// Dest pair: r_ctrl (low/high). Dedicated scratch pair reserved in the
			// regcount patch for controlled stubs.
			insts.push_back(inst_ldg64_desc(r.r_ctrl, r.r_ptr, 4));
			ldg_wait();
			return;
		}
		// Base address already lives in `URx/URx+1`; use imm16 load (bring-up only).
		insts.push_back(inst_ldg_e_u32_addr_ur(r.r_ctrl, r.r_ptr, ur, off_imm8));
		ldg_wait();
	};

		// control.enable
		load_ctrl_u64_pair(kOffEnable);
		if (dbg_device_reads) {
			// reserved0 = device-read mode
			insts.push_back(inst_mov_imm_u32(r.r_ptr, 0u));
			insts.push_back(inst_mov_imm_u32(uint8_t(r.r_ptr + 1u), 0u));
			insts.push_back(inst_stg_e_u32_addr_ur(
				r.r_ptr, ur, uint8_t(r.r_ctrl + 1u), /*imm8=*/0x18u));
		}
		if (dbg_store_ctrl) {
			// slots[0] = enable (debug)
			if (control_use_desc) {
				const uint32_t alo = lo + uint32_t(kOffSlots);
				const uint32_t ahi = hi + uint32_t(alo < lo ? 1u : 0u);
			insts.push_back(inst_mov_imm_u32(r.r_ptr, alo));
			insts.push_back(inst_mov_imm_u32(uint8_t(r.r_ptr + 1u), ahi));
			addr_wait();
			insts.push_back(inst_stg_e_u32_off(/*ur_base=*/4, r.r_ptr,
							   r.r_ctrl, 0));
		} else {
			insts.push_back(inst_stg_e_u32_addr_ur(r.r_ptr, ur, r.r_ctrl,
							      kOffSlots));
		}
	}
	// P0 = (enable != 0)
	insts.push_back(inst_isetp_ne_u32_and_p0(r.r_ctrl));
	// Debug bring-up: only touch `enable` and then exit.
		if (env_truthy("BPFTIME_CUDA_SASS_DETOUR_CONTROL_ONLY_ENABLE")) {
			insts.push_back(inst_r2p_pr_mask(r.r_pr,
							 sass_detour_pr_save_mask()));
			insts.push_back(inst_umov_ur_ur(ur, ur_save));
			insts.push_back(inst_umov_ur_ur(uint8_t(ur + 1u), uint8_t(ur_save + 1u)));
			return insts;
		}
	const size_t bra_if_enabled = insts.size();
	insts.push_back(SassInst {}); // @P0 BRA label_enabled
	const size_t bra_to_exit_disabled = insts.size();
	insts.push_back(SassInst {}); // BRA label_exit_common

	const size_t label_enabled = insts.size();

	// control.mode is loaded together with enable via the u64 at offset 0x08.
	if (dbg_store_ctrl) {
		// slots[1] = mode (debug)
		const uint8_t off = uint8_t(kOffSlots + 0x4u);
		if (control_use_desc) {
			const uint32_t alo = lo + uint32_t(off);
			const uint32_t ahi = hi + uint32_t(alo < lo ? 1u : 0u);
			insts.push_back(inst_mov_imm_u32(r.r_ptr, alo));
			insts.push_back(inst_mov_imm_u32(uint8_t(r.r_ptr + 1u), ahi));
			addr_wait();
			// r_ctrl+1 holds mode (upper 32 bits of the 0x08 u64).
			insts.push_back(inst_stg_e_u32_off(/*ur_base=*/4, r.r_ptr,
							   uint8_t(r.r_ctrl + 1u), 0));
		} else {
			insts.push_back(inst_stg_e_u32_addr_ur(
				r.r_ptr, ur, uint8_t(r.r_ctrl + 1u), off));
		}
	}

	// if (mode == Identify) { slots[smid] = func_id; goto exit; }
	insts.push_back(inst_isetp_ne_u32_and_p0_imm(
		uint8_t(r.r_ctrl + 1u),
		(uint32_t)Sm120SassControlMode::Identify)); // P0 = (mode != Identify)
	const size_t bra_not_identify = insts.size();
	insts.push_back(SassInst {}); // @P0 BRA label_not_identify

	// Identify path.
	// Gate identify store to lane0 to reduce write density.
	// Avoid relying on SR_LANEID at function entry. Some cubin-only kernels can
	// observe surprising SR_LANEID behavior before their own prologues.
	// Derive lane_id from tid_x: lane_id = tid_x & 31.
	insts.push_back(inst_s2r(r.r_tid, SM120_SR_TID_X));
	{
		const int s2r_wait_nops = [] {
			if (auto v = env_u32("BPFTIME_CUDA_SASS_DETOUR_S2R_WAIT_NOPS"))
				return int(std::min<uint32_t>(*v, 64u));
			return 4;
		}();
		for (int i = 0; i < s2r_wait_nops; i++)
			insts.push_back(inst_nop());
	}
	insts.push_back(inst_lop3_and_imm_u32(r.r_lane, r.r_tid, 0x1fu));
	insts.push_back(inst_isetp_ne_u32_and_p0(r.r_lane)); // P0 = (lane != 0)
	insts.push_back(inst_s2r(r.r_smid, SM120_SR_VIRTUALSMID));
	insts.push_back(inst_lop3_and_imm_u32(
		r.r_smid, r.r_smid, kSm120SassControlSlotsCount - 1u));
			if (!env_truthy("BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_NO_STORE")) {
				if (control_use_desc) {
				const uint32_t alo = lo + uint32_t(kOffSlots);
				const uint32_t ahi = hi + uint32_t(alo < lo ? 1u : 0u);
				insts.push_back(inst_mov_imm_u32(r.r_ptr, alo));
				insts.push_back(inst_mov_imm_u32(uint8_t(r.r_ptr + 1u), ahi));
				addr_wait();
				// addr = base + slots_off + (smid&7)*4
				insts.push_back(inst_imad_wide_u32(
					r.r_ptr, r.r_smid, 0x4, r.r_ptr));
				addr_wait();
					// Only lane0 stores (avoid relying on predicated STG encoding).
					const size_t bra_skip_lane0 = insts.size();
					insts.push_back(SassInst {}); // @P0 BRA label_skip_lane0_store
					// Identify closure disambiguation: record the current code object's
					// image_id into the control header (reserved0).
					insts.push_back(inst_hfma2_imm_u32(r.r_tmp, image_id));
					addr_wait();
					insts.push_back(inst_stg_e_u32_addr_ur(
						/*addr_pair=*/0xff, ur, r.r_tmp, kOffReserved0));
					insts.push_back(inst_mov_imm_u32(r.r_tmp, func_id));
					// Similar to address materialization, we observed occasional stale
					// values when a just-materialized GPR is consumed by STG/LDG at
					// function entry. Leave a small gap before the store.
				addr_wait();
				// Use the "+imm" encoding with imm=0; this avoids ambiguities in
				// nvdisasm decoding between STG variants and keeps fields stable.
				insts.push_back(
					inst_stg_e_u32_off(/*ur_base=*/4, r.r_ptr, r.r_tmp, 0));
				const size_t label_skip_lane0_store = insts.size();
				bra_lane0_skip_store =
					std::make_pair(bra_skip_lane0, label_skip_lane0_store);
				} else {
					// r_ptr(.64) = (smid&7) * 4 (offset into slots[])
					insts.push_back(
						inst_imad_wide_u32(r.r_ptr, r.r_smid, 0x4, 0xff));
						// Only lane0 stores. Prefer predicated BRA over predicated STG for
						// better reliability on cubin-only kernels.
						const size_t bra_skip_lane0 = insts.size();
						insts.push_back(SassInst {}); // @P0 BRA label_skip_lane0_store
						// Identify closure disambiguation: record image_id into reserved0.
						insts.push_back(inst_hfma2_imm_u32(r.r_tmp, image_id));
						addr_wait();
						insts.push_back(inst_stg_e_u32_addr_ur(
							/*addr_pair=*/0xff, ur, r.r_tmp, kOffReserved0));
							insts.push_back(inst_hfma2_imm_u32(r.r_tmp, func_id));
							addr_wait();
							insts.push_back(inst_stg_e_u32_addr_ur(
								r.r_ptr, ur, r.r_tmp, kOffSlots));
					const size_t label_skip_lane0_store = insts.size();
					bra_lane0_skip_store =
						std::make_pair(bra_skip_lane0,
							       label_skip_lane0_store);
			}
		}
	const size_t bra_after_identify = insts.size();
	insts.push_back(SassInst {}); // BRA label_exit_common

	const size_t label_not_identify = insts.size();

	// if (mode != Target) goto restore_only;
	insts.push_back(inst_isetp_ne_u32_and_p0_imm(
		uint8_t(r.r_ctrl + 1u),
		(uint32_t)Sm120SassControlMode::Target)); // P0 = (mode != Target)
	const size_t bra_exit_if_not_target_mode = insts.size();
	insts.push_back(SassInst {}); // @P0 BRA label_exit_common

		// if (target_func_id != func_id) goto restore_only;
		load_ctrl_u64_pair(kOffTargetFunc - 0x4u /* 0x10: epoch+target */);
		if (dbg_device_reads) {
			// reserved1 = device-read target_func_id
			insts.push_back(inst_mov_imm_u32(r.r_ptr, 0u));
			insts.push_back(inst_mov_imm_u32(uint8_t(r.r_ptr + 1u), 0u));
			insts.push_back(inst_stg_e_u32_addr_ur(
				r.r_ptr, ur, uint8_t(r.r_ctrl + 1u), /*imm8=*/0x1cu));
		}
	if (dbg_store_ctrl) {
		// slots[2] = target_func_id (debug)
		const uint8_t off = uint8_t(kOffSlots + 0x8u);
		if (control_use_desc) {
			const uint32_t alo = lo + uint32_t(off);
			const uint32_t ahi = hi + uint32_t(alo < lo ? 1u : 0u);
			insts.push_back(inst_mov_imm_u32(r.r_ptr, alo));
			insts.push_back(inst_mov_imm_u32(uint8_t(r.r_ptr + 1u), ahi));
			addr_wait();
			// r_ctrl+1 holds target_func_id (upper 32 bits of the 0x10 u64).
			insts.push_back(inst_stg_e_u32_off(/*ur_base=*/4, r.r_ptr,
							   uint8_t(r.r_ctrl + 1u), 0));
		} else {
			insts.push_back(
				inst_stg_e_u32_addr_ur(r.r_ptr, ur, uint8_t(r.r_ctrl + 1u),
						       off));
		}
	}
	insts.push_back(inst_isetp_ne_u32_and_p0_imm(uint8_t(r.r_ctrl + 1u), func_id));
	const size_t bra_exit_if_not_target_func = insts.size();
	insts.push_back(SassInst {}); // @P0 BRA label_exit_common

		// Target hit: execute sampling body.
		if (dbg_target_hit) {
			// reserved0 = "TARG", reserved1 = func_id
			insts.push_back(inst_mov_imm_u32(r.r_ptr, 0u));
			insts.push_back(inst_mov_imm_u32(uint8_t(r.r_ptr + 1u), 0u));
			insts.push_back(inst_hfma2_imm_u32(r.r_tmp, 0x54415247u));
			insts.push_back(inst_stg_e_u32_addr_ur(
				r.r_ptr, ur, r.r_tmp, /*imm8=*/0x18u));
				insts.push_back(inst_hfma2_imm_u32(r.r_tmp, func_id));
				insts.push_back(inst_stg_e_u32_addr_ur(
					r.r_ptr, ur, r.r_tmp, /*imm8=*/0x1cu));
			}
		{
		// Some sampling modes use a different prologue shape (e.g. materialize a
		// base pointer in GPRs and/or load a descriptor into `cfg.desc_ur`).
		// Do not try to strip a fixed UR-save/restore wrapper in those cases;
		// embed the full stub.
		if ((cfg.mode == Sm120SamplingConfig::Mode::ThreadMap &&
		     cfg.thread_map_device) ||
		    (cfg.mode == Sm120SamplingConfig::Mode::PcMarker)) {
			// The controlled stub uses `ldcu_desc` (when present) for UR4/UR5
			// control/header loads. ThreadMap(device) sampling, however, uses
			// `cfg.desc_ur` as the descriptor base for its STG/LDG instructions.
			// If we reuse the UR4 template here, we'd end up loading the descriptor
			// into UR4 while emitting stores that reference `cfg.desc_ur` (UR62 by
			// default), which breaks sampling. Patch the template destination to
			// `cfg.desc_ur` before handing it to the sampling stub.
			std::optional<SassInst> sampling_ldcu_desc = ldcu_desc;
			if (sampling_ldcu_desc) {
				sampling_ldcu_desc->w0 =
					set_u64_byte(sampling_ldcu_desc->w0, 2, cfg.desc_ur);
			}
			auto full = build_sm120_sampling_stub(cfg, r, tag, sampling_ldcu_desc);
			insts.insert(insts.end(), full.begin(), full.end());
		} else {
			auto body = build_sm120_sampling_body_assuming_ur(cfg, r, tag);
			insts.insert(insts.end(), body.begin(), body.end());
		}
	}
	// Fallthrough to exit_common after body.
	const size_t label_exit_common = insts.size();

		// Epilogue: restore P0 then UR pair.
		insts.push_back(inst_r2p_pr_mask(r.r_pr,
						 sass_detour_pr_save_mask()));
		insts.push_back(inst_umov_ur_ur(ur, ur_save));
		insts.push_back(inst_umov_ur_ur(uint8_t(ur + 1u), uint8_t(ur_save + 1u)));

		// Patch branch deltas now that labels are known.
	auto patch_bra = [&](size_t at, size_t target, bool pred_p0) {
		const int64_t delta =
			int64_t(target) * (int64_t)SASS_INST_BYTES -
			int64_t(at) * (int64_t)SASS_INST_BYTES;
		auto b = pred_p0 ? encode_bra_sm120_pred_p0(delta)
				 : encode_bra_sm120_unpred(delta);
		assert(b.has_value());
		insts[at] = SassInst { b->w0, b->w1 };
	};
		patch_bra(bra_if_enabled, label_enabled, /*pred_p0=*/true);
		patch_bra(bra_to_exit_disabled, label_exit_common, /*pred_p0=*/false);
		patch_bra(bra_not_identify, label_not_identify, /*pred_p0=*/true);
		patch_bra(bra_after_identify, label_exit_common, /*pred_p0=*/false);
		patch_bra(bra_exit_if_not_target_mode, label_exit_common, /*pred_p0=*/true);
		patch_bra(bra_exit_if_not_target_func, label_exit_common, /*pred_p0=*/true);
		if (bra_lane0_skip_store)
			patch_bra(bra_lane0_skip_store->first,
				  bra_lane0_skip_store->second, /*pred_p0=*/true);
		(void)label_exit_common;

	return insts;
}
} // namespace

std::optional<int> infer_sm_version_from_elf(std::span<const uint8_t> elf)
{
	Elf64_Ehdr ehdr {};
	if (!read_pod(elf, 0, ehdr))
		return std::nullopt;
	if (std::memcmp(ehdr.e_ident, ELF_MAGIC, sizeof(ELF_MAGIC)) != 0)
		return std::nullopt;
	if (ehdr.e_ident[4] != 2 /* ELFCLASS64 */ ||
	    ehdr.e_ident[5] != 1 /* ELFDATA2LSB */)
		return std::nullopt;

	const uint32_t f = ehdr.e_flags;
	// Observed patterns (nvptxcompiler):
	// - sm_70..sm_90: f == (sm << 16) | (0x05 << 8) | sm
	// - sm_100+:      f == (0x06 << 24) | (sm << 8) | 0x02
	const uint32_t lo = f & 0xffu;
	const uint32_t hi = (f >> 16) & 0xffu;
	const uint32_t mid = (f >> 8) & 0xffu;
	if (mid == 0x05u && lo != 0 && lo == hi)
		return static_cast<int>(lo);

	if (((f >> 24) & 0xffu) == 0x06u && (f & 0xffu) == 0x02u &&
	    ((f >> 16) & 0xffu) == 0x00u && mid != 0)
		return static_cast<int>(mid);

	// CUDA 12.8+ sometimes varies the low 16 bits for SM100+; still treat
	// bits[15:8] as the SM version when the top byte indicates “new format”.
	if (((f >> 24) & 0xffu) == 0x06u && mid >= 100)
		return static_cast<int>(mid);

	return std::nullopt;
}

		std::optional<DetourResult> apply_elf_text_detours_sm120(
			std::vector<uint8_t> &elf_bytes, std::string_view section_name_filter,
			std::string_view sample_section_filter,
			const Sm120SamplingConfig *sampling_cfg,
			const std::unordered_set<uint32_t> *extra_filter_func_ids,
			bool patch_all_is_fallback)
		{
		DetourResult result;
		Elf64_Ehdr ehdr {};
		if (!read_pod(std::span<const uint8_t>(elf_bytes.data(), elf_bytes.size()),
			      0, ehdr)) {
			result.reason = "not an ELF64 header";
			return result;
		}
		if (std::memcmp(ehdr.e_ident, ELF_MAGIC, sizeof(ELF_MAGIC)) != 0) {
			result.reason = "ELF magic mismatch";
			return result;
		}
		if (ehdr.e_ident[4] != 2 /* ELFCLASS64 */ ||
		    ehdr.e_ident[5] != 1 /* ELFDATA2LSB */) {
			result.reason = "unsupported ELF class/endianness";
			return result;
		}
		if (ehdr.e_shoff == 0 || ehdr.e_shentsize == 0 || ehdr.e_shnum == 0 ||
		    ehdr.e_shentsize != sizeof(Elf64_Shdr)) {
			result.reason = "missing section headers";
			return result;
		}

	auto sm = infer_sm_version_from_elf(
		std::span<const uint8_t>(elf_bytes.data(), elf_bytes.size()));
	if (!sm || *sm != 120) {
		result.reason = "SM != 120";
		return result;
	}

			auto shstrtab =
				get_shstrtab(std::span<const uint8_t>(elf_bytes.data(),
								      elf_bytes.size()),
					     ehdr);
		if (!shstrtab) {
			result.reason = "missing shstrtab";
			return result;
		}

	size_t shoff = static_cast<size_t>(ehdr.e_shoff);
	const size_t shnum = static_cast<size_t>(ehdr.e_shnum);

		std::vector<Elf64_Shdr> shdrs(shnum);
			for (size_t i = 0; i < shnum; i++) {
				if (!read_pod(std::span<const uint8_t>(elf_bytes.data(),
								      elf_bytes.size()),
					      shoff + i * sizeof(Elf64_Shdr), shdrs[i])) {
					result.reason = "unable to read section headers";
					return result;
				}
			}

		std::vector<Elf64_Phdr> phdrs;
		if (ehdr.e_phoff != 0 && ehdr.e_phnum != 0) {
			if (ehdr.e_phentsize == sizeof(Elf64_Phdr)) {
				const size_t phoff = static_cast<size_t>(ehdr.e_phoff);
				const size_t phnum = static_cast<size_t>(ehdr.e_phnum);
				if (phoff + phnum * sizeof(Elf64_Phdr) <= elf_bytes.size()) {
					phdrs.resize(phnum);
					for (size_t i = 0; i < phnum; i++) {
						if (!read_pod(std::span<const uint8_t>(
								      elf_bytes.data(),
								      elf_bytes.size()),
							      phoff +
								      i * sizeof(Elf64_Phdr),
							      phdrs[i])) {
							phdrs.clear();
							break;
						}
					}
				}
			}
		}

		// Stable per-code-object id:
		// hash only the "meaningful" ELF extent (section payloads + tables), not
		// the entire host-provided buffer (raw-ELF loads can include extra bytes).
		size_t file_end = 0;
		for (const auto &sh : shdrs) {
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
			file_end = std::max(file_end, phoff + phnum * sizeof(Elf64_Phdr));
		}
		if (file_end == 0 || file_end > elf_bytes.size())
			file_end = elf_bytes.size();
		const uint32_t image_id = sass_image_id(std::span<const uint8_t>(
			elf_bytes.data(), std::min(file_end, elf_bytes.size())));
		result.image_id = image_id;
		if (auto want = env_u32("BPFTIME_CUDA_SASS_DETOUR_FILTER_IMAGE_ID");
		    want && *want != image_id) {
			if (env_truthy("BPFTIME_CUDA_SASS_DETOUR_DEBUG")) {
				SPDLOG_INFO(
					"SASS detour: SM120 ELF skipped by image_id (got=0x{:08x} want=0x{:08x})",
					image_id, *want);
			}
			result.reason = "filtered by image_id";
			return result;
		}
		if (auto want = env_u32("BPFTIME_CUDA_SASS_DETOUR_FILTER_IMAGE_ID");
		    want && env_truthy("BPFTIME_CUDA_SASS_DETOUR_DEBUG")) {
			SPDLOG_INFO("SASS detour: SM120 ELF matched image_id=0x{:08x}",
				    image_id);
		}

		// Determine the nearest following offset for each section (including
		// section header table boundary).
			const auto dump_dir = env_dir("BPFTIME_CUDA_SASS_DETOUR_DUMP_DIR");
			const bool dump_unpatched =
				env_truthy("BPFTIME_CUDA_SASS_DETOUR_DUMP_UNPATCHED");
			bool dumped = false;

				// Some vendor/JITed cubins don't preserve readable names in `.text.*`
				// section names. Try additional match sources (symtab / nv.info.*).
				std::unordered_set<size_t> filter_text_section_indices;
				struct TextEntryPointMatch {
					std::string sym_name;
					uint64_t entry_off = 0; // offset within the containing section
					uint32_t func_id = 0; // best-effort from .nv.info.<name> (sh_info)
				};
				// `.text` aggregate section index -> matched symbol entrypoints.
				// Used when the ELF lacks per-function `.text.<name>` sections.
				std::unordered_map<size_t, std::vector<TextEntryPointMatch>>
					filter_text_entrypoints;
				std::unordered_set<uint32_t> filter_func_ids;
				if (!section_name_filter.empty()) {
					// `.nv.info.<kernel>` sections usually preserve the symbol name
					// even if `.text.*` is anonymized; link to `.text.*` via
				// `sh_info` (func_id).
				for (size_t i = 0; i < shnum; i++) {
					const auto &ish = shdrs[i];
					if (ish.sh_offset == 0 || ish.sh_size == 0)
						continue;
					auto iname = safe_cstr(*shstrtab, ish.sh_name);
					const bool is_nvinfo_name =
						iname.starts_with(".nv.info.") ||
						iname.starts_with(".nv.merc.nv.info.");
					if (is_nvinfo_name &&
					    iname.find(section_name_filter) !=
						    std::string_view::npos) {
						filter_func_ids.insert(ish.sh_info);
					}
				}

					// `.symtab` may preserve kernel symbol names; link to `.text.*`
					// via `st_shndx` (section index).
					for (size_t i = 0; i < shnum; i++) {
						const auto &sym_sh = shdrs[i];
					auto sname = safe_cstr(*shstrtab, sym_sh.sh_name);
					if (sname != ".symtab" || sym_sh.sh_entsize == 0)
						continue;
					const size_t sym_off =
						static_cast<size_t>(sym_sh.sh_offset);
					const size_t sym_sz =
						static_cast<size_t>(sym_sh.sh_size);
					if (sym_off + sym_sz > elf_bytes.size())
						break;
					const size_t str_idx =
						static_cast<size_t>(sym_sh.sh_link);
					if (str_idx >= shnum)
						break;
					const auto &str_sh = shdrs[str_idx];
					const size_t str_off =
						static_cast<size_t>(str_sh.sh_offset);
					const size_t str_sz =
						static_cast<size_t>(str_sh.sh_size);
					if (str_off + str_sz > elf_bytes.size())
						break;
					auto strtab = std::span<const uint8_t>(
						elf_bytes.data() + str_off, str_sz);

					struct Elf64_SymLocal {
						uint32_t st_name;
						uint8_t st_info;
						uint8_t st_other;
						uint16_t st_shndx;
						uint64_t st_value;
						uint64_t st_size;
					};
					const size_t entsize =
						static_cast<size_t>(sym_sh.sh_entsize);
						if (entsize < sizeof(Elf64_SymLocal))
							break;
						auto find_nvinfo_func_id_for_sym =
							[&](std::string_view sym_name)
							-> std::optional<uint32_t> {
							// Exact match on `.nv.info.<sym>` or
							// `.nv.merc.nv.info.<sym>`.
							for (size_t k = 0; k < shnum; k++) {
								const auto &ish = shdrs[k];
								if (ish.sh_offset == 0 || ish.sh_size == 0)
									continue;
								auto iname =
									safe_cstr(*shstrtab, ish.sh_name);
								const bool is_nvinfo_name =
									iname.starts_with(".nv.info.") ||
									iname.starts_with(".nv.merc.nv.info.");
								if (!is_nvinfo_name)
									continue;
								const std::string_view prefix =
									iname.starts_with(".nv.info.")
										? std::string_view(".nv.info.")
										: std::string_view(
											  ".nv.merc.nv.info.");
								if (iname.size() == prefix.size() + sym_name.size() &&
								    iname.substr(prefix.size()) == sym_name) {
									return ish.sh_info;
								}
							}
							return std::nullopt;
						};
						for (size_t off = 0;
						     off + sizeof(Elf64_SymLocal) <= sym_sz;
						     off += entsize) {
							Elf64_SymLocal sym {};
						std::memcpy(&sym, elf_bytes.data() + sym_off + off,
							    sizeof(sym));
						if (sym.st_name == 0)
							continue;
						if (sym.st_shndx == 0 ||
						    size_t(sym.st_shndx) >= shnum)
							continue;
						auto sym_name = safe_cstr(strtab, sym.st_name);
						if (sym_name.empty())
							continue;
							if (sym_name.find(section_name_filter) ==
							    std::string_view::npos)
								continue;
							const size_t sec_idx = size_t(sym.st_shndx);
							auto tname = safe_cstr(*shstrtab,
									       shdrs[sec_idx].sh_name);
							if (tname.starts_with(".text.")) {
								filter_text_section_indices.insert(sec_idx);
							} else if (tname == ".text") {
								// Aggregate `.text` section: record symbol entry offset so we
								// can patch a specific function even without `.text.<name>`
								// sections.
								TextEntryPointMatch m;
								m.sym_name = std::string(sym_name);
								m.entry_off = sym.st_value;
								if (auto id =
									    find_nvinfo_func_id_for_sym(sym_name);
								    id) {
									m.func_id = *id;
								}
								filter_text_entrypoints[sec_idx].push_back(
									std::move(m));
							}
						}
						break;
					}
				}
	if (extra_filter_func_ids) {
		for (uint32_t id : *extra_filter_func_ids)
			filter_func_ids.insert(id);
	}

	// Identify-closure patch-all fallback can be overly invasive if we detour
	// every `.text.*` section (including internal helper functions). Prefer a
	// narrower fallback when possible: only detour `.text.*` sections that have
	// a corresponding `.nv.info.*` / `.nv.merc.nv.info.*` section (a strong hint
	// that the function is a kernel entrypoint).
	if (patch_all_is_fallback && section_name_filter.empty() &&
	    extra_filter_func_ids == nullptr &&
	    filter_text_section_indices.empty() && filter_func_ids.empty()) {
		for (size_t i = 0; i < shnum; i++) {
			const auto &ish = shdrs[i];
			auto iname = safe_cstr(*shstrtab, ish.sh_name);
			const bool is_nvinfo =
				iname.starts_with(".nv.info.") ||
				iname.starts_with(".nv.merc.nv.info.");
			if (!is_nvinfo)
				continue;
			if (ish.sh_info != 0)
				filter_func_ids.insert(static_cast<uint32_t>(ish.sh_info));
		}
	}

		// Some CUDA cubins (observed in vendor / flashattention objects) do not use
		// `sh_info` on `.text.*` sections as the stable per-function ID. Instead,
		// the "func_id" used by `.nv.info.*` sometimes matches the *section index*
		// of the corresponding `.text.*` section.
		//
		// To make func_id-based detouring robust (strip scenario), detect this shape
		// and allow matching by section index as a fallback.
		//
		// Note: do NOT blindly prefer section indices: some cubins (e.g. vLLM custom
		// kernels) have a valid `sh_info` that matches the global regcount table,
		// while `.nv.info.<name>` uses a different identifier (often the section
		// index of `.nv.merc.nv.info.<name>` or `.nv.info.<name>`). We select the
		// best candidate per-section below.
		bool func_id_matches_text_section_index = false;
		if (!filter_func_ids.empty()) {
			for (uint32_t id : filter_func_ids) {
				if (id >= shnum)
					continue;
				auto n = safe_cstr(*shstrtab, shdrs[id].sh_name);
				if (n.starts_with(".text.")) {
					func_id_matches_text_section_index = true;
					break;
				}
			}
		}

	size_t seen_text_sections = 0;

		std::optional<NvInfoGlobalRegcountIndex> global_nvinfo_reg_idx;
		std::optional<NvInfoGlobalRegcountIndex> global_nvmerc_reg_idx;
		if (sampling_cfg && sampling_cfg->enabled) {
			auto elf_span = std::span<const uint8_t>(elf_bytes.data(),
							 elf_bytes.size());
		global_nvinfo_reg_idx = build_nvinfo_global_regcount_index(
			elf_span, shdrs, *shstrtab, ".nv.info");
		global_nvmerc_reg_idx = build_nvinfo_global_regcount_index(
			elf_span, shdrs, *shstrtab, ".nv.merc.nv.info");
	}

		for (size_t i = 0; i < shnum; i++) {
			auto &sh = shdrs[i];
			if (sh.sh_offset == 0 || sh.sh_size < SASS_INST_BYTES)
				continue;
				auto name = safe_cstr(*shstrtab, sh.sh_name);
				if (!(name.starts_with(".text.") || name == ".text"))
					continue;
				const bool is_text_aggregate = (name == ".text");
				std::optional<TextEntryPointMatch> text_agg_entry;
				const bool text_agg_has_entrypoint = [&]() -> bool {
					auto it = filter_text_entrypoints.find(i);
					return (it != filter_text_entrypoints.end() &&
						!it->second.empty());
				}();
				const bool text_agg_allow_fallback_patch =
					(is_text_aggregate && !text_agg_has_entrypoint &&
					 // If we only have a single aggregate `.text` section, we may
					 // still want a best-effort detour in patch-all fallback mode to
					 // avoid "no records" in strip/sectionless workloads.
					 (patch_all_is_fallback ||
					  (section_name_filter.empty() &&
					   filter_text_section_indices.empty() &&
					   filter_func_ids.empty())));
				const bool text_agg_force_exit_detour =
					(is_text_aggregate && !text_agg_has_entrypoint);
					std::string synth_text_name_storage;
					std::string_view match_text_name = name;
					size_t entry_rel = 0;
					// Packed predicate code: low3=pred index, bit7=negate, pred==7 => unpredicated.
					uint8_t detour_site_pred = sm120_pack_pred(7u, false);
					bool reg255_pre_exit_detour = false;
					uint32_t func_id_for_entry = 0;
				if (is_text_aggregate) {
					if (text_agg_has_entrypoint) {
						auto it = filter_text_entrypoints.find(i);
						// Minimal bring-up: patch only the first matched symbol in `.text`
						// to avoid exhausting the single-section code cave.
						text_agg_entry = it->second.front();
						entry_rel =
							static_cast<size_t>(text_agg_entry->entry_off);
						func_id_for_entry = text_agg_entry->func_id;
						synth_text_name_storage =
							std::string(".text.") +
							text_agg_entry->sym_name;
						match_text_name = synth_text_name_storage;
					} else if (text_agg_allow_fallback_patch) {
						// We don't know function boundaries inside the aggregate `.text`
						// segment. Prefer a tail EXIT detour later for best-effort
						// liveness safety.
						entry_rel = 0;
						func_id_for_entry =
							static_cast<uint32_t>(i);
						synth_text_name_storage = ".text.__aggregate__";
						match_text_name = synth_text_name_storage;
					} else {
						continue;
					}
				}
					seen_text_sections++;
						bool detour_match_by_text_name = false;
						bool detour_match_by_symtab = false;
						bool detour_match_by_func_id = false;
						uint32_t text_func_id = 0;
						if (is_text_aggregate) {
							text_func_id = func_id_for_entry;
						} else {
							const uint32_t cand_shinfo =
								static_cast<uint32_t>(sh.sh_info);
							const uint32_t cand_index =
								static_cast<uint32_t>(i);
							auto in_global_regcount_index =
								[&](uint32_t id) -> bool {
								if (global_nvinfo_reg_idx &&
								    global_nvinfo_reg_idx
										    ->func_to_regcount_file_off
										    .count(id) != 0)
									return true;
								if (global_nvmerc_reg_idx &&
								    global_nvmerc_reg_idx
										    ->func_to_regcount_file_off
										    .count(id) != 0)
									return true;
								return false;
							};
							// Prefer `sh_info` when it is present and usable for regcount
							// patching; only fall back to section index when `sh_info` is
							// absent or clearly doesn't match the available regcount tables.
							if (cand_shinfo != 0 &&
							    (in_global_regcount_index(cand_shinfo) ||
							     !in_global_regcount_index(cand_index))) {
								text_func_id = cand_shinfo;
							} else if (in_global_regcount_index(cand_index)) {
								text_func_id = cand_index;
							} else if (cand_shinfo != 0) {
								text_func_id = cand_shinfo;
							} else if (func_id_matches_text_section_index) {
								text_func_id = cand_index;
							} else {
								text_func_id = cand_index;
							}
						}
						const uint32_t func_id = text_func_id;
						const bool need_filter =
							(!section_name_filter.empty() ||
							 !filter_text_section_indices.empty() ||
							 !filter_func_ids.empty() ||
							 (is_text_aggregate && text_agg_entry.has_value()));
					if (need_filter) {
						detour_match_by_text_name =
							(!section_name_filter.empty() &&
							 match_text_name.find(section_name_filter) !=
								 std::string_view::npos);
						detour_match_by_symtab =
							(!filter_text_section_indices.empty() &&
							 filter_text_section_indices.count(i) != 0);
						if (!detour_match_by_symtab && is_text_aggregate &&
						    text_agg_entry.has_value()) {
							// Aggregate `.text` symbol match from symtab.
							detour_match_by_symtab = true;
						}
						detour_match_by_func_id =
							(!filter_func_ids.empty() &&
							 filter_func_ids.count(text_func_id) != 0);
						if (!detour_match_by_text_name &&
					    !detour_match_by_symtab &&
					    !detour_match_by_func_id) {
						result.skipped_text_sections++;
						continue;
					}
					} else {
						// Patch-all: treat as “matched” for downstream decisions.
						detour_match_by_text_name = true;
					}
					// For “patch-all fallback” (used by identify closure), we still
					// need to inject a minimal stub to collect func_id, but we must
					// NOT treat every patched section as a detour-selected target for
					// sampling/control reads.
					const bool detour_selected =
						patch_all_is_fallback ? false :
									(need_filter
										 ? (detour_match_by_text_name ||
										    detour_match_by_symtab ||
										    detour_match_by_func_id)
										 : true);

			const size_t start = static_cast<size_t>(sh.sh_offset);
			const size_t size = static_cast<size_t>(sh.sh_size);
			if (start + size > elf_bytes.size()) {
				result.skipped_text_sections++;
				continue;
			}
			if ((size % SASS_INST_BYTES) != 0) {
				result.skipped_text_sections++;
				continue;
			}
			if (entry_rel + SASS_INST_BYTES > size) {
				result.skipped_text_sections++;
				continue;
			}
			if ((entry_rel % SASS_INST_BYTES) != 0) {
				result.skipped_text_sections++;
				continue;
			}

			auto section_span = std::span<const uint8_t>(
				elf_bytes.data() + start, size);
			std::optional<size_t> cave_rel;

		// Decide how many prefix instructions to replay in the trampoline.
		// Default is 1 (the detoured instruction). Larger replay windows are
		// risky because the early instructions of some kernels can include
		// PC-relative control flow; keep it opt-in via env.
		size_t replay_n = 1;
		if (const char *v = std::getenv("BPFTIME_CUDA_SASS_DETOUR_REPLAY_N");
		    v && *v) {
			long x = std::strtol(v, nullptr, 10);
			if (x >= 1 && x <= 64)
				replay_n = static_cast<size_t>(x);
		}
		// PcMarker is intended to be a very small, low-risk multi-point bring-up;
		// keep the replay window at exactly 1 instruction.
		if (sampling_cfg && sampling_cfg->enabled &&
		    sampling_cfg->mode == Sm120SamplingConfig::Mode::PcMarker)
			replay_n = 1;
		if (size < replay_n * SASS_INST_BYTES)
			replay_n = std::max<size_t>(1, size / SASS_INST_BYTES);

				size_t entry_file_off = start + entry_rel;
				size_t pc0_base_file_off = entry_file_off;
				std::vector<uint8_t> prefix_bytes(replay_n * SASS_INST_BYTES);
				std::memcpy(prefix_bytes.data(), elf_bytes.data() + entry_file_off,
					    prefix_bytes.size());

		std::array<uint8_t, SASS_INST_BYTES> first_inst {};
		std::memcpy(first_inst.data(), prefix_bytes.data(), SASS_INST_BYTES);
		uint64_t first_w0 = 0;
		std::memcpy(&first_w0, first_inst.data(), sizeof(first_w0));
		const uint16_t first_op16 = uint16_t(first_w0 & 0xffffu);
		(void)first_op16;

			std::vector<SassInst> stub_insts;
				std::optional<Sm120RegLayout> layout;
				bool sampling_enabled = false;
				bool stub_no_regcount_fallback = false;
				bool request_exit_detour = false;
				bool exit_thread_map_no_regcount = false;
					// When we build a no-regcount EXIT stub, keep the kernel's original
					// regcount (if known) so nvdisasm-based "dead reg" picking does not
					// choose out-of-range registers (can crash on non-255 kernels).
					std::optional<uint32_t> exit_stub_old_regcount;
					std::optional<std::array<uint8_t, 6>> exit_thread_map_scratch_regs;
					std::optional<uint8_t> exit_stub_existing_desc_ur_base;
						uint32_t tag = 0;
					const bool want_sampling =
						(sampling_cfg && sampling_cfg->enabled &&
				    sampling_cfg->sample_buffer_device_ptr != 0 &&
				    (sampling_cfg->mode == Sm120SamplingConfig::Mode::SmidBitmap ||
				     (sampling_cfg->mode ==
					      Sm120SamplingConfig::Mode::PcMarker &&
				      sampling_cfg->marker_ring_entries != 0) ||
				     sampling_cfg->max_records != 0));
		const bool want_control =
			(sampling_cfg && sampling_cfg->enabled &&
			 sampling_cfg->control_enabled &&
			 sampling_cfg->sample_buffer_device_ptr != 0);
				// The “patch-all fallback” path is used for identify closure on
				// stripped images; keep it conservative and avoid sampling there.
				const bool want_sampling_for_this_call =
					(want_sampling && !patch_all_is_fallback);
				bool sample_match =
					(sample_section_filter.empty() ||
					 match_text_name.find(sample_section_filter) !=
						 std::string_view::npos);
		// Strip scenario: the SM120 `.text.*` section name may not contain the
		// kernel name, but the section can still be selected via `.nv.info.*`
		// `sh_info` (func_id) propagation from other SMs in the same fatbin.
		//
		// If the user uses the same string for detour + sampling filters (a
		// common setup), treat any detoured section as a sample match as well
		// when the detour selection came from func_id/symtab rather than the
		// `.text.<name>` string.
		//
		// Also handle the "func_id-only detour" mode (section_name_filter==""):
		// once we have a precise func_id, we may intentionally ignore the text
		// name substring to avoid patching many template instantiations; in that
		// case sampling should still follow the detour selection.
		if (!sample_match && !sample_section_filter.empty() &&
		    (detour_match_by_symtab || detour_match_by_func_id) &&
		    (section_name_filter.empty() ||
		     sample_section_filter == section_name_filter))
			sample_match = true;
				const bool need_stub_for_section =
					(want_sampling_for_this_call && sample_match) || want_control;
			if (need_stub_for_section) {
				const std::string_view kernel_name = [&]() -> std::string_view {
					if (is_text_aggregate) {
						if (text_agg_entry)
							return std::string_view(text_agg_entry->sym_name);
						return std::string_view("__aggregate__");
					}
					return match_text_name.substr(
						std::string_view(".text.").size());
				}();
				tag = fnv1a32(kernel_name);

				// Identify-closure patch-all fallback:
				// Always inject the minimal "write-only identify" stub, even when
				// regcount patching would be possible. The controlled stub path is
				// designed to avoid global reads/writes for non-selected targets and
				// intentionally does not populate `control.slots[]`, which would
				// break identify closure.
				if (patch_all_is_fallback && want_control && sampling_cfg &&
				    sampling_cfg->control_enabled) {
					uint8_t scratch_rd = uint8_t((first_w0 >> 16) & 0xffu);
					if (scratch_rd == 0xffu)
						scratch_rd = 1u;
					const auto ldcu_desc_fallback =
						find_ldcu64_desc_template_sm120(
							section_span,
							/*override_ur_base=*/std::nullopt);
					const auto ldg32_desc_fallback =
						find_ldg_e32_desc_template_sm120(
							section_span,
							/*override_ur_base=*/std::nullopt);
						stub_insts = build_sm120_identify_only_stub_no_regcount(
							*sampling_cfg, scratch_rd, func_id, image_id,
							ldcu_desc_fallback, ldg32_desc_fallback,
							/*detour_selected=*/false);
					sampling_enabled = !stub_insts.empty();
					stub_no_regcount_fallback = sampling_enabled;
					if (stub_no_regcount_fallback &&
					    env_truthy("BPFTIME_CUDA_SASS_DETOUR_DEBUG")) {
						SPDLOG_INFO(
							"SASS detour: patch-all identify-only stub injected text='{}' func_id={}",
							std::string(name), func_id);
					}
				} else {
			// Preferred: patch global EIATTR_REGCOUNT tables by `sh_info`
			// (func_id). On CUDA 12.x SM100+, `.nv.merc.nv.info` is the
			// runtime table, and `.nv.info` may differ.
			bool reg_patched = false;
			uint32_t old_rc_u32 = 0;

			auto read_global_regcount =
				[&](const std::optional<NvInfoGlobalRegcountIndex> &idx,
				    uint32_t *out_old_rc) -> bool {
				if (!idx)
					return false;
				auto it = idx->func_to_regcount_file_off.find(func_id);
				if (it == idx->func_to_regcount_file_off.end())
					return false;
				const size_t off = it->second;
				if (off + 4 > elf_bytes.size())
					return false;
				std::memcpy(out_old_rc, elf_bytes.data() + off, 4);
				return true;
			};

			auto patch_global_regcount =
				[&](const std::optional<NvInfoGlobalRegcountIndex> &idx,
				    uint32_t new_rc) -> bool {
				if (!idx)
					return false;
				auto it = idx->func_to_regcount_file_off.find(func_id);
				if (it == idx->func_to_regcount_file_off.end())
					return false;
				const size_t off = it->second;
				if (off + 4 > elf_bytes.size())
					return false;
				std::memcpy(elf_bytes.data() + off, &new_rc, 4);
				return true;
			};

				// Note: CUDA's EIATTR_REGCOUNT payload can be a packed u32 where the
				// low 8 bits contain the register count and higher bits may carry
				// additional flags/metadata. Treat the low byte as the regcount and
				// preserve the upper bits when patching.
				uint32_t old_rc_word_merc = 0;
				uint32_t old_rc_word_nvinfo = 0;
				const bool have_merc =
					read_global_regcount(global_nvmerc_reg_idx, &old_rc_word_merc);
				const bool have_nvinfo =
					read_global_regcount(global_nvinfo_reg_idx, &old_rc_word_nvinfo);

					if (have_merc || have_nvinfo) {
						old_rc_u32 =
							have_merc ? old_rc_word_merc : old_rc_word_nvinfo;
						const uint32_t old_rc =
							uint32_t(old_rc_u32 & 0xffu);
						if (old_rc == 255u && sampling_cfg && sampling_cfg->enabled &&
						    sampling_cfg->mode ==
							    Sm120SamplingConfig::Mode::ThreadMap &&
						    sampling_cfg->thread_map_device &&
						    want_sampling_for_this_call && sample_match &&
						    detour_selected) {
								// regcount=255 + thread_map(device): prefer EXIT detour with a
								// no-regcount stub, rather than risking entry/prologue patching.
								stub_insts =
									build_sm120_thread_map_stub_no_regcount_at_exit(
										*sampling_cfg, func_id,
										/*scratch_regs=*/std::nullopt);
							sampling_enabled = !stub_insts.empty();
							request_exit_detour = sampling_enabled;
							stub_no_regcount_fallback = sampling_enabled;
							exit_thread_map_no_regcount = sampling_enabled;
							if (sampling_enabled)
								exit_stub_old_regcount = 255u;
							if (sampling_enabled &&
							    env_truthy("BPFTIME_CUDA_SASS_DETOUR_DEBUG")) {
								SPDLOG_INFO(
									"SASS detour: reg255 thread_map(device) EXIT stub enabled text='{}' func_id={}",
								std::string(name), func_id);
						}
					}
						layout = compute_sm120_layout(old_rc,
									      sampling_cfg->mode);
						// regcount-saturated kernels (e.g., flashattention variants with
						// old_rc close to 255) can't fit our entry stubs because there is no
						// room to increase regcount for scratch registers. For
						// thread_map(device), prefer a no-regcount EXIT stub rather than
						// falling back to identify-only.
						if (!layout && sampling_cfg && sampling_cfg->enabled &&
						    sampling_cfg->mode ==
							    Sm120SamplingConfig::Mode::ThreadMap &&
						    sampling_cfg->thread_map_device &&
						    want_sampling_for_this_call && sample_match &&
						    detour_selected) {
							stub_insts =
								build_sm120_thread_map_stub_no_regcount_at_exit(
									*sampling_cfg, func_id,
									/*scratch_regs=*/std::nullopt);
							sampling_enabled = !stub_insts.empty();
							request_exit_detour = sampling_enabled;
							stub_no_regcount_fallback = sampling_enabled;
							exit_thread_map_no_regcount = sampling_enabled;
							if (sampling_enabled)
								exit_stub_old_regcount = old_rc;
							if (sampling_enabled &&
							    env_truthy("BPFTIME_CUDA_SASS_DETOUR_DEBUG")) {
								SPDLOG_INFO(
									"SASS detour: reg-saturated thread_map(device) EXIT stub enabled (old_rc={}) text='{}' func_id={}",
									old_rc, std::string(name), func_id);
							}
						}
						if (layout) {
							const uint32_t new_rc = layout->new_regcount;
							const uint32_t new_word_merc =
								(old_rc_word_merc & ~0xffu) | (new_rc & 0xffu);
							const uint32_t new_word_nvinfo =
								(old_rc_word_nvinfo & ~0xffu) | (new_rc & 0xffu);
							bool patched_any = false;
							if (have_merc)
								patched_any |= patch_global_regcount(
									global_nvmerc_reg_idx,
									new_word_merc);
							if (have_nvinfo)
								patched_any |= patch_global_regcount(
									global_nvinfo_reg_idx,
									new_word_nvinfo);
							reg_patched = patched_any;
						}
					} else if (sampling_cfg && sampling_cfg->enabled &&
						   sampling_cfg->mode ==
							   Sm120SamplingConfig::Mode::ThreadMap &&
						   sampling_cfg->thread_map_device &&
						   want_sampling_for_this_call && sample_match &&
						   detour_selected) {
						const bool force_exit_stub =
							env_truthy("BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_EXIT_STUB_ON_NO_REGCOUNT") ||
							// When we are already in a precise func_id-targeted detour
							// (identify closure -> on-demand upgrade), prefer forcing the
							// no-regcount EXIT stub rather than falling back to identify-only.
							// This keeps the effect narrow (target-only) while enabling
							// device-per-thread bring-up on regcount-saturated kernels that
							// don't ship readable regcount tables.
							(detour_match_by_func_id &&
							 !env_truthy("BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_NO_AUTO_EXIT_STUB_ON_TARGET"));
						if (!force_exit_stub) {
							// Continue to the per-kernel nvinfo patch path; if that also
							// fails, we'll fall back to identify-only (stable but sparse).
						} else {
						// Some cubin-only vendor kernels don't ship readable regcount tables.
						// For bring-up, allow forcing the no-regcount EXIT stub even when we
						// can't confirm regcount==255 from `.nv.info`/`.nv.merc` metadata.
							stub_insts =
								build_sm120_thread_map_stub_no_regcount_at_exit(
									*sampling_cfg, func_id,
									/*scratch_regs=*/std::nullopt);
						sampling_enabled = !stub_insts.empty();
						request_exit_detour = sampling_enabled;
						stub_no_regcount_fallback = sampling_enabled;
						exit_thread_map_no_regcount = sampling_enabled;
						// regcount is unknown in this branch; keep exit_stub_old_regcount unset
						// to disable nvdisasm-based dead-reg picking.
						if (sampling_enabled &&
						    env_truthy("BPFTIME_CUDA_SASS_DETOUR_DEBUG")) {
							SPDLOG_INFO(
								"SASS detour: thread_map(device) EXIT stub forced (no regcount table) text='{}' func_id={}",
								std::string(name), func_id);
						}
						}
					}

			// Fallback: patch per-kernel nvinfo payload by `func_id` (sh_info).
			//
			// Many stripped/JITed cubins do not preserve readable kernel names in
			// section headers, but still carry per-function `.nv.info.*` /
			// `.nv.merc.nv.info.*` sections whose `sh_info` matches the
			// corresponding `.text.*` section. Prefer that stable linkage over
			// string matching.
			if (!reg_patched) {
				const std::string nvinfo_name =
					std::string(".nv.info.") + std::string(kernel_name);
				const std::string nvmerc_name = std::string(".nv.merc.nv.info.") +
								std::string(kernel_name);
				for (size_t j = 0; j < shnum; j++) {
					auto &ish = shdrs[j];
					if (ish.sh_offset == 0 || ish.sh_size < 4)
						continue;
					auto iname = safe_cstr(*shstrtab, ish.sh_name);
					const bool name_match =
						(iname == nvinfo_name || iname == nvmerc_name);
					const bool info_match =
						(ish.sh_info == func_id &&
						 (iname.starts_with(".nv.info.") ||
						  iname.starts_with(".nv.merc.nv.info.")));
					if (!name_match && !info_match)
						continue;
					const size_t istart =
						static_cast<size_t>(ish.sh_offset);
					const size_t isize =
						static_cast<size_t>(ish.sh_size);
					if (istart + isize > elf_bytes.size())
						continue;
					auto nvspan = std::span<uint8_t>(
						elf_bytes.data() + istart, isize);
					auto old_rc = read_nvinfo_regcount(
						std::span<const uint8_t>(nvspan.data(),
									 nvspan.size()));
					if (!old_rc)
						continue;
					layout = compute_sm120_layout(*old_rc,
								      sampling_cfg->mode);
					if (!layout)
						continue;
					auto patched = patch_nvinfo_regcount(
						nvspan, layout->new_regcount);
					if (patched) {
						old_rc_u32 = patched->first;
						reg_patched = true;
					}
					if (reg_patched)
						break;
				}
			}

				if (layout && reg_patched) {
						if (want_sampling_for_this_call && sample_match) {
							result.sampled_kernels.push_back(InstrumentedKernelInfo {
								std::string(name),
								std::string(kernel_name),
							tag,
							layout->old_regcount,
							layout->new_regcount,
							});
						}
							if (sampling_cfg && sampling_cfg->control_enabled) {
								const auto ldcu_desc =
									find_ldcu64_desc_template_sm120(
										section_span, /*override_ur_base=*/4u);
								const auto ldg64_desc =
									find_ldg_e64_desc_template_sm120(
										section_span, /*want_ur_base=*/4);
								if (env_truthy("BPFTIME_CUDA_SASS_DETOUR_DEBUG") &&
								    (detour_match_by_text_name ||
								     detour_match_by_symtab ||
								     detour_match_by_func_id)) {
								SPDLOG_INFO(
										"SASS detour: controlled stub templates text='{}' func_id={} old_rc={} new_rc={} ldcu_desc={} ldg64_desc={}",
										std::string(name), func_id,
										layout->old_regcount,
										layout->new_regcount,
										ldcu_desc.has_value() ? 1 : 0,
										ldg64_desc.has_value() ? 1 : 0);
								}
										stub_insts = build_sm120_controlled_sampling_stub(
											*sampling_cfg, *layout, tag, func_id, image_id,
											detour_selected, ldcu_desc, ldg64_desc);
								} else {
							const auto ldcu_desc =
							find_ldcu64_desc_template_sm120(
								section_span,
								/*override_ur_base=*/sampling_cfg
									? sampling_cfg->desc_ur
									: uint8_t(4));
						stub_insts = build_sm120_sampling_stub(
							*sampling_cfg, *layout, tag, ldcu_desc);
					}
					sampling_enabled = !stub_insts.empty();
				}
			}
			// Optional: for thread-map(device), prefer detouring at the tail EXIT
			// rather than at the entry/prologue. This is a stability knob for kernels
			// whose early instructions use a wide live register set (some ATen/cutlass
			// helpers), where our entry stub may clobber a live GPR and crash.
			if (!request_exit_detour && sampling_enabled && sampling_cfg &&
			    sampling_cfg->enabled &&
			    sampling_cfg->mode == Sm120SamplingConfig::Mode::ThreadMap &&
			    sampling_cfg->thread_map_device &&
			    env_truthy("BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_PREFER_EXIT")) {
				request_exit_detour = true;
			}
			if (text_agg_force_exit_detour)
				request_exit_detour = true;
			// Identify-closure fallback for regcount-saturated kernels:
			// if we couldn't patch regcount to reserve scratch registers, still
			// inject a minimal identify stub so the host can learn func_id within
			// the same run.
				if (!sampling_enabled && want_control &&
				    sampling_cfg && sampling_cfg->control_enabled) {
					// No-regcount fallback: avoid clobbering arbitrary registers on
					// regcount-saturated kernels by borrowing a scratch GPR that will
					// be overwritten by the replayed prologue prefix (we use the
					// destination register of the first instruction by convention).
					uint8_t scratch_rd = uint8_t((first_w0 >> 16) & 0xffu);
					if (scratch_rd == 0xffu)
						scratch_rd = 1u;
					const uint8_t ctrl_desc_ur = 4u;
					const auto ldcu_desc_fallback =
						find_ldcu64_desc_template_sm120(
							section_span,
							/*override_ur_base=*/std::nullopt);
					const auto ldg32_desc_fallback =
						find_ldg_e32_desc_template_sm120(
							section_span,
							/*override_ur_base=*/std::nullopt);
						stub_insts = build_sm120_identify_only_stub_no_regcount(
							*sampling_cfg, scratch_rd, func_id, image_id,
							ldcu_desc_fallback,
							ldg32_desc_fallback,
								detour_selected);
					sampling_enabled = !stub_insts.empty();
					stub_no_regcount_fallback = sampling_enabled;
					if (stub_no_regcount_fallback &&
					    env_truthy("BPFTIME_CUDA_SASS_DETOUR_DEBUG")) {
						SPDLOG_INFO(
							"SASS detour: identify-only fallback stub injected text='{}' func_id={} (regcount patch unavailable, ctrl_desc_ur={} ldcu_desc={} ldg32_desc={})",
							std::string(name), func_id,
							(unsigned)ctrl_desc_ur,
							ldcu_desc_fallback.has_value() ? 1 : 0,
							ldg32_desc_fallback.has_value() ? 1 : 0);
					}
					}

				} // !patch_all_is_fallback
			if (request_exit_detour &&
			    (!is_text_aggregate || !text_agg_entry.has_value())) {
				// Switch the detour site from the entry instruction to the tail EXIT.
						// This is only used for regcount=255 + thread_map(device) stubs.
						// Allowing predicated EXIT can be unsafe when the predicate is
						// divergent within a warp. Keep a conservative default:
						// - `...ALLOW_PREDICATED_EXIT=1`: allow @P0 EXIT only (legacy behavior)
						// - `...ALLOW_PREDICATED_EXIT_ANY=1`: allow any predicate P0..P6
						const bool allow_pred_p0 = env_truthy(
							"BPFTIME_CUDA_SASS_DETOUR_ALLOW_PREDICATED_EXIT");
						const bool allow_pred_any = env_truthy(
							"BPFTIME_CUDA_SASS_DETOUR_ALLOW_PREDICATED_EXIT_ANY");
						uint8_t exit_pred = 7u;
						bool exit_neg = false;
						if (auto exit_rel = find_tail_exit_sm120(
							    section_span, /*prefer_unpred=*/true,
							    /*allow_pred_p0=*/allow_pred_p0,
							    /*allow_pred_any=*/allow_pred_any,
							    /*out_pred=*/&exit_pred,
							    /*out_neg=*/&exit_neg);
						    exit_rel) {
						if (exit_pred == 7u)
							exit_neg = false;
						detour_site_pred = sm120_pack_pred(exit_pred, exit_neg);
						entry_rel = *exit_rel;
						replay_n = 1; // replay exactly one instruction at the detour site
						if (exit_pred == 7u &&
						    env_truthy("BPFTIME_CUDA_SASS_DETOUR_REG255_PRE_EXIT")) {
							const size_t max_scan_insts =
								env_u32("BPFTIME_CUDA_SASS_DETOUR_REG255_PRE_EXIT_SCAN_INSTS")
									.value_or(8u);
						if (auto nop_rel = find_pre_exit_nop_sm120(
							    section_span, *exit_rel, max_scan_insts);
						    nop_rel) {
							entry_rel = *nop_rel;
							reg255_pre_exit_detour = true;
						}
					}
					entry_file_off = start + entry_rel;
					pc0_base_file_off = entry_file_off;
					prefix_bytes.resize(replay_n * SASS_INST_BYTES);
					std::memcpy(prefix_bytes.data(),
						    elf_bytes.data() + entry_file_off,
						    prefix_bytes.size());
					std::memcpy(first_inst.data(), prefix_bytes.data(),
						    SASS_INST_BYTES);
					std::memcpy(&first_w0, first_inst.data(),
						    sizeof(first_w0));

					// Optional: pick scratch registers from "dead regs after the
					// detour point" (nvdisasm-based). This is a minimal liveness
					// approximation: any GPR not mentioned in the suffix is safe to
					// clobber at the detour site.
					//
					// This is particularly useful for regcount=255 vendor kernels
					// where we want to avoid relying on fixed low registers.
						if (exit_thread_map_no_regcount &&
						    env_truthy(
							    "BPFTIME_CUDA_SASS_DETOUR_EXIT_STUB_USE_DEAD_REGS") &&
						    sampling_cfg) {
							if (!exit_stub_old_regcount) {
								// Without a trustworthy regcount, dead-reg selection can pick
								// out-of-range registers and crash. Keep fixed low scratch regs.
								// (Set BPFTIME_CUDA_SASS_DETOUR_DEBUG for visibility.)
							} else {
							auto dead = compute_dead_gprs_by_nvdisasm(
								std::span<const uint8_t>(
									elf_bytes.data(), elf_bytes.size()),
								std::string_view(name),
								/*old_regcount=*/*exit_stub_old_regcount,
								/*from_pc=*/entry_rel + SASS_INST_BYTES);
							if (dead) {
								auto picked =
									pick_exit_thread_map_scratch_regs_from_dead(
										*dead);
								if (picked) {
										exit_thread_map_scratch_regs = *picked;
										stub_insts =
											build_sm120_thread_map_stub_no_regcount_at_exit(
												*sampling_cfg, func_id,
												exit_thread_map_scratch_regs);
									sampling_enabled = !stub_insts.empty();
									if (env_truthy(
										    "BPFTIME_CUDA_SASS_DETOUR_DEBUG")) {
										SPDLOG_INFO(
											"SASS detour: reg255 exit-stub dead-reg scratch picked ptr={} cta={} tid={} lane={} smid={} idx={} text='{}' func_id={}",
										(unsigned)exit_thread_map_scratch_regs
											->at(0),
										(unsigned)exit_thread_map_scratch_regs
											->at(1),
										(unsigned)exit_thread_map_scratch_regs
											->at(2),
										(unsigned)exit_thread_map_scratch_regs
											->at(3),
										(unsigned)exit_thread_map_scratch_regs
											->at(4),
										(unsigned)exit_thread_map_scratch_regs
											->at(5),
										std::string(name), func_id);
									}
								}
								}
								}
							}

							// For no-regcount EXIT stubs in "no-UR-write" mode, try to reuse an
							// already-initialized `desc[URx]` from the kernel's own epilogue.
							//
							// This is especially important when we detour a predicated EXIT: the
							// detoured lanes may jump to the trampoline before the kernel's own
							// prologue loads the conventional `UR4/UR5` descriptor, so assuming
							// `desc[UR4]` is ready can crash with "illegal instruction".
							if (exit_thread_map_no_regcount && sampling_cfg &&
							    sampling_cfg->control_enabled &&
							    !env_truthy(
								    "BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_EXIT_STUB_USE_UR")) {
								const size_t scan_insts =
									env_u32("BPFTIME_CUDA_SASS_DETOUR_EXIT_STUB_INFER_DESC_SCAN_INSTS")
										.value_or(128u);
								exit_stub_existing_desc_ur_base =
									infer_existing_desc_ur_base_near_sm120(
										section_span, entry_rel, scan_insts);
								if (exit_stub_existing_desc_ur_base) {
									stub_insts =
										build_sm120_thread_map_stub_no_regcount_at_exit(
											*sampling_cfg, func_id,
											exit_thread_map_scratch_regs,
											exit_stub_existing_desc_ur_base);
									sampling_enabled = !stub_insts.empty();
									if (env_truthy("BPFTIME_CUDA_SASS_DETOUR_DEBUG")) {
										SPDLOG_INFO(
											"SASS detour: reg255 EXIT stub inferred desc_ur_base={} text='{}' func_id={}",
											unsigned(*exit_stub_existing_desc_ur_base),
											std::string(name), func_id);
									}
								}
							}
						} else {
							// If we *require* an EXIT detour (regcount=255 + no-regcount exit
							// stub), we cannot safely fall back to entry detour: the stub assumes
							// it runs at EXIT and may clobber low registers.
						if (exit_thread_map_no_regcount) {
							if (env_truthy("BPFTIME_CUDA_SASS_DETOUR_DEBUG")) {
								SPDLOG_INFO(
									"SASS detour: reg255 EXIT stub requested but no EXIT found text='{}' func_id={}",
									std::string(name), func_id);
							}
							result.skipped_text_sections++;
							continue;
						}
						// Best-effort prefer-exit: keep the entry detour when there is no
						// recognizable tail EXIT in this section (coverage over preference).
						if (env_truthy("BPFTIME_CUDA_SASS_DETOUR_DEBUG") ||
						    env_truthy(
							    "BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_PREFER_EXIT")) {
							SPDLOG_INFO(
								"SASS detour: prefer-exit requested but no tail EXIT found; falling back to entry detour text='{}' func_id={}",
								std::string(name), func_id);
						}
					}
				}

				const size_t tramp_insts =
					(sampling_enabled ? stub_insts.size() : 0) + replay_n + 1;
				const size_t per_tramp_bytes = SASS_INST_BYTES * tramp_insts;
				// Multi-site patching:
				// - PcMarker: patch multiple offsets within the same `.text.*` section.
				// - ThreadMap prefer-exit: patch multiple EXIT sites to avoid "record=0"
				//   when the tail EXIT is not on the executed path (multi-exit / dead code).
				//
				// Pre-compute the required cave size so we can allocate one contiguous
				// cave and place one trampoline per site.
					std::vector<size_t> pc_marker_sites_rel;
					std::vector<std::pair<size_t, uint8_t>>
						threadmap_exit_sites_rel;
				size_t cave_need_bytes = per_tramp_bytes;
				if (sampling_enabled && sampling_cfg &&
				    sampling_cfg->mode == Sm120SamplingConfig::Mode::PcMarker) {
					for (uint32_t off : sampling_cfg->marker_offsets) {
					const size_t rel = static_cast<size_t>(off);
					if (rel + SASS_INST_BYTES > size)
						continue;
					if ((rel % SASS_INST_BYTES) != 0)
						continue;
					pc_marker_sites_rel.push_back(rel);
				}
				if (pc_marker_sites_rel.empty())
					pc_marker_sites_rel.push_back(entry_rel);
				std::sort(pc_marker_sites_rel.begin(),
					  pc_marker_sites_rel.end());
				pc_marker_sites_rel.erase(
					std::unique(pc_marker_sites_rel.begin(),
						    pc_marker_sites_rel.end()),
					pc_marker_sites_rel.end());
				if (pc_marker_sites_rel.size() > 16)
					pc_marker_sites_rel.resize(16);
				const size_t n = pc_marker_sites_rel.size();
					if (n != 0 && per_tramp_bytes != 0 &&
					    n <= (std::numeric_limits<size_t>::max() /
						  per_tramp_bytes))
						cave_need_bytes = per_tramp_bytes * n;
				} else if (sampling_enabled && sampling_cfg && request_exit_detour &&
					   sampling_cfg->mode ==
						   Sm120SamplingConfig::Mode::ThreadMap &&
					   sampling_cfg->thread_map_device &&
					   env_truthy(
						   "BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_PREFER_EXIT")) {
					// Prefer-exit for thread-map(device): patch multiple EXIT sites.
					// This improves stability for kernels where the last EXIT belongs
					// to an untaken path (e.g. multi-exit or dead code blocks).
					//
					// We only patch unconditional EXIT by default to avoid depending on
					// predicate state at the exit site.
					// For multi-exit prefer-exit patching we can safely include
					// predicated EXIT sites as well:
						// - patch site: `@P{pred} BRA` to trampoline (so we only detour when it
						//   would have exited)
						// - trampoline: run stub, then execute an unconditional EXIT
					//
						// This avoids "record=0" on kernels that only use predicated EXIT.
						// Default: prefer unconditional EXIT sites (more reliable coverage);
						// allow predicated sites only when explicitly enabled.
						const bool allow_pred_p0 = env_truthy(
							"BPFTIME_CUDA_SASS_DETOUR_ALLOW_PREDICATED_EXIT");
						const bool allow_pred_any = env_truthy(
							"BPFTIME_CUDA_SASS_DETOUR_ALLOW_PREDICATED_EXIT_ANY");
					// Cap the number of patched EXIT sites to keep cave size bounded.
					const size_t max_sites =
						env_u32("BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_EXIT_SITES_MAX")
							.value_or(16u);
						const size_t scan_insts =
							env_u32("BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_EXIT_SCAN_INSTS")
								.value_or(512u);
						// (rel, pred) where pred==7 means unpredicated EXIT, else @P{pred} EXIT.
						std::vector<std::pair<size_t, uint8_t>> unpred_sites;
						std::vector<std::pair<size_t, uint8_t>> pred_sites;
							size_t scanned = 0;
							for (size_t off = size;
							     off >= SASS_INST_BYTES && scanned < scan_insts;
							     off -= SASS_INST_BYTES, scanned++) {
								const size_t rel = off - SASS_INST_BYTES;
							auto w0 = read_u64_le(section_span, rel);
							if (!w0)
								continue;
							const uint16_t op16 =
								uint16_t(*w0 & 0xffffu);
							uint8_t pred = 0;
							bool neg = false;
							const auto kind =
								exit_kind_from_op16_sm120(op16, &pred, &neg);
							if (kind == Sm120ExitKind::ExitUnpred) {
								unpred_sites.emplace_back(
									rel, sm120_pack_pred(7u, false));
								continue;
							}
							const bool pred_ok =
								(pred == 0u && allow_pred_p0) ||
								(pred != 7u && allow_pred_any);
							if (pred_ok && kind == Sm120ExitKind::ExitPred) {
								pred_sites.emplace_back(
									rel, sm120_pack_pred(pred, neg));
								continue;
							}
						}
							// Prefer unconditional EXIT sites (executed by all threads).
							// Only include predicated EXIT sites when requested and when we still
							// have room (or when there are no unconditional sites at all).
							if (!unpred_sites.empty()) {
								threadmap_exit_sites_rel = std::move(unpred_sites);
							if (!pred_sites.empty() &&
							    threadmap_exit_sites_rel.size() < max_sites) {
								const size_t room =
									max_sites - threadmap_exit_sites_rel.size();
								if (pred_sites.size() > room)
									pred_sites.resize(room);
								threadmap_exit_sites_rel.insert(
									threadmap_exit_sites_rel.end(),
									pred_sites.begin(), pred_sites.end());
							}
							} else if (allow_pred_p0 || allow_pred_any) {
								threadmap_exit_sites_rel = std::move(pred_sites);
							}
					if (threadmap_exit_sites_rel.size() > max_sites &&
					    max_sites >= 2) {
							// Downsample: keep the first + last, then pick evenly spaced
							// middle points.
						std::vector<std::pair<size_t, uint8_t>> sampled;
							sampled.reserve(max_sites);
							sampled.push_back(threadmap_exit_sites_rel.front());
							const size_t n = threadmap_exit_sites_rel.size();
							const size_t mids = max_sites - 2;
							for (size_t i = 0; i < mids; i++) {
							const size_t idx =
								1 +
								((i + 1) * (n - 2)) /
									mids;
							sampled.push_back(
								threadmap_exit_sites_rel[idx]);
						}
						sampled.push_back(threadmap_exit_sites_rel.back());
						threadmap_exit_sites_rel = std::move(sampled);
					} else if (threadmap_exit_sites_rel.size() > max_sites) {
						threadmap_exit_sites_rel.resize(max_sites);
					}
					if (!threadmap_exit_sites_rel.empty()) {
						const size_t n =
							threadmap_exit_sites_rel.size();
						if (n <=
						    (std::numeric_limits<size_t>::max() /
						     per_tramp_bytes)) {
							cave_need_bytes =
								per_tramp_bytes *
								n;
						}
					}
					if (!threadmap_exit_sites_rel.empty() &&
					    env_truthy("BPFTIME_CUDA_SASS_DETOUR_DEBUG")) {
						SPDLOG_INFO(
							"SASS detour: thread_map prefer-exit: text='{}' func_id={} exit_sites={} max_sites={} cave_need_bytes={}",
							std::string(name), func_id,
							threadmap_exit_sites_rel.size(),
							(unsigned)max_sites,
							(unsigned long long)cave_need_bytes);
					}
				}
				section_span = std::span<const uint8_t>(elf_bytes.data() + start,
									size);
			cave_rel =
			find_tail_nop_cave_sm120(section_span, cave_need_bytes);
		if (!cave_rel) {
			// gap-cave: extend this `.text.*` section into the alignment
			// gap before the next section / section header table, without
			// relocating any other section.
			const size_t end = start + size;
			size_t next_off = elf_bytes.size();
			for (const auto &osh : shdrs) {
				if (osh.sh_offset == 0)
					continue;
				const size_t ooff = static_cast<size_t>(osh.sh_offset);
				if (ooff >= end && ooff < next_off)
					next_off = ooff;
			}
			if (shoff >= end && shoff < next_off)
				next_off = shoff;
			if (next_off > end && (next_off - end) >= cave_need_bytes) {
				cave_rel = size;
				sh.sh_size += cave_need_bytes;
			}
		}
		if (!cave_rel) {
			// insert-cave: if there's no tail NOP cave and no alignment gap,
			// shift the ELF file layout to create space at the end of this
			// `.text.*` section (hetGPU-style).
			if (!phdrs.empty()) {
				const size_t insert_off = start + size;
				if (insert_sm120_text_cave_by_shifting(
					    elf_bytes, ehdr, shdrs, phdrs,
					    insert_off, cave_need_bytes)) {
					shoff = static_cast<size_t>(ehdr.e_shoff);
					sh.sh_size += cave_need_bytes;
					cave_rel = size;

					// Recompute shstrtab/global indexes since all offsets
					// beyond `insert_off` have moved.
					auto elf_span = std::span<const uint8_t>(
						elf_bytes.data(), elf_bytes.size());
					auto shstr2 = get_shstrtab_from_shdrs(
						elf_span, ehdr, shdrs);
					if (!shstr2) {
						result.reason = "missing shstrtab after insert-cave";
						return std::nullopt;
					}
					shstrtab = shstr2;
					if (sampling_cfg && sampling_cfg->enabled) {
						global_nvinfo_reg_idx =
							build_nvinfo_global_regcount_index(
								elf_span, shdrs, *shstrtab,
								".nv.info");
						global_nvmerc_reg_idx =
							build_nvinfo_global_regcount_index(
								elf_span, shdrs, *shstrtab,
								".nv.merc.nv.info");
					}
				}
			}
		}
		if (!cave_rel) {
			result.skipped_text_sections++;
			continue;
		}
			// PcMarker: patch multiple sites within the same `.text.*` section.
			// Each site gets its own trampoline (replay original inst, then write a
			// marker record), laid out sequentially in a single cave.
				if (sampling_enabled && sampling_cfg &&
				    sampling_cfg->mode == Sm120SamplingConfig::Mode::PcMarker &&
				    layout && !pc_marker_sites_rel.empty()) {
				const size_t cave_base_rel = *cave_rel;
				const size_t n_sites = pc_marker_sites_rel.size();
				const auto ldcu_desc = find_ldcu64_desc_template_sm120(
					section_span,
					/*override_ur_base=*/sampling_cfg->desc_ur);

				std::vector<std::array<uint8_t, SASS_INST_BYTES>> orig(n_sites);
				std::vector<size_t> site_file_off(n_sites);
				for (size_t si = 0; si < n_sites; si++) {
					const size_t site_rel = pc_marker_sites_rel[si];
					site_file_off[si] = start + site_rel;
					std::memcpy(orig[si].data(),
						    elf_bytes.data() + site_file_off[si],
						    SASS_INST_BYTES);
				}

				bool ok = true;
				for (size_t si = 0; si < n_sites; si++) {
					const size_t site_rel = pc_marker_sites_rel[si];
					const size_t site_off = site_file_off[si];
					const size_t tramp_rel = cave_base_rel + si * per_tramp_bytes;
					const size_t tramp_file_off = start + tramp_rel;

					// Patch site with BRA to trampoline.
					const int64_t fwd_delta =
						static_cast<int64_t>(tramp_rel - site_rel);
					auto bra_fwd = encode_bra_sm120_unpred(fwd_delta);
					if (!bra_fwd ||
					    !write_sass_inst(std::span<uint8_t>(
								     elf_bytes.data(),
								     elf_bytes.size()),
							     site_off, bra_fwd->w0,
							     bra_fwd->w1)) {
						ok = false;
						break;
					}

					// Build per-site marker stub. marker_off is the detour site offset
					// (relative to the `.text.*` section start).
					const uint32_t marker_off =
						static_cast<uint32_t>(site_rel);
					auto marker_stub = build_sm120_pc_marker_stub(
						*sampling_cfg, *layout, tag, marker_off, ldcu_desc);
					if (marker_stub.empty()) {
						ok = false;
						break;
					}

					size_t tramp_write_off = tramp_file_off;
					const size_t pc0_base_file_off = site_off;

					uint64_t first_w0 = 0;
					std::memcpy(&first_w0, orig[si].data(),
						    sizeof(first_w0));
					const uint16_t op16 = uint16_t(first_w0 & 0xffffu);
					const bool first_is_unpred_bra = (op16 == 0x7947);
					std::optional<int64_t> first_bra_target;
					if (replay_n == 1 && first_is_unpred_bra)
						first_bra_target =
							decode_bra_delta_bytes_sm120(first_w0);
					const bool replay_prefix =
						!(replay_n == 1 && first_bra_target);

					auto write_prefix = [&](int64_t prefix_tramp_pc0) -> bool {
						for (size_t k = 0; k < replay_n; k++) {
							uint64_t w0 = 0;
							uint64_t w1 = 0;
							std::memcpy(&w0,
								    orig[si].data() +
									    k * SASS_INST_BYTES,
								    sizeof(w0));
							std::memcpy(&w1,
								    orig[si].data() +
									    k * SASS_INST_BYTES +
									    sizeof(w0),
								    sizeof(w1));
							// Relocate PC-relative control-flow immediates inside the replay
							// window. When we execute the prefix from the trampoline, the PC
							// changes by `prefix_tramp_pc0`, so any rel-imm instructions
							// (BRA/CALL) must be re-encoded to target the same original address.
							const uint16_t op16_i = uint16_t(w0 & 0xffffu);
							if (is_rel_imm36_ctrl_sm120(op16_i)) {
								if (auto old_delta =
									    decode_bra_delta_bytes_sm120(w0)) {
									const int64_t new_delta =
										*old_delta - prefix_tramp_pc0;
									if (auto nw0 =
										    patch_rel_imm36_in_w0_sm120(
											    w0, new_delta)) {
										w0 = *nw0;
									}
								}
							}
							if (!write_sass_inst(
								    std::span<uint8_t>(
									    elf_bytes.data(),
									    elf_bytes.size()),
								    tramp_write_off, w0, w1))
								return false;
							tramp_write_off += SASS_INST_BYTES;
						}
						return true;
					};

					auto write_stub = [&](const std::vector<SassInst> &v) -> bool {
						for (const auto &inst : v) {
							if (!write_sass_inst(
								    std::span<uint8_t>(
									    elf_bytes.data(),
									    elf_bytes.size()),
								    tramp_write_off, inst.w0,
								    inst.w1))
								return false;
							tramp_write_off += SASS_INST_BYTES;
						}
						return true;
					};

					// Prefer prefix-first so the marker is emitted *after* executing
					// the detoured instruction (minimal semantic change).
					if (replay_prefix) {
						if (!write_prefix(static_cast<int64_t>(
							    tramp_write_off - pc0_base_file_off))) {
							ok = false;
							break;
						}
					}
					if (!write_stub(marker_stub)) {
						ok = false;
						break;
					}

					const int64_t bra_pc = static_cast<int64_t>(
						tramp_write_off - pc0_base_file_off);
					const int64_t target_pc =
						first_bra_target
							? *first_bra_target
							: static_cast<int64_t>(
								  replay_n * SASS_INST_BYTES);
					const int64_t back_delta = target_pc - bra_pc;
					auto bra_back = encode_bra_sm120_unpred(back_delta);
					if (!bra_back ||
					    !write_sass_inst(std::span<uint8_t>(
								     elf_bytes.data(),
								     elf_bytes.size()),
							     tramp_write_off,
							     bra_back->w0,
							     bra_back->w1)) {
						ok = false;
						break;
					}
				}

				if (!ok) {
					// Restore original site instructions on failure.
					for (size_t si = 0; si < n_sites; si++) {
						std::memcpy(elf_bytes.data() + site_file_off[si],
							    orig[si].data(),
							    SASS_INST_BYTES);
					}
					result.skipped_text_sections++;
					continue;
				}

				result.patched_text_sections++;
					result.sampled_text_sections++;
					continue;
				}
				// ThreadMap prefer-exit: patch multiple EXIT sites and reuse the same
				// thread-map stub for each site.
					if (sampling_enabled && sampling_cfg && request_exit_detour &&
					    sampling_cfg->mode == Sm120SamplingConfig::Mode::ThreadMap &&
					    sampling_cfg->thread_map_device &&
					    env_truthy(
						    "BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_PREFER_EXIT") &&
					    !threadmap_exit_sites_rel.empty()) {
							if (env_truthy("BPFTIME_CUDA_SASS_DETOUR_DEBUG")) {
								std::string sites;
								for (size_t si = 0;
								     si < std::min<size_t>(threadmap_exit_sites_rel.size(), 6);
						     si++) {
							if (!sites.empty())
								sites += ",";
							const auto rel = threadmap_exit_sites_rel[si].first;
							const uint8_t code = threadmap_exit_sites_rel[si].second;
							const uint8_t p = sm120_unpack_pred(code);
							const bool n = sm120_unpack_pred_neg(code);
							sites += std::to_string((unsigned long long)rel);
							if (p != 7u) {
								sites += n ? "@!P" : "@P";
								sites += std::to_string(unsigned(p));
							}
						}
						SPDLOG_INFO(
							"SASS detour: thread_map prefer-exit: patching {} EXIT sites (first={}) text='{}' func_id={}",
							threadmap_exit_sites_rel.size(),
							sites, std::string(name),
							func_id);
					}
					const size_t cave_base_rel = *cave_rel;
					const size_t n_sites =
						threadmap_exit_sites_rel.size();
					std::vector<std::array<uint8_t, SASS_INST_BYTES>> orig(
						n_sites);
					std::vector<size_t> site_file_off(n_sites);
						for (size_t si = 0; si < n_sites; si++) {
							const size_t site_rel =
								threadmap_exit_sites_rel[si].first;
							site_file_off[si] = start + site_rel;
						std::memcpy(orig[si].data(),
							    elf_bytes.data() +
								    site_file_off
									    [si],
								    SASS_INST_BYTES);
						}

						// Prefer using the original EXIT site's control word (w1) for our injected
						// forward branches and internal stub gating branches.
						std::optional<uint64_t> bra_w1_fwd_template;
						if (!orig.empty()) {
							uint64_t w1_first = 0;
							std::memcpy(&w1_first, orig[0].data() + 8,
								    sizeof(w1_first));
							bra_w1_fwd_template = w1_first;
						}
						if (exit_thread_map_no_regcount && sampling_cfg) {
							stub_insts =
								build_sm120_thread_map_stub_no_regcount_at_exit(
									*sampling_cfg, func_id,
									exit_thread_map_scratch_regs,
									exit_stub_existing_desc_ur_base,
									bra_w1_fwd_template);
						}

						bool ok = true;
						for (size_t si = 0; si < n_sites; si++) {
								const size_t site_rel =
									threadmap_exit_sites_rel[si].first;
							const uint8_t site_pred_code =
								threadmap_exit_sites_rel[si].second;
							const size_t site_off = site_file_off[si];
							const size_t tramp_rel =
								cave_base_rel +
								si * per_tramp_bytes;
						const size_t tramp_file_off =
							start + tramp_rel;

						// Patch site with BRA to trampoline.
							const int64_t fwd_delta =
								static_cast<int64_t>(tramp_rel -
										     site_rel);
							const uint8_t site_pred = sm120_unpack_pred(site_pred_code);
							const bool site_neg = sm120_unpack_pred_neg(site_pred_code);
							const bool is_predicated = (site_pred != 7u);
							auto bra_fwd =
								is_predicated
									? encode_bra_sm120_pred(fwd_delta, site_pred,
												site_neg)
									: encode_bra_sm120_unpred(fwd_delta);
							uint64_t bra_w1 = 0;
							std::memcpy(&bra_w1, orig[si].data() + 8, sizeof(bra_w1));
							if (!bra_fwd ||
							    !write_sass_inst(
								    std::span<uint8_t>(
									    elf_bytes.data(),
									    elf_bytes.size()),
								    site_off, bra_fwd->w0,
								    bra_w1)) {
								ok = false;
								break;
							}

						// Build trampoline: [stub][orig EXIT][BRA back]
						size_t tramp_write_off =
							tramp_file_off;
						const size_t pc0_base_file_off =
							site_off;

						auto write_stub = [&](const std::vector<SassInst> &v)
							-> bool {
							for (const auto &inst : v) {
								if (!write_sass_inst(
									    std::span<uint8_t>(
										    elf_bytes.data(),
										    elf_bytes.size()),
									    tramp_write_off,
									    inst.w0, inst.w1))
									return false;
								tramp_write_off +=
									SASS_INST_BYTES;
							}
							return true;
						};
						if (!write_stub(stub_insts)) {
							ok = false;
							break;
						}

						// Replay the original EXIT instruction.
						uint64_t w0 = 0, w1 = 0;
						std::memcpy(&w0, orig[si].data(),
							    sizeof(w0));
							std::memcpy(&w1,
								    orig[si].data() + 8,
								    sizeof(w1));
							// For predicated EXIT sites, we only jump to the trampoline
							// when the predicate is true (@P{pred} BRA). To avoid depending on predicate
							// state after the stub, make the replayed EXIT unconditional.
							if (is_predicated) {
								w0 = (w0 & ~uint64_t(0xffffu)) |
								     uint64_t(0x794du);
							}
						if (!write_sass_inst(
							    std::span<uint8_t>(
								    elf_bytes.data(),
								    elf_bytes.size()),
							    tramp_write_off, w0, w1)) {
							ok = false;
							break;
						}
						tramp_write_off += SASS_INST_BYTES;

						// Branch back (won't execute on EXIT, but keeps layout consistent).
						const int64_t bra_pc = static_cast<int64_t>(
							tramp_write_off - pc0_base_file_off);
						const int64_t target_pc = static_cast<int64_t>(
							SASS_INST_BYTES);
						const int64_t back_delta = target_pc - bra_pc;
						auto bra_back =
							encode_bra_sm120_unpred(
								back_delta);
						if (!bra_back ||
						    !write_sass_inst(
							    std::span<uint8_t>(
								    elf_bytes.data(),
								    elf_bytes.size()),
							    tramp_write_off,
							    bra_back->w0,
							    bra_back->w1)) {
							ok = false;
							break;
						}
					}

					if (!ok) {
						// Restore patched sites.
						for (size_t si = 0; si < n_sites;
						     si++) {
							const size_t site_off =
								site_file_off[si];
							std::memcpy(elf_bytes.data() + site_off,
								    orig[si].data(),
								    SASS_INST_BYTES);
						}
						result.skipped_text_sections++;
						continue;
					}

					result.patched_text_sections++;
					result.sampled_text_sections++;
					continue;
				}
				const size_t tramp_off2 = *cave_rel;
				const size_t tramp_file_off2 = start + tramp_off2;

			// Patch entry with BRA to trampoline (PC=0).
			const int64_t fwd_delta =
				static_cast<int64_t>(tramp_off2 - entry_rel);
					const uint8_t pred = sm120_unpack_pred(detour_site_pred);
					const bool neg = sm120_unpack_pred_neg(detour_site_pred);
					auto bra_fwd =
						(pred == 7u)
							? encode_bra_sm120_unpred(fwd_delta)
							: encode_bra_sm120_pred(fwd_delta, pred,
										neg);
				if (!bra_fwd) {
					result.skipped_text_sections++;
					continue;
				}
			if (!write_sass_inst(
				    std::span<uint8_t>(elf_bytes.data(),
						       elf_bytes.size()),
				    entry_file_off, bra_fwd->w0, bra_fwd->w1)) {
				result.skipped_text_sections++;
				continue;
			}

			// Trampoline order matters.
			//
			// Default: execute the sampling stub *before* replaying the kernel
			// prologue prefix. Many cubin-only kernels initialize UR registers and
			// predicates in the first few instructions; running our stub after the
			// prologue can clobber that state and crash workloads (e.g. vLLM/cutlass).
			//
			// Exception (identify/closure control): controlled stubs may need the
			// very first prologue instruction(s) (e.g. LDC R1, c[0][...]) to run
			// before using constant-memory based operations (LDCU/LDG). In that
			// case we place the replay prefix first, then the controlled stub.
			//
			// Layout:
			// - default:   [stub...][orig_prefix...][BRA back]
			// - controlled:[orig_prefix...][stub...][BRA back]
			size_t tramp_write_off = tramp_file_off2;

			// Replay the prologue prefix. If the entry instruction is an
			// unconditional BRA, do NOT replay it from the trampoline (PC-relative
			// immediate); instead, run the stub and branch to the original target.
			const uint16_t op16 = uint16_t(first_w0 & 0xffffu);
			const bool first_is_unpred_bra = (op16 == 0x7947);
			std::optional<int64_t> first_bra_target;
			if (replay_n == 1 && first_is_unpred_bra)
				first_bra_target = decode_bra_delta_bytes_sm120(first_w0);

						const bool first_is_exit =
							(exit_kind_from_op16_sm120(op16, nullptr,
										   nullptr) !=
							 Sm120ExitKind::NotExit);
					const bool force_prefix_first =
						// reg255 thread-map: when we detour at a pre-exit NOP (instead of
						// the EXIT itself), we want the trampoline to replay the NOP
						// before running the stub, then fall through to the original EXIT.
						reg255_pre_exit_detour;
					const bool prefer_prefix_first =
						force_prefix_first ||
						(!first_is_exit &&
						 ((sampling_cfg && sampling_cfg->control_enabled &&
						   !stub_no_regcount_fallback) ||
						  (sampling_cfg &&
						   sampling_cfg->mode ==
							   Sm120SamplingConfig::Mode::ThreadMap &&
						   sampling_cfg->thread_map_device &&
						   !env_truthy(
							   "BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_USE_UR"))));
				const bool replay_prefix =
					!(replay_n == 1 && first_bra_target);

					auto write_prefix = [&](int64_t prefix_tramp_pc0) -> bool {
					// Relocate PC-relative control-flow immediates inside the replay
					// window. When we execute the prefix from the trampoline, the PC
					// changes by `prefix_tramp_pc0`, so any rel-imm instructions (BRA/CALL)
					// must be re-encoded to target the same original address.
					bool prefix_write_ok = true;
					for (size_t inst_off = 0; inst_off < prefix_bytes.size();
					     inst_off += SASS_INST_BYTES) {
						uint64_t w0 = 0, w1 = 0;
					std::memcpy(&w0, prefix_bytes.data() + inst_off,
						    sizeof(w0));
					std::memcpy(&w1,
						    prefix_bytes.data() + inst_off + 8,
						    sizeof(w1));
					const uint16_t op16_i = uint16_t(w0 & 0xffffu);
					if (is_rel_imm36_ctrl_sm120(op16_i)) {
						if (auto old_delta =
							    decode_bra_delta_bytes_sm120(w0)) {
							const int64_t new_delta =
								*old_delta - prefix_tramp_pc0;
							if (auto nw0 =
								    patch_rel_imm36_in_w0_sm120(
									    w0, new_delta)) {
								w0 = *nw0;
							}
						}
					}
						if (!write_sass_inst(
							    std::span<uint8_t>(
								    elf_bytes.data(),
								    elf_bytes.size()),
							    tramp_write_off, w0, w1)) {
						prefix_write_ok = false;
						break;
					}
						tramp_write_off += SASS_INST_BYTES;
					}
					return prefix_write_ok;
				};

				auto write_stub = [&]() -> bool {
					for (const auto &inst : stub_insts) {
						if (!write_sass_inst(
							    std::span<uint8_t>(elf_bytes.data(),
									       elf_bytes.size()),
							    tramp_write_off, inst.w0, inst.w1)) {
							return false;
						}
						tramp_write_off += SASS_INST_BYTES;
					}
					return true;
				};

					if (prefer_prefix_first && replay_prefix) {
						// Prefix at the trampoline entry (pc0 = tramp_off2).
						if (!write_prefix(static_cast<int64_t>(
							    tramp_write_off - pc0_base_file_off))) {
							// Restore entry on failure.
							std::memcpy(elf_bytes.data() + entry_file_off,
								    first_inst.data(),
								    SASS_INST_BYTES);
							result.skipped_text_sections++;
							continue;
						}
						if (!write_stub()) {
							std::memcpy(elf_bytes.data() + entry_file_off,
								    first_inst.data(),
								    SASS_INST_BYTES);
							result.skipped_text_sections++;
							continue;
						}
					} else {
						// Default order: stub then prefix (or stub only for first BRA).
						if (!write_stub()) {
							std::memcpy(elf_bytes.data() + entry_file_off,
								    first_inst.data(),
								    SASS_INST_BYTES);
							result.skipped_text_sections++;
							continue;
						}
						if (replay_prefix) {
							// Prefix starts after the stub (pc0 = tramp_off2 + stub_bytes).
							if (!write_prefix(
								    static_cast<int64_t>(
									    tramp_write_off -
									    pc0_base_file_off))) {
								std::memcpy(elf_bytes.data() +
										    entry_file_off,
									    first_inst.data(),
									    SASS_INST_BYTES);
								result.skipped_text_sections++;
								continue;
							}
					}
				}

				// Return from the trampoline:
				// - default: entry + replay_n*0x10 (next instruction after replay window)
				// - if the entry instruction is an unconditional BRA: branch to its target
				//   (so our stub executes once, then control flow matches the original)
				const int64_t bra_pc =
					static_cast<int64_t>(tramp_write_off - pc0_base_file_off);
				const int64_t target_pc = first_bra_target
								  ? *first_bra_target
								  : static_cast<int64_t>(
									    replay_n * SASS_INST_BYTES);
				const int64_t back_delta = target_pc - bra_pc;
			auto bra_back = encode_bra_sm120_unpred(back_delta);
			if (!bra_back ||
				    !write_sass_inst(std::span<uint8_t>(elf_bytes.data(),
								       elf_bytes.size()),
						     tramp_write_off,
						     bra_back->w0, bra_back->w1)) {
				// Restore entry on failure.
				std::memcpy(elf_bytes.data() + entry_file_off, first_inst.data(),
					    SASS_INST_BYTES);
				result.skipped_text_sections++;
				continue;
			}

			result.patched_text_sections++;
			if (sampling_enabled)
				result.sampled_text_sections++;
		}

		if (result.patched_text_sections == 0 && result.reason.empty()) {
				if (seen_text_sections == 0) {
					result.reason = "no .text.* sections";
				} else if (!section_name_filter.empty()) {
					result.reason =
						"no .text.* sections matched filter (text/nvinfo/symtab)";
				} else {
					result.reason = "no patchable .text.* sections";
				}
			}

		if (!write_updated_headers(elf_bytes, ehdr, phdrs, shdrs)) {
			result.reason = "unable to write updated ELF headers";
			return result;
		}

		if (dump_dir && result.patched_text_sections > 0 && !dumped) {
			auto span = std::span<const uint8_t>(elf_bytes.data(),
							     elf_bytes.size());
		const uint64_t h = fnv1a64(span);
		std::string fn = "sm120_detoured_" +
				 std::to_string(result.patched_text_sections) +
				 "_" + std::to_string(h) + ".cubin";
			maybe_dump_file(dump_dir, fn, span);
			dumped = true;
		}

		if (dump_dir && dump_unpatched && result.patched_text_sections == 0 &&
		    !dumped) {
			auto span = std::span<const uint8_t>(elf_bytes.data(),
							     elf_bytes.size());
			const uint64_t h = fnv1a64(span);
			const std::string why =
				sanitize_filename_component(result.reason, 64);
			std::string fn =
				"sm120_unpatched_" + why + "_" + std::to_string(h) +
				".cubin.raw";
			maybe_dump_file(dump_dir, fn, span);
			dumped = true;
		}

		return result;
	}

std::vector<std::array<uint64_t, 2>>
sm120_build_thread_map_stub_for_test(const Sm120SamplingConfig &cfg,
				     uint32_t old_regcount)
{
	auto layout = compute_sm120_layout(old_regcount,
					   Sm120SamplingConfig::Mode::ThreadMap);
	if (!layout)
		return {};
	auto insts = build_sm120_thread_map_stub(cfg, *layout, std::nullopt);
	std::vector<std::array<uint64_t, 2>> out;
	out.reserve(insts.size());
	for (const auto &i : insts)
		out.push_back(std::array<uint64_t, 2> { i.w0, i.w1 });
	return out;
}

} // namespace bpftime::attach::sass_detour
