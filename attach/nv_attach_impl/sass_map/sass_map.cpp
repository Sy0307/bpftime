#include "sass_map.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <optional>
#include <string_view>
#include <unordered_map>

namespace bpftime::attach::sass_map
{
namespace
{
constexpr uint8_t ELF_MAGIC[4] = { 0x7f, 'E', 'L', 'F' };

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

template <typename T>
static bool read_pod(std::span<const uint8_t> bytes, size_t offset, T &out)
{
	if (offset + sizeof(T) > bytes.size())
		return false;
	std::memcpy(&out, bytes.data() + offset, sizeof(T));
	return true;
}

static std::optional<std::span<const uint8_t>>
find_elf_section(std::span<const uint8_t> elf, std::string_view section_name)
{
	Elf64_Ehdr ehdr {};
	if (!read_pod(elf, 0, ehdr))
		return std::nullopt;
	if (std::memcmp(ehdr.e_ident, ELF_MAGIC, sizeof(ELF_MAGIC)) != 0)
		return std::nullopt;
	// Only support ELF64 little-endian for now.
	if (ehdr.e_ident[4] != 2 /* ELFCLASS64 */ ||
	    ehdr.e_ident[5] != 1 /* ELFDATA2LSB */)
		return std::nullopt;

	if (ehdr.e_shoff == 0 || ehdr.e_shentsize == 0 || ehdr.e_shnum == 0)
		return std::nullopt;
	if (ehdr.e_shentsize != sizeof(Elf64_Shdr))
		return std::nullopt;

	const size_t shoff = static_cast<size_t>(ehdr.e_shoff);
	const size_t shnum = static_cast<size_t>(ehdr.e_shnum);
	const size_t shstrndx = static_cast<size_t>(ehdr.e_shstrndx);
	if (shoff + shnum * sizeof(Elf64_Shdr) > elf.size())
		return std::nullopt;
	if (shstrndx >= shnum)
		return std::nullopt;

	Elf64_Shdr shstr {};
	if (!read_pod(elf, shoff + shstrndx * sizeof(Elf64_Shdr), shstr))
		return std::nullopt;
	if (shstr.sh_offset + shstr.sh_size > elf.size())
		return std::nullopt;
	auto shstrtab = elf.subspan(static_cast<size_t>(shstr.sh_offset),
				    static_cast<size_t>(shstr.sh_size));

	for (size_t i = 0; i < shnum; i++) {
		Elf64_Shdr sh {};
		if (!read_pod(elf, shoff + i * sizeof(Elf64_Shdr), sh))
			return std::nullopt;
		if (sh.sh_name >= shstrtab.size())
			continue;
		const char *name_c =
			reinterpret_cast<const char *>(shstrtab.data() + sh.sh_name);
		const size_t max_len = shstrtab.size() - sh.sh_name;
		const size_t len = strnlen(name_c, max_len);
		std::string_view name(name_c, len);
		if (name != section_name)
			continue;
		if (sh.sh_offset + sh.sh_size > elf.size())
			return std::nullopt;
		return elf.subspan(static_cast<size_t>(sh.sh_offset),
				   static_cast<size_t>(sh.sh_size));
	}
	return std::nullopt;
}

struct UlebResult {
	uint64_t value;
	size_t bytes;
};

static std::optional<UlebResult> decode_uleb(std::span<const uint8_t> data,
					     size_t offset)
{
	uint64_t value = 0;
	uint32_t shift = 0;
	size_t consumed = 0;
	while (offset + consumed < data.size()) {
		uint8_t b = data[offset + consumed];
		value |= (uint64_t)(b & 0x7f) << shift;
		consumed++;
		if ((b & 0x80) == 0)
			return UlebResult { value, consumed };
		shift += 7;
		if (shift > 63)
			return std::nullopt;
	}
	return std::nullopt;
}

struct SlebResult {
	int64_t value;
	size_t bytes;
};

static std::optional<SlebResult> decode_sleb(std::span<const uint8_t> data,
					     size_t offset)
{
	int64_t value = 0;
	uint32_t shift = 0;
	size_t consumed = 0;
	uint8_t b = 0;
	while (offset + consumed < data.size()) {
		b = data[offset + consumed];
		value |= (int64_t)(b & 0x7f) << shift;
		shift += 7;
		consumed++;
		if ((b & 0x80) == 0)
			break;
		if (shift > 63)
			return std::nullopt;
	}
	if (consumed == 0)
		return std::nullopt;
	// sign extend if needed
	if (shift < 64 && (b & 0x40) != 0)
		value |= -((int64_t)1 << shift);
	return SlebResult { value, consumed };
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

struct DebugLineHeader {
	uint8_t min_insn_len = 1;
	uint8_t default_is_stmt = 1;
	int8_t line_base = -5;
	uint8_t line_range = 14;
	uint8_t opcode_base = 13;
};

struct FileEntry {
	std::string name;
	uint64_t dir_index = 0;
};

static bool parse_debug_line_header_and_tables(
	std::span<const uint8_t> unit, uint16_t version, size_t header_length,
	size_t &line_program_offset, DebugLineHeader &hdr,
	std::vector<std::string> &include_dirs, std::vector<FileEntry> &files)
{
	// unit points to the bytes right after (unit_length, version, header_length),
	// i.e. "header" payload of length header_length.
	if (header_length < 5 || unit.size() < header_length)
		return false;
	size_t off = 0;
	hdr.min_insn_len = unit[off++];
	hdr.default_is_stmt = unit[off++];
	hdr.line_base = static_cast<int8_t>(unit[off++]);
	hdr.line_range = unit[off++];
	hdr.opcode_base = unit[off++];
	if (hdr.opcode_base == 0)
		return false;

	// standard_opcode_lengths[opcode_base-1]
	if (off + (hdr.opcode_base - 1) > header_length)
		return false;
	off += (hdr.opcode_base - 1);

	// include directories: sequence of null-terminated strings, terminated by empty string
	while (off < header_length) {
		auto s = read_cstr(unit, off);
		if (!s)
			return false;
		off += s->size() + 1;
		if (s->empty())
			break;
		include_dirs.emplace_back(*s);
	}

	// file names: sequence of entries, terminated by empty string
	while (off < header_length) {
		auto name = read_cstr(unit, off);
		if (!name)
			return false;
		off += name->size() + 1;
		if (name->empty())
			break;

		// dir index (ULEB128), time (ULEB128), size (ULEB128)
		auto dir = decode_uleb(unit, off);
		if (!dir)
			return false;
		off += dir->bytes;
		auto time = decode_uleb(unit, off);
		if (!time)
			return false;
		off += time->bytes;
		auto size = decode_uleb(unit, off);
		if (!size)
			return false;
		off += size->bytes;

		FileEntry fe;
		fe.name = std::string(*name);
		fe.dir_index = dir->value;
		files.emplace_back(std::move(fe));
	}

	line_program_offset = header_length;
	return true;
}

static std::optional<std::string>
resolve_file(uint64_t file_index, const std::vector<std::string> &include_dirs,
	     const std::vector<FileEntry> &files)
{
	// DWARF file index is 1-based.
	if (file_index == 0)
		return std::nullopt;
	const size_t idx = static_cast<size_t>(file_index - 1);
	if (idx >= files.size())
		return std::nullopt;
	const auto &f = files[idx];
	if (f.dir_index == 0)
		return f.name;
	const size_t d = static_cast<size_t>(f.dir_index - 1);
	if (d >= include_dirs.size())
		return f.name;
	if (include_dirs[d].empty())
		return f.name;
	return include_dirs[d] + "/" + f.name;
}

static std::optional<LineTable>
parse_debug_line_unit(std::span<const uint8_t> unit_payload, uint16_t version)
{
	if (unit_payload.size() < 4)
		return std::nullopt;

	// header_length is 4 bytes for DWARF 2/3, 8 bytes for DWARF 64-bit with version >= 5.
	// For CUDA CUBINs we expect DWARF2/3 style (header_length u32).
	uint32_t header_length = 0;
	std::memcpy(&header_length, unit_payload.data(), sizeof(header_length));
	size_t header_len = header_length;
	if (unit_payload.size() < 4 + header_len)
		return std::nullopt;

	std::span<const uint8_t> header_bytes =
		unit_payload.subspan(4, header_len);
	std::span<const uint8_t> program_bytes =
		unit_payload.subspan(4 + header_len);

	DebugLineHeader hdr {};
	std::vector<std::string> include_dirs;
	std::vector<FileEntry> files;
	size_t program_off = 0;
	if (!parse_debug_line_header_and_tables(header_bytes, version, header_len,
						program_off, hdr, include_dirs,
						files))
		return std::nullopt;

	LineTable table;

	// Line number program state machine (DWARF 2/3-ish).
	uint64_t address = 0;
	uint64_t file = 1;
	uint32_t line = 1;
	uint32_t column = 0;
	bool is_stmt = hdr.default_is_stmt != 0;

	auto emit_row = [&]() {
		auto resolved = resolve_file(file, include_dirs, files);
		SourceLocation loc;
		loc.file = resolved ? *resolved : "unknown";
		loc.line = line;
		loc.column = column;
		loc.is_stmt = is_stmt;
		table.entries.push_back(LineEntry { address, std::move(loc) });
	};

	size_t off = 0;
	while (off < program_bytes.size()) {
		uint8_t opcode = program_bytes[off++];
		if (opcode == 0) {
			// extended opcode
			auto len = decode_uleb(program_bytes, off);
			if (!len)
				return std::nullopt;
			off += len->bytes;
			if (len->value == 0)
				continue;
			if (off >= program_bytes.size())
				return std::nullopt;
			uint8_t ext = program_bytes[off++];
			const size_t ext_payload_len =
				static_cast<size_t>(len->value - 1);
			if (off + ext_payload_len > program_bytes.size())
				return std::nullopt;
			switch (ext) {
			case 1: // DW_LNE_end_sequence
				// reset state per spec
				address = 0;
				file = 1;
				line = 1;
				column = 0;
				is_stmt = hdr.default_is_stmt != 0;
				break;
			case 2: // DW_LNE_set_address
				if (ext_payload_len == 8) {
					uint64_t a = 0;
					std::memcpy(&a, program_bytes.data() + off,
						    sizeof(a));
					address = a;
				} else if (ext_payload_len == 4) {
					uint32_t a = 0;
					std::memcpy(&a, program_bytes.data() + off,
						    sizeof(a));
					address = a;
				}
				break;
			case 3: // DW_LNE_define_file (rare)
				// ignore for now
				break;
			default:
				break;
			}
			off += ext_payload_len;
			continue;
		}

		if (opcode < hdr.opcode_base) {
			switch (opcode) {
			case 1: // DW_LNS_copy
				emit_row();
				break;
			case 2: { // DW_LNS_advance_pc
				auto v = decode_uleb(program_bytes, off);
				if (!v)
					return std::nullopt;
				off += v->bytes;
				address += v->value * hdr.min_insn_len;
				break;
			}
			case 3: { // DW_LNS_advance_line
				auto v = decode_sleb(program_bytes, off);
				if (!v)
					return std::nullopt;
				off += v->bytes;
				line = static_cast<uint32_t>(
					static_cast<int64_t>(line) + v->value);
				break;
			}
			case 4: { // DW_LNS_set_file
				auto v = decode_uleb(program_bytes, off);
				if (!v)
					return std::nullopt;
				off += v->bytes;
				file = v->value;
				break;
			}
			case 5: { // DW_LNS_set_column
				auto v = decode_uleb(program_bytes, off);
				if (!v)
					return std::nullopt;
				off += v->bytes;
				column = static_cast<uint32_t>(v->value);
				break;
			}
			case 6: // DW_LNS_negate_stmt
				is_stmt = !is_stmt;
				break;
			case 8: { // DW_LNS_const_add_pc
				const uint8_t adj = 255 - hdr.opcode_base;
				const uint64_t inc =
					(adj / hdr.line_range) * hdr.min_insn_len;
				address += inc;
				break;
			}
			case 9: { // DW_LNS_fixed_advance_pc
				if (off + 2 > program_bytes.size())
					return std::nullopt;
				uint16_t v = 0;
				std::memcpy(&v, program_bytes.data() + off, sizeof(v));
				off += 2;
				address += v;
				break;
			}
			default: {
				// Unsupported standard opcode: skip its operands based on standard lengths table.
				// We don't have the per-op lengths here; for CUDA use-cases this is usually fine.
				break;
			}
			}
			continue;
		}

		// special opcode
		const uint8_t adj = opcode - hdr.opcode_base;
		const uint64_t addr_inc =
			(adj / hdr.line_range) * hdr.min_insn_len;
		const int64_t line_inc =
			static_cast<int64_t>(hdr.line_base) +
			static_cast<int64_t>(adj % hdr.line_range);
		address += addr_inc;
		line = static_cast<uint32_t>(static_cast<int64_t>(line) + line_inc);
		emit_row();
	}

	return table;
}

static std::optional<LineTable> parse_debug_line(std::span<const uint8_t> sec)
{
	size_t off = 0;
	LineTable out;

	while (off + 4 <= sec.size()) {
		uint32_t unit_length = 0;
		std::memcpy(&unit_length, sec.data() + off, sizeof(unit_length));
		off += 4;

		uint64_t length64 = unit_length;
		if (unit_length == 0xffffffff) {
			// DWARF64
			if (off + 8 > sec.size())
				return std::nullopt;
			std::memcpy(&length64, sec.data() + off, sizeof(length64));
			off += 8;
		}
		if (length64 == 0)
			break;
		if (off + length64 > sec.size())
			break;

		if (off + 2 > sec.size())
			return std::nullopt;
		uint16_t version = 0;
		std::memcpy(&version, sec.data() + off, sizeof(version));
		off += 2;
		const size_t unit_payload_len = static_cast<size_t>(length64 - 2);
		if (off + unit_payload_len > sec.size())
			return std::nullopt;

		auto unit_payload = sec.subspan(off, unit_payload_len);
		off += unit_payload_len;

		auto table = parse_debug_line_unit(unit_payload, version);
		if (!table)
			continue;
		out.entries.insert(out.entries.end(),
				   std::make_move_iterator(table->entries.begin()),
				   std::make_move_iterator(table->entries.end()));
	}

	if (out.entries.empty())
		return std::nullopt;
	return out;
}
} // namespace

std::optional<LineTable>
parse_elf_debug_line(std::span<const uint8_t> elf_bytes)
{
	auto sec = find_elf_section(elf_bytes, ".debug_line");
	if (!sec)
		return std::nullopt;
	return parse_debug_line(*sec);
}

} // namespace bpftime::attach::sass_map

