#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace bpftime::attach::sass_map
{

struct SourceLocation {
	std::string file;
	uint32_t line = 0;
	uint32_t column = 0;
	bool is_stmt = false;
};

struct LineEntry {
	uint64_t address = 0;
	SourceLocation loc;
};

struct LineTable {
	std::vector<LineEntry> entries;
};

// Parse ELF(CUBIN) and extract a DWARF .debug_line mapping (address -> source).
// Returns nullopt if the input isn't an ELF or doesn't contain .debug_line.
std::optional<LineTable>
parse_elf_debug_line(std::span<const uint8_t> elf_bytes);

} // namespace bpftime::attach::sass_map

