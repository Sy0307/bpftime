#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include <optional>
#include <unordered_set>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace bpftime::attach::sass_detour
{

// Control/header layout at the beginning of the SM120 SASS sampler buffer.
// Used by the "identify/tag closure" logic to arm identify windows and to
// switch to a target func_id within the same process (single-run closure).
constexpr uint32_t kSm120SassControlMagic = 0x53535042u; // "BPSS" (LE)
constexpr uint32_t kSm120SassControlVersion = 1u;
constexpr uint32_t kSm120SassControlHeaderBytes = 0x40u;
constexpr uint32_t kSm120SassControlSlotsOffset = 0x20u;
// Keep slots within the first 0x40 bytes so sampler data can stay at an imm8
// offset. Collisions are acceptable (all SMs of the same kernel write the same
// func_id in Identify mode).
constexpr uint32_t kSm120SassControlSlotsCount = 8u;
constexpr uint32_t kSm120SassControlDataOffset = 0x40u;

enum class Sm120SassControlMode : uint32_t {
	Off = 0,
	Identify = 1,
	Target = 2,
};

struct Sm120SassControlHeader {
	uint32_t magic;
	uint32_t version;
	uint32_t enable; // 0=off, 1=on
	uint32_t mode; // Sm120SassControlMode
	uint32_t epoch;
	uint32_t target_func_id; // valid in Target mode
	uint32_t reserved0;
	uint32_t reserved1;
};
static_assert(sizeof(Sm120SassControlHeader) == 0x20);

struct Sm120SamplingConfig {
	enum class Mode : uint8_t {
		// Writes a 256-entry u32 bitmap: bitmap[smid] = 1.
		SmidBitmap = 0,
		// Writes a CTA->SMID map: out[ctaid.x] = smid_raw.
		CtaSmid = 1,
		// Writes per-warp records (indexed by ctaid.x and warp_id).
		WarpMap = 2,
		// Writes per-thread records (indexed by ctaid.x and tid.x).
		ThreadMap = 3,
		// Writes per-hit marker events into a ring buffer (best-effort hotspot):
		// each patched point emits a record `{seq, marker_off, tag, smid_raw, ctaid_x, tid_x}`.
		PcMarker = 4,
	};

	// Enables a minimal sampler that writes into `sample_buffer_device_ptr`
	// (device global memory).
	//
	// Buffer layout (little-endian):
	// - `SmidBitmap`: u32 bitmap[256]
	// - `CtaSmid`:   u32 cta_smid[max_records] (init to 0xffffffff)
	// - `WarpMap`:   u32 smid_slots[max_records * 32] (init to 0xffffffff).
	//               Device writes only the low byte (smid_lo8) into the slot.
	// - `ThreadMap`: same device buffer as `WarpMap`, but the host expands each
	//               warp slot into 32 lane records (tid_x = warp_id*32+lane).
	// - `PcMarker`:  u32 write_idx; u32 pad[3]; then `marker_ring_entries` records:
	//               record (6*u32): `{seq, marker_off, tag, smid_raw, ctaid_x, tid_x}`
	bool enabled = false;
	Mode mode = Mode::SmidBitmap;
	uint64_t sample_buffer_device_ptr = 0;
	// Byte offset from `sample_buffer_device_ptr` to the sampler data region.
	// When non-zero, the buffer begins with a control/header block.
	uint32_t buffer_data_offset = 0;
	// When true, build "controlled" stubs that consult the control header
	// (enable/mode/target_func_id) to perform identify/targeted sampling.
	bool control_enabled = false;
	uint32_t max_records = 0;
	// If `mode==ThreadMap`:
	// - false: device writes per-warp slots, host expands to per-thread.
	// - true:  device writes per-thread bytes directly.
	bool thread_map_device = false;
	// If `mode==ThreadMap && thread_map_device==true`:
	// - lane0_only: only lane0 writes (debug/bring-up).
	// - warp0_only: only warp0 writes (reduce density by 32x).
	// - stride4: write each thread at aligned offset (idx*4), trading size for
	//   robustness on some kernels.
	bool thread_map_device_lane0_only = false;
	bool thread_map_device_warp0_only = false;
	bool thread_map_device_stride4 = false;
	// If `mode==ThreadMap && thread_map_device==true && max_records!=0`:
	// - false (default): ctaid_mod = ctaid.x & (max_records-1) (mask; may collide)
	// - true: skip stores when ctaid.x >= max_records (clamp; reduces contention)
	bool thread_map_device_cta_clamp = false;
	// Debug: build the per-thread mapping index but don't write anything.
	bool thread_map_device_no_store = false;
	// regcount=255 bring-up: reserve a small spill area after the sampler data.
	// When enabled, the reg255 EXIT stub can save/restore a small subset of GPRs
	// and predicates without needing extra address scratch (uses [RZ+UR+imm]).
	//
	// Spill base is at:
	//   sample_buffer_device_ptr + buffer_data_offset + data_bytes
	// where data_bytes depends on mode/max_records/stride (same as host allocation).
	bool reg255_thread_map_spill_enable = false;
	// When true, the spill area is "per-thread": each (ctaid_mod, tid_x) has an
	// independent spill slot, so multi-thread stubs can save/restore without
	// races.
	bool reg255_thread_map_spill_per_thread = false;
	// Bytes per spill slot when `reg255_thread_map_spill_per_thread==true`.
	// Must be >= 0x10 (we use 0x10 bytes in the minimal implementation).
	uint32_t reg255_thread_map_spill_stride_bytes = 0;
	// Total bytes reserved for the spill area (host allocation size).
	uint32_t reg255_thread_map_spill_bytes = 0;
	// If `mode==PcMarker`:
	// - ring buffer capacity (power-of-two, mask-based indexing).
	// - offsets (relative to `.text.*` section start) to patch in addition to the
	//   default entry detour; empty => patch entry only.
	// - density gates (similar to ThreadMap device bring-up).
	uint32_t marker_ring_entries = 0;
	std::vector<uint32_t> marker_offsets;
	bool marker_lane0_only = true;
	bool marker_warp0_only = false;
	bool marker_cta_clamp = true;
	// Which UR register pair to use as global memory descriptor base.
	// Default 4 uses the conventional UR4/UR5 pair (most robust).
	uint8_t desc_ur = 4;
};

struct InstrumentedKernelInfo {
	std::string text_section_name; // e.g. ".text.foo"
	std::string kernel_name; // e.g. "foo"
	uint32_t tag = 0;
	uint32_t old_regcount = 0;
	uint32_t new_regcount = 0;
};

struct DetourResult {
	// Best-effort stable identifier of the SM120 ELF code object (hash of the
	// meaningful ELF extent). Used to scope func_id caches and avoid cross-module
	// func_id collisions.
	uint32_t image_id = 0;
	size_t patched_text_sections = 0;
	size_t skipped_text_sections = 0;
	size_t sampled_text_sections = 0;
	std::vector<InstrumentedKernelInfo> sampled_kernels;
	std::string reason;
};

// Apply a minimal SASS detour to every matching `.text.*` section in a CUBIN ELF
// image (SM120 only for now):
// - overwrite the first instruction with a `BRA` to a trampoline placed in
//   slack space at the end of the section
// - trampoline replays the original first instruction and `BRA`s back
//
// This validates “cubin-only” SASS rewriting without needing PTX.
std::optional<DetourResult> apply_elf_text_detours_sm120(
	std::vector<uint8_t> &elf_bytes, std::string_view section_name_filter,
	std::string_view sample_section_filter,
	const Sm120SamplingConfig *sampling_cfg = nullptr,
	const std::unordered_set<uint32_t> *extra_filter_func_ids = nullptr,
	bool patch_all_is_fallback = false);

// Infer SM version from CUBIN ELF e_flags (best-effort). Returns e.g. 120 for
// sm_120.
std::optional<int> infer_sm_version_from_elf(std::span<const uint8_t> elf);

#ifdef BPFTIME_SASS_DETOUR_TESTING
// Test helper: build the SM120 thread-map sampler stub (instruction words),
// without requiring a full CUBIN patch/cave.
std::vector<std::array<uint64_t, 2>>
sm120_build_thread_map_stub_for_test(const Sm120SamplingConfig &cfg,
				     uint32_t old_regcount);
#endif

} // namespace bpftime::attach::sass_detour
