# bpftime GPU SASS ↔ PTX/LLVM Mapping: Design Notes (dev/sass)

This document sketches how bpftime can grow from **PTX-string rewriting** to a
pipeline that can also **understand SASS (machine code)** and build robust
**bidirectional mappings**:

- SASS PC/address ↔ (PTX file,line,column)
- SASS PC/address ↔ (bpftime “injected region”, attach entry, eBPF instruction index)
- (optional) SASS ↔ LLVM IR metadata, enabling profiling/replay workflows

Reference inspiration: hetGPU `tmatmul` commit `c7b91063...` (“native SASS parsing
and LLVM inlining infrastructure”).

For a practical “vLLM + Qwen3 + cubin-only SASS detour” validation walkthrough
and a roadmap towards SM/warp/thread observability, see
`design/gpu_sass_observability.md:1`.

## hetGPU（tmatmul 分支）当前实现进度与成熟度

结合 `tmatmul` 分支最近 5 个提交（`28301d2`/`5805646`/`981a124`/`56aff72`/`c7b9106`）的实际代码状态，可以把 hetGPU 的进展分成三类：

### 1) “接口/数据结构/集成点”已经搭好（相对完善）

- 新增 `ptx/src/sass/*` 模块骨架：`cubin_parser` / `disassembler` / `dwarf_parser` / `instruction` / `llvm_inline`
- `SassPtxMapper`：提供 SASS address ↔ PTX(file,line) 的双向表结构、JSON 导入导出、nearest 查询等（在 `ptx/src/debug.rs`）
- “加载入口”设计：`CubinDebugInfo::parse_cubin_native()`、`HetGpuDebugInterface::load_from_cubin_native()` 这类 API 已经把“未来应该怎么接”的路铺好
- LLVM 侧的“注入策略枚举”也完整：InlineAssembly / PtxReconstruction / MetadataOnly / Hybrid（在 `ptx/src/sass/llvm_inline.rs`）

### 2) “能跑但偏工具依赖”的路径更成熟：cuobjdump 输出解析

`SassPtxMapper::parse_cuobjdump_output()` 这条路径（解析 `cuobjdump -sass -lineinfo` 文本）逻辑相对实用；
很多测试也围绕样例 `SAMPLE_CUOBJDUMP_OUTPUT` 在跑（见 `ptx/src/debug.rs` 末尾）。

也就是说：目前更像是“先把 mapping 能用起来”，而不是“native cubin+dwarf 一定正确”。

### 3) “native CUBIN/ELF + DWARF line + 真实 SASS 解码”仍是早期/未完成

从代码可以直接看出有多处 stub/placeholder：

- `cubin_parser.rs`
  - `extract_sm_version(...)` 目前直接 `Ok(61)` 默认值（没有真的从 `.nv.info` 或 ELF flags 解析）
  - `parse_debug_lines(...)` 目前直接返回空 map（注释写明 “For now, return empty map”），导致 `parse_cubin_native()` 这条路径即使能拿到 kernel code，也拿不到 “SASS address → PTX 行号” 的核心数据
- `dwarf_parser.rs`
  - 在 `5805646` 里确实补进了大量 DWARF 解析逻辑（`.debug_info/.debug_abbrev/.debug_str` 用 gimli），并实现了一个简化版 `.debug_line` state machine；但它目前还没有被 `cubin_parser.rs` 串起来使用，而且 `.debug_line` 解析里对 `opcode_base/line_range/line_base` 等参数有硬编码假设，跨 CUDA/SM/编译器组合时仍可能不稳
- `disassembler.rs`
  - opcode table 注释写了 “approximate - real encoding varies by SM version”，属于启发式骨架，不是严格的 SASS 反汇编器

结论：**hetGPU 的“结构设计 + 接口”很有参考价值，但其“native SASS/DWARF 实现”在该提交里还不算完善/可作为强依赖。**

## 0) Where bpftime is today (baseline)

bpftime’s NVIDIA path (`attach/nv_attach_impl`) does:

1. Intercept `__cudaRegisterFatBinary`/`__cudaRegisterFunction` etc (Frida-gum)
2. Extract PTX from fatbin (currently via `cuobjdump --extract-ptx all`)
3. Apply PTX passes (JSON stdin/stdout) and optionally add trampoline
4. Compile patched PTX to an ELF/CUBIN blob via `nvPTXCompiler` (no `nvcc` needed)
5. Load module with `cuModuleLoadDataEx`, then redirect launches to patched `CUfunction`

**Key observation:** we already have the **compiled ELF/CUBIN bytes in memory**
(`compiled_program` in `fatbin_record::compile_ptxs`). That’s the best insertion
point for native SASS + DWARF parsing without re-running external tools.

## 1) What we want (capabilities)

### A. SASS-side visibility

For each compiled module (ELF/CUBIN) and each kernel inside it, be able to:

- enumerate kernel functions, address ranges, and code bytes
- build a **line table** mapping machine address → (PTX file,line,column)
- optionally build an instruction table (address → {bytes, decoded mnemonic, classification})

### B. PTX/LLVM “origin tracking”

When bpftime injects/compiles code, be able to attribute SASS PCs to:

- original kernel PTX lines (if present)
- injected callsites (kprobe/kretprobe/memcapture, etc.)
- eBPF-generated PTX function bodies

This requires consistent “origin markers” that survive PTX → SASS compilation.

### C. Practical outputs

- runtime API to query mapping for a kernel / CUfunction
- `bpftimetool` (future) to dump mapping as JSON for debugging
- optional offline “cubin-inspect” tool for CI and local sanity checks

## 2) Proposed architecture

### 2.1 Data model (minimal, first milestone)

In `attach/nv_attach_impl`, add an internal mapping cache keyed by compiled ELF SHA:

```text
CompiledElfKey = sha256(elf_bytes)

SassModuleMapping {
  sm_arch: "sm_86",
  kernels: map<kernel_name, SassKernelMapping>
}

SassKernelMapping {
  name,
  text_range: [start,end),
  line_table: vec<LineEntry { pc, file, line, column }>,
  // optional later:
  insts: vec<InstEntry { pc, size, bytes[...], decoded? }>,
}
```

### 2.2 Where to build the mapping (runtime)

Hook point: `fatbin_record::compile_ptxs()` (or immediately after compilation).

- Input: `ptx_fixed` (PTX text) + `compiled_program` (ELF/CUBIN bytes)
- Output: `SassModuleMapping` stored in `nv_attach_impl` cache

This keeps the mapping aligned with the *actual* module we load via
`cuModuleLoadDataEx`.

### 2.3 Native CUBIN/ELF parsing strategy (bpftime-side)

Prefer LLVM’s object + DWARF readers (already a dependency when CUDA attach is on):

- `llvm::object::ObjectFile` / `llvm::object::ELFObjectFile`
- `llvm::DWARFContext` + debug line parsing

What we need from ELF:

- section(s) containing kernel code (`.text` / `.text.<kernel>` variants)
- symbol table entries for functions (kernel boundaries)
- DWARF `.debug_line` or CUDA lineinfo sections (address ↔ source location)

Fallback (Phase 0 / pragmatic):

- if native parsing is too slow/fragile initially, allow an env-gated path that
  shells out to `cuobjdump -sass -lineinfo` and parses its output, just for mapping
  generation. This is useful for rapid iteration but not the end goal.

## 3) Making PTX ↔ SASS mapping “work” in bpftime

The biggest gap is not “reading DWARF”, it’s **ensuring the compiler emits line
info that encodes bpftime’s intent**.

### 3.1 PTX compilation flags (nvPTXCompiler)

We should add a controlled way to compile with line info:

- default keep current `-O3` behavior
- when `BPFTIME_CUDA_LINEINFO=1`, add a lineinfo/debug flag supported by nvptxcompiler
  (exact spelling varies by toolkit; we should probe and gate with a runtime check)

bpftime 落地细节建议：

- 在 `attach/nv_attach_impl/nv_attach_fatbin_record.cpp` 的 `compile_ptxs()` 里集中拼装 nvptxcompiler compile options：
  - 保留现有：`--gpu-name=<sm>`、`--verbose`、`-O3`
  - 增加 env gate：`BPFTIME_NVPTX_LINEINFO=1` / `BPFTIME_NVPTX_DEBUG=1`
  - 做“探测式降级”：
    1) 先按用户开关加入候选选项（比如 lineinfo/debug 相关）
    2) 如果编译失败，从 `get_error_log()` 判断是 “unknown option”，则移除该 option 重新编译
    3) 最终把“这次编译实际启用的选项列表”打到 log + 也写入 mapping JSON 的 metadata

这样可以避免不同 CUDA 版本的 option 差异导致整条链路炸掉。

### 3.2 Origin marking scheme (PTX-level)

We need a stable mapping key that will show up in DWARF line tables:

- Use `.file` and `.loc` directives in PTX around injected regions
  - file: `"bpftime://inject/<pass_name>/<kernel>"`
  - line: `attach_id` or (attach_id << 16 | site_index)
  - column: optional sub-id (e.g., “before/after”)

For eBPF-generated PTX functions:

- Best: have the LLVM→PTX generator emit debug locations per eBPF instruction
  (line = eBPF PC, column = sub-op / helper id).
- Acceptable first step: emit coarse-grained `.loc` per basic block / per helper call,
  sufficient to identify “this SASS range is bpftime-ebpf”.

### 3.3 LLVM mapping (what “LLVM inlining” means for bpftime)

bpftime does not have the kernel’s LLVM IR, so we can’t “inline kernel SASS into IR”
the way hetGPU experiments do.

What we *can* do (and is still valuable):

- treat LLVM as the authoritative place to attach **debug metadata** for bpftime
  generated code (eBPF → PTX path)
- optionally build a small LLVM IR “shadow module” for mapping/profiling:
  - functions represent kernels and injected probes
  - basic blocks correspond to SASS basic blocks (optional later)
  - attach metadata: `bpftime.sass.pc`, `bpftime.attach_id`, `bpftime.ebpf_pc`

This “LLVM mapping” is primarily for tooling and analysis, not for codegen.

## 3.4（bpftime 特有）为什么我们比 hetGPU 更容易把 native mapping 做“真”

hetGPU 当前 Rust 侧的 DWARF/SM 解析是自实现/骨架状态；bpftime 在 CUDA attach 场景里已经强依赖 LLVM，
因此更建议直接用 LLVM 的成熟实现来做“ELF + DWARF line table”：

- `llvm::object::ObjectFile` / `llvm::object::ELFObjectFileBase`
- `llvm::DWARFContext` + `DWARFDebugLine`（读取 `.debug_line` / line table）

对应落地方式：

- 新增一个小模块（建议放在 `attach/nv_attach_impl/` 下）：
  - `attach/nv_attach_impl/sass_map/sass_map.hpp`
  - `attach/nv_attach_impl/sass_map/sass_map.cpp`
  - 只做两件事：
    1) 输入：`std::span<const uint8_t>`（compiled ELF/CUBIN bytes）
    2) 输出：`SassModuleMapping`（kernel ranges + line table entries + metadata）

这比在 bpftime 里手搓 DWARF state machine 风险低很多，也更接近“可维护的工程实现”。

## 4) Verification strategy (how we know it’s correct)

### 4.1 Unit-level (no GPU execution required, but needs CUDA toolchain)

Add tests under `attach/nv_attach_impl/test` that:

1. Compile a tiny PTX snippet with known `.file/.loc` into ELF via nvPTXCompiler
2. Parse the resulting ELF and assert that:
   - function symbol(s) exist
   - line table contains expected (file,line) entries

This validates “PTX markers survive compilation” + “parser reads line table”.

### 4.2 Integration-level (GPU runner, existing workflow)

On the self-hosted GPU CI (`.github/workflows/test-gpu-examples.yml`):

- run one representative example (e.g., `example/gpu/kernel_trace`)
- enable mapping dump via env vars, then assert artifacts exist:
  - dumped compiled CUBIN/ELF
  - dumped JSON mapping (if enabled)
- optionally compare a few addresses against `cuobjdump -sass -lineinfo` output

Minimal CI hook (suggestion):

- set env vars for the test job (or per-matrix entry):
  - `BPFTIME_NV_ATTACH_DUMP_PTX_DIR=/tmp/bpftime-ptx`
  - `BPFTIME_NV_ATTACH_DUMP_CUBIN_DIR=/tmp/bpftime-cubin`
  - `BPFTIME_NV_ATTACH_DUMP_SASSMAP_DIR=/tmp/bpftime-sassmap`
- add an `actions/upload-artifact` step at the end of the job to upload those
  directories (if they exist) for post-mortem debugging.

### 4.3 Runtime sanity checks (human-friendly)

Add env-controlled dumps (debug-only):

- `BPFTIME_NV_ATTACH_DUMP_PTX_DIR=/tmp/bpftime-ptx`
- `BPFTIME_NV_ATTACH_DUMP_CUBIN_DIR=/tmp/bpftime-cubin`
- `BPFTIME_NV_ATTACH_DUMP_SASSMAP_DIR=/tmp/bpftime-sassmap`

So a developer can:

1. run an example with agent preloaded
2. inspect dumped `.ptx`/`.cubin`
3. use `cuobjdump -sass -lineinfo` on the dumped cubin as a reference
4. compare with bpftime’s generated mapping JSON

## 7) Current prototype (bpftime dev/sass)

There is now a minimal native “ELF(CUBIN) + DWARF `.debug_line` → JSON mapping”
prototype in bpftime:

- Parser: `attach/nv_attach_impl/sass_map/sass_map.cpp`
- Hook point: `attach/nv_attach_impl/nv_attach_fatbin_record.cpp` (right after nvPTXCompiler produces the ELF/CUBIN bytes)

Enable it via env:

- `BPFTIME_NVPTX_LINEINFO=1` and/or `BPFTIME_NVPTX_DEBUG=1`:
  - tries additional nvptxcompiler flags (varies across CUDA versions; bpftime probes by dropping options until compilation succeeds)
- `BPFTIME_NV_ATTACH_DUMP_CUBIN_DIR=/tmp/bpftime-cubin`:
  - dumps compiled ELF/CUBIN bytes as `<elf_sha256>.cubin`
- `BPFTIME_NV_ATTACH_DUMP_SASSMAP_DIR=/tmp/bpftime-sassmap`:
  - dumps mapping JSON as `<elf_sha256>.json` containing:
    - `sm_arch`, `ptx_name`, `ptx_sha256`, `elf_sha256`, `compile_key`
    - `has_debug_line`, `entry_count`, and raw `entries` (address/file/line/column)

Notes:

- If `.debug_line` is not present in the compiled ELF/CUBIN, `has_debug_line=false` and `entries` is omitted.
- For “deep trace” you still need either:
  - reliable lineinfo emission (compiler flags), and/or
  - full SASS disassembly/patching (separate milestone).

Helper tool:

- `tools/bpftimetool` now has `dump-sassmap <cubin_or_elf_path>` to parse a dumped `.cubin` (or any ELF) offline and print the `.debug_line` mapping as JSON.

## 5) Milestones (recommended order)

1. **Dump-only plumbing**: dump patched PTX + compiled ELF (env-gated)
2. **Lineinfo on demand**: add compile flags toggle + smoke tests
3. **Native ELF + DWARF line parser**: produce `SassModuleMapping` JSON
4. **Origin markers**: `.file/.loc` around injected callsites and (coarse) eBPF code
5. **Tooling**: `bpftimetool dump-sass-map <kernel>` for runtime inspection
6. **Optional**: instruction decoding/classification; basic blocks; richer LLVM metadata

## 6) Risks / unknowns

- nvPTXCompiler accepted flags for lineinfo/debug differ across CUDA versions.
- Not all upstream PTX contains meaningful `.loc` info; bpftime must add its own.
- CUBIN layouts vary across SM generations; section naming and symbol types can differ.
- Full native SASS disassembly is hard; start from “address + bytes + line table”.
