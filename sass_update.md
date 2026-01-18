# bpftime SASS/IR 更新日志（hetGPU 风格演进）

目标：把 bpftime 的 GPU 观测从“纯 cubin/SASS 原地 patch”逐步补齐到类似 hetGPU 的“可重编译/IR 化”能力，用来从根上解决：
- regcount=255（flashattention）入口敏感：entry/prologue 难以安全插长 stub
- 没有 dead regs：无法稳定借 scratch
- 需要稳定 spill/save/restore：纯 SASS patch 很难做到可证明不破坏
- 需要更长、更复杂的 stub：code cave/布局限制明显

非目标（本阶段）：
- 直接把 cubin-only 的 flashattention 完全 lift 到 LLVM 并重生成（工程量太大，需要后续分阶段）
- 侵入式修改 vLLM（不改 vLLM）

## 路线拆解

### Route A（已做/默认稳定）：SASS identify/closure + 精准 detour
- 单次运行内 vLLM flash 已能 “稳定命中 + 至少 1 条 record”（见 `vllm_issues.md` 最新追加）。

### Route B（本文件跟踪）：JIT-link PTX/IR 插桩（hetGPU 风格最小子集）
核心思路：当 fatbin 内存在 PTX/NVVM（bitcode）时，不在 SASS 里“硬塞指令”，而是：
1) 从 fatbin 抽取 PTX/NVVM
2) 在 PTX/IR 层注入观测逻辑（不改变 kernel signature）
3) `cuLink` / `nvJitLink` 生成 sm_120 cubin
4) （可选）再走 SASS detour 做更底层观测

这样 “寄存器/spill/布局” 由编译器/链接器解决，天然克服 “无 dead regs / 需要长 stub”。

限制：
- 对 **cubin-only**（只带 SASS、无 PTX/bitcode）的 flashattention 仍不适用；这类要么继续走纯 patch + liveness/save/restore，要么引入 SASS→IR lift（后续阶段）。

## 2026-01-12 计划（开始）

Step 0（最小可验证）：在 JIT-link PTX 路径里做一个 “PTX marker” 插桩
- 注入一个 `.global .u32 __bpftime_ptx_marker`
- 在目标 `.entry` 内插入一次 `st.global.u32 [__bpftime_ptx_marker], 1`（不改参数，不读控制字）
- 写一个 host smoke test：构造 PTX-only fatbin → 走 `cuLibraryLoadData` → launch kernel → `cuModuleGetGlobal` 读回 marker

验收：
- 在 bpftime 下跑 smoke test，marker==1
- 在不开 bpftime 的情况下 smoke test 不要求 marker（符号不存在）

### 2026-01-12 进度（Step 0 完成）

- bpftime 侧：
  - 在 `maybe_patch_cuda_image_sass_detour()` 的 JIT-link PTX 提取路径中，新增 `BPFTIME_CUDA_SASS_DETOUR_JITLINK_PTX_MARKER=1`：
    - 对匹配 `BPFTIME_CUDA_SASS_DETOUR_FILTER` 的 PTX 输入，注入 `__bpftime_ptx_marker` 全局符号
    - 并在对应 `.entry` 的首条指令前插入一次 `st.global.u32` 写 1（PTX-level）
- 测试侧：
  - 新增 `benchmark/gpu/host/ptx_jitlink_marker_test.py`：生成 PTX-only fatbin，并通过 `cuLibraryLoadData`（FATBINC wrapper）触发 bpftime 的 JIT-link PTX 路径

回归命令（示例）：

```bash
BPFTIME_CUDA_SASS_DETOUR=1 \
BPFTIME_CUDA_SASS_DETOUR_FILTER=bpftime_ptx_marker_kernel \
BPFTIME_CUDA_SASS_DETOUR_JITLINK_PTX=1 \
BPFTIME_CUDA_SASS_DETOUR_JITLINK_PTX_MARKER=1 \
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
python3 benchmark/gpu/host/ptx_jitlink_marker_test.py
```

期望输出：
- `__bpftime_ptx_marker=1`

后续 Step 1（向“真实观测”靠近）：
- 用类似方法注入 “thread-map(device) 的紧凑写出” 或 “pc-marker”
- 把写出目标变成 bpftime 的 device buffer（需要 module 内可定位/可初始化的 global symbol 或其它绑定机制）

## 2026-01-12 进度（Step 1：PTX-level thread-map bring-up 完成）

本步骤目标：把 PTX 注入从“单个 marker”扩展到真正的 device-side 写出（thread-map 的最小形态），用于证明：
- 不需要找 dead regs
- 不需要 code cave（stub 可以很长）
- spill/寄存器分配由 toolchain 解决

实现：
- 新增 env：`BPFTIME_CUDA_SASS_DETOUR_JITLINK_PTX_THREADMAP=1`
- 在 JIT-link PTX 路径里注入：
  - module globals（由 bpftime 在 launch 时绑定）：
    - `.visible .global .align 8 .u64 __bpftime_ptx_out_ptr;`
    - `.visible .global .align 4 .u32 __bpftime_ptx_out_cap;`
    - `.visible .global .align 4 .u32 __bpftime_ptx_out_gate;`
  - 在匹配的 `.entry` 里插入 thread-map 写出逻辑（写到 sampler buffer）：
    - 读取 `__bpftime_ptx_out_ptr/cap/gate`
    - 按 gate 做 lane0-only / warp0-only / CTA0-only clamp
    - `idx=tid.x`，写 `smid` 到 `out_ptr[idx]`（u32 slot）

验证：
- 新增 `benchmark/gpu/host/ptx_jitlink_threadmap_test.py`
- 回归命令：

```bash
BPFTIME_CUDA_SASS_DETOUR=1 \
BPFTIME_CUDA_SASS_DETOUR_FILTER=bpftime_ptx_threadmap_kernel \
BPFTIME_CUDA_SASS_DETOUR_JITLINK_PTX=1 \
BPFTIME_CUDA_SASS_DETOUR_JITLINK_PTX_THREADMAP=1 \
BPFTIME_CUDA_SASS_SAMPLE=1 \
BPFTIME_CUDA_SASS_SAMPLE_MODE=thread \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE=1 \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_STRIDE4=1 \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_LANE0_ONLY=1 \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_CTA_CLAMP=1 \
BPFTIME_CUDA_SASS_SAMPLE_MAX_RECORDS=1 \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_PATH=/tmp/bpftime-ptx-threadmap.jsonl \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_ON_SYNC=1 \
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
python3 benchmark/gpu/host/ptx_jitlink_threadmap_test.py
```

期望输出：
- `ok: lane0-only bpftime_sass_thread records for [0, 32, 64, 96, 128, 160, 192, 224]`

## 2026-01-12 进度（Step 2：接入 bpftime JSONL dump/限流/目标选择 完成）

本步骤目标：让 “PTX-level 写出” 直接走 bpftime 现有的 sampler buffer + JSONL dump（而不是写到 module 自己的固定数组），从而：
- 和现有的 `BPFTIME_CUDA_SASS_SAMPLE_*` / `BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_*` 限流/gate 复用
- 和 vLLM bring-up 的整体观测链路对齐（dump-on-sync / 之后可扩展到 dump-on-exit 等）

实现（核心点）：
- PTX 注入侧（`attach/nv_attach_impl/nv_attach_impl_frida_setup.cpp`）：
  - 把 thread-map 的写出目标从 `__bpftime_ptx_out[1024]` 改为 `__bpftime_ptx_out_ptr/cap/gate`
  - 写出内容从 `tid.x` 改为 `smid`，以匹配 bpftime 现有 `bpftime_sass_thread` 的 slot 语义（slot->tid_x，value->smid_lo8）
- host 侧绑定（`cuLaunchKernel` hook）：
  - `resolve_cumodule_for_cufunction()` 获取 module（兼容 cuLibrary path）
  - `cuModuleGetGlobal_v2` 找到 `__bpftime_ptx_out_*` 三个符号
  - 用 `cuMemcpyHtoDAsync_v2` 把 `out_ptr` 绑定到 `sass_sampling.device_buffer + buffer_data_offset`
  - 按 env 把 gate/cap 写入，且在 bind 时对首个 record 做一次 `0xffffffff` 清理

对 vLLM 的意义（这条 Route B 的适用范围）：
- 只要目标 kernel 的 code object 路径里确实有 PTX/NVVM（bitcode）并走 JIT-link（nvJitLink/cuLink），就能用这条 IR/重编译路线避开 “reg255 + entry/prologue 入口敏感” 的根问题。
- 对 cubin-only（很多 flashattention sm_120）仍需要另一条线：纯 SASS patch + 更系统的 liveness/save/restore/spill + 更安全 patch 点（后续阶段）。

## Step 3（计划）：覆盖真实 vLLM 的“直接 JIT-link”入口（cuLink/nvJitLink）

现状缺口：
- Step 2 的 PTX 注入目前主要发生在 “fatbin wrapper 解析→bpftime 自己做 cuLink(JIT-link) fallback” 这条路径；
- 但真实 vLLM/PyTorch/Triton/NVRTC 很常见的链路是：直接调用 `cuLinkAddData(PTX/NVVM)` / `nvJitLinkAddData` → `*_Complete` → `cuModuleLoadDataEx`；
  这种情况下，如果我们不 hook `*LinkAddData`，就无法在 PTX/IR 层注入（仍会退回到 cubin/SASS 的困难模式）。

计划落地：
1) hook `cuLinkAddData`（以及必要的 `cuLinkAddData_v2` / `cuLinkDestroy`）：
   - 对 `CU_JIT_INPUT_PTX` 的输入做 “filter 命中→注入 marker/threadmap/pc-marker（后续）”
   - 维护 per-`CUlinkState` 的 owned buffer，确保在 `cuLinkComplete` 前数据不被释放
2) hook `nvJitLinkAddData`（如果本机/驱动栈实际使用 nvJitLink）：
   - 同上，对 PTX/NVVM 输入注入并保活
3) vLLM 验证：
   - trace→挑选一个确定走 JIT-link 的 kernel substring（通常来自 Triton/NVRTC）
   - 用现有 sampler-bound thread-map（lane0-only + CTA clamp + max_records=1）验证 JSONL 必出数据

### 2026-01-12 进度（Step 3：cuLinkAddData/c uLinkAddData_v2 覆盖完成）

实现：
- hook 了真实 driver 链路的 `cuLinkAddData` + `cuLinkAddData_v2` + `cuLinkDestroy`：
  - 当 `CU_JIT_INPUT_PTX` 且命中 `BPFTIME_CUDA_SASS_DETOUR_FILTER`（或 `BPFTIME_CUDA_SASS_SAMPLE_FILTER`）时：
    - 注入与 Step 2 一致的 PTX snippet（sampler-bound thread-map / marker）
    - 通过 `nv_attach_impl::culink_track_owned_input()` 把注入后的 PTX bytes 绑定到 `CUlinkState` 生命周期，直到 `cuLinkDestroy` 清理
- 同时把 sampler globals 的 bind 从“依赖 kernel_name 命中”改为“按 module 一次性探测并缓存”，避免 strip/未知名场景下 out_ptr 永远不下发。

验证：
- 新增 `benchmark/gpu/host/ptx_culink_adddata_threadmap_test.py`
- 回归命令：

```bash
BPFTIME_CUDA_SASS_DETOUR=1 \
BPFTIME_CUDA_SASS_DETOUR_FILTER=bpftime_culink_adddata_kernel \
BPFTIME_CUDA_SASS_DETOUR_JITLINK_PTX=1 \
BPFTIME_CUDA_SASS_DETOUR_JITLINK_PTX_THREADMAP=1 \
BPFTIME_CUDA_SASS_SAMPLE=1 \
BPFTIME_CUDA_SASS_SAMPLE_MODE=thread \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE=1 \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_STRIDE4=1 \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_LANE0_ONLY=1 \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_CTA_CLAMP=1 \
BPFTIME_CUDA_SASS_SAMPLE_MAX_RECORDS=1 \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_ON_SYNC=1 \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_PATH=/tmp/bpftime-ptx-culink-adddata-threadmap.jsonl \
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
python3 benchmark/gpu/host/ptx_culink_adddata_threadmap_test.py
```

期望：
- `/tmp/bpftime-ptx-culink-adddata-threadmap.jsonl` 中 `bpftime_sass_thread` 至少包含 tid_x `[0,32,64,...,224]` 的 lane0-only 记录

后续：
- nvJitLink：若真实 vLLM 的 Triton/NVRTC 链路主要用 `nvJitLinkAddData`，还需要同样 hook 那组 API（本机先以 cuLink 覆盖为主）。

## Step 4（计划）：cubin-only + regcount=255（FlashAttention）稳定高密度 thread-map(device)

目标（对齐你要的“稳定高密度 thread-map(device)”）：
- 在 **真实 vLLM** 的 flashattention（cubin-only、regcount=255）上：
  - 能稳定插桩（不崩、不触发 cublas internal error）
  - 线程级写出密度可逐步放开，最终至少做到：
    - **CTA0 全量 per-thread 写出**（tid.x 0..blockDim.x-1 都写），并且每次 run 都能 dump 出大量 `bpftime_sass_thread` 记录

为什么需要“系统化 patch/layout + CFG/liveness + spill”：
- reg255 kernel 入口/前段极敏感，长 stub/读控制字/复杂谓词 很容易引发随机崩溃。
- 现在我们默认走 “prefer-exit + multi-exit” 来把风险压到最低；要把 patch 点推到更一般的位置、并稳定扩大写出密度，就必须能：
  - 选更安全的 patch 点（不仅 tail EXIT）
  - 找 dead regs / 可用谓词/UR（避免借 live regs）
  - 找不到时退化（L0→L3），必要时做 per-thread 独占 slot 的 spill/save/restore

实施路线（从最稳到最激进，逐步放开，每一步都要能回归）：

Step 4.1（闭环脚本 + 最小目标选择）：
- 用 identify/closure（Route A）拿到本次 vLLM flash 的 `target_func_id`
- 第二次 run 用 `BPFTIME_CUDA_SASS_DETOUR_FILTER_FUNC_IDS=<target_func_id>` 做精准 patch（避免 patch-all 扩大影响面）
- 把采样模式设为 `thread_map(device)`，先从 **tid0-only**（lane0+warp0）开始验证 “不崩 + 必有 records”

Step 4.2（EXIT/pre-exit patch 点稳定化）：
- 继续用 `prefer-exit + multi-exit`，避免 “tail EXIT 不执行导致 record=0”
- 若 store 仍引发不稳定：切到 `BPFTIME_CUDA_SASS_DETOUR_REG255_PRE_EXIT=1`（在 EXIT 前一条 NOP/稳定点 detour，并先 replay 再跑 stub）
- 选用 `BPFTIME_CUDA_SASS_DETOUR_EXIT_STUB_USE_DEAD_REGS=1`，用 “dead regs after patch site” 当 scratch

Step 4.3（高密度放开：lane0-only → warp0-only → full threads）：
- 始终保持：
  - `CTA_CLAMP=1` + `MAX_RECORDS=1`（先只让 CTA0 写，避免全网格写爆 buffer）
  - `STRIDE4=1`（更稳的 32-bit 对齐写）
  - dump-on-sync（稳定复现）
- 逐步放开 gate：
  - tid0-only（lane0+warp0）
  - lane0-only（每 warp 一个线程写）
  - warp0-only（一个 warp 全量写）
  - full（CTA0 全量 per-thread 写）

Step 4.4（必要时引入 per-thread spill/save/restore（L3））：
- 若在放开密度时出现稳定性问题，则启用：
  - `BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_REG255_SPILL=1`
  - `BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_REG255_SPILL_PER_THREAD=1`
  - 并把保存/恢复从 “只 P0” 扩到 “少量 GPR + 多个 predicate”（需要扩展 stub 逻辑）

验收（最终）：
- 在 vLLM Qwen3-0.6B 的 in-process run 里，启用 `full threads` + `CTA0` 时：
  - 稳定不崩，且 `bpftime_sass_thread` 记录数 >= `blockDim.x`（至少数百条）
  - 连续跑 3 次都能过（避免偶发）

## 20260112 进度追加（reg255 flash：predicated EXIT 语义修正 + 目标 gating）

本轮改动/结论（和 Step 4 直接相关）：

1) **predicated EXIT 之前的实现缺了 `@!P? EXIT`（negation）语义，导致“允许 predicated EXIT”时可能破坏控制流**
- 现象：FlashAttention 的 sm_120 SASS 里真实存在 `@!P0 EXIT`（cuobjdump 可见 `0x894d`），但我们旧的解析只看 `(pred<<12)|0x094d`，会把 `@!P0 EXIT` 当成 `@P0 EXIT`。
- 修复：
  - `exit_kind_from_op16_sm120()` 现在同时解析 `pred` + `neg`（bit15 / `0x8000`）。
  - `encode_bra_sm120_pred()` 支持生成 `@!P? BRA`（同样用 bit15）。
  - thread-map prefer-exit 的 debug 日志会显示 `rel@P0` / `rel@!P0`，便于核对。

2) **把 `ALLOW_PREDICATED_EXIT` 分成两档，避免默认过激**
- `BPFTIME_CUDA_SASS_DETOUR_ALLOW_PREDICATED_EXIT=1`：只允许 `P0`（含 `@P0` 与 `@!P0`）。
- `BPFTIME_CUDA_SASS_DETOUR_ALLOW_PREDICATED_EXIT_ANY=1`：允许 `P0..P6`（高风险，真实 vLLM 上仍可能 segfault，保持 opt-in）。

3) **reg255 EXIT thread-map(device) stub 增加 control header gating（在 EXIT 点读控制字，避免 entry/prologue 的 _UNSAFE）**
- 在 `build_sm120_thread_map_stub_no_regcount_at_exit()`：若 `cfg.control_enabled==true`，会读取 control header：
  - `enable==1 && mode==Target(2) && target_func_id==this_func_id` 才执行写出；
  - 否则直接跳过写出（减少跨 kernel 互相覆盖、降低“最后一次 dump 看到的不是目标 kernel”概率）。
- bring-up 脚本 `benchmark/gpu/vllm_observability/vllm_flash_reg255_threadmap_bringup.py`：
  - thread-map stages 会设置 `BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE_FORCE_TARGET_FUNC_ID=<target_func_id>`，让 host 直接进入 Target 模式（dump meta 里能看到 `control.mode=2`）。

当前仍卡住的核心问题（离“高密度 per-thread 写出”最近的 blocker）：
- 即使 `control.mode=2` 且 `target_func_id` 已设置，thread-map(device) 的 JSONL 里仍常只有 `slot=0` 一条记录。
- 通过 `BPFTIME_CUDA_TRACE_PATH` 抓到的真实 vLLM trace 显示：`flash_fwd_splitkv_kernel` 的 `blockDim.x=128` 且被频繁 launch（理论上 `warp0-only` 至少应有 32 条记录）。
- 这意味着我们**还没有稳定命中“真实执行的那一个 flashattention SM120 code object/entrypoint”**（典型原因是 strip 场景下 `.text.<name>` 不可用，或 func_id→entrypoint 映射还不够强），因此高密度写出仍无法验证通过。

下一步（仍按 Step 4 路线，且与 hetGPU 对齐的关键）：
- 补齐“可命中覆盖面”的那块：让“真实执行的 flash_fwd_splitkv_kernel（可能是 strip 的 SM120 code object）”能被稳定定位并打上 thread-map(device) stub；
  - 需要更强的 name→entrypoint 映射（`.nv.info.<name>` / symtab / 以及 identify/tag 闭环得到 func_id 后再精确 patch）。

## 20260112 进度追加（image_id 一致性 + bring-up 脚本回归）

本轮目标：把“按 kernel 名字精准 detour”的闭环再收敛一步，减少跨模块 func_id 冲突与 host 侧重复写控制字导致的噪声。

进展：

1) **统一 image_id 计算口径（避免 cache key 与 patcher 的 image_id 不一致）**
- 问题：`nv_attach_impl_frida_setup.cpp` 里用 `fnv1a64(full_bytes)` 算 image_id，而 `sass_detour.cpp` 会先按 ELF 的“有效 extent”（section/headers）裁剪后再 hash；在 raw-ELF / fatbinc padding 场景下，两边 image_id 可能不一致，导致 `learned_func_ids_by_image_id` 命中率下降。
- 修复：`attach/nv_attach_impl/nv_attach_impl_frida_setup.cpp` 新增 `sass_elf_image_id32()`，按 ELF header/section table 计算 `file_end` 后再 hash，使其与 `sass_detour::DetourResult::image_id` 对齐。

2) **identify->Target 写控制字后同步更新 host-side dedupe 状态**
- 问题：identify dump 成功后 `sass_control_maybe_dump_identify()` 会直接把 header 切到 Target，但 host 的 `last_control_*` 仍是旧值，后续每个 matching launch 会再次走 `sass_control_set_target()` 做一次 HtoD + 打 log（log spam + 不必要开销）。
- 修复：`attach/nv_attach_impl/nv_attach_impl_frida_setup.cpp` 在 identify 成功写 Target header 后同步更新 `sass_sampling_state_t::last_control_*`，让后续 launch 能被 dedupe 掉。

3) **bring-up 脚本在“无 set_target log”情况下仍能自动提取 func_id**
- 由于上面 (2) 的收敛，identify run 里可能不再出现 `SASS control: set target func_id=...`；因此 `benchmark/gpu/vllm_observability/vllm_flash_reg255_threadmap_bringup.py` 增强了 log 解析：优先从 `identify dump ... slots=[...]` 提取 func_id。

验证（当前 checkpoint）：
- 默认（single-run closure + on-demand upgrade）仍可跑通：`tid0` stage 能稳定产出 `bpftime_sass_thread` 记录（目前 `MAX_RECORDS=1`，因此 record 数固定为 1，主要用于验证“命中 + 写出链路不崩”）。

注意：
- “legacy 两次运行”（identify 一次、thread-map 再起一次）场景下，**跨 run 用 image_id 做强约束可能降低命中率**（某些 JIT/link 路径会生成不同的 code object）。因此脚本把 `FILTER_IMAGE_ID` 变为可选（`--legacy-use-image-id`），默认仍以命中率优先（func_id-only）。
