# SM120 SASS “Identify/Tag 闭环”（单次运行内 name→func_id）

目标：解决 vLLM/flashattention 在 **cold run** 场景里常见的 “SM120 raw-ELF 先加载、fatbinc 后出现” 导致 **同一次运行来不及用跨 SM 传播得到的 func_id 去 patch** 的问题。

本方案在 **单次运行内** 完成闭环：

- 输入：用户只提供 `BPFTIME_CUDA_SASS_DETOUR_FILTER=<kernel_substr>`（例如 `flash_fwd_splitkv_kernel`）
- 输出：自动得到对应的 `func_id`（`sh_info`）并在同一进程后续执行中生效（无需重启）

## 背景：为什么跨 SM 传播在 vLLM 里“来不及”

在当前 vLLM 0.13.0 + Qwen/Qwen3-0.6B 的 in-process 场景里，实测顺序几乎固定：

1. 先出现 `magic=0x464c457f` 的 **raw ELF（常见是 SM120）**
2. 后出现包含 `.nv.info.<kernel>` 名称的 **fatbinc**
3. 因此跨 SM 传播（在 fatbin 内从其它 SM 提取 func_id）只能在 “raw ELF 已经 load 完” 之后才拿到 func_id

跨-run cache 可以让 **第二次启动**命中，但我们要做的是 **第一次启动就闭环**。

## 设计原则

- 默认不扰动：绝大多数时间 stub 只做一次 `LDG` 读取 enable，并快速返回（近似 no-op）
- 显式武装窗口：仅当 host 侧“识别到目标 kernel launch”时，短时间把 enable 打开
- 无需 kernel 名字存在于 SM120 cubin：闭环基于 `func_id(sh_info)`，不依赖 `.text.<name>`
- 仅针对 SM120：当前实现阶段只覆盖 `sm_120`

## 控制面：control buffer layout（device memory）

复用现有 SASS sampler 的 device buffer，在 buffer 起始处预留一个固定大小的 header，用于 host↔device 的控制与 identify 输出。

```
base = sample_buffer_device_ptr  (stub 内用 UR pair 物化)

offset 0x00: magic(u32)
offset 0x04: version(u32)
offset 0x08: enable(u32)            // 0=off, 1=on
offset 0x0c: mode(u32)              // 1=IDENTIFY, 2=TARGET
offset 0x10: epoch(u32)             // host 增加，用于分次
offset 0x14: target_func_id(u32)    // mode=TARGET 时使用
offset 0x18: reserved
offset 0x40: data region (原 sampler 输出区，从这里开始)
```

identify 输出（最小可用）：

- `slots[smid_lo8] = func_id`（u32），写入地址：`base + 0x20 + smid_lo8*4`
- host 侧在同步点（`cuCtxSynchronize/cuStreamSynchronize`）拉回 header+slots，得到唯一 `func_id` 集合

## stub：device 侧执行逻辑（SM120）

所有被 detour 的 `.text.*` 入口都执行同一个 stub。

下面先描述“目标形态（read-control）”，再说明目前实现到哪一步。

```
// 常态：enable=0，快速退出（近似 no-op）
enable = LDG.U32 [base + 0x08]
if (enable == 0) goto exit

mode = LDG.U32 [base + 0x0c]
if (mode == IDENTIFY) {
  smid = S2R VIRTUALSMID
  slots[smid_lo8] = func_id(sh_info)   // 写立即数
  goto exit
}

if (mode == TARGET) {
  if (func_id != target_func_id) goto exit
  // 这里执行“真正采样/写出”（后续扩展：thread/warp map）
}

exit:
  // restore UR/predicates, branch back
```

### 当前实现（2026-01-07）：write-only identify + host 侧强制对齐

目前在 detour trampoline 内执行 `LDG` 读取 control header 仍会触发
`CUDA_ERROR_ILLEGAL_ADDRESS(700)`（见进度日志），所以现阶段先采用“只写不读”的
identify stub：

- stub 不做任何 global load（不读 enable/mode/target）
- stub 直接把 `func_id(sh_info)` 写入 `slots[smid&7]`（默认只让 lane0 写）
- host 在 `cuLaunchKernel` 处 `arm(清 slots)`，并可选在该次 launch 后做一次强制同步，
  立刻 dump slots 学到 `func_id` 并写入 cache（避免依赖应用自己调用 sync）

需要的最小 opcode 模板：

- `LDG.E.U32`（读取 enable/mode/target）
- `ISETP.EQ.U32`（判断 enable==0、mode==X、func_id==target）
- `MOV imm32`（把 func_id 写入寄存器，用于 store）

## host：arm / dump / learn（闭环）

### 触发（arm）

当 `cuLaunchKernel` 的 `kernel_name` 命中 `BPFTIME_CUDA_SASS_DETOUR_FILTER` 且：

- 当前进程内尚无 mapping（memory/cache）可用

则：

1. 通过 `cuMemcpyHtoD` 更新 control header：
   - `enable=1, mode=IDENTIFY, epoch++`
   - 清空 identify slots（写 0）
2. 记录 “pending identify = true”，等待后续同步点读取结果

### 读取（dump）

在 `cuCtxSynchronize`/`cuStreamSynchronize` hook：

- 若 pending identify：
  - `cuMemcpyDtoH` 拉回 header+slots
  - 提取非 0 的 `func_id` 集合（可去重）
  - 写入 mapping：
    - 内存：`filter/kernel_name_substr -> func_id[]`
    - 持久化：复用现有 `BPFTIME_CUDA_SASS_DETOUR_FUNC_ID_CACHE_PATH`
  - 切换到 `mode=TARGET`（可选）让后续采样更精确

### 可选：单次 launch 内完成 “arm→sync→dump”（已实现）

在 `cuLaunchKernel` hook 中，如果本次命中 filter 且需要 identify：

- 先 `arm identify`（header + clear slots）
- 若设置 `BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_SYNC_AFTER_LAUNCH=1`：
  - launch 返回后立刻对当前 stream/ctx 做一次同步
  - 立刻 dump slots 并学习 func_id（减少 “raw-ELF 先到” 的时序敏感性）

## 实施计划（逐步验收 + 进度记录）

### Step 1（本轮）

- [x] 在 sampler buffer 里引入 header（`data_offset=0x40`）与 slots
- [x] `cuLaunchKernel` arm + `cu*Synchronize` dump/learn（并支持 `IDENTIFY_SYNC_AFTER_LAUNCH`）
- [x] write-only identify：device 侧写出 `func_id`（lane0 gate + 稳定写入）
- [ ] read-control：stub 内 `LDG` 读 enable/mode/target（当前仍会触发 700）
- [ ] 回归：vLLM cold run（清 cache）一次启动内完成 identify 并在同进程后续请求生效（依赖 read-control 或 patch-all 策略）

### Step 2（后续）

- [ ] raw-ELF 早到时启用 “identify-ready patch-all”（仅 SM120、enable=0 常态）
- [ ] TARGET 模式下做 thread/warp map 的低扰动采样（lane0/warp0 gate）

## 进度日志

（每完成一小步都在这里追加一条，包含：变更文件、验证命令、结果摘要）

- 2026-01-07：补齐 SM120 的 `S2R/ISETP/LDG` 关键模板（以 `nvcc -arch=sm_120` + `cuobjdump --dump-sass` probe 为准），并修复 `LANE0_ONLY` gate 的谓词/写出（见 `attach/nv_attach_impl/sass_detour/sass_detour.cpp`）。
- 2026-01-07：确认容器 GPU driver 可用（`python3 -c "import os; os.open('/dev/nvidiactl', os.O_RDWR)"` + `python3 -c "import ctypes; print(ctypes.CDLL('libcuda.so.1').cuInit(0))"` 返回 `nvidiactl rw ok` + `0`）。
- 2026-01-07：当前 `LDG.E desc[UR4][addr]` 形式的 “read-control” 在 detour trampoline 内仍会触发 `cuCtxSynchronize=700 (CUDA_ERROR_ILLEGAL_ADDRESS)`（最小复现：`benchmark/gpu/host/cubin_call_replay_test.py` + `BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE=1` + `BPFTIME_CUDA_SASS_DETOUR_CONTROL_READ_GLOBAL=1`）。因此现阶段默认走 “write-only identify（不做任何 global load）+ 跨 run func_id cache” 的闭环（`BPFTIME_CUDA_SASS_DETOUR_FUNC_ID_CACHE_PATH`），单次运行闭环仍待继续攻关。
- 2026-01-07：write-only identify 在 `call_entry` 最小例可稳定写出：`BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_SYNC_AFTER_LAUNCH=1` 下可观测到 `slots=[7,0,0,0,0,0,0,0]`（func_id=7，lane0-only），并写入 `/tmp/bpftime-sass-funcid-cache.json` 供下次运行复用。
- 2026-01-07：修复 `thread_map(device)` 写出稳定性（slot 映射 + LANE0_ONLY gate）：
  - 问题：`idx` materialize 后紧跟 `IMAD.WIDE` 会出现“stale idx（常见为 0）→ 大量写到 slot0 / 甚至触发 717”的非确定性，导致看起来像 “LANE0_ONLY 不生效/slot 映射错乱”。
  - 修复：
    - `idx = (ctaid<<10) | (tid&0x3ff)`（`SHF.L + LOP3.OR`）
    - `idx` 与 `IMAD.WIDE` 之间默认插入 `BPFTIME_CUDA_SASS_DETOUR_IDX_WAIT_NOPS=8` 个 NOP（可调）
    - gate 的 lane0 判断改为 `lane=(tid&31)`（避免依赖 `SR_LANEID` 在入口处的异常行为）
    - 新增调试输出：`BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_DEBUG_STORE_IDX=1`/`...DEBUG_STORE_TID=1`，host dump 增加 `raw_u32`
  - 验证（期望：dump 32 条且 `slot==raw_u32` 且 slot 集合为 `0,32,...,992`）：
    - `BPFTIME_CUDA_SASS_SAMPLE=1 BPFTIME_CUDA_SASS_SAMPLE_MODE=thread BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE=1 BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_LANE0_ONLY=1 BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_STRIDE4=1 BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_DEBUG_STORE_TID=1 BPFTIME_TEST_BLOCK=1024 LD_PRELOAD=build/runtime/agent/libbpftime-agent.so python3 benchmark/gpu/host/cubin_call_replay_test.py`
