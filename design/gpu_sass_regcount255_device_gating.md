# regcount=255（SM120）device-side control reads / Target gating：路线 B 方案

目标：让 `BPFTIME_CUDA_SASS_DETOUR_NO_REGCOUNT_CONTROL_UNSAFE=1` 这条“在 kernel 入口 device 侧 `LDG` 读取 control header 并做 Target gating”的路径，能在真实 vLLM（flashattention/cutlass）上稳定运行，不再触发 `CUBLAS_STATUS_INTERNAL_ERROR` 这类后效崩溃。

背景问题：当 kernel `regcount=255` 时，我们无法通过“提升 regcount 预留 scratch GPR”的方式保证 stub 的寄存器/谓词/UR 使用是可证明不破坏的。device-side 读取 control header 会引入更多指令、寄存器依赖、UR descriptor 依赖与谓词分支窗口；在 cubin-only vendor kernel 的入口，这些都非常敏感，任何假设不成立都会变成随机崩溃。

## 总体策略（分层收敛）

路线 B 的核心不是“把 `LDG` 编码出来”，而是把它放进一个 **可证明低风险** 的执行框架里：

1) **极早的 single-thread guard**：绝大多数线程在入口立刻跳过（不做 `LDG`/不写内存），把风险与扰动压到最小。
2) **只在 guard 通过后触碰全局内存**：device-side `LDG`（control header）与任何 `STG` 只允许单线程执行。
3) **对被触碰的状态做最小 save/restore**：至少覆盖 P0（以及必要时的 UR pair / 少量 GPR），并避免跨 warp/CTA 竞争。
4) **模板化 + kernel-local template**：控制字/descriptor 的 w1/w0 通过从目标 kernel 中提取模板（hetGPU 风格）来保证稳定性，而不是硬编码。

## 1) single-thread guard（必须优先做）

### 目标
让 stub 的“重逻辑”（读取 control、Target 判断、写出）只在单线程执行，例如：

- `(ctaid.x == 0 && tid.x == 0)`（最保守）
- 或者先 `(tid.x == 0)`（每 CTA 1 线程），但更容易写爆 buffer/影响面更大。

### 为什么必须
即使我们能做到不破坏寄存器，`regcount=255` kernel 的入口对额外指令窗口仍非常敏感。把 `LDG`/分支/地址计算的执行次数从 “每线程” 降到 “一次”，是稳定性的最大杠杆。

### 推荐实现思路（不依赖 GPR/P0）
优先用 **UR + uniform predicate（UP）** 来构造 guard，让普通线程几乎不触碰 GPR/P0：

- `S2UR` 读取 `SR_CTAID.X` / `SR_TID.X` 到 UR
- `UISETP.*` 产生 `UP0`（例如 `tid_x != 0 || ctaid_x != 0`）
- `BRA.U` 基于 `UP0`/`!UP0` 跳过重逻辑

实现要点：
- guard 必须放在 stub 最开头，且在 guard 通过前 **不做任何全局内存访问**；
- 需要补齐 SM120 的 `UISETP` / `BRA.U` 编码模板（从实际 sm_120 cubin 中提取 w0/w1，禁止拍脑袋硬写）。

## 2) save/restore（最小集合）

即使采用 UR/UP guard，guard 通过的那 1 个线程仍需要执行 `LDG`、分支与写出。此时必须保证：

- **恢复 P0**：避免破坏原 kernel 的 predicate 状态（当前已经在部分 stub 做了 `P2R/R2P` 保存恢复，路线 B 需要强制化）。
- **避免 clobber kernel 入口常用 UR**：如 UR4/UR5 常见用于 descriptor；如果要用 descriptor-form `LDG/STG`，建议：
  - 尽量沿用 kernel 内已有的 descriptor base（从 prologue 提取 `LDCU.64` 模板）；
  - 或者借用较高 UR pair（例如 UR62/UR63），但必须考虑 kernel 入口是否已经 live（regcount=255 常见）。

如果仍需要临时 GPR（地址对/临时值）：
- 只用极少量（例如一对 `.64` + 1～2 个 `.32`），并尽量让它们在 replay prefix 之前被 kernel prologue 覆盖；
- 或者在 guard 通过后使用更复杂的 per-thread spill（见下一节）。

## 3) per-thread spill（可选，但用于“更通用”的稳定性）

当 guard 仍不足以完全消除崩溃（例如入口对 P0/GPR 的任何改写都敏感），就需要一个“不会跨线程竞争”的 spill 通道。

可行方向：

- **local memory spill（per-thread）**：用 `STL/LDL`（或等价 local 指令）把需要保存的状态写到该线程的 local frame。
  - 前提：我们能从 cubin 的 `.nv.info*` 获取 local frame / stack size / local memory size，并选择不会越界的固定偏移。
  - 这是 hetGPU 系统化 patch/layout 常见的路线：先能稳定解析/建模，再插入更复杂 stub。

不推荐：
- 用单个全局 scratch（跨 warp/CTA 会竞争）。

## 4) device-side control reads（在 guard + save/restore 后实现）

在上述框架成立后，再实现 device-side `LDG` 读取 control header 并做 Target gating：

- 读 `enable/mode/target_func_id`（建议打包成 `LDG.E.64` 两次读取，减少指令数与依赖）
- 用 `ISETP/UISETP` 做比较
- Target hit 才写出（写出也建议 lane0/单线程，避免压力与冲突）

关键验收：
- 连续多次 vLLM cold run 不触发 cublas internal error
- `Target-hit marker + 至少 1 条 record` 稳定出现

## 5) 里程碑建议（实现顺序）

1) 补齐 `UISETP` + `BRA.U` 模板（从 sm_120 cubin 提取完整 w0/w1）
2) 在 regcount=255 的 no-regcount stub 入口加入 `ctaid==0 && tid==0` guard（先不做任何 `LDG`）
3) guard 通过后只写一个 marker（不读 control），验证稳定性
4) guard 通过后加 `LDG` 读取 enable/mode（只读不写），验证稳定性
5) 最后才打开 Target gating + 写出（逐步扩大功能）

本文件仅描述路线 B 的工程化方案；路线 A（host 闭环 + func_id 精准 patch）用于短期默认稳定能力，并不依赖路线 B。  
