可 patch 覆盖面不足（最致命）：很多关键 kernel（flash/cublasLt 等）对应的 SM120 code object 可能是 stripped / JIT/link 产物 / 不带可用 .text.* 命名与符号信息，导致“按名字 filter”打不中，甚至拿不到可 patch 的那份最终 cubin。
稳定性风险（会直接崩/illegal access）：device 写出/更长 stub 会引入寄存器/谓词破坏、地址计算错误、并发写冲突等；在 vLLM 的大 grid/block、sharedmem、并发 stream 下更容易放大。
性能与扰动（会把观测变成被观测对象）：patch-all 或高密度写出会显著拖慢 kernel、增加 L2/DRAM 压力、引入延迟抖动；需要强 gate/采样率/只对目标 kernel 生效。
归因不准（数据“看似有”，但对不上）：你看到的记录来自哪个 kernel/哪个入口、是否被多个 CTA/stream 覆盖、是否跨进程混写；如果没有闭环映射，容易出现“记录有了但不知道对应哪一个 kernel/阶段”。
cudagraph / 多入口路径导致漏采：vLLM 会 capture/replay graph，部分执行路径不经过你期望的 hook 点或时序不同，导致采样触发不稳定或对不上某次请求。
多进程与输出/状态隔离：engine 多进程会重复初始化/写文件/重复 patch；虽然 %p 解决了 dump 覆盖，但 tag 映射表/控制开关也要按进程/模块隔离。


你深入分析 规划一下 可 patch 覆盖面不足（最致命） 和 稳定性风险这俩怎么解决掉。给个方案出来。


下面这俩问题本质上是两条线：覆盖面（能不能 patch 到目标 code object/entrypoint）和稳定性（patch 了会不会崩/扰动）。我给一个按优先级可落地的方案。

1) “可 patch 覆盖面不足（最致命）”怎么解决
1.1 先把“能拿到最终 code object”补齐（入口覆盖）
目前我们主要在 cuLibraryLoadData/Ex、cuModuleLoadData/Ex 这类路径 patch。真实 vLLM 里还可能走：

JIT/link 路径：cuLinkCreate/cuLinkAddData/cuLinkAddFile/cuLinkComplete（最终产物往往是 link 后的 cubin，host 侧最该 patch 的就是这份）
load-from-file 路径：cuModuleLoad / cuLibraryLoadFromFile（有些环境会从缓存/磁盘加载）
graph 相关路径（不是 code object 入口，但影响“何时/哪次 launch”）：graph capture/replay 会改变调用链与 hook 时序
方案：把上述 API 全部 hook 起来，确保不管 vLLM 走哪个编译/加载路径，我们都能在“最终 cubin bytes”出现的那一刻 patch（并做 code hash 去重缓存）。

验收：跑 vLLM load test，打开 BPFTIME_CUDA_SASS_DETOUR_DUMP_DIR，能看到 patch 覆盖的 code object 数显著增加；至少 flash/cublasLt 相关 launch 不再出现“完全没有对应 code object 被处理”的情况。

1.2 再解决“strip 场景按 kernel 名字 filter 打不中”
这就是你前面问的 **“tag 归因/映射闭环”**要解决的核心痛点：

kernel_name 在 host 侧是有的（cuLaunchKernel/cuModuleGetFunction/cuKernelGetName 都能拿到）
但 cubin 内的 .text.<name> / symtab / 字符串可能被 strip 掉，导致我们在 patch 阶段无法用名字定位到“该 kernel 的 entrypoint/section”
解决思路（闭环）：

现状（已做的最小闭环，flash 首个目标可跑通）：

- 通过 fatbin 中“仍保留 `.nv.info.<kernel>` section name”的 code object，提取 `sh_info(func_id)` 集合（不依赖 `.text.<name>`）。
- 在 SM120 code object 的 `.text.*` 名字被 strip 的情况下，允许用 `func_id` 过滤来命中目标入口（`BPFTIME_CUDA_SASS_DETOUR_FILTER_FUNC_IDS` 或自动学习的 `func_id` 集合）。
- 增加持久化缓存：`BPFTIME_CUDA_SASS_DETOUR_FUNC_ID_CACHE_PATH`（默认 `/tmp/bpftime-sass-funcid-cache.json`）把 “filter_substr → func_id[]” 写盘，下一次运行无需手工抄 `func_id`。
- 为了避免“func_id 小整数在不同模块里碰撞导致误 patch”，raw-ELF 路径只在“近期 fatbinc 命中过 filter”（fallback budget > 0）时才启用 learned func_id（把影响范围收敛到同一批次的 code object）。

验收（当前可以做的）：

- 只设置 `BPFTIME_CUDA_SASS_DETOUR_FILTER=flash_fwd_splitkv_kernel`（不手写 func_id），跑真实 vLLM（Qwen/Qwen3-0.6B）：
  - 第一次运行会学习并写出 cache（日志：`learned ... func_id`）
  - 后续运行会读取 cache 并对 SM120 cubin 产生 `sm120_detoured_*.cubin`（dump dir 可见）

仍未完成/后续要补的“完整闭环”（更像 hetGPU 的系统化做法）：

- 真正的“单次运行内 name→entrypoint 精准闭环”：需要 identify/tag 输出机制，把 launch 的 `kernel_name` 和 device 写出的 tag/func_id 对齐，避免依赖跨 run cache。
- 更强的 name→entrypoint 枚举：覆盖只有 `.text` 聚合段、无 `.text.*` 的场景（见 1.3）。
- 对 strip 场景的“归因闭环”要输出更结构化的 mapping（比如记录到 JSON：kernel_name → {module_hash, func_id[]}），用于稳定 detour + 采样范围控制。

实测（当前环境：vLLM 0.13.0 + Qwen/Qwen3-0.6B + in-process 脚本）：

- “raw-ELF 先出现、fatbinc 后出现”的顺序非常常见：连续 8 次运行里，日志里 `magic=0x464c457f` 的 raw ELF 总是在首次 `learned ... func_id`/`fatbinc contains filter name` 之前出现（8/8）。
- 更关键的是：单次运行内目前确实会“来不及 patch SM120”：
  - 5 次 cold run（每次都清空 func_id cache）里，`BPFTIME_CUDA_SASS_DETOUR_DUMP_DIR` 下 **没有任何** `sm120_detoured_*.cubin`（0/5）。
  - 同一进程配置下做两次运行（Run1 生成 cache，Run2 复用 cache），Run2 每次都会产出 `sm120_detoured_*.cubin`（3/3，且每次 8 个 detoured cubin）。

结论：对 flash 这类目标，“只靠跨 SM 传播”在当前 vLLM load 顺序下往往需要 **至少 2 次启动**（Run1 学习/写 cache，Run2 才能稳定命中 SM120）；要做到“一次启动就命中”，必须上 identify/tag 闭环或更激进的 load-delay/patch-all 策略（不推荐）。

下一步：落地“单次运行 identify/tag 闭环”的 Step 1 设计与实现记录见 `design/gpu_sass_identify_closure.md:1`。

patch 阶段不再依赖名字：对 cubin 里所有可 patch 的 entrypoint/func 赋一个稳定 tag（优先用 sh_info(func_id) 这种“即使 stripped 也存在”的标识）。
stub 在 device 侧把“我是谁（tag）”写出来（一次即可，低成本）。
host 侧把这次 launch 的 kernel_name 和 device 写出的 tag 对齐，形成 kernel_name -> tag 映射表。
后续就能做到：用户给 kernel_name，我们按 tag 精准启用/禁用 detour/采样，不需要 cubin 内有名字。
验收：用 flash_* 做目标，哪怕它对应的 cubin stripped、.text.* 名字不可用，也能通过一次“identify”建立映射，然后稳定只对 flash 的 tag 打开采样/更重 stub。

1.3 处理“没有 .text.* section 名称/甚至只有 .text 聚合段”
这是更硬的覆盖问题：如果 ELF 只有一个 .text 段而没有 per-function .text.<name>，那就缺少“每个函数入口偏移”的枚举依据。

如果还保留 symtab 或 nvinfo 能给出入口偏移：可枚举 entrypoint。
如果两者都没有：就只能退化为 patch-all（粗粒度）或放弃“按函数入口精确插桩”（除非走更底层的 driver 私有接口/反汇编恢复函数边界，复杂且不稳）。
方案：分级 fallback

L0：.text.<func> 可用 → 精准
L1：只有 .text 但能从 symtab/nvinfo 拿到 entry offsets → 精准
L2：拿不到 offsets → 只能 patch-all（或者仅做 host 侧观测）
验收：dump 出来的 “unpatched raw cubin” 分类原因减少；至少对 vLLM 常见 code object 不再大量落入 L2。

2) “稳定性风险”怎么解决（崩/illegal access/扰动）
核心原则：把 patch-all 的“结构性风险”和采样写出的“运行时风险”拆开控制。

2.1 分两级 stub：默认极轻，按需变重
Level 0（默认、安全）：只做最小 detour trampoline（replay + 回跳），不做任何写出；目的是“patch-all 不至于把 vLLM 弄崩”。
Level 1（按 tag enable）：才执行写出/采样；并且必须可控（lane0/warp0/tid0、采样率、上限、只写一次等）。
这样即使为了覆盖面采取 patch-all，也不会把所有 kernel 都变成高开销/高风险。

2.2 写出通道做成“硬边界 + 可限流”
你已经看到：全线程写出在真实 workload 下容易触发非法访存/不稳定。
稳定化手段（按优先级）：

硬边界：任何写出必须保证 index 永不越界（mask/上限/原子分配 + cap）。
密度控制默认收紧：默认至少 lane0_only 或 warp0_only；需要用户显式放开全写。
一次性/采样率：支持“只记录第一次命中/每 N 次命中”，避免 graph warmup 阶段刷爆 buffer。
并发写冲突可接受但要可解释：允许覆盖（debug 场景），但要能通过 tag/slot/epoch 判断“覆盖发生了”。
2.3 寄存器/谓词保护系统化（防 silent wrong）
目前我们已经把 P0 gate 修正了，但真实 vLLM 下一步要做的是：

PR（谓词）全量保存/恢复（hetGPU 风格）：避免未来 stub 用到更多 predicate 时踩到原 kernel 的 PR 状态。
UR/GPR 更通用的分配与 headroom：如果 regcount patch 失败，就必须自动降级到 Level0（不允许写出）。
control word 生成/校验策略：像这次 ISETP 就是典型“编码能跑但语义不稳”，后续必须用小 probe cubin + cuobjdump 对照来维护一组“可信模板”。
2.4 回归验证策略（必须能定位到“哪个 kernel/哪个 code object 触发问题”）
每次崩溃都要能输出：kernel_name、对应 code hash、是否 patched、以及 dump 出对应 cubin（或 raw cubin）。
先在单 kernel filter + 低密度 gate 下稳定，再逐步扩大范围（扩大 tag 集合、扩大写出密度、扩大 workload）。

## 当前 checkpoint（2026-01-07）

- **thread_map(device) LANE0_ONLY gate + slot 映射已收敛（最小例验证）**：
  - `call_entry`（`benchmark/gpu/host/cubin_call_replay_test.py`）：
    - `BPFTIME_TEST_BLOCK=128`：`LANE0_ONLY`≈4 条、`WARP0_ONLY`≈31 条、两者都开≈1 条（写出密度降到安全范围）
    - `BPFTIME_TEST_BLOCK=1024`：`LANE0_ONLY`=32 条，并满足 `slot == raw_u32(tid)`（用 `BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_DEBUG_STORE_TID=1` 断言）
  - 关键修复：`idx` 计算与 `IMAD.WIDE` 之间必须留 gap，否则会出现“stale idx（常见为 0）→ 大量写到 slot0 / 甚至触发 717”的非确定性；当前默认插入 `BPFTIME_CUDA_SASS_DETOUR_IDX_WAIT_NOPS=8` 个 NOP（可调）。
- **identify/tag 闭环**：
  - write-only identify 可用：`BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE=1` + `BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_SYNC_AFTER_LAUNCH=1` 时，stub 可写出 `func_id(sh_info)` 到 control slots，host 当次 launch 后 dump 学习并写入 `/tmp/bpftime-sass-funcid-cache.json`（最小复现：`benchmark/gpu/host/cubin_call_replay_test.py`）。
  - read-control 仍不可用：在 trampoline 内执行 `LDG` 读取 control header 依旧可能触发 `cuCtxSynchronize=700`（`BPFTIME_CUDA_SASS_DETOUR_CONTROL_READ_GLOBAL=1`），需要后续继续攻关/替代控制通路。


## 20260108手动checkpoint

目前进度可以分成 3 条线（“能跑起来/能写出/能闭环”）：

SASS detour + device-per-thread 写出：链路已跑通

我们已经能在 SM120 的 cubin-only（无 PTX） 上做 .text.* 级别 patch，并把 thread_map 的结果从 device 侧写到 buffer，再由 host dump 成 JSONL。
之前你看到的 “LANE0_ONLY 仍然很多非 lane0 写、甚至 vLLM illegal access” 的核心根因更偏底层：idx/地址计算的指令依赖不稳定（stale 值），导致大量线程写到了错误 slot（看起来就像 gate 失效/到处在写）。这条已经做了修复：把 idx 生成改成更稳的组合逻辑，并在关键依赖点插入可调 NOP（IDX_WAIT_NOPS / S2R_WAIT_NOPS），同时把 gate 改成 predicated BRA skip-store（更可靠、也更容易 debug），并加了 DEBUG_STORE_TID/IDX 方便对齐校验。
真实 vLLM（Qwen3-0.6B）能启动：已打通，但有运行形态约束

vLLM 默认会走多进程（V1 engine 的 multiprocessing），LD_PRELOAD + CUDA init 在子进程里容易出问题；目前通过 VLLM_ENABLE_V1_MULTIPROCESSING=0 强制单进程 可以稳定跑通 bpftime + vLLM，并能产出 detoured cubin dumps / CUDA trace。
1.2：flashattention “strip 场景闭环”推进到一半（卡在采样/归因闭环）

好消息：已经能从 vLLM 的 CUDA trace 里定位到 flash kernel 名称（flash_fwd_splitkv_kernel...），并且能在 fatbin 的“其它 SM code object（名字没 strip）”里提取到 func_id 集合，写到 bpftime-sass-funcid-cache.json（用于 patch SM120 stripped code object）。这说明 “name→func_id 学习”路径是通的。
现在卡点：patch 命中了（能 detour），但 sample 仍可能 0 records。原因不是没 patch 到，而是我们早期的 sampling 插桩条件仍偏 “按 section 名称匹配”，对 strip 掉名字的 SM120 .text.* 会判定为不该 sample，于是 stub 没插进去 → dump 0。这个属于 1.2 自动闭环的缺口：detour/filter 逻辑已经能靠 func_id 命中，但 sampling/归因没跟上（“命中了却不写出”）。
接下来还需要做什么（按优先级）：

把 1.2 真正做成“自动闭环”：当通过 name→func_id 已经选中了某个 SM120 section 时，sampling 也必须“跟随命中”插入 stub，而不是依赖 .text.<name> 字符串匹配；否则 flash 这类 strip 场景永远看不到采样数据。
解决“单次运行闭环”的 load 顺序问题：如果 SM120 raw-ELF 先加载、fatbin 后出现，那么同一次运行里学不到 func_id 就来不及 patch；跨 run 的 cache 只能让“第二次运行好”。要做到单次闭环，需要引入你之前认可的 identify/tag 机制（先粗粒度插桩写出 func_id/tag → 对齐 launch → 形成映射 → 回到精确 detour）。
补齐覆盖面（最致命）：面对 “只有 .text 聚合段 / 没有 .text.* / 没有 section header” 的 code object，需要落地 L0/L1/L2 分级 fallback（symtab/nvinfo entry offsets、否则 patch-all/降级 host-only），不然真实 vLLM 里会经常出现 no_cubin_dumps / 0 records。
继续收敛稳定性风险：即使 lane0-only 正确，也要控制写出密度/格式，避免对 flash/cutlass 这类 kernel 触发 address-space 假设/并发写越界（这也是你强调“先收敛再放开 per-thread”的原因）。

## 20260108 进度记录（1.2 自动闭环：sampling 跟随 func_id 命中）

问题：在 strip 场景下（SM120 `.text.*` 没有 kernel 名字），detour 可以通过 fatbin 内其它 SM 的 `.nv.info.<kernel>` 提取到 `func_id(sh_info)` 并命中 SM120 入口；但 sampling 以前默认依赖 `.text.<name>` 字符串匹配，导致“detour 命中但 sample stub 没插入 → dumped 0 records”。

修复：当用户未显式设置 `BPFTIME_CUDA_SASS_SAMPLE_FILTER` 时，不再把 sample filter 默认继承为 detour filter 字符串；改为传空（表示 sampling 只要“该 section 已被 detour 选中”就插 stub）。这样在 `func_id` 命中的 strip 场景里，sampling 会自动跟随命中，不再依赖 `.text.<name>`。

代码位置：
- `attach/nv_attach_impl/nv_attach_impl_frida_setup.cpp`：`sample_filter_sv` 默认改为空（仅在用户显式设置时才生效）。
- `attach/nv_attach_impl/sass_detour/sass_detour.cpp`：保留“detour+sample 使用同一字符串时，detour 通过 func_id/symtab 命中也视为 sample_match”的兜底逻辑（避免用户显式设置 sample_filter 时仍掉坑）。

## 20260108 进度记录（func_id collision 修复 + vLLM 回归现象）

现象（修复前）：在开启 `flash_fwd_splitkv_kernel` 的 `func_id` 过滤时，出现过 vLLM `CUDA illegal instruction`；排查发现被 detour 的并不只有 flashattention，对一些 `at::native::*` 内核也插入了 stub。

根因：`func_id(sh_info)` **不是全局唯一**，不同 fatbin/ELF 间会复用相同数值。我们把“学习到的 func_id 集合”全局应用到了所有 fatbin 的 SM120 `.text.*` 过滤上，导致在不相关模块里发生 `func_id` 数值碰撞，从而误 patch。

修复：在 fatbin patch 路径上，只有当该 fatbin **确认包含 filter 名字**（`saw_filter_name_anywhere`）时，才启用 learned/cached 的 `func_id` 过滤；否则仅允许 `.text.<name>` 字符串匹配（以及用户手工 `BPFTIME_CUDA_SASS_DETOUR_FILTER_FUNC_IDS`）。

回归结果：
- 修复后，真实 vLLM cold run 可正常完成（不再触发 illegal instruction）。
- 但在 “raw-ELF 先于 fatbinc 出现” 的 load 顺序下，flashattention 的 SM120 raw-ELF 仍可能在同一次运行里 **来不及** 使用从 fatbin 学到的 `func_id` 去 patch（所以 `dump_on_first_sampled_launch` 仍可能出现 0 records）。这属于后续 Step 1（identify/tag 单次闭环）要解决的 load 顺序问题。

## 20260108 进度记录（Step 1：单次运行闭环的 load 顺序修复 - identify_pending gated patch-all）

问题：在 “raw-ELF first” 场景里，SM120 的 raw-ELF code object 在同一次运行里可能比 fatbinc 更早出现；而 raw-ELF 往往不包含 `flash_fwd_splitkv_kernel` 这类可读字符串，导致按名字/func_id 都匹配不到，于是 **来不及在第一次运行里把 stub 插进去**（跨 run 的 func_id cache 只能让第二次运行生效）。

关键观察：很多 PTX→SASS 的 JIT/加载是 **lazy** 的，可能发生在第一次 `cuLaunchKernel` 周期内（或者紧邻 launch 的内部路径）。这意味着我们可以在 host 侧“先 arm Identify”，并把“接下来一小段窗口内出现的 raw-ELF SM120”视为闭环候选，然后对这些 raw-ELF 做 patch-all（但 stub 依然是 gated 的，默认 enable=0，不会长期扰动）。

落地：在 raw-ELF patch 路径里引入 `identify_pending` 作为 gating：
- 当 `cuLaunchKernel` 里触发 `sass_control_arm_identify()` 后，`identify_pending=true`。
- 若此时出现 SM120 raw-ELF 且 `apply_elf_text_detours_sm120(..., filter=...)` 返回 `no .text.* sections matched filter`，就启用 patch-all（`filter=""`）把 controlled stub 插进去。
- 这样同一次运行里：第一次 launch 走 Identify 写出 slots → host sync/dump 选出 func_id → 立即切 Target → 后续 launch 立刻能 sample（不依赖第二次运行）。

代码位置：
- `attach/nv_attach_impl/nv_attach_impl_frida_setup.cpp`：raw-ELF 分支里用 `identify_pending` 触发 identify-closure patch-all fallback（避免对所有 raw-ELF 无差别 patch-all）。

## 20260108 进度记录（Step 1 补强：raw-ELF “contains filter” 提前 patch-all）

新增问题：有些 “raw-ELF first” 的 code object **并不发生在 `cuLaunchKernel` 内部的 lazy load**，而是更早（模型/库初始化阶段）就被 `cuModuleLoadData*` 加载；这样 `identify_pending` 还没被置位，raw-ELF 也就不会触发 gated patch-all，导致同一次运行里永远学不到 slots（第一次 launch 也不会写）。

补强策略：如果 raw-ELF 本体里还能搜到 filter 字符串（例如保留了 `.nv.info.<kernel>` / debug / 元数据），那我们就可以在 “identify_pending=false” 的情况下也判断它**很可能就是目标模块**，从而提前启用 patch-all（仍然是 controlled stub，默认 enable=0，只有 host arm Identify/Target 时才会真的写出/采样）。

落地改动：
- `attach/nv_attach_impl/nv_attach_impl_frida_setup.cpp`：raw-ELF 分支 patch-all 条件从 `identify_pending || force_on_miss` 扩展为 `identify_pending || force_on_miss || raw_elf_contains_filter`（仍然要求 `BPFTIME_CUDA_SASS_DETOUR_PATCH_ALL_FALLBACK=1` 且 reason 为 `no .text.* sections matched filter`）。

最小回归：
- `benchmark/gpu/host/cubin_call_replay_test.py` 仍可正常启动并写出（CTA 模式 1 条记录），未引入构建/运行回归。

## 20260108 进度记录（regcount=255：vLLM flash 上跑出 marker + ≥1 record）

目标：在真实 vLLM in-process（`benchmark/gpu/vllm_observability/vllm_inprocess_generate.py`，vLLM 0.13.0）里，针对 flashattention（大量 regcount=255 的 cubin-only 代码对象），先把“最小可观测链路”跑通：能看到 **Target 选择 marker** + **至少 1 条 sampler record**（warp_map）。

现状与结论：
- **vLLM in-process 可跑**：`VLLM_ENABLE_V1_MULTIPROCESSING=0` + `--enforce-eager` 下可稳定启动并完成一次 generate。
- **flashattention code object 可被处理**：需要 `BPFTIME_CUDA_SASS_DETOUR_JITLINK_PTX=1`（PTX→cubin 的 jitlink 路径），并能看到 `cuLibraryLoadData jitlink PTX->cubin ... patched_text_sections=...`。
- **regcount=255 的“device 侧 Target 控制读取”仍不稳定**：对 regcount=255 kernel 试图在 trampoline 里读取 control header（LDG/ISETP/BRA gating）会在真实 workload 上触发不稳定（常见表现为后续 `CUBLAS_STATUS_INTERNAL_ERROR`）。目前将该路径标记为 bring-up-only，需要显式打开 `BPFTIME_CUDA_SASS_DETOUR_NO_REGCOUNT_CONTROL_UNSAFE=1` 才会启用。
- **先用“host marker + device 最小 record”打通闭环**（稳定）：
  - host 在 `set_target` 时把 `control.reserved0="TARG"`、`control.reserved1=target_func_id` 写到 control header（用于对齐/验证 Target 已经选中）。
  - regcount=255 的 identify-only fallback stub 在 lane0 额外写一条最小 record（写到 warp_map 的 `data[0]`），保证 dump 至少有 1 条 `bpftime_sass_warp`。

复现命令（示例）：
```bash
HF_HOME=/tmp/hf-qwen3 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
BPFTIME_CUDA_SASS_DETOUR=1 \
BPFTIME_CUDA_SASS_DETOUR_JITLINK_PTX=1 \
BPFTIME_CUDA_SASS_DETOUR_FILTER=flash_fwd_splitkv_kernel \
BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE=1 \
BPFTIME_CUDA_SASS_DETOUR_NO_REGCOUNT_CONTROL=1 \
BPFTIME_CUDA_SASS_SAMPLE=1 \
BPFTIME_CUDA_SASS_SAMPLE_MODE=warp \
BPFTIME_CUDA_SASS_SAMPLE_MAX_RECORDS=1 \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_PATH=/tmp/bpftime-vllm-flash-safe2.jsonl \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_ON_SYNC=1 \
BPFTIME_CUDA_SASS_SAMPLE_SYNC_DUMP_MAX_ATTEMPTS=2000 \
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
.venv-vllm/bin/python3 benchmark/gpu/vllm_observability/vllm_inprocess_generate.py \
  --model Qwen/Qwen3-0.6B --max-tokens 1 --enforce-eager
```

期望输出（关键点）：
- `/tmp/bpftime-vllm-flash-safe2.jsonl` 至少两行：
  - meta 行里 `control.mode==2`（Target），`control.target_func_id!=0`，并且 `control.reserved0==0x54415247`（"TARG"）；
  - 至少一条 `{"type":"bpftime_sass_warp",...}` record。

下一步（真正“device 侧 Target-hit gating”要补的东西）：
- regcount=255 kernel 里不能靠“共享 scratch + save/restore”去保护寄存器/谓词（存在跨 warp/CTA 的数据竞争）；要做成稳定的 Target gating，需要更像 hetGPU 的：
  - 可证明不破坏的寄存器/谓词保存策略（例如用不会冲突的 scratch 方案，或更体系化的 UR/stack 保存），
  - 更系统的 patch/layout（更长 stub、更可靠的 code cave/reloc），
  - 再逐步把 `BPFTIME_CUDA_SASS_DETOUR_NO_REGCOUNT_CONTROL_UNSAFE` 变成默认可用路径。

## 20260108 进度记录（regcount=255 Target gating 稳定性：先收敛“影响面”，再做单次闭环）

现象回归：
- 开启 `BPFTIME_CUDA_SASS_DETOUR_NO_REGCOUNT_CONTROL_UNSAFE=1`（device 侧读 control header + Target gating）仍会在真实 vLLM workload 上高概率触发 `CUBLAS_STATUS_INTERNAL_ERROR`（更像“异步非法访问/状态被破坏”的后效表现）。
- 关闭该开关（只做 write-only identify + 最小 record）则 vLLM 可稳定跑完，并且 dump 能看到：
  - `control.mode=Target` + `reserved0="TARG"`（host marker）
  - 至少 1 条 `bpftime_sass_warp` record（device 最小写出）

已做的收敛改造（优先降低误伤/扩大影响面导致的“随机崩溃”）：
- patch-all fallback（identify closure 用）不再做 sampling/控制读取：新增 `patch_all_is_fallback` 路径，让 patch-all 仅插入最小 identify stub（避免在“为了闭环而 patch-all”的阶段对大量非目标 kernel 执行 LDG/采样写出）。
  - 代码：`attach/nv_attach_impl/nv_attach_impl_frida_setup.cpp` → 调用 `apply_elf_text_detours_sm120(..., patch_all_is_fallback=true)`
  - 代码：`attach/nv_attach_impl/sass_detour/sass_detour.cpp` → `apply_elf_text_detours_sm120(..., patch_all_is_fallback)`
- regcount=255 no-regcount stub 做了更保守的寄存器/谓词处理：
  - 入口保存/恢复 P0（降低 “entry predicate state” 带来的不确定性）。
  - lane0 gate 改为 `tid_x & 31`（避免依赖 `SR_LANEID`）。
  - 优先用内核自身的 `LDCU` descriptor 模板走 `STG desc[URx][R.64+imm]` 写出（尽量不去 clobber `cfg.desc_ur` 的 UR base ptr）。
  - 代码：`attach/nv_attach_impl/sass_detour/sass_detour.cpp` → `build_sm120_identify_only_stub_no_regcount(...)`

下一步（单次运行闭环的“无 device-load Target gating”路线，优先落地）：
- 目标：不依赖 regcount=255 kernel 内部的 `LDG` 读取 control header（这条路目前不稳定），而是在 host 侧完成 “name→func_id→精确 patch”：
  1) 第一次命中 kernel_name 时：允许 on-demand patch-all（只为让 identify stub 存在），arm Identify，sync 后 dump 出 func_id。
  2) 从 dump 得到 `kernel_name -> func_id` 后：下一次 launch 时走 **on-demand func_id 定向 patch**（只 patch 目标 `.text.*(sh_info==func_id)`），并记录 `kernel_name -> patched CUfunction`，后续 launch 不再走 patch-all。
  3) 在这个 “func_id 定向 patch” 的版本里，保持 write-only stub（不做 device control reads），用 “只 patch 目标 kernel” 实现 Target gating（影响面极小，且更稳定）。
  - 代码已补齐：`attach/nv_attach_impl/nv_attach_impl_frida_setup.cpp` → `sass_detour_on_demand_patch_func_id_for_kernel(...)`

## 20260108 进度追加（no-regcount 最小 record + dump 时机收敛）

本轮目标：先把 **真实 vLLM（in-process, Qwen/Qwen3-0.6B）** 上 `flash_fwd_splitkv_kernel` 的 **regcount=255 场景** 跑出
“Target 已选中（host marker）+ 至少 1 条 record”，再收敛 dump 时机/次数。

已完成：
- **regcount=255 no-regcount identify-only stub 的 write-only 写出改造**（可稳定写出）：
  - 不再尝试通过 `LDCU.64` 得到的 descriptor 做 `STG.E desc[...]` 写出（在 flashattention 上出现“slots 仍为 0 / 看起来像 silent no-store”的现象）。
  - 改为使用 **UR-base `[RZ + UR + imm]`** 写出（默认使用 `desc_ur=62`，尽量避免 clobber 编译器常用的 `UR4/UR5`）。
  - write-only identify 写出内容：
    - `control.slots[0] = func_id`（用于单次闭环 identify）
    - `data[0] = smid_raw`（保证 dump 至少产生 1 条 `bpftime_sass_warp` record）
  - 代码：`attach/nv_attach_impl/sass_detour/sass_detour.cpp` → `build_sm120_identify_only_stub_no_regcount(...)` 的 `write_only_identify` 分支。
- **dump-on-sync 的“早期 sync 消耗 attempts”问题修复**：
  - vLLM 初始化阶段会触发多次 `cuCtxSynchronize`，之前会提前 dump（但没有 record），从而把 `BPFTIME_CUDA_SASS_SAMPLE_SYNC_DUMP_MAX_ATTEMPTS` 很快消耗完，导致真正命中 flash kernel 后反而不 dump。
  - 现在：当 identify closure enabled 时，只有在 **identify 已 armed（pending/epoch）或 target 已设置（identify_target_func_id）** 后才允许 dump-on-sync，避免“初始化阶段空 dump”污染。
  - 代码：`attach/nv_attach_impl/nv_attach_impl.cpp` → `nv_attach_impl::maybe_dump_sass_samples_on_sync()`。

验证结果（关键点）：
- 运行时日志能看到：
  - `SASS control: armed identify epoch=1 ...`
  - `SASS control: identify dump ... slots=[269,0,0,0,0,0,0,0] unique_ids=1 filter='flash_fwd_splitkv_kernel'`
  - `SASS control: set target func_id=269 ...`（host marker）
  - `SASS sample: dumped 1 records to ...`
- dump 文件示例（两行）：
  - meta 行里 `control.mode==2`（Target），`control.reserved0==0x54415247`（"TARG"）
  - 至少一条 `{"type":"bpftime_sass_warp",...}`，且 `smid_raw` 是合理的 SMID（0~255）

复现（dump-on-sync）：
```bash
HF_HOME=/tmp/hf-qwen3 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_ENFORCE_EAGER=1 \
BPFTIME_CUDA_SASS_DETOUR=1 \
BPFTIME_CUDA_SASS_DETOUR_JITLINK_PTX=1 \
BPFTIME_CUDA_SASS_DETOUR_FILTER=flash_fwd_splitkv_kernel \
BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE=1 \
BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_SYNC_AFTER_LAUNCH=1 \
BPFTIME_CUDA_SASS_SAMPLE=1 \
BPFTIME_CUDA_SASS_SAMPLE_MODE=warp \
BPFTIME_CUDA_SASS_SAMPLE_MAX_RECORDS=1 \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_PATH=/tmp/bpftime-vllm-flash-sync.jsonl \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_ON_SYNC=1 \
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
.venv-vllm/bin/python3 benchmark/gpu/vllm_observability/vllm_inprocess_generate.py \
  --model Qwen/Qwen3-0.6B --max-tokens 1 --enforce-eager
```

## 20260109 进度记录（1.2 自动闭环：func_id-only 精准 detour 默认路径）

目标：把 “按 kernel 名字精准 detour” 做成默认稳定路径（不依赖 `.text.<name>` 字符串匹配），在真实 vLLM（flashattention/strip 场景）上能做到：

- 第一次运行：identify 学到 `func_id`（并写入 `BPFTIME_CUDA_SASS_DETOUR_FUNC_ID_CACHE_PATH`）。
- 后续运行：load-time 直接只 detour 命中的 `func_id`（避免 patch-all / 避免 detour 过多 template instantiation），并且 sampler 插桩也“跟随命中”。

关键修复与结论：

1) `func_id` 的语义修复：有些 flashattention code object 里，`.nv.info.*` 的 `sh_info` 并不对应 `.text.*` 的 `sh_info`，而是对应 **`.text.* section index`**。
- 现在在 `apply_elf_text_detours_sm120()` 内会探测这种形态（`func_id_matches_text_section_index`），并把用于 stub 写出/匹配的 `func_id` 统一为 `text_func_id`（section index 或 `sh_info` 二选一）。
- 直接收益：identify 产出的 `func_id` 能被后续 “按 func_id patch” 命中，不会出现 “学到 269 但 patch 命中不到” 这类假闭环。

2) 精准 detour 的默认策略：当同一个 filter 最终只解析到 **唯一的 func_id** 时，load-time **忽略文本名 substring filter**（`apply_filter_sv=""`），只用 `func_id` 去选择要 patch 的 `.text.*`。
- 目的：避免 flashattention 这类 “filter 命中很多 `.text.*`（大量模板实例）” 时的过度 detour（影响面/稳定性/性能）。
- 结果：日志可看到 `patched_text_sections=1`，而不是动辄几十个。

3) sampling “跟随命中”的闭环：`BPFTIME_CUDA_SASS_SAMPLE_FILTER` 默认保持为空（不做 `.text` 名字过滤），因此 sampling 会对所有 detour-selected 的 section 生效；并额外修正了 “func_id-only detour 时 sampling 仍需视为 match” 的逻辑，确保 strip 场景不会因为 `.text.<name>` 不可用而永远 0 records。

4) on-demand upgrade（单次运行内从 identify → 立刻精准 detour）仍非默认：
- 我们实现了 `BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE_ON_DEMAND_UPGRADE=1` 的 on-demand patch（在 `cuLaunchKernel` 里尝试生成/替换为 func-id-targeted patched CUfunction）。
- 但在真实 vLLM flashattention 上该路径仍会出现 `CUDA_ERROR_INVALID_VALUE / invalid argument` 风险；因此目前 **默认关闭**，优先保证 “跨 run cache + load-time func_id-only 精准 detour” 的稳定性。

复现建议（两次运行闭环，第二次即为默认精准 detour）：

```bash
# Run 1: learn func_id (writes cache)
HF_HOME=/tmp/hf VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_ENFORCE_EAGER=1 \
BPFTIME_CUDA_SASS_DETOUR=1 \
BPFTIME_CUDA_SASS_DETOUR_JITLINK_PTX=1 \
BPFTIME_CUDA_SASS_DETOUR_FILTER=flash_fwd_splitkv_kernel \
BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE=1 \
BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_SYNC_AFTER_LAUNCH=1 \
BPFTIME_CUDA_SASS_DETOUR_DEBUG=1 \
BPFTIME_CUDA_SASS_DETOUR_FUNC_ID_CACHE_PATH=/tmp/bpftime-sass-funcid-cache-flash.json \
BPFTIME_CUDA_SASS_SAMPLE=1 BPFTIME_CUDA_SASS_SAMPLE_MODE=warp BPFTIME_CUDA_SASS_SAMPLE_MAX_RECORDS=1 \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_PATH=/tmp/bpftime-vllm-flash-run1.jsonl \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_ON_SYNC=1 \
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
.venv-vllm/bin/python3 benchmark/gpu/vllm_observability/vllm_inprocess_generate.py \
  --model Qwen/Qwen3-0.6B --max-tokens 1 --enforce-eager

# Run 2: cache hit => load-time func_id-only => patched_text_sections=1 + ≥1 record
HF_HOME=/tmp/hf VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_ENFORCE_EAGER=1 \
BPFTIME_CUDA_SASS_DETOUR=1 \
BPFTIME_CUDA_SASS_DETOUR_JITLINK_PTX=1 \
BPFTIME_CUDA_SASS_DETOUR_FILTER=flash_fwd_splitkv_kernel \
BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE=1 \
BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_SYNC_AFTER_LAUNCH=1 \
BPFTIME_CUDA_SASS_DETOUR_DEBUG=1 \
BPFTIME_CUDA_SASS_DETOUR_FUNC_ID_CACHE_PATH=/tmp/bpftime-sass-funcid-cache-flash.json \
BPFTIME_CUDA_SASS_SAMPLE=1 BPFTIME_CUDA_SASS_SAMPLE_MODE=warp BPFTIME_CUDA_SASS_SAMPLE_MAX_RECORDS=1 \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_PATH=/tmp/bpftime-vllm-flash-run2.jsonl \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_ON_SYNC=1 \
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
.venv-vllm/bin/python3 benchmark/gpu/vllm_observability/vllm_inprocess_generate.py \
  --model Qwen/Qwen3-0.6B --max-tokens 1 --enforce-eager
```

验收点（Run 2）：
- 日志包含：`patched_text_sections=1`（或接近 1）；
- dump 文件至少两行（meta + 1 条 record），meta 里 `reserved0=="TARG"`（0x54415247）可用于对齐 Target 已生效；
- 不再需要手工抄 `func_id` / 不依赖 `.text.<name>` 匹配。

## 20260109 进度记录（日志/退出行为收敛：set_target 去重 + dump-on-exit 安静）

问题：
- `SASS control: set target func_id=...` 在真实 vLLM workload 下会被 `cuLaunchKernel` 频繁触发，导致 log spam（且每次都做一次 `cuMemcpyHtoD`）。
- 进程退出阶段偶发 `SASS sample: cuMemcpyDtoH_v2 failed, skip dump`：析构函数里仍尝试 dump，但此时 CUDA context 可能已销毁。

修复：
- host 侧按 `(epoch, func_id)` 去重：如果 target 没变，就跳过 `cuMemcpyHtoD` 与 log。
  - 代码：`attach/nv_attach_impl/nv_attach_impl_frida_setup.cpp` 的 `sass_control_set_target(...)`
  - 状态字段：`attach/nv_attach_impl/nv_attach_impl.hpp` 的 `sass_sampling_state_t::last_control_*`
- 析构 dump 收敛：
  - 若已成功 dump 过一次（`dumped_once=true`），析构时不再 dump（避免 teardown 时调用 CUDA API）。
  - 对 DtoH 的 deinit/invalid-context 错误降级为静默（仅 debug 下提示）。
  - 代码：`attach/nv_attach_impl/nv_attach_impl.cpp` 的 `maybe_dump_sass_samples()` / `dump_sass_samples_locked(...)`

回归（真实 vLLM flash，cache 命中 + 精准 detour）：
- 现在 Run 内只出现 1 条 `SASS control: set target func_id=261 ...`，且不再出现退出阶段的 `cuMemcpyDtoH_v2 failed` warn。

## 20260109 进度记录（单次运行闭环：on-demand upgrade 变稳定）

目标：不依赖“第二次运行 cache 命中”，在**同一次 vLLM 运行**里完成闭环：

1) identify 学到 `func_id`（slots 写出 + dump）。
2) 立刻把当前 kernel 升级为 **func-id-targeted** 版本（on-demand reload 新 module/library + 选择新 CUfunction）。
3) 后续同名 kernel 的 launch 使用新 CUfunction（精确 detour，影响面更小）。

关键问题（之前的 hard blocker）：

- on-demand upgrade 后，紧接着的 `cuLaunchKernel` 会返回 `CUDA_ERROR_INVALID_VALUE`，vLLM 侧表现为 flashattention `CUDA error: invalid argument`。
- 复现特征：日志出现 `SASS detour: on-demand func_id=... patched launch ...` 后，同一个 launch 的 `cuLaunchKernel exit res=1`。

根因（确认路径）：

- on-demand upgrade 会新加载一个 `CUfunction`（属于新 `CUlibrary/CUmodule`）。
- 但框架（vLLM/PyTorch/flashattention）可能在原始 `CUfunction` 上配置了 launch 相关状态（最典型是 **max dynamic shared memory**，也可能包含 cache/sharedmem config）。
- 新 `CUfunction` 不继承这些设置 → launch 参数在新函数上变为非法 → `CUDA_ERROR_INVALID_VALUE`。

修复：

- 在 on-demand upgrade 生成 `new_func` 后，自动把原始 `CUfunction` 的关键 launch 属性**克隆**到 `new_func`：
  - `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`
  - `CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT`
  - `cuFuncGet/SetCacheConfig`（若可用）
  - `cuFuncGet/SetSharedMemConfig`（若可用）
- 代码：`attach/nv_attach_impl/nv_attach_impl_frida_setup.cpp` 的 `maybe_clone_launch_attributes()`，以及 `sass_detour_on_demand_patch_*` 两个路径调用。

额外稳定性修复（jitlink PTX bring-up）：

- 修复了一个 string_view 悬空导致的崩溃：rewrite `.target sm_XX -> sm_120` 时先保存旧 token，再 mutate `blob`。（否则 `SPDLOG` 读到已失效的 view）
- 修复了 cuLinkAddData 在超大 PTX 上的驱动不稳定：默认只选择**较小**且包含 filter 的 PTX 输入，并引入上限：
  - `BPFTIME_CUDA_SASS_DETOUR_JITLINK_MAX_PTX_BYTES`（默认 20MiB）
  - `BPFTIME_CUDA_SASS_DETOUR_JITLINK_MAX_INPUTS`（默认 4）

回归验证（真实 vLLM + flashattention）：

- `--max-tokens 1`：能看到 `identify dump slots=[261,...]`，随后出现 `on-demand func_id=261 patched launch ...` 且该次 `cuLaunchKernel exit res=0`；无 `invalid argument`。
- `--max-tokens 8`：同样无 `invalid argument`，并且避免了先前 cuLinkAddData 的 segfault。

## 20260112 进度记录（image_id 口径统一 + bring-up 脚本恢复自动闭环）

问题（影响“按 kernel 名字精准 detour”的稳定性）：
- `learned_func_ids_by_image_id` 的 key（host-side 计算）与 `sass_detour::DetourResult::image_id`（patcher 内计算）口径不一致：前者 hash 全 buffer，后者 hash ELF 有效 extent；在 fatbinc padding / raw-ELF 推断 size 场景，会导致 image-scoped cache 命中率下降。
- identify->Target 切换后，host-side 的 `last_control_*` 未更新：后续每个 matching launch 都会重复走 `sass_control_set_target()` 做 `cuMemcpyHtoD` + 打 log（log spam）。
- bring-up 脚本依赖 `set target func_id=...` 日志提取 func_id；当 log 被 dedupe 后，脚本会误判“无法提取 target_func_id”。

修复：
- `attach/nv_attach_impl/nv_attach_impl_frida_setup.cpp`：新增 `sass_elf_image_id32()`，按 ELF section/header 计算 `file_end` 后再 hash，使 image_id 口径与 `sass_detour.cpp` 对齐。
- `attach/nv_attach_impl/nv_attach_impl_frida_setup.cpp`：`sass_control_maybe_dump_identify()` 写 Target header 后同步更新 `sass_sampling.last_control_*`，避免后续重复 `set_target`。
- `benchmark/gpu/vllm_observability/vllm_flash_reg255_threadmap_bringup.py`：func_id 提取新增 fallback：从 `SASS control: identify dump ... slots=[...]` 解析（不再依赖 `set target func_id` 日志）。

备注（关于跨 run 的 image_id 稳定性）：
- 单次运行内：image_id 可用于精确限定 code object（避免跨模块 func_id 冲突），对 “single-run closure + on-demand upgrade” 有益。
- 两次运行（identify 一次、thread-map 再起一次）：某些 JIT/link 路径可能生成不同 code object，跨 run 强制 `FILTER_IMAGE_ID` 可能降低命中率；因此 bring-up 脚本把它做成可选（`--legacy-use-image-id`）。

复现命令（单次闭环开启）：

```bash
HF_HOME=/tmp/hf VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_ENFORCE_EAGER=1 \
BPFTIME_CUDA_SASS_DETOUR=1 \
BPFTIME_CUDA_SASS_DETOUR_JITLINK_PTX=1 \
BPFTIME_CUDA_SASS_DETOUR_FILTER=flash_fwd_splitkv_kernel \
BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE=1 \
BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_SYNC_AFTER_LAUNCH=1 \
BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE_ON_DEMAND_UPGRADE=1 \
BPFTIME_CUDA_SASS_DETOUR_DEBUG=1 BPFTIME_CUDA_SASS_DETOUR_DEBUG_LAUNCH=1 \
BPFTIME_CUDA_SASS_DETOUR_FUNC_ID_CACHE_PATH=/tmp/bpftime-sass-funcid-cache-single-run.json \
BPFTIME_CUDA_SASS_SAMPLE=1 BPFTIME_CUDA_SASS_SAMPLE_MODE=warp BPFTIME_CUDA_SASS_SAMPLE_MAX_RECORDS=1 \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_PATH=/tmp/bpftime-vllm-flash-single-run.jsonl \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_ON_SYNC=1 \
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
.venv-vllm/bin/python3 benchmark/gpu/vllm_observability/vllm_inprocess_generate.py \
  --model Qwen/Qwen3-0.6B --max-tokens 1 --enforce-eager
```

## 20260109 进度记录（ThreadMap(device) 降密度：CTA clamp，便于逐步放开 device-per-thread）

背景 / 痛点：

- ThreadMap(device) 走 “device 侧每 thread 写出” 的时候，如果仍用 `ctaid_mod = ctaid & (max_records-1)`（mask-based），当 `max_records` 很小（比如 1）时，所有 CTAs 都会写到同一段 buffer，导致**极高 contention**（虽然不一定越界，但会放大稳定性风险/性能干扰）。
- 我们需要一个“更安全的逐步放开”开关：优先保证 **不越界 + 低干扰**，再逐步提高采样密度。

实现：

- 新增 ThreadMap(device) 的 CTA gating 模式：`BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_CTA_CLAMP=1`
  - 含义：当 `max_records!=0` 时，改为 **clamp**：
    - `if (ctaid.x >= max_records) skip stores`
  - 好处：当 `max_records` 很小时，只有前 `max_records` 个 CTA 会写出，避免所有 CTA 撞到同一段 buffer。
- SM120 stub 侧实现：`attach/nv_attach_impl/sass_detour/sass_detour.cpp` 的 `build_sm120_thread_map_stub(...)`
  - 复用 P0 做 `ISETP.GE + @P0 BRA`，并统一保存/恢复 P0（避免破坏 kernel entry 的 predicate state）。
- host 侧 meta 增加 `cta_clamp` 字段，便于 dump 里确认该模式生效：`attach/nv_attach_impl/nv_attach_impl.cpp`。

最小验证（无需 vLLM，先验证 stub 行为与映射正确性）：

```bash
BPFTIME_LOG_OUTPUT=console \
BPFTIME_CUDA_TRACE_PATH=/tmp/bpftime-testtrace.jsonl \
BPFTIME_CUDA_SASS_DETOUR=1 \
BPFTIME_CUDA_SASS_SAMPLE=1 \
BPFTIME_CUDA_SASS_SAMPLE_FILTER=call_entry \
BPFTIME_CUDA_SASS_SAMPLE_MODE=thread \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE=1 \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_STRIDE4=1 \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_CTA_CLAMP=1 \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_DEBUG_STORE_TID=1 \
BPFTIME_CUDA_SASS_SAMPLE_MAX_RECORDS=1 \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_MAX=40 \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_PATH=/tmp/bpftime-threadmap-clamp.jsonl \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_ON_SYNC=1 \
BPFTIME_TEST_BLOCK=1024 BPFTIME_TEST_GRID=2 \
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
python3 benchmark/gpu/host/cubin_call_replay_test.py
```

预期：

- `/tmp/bpftime-threadmap-clamp.jsonl` 第一行 meta 包含：`"thread_device":true` 且 `"cta_clamp":true`
- 记录里 `raw_u32 == tid_x`（debug_store_tid），说明 device-per-thread 的 slot->(ctaid,tid) 映射正确；
- `max_records=1` 时等价于“只采样 CTA0”，用于安全 bring-up（进一步放开可逐步调大 `max_records`）。

已知限制（和 vLLM flash 的关系）：

- regcount=255 的 kernel 目前无法做 ThreadMap(device)（`compute_sm120_layout()` 无法再提升 regcount 来预留 scratch），因此 flashattention 仍走 no-regcount identify-only 路径；ThreadMap(device) 的放开主要先用于 **非 regcount=255** 的 kernel（layernorm/rotary 等），作为下一阶段。

---

## 2026-01-09 进度追加：vLLM 上把 ThreadMap(device) 真跑出 record（rotary）

目标（阶段性 checkpoint）：

- 真实 vLLM in-process（v0.13.0）能启动并生成；
- 选一个 **非 regcount=255** 且可 patch 的 SM120 kernel（这里用 `vllm::rotary_embedding_kernel`）；
- ThreadMap(device) 在该 kernel 上能稳定输出：至少 `meta + >=1 条 bpftime_sass_thread record`；
- 然后逐步放开写出密度（lane0 → warp0 → full per-thread），确认不会因为写出密度直接把 vLLM 打崩。

### 关键卡点：一直只有 meta（0 records）的根因

现象：

- 即使 `module_patched=true`（`cuFuncGetAttribute` 看到寄存器数已被我们提升到 56），且用 `cuobjdump` 能确认 stub 里确实有 `STG.E [R..+UR62], R..`，dump 文件仍长期只有一行 meta（0 records）。

根因：

- `BPFTIME_CUDA_SASS_SAMPLE_DUMP_ON_SYNC=1` 时，vLLM/PyTorch 初始化阶段会触发很多早期 sync；
- `maybe_dump_sass_samples_on_sync()` 在**非 identify-closure 场景**默认 `max_attempts=1`，导致“第一次早期 sync”就把 dump 机会消耗掉；
- 后续即使 `BPFTIME_CUDA_SASS_SAMPLE_SYNC_AFTER_LAUNCH=1` 在匹配 kernel 后做了显式 sync，也会因为 attempts 已耗尽而不会再 dump，于是你看到的永远是“meta-only”。

### 修复：sync-after-launch 路径强制 dump（不走 attempts 预算）

代码：

- 新增 `nv_attach_impl::dump_sass_samples_force()`：`attach/nv_attach_impl/nv_attach_impl.hpp` / `attach/nv_attach_impl/nv_attach_impl.cpp`
- 在 `BPFTIME_CUDA_SASS_SAMPLE_SYNC_AFTER_LAUNCH=1` 的路径里，sync 后改为调用 `dump_sass_samples_force()`：
  - `attach/nv_attach_impl/nv_attach_impl_frida_setup.cpp`

推荐 bring-up 组合：

- `BPFTIME_CUDA_SASS_SAMPLE_DUMP_ON_SYNC=0`（避免早期 sync 抢占 dump 机会）
- `BPFTIME_CUDA_SASS_SAMPLE_SYNC_AFTER_LAUNCH=1`（在首次命中 filter 的 sampled kernel 后，显式 sync 并强制 dump）

### vLLM 回归验证（Qwen3-0.6B + rotary 作为 target kernel）

最小命中（tid0-only：lane0+warp0）：

```bash
HF_HOME=/tmp/hf \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_ENFORCE_EAGER=1 \
BPFTIME_CUDA_SASS_DETOUR=1 \
BPFTIME_CUDA_SASS_SAMPLE=1 \
BPFTIME_CUDA_SASS_DETOUR_FILTER=_ZN4vllm23rotary_embedding_kernelIN3c108BFloat16ELb1EEEvPKlPT_S6_PKS5_illliii \
BPFTIME_CUDA_SASS_SAMPLE_FILTER=_ZN4vllm23rotary_embedding_kernelIN3c108BFloat16ELb1EEEvPKlPT_S6_PKS5_illliii \
BPFTIME_CUDA_SASS_SAMPLE_MODE=thread \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE=1 \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_STRIDE4=1 \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_CTA_CLAMP=1 \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_LANE0_ONLY=1 \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_WARP0_ONLY=1 \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_USE_UR=1 \
BPFTIME_CUDA_SASS_SAMPLE_MAX_RECORDS=1 \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_PATH=/tmp/bpftime-vllm-threadmap-rotary.jsonl \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_ON_SYNC=0 \
BPFTIME_CUDA_SASS_SAMPLE_SYNC_AFTER_LAUNCH=1 \
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
.venv-vllm/bin/python3 benchmark/gpu/vllm_observability/vllm_inprocess_generate.py \
  --model Qwen/Qwen3-0.6B --max-tokens 1 --enforce-eager
```

预期：

- `/tmp/bpftime-vllm-threadmap-rotary.jsonl` 至少 2 行：`bpftime_sass_sample_meta` + 1 条 `bpftime_sass_thread`。

逐步放开密度（都保持 `max_records=1` + `CTA_CLAMP=1`，只让 CTA0 写，降低对真实负载干扰）：

- lane0-only（每 warp 1 条，共 ~16 条，block=512）：
  - `LANE0_ONLY=1, WARP0_ONLY=0`
- warp0-only（warp0 全 lane，共 32 条）：
  - `LANE0_ONLY=0, WARP0_ONLY=1`
- full per-thread（CTA0 的 512 threads 全写，共 512 条）：
  - `LANE0_ONLY=0, WARP0_ONLY=0`

备注：

- vLLM 会偶发打印 `cpuinfo` 的 `JSONDecodeError`（usage worker 线程），不影响主推理与本次观测链路。

---

## 为什么 regcount=255 的 flashattention 还不能 ThreadMap(device)？如果要做，怎么做（规划）

### 为什么现在不行（本质原因）

对 SM120 的“ThreadMap(device)”来说，我们的 stub 至少需要：

- 若干 GPR（`.64` 指针寄存器对 + idx/tid/lane/smid 等 scratch）；
- 至少一个谓词寄存器（P0）做 gate/跳转；
- 若干 UR（用于 base 地址或 desc）。

而 flashattention 这类 kernel 往往是 **regcount=255**：

- regcount=255 代表该 kernel 已经占满了可用的通用寄存器编号空间（0..254/255 的那种上限形态）。
- 对于“非 255”kernel，我们用“提升 regcount”这招给 stub 预留一段高位 scratch GPR（不影响原 kernel 的寄存器分配，因为原 kernel 根本没用到那些高编号寄存器）。
- **但 regcount=255 时无法再提升**，也就无法“凭空获得”新的 scratch GPR。

这会导致两个硬问题：

1) **没有可用 scratch 寄存器**：stub 写任何 GPR 都可能覆盖原 kernel 入口就要用的 live state（尤其是 flash/cutlass 这类高度手工优化的 prologue），从而触发看似随机的崩溃（如 `CUBLAS_STATUS_INTERNAL_ERROR` / illegal access）。

2) **save/restore 很难做到可证明安全**：
   - 你想“借用某些寄存器”，就必须保存再恢复；
   - 保存需要一个写出通道（global/local/shared）+ 地址计算 + 并发安全；
   - 但地址计算本身又需要 scratch 寄存器/谓词/UR，进入循环依赖；
   - shared scratch 会有跨 warp/CTA 竞争；local memory 需要 stack/地址寄存器（同样会扰动 prologue）；global scratch 会引入额外的内存访问与潜在的 address-space 假设破坏。

为什么 identify-only 能跑：

- identify-only stub 目前是“no-regcount fallback”的极小实现：只借用 **第一条指令的目的寄存器**（随后会被 replay 覆盖），并用最少的 UR-base 地址形式做一次写出，因此对 regcount=255 的侵入性小得多；但它不具备 thread-map 这种高密度写出能力。

### 如果要做（路线 B / 长期）：把 regcount=255 的 ThreadMap(device) 做成稳定能力

目标：在不依赖 “提升 regcount 预留新 GPR” 的前提下，仍能在 flashattention 上做 **可控密度** 的 device-side 写出（至少 lane0-only / warp0-only），并逐步扩展到更高密度。

建议分阶段：

1) 选“更安全的 patch 点”（尽量不在 entry）
   - 优先在 kernel 的 **epilogue/return 附近**或某段稳定的同步点后插入（此时寄存器 live 集合通常更小，且 SR_LANEID/CTAID/TID 语义更稳定）。
   - entry 插桩对 reg255 风险最高（prologue 极敏感）。

2) 做最小的“寄存器可用性/活跃性分析”（hetGPU-style 的关键能力）
   - 解析 SASS 指令，构建基本块/控制流（至少能覆盖最常见的 BRA/SSY/BSSY/RET 形态）。
   - 在候选 patch 点附近做 live-in/live-out 近似分析，找出：
     - 真的“dead” GPR（可借用而无需 save）；
     - 可用谓词（或可保存/恢复的谓词位）；
     - 可用 UR 对（尽量用高位 UR 并保存/恢复）。
   - 如果找不到足够 dead 寄存器：只能退化到更小 stub（例如只写一个 marker/func_id），而不是 thread-map。

3) 把 thread-map stub 拆成“最小可用形态”，并严格 gate
   - 从 **lane0-only** 开始：每 warp 1 次写出（1/32），写一个紧凑字段（例如 `u32 {smid_lo8 | (warp<<8) | (lane<<13) | (ctaid_mod<<18)}`）。
   - 再到 **warp0-only**：warp0 的 32 lanes（32 次写出）。
   - 再到 full per-thread：只在确认稳定性后放开。
   - 全程保留 `CTA_CLAMP + max_records=1` 这类“只让 CTA0 写”的安全阀。

4) 设计“可证明不破坏”的 save/restore（如果必须借 live 寄存器）
   - 优先策略：只借用 dead GPR（不需要保存）。
   - 如果必须保存：
     - 保存目标尽量小（只保存被借用的 1~2 个寄存器 + P0）；
     - 保存介质建议先用 global buffer 的 per-thread slot（已经有输出 buffer），但要确保地址计算本身只用到“可借用的最小 scratch”；
     - 并发问题：每 thread 独占 slot（用 `(ctaid_mod<<10)|tid`），不需要原子。

5) 验证策略（务必分级）
   - L0：只打 patch，不写出（NO_STORE），确认 vLLM+flash 不崩；
   - L1：lane0-only + CTA_CLAMP=1 + max_records=1；
   - L2：warp0-only + 同上；
   - L3：逐步放开 CTA_CLAMP/max_records；
   - 全程用 filter 精准命中 flash kernel，避免 patch-all 扩大影响面。

短期结论：

- 现在的路线 A（identify closure + 精准 func_id patch）是“能稳定在真实 vLLM 上跑起来”的主线；
- regcount=255 的 ThreadMap(device) 属于路线 B，需要更系统化的 patch/layout + 活跃性分析（hetGPU 风格），否则很难做到稳定且可解释。

---

## 面向 flashattention（regcount=255）的可行路线：借用“prefix replay 覆盖”的目的寄存器当 scratch（尽量不碰 live）

背景：regcount=255 不能靠“提升 regcount”拿到新 scratch GPR。若仍想在同一 run 内对 flash 做 thread/warp 观测，就必须在“不增加可见寄存器压力”的前提下获得临时寄存器。

### 核心想法（可行性：中等，工程量明显小于“重编译管线”）

我们的 detour trampoline 结构本来就是：

- entry：用 `BRA` 跳转到 code cave
- cave：先跑 stub（我们的采样/marker），再 replay prologue prefix，最后跳回 entry+prefix_len

关键点：**如果我们只使用“prefix 指令写到的目的寄存器（RD）”当作 stub scratch**，那么：

- stub 先 clobber 这些 RD；
- 随后 replay 的 prefix 会把这些 RD 写回为“原 kernel 正常执行时应该得到的值”；
- 等价于我们“借用了未来会被覆盖的寄存器”，无需 save/restore，也无需提升 regcount。

这对 regcount=255 很重要：在入口处很难保证有真正 dead regs，但 prefix 往往会写一批寄存器（LDC/LDCU/S2R 等），可提供 2~8 个 scratch。

### 关键风险与约束

1) replay_n 不能太大：prefix 内如果包含 PC-relative 控制流（如 `BSSY/SSY/BRA/CALL` 及其相对偏移），在 cave 里 replay 会因为 PC 不同而语义改变。
   - 因此需要“自动挑选 replay window”：从 entry 开始扫描，遇到控制流/PC-relative 立即停止。

2) 能借到的 scratch 寄存器数量有限且不稳定（不同 kernel prologue 形态不同）。
   - 所以必须从低密度 stub 开始（marker / lane0-only），逐步扩展。

3) 只能保证“不破坏被 replay 覆盖的寄存器”；对谓词/UR 仍要显式保存/恢复（P2R/R2P，UMOV UR save/restore）。

### 落地计划（Step-by-step）

Step 0（观测与评估）：

- 对 vLLM flash kernel（`flash_fwd_splitkv_kernel*`）dump patched/unpatched cubin；
- 用我们自己的 decoder 在 entry 附近扫描前 N 条指令，统计：
  - 控制流出现的最早位置（可安全 replay 的上限）
  - 前缀里被写的 RD 集合（可借用 scratch 的数量）

验收：能找到至少 3~4 个可借用 GPR（含一对 `.64` 地址寄存器）且 prefix 在控制流前有足够长度（比如 >= 6~12 条）。

Step 1（实现：自动选择“安全 replay_n + 可借用 scratch regs”）：

- 在 detour 生成阶段，为每个目标 section：
  - 扫描 prefix，计算 `safe_replay_n`
  - 提取 `scratch_pool = {RD_i}`（去重、排除 RZ/特殊编码）
  - 生成 stub 时只从 scratch_pool 取寄存器（不再依赖 regcount patch 的高位寄存器布局）

验收：对 flash kernel 能生成 stub（哪怕只 NO_STORE），并且 vLLM 不崩。

Step 2（最小 marker stub，验证稳定性）：

- stub 只做：lane0-only + 写 1 个 u32 marker（例如写 smid_lo8 或常量 tag）到 `sample_buffer[0]`
- 仍然 `CTA_CLAMP + max_records=1`，把干扰压到最低

验收：`/tmp/*.jsonl` 有 `meta + >=1 record`，且 vLLM 端不出现 cublas/illegal access。

Step 3（lane0-only threadmap，逐步放开）：

- 只写 lane0（每 warp 1 条，1/32），slot 选择用 `(ctaid_mod<<10)|tid_x` 或更稀疏映射（先 CTA0）
- 后续再试 warp0-only、再试 full per-thread（每一步都要先在真实 vLLM 上跑通）

验收：记录密度按预期变化，且稳定性可控。

Step 4（如果 Step 1 找不到足够 scratch）：

- 退化路径：
  - L2：换 patch 点（epilogue/return 前的稳定点）再做同样的“借用 RD”策略
  - L3：才考虑 save/restore spill 或重编译管线（hetGPU 风格）

---

## 规划：SM120 上把 flashattention 也做成“可控密度的 device-side 观测”（不在 entry，先 exit/epilogue）

现状说明（为什么你“没看到 flash 的 thread/warp mapping”）：

- 目前我们对 vLLM 的 flashattention（`flash_fwd_splitkv_kernel*`）主要跑通的是 **identify-only/closure**：
  - 用最小 no-regcount stub 写出 `func_id/tag`，用于 strip 场景下的闭环（name → func_id → 精准 patch）。
- 但 flashattention 大多是 `regcount=255`，我们还没有把 **thread_map(device) / warp_map** 的 stub 安全地放进 flash kernel：
  - entry/prologue 处寄存器/谓词状态高度敏感；
  - regcount=255 不能靠“提升 regcount”获得新 scratch GPR；
  - 目前还缺一个“更安全 patch 点 + 最小 CFG/liveness + save/restore/spill”的闭环实现。

目标：

- 在真实 vLLM（SM120）负载下，对 flashattention kernel 先实现 **lane0-only / warp0-only** 的 device-side 写出；
- 保持 `CTA_CLAMP + max_records=1` 作为安全阀（先只让 CTA0 写，降低干扰与崩溃风险）；
- 验收标准：vLLM 不崩 + dump 至少 `meta + >=1 record`，再逐步放开密度。

### Phase 1：先选更安全 patch 点（EXIT/RET 附近），做最小 marker（最快闭环）

动机：

- “patch 在 entry”对 reg255 风险最高；而在 **epilogue/return 前**，寄存器 live 集合通常小很多，插桩更稳。

步骤：

1) 实现 SM120 的 `EXIT/RET` 指令识别（基于 op16/w0 模式），在 `.text.*` 中定位所有 exit 点。
2) 在 exit 点做 detour（BRA → cave），cave 里跑最小 stub（lane0-only 写 1 个 u32 marker），再执行原 EXIT/RET。

验收：

- patched cubin 用 `cuobjdump --dump-sass --function flash...` 能看到 exit 附近出现我们的 `BRA` 与 cave stub；
- vLLM+flash 运行不崩，dump 有至少 1 条 record（哪怕只有 marker）。

### Phase 2：最小 CFG + 最小 liveness（hetGPU 风格子集），自动找可借用 dead regs

动机：

- exit 点不是唯一“安全点”；且有些 kernel 的 epilogue 可能较复杂。
- 需要一个保守 liveness 来告诉我们：哪些 GPR/P0/UR 在某个 patch 点确实 dead（可直接借用）。

步骤：

1) 最小 CFG：识别 BRA（含 pred）并建基本块（fallthrough + branch）。
2) 最小 defs/uses：只覆盖我们 stub 会用到的指令子集；未知指令保守视为“读写很多寄存器”（宁可找不到 dead regs）。
3) 反向数据流求 live-in/live-out，导出 patch 点附近的 dead regs 池。

验收：

- 对 flash kernel 的 exit block 能导出至少 2~4 个可借用 GPR（最好包含一对 `.64` 地址寄存器）；并输出 debug（dead regs 数量、候选 patch 点）。

### Phase 3：实现“可控密度”的 flash thread/warp mapping（逐步放开）

顺序（每一步都要回归 vLLM+flash，且默认 target-only 精准 patch）：

- L0：marker-only（1 次写出）
- L1：lane0-only（每 warp 1 条，1/32）
- L2：warp0-only（warp0 32 条）
- L3：更高密度（谨慎推进；必要时先换更紧凑格式）

安全阀：

- `CTA_CLAMP=1 + max_records=1` 先固定（只让 CTA0 写），稳定后再增大 `max_records`。

### Phase 4：如果 liveness 找不到足够 dead regs（退化路径）

优先退化而不是“硬插”导致 vLLM 崩溃：

- 退化为更短 stub（marker / lane0-only）
- 换 patch 点（更靠近 return 的稳定点）
- 仍不行才考虑最小 save/restore spill（每 thread 独占 slot，避免原子/共享竞争）

---

## 20260109 进度追加（flash reg255：EXIT detour + thread_map(device) lane0-only 跑通）

### 这次实际做到的（已验证）

1) **flashattention（regcount=255）现在可以跑 thread_map(device)（通过 EXIT detour）**

- detour 点从 entry 切到 **tail EXIT**（避免入口敏感），并用 no-regcount exit stub 做 device 写出。
- vLLM in-process（`VLLM_ENABLE_V1_MULTIPROCESSING=0`）下，`flash_fwd_splitkv_kernel*` 通过 `func_id` 精准过滤后，dump 能看到 `meta + 多条 bpftime_sass_thread`。

2) **修正了 SM120 `S2R` 的 w1 编码常量（之前不准）**

- `SM120_SR_CTAID_X`、`SM120_SR_VIRTUALSMID` 的 w1 需要以 `readelf -x .text.*` / `nvdisasm` 的实际输出为准。
- 之前常量错误会导致 `ctaid_x`/`smid` 读出来不可信；现已修正为与 probe cubin 一致。

3) **为什么 dump 里常看到 `smid_lo8=0`：不是 bug，通常是我们把写出密度压到“只跑 CTA0”导致**

- 我们默认为了安全会开 `CTA_CLAMP=1 + max_ctas=1`（只让 `ctaid.x==0` 的 CTA 写）。
- vLLM 里如果只跑一个 CTA，这个 CTA 很容易稳定落在同一个 SM（比如 SM0），因此 `smid_lo8` 经常是 0。
- 想看到多个 SM 的分布，需要逐步放开 `max_ctas`（同时保持 lane0-only / max_records 上限）。

### 复现命令（真实 vLLM + Qwen3-0.6B）

先确保用 in-process（否则多进程 child 不一定继承 detour 状态）：

- `VLLM_ENABLE_V1_MULTIPROCESSING=0`

示例（lane0-only，CTA0-only，目标 func_id=412）：

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
BPFTIME_CUDA_SASS_DETOUR=1 \
BPFTIME_CUDA_SASS_SAMPLE=1 \
BPFTIME_CUDA_SASS_SAMPLE_MODE=thread \
BPFTIME_CUDA_SASS_DETOUR_FILTER=flash_fwd_splitkv_kernel \
BPFTIME_CUDA_SASS_DETOUR_FILTER_FUNC_IDS=412 \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE=1 \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_LANE0_ONLY=1 \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_STRIDE4=1 \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_CTA_CLAMP=1 \
BPFTIME_CUDA_SASS_SAMPLE_MAX_RECORDS=1 \
BPFTIME_CUDA_SASS_SAMPLE_SYNC_AFTER_LAUNCH=1 \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_ON_SYNC=1 \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_PATH=/tmp/bpftime-vllm-flash-thread-exit.jsonl \
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
.venv-vllm/bin/python3 benchmark/gpu/vllm_observability/vllm_inprocess_generate.py \
  --model Qwen/Qwen3-0.6B --max-tokens 4
```

预期：`/tmp/bpftime-vllm-flash-thread-exit.jsonl` 至少包含：

- `bpftime_sass_sample_meta`
- `bpftime_sass_thread`（slot=0,32,64,... 对应 lane0-only 的 warp0/warp1/...）

### 下一步（对应你要的 1/2/3：更安全 patch 点 + CFG/liveness + spill）

- **1 更安全 patch 点**：当前已把 reg255 的 thread_map(device) 默认切到 tail EXIT；下一步扩展到 “RET / 其它稳定同步点附近”的候选点，并做更强 verify（dump 出被替换指令位置）。
- **2 最小 CFG+liveness**：先做“detour 点之后的寄存器使用集”分析（不需要全语义解码），输出 dead GPR 候选；再逐步扩展到基本块级 CFG。
- **3 save/restore/spill**：
  - 已补：stub 的 predicate 保存/恢复支持可配置 mask（`BPFTIME_CUDA_SASS_DETOUR_PR_SAVE_MASK`，默认 P0，调试可设 `0xff`）。
  - 待补：当找不到足够 dead regs 时，按 L0→L3 退化链引入 per-thread spill（每 thread 独占 slot），先支持 lane0-only/warp0-only 的低密度路径。

---

## 20260109 进度追加（L3 最小：reg255 spill 缓冲 + 放开到全 thread 写出（受 CTA_CLAMP 控制））

### 新增/变更点

- 新增 reg255 spill 配置（仅用于 thread_map(device) 的 reg255 EXIT stub）：
  - `BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_REG255_SPILL=1`
  - `BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_REG255_SPILL_BYTES=256`（默认 256，对齐到 16）
  - 默认安全阀：只有在 **tid0-only + CTA0-only** 的情况下才会真的执行 spill save/restore；需要强行打开可用 `BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_REG255_SPILL_UNSAFE=1`（仅用于调试）。

### 20260110 追加：真正的 L3（per-thread 独占 slot spill）

为了解决“共享 spill header 在多线程下必然冲突/覆盖”的问题，新增 per-thread spill：

- `BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_REG255_SPILL_PER_THREAD=1`
- `BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_REG255_SPILL_STRIDE_BYTES=16`（每 thread 独占 16B slot；host 会按 `max_records*1024*stride` 分配）

验证（真实 vLLM + flash reg255，`MAX_RECORDS=4`，全 thread 写出）：

- `/tmp/bpftime-vllm-flash-thread-ctas4-spillpt.jsonl`：`thread_records=512`，`per_cta={0:128,1:128,2:128,3:128}`
- meta 中 `device_bytes` 会明显增大（因为多出来 per-thread spill 区），例如本次为 `81920`（数据区 `4*1024*4=16384` + spill 区 `4*1024*16=65536`）
- host 侧在分配 sampler buffer 时，会在 data 区后面额外加上 spill 区（所以 meta 里的 `device_bytes` 会比原来的 `max_records*1024*4` 多出一段）。
- stub 侧的 spill 目前是“最小可用”的 save/restore：
  - 使用 `[RZ.64 + UR + imm]` 的绝对寻址，把少量 GPR + predicate snapshot 写到 spill 区；
  - 结束前再读回来并恢复（为后续把 patch 点从 EXIT 推到其它点做准备）。

### 回归验证（真实 vLLM + flashattention reg255）

共同前提：
- `VLLM_ENABLE_V1_MULTIPROCESSING=0`（保证 detour 生效在同进程）
- `BPFTIME_CUDA_SASS_DETOUR_FILTER_FUNC_IDS=412`（精准只 patch flash 的目标 func_id）

#### 1) CTA0 内全 thread 写出（关闭 lane0/warp0 gate）

- 参数（关键差异）：`LANE0_ONLY=0`、`WARP0_ONLY=0`、`CTA_CLAMP=1`、`MAX_RECORDS=1`
- 结果：
  - `/tmp/bpftime-vllm-flash-thread-fullcta.jsonl`：`thread_records=128`
  - 解释：flash kernel 的 `blockDim.x` 在该场景下就是 128，所以“全 thread”= 128 条。

#### 2) 放开到多个 CTA（仍 clamp，最多写 CTA[0..3]）

- 参数（关键差异）：`MAX_RECORDS=4`（且 `CTA_CLAMP=1`）
- 结果：
  - `/tmp/bpftime-vllm-flash-thread-ctas4.jsonl`：`thread_records=512`，`per_cta={0:128,1:128,2:128,3:128}`

### 距离“真实 vLLM/flash 的稳定全量观测”还有多远？

现在已经可以做到：
- 在 flashattention(reg255) 上，**按 CTA_CLAMP 控制影响面**，做到“每个被允许的 CTA 内，所有实际存在的 threads 都写出记录”（并且 vLLM cold-run 可稳定跑通）。

仍需要继续收敛的点（按重要程度）：
- **覆盖面/规模**：要“全量”覆盖 flash 的所有 CTA（甚至所有 launch），buffer 会随 grid 线性膨胀；需要进一步的采样/限流机制（例如只采样前 N 个 CTA、或按 stream/kernel 次数采样）。
- **扰动/性能**：全 thread 写出会给 L2/DRAM 带来明显压力；需要更紧凑的写出格式（比如 u8/u16 压缩、或者只写一次/分层采样）。
- **更通用 patch 点 + 更强 save/restore/spill**：当前主路径仍是 EXIT detour；如果后续要在更早的点插桩（例如同步点/epilogue 前）并且线程继续执行，就必须扩展 spill 的“每 thread 独占 slot”与可配置保存集（不仅是少量 GPR/P0）。

---

## 20260110 追加：reg255 “pre-exit NOP detour”（prefix-first：replay NOP → stub → 原始 EXIT）

### 动机

reg255 的 thread_map(device) 目前默认 detour 在 tail `EXIT`，这样 stub 可以随意用低寄存器做 scratch（因为 `EXIT` 后线程立刻结束）。

但如果我们想把 patch 点 **从 `EXIT` 往前推一小步**（更接近一般的 “epilogue/同步点附近可插桩” 的形态），同时又要保证 trampoline 的顺序是：

1) **先 replay 被覆盖指令**
2) **再跑 stub**
3) **再回到原始控制流继续执行**

那么 `EXIT` 自己就不适合放在 prefix-first 里（replay `EXIT` 会直接结束线程，stub 永远跑不到）。

### 实现

新增一个可选模式：在找到 **unpredicated tail `EXIT`** 后，向前扫描若干条指令，找到 **离 `EXIT` 最近的 `NOP`** 并把 detour 点放在这个 `NOP` 上：

- 入口 patch：`NOP → BRA tramp`
- trampoline 顺序：`replay NOP → stub → BRA back`（back 到 `NOP` 的下一条，也就是原始 `EXIT`）

限制：
- 只对 **unpredicated `EXIT`** 启用（predicated `EXIT` 目前不做 pre-exit，避免“未退出线程也执行 stub”带来副作用）。

### 新增环境变量

- `BPFTIME_CUDA_SASS_DETOUR_REG255_PRE_EXIT=1`：启用 pre-exit NOP detour（默认关闭）。
- `BPFTIME_CUDA_SASS_DETOUR_REG255_PRE_EXIT_SCAN_INSTS=8`：从 `EXIT` 往前最多扫描多少条指令找 `NOP`（默认 8）。

### 回归验证（真实 vLLM + flash reg255）

在已有的 per-thread spill + 多 CTA clamp 的基础上开启 pre-exit：

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
BPFTIME_CUDA_SASS_DETOUR=1 \
BPFTIME_CUDA_SASS_SAMPLE=1 \
BPFTIME_CUDA_SASS_SAMPLE_MODE=thread \
BPFTIME_CUDA_SASS_DETOUR_FILTER=flash_fwd_splitkv_kernel \
BPFTIME_CUDA_SASS_DETOUR_FILTER_FUNC_IDS=412 \
BPFTIME_CUDA_SASS_DETOUR_REG255_PRE_EXIT=1 \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE=1 \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_LANE0_ONLY=0 \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_WARP0_ONLY=0 \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_STRIDE4=1 \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_CTA_CLAMP=1 \
BPFTIME_CUDA_SASS_SAMPLE_MAX_RECORDS=4 \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_REG255_SPILL=1 \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_REG255_SPILL_PER_THREAD=1 \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_REG255_SPILL_STRIDE_BYTES=16 \
BPFTIME_CUDA_SASS_SAMPLE_SYNC_AFTER_LAUNCH=1 \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_ON_SYNC=1 \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_PATH=/tmp/bpftime-vllm-flash-preexit-ctas4-spillpt.jsonl \
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
.venv-vllm/bin/python3 benchmark/gpu/vllm_observability/vllm_inprocess_generate.py \
  --model Qwen/Qwen3-0.6B --max-tokens 1
```

预期/结果：`/tmp/bpftime-vllm-flash-preexit-ctas4-spillpt.jsonl` dump `512 records`，并且 vLLM cold run 可稳定完成。

---

## 观测目标补全：flash 内部阶段/路径信息/硬件计数器（规划）

目前我们在 flashattention(reg255) 上主要是在 `EXIT` / `EXIT` 前 `NOP` 这种“尾部安全点”做 thread_map(device) 写出；它适合验证“能命中 + 不崩 + 有数据”，但不足以回答：

- flash 内部各阶段（QKV 读/softmax/写回）到底谁慢、慢在哪条路径
- warp 在哪些阶段 stall/分歧严重
- 某次 TTFT 抖动到底是 flash 内部计算慢、还是 memory/同步/别的 kernel 慢

下面是把这些补齐的可落地路线（按工程难度从低到高）。

### A. “阶段拆分”（内部阶段 begin/end）怎么做

**A1. 先做 kernel 级（不需要 SASS）：CUDA events 计时**
- 目标：拿到 “每次 flash kernel 的 GPU duration” 做 request 级对齐。
- 方法：在 `cuLaunchKernel` hook 中，对目标 kernel（name/filter 命中）自动插入：
  - `cuEventRecord(start, stream)` → 原始 launch → `cuEventRecord(end, stream)`
  - 在合适的同步点/采样点用 `cuEventElapsedTime` 读出 duration（可采样/限流）
- 优点：实现简单，风险小；能立刻把“launch 时间点”升级到“GPU 时间段”。
- 缺点：只能得到 kernel 总时长，不能拆到内部阶段。

**A2. 再做 kernel 内部分阶段（需要 SASS 多点插桩 + liveness）**
- 目标：在 flash 内部若干关键位置插 `marker`（可选带时间戳），形成阶段时间线。
- 方法（最小可行形态）：
  - 选定 3~5 个 patch 点（例如：主要 loop 前、softmax 前后、writeback 前后）
  - 每个点插一个极短 stub：写出 `{tag, pc, ctaid, tid/warp/lane, ts(optional)}` 到 buffer
  - host 侧按 `{tag, pc}` 做阶段归因（pc 区间→阶段名）
- 时间戳来源：
  - 若能安全插入，读 `globaltimer/clock64` 并写出（精度高，但需要更多 scratch）
  - 否则先只写 `{pc}`，阶段边界靠“事件序列”近似（精度较差）
- 关键前提：需要 **CFG+liveness + save/restore/spill**，否则 reg255 在非尾部点很容易破坏 live 寄存器/谓词导致崩溃。

### B. “中间路径信息”（分支/循环行为）怎么做

目标是回答“慢是不是因为走了某条罕见分支/循环迭代次数变多/某些 warp 分歧严重”。

**B1. 分支计数（需要 SASS，多点、极短）**
- 在几个关键分支点插入：`if (lane0) atomicAdd(counter[branch_id], 1)` 或者写出稀疏事件 `{branch_id, taken}`。
- 为了不引入原子热点，建议：
  - 只采 lane0/warp0
  - 或者每 CTA 只让一个 warp 写
  - 或者只对前 N 个 CTA 生效（CTA_CLAMP）

**B2. PC 采样/指令级热点（更适合 CUPTI，而不是纯 SASS patch）**
- CUPTI PC Sampling 可以直接告诉你热点 PC、stall reason 的分布；它比“手工在每个分支点插桩”更系统化，但引入 CUPTI 依赖。

### C. “warp stall / pipe 利用率 / 硬件计数器”怎么做

这类指标本质上来自硬件 PM 计数器；纯二进制 patch 很难稳定、通用地读到完整集合。

**C1. CUPTI / Nsight Compute 指标采集（推荐）**
- 目标：拿到 `sm__*`, `smsp__*`, `lts__*`, `dram__*` 等指标（warp stall、pipe 利用率、cache 命中、带宽等）。
- 方法：引入 CUPTI metrics/activity（可采样、按 kernel name 过滤），把结果写到同一份 JSONL 里。
- 价值：直接补齐 Nsight 的那类指标，同时仍保留 bpftime 的“按 request/marker 对齐”的能力。

**C2. Nsight 做基线，bpftime 做线上轻量化**
- 短期可用 Nsight 先离线定位“哪些 PC/哪些阶段最慢”，再回到 bpftime 只对这些点做 SASS marker（减少插桩点数量和风险）。

### D. 里程碑（建议）

- D0（已有）：flash reg255 尾部安全点 thread_map(device) 写出稳定，且可 CTA_CLAMP 放开到多个 CTA。
- D1（建议下一步）：对 flash kernel 先做 CUDA events 计时（kernel 总时长），并能和 bpftime trace/request marker 对齐。
- D2：做 3~5 个内部阶段 marker（pc-based），先 lane0/warp0 + CTA_CLAMP；逐步扩展点数/密度。
- D3：集成 CUPTI counters/PC sampling（补齐 warp stall/pipe 利用率），并把输出与我们的 tag/pc 合并。

---

## 20260110 追加：KV cache 路线验证（trace → 选目标 → 最小插桩）

目标：先把 vLLM 的 KV cache 更新相关 kernel 也纳入“可观测”，作为 flashattention 之外的第二条闭环样例。

### Step 1：用 bpftime trace 找 KV 相关 kernel 名

对 `vllm_inprocess_generate.py` 开启 `BPFTIME_CUDA_TRACE_PATH` 后，统计 top kernels，能看到 KV cache 更新相关的 kernel（示例为 Qwen3-0.6B）：

- `vllm::reshape_and_cache_flash_kernel<...>`（mangled 名里包含 `reshape_and_cache_flash_kernel`，launch 次数约 1988）

这类 kernel 对应“把本次 step 的 K/V 写入 paged KV cache”的更新路径，是 KV cache 观测的关键入口之一。

### Step 2：对目标 KV kernel 做最小 SASS sampling（先走稳定的 host-expanded）

先用最稳的 `thread` mode 的 host-expanded 形态（device 不直接逐 thread 写，而是每 CTA 写 32 个 warp slot，host 展开成 per-thread 记录）：

- `BPFTIME_CUDA_SASS_DETOUR_FILTER=reshape_and_cache_flash_kernel`
- `BPFTIME_CUDA_SASS_SAMPLE_MODE=thread`
- `BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE=0`
- `BPFTIME_CUDA_SASS_SAMPLE_MAX_RECORDS=1`

回归结果（真实 vLLM in-process）：

- `/tmp/bpftime-vllm-kv-threadmap-warp.jsonl`：
  - `bpftime_sass_sample_meta.thread_device=false`
  - `bpftime_sass_thread` 记录数为 `1024`（32 warps × 32 lanes 的 host expand）

### 当前限制 / 下一步

- host-expanded 的 `thread` 形态默认按 “32 warps × 32 lanes = 1024 threads/CTA” 展开；
  如果目标 kernel 的 `blockDim.x` 不是 1024，展开后的“tid 语义”会失真（但仍可用于低成本验证“命中+稳定+有数据”）。
- 下一步要做更准确的 KV kernel thread/warp 观测：
  - 换成 `BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE=1`（device-per-thread），并启用 gate + CTA_CLAMP 逐步放开；
  - 或者在 host 侧把 launch 里的 `blockDim`（来自 trace）写入 meta，用它来做更准确的 host expansion（更小改动、风险更低）。

---

## 20260110 追加：非侵入式 KV kernel GPU duration + args fingerprint（bpftime-only，无需改 vLLM）

目标：在“不改 vLLM”的前提下，把 KV cache 相关 kernel 的观测从 “launch 点”升级为：

- **GPU duration（cuEvent 计时）**：回答“KV update kernel 真实 GPU 上跑多久”
- **args/指针 fingerprint（best-effort）**：把同名 kernel 的不同调用形态分桶（用于定位“为什么同名 kernel 有时快有时慢”）

### 1) GPU duration（`kernel_timing` 记录）

实现方式：
- 在 `cuLaunchKernel` hook 中（仅对 filter 命中的 kernel，且不在 CUDA graph capture 中）：
  - launch 前 `cuEventRecord(start, stream)`
  - launch 后 `cuEventRecord(end, stream)`
- 在 `cuStreamSynchronize` / `cuCtxSynchronize` 时，批量 `cuEventElapsedTime` 并写 trace：
  - `{"type":"kernel_timing", "launch_seq":..., "gpu_ms":..., "flushed_by_api":..., "flushed_by_seq":...}`

验证（真实 vLLM in-process）：
- `BPFTIME_CUDA_KERNEL_TIMING_FILTER=reshape_and_cache_flash_kernel`
- `BPFTIME_CUDA_KERNEL_TIMING_SAMPLE_EVERY=10`（采样，避免对所有 launch 计时）
- trace summary 能看到 `Timed kernels` 里出现 `reshape_and_cache_flash_kernel` 的 count/avg_ms/total_ms

### 2) args fingerprint（`launch.args_fingerprint` 字段）

实现方式：
- 对 `cuLaunchKernel(kernelParams=...)`（不是 `extra` 形态）：
  - 用 `process_vm_readv` 做 **安全读取**（避免越界/不可读导致 SIGSEGV）
  - 读取最多 `BPFTIME_CUDA_TRACE_ARGS_MAX` 个 argument 的 u64 值（best-effort）
  - 计算 `args_fingerprint=fnv1a64(words)` 写入 `launch` 记录
- 为了避免对所有 kernel 都读参数，默认要求提供 filter：
  - `BPFTIME_CUDA_TRACE_ARGS_FILTER=reshape_and_cache_flash_kernel`

验证（真实 vLLM in-process）：
- trace 中 `launch` 记录里能看到 `args_fingerprint`（同一个 kernel 名可能出现不同 fingerprint）。

### 注意事项 / 局限

- duration 依赖 “有同步点” 才能 flush（当前实现只在 `cuStreamSynchronize` / `cuCtxSynchronize` flush）。
- args fingerprint 是 best-effort：没有类型信息，只取若干 u64 word；用于分桶/对比，不保证语义可解释。

---

## TODO（下一步三项，仍保持“完全不改 vLLM”的非侵入式路线）

下面三项对应我们之前讨论的剩余能力：并发/串行归因、KV kernel 的 device-per-thread 精确 thread-map、pc-marker/hotspot。

### 1) 并发/串行关系（“哪个 kernel 被哪个 sync 卡住”）怎么做

**目标**
- 把 “sync 花了多久” 归因到 “它在等哪些 kernel”（按 stream/ctx）。
- 输出适合回归：每个 `sync` 事件对应的“等待集合 top-k（按累计 gpu_ms）”。

**现状可用数据**
- trace 已有：
  - `launch`（包含 `seq/ts_ns/stream/name/grid/block`）
  - `sync`（包含 `api/obj=stream/duration_ns`）
  - `kernel_timing`（包含 `launch_seq/name/stream/gpu_ms`，在 sync 时 flush）

**落地路线（推荐先做离线分析工具，避免 runtime 额外开销）**
- 新增一个分析脚本（例如 `benchmark/gpu/vllm_observability/bpftime_trace_analyze.py`）：
  1) 建立 `launch_seq -> {name, stream, ts_ns}` 索引
  2) 建立 `stream -> launches_in_order`（按 `seq`）
  3) 读取 `kernel_timing`，把 `launch_seq -> gpu_ms` 填回
  4) 对每条 `sync`：
     - `cuStreamSynchronize(stream)`：取该 stream 上 “上一次同 stream sync 之后” 的 launch 集合
     - `cuCtxSynchronize()`：取 “上一次 ctx sync 之后” 的所有 stream 的 launch 集合
     - 统计 top-k：`sum(gpu_ms)`、`count`，并输出 `sync_wait_top_kernels`
  5) （可选）把 `stream_wait_event`、`event_record` 也纳入依赖推断，增强“跨 stream 等待”的解释力。

**验收/验证**
- 用 `vllm_inprocess_generate.py` 跑一次 trace + timing（已有命令）。
- 运行分析脚本能输出类似：
  - `cuStreamSynchronize(stream=...) waited_cpu_ms=... top_kernels=[(reshape_and_cache..., total_gpu_ms=..., count=...), ...]`

### 2) KV kernel（reshape_and_cache_flash_kernel）推进到 device-per-thread 精确写出

**目标**
- 对 KV update kernel 做到和 flash 类似的“device 侧每 thread 写出”，使 thread/warp/lane 分布在真实 `blockDim.x` 下完全准确。

**现状**
- 已验证 host-expanded（`thread_device=false`）路径能稳定命中并产出记录；
  但它默认按 1024 threads/CTA 展开，遇到 `blockDim.x != 1024` 时 tid 语义会失真。

**落地路线**
- Step 0：用 trace 确认该 kernel 的典型 `blockDim.x` / `gridDim.x`（已可从 JSONL 统计）。
- Step 1（最保守 bring-up）：
  - `BPFTIME_CUDA_SASS_SAMPLE_MODE=thread`
  - `BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE=1`
  - `BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_STRIDE4=1`
  - `BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_CTA_CLAMP=1`
  - `BPFTIME_CUDA_SASS_SAMPLE_MAX_RECORDS=1`
  - gate：`BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_LANE0_ONLY=1`
  - 预期：dump 记录数 ≈ `warp_count = ceil(blockDim.x/32)`（因为只有 lane0 写）
- Step 2（逐步放开密度）：
  - 先 `WARP0_ONLY=1, LANE0_ONLY=0`（每 CTA 32 threads）
  - 再 `LANE0_ONLY=0, WARP0_ONLY=0`（每 CTA 全 threads）
  - 仍保持 `CTA_CLAMP=1`，`MAX_RECORDS=1 → 4` 逐步放开
  - 预期：dump 记录数 ≈ `max_records * blockDim.x`（线程不存在的 slot 仍为 `0xffffffff`，host 不会输出）

**失败时的退化/修复方向**
- 若出现 illegal access / 内存错误：
  - 先回退密度（lane0-only / warp0-only）
  - 再只针对该 kernel 做 targeted detour（必要时用 func_id filter）
  - 再切换写出寻址形态（desc vs UR base：`BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_USE_UR`）

**验收/验证**
- 至少跑通一个 checkpoint：`MAX_RECORDS=1 + lane0-only` 不崩且有记录；
  然后跑 `MAX_RECORDS=4 + 全 threads` 能稳定输出（类似 flash 的 `per_cta` 统计）。

### 3) pc-marker / hotspot（非侵入，但需要新增 sampler mode）怎么做

**目标**
- 在目标 kernel 内部插入少量 marker，用于回答：
  - “kernel 内部哪些位置/阶段被频繁执行”
  - “是否存在某些路径/循环导致异常多的 marker 命中”
- 注意：这不是 Nsight 的硬件 stall reason；它更像“我们自己定义的阶段/热点事件流”。

**落地路线（分两阶段）**
- Phase 1（先把基础设施跑通，低风险）：
  - 新增一个采样 mode（例如 `Mode::PcMarker`），输出一个 ring buffer：
    - record: `{marker_id, tag/func_id, ctaid_x, tid_x/warp/lane, smid, (optional clock)}`。
  - marker_id 的来源：
    - 每个插桩点在 patch 阶段分配一个常量 `marker_id`（不依赖“读取 PC”）。
  - 默认 gate：lane0-only + CTA_CLAMP（避免输出爆炸）。
  - 先在非 reg255 的 KV kernel 上验证（entry / pre-exit / EXIT 附近）：
    - 目标只是验证 ring buffer 写出 + dump + 不崩。
- Phase 2（真正 hotspot：多点插桩 + 自动选点）：
  - 给定 kernel，自动找到 N 个候选点（无需 vLLM 改动）：
    - 优先同步点附近（例如 `BAR.SYNC`）或 epilogue 附近（live 更小）
    - 如果缺少可自动识别点：允许用户传 offsets（`BPFTIME_CUDA_SASS_MARKER_OFFSETS=0x...`）做 bring-up
  - 每个点插同一个最小 marker stub（只写 marker_id + 少量上下文）。
  - 输出：按 `marker_id` 聚合 hit count，形成“热点分布”。

**与 reg255 flash 的关系**
- flash(reg255) 想在“非尾部点”插 marker，仍然要依赖我们已有的 spill/退化链（L0→L3）；
  所以 pc-marker 建议先在 KV kernel 跑通 Phase 1，再逐步移植到 flash（先尾部点，后中间点）。

## 20260110 进度追加（KV：离线 sync 归因 + device-per-thread bring-up）

本轮按优先级先做两件事：（1）离线并发/串行归因脚本；（2）KV kernel 的 device-per-thread bring-up（lane0-only → warp0-only → full）。

### 1) 离线并发/串行归因脚本：sync waits → kernels

**新增**
- `benchmark/gpu/vllm_observability/bpftime_trace_analyze.py`：读取 bpftime CUDA trace JSONL（容忍尾部截断行），对每条 `cuStreamSynchronize` / `cuCtxSynchronize` 做 best-effort 归因：
  - 取“上一次同 stream/ctx sync 之后”的 launch 集合；
  - 结合 `kernel_timing.gpu_ms` 聚合出 top-k（总 GPU 时间 / 次数）。

**验证（示例）**
- 先采集 KV trace + GPU timing（只对目标 kernel 计时，降低开销）：
  - `BPFTIME_CUDA_KERNEL_TIMING=1`
  - `BPFTIME_CUDA_KERNEL_TIMING_FILTER=reshape_and_cache_flash_kernel`
  - trace：`/tmp/bpftime-vllm-trace-kv-timing.jsonl`
- 运行分析脚本：
  - `python benchmark/gpu/vllm_observability/bpftime_trace_analyze.py --trace /tmp/bpftime-vllm-trace-kv-timing.jsonl --kernel-filter 'reshape_and_cache_flash_kernel' --top 5`
  - 期望：在若干 `cuCtxSynchronize` 的“等待集合”里能看到 `reshape_and_cache_flash_kernel`（`timed_launches>0`）。

### 2) KV kernel：reshape_and_cache_flash_kernel device-per-thread bring-up（CTA0 clamp）

目标：让 `tid_x/warp/lane` 在真实 `blockDim.x` 下完全准确（不走 host 展开假设），并用 gate 收敛写出密度，确保真实 vLLM workload 稳定。

**checkpoint A：lane0-only（最保守，1/32 写出）**
- env：`BPFTIME_CUDA_SASS_SAMPLE_MODE=thread` + `BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE=1` + `..._LANE0_ONLY=1` + `..._CTA_CLAMP=1` + `MAX_RECORDS=1`
- dump：`/tmp/bpftime-vllm-kv-device-lane0.jsonl`
- 结果：dump **16 records**（warp_id 0..15）→ 推断该 kernel 常见 `blockDim.x≈512`（16 warps × 32 lanes）。

**checkpoint B：warp0-only（每 CTA 32 threads）**
- env：`LANE0_ONLY=0` + `WARP0_ONLY=1`
- dump：`/tmp/bpftime-vllm-kv-device-warp0.jsonl`
- 结果：dump **32 records**（tid_x 0..31）。

**checkpoint C：full（每 CTA 全 threads）**
- env：`LANE0_ONLY=0` + `WARP0_ONLY=0`
- dump：`/tmp/bpftime-vllm-kv-device-full.jsonl`
- 结果：dump **512 records**（ctaid_x=0，CTA clamp 生效），说明在该 KV kernel 上 device-per-thread 全开可稳定写出（至少在 CTA0）。

**注意**
- 日志可能出现 `instrumented N .text.* sections (mode=thread)`：因为 substring filter 命中了多个 `.text.*` 实例（模板/多版本）；当前 checkpoint 仍可用，但后续若要更精确可引入 func_id-targeted filter（非本轮阻塞项）。

**下一步（继续收敛/扩大覆盖）**
- 在 KV kernel 上把 `MAX_RECORDS` 从 `1 → 4`，逐步放开更多 CTA（仍保持 clamp + 限流），验证：
  - full 模式期望记录数 ≈ `max_records * blockDim.x`（例如 `4 * 512 = 2048`，取决于实际 grid 是否足够大）。

## 20260110 进度追加（pc-marker / hotspot：Mode::PcMarker + 多点插桩）

本轮完成第 (3) 项：pc-marker / hotspot（非侵入、SM120-only），用于在不改 vLLM 的前提下得到“kernel 内部若干位置的命中频率”。

### 实现点
- 新增采样模式：`BPFTIME_CUDA_SASS_SAMPLE_MODE=marker`（`Sm120SamplingConfig::Mode::PcMarker`）。
- 新增多点插桩：`BPFTIME_CUDA_SASS_MARKER_OFFSETS=0,0x10,...`（相对于 `.text.*` section start 的字节偏移，要求 0x10 对齐），同一 section 内每个 offset 都会被 patch 成 “BRA → trampoline”，trampoline 先 replay 原指令，再写出 marker record，再跳回。
- device 写出格式（ring buffer）：
  - header：`write_idx`（u32）+ padding（总 0x10 bytes）
  - record（24 bytes）：`{seq, marker_off, tag, smid_raw, ctaid_x, tid_x}`
  - gate：默认 lane0-only + CTA clamp（可用 env 放开）。
- 新增分析脚本：`benchmark/gpu/vllm_observability/bpftime_sass_marker_analyze.py`，按 `marker_off` 聚合 hit counts。

### 验证
- 小例子（cubin_call_replay_test）验证 `marker_off` 正确写出：
  - `BPFTIME_CUDA_SASS_DETOUR_FILTER=call_entry`
  - `BPFTIME_CUDA_SASS_MARKER_OFFSETS=0,0x10`
  - dump 中 `bpftime_sass_marker.marker_off` 能看到 `16`（说明多点插桩链路工作）。
- 真实 vLLM（in-process, Qwen/Qwen3-0.6B）验证 KV kernel：
  - `BPFTIME_CUDA_SASS_DETOUR_FILTER=reshape_and_cache_flash_kernel`
  - `BPFTIME_CUDA_SASS_MARKER_OFFSETS=0,0x10`
  - dump 里能同时看到 `marker_off=0` 与 `marker_off=16` 的记录（多点命中）。

### 已知限制/下一步
- 当前 `smid_raw` 基于 `SR_VIRTUALSMID`，在本机观测结果常为 `0`（对热点统计不阻塞，但对 SM 归因价值较低）；后续可补一个 SR 选择/探针来确定正确的“物理 SMID”来源。
- offsets 目前是手工指定（bring-up）。下一步可以在目标 kernel 内做“自动找点”的最小实现（比如扫描 `BAR.SYNC` / epilogue 附近稳定点），把热点带入更一般场景。

## 20260110 进度追加（CTA 扩容验证 + smid_raw=0 的澄清）

### CTA 扩容（ThreadMap device）

目标：把 KV kernel 的 device-per-thread 从 `CTA0` 扩到更多 CTA，并用 gate 做“限流/稳定性优先”的推进。

验证（真实 vLLM, `reshape_and_cache_flash_kernel`）：
- `MAX_RECORDS=64 + CTA_CLAMP=1 + lane0-only`：记录数 = `64 * 16 = 1024`（warp_count=16），可用于低开销的多 CTA 采样。
- `MAX_RECORDS=64 + CTA_CLAMP=1 + warp0-only`：记录数 = `64 * 32 = 2048`。
- `MAX_RECORDS=4 + CTA_CLAMP=1 + full`：记录数 = `4 * 512 = 2048`（blockDim.x≈512）。

分析工具：
- `benchmark/gpu/vllm_observability/bpftime_sass_thread_analyze.py`：统计 `ctaid_x` 分布、推断 `blockDim.x`、输出 `smid_lo8` 直方图。

### smid_raw 常为 0：不是 SR 错误，通常是采样配置导致

现象：当 `MAX_RECORDS=1` 且 `CTA_CLAMP=1` 时，只有 `ctaid_x=0` 会写出；该 CTA 有可能被调度到 SM0，因此 `smid_lo8` 看起来“总是 0”。

澄清：把 `MAX_RECORDS` 提高到 `64` 后，`smid_lo8` 会出现多种值（多个 CTA 被调度到不同 SM），说明 `SR_VIRTUALSMID` 路线在本机是可用的，“全 0”主要是因为只观测到 CTA0。

## 20260111 进度追加（KV 覆盖矩阵：trace→pick→逐个 bring-up 的入口）

目标：把 KV 覆盖从“一个 kernel（`reshape_and_cache_flash_kernel`）”扩到“KV 路径的一组 kernels”（paged attention / swap/copy / prefill/decode 变体），但仍保持 **不改 vLLM** 的非侵入式观测路径。

新增：
- `benchmark/gpu/vllm_observability/vllm_kv_kernel_coverage.py`：运行一组短 workload profile（baseline/batch/prefill-heavy/kv-pressure），每个 profile 产出一个 bpftime trace，并自动运行：
  - `bpftime_trace_summarize.py --json`
  - `bpftime_trace_pick_kernels.py --json`
  最终输出汇总 `kv_coverage_report.json`，方便从 trace 中挑选下一批“要 bring-up 的 KV/attn kernels”。
- `benchmark/gpu/vllm_observability/vllm_kv_kernel_bringup.py`：针对单个 kernel substring 做“一键 bring-up”（timing / marker / thread-map(device)），避免每次手工拼环境变量。
- coverage report 也会汇总 `memcpy/memset`（按 kind 的 count/bytes + top ops），用于判断“swap/copy 是否真的发生”（这类路径常表现为 memcpy 事件而不是 kernel）。

下一步（按优先级）：
1) 先用 coverage report 确认本机能稳定出现哪些 vLLM kernels（尤其关注 paged attention / swap/copy 是否出现）。
2) 对 top-N kernels 逐个跑 bring-up：先开 `kernel_timing`（duration），再开 `marker`（多点命中），最后视需要开 `thread_map(device)`（lane0→warp0→full，配合 `CTA_CLAMP/MAX_RECORDS` 限流）。

当前观察（初步）：
- `baseline_decode/batch_decode/long_prefill_short_decode` 组合下，report 里除了 `reshape_and_cache_flash_kernel`，还能稳定看到一组 `flash_fwd_splitkv_*` / `flash_fwd_splitkv_combine_*`（attention 路径），说明“按 trace 扩展目标 kernel 集合”的闭环有效。
- `paged_attention` / `swap_or_copy` 是否出现仍取决于是否触发对应路径；后续需要用更强的 memory-pressure profile（更小 `kv_cache_memory_bytes` / 更大并发 / 更长上下文）逼出该路径，再把这些 kernel 纳入 bring-up。
- 新增 server 覆盖矩阵：`benchmark/gpu/vllm_observability/vllm_kv_server_coverage.py`，用于在真实 OpenAI server 并发负载下尝试触发 paged/offload/swap（best-effort），并输出 `kv_server_coverage_report.json`（含 trace + picked kernels + memcpy/memset 汇总）。

验证（真实 vLLM OpenAI server）：
- 默认 profiles 可以跑通并产出 trace + report。
- `server_kv_offload_pressure` 能观测到显著的 `DtoHAsync` memcpy bytes（~0.8GB 量级），说明该配置下确实触发了“host<->device 搬运”（可视为 swap/offload/copy 路径的非侵入证据）。
- 本机 CUDA trace 中未必存在名字里包含 `paged_attention` 的 kernel：默认 attention 路径更多体现为 `flash_*`（FLASH_ATTN）或 GEMM/ATen kernels（非 flash backend）。因此覆盖目标以 “trace 里真实出现的 kernels + memcpy/memset” 为准（trace→pick→逐个 bring-up）。

非侵入式 TODO（记录）：
- GPU compute 覆盖扩展：把 bring-up 从 KV/flash 扩到 vLLM 的 top-N 热点（norm/rotary + cutlass/cublasLt + ATen kernels），并为每类目标形成固定观测模板（filter+sampling+marker offsets）。
- 内存/搬运归因增强：新增离线分析，把 `memcpy/memset` 按 stream+sync 窗口与 kernel 序列做更强关联（区分 offload/swap/copy 的时间关系），并结合 `BPFTIME_CUDA_TRACE_ARGS=1` 的 args_fingerprint 做更细的“可能 KV vs 其它”分桶。
- 库/运行时层（后续）：hook `cublasLt` / `nccl` / flashinfer 等 API 输出语义级事件（不改 vLLM）。

## 20260111 进度追加（GPU compute bring-up 扩展 + memcpy 归因增强）

### A) GPU compute bring-up 扩展（top-N hotspots）

目标：把观测从“KV/flash 单点”扩到 vLLM 真实热点（norm/rotary/MLP GEMM/ATen 等）。

实现：
- `BPFTIME_CUDA_KERNEL_TIMING_FILTERS`：支持在一次运行里为多个 kernel substring 同时开启 `kernel_timing`（避免逐个重跑 vLLM）。
- `benchmark/gpu/vllm_observability/bpftime_trace_pick_kernels.py` 支持 `--sort gpu_ms`（基于 `kernel_timing.gpu_ms` 排序，找真正的 GPU 热点）。

回归（真实 vLLM in-process, Qwen/Qwen3-0.6B, `--enforce-eager`）：
- trace: `/tmp/bpftime-vllm-hot-timing4.jsonl`
- timed kernels（按 `gpu_ms_total` 排序，摘录）：
  - `cutlass_80_wmma_tensorop_*gemm*`（MLP/Matmul 相关，GPU 时间最高）
  - `vllm::fused_add_rms_norm_kernel` / `vllm::rms_norm_kernel`
  - `flash_fwd_splitkv_kernel`（attention）
  - `cublasLt::splitKreduce_kernel`（GEMM/reduction）
  - `vllm::rotary_embedding_kernel`
  - `vllm::act_and_mul_kernel`
  - `vllm::reshape_and_cache_flash_kernel`（KV write）

### B) memcpy 搬运归因增强（非侵入 best-effort）

目标：在不改 vLLM 的前提下，把 “大量 DtoH/HtoD 发生在哪些时间窗/附近哪些 kernel” 变得可读。

实现：
- `benchmark/gpu/vllm_observability/bpftime_trace_mem_analyze.py`
  - 支持 `--require-memcpy-kind/--require-memcpy-bytes`：筛出包含特定 kind（如 `DtoHAsync`）且超过阈值的 sync-window。
  - 增加 `class_guess`（kv_write/attention_flash/gemm/norm/rotary/other）作为粗粒度归因标签。

回归（真实 vLLM OpenAI server, `server_kv_offload_pressure`）：
- 全局 `DtoHAsync` 总量 ~0.8GB（说明 host<->device 搬运确实发生）。
- `DtoHAsync` 多为“多次中小 memcpy”分布在多个 `cuStreamSynchronize` 窗口内；需要结合 window 的 top kernels 做 best-effort 归因（仍不等于语义级 KV evict/swap 原因）。

## 20260111 进度追加（thread-map(device) 系统化验证 + 失败分类）

目标：把 `thread-map(device)` 从“KV 单点 bring-up”扩到 vLLM 真实热点 kernels（norm/rotary/cutlass/部分 ATen），并形成稳定的限流模板；对失败目标给出下一步路线。

### 新增/调整（为系统化 sweep 服务）

- 新增稳定性开关：`BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_PREFER_EXIT=1`
  - 用途：当 entry/prologue detour 会 clobber live regs 导致 crash（典型是部分 ATen elementwise kernel）时，best-effort 改成 tail-EXIT detour。
  - 现状：对 `unrolled_elementwise_kernel` 的 entry crash 有帮助，但仍可能因为“选到不执行的 EXIT（多 exit/死代码）”导致 record=0（需要后续增强 exit 选点/多点 patch）。
- 修正 bring-up dump 触发：`BPFTIME_CUDA_SASS_SAMPLE_SYNC_AFTER_LAUNCH_REQUIRE_SAMPLED=1`
  - 用途：避免 `SYNC_AFTER_LAUNCH` 在 vLLM 初始化阶段“误命中/过早 dump”，导致 dump 为空且后续不再 dump。
  - 实现：`cuLaunchKernel` wrapper 的 one-shot sync+dump 优先匹配 `tag_to_kernel`（实际 instrumented kernels），只有当不要求 sampled 时才回退到 substring match。
- `benchmark/gpu/vllm_observability/vllm_threadmap_sweep.py`
  - 支持 `--retry-prefer-exit-on-failure`：当 entry run 非 0 退出时自动 retry 一次 prefer-exit，并把两次 attempt 都写进 report（便于定位“crash vs no-record”）。

### 回归（真实 vLLM in-process, Qwen/Qwen3-0.6B, `--enforce-eager`）

命令（示例）：

```bash
source /tmp/vllm-venv/bin/activate
HF_HOME=/tmp/hf \
python benchmark/gpu/vllm_observability/vllm_threadmap_sweep.py \
  --model Qwen/Qwen3-0.6B \
  --kernels 'reshape_and_cache_flash_kernel,rms_norm_kernel,fused_add_rms_norm_kernel,rotary_embedding_kernel,act_and_mul_kernel,cutlass_80_wmma_tensorop,unrolled_elementwise_kernel' \
  --modes lane0,warp0 \
  --max-records 1 \
  --out-dir /tmp/bpftime-vllm-threadmap-sweep6
```

结果（摘录，见 `/tmp/bpftime-vllm-threadmap-sweep6/threadmap_sweep_report.json`）：
- vLLM kernels（entry detour 可用）：
  - `reshape_and_cache_flash_kernel`：lane0=16、warp0=32
  - `rms_norm_kernel`：lane0=4、warp0=32
  - `fused_add_rms_norm_kernel`：lane0=4、warp0=32
  - `rotary_embedding_kernel`：lane0=16、warp0=32
  - `act_and_mul_kernel`：lane0=32、warp0=32
- cutlass GEMM（entry detour 可用）：
  - `cutlass_80_wmma_tensorop`：lane0=4、warp0=32
- ATen/torch elementwise（目前不稳定/需要额外机制）：
  - `unrolled_elementwise_kernel`：entry detour 会触发 `CUDA illegal instruction`（但 crash 前能写出少量 record）；prefer-exit 能避免 crash，但当前版本仍可能 record=0（选点问题）。

### 下一步（按优先级）

1) 让 `PREFER_EXIT` 对 multi-exit kernel 可靠：对同一 `.text.*` 内的多个 EXIT 做多点 patch（类似 pc-marker 的多点 trampoline），或做最小 CFG/启发式选点（保证选到“真实路径”的 EXIT）。
2) 引入 hetGPU 风格的最小活跃性/死寄存器分析：对 entry/prologue detour 自动找 dead regs / 可用谓词，避免依赖固定 scratch layout（减少 ATen/cutlass 的 crash 风险）。
3) 扩展 sweep 覆盖面：基于 `--sort gpu_ms` 的 top-N 列表，把更多热点（更多 cutlass/cublasLt/ATen）逐个加入，并为每类 kernel 固化“限流模板”（lane0/warp0/max_records/cta_clamp）。

## 20260111 进度追加（unrolled_elementwise_kernel：prefer-exit 稳定化）

现象（之前）：
- entry detour：会触发 `CUDA illegal instruction`
- prefer-exit：能避免 crash，但在部分 run 里出现 `record=0`
- prefer-exit + store：在更一般负载（例如 `--batch-size 2 --max-tokens 16`）里会触发 `CUBLAS_STATUS_INTERNAL_ERROR`
- prefer-exit + `BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_NO_STORE=1`：不会触发 cublas 错误

定位结论：
- “record=0”的根因是：目标 ATen kernel 的退出点常用 `@P0 EXIT`（predicated exit），只 patch unconditional EXIT 会错过执行路径。
- `CUBLAS_STATUS_INTERNAL_ERROR` 的根因是：thread-map(device) 默认走 descriptor store（`STG.E desc[...]`）在这类 ATen kernel 上不稳定；改成 UR-base addressing（`STG.E [R+UR]`）后稳定。

修复（本轮已实现并回归）：
- prefer-exit 的 multi-exit 采集默认包含 `@P0 EXIT`：
  - site 侧用 `@P0 BRA` detour（仅在原本会 exit 的 lane 进入 trampoline）
  - trampoline 内把 replayed exit 强制为 unconditional EXIT（不依赖 P0 在 stub 后的状态）
- prefer-exit 模式下，thread-map(device) 默认改用 UR-base addressing（相当于默认开启 `BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_USE_UR=1`）

回归命令（真实 vLLM in-process, Qwen/Qwen3-0.6B）：

```bash
source /tmp/vllm-venv/bin/activate
HF_HOME=/tmp/hf \
python benchmark/gpu/vllm_observability/vllm_threadmap_sweep.py \
  --model Qwen/Qwen3-0.6B \
  --kernels unrolled_elementwise_kernel \
  --modes lane0 \
  --max-tokens 16 --batch-size 2 --max-records 1 \
  --prefer-exit --no-retry-prefer-exit-on-failure \
  --out-dir /tmp/bpftime-vllm-threadmap-sweep12
```

结果：
- `best_ok=true`，`thread_records=4`（blockDim.x≈128 的 lane0-only）

## 20260111 进度追加（1.2：identify closure 在单次运行内稳定命中 flash + 至少 1 条 record）

目标：
- “任何负载都能稳定命中并采到数据”（优先补齐 1.1/1.2 的稳定性）
- strip/JIT/load 顺序不确定时，不依赖“第二次运行才命中”的跨 run cache

### 问题（之前）

- `cuLaunchKernel` 的 identify-closure 会尝试 on-demand patch-all(fallback)，但对 flashattention 常失败：
  - `module_image` 来自 `cuLibraryGetModule`，格式是 `FATBINC (0x466243b1)`，且经常 `size=0`
  - on-demand patch-all 旧实现默认把 `module_image` 当 raw-ELF 读 `image_size` 来 patch，结果直接拿不到可 patch 的 bytes → `0 records`

### 修复（本轮实现）

1) **FATBINC “size=0” 修复：保留 owned image span（即使 no_patch）**
- 在 `maybe_patch_cuda_image_sass_detour()` 的 FATBINC 路径：当 `code_size==0` 且我们能构造自包含 wrapper 时，
  即使没有 detour，也会把 `fatbinc wrapper + fatbin bytes` 拷贝成一段连续的 owned image 并缓存 size，
  让后续 on-demand patch 能稳定 re-read 该 image（不再出现 `module_image size=0`）。

2) **identify-closure 的 patch-all fallback 默认打开（仅在 closure 模式）**
- `BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE=1` 时，默认允许 patch-all fallback（等价于 `PATCH_ALL_FALLBACK` 的安全子集），
  避免用户额外手动配置导致“闭环必现失败”。
- 可禁用：`BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE_NO_PATCH_ALL_FALLBACK_DEFAULT=1`

3) **PTX/JIT 场景的覆盖增强：closure 下默认允许 JIT-link PTX→sm_120 cubin（仅当 fatbin 无可 patch 的 sm120 ELF 时）**
- 这是为了解决“fatbin 里只有旧 SM SASS + PTX，真实 sm120 代码来自 driver JIT”的场景。
- 可禁用：`BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE_NO_JITLINK_PTX_DEFAULT=1`

4) **on-demand patch-all(fallback) 不再假设 raw-ELF**
- `cuLaunchKernel` 的 on-demand patch-all：对非 ELF `module_image`（fatbin/fatbinc/未知 size）改为复用 `maybe_patch_cuda_image_sass_detour()`，
  让 fatbinc/fatbin/JIT-link 路径都能走同一套 patch 逻辑。

### 验收（真实 vLLM）

环境：vLLM 0.13.0 + Qwen/Qwen3-0.6B（in-process, `--enforce-eager`）

```bash
source /tmp/vllm-venv/bin/activate
HF_HOME=/tmp/hf \
BPFTIME_CUDA_SASS_DETOUR=1 \
BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE=1 \
BPFTIME_CUDA_SASS_DETOUR_FILTER=flash \
BPFTIME_CUDA_SASS_SAMPLE=1 \
BPFTIME_CUDA_SASS_SAMPLE_MODE=warp \
BPFTIME_CUDA_SASS_SAMPLE_MAX_RECORDS=1 \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_ON_SYNC=1 \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_PATH=/tmp/bpftime-vllm-flash-sync.jsonl \
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
python benchmark/gpu/vllm_observability/vllm_inprocess_generate.py \
  --model Qwen/Qwen3-0.6B --max-tokens 1 --enforce-eager
```

期望现象：
- 日志出现：`SASS control: identify dump ... slots=[...]`
- dump 文件 `/tmp/bpftime-vllm-flash-sync.jsonl` 至少 2 行（meta + 1 条 `bpftime_sass_warp`）

## 20260112 进度追加（reg255 flash：predicated EXIT negate 修复 + Target gating 验证）

### 1) predicated EXIT 的关键修复：支持 `@!P? EXIT`

FlashAttention 的 sm_120 SASS 里存在 `@!P0 EXIT`（cuobjdump 可见 `0x894d`）。此前我们只按 `pred<<12|0x094d` 解析 predicated EXIT，会把 `@!P0 EXIT` 误判成 `@P0 EXIT`，一旦允许 predicated EXIT 并在 site 上生成 `@P0 BRA`，就会改变控制流，导致真实 vLLM 中出现随机崩溃（包括 segfault/futex 异常等）。

本轮修复：
- `exit_kind_from_op16_sm120()` 增加 `neg`（bit15 / `0x8000`）解析；
- `encode_bra_sm120_pred()` 支持生成 `@!P? BRA`；
- prefer-exit 的 debug 日志会显示 `rel@P0` / `rel@!P0`，便于核对。

环境变量分级（避免默认过激）：
- `BPFTIME_CUDA_SASS_DETOUR_ALLOW_PREDICATED_EXIT=1`：只允许 `P0`（包含 `@P0` 和 `@!P0`）
- `BPFTIME_CUDA_SASS_DETOUR_ALLOW_PREDICATED_EXIT_ANY=1`：允许 `P0..P6`（真实 vLLM 上仍可能触发 segfault，保持 opt-in）

### 2) reg255 EXIT stub 增加 Target gating（避免 entry/prologue 的 _UNSAFE）

为避免 reg255 kernel 的 entry/prologue 路径做复杂 device-side control read/target gating（那条 `_UNSAFE` 仍会引发稳定性问题），我们把 gating 移到 EXIT stub：
- 只在 `enable==1 && mode==Target(2) && target_func_id==this_func_id` 时执行 thread-map 写出
- 否则跳过写出（减少跨 kernel 覆盖/干扰）

bring-up 脚本侧验证：
- `benchmark/gpu/vllm_observability/vllm_flash_reg255_threadmap_bringup.py` 的 thread-map stages 会设置：
  - `BPFTIME_CUDA_SASS_DETOUR_IDENTIFY_CLOSURE_FORCE_TARGET_FUNC_ID=<target_func_id>`
- dump meta 里可见 `control.mode=2` 且 `control.target_func_id=<...>`（Target 已生效）

示例（warp0-only）：
```bash
BPFTIME_CUDA_TRACE_PATH=/tmp/bpftime-trace-gated2.jsonl \
BPFTIME_CUDA_SASS_DETOUR_ALLOW_PREDICATED_EXIT=1 \
BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_EXIT_SCAN_INSTS=16384 \
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
.venv-vllm/bin/python benchmark/gpu/vllm_observability/vllm_flash_reg255_threadmap_bringup.py \
  --kernel-filter flash --density warp0 --max-tokens 8 --batch-size 1 \
  --out-dir /tmp/bpftime-reg255-gated2
```

### 3) 关键卡点：真实 flash kernel 的“命中与写出”仍未达到高密度

用 `BPFTIME_CUDA_TRACE_PATH` 的真实 trace 统计：
- `_ZN5flash24flash_fwd_splitkv_kernel...` 在该 workload 下频繁被 launch
  - `blockDim.x = 128`
  - `grid.z = 8/16`（不同变体）

但 thread-map(device) dump 目前仍常只有 `slot=0` 单条记录（即便 `warp0-only`），这与 `blockDim.x=128` 的预期不符。

推断：
- 当前仍没有稳定把 thread-map(device) stub 命中到“真实执行的那一个 SM120 flashattention code object/entrypoint”；
- 典型原因是 strip 场景下 `.text.<name>` 不可用、或 func_id→entrypoint/section 映射仍不完备（即使能看到 kernel name 在 trace 里出现）。

下一步（按重要程度）：
1) 做“可命中覆盖面”闭环：让 strip 的 SM120 flash code object 也能通过 func_id/tag 闭环定位并精确 patch；
2) 然后再回到密度：lane0-only → warp0-only → full threads，验证 `bpftime_sass_thread` 记录数随 gate 放开增长到 O(blockDim.x)。
