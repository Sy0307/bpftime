# bpftime CUDA：SASS 级可观测（设计 + vLLM 实测流程）

本文件记录 bpftime 当前的 **SASS（cubin-only）级插桩/可观测**进展、如何用 **真实 vLLM** 验证，以及要做成“可观测 SM/warp/thread”还缺哪些关键能力。

## 对齐 hetGPU 的“最小子集”改造进度（已做/未做）

这里的“hetGPU 最小子集”指：具备足够系统化的 **patch/布局能力**（code cave 扩容）、**寄存器/谓词保护**、以及一套可扩展的 **SASS 指令生成/编码抽象**（至少覆盖 sm_120 观测 stub 所需的 op），并在此基础上实现可控的 mapping stub，并用真实 vLLM 回归稳定性。

### 已完成

- **code cave 扩容（gap-cave）**：当 `.text.*` 尾部 NOP cave 不够时，将该 section 向后扩展到下一个 section offset / `e_shoff` 前的 gap，并更新 `sh_size`，无需搬移其它 section（`attach/nv_attach_impl/sass_detour/sass_detour.cpp:1234`）。
- **code cave 扩容（insert-cave / shift-cave，hetGPU 风格）**：当既没有尾部 NOP cave、也没有 alignment gap 时，在该 `.text.*` section 末尾 **插入一段 NOP gap**，并整体搬移后续 ELF 内容（更新 `e_shoff/e_phoff`、所有 `sh_offset`、以及相关 `PT_LOAD` 段的 `p_filesz/p_memsz`），保证 trampoline/stub 仍能落地（`attach/nv_attach_impl/sass_detour/sass_detour.cpp`）。
- **入口 detour + trampoline 框架（SM120）**：把入口第一条指令改成 `BRA trampoline`；trampoline 内先执行采样 stub，再回放原 prologue prefix，最后 `BRA` 回到“第一条未回放的指令”（`attach/nv_attach_impl/sass_detour/sass_detour.cpp:1260`）。这个顺序对 vLLM/cutlass 的 cubin-only kernel 很关键（避免 stub 破坏 prologue 初始化的 UR/谓词状态）。
- **replay window 的控制流重定位（关键鲁棒性）**：
  - 修复 `replay_n > 1` 时 trampoline 回跳地址错误（现在回到 `entry + replay_n*16`）。
  - 对 replay window 内的 PC-relative 控制流指令（BRA/CALL 等）做 immediate 重定位，避免在 trampoline 中“跳到错误地址”导致 silent wrong / crash（`attach/nv_attach_impl/sass_detour/sass_detour.cpp`）。
  - 增加离线单测：`attach/nv_attach_impl/test/test_sass_detour_replay_reloc.cpp`（不依赖 GPU，直接对 cubin 做 patch 并验证 CALL target）。
- **寄存器布局（最小可用）**：为 stub 分配 scratch GPR（基于 old regcount 向上分配 + headroom），避免落在边界寄存器导致的 SM120 kernel 不稳定（`attach/nv_attach_impl/sass_detour/sass_detour.cpp:727`）。
- **regcount 元数据修正（关键能力）**：对 `.nv.merc.nv.info` 与 `.nv.info` 的 global regcount entry（type=0x2f04,size=8）按 `sh_info(func_id)` 定位并写入新 regcount；没有这步真实 vLLM/cutlass cubin-only kernel 会崩（`attach/nv_attach_impl/sass_detour/sass_detour.cpp:1112`）。
- **nv.info small regcount ID 修正（SM120 实测）**：CUDA 12.x SM120 的 `.nv.info.<kernel>` 中 regcount small entry ID 为 `0x1b`（而不是 `0x19`）；已修正读取/patch 逻辑并保留 `0x19` 作为兼容 fallback（`attach/nv_attach_impl/sass_detour/sass_detour.cpp`）。
- **指令生成/编码抽象（薄层，够用）**：实现了一组 `inst_*`/`encode_*` 生成 stub 所需的 SM120 指令编码（S2R/IMAD/LOP3/SHF/ISETP/UMOV/STG/BRA 等），支撑 `warp_map/thread_map` 等 stub（`attach/nv_attach_impl/sass_detour/sass_detour.cpp:401` 起）。
- **可控 mapping stub + vLLM 验证**
  - **per-warp**：`BPFTIME_CUDA_SASS_SAMPLE_MODE=warp`，device 侧写 `slots[(ctaid*32)+warp_id]=smid_lo8`；`warp_id = (tid.x>>5)`（`attach/nv_attach_impl/sass_detour/sass_detour.cpp:876`）。
  - **thread debug（host 展开）**：`BPFTIME_CUDA_SASS_SAMPLE_MODE=thread`，device 侧仍写 per-warp slot，但 host dump 展开每个 warp slot → 32 lane 形成 per-thread 记录（`attach/nv_attach_impl/sass_detour/sass_detour.cpp:913`、`attach/nv_attach_impl/nv_attach_impl.cpp:889` 一段）。
	  - **thread_map（device 真写出，实验性）**：
	    - `BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE=1` 时，device 侧按 `idx=(ctaid_mod<<10)+tid.x` 写入（不再依赖 OR 路径），并在 `stride4` 下使用对齐的 `STG.E` 写 32-bit（`attach/nv_attach_impl/sass_detour/sass_detour.cpp`）。
	    - 为了 vLLM 稳定性/开销控制，可配合 gate 降低写入密度：
	      - `BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_LANE0_ONLY=1`：只让每个 warp 的 lane0 写（1/32）
	      - `BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_WARP0_ONLY=1`：只让 warp0 写（1/32）
	      - 两者同时开启时等价于 `tid.x==0`（1/1024）
		      - gate 使用 `tid.x` 派生 `lane=(tid&31)` 生成谓词，并对 `P0` 做最小保存/恢复（`P2R/R2P`），避免破坏原 kernel 的谓词状态；同时采用 “predicated BRA skip_store” 来规避 predicated STG 在入口处的不稳定行为：
		        - `LANE0_ONLY`：`lane=(tid&31)` + `ISETP.NE` → `@P0 BRA skip_store`
		        - `WARP0_ONLY`：`ISETP.GE tid.x, 32` → `@P0 BRA skip_store`
		        - 两者都开：`ISETP.NE tid.x, 0` → `@P0 BRA skip_store`
		      - `idx` materialize 后默认插入 `BPFTIME_CUDA_SASS_DETOUR_IDX_WAIT_NOPS=8` 个 NOP：否则会出现“stale idx（常见为 0）→ 大量写到 slot0”的非确定性。
		      - 最小正确性验证：
		        - `BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_DEBUG_STORE_TID=1`：要求 `slot == raw_u32(tid)`
		        - `BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE_DEBUG_STORE_IDX=1`：要求 `slot == raw_u32(idx)`
	      - checkpoint（最小 call_entry）：
	        - `BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE=1 + STRIDE4=1` 时，`call_entry`（block=128, grid=1）无 gate 约 120 条；
	        - 开 `LANE0_ONLY=1` 后约 4 条；开 `WARP0_ONLY=1` 后约 31 条；两者都开约 1 条。
	    - 计算 `idx*stride` 的 `IMAD.WIDE` 在 device 写出路径改为 **in-place**（`rb==rd`），降低 SM120 cubin-only kernel 对 wide-op 边界寄存器的敏感性。
- **UR 寄存器最小保护（save/restore）**：对采样 stub 临时使用的 UR pair 做 `UMOV URsave, URx` 保存，并在返回原 kernel 前恢复（`attach/nv_attach_impl/sass_detour/sass_detour.cpp:737` 起，`inst_umov_ur_ur` + stub 末尾 restore）。
  - **真实 vLLM（Qwen/Qwen3-0.6B）回归**：能启动 openai server、完成请求；并能 dump detoured cubin（无 PTX）以及 JSONL 样本（`benchmark/gpu/vllm_observability/vllm_openai_server_load_test.py:237`）。
  - **raw CUBIN（ELF）输入的 size 推断修正**：`cuModuleLoadData` 传入的 cubin 有时 program header table 在 section header table 之后；现在 size 推断会包含 `e_phoff + e_phnum*e_phentsize`，避免“读不到 phdr 导致 detour/insert-cave 无法工作”（`attach/nv_attach_impl/nv_attach_impl_frida_setup.cpp`）。

### 未完成（与 hetGPU 差距最大的部分）

- **谓词寄存器保护（PR save/restore）**：目前 stub 没有像 hetGPU 那样通用地保存/恢复 `PR`（例如 `P2R/R2P`），也没有“使用前保存、使用后恢复”的统一框架；依赖 stub 尽量不破坏原有谓词状态。
  - 目前只有 `LANE0_ONLY` gate 对 `P0` 做了点状保存/恢复；要进一步系统化，需要把“可用谓词选择 + 保存/恢复 + 约定”做成统一组件。
- **更通用的寄存器保护策略**：
  - 目前主要依赖“提高 regcount + 使用新增 scratch GPR”，但尚未实现更系统化的 caller-save/callee-save 约束、以及对特殊寄存器/UR/谓词的统一保护。
- **指令编码抽象的系统化程度不足**：
  - 当前是“够用的 inst_* 函数集合”，缺少 hetGPU 那种更可组合的 patch pipeline（例如按 block 组装、自动分配寄存器/谓词、对控制字/调度做校验/回退）。
  - 控制字段（control word）目前主要来自观察与经验，缺少统一的“生成策略 + 校验/对比”工具链。
- **更通用的 code cave/布局策略**：
  - 已有 tail-cave + gap-cave，但尚未实现 hetGPU 风格的更系统化布局（例如跨 section 的 code cave 管理、multi-stub 放置、对齐/填充策略、以及更复杂的 ELF 重排/重定位处理）。
- **device-per-thread 写出在 vLLM 上仍需进一步收敛**：
  - 已实现 `thread_map_device + stride4` 的写出通路，但：
    - `LANE0_ONLY` gate + slot 映射在 `call_entry` 最小例已做到可预测/可断言（见上面的 debug_store_t*）。
    - 真实 vLLM 下仍需继续扩大覆盖面做回归（尤其 flash/cublasLt），并把“失败的 kernel 名称/section dump”纳入自动化隔离流程。
- **更强的可控性（采样 gate/限流/按 stream/按 launch 选择）**：
  - 目前主要靠 kernel 名称 filter 与 max_records mask；缺少更细粒度的 gate（采样率、只采样部分 CTA/warp、或者按 tag/launch 次数触发）。

### 已建立的 checkpoint（便于持续迭代）

- **Checkpoint A：只做 detour，不采样**：验证 cubin-only 可 patch 且不影响 vLLM 正常推理。
- **Checkpoint B：CTA/warp/thread mapping**：在单 kernel filter 下开启 sampling，验证：
  - vLLM server 可启动并完成请求（正确性/稳定性）
  - dump 的 cubin `--extract-ptx` 显示无 PTX、`/*0000*/` 入口为 `BRA ...`（detour 生效）
  - JSONL 输出可解析并具备 `ctaid/warp_id/lane_id` 等字段（可观测数据链路通）

## 背景：为什么必须做 SASS

在常见的 `torch`/`vllm` 预编译 wheel 环境里，很多 CUDA kernel 以 **cubin-only fatbin（SASS-only）** 形式发布，fatbin 内 **不带 PTX**。

这会导致：
- 仅靠 bpftime 现有的 “抽 PTX → patch PTX → nvptxcompiler 重新编译” 这条链路无法覆盖这部分 kernel
- 你能 trace 到 launch（host 侧），但无法把 probe/采样真正“插到 kernel 里”

因此需要 **直接对 CUBIN ELF 的 `.text.*`（SASS 机器码）做改写**。

## 当前实现：SASS detour + 最小采样（可跑真实 vLLM）

bpftime 现在实现了一个 **“入口 detour + trampoline”**（SM120）：
- 在 CUBIN ELF 的 `.text.*` section 里找尾部的 NOP padding（code cave）
- 把入口第一条指令改写成 `BRA trampoline`
- trampoline 里可选执行一段“采样 stub”，随后执行“原本的第一条指令”，再 `BRA` 回到入口 `+0x10`

这一步的意义是：在 **cubin-only fatbin** 的真实 workload（vLLM）里验证：
- 二进制确实可 patch，且不破坏程序运行
- 采样 stub 确实在 kernel 内执行（写入 GPU buffer → 同步点 dump 到 JSONL）

实现位置：
- SASS detour patcher：`attach/nv_attach_impl/sass_detour/sass_detour.cpp:1`
- 注入点：`attach/nv_attach_impl/nv_attach_impl_frida_setup.cpp:1`（hook `cuLibraryLoadData`，支持 FATBINC wrapper / FATBIN payload / raw ELF）
- 配套文档：`attach/nv_attach_impl/README.md:1`

当前限制：
- 仅支持 **SM120**（你机器是 sm_120）
- trampoline 代码目前依赖 `.text.*` 尾部 NOP cave（stub 必须足够短）

### 已实现的两个最小采样模式

通过环境变量 `BPFTIME_CUDA_SASS_SAMPLE=1` 打开：

- `BPFTIME_CUDA_SASS_SAMPLE_MODE=smid_bitmap`
  - 每个被插桩 kernel，把执行到的 SMID 标记到 `smid_bitmap[smid]=1`
  - 适合快速验证“cubin-only kernel 内确实执行到了采样 stub”
- `BPFTIME_CUDA_SASS_SAMPLE_MODE=cta` / `records`（语义：CTA→SMID）
  - 每个 CTA 写 `buffer[ctaid.x] = smid_raw`
  - 用于观察 block 被调度到“哪个 SM/哪个虚拟 SM”
- `BPFTIME_CUDA_SASS_SAMPLE_MODE=warp`（语义：CTA×Warp→SM）
  - 写 `slots[(ctaid.x * 32) + warp_id] = smid_lo8`（device 侧只写低 8bit）
  - 用于做 **SM/Warp/Lane/Thread mapping** 的“底座”（warp 内所有 lane 都在同一 SM 上）
- `BPFTIME_CUDA_SASS_SAMPLE_MODE=thread`（语义：Thread mapping，host 展开）
  - device 侧仍然是 per-warp slot 写入（同 `warp`）
  - host dump 时把每个 warp slot 展开成 32 个 lane：`tid_x = warp_id*32 + lane_id`
  - 最终输出 JSONL 里就是 per-thread 记录（便于直接联调/可视化）

## vLLM 实测：用 Qwen/Qwen3-0.6B 验证 cubin-only 可插桩

这里用真实 vLLM OpenAI server（`vllm.entrypoints.openai.api_server`）启动服务，并发请求，再用日志 + cuobjdump 验证：
- detour 是否真的触发
- dump 出来的 CUBIN 是否 **无 PTX**
- SASS 入口是否已经变成 `BRA ...`（detour 生效）
- `thread_map(device)` 的 **LANE0_ONLY gate 是否真的生效**、以及“全 lane 写出”是否会引发稳定性问题

### 1) 环境准备

建议使用 venv（如果你已有可跳过）：

```bash
python3 -m venv /tmp/vllm-venv
source /tmp/vllm-venv/bin/activate
pip install -U pip wheel setuptools
pip install vllm
```

### 2) 跑真实 vLLM + bpftime SASS detour（负载测试）

脚本：`benchmark/gpu/vllm_observability/vllm_openai_server_load_test.py:1`

```bash
source /tmp/vllm-venv/bin/activate

BPFTIME_ALLOW_NO_SHM=1 \
BPFTIME_LOG_OUTPUT=console \
BPFTIME_CUDA_TRACE_PATH=/tmp/bpftime-vllm-qwen3-trace.jsonl \
BPFTIME_CUDA_SASS_DETOUR=1 \
BPFTIME_CUDA_SASS_DETOUR_FILTER=cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_tn_align8 \
	BPFTIME_CUDA_SASS_DETOUR_DUMP_DIR=/tmp/bpftime-sass-qwen3 \
	BPFTIME_CUDA_SASS_SAMPLE=1 \
	BPFTIME_CUDA_SASS_SAMPLE_MODE=smid_bitmap \
	BPFTIME_CUDA_SASS_SAMPLE_DUMP_PATH=/tmp/bpftime-sass-samples-%p.jsonl \
	BPFTIME_CUDA_SASS_SAMPLE_DUMP_ON_SYNC=1 \
	LD_PRELOAD=$PWD/build/runtime/agent/libbpftime-agent.so \
HF_HOME=/tmp/hf-qwen3 \
VLLM_MODEL=Qwen/Qwen3-0.6B \
VLLM_CONCURRENCY=1 \
VLLM_TOTAL_REQUESTS=2 \
VLLM_STREAM=0 \
python benchmark/gpu/vllm_observability/vllm_openai_server_load_test.py
```

预期：
- 输出 JSON 中：
  - `sass_detour_patched: true`（至少能看到 detoured cubin dump）
  - `sass_detour_cubin_verify.no_ptx: true`
  - `sass_detour_cubin_verify.entry_bra: true`（至少有一个函数入口是 `/*0000*/ BRA ...`）

### 3) Strip 场景闭环（flashattention）：name→entrypoint 不可用时，用 func_id(tag) 精准 detour

在真实 vLLM（`vllm/_C.abi3.so`）里，FlashAttention 的 SM120 cubin 经常出现“`.text.*`/`.nv.info.*`/`.symtab` 名称不可用（或匿名化）”的情况：
- CUDA trace 里能看到 kernel 名（例如 `flash_fwd_splitkv_kernel`）
- 但在 CUBIN ELF 里 **无法用 name filter 稳定定位到 `.text.*` section**
- 结果就是 `BPFTIME_CUDA_SASS_DETOUR_FILTER=flash_fwd_splitkv_kernel` 会出现 `no .text.* sections matched filter`

最小闭环做法是：引入 **func_id(tag) 过滤**，绕开名字依赖：

- 新增：`BPFTIME_CUDA_SASS_DETOUR_FILTER_FUNC_IDS=<u32[,u32...]>`（支持 `0x..`/十进制）
  - 语义：额外按 ELF section header 的 `sh_info(func_id)` 命中 `.text.*`（即使 section 名被 strip）

如何拿到 func_id：
- 从同一套 kernels 的“有名字”的 cubin（例如 SM80）里，`cuobjdump --dump-elf` 的 section 表中，
  `.nv.info._ZN5flash24flash_fwd_splitkv_kernel...` 这一行的 `Info` 列就是 `func_id`。
- 例如我们在一次 vLLM dump 中抽到 `flash_fwd_splitkv_kernel` 的 func_id 范围为 `0x85..0xac`（连续区间）。

验证命中（真实 vLLM server + 1 次请求）：

```bash
source /tmp/vllm-venv/bin/activate

BPFTIME_ALLOW_NO_SHM=1 \
BPFTIME_LOG_OUTPUT=console \
BPFTIME_CUDA_TRACE_PATH=/tmp/bpftime-vllm-trace.jsonl \
BPFTIME_CUDA_SASS_DETOUR=1 \
BPFTIME_CUDA_SASS_DETOUR_DEBUG=1 \
BPFTIME_CUDA_SASS_DETOUR_FILTER_FUNC_IDS=0x85,0x86,...,0xac \
BPFTIME_CUDA_SASS_DETOUR_DUMP_DIR=/tmp/bpftime-sass-flash-funcid \
LD_PRELOAD=$PWD/build/runtime/agent/libbpftime-agent.so \
HF_HOME=/tmp/hf-qwen3 \
VLLM_MODEL=Qwen/Qwen3-0.6B \
VLLM_CONCURRENCY=1 \
VLLM_TOTAL_REQUESTS=1 \
VLLM_STREAM=0 \
python benchmark/gpu/vllm_observability/vllm_openai_server_load_test.py
```

预期：
- JSON summary 里 `sass_detour_patched: true`
- `/tmp/bpftime-sass-flash-funcid/` 里出现 `sm120_detoured_*.cubin`

进一步：对 flash kernel 做 SM/Warp mapping（dump 时机对齐到 flash launch）

```bash
BPFTIME_CUDA_SASS_SAMPLE=1 \
BPFTIME_CUDA_SASS_SAMPLE_MODE=warp \
BPFTIME_CUDA_SASS_SAMPLE_MAX_RECORDS=256 \
BPFTIME_CUDA_SASS_SAMPLE_FILTER= \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_PATH=/tmp/bpftime-sass-flash-warp-%p.jsonl \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_ON_FIRST_SAMPLED_LAUNCH=1 \
BPFTIME_CUDA_SASS_SAMPLE_DUMP_ON_SYNC=0 \
BPFTIME_CUDA_SASS_DETOUR_FILTER=flash_fwd_splitkv_kernel \
...
```

说明：
- `BPFTIME_CUDA_SASS_SAMPLE_FILTER=`（显式空字符串）用于“采样 stub 对所有被 detour 的 section 生效”（避免 strip 导致 sample filter 匹配失败）
- `BPFTIME_CUDA_SASS_DETOUR_FILTER=flash_fwd_splitkv_kernel` 仅用于让 “dump_on_first_sampled_launch” 在 flash kernel launch 时触发 dump（不影响 func_id 过滤）
- `/tmp/bpftime-sass-samples.jsonl` 非空，能看到 `bpftime_sass_smid` 记录

### 3) 手动验证（可选）

对任意一个 dump 的 `.cubin`：

```bash
/usr/local/cuda/bin/cuobjdump --dump-sass /tmp/bpftime-sass-qwen3/<file>.cubin | head
/usr/local/cuda/bin/cuobjdump --extract-ptx all /tmp/bpftime-sass-qwen3/<file>.cubin
```

预期现象：
- `--dump-sass` 中 `/*0000*/` 处是 `BRA ...`（入口已被 detour）
- `--extract-ptx` 明确提示 `No PTX file found ...`

### 4) vLLM 实测 checkpoint（待复验）

以下是历史跑通过的配置；由于近期修复了 `idx`→`IMAD.WIDE` 的依赖稳定性并引入 `BPFTIME_CUDA_SASS_DETOUR_IDX_WAIT_NOPS`，需要在真实 vLLM 上重新回归确认：

- **LANE0_ONLY gate（device-per-thread）在真实 vLLM 下按预期工作**
  - 配置：`BPFTIME_CUDA_SASS_SAMPLE_MODE=thread` + `BPFTIME_CUDA_SASS_THREAD_MAP_DEVICE=1` + `..._LANE0_ONLY=1` + `..._STRIDE4=1` + `MAX_RECORDS=1`
  - 选择 kernel：`BPFTIME_CUDA_SASS_DETOUR_FILTER=cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_tn_align8`
  - 结果：`/tmp/bpftime-vllm-thread.jsonl` 里只有 lane0 的记录（`lane_id==0`），`tid_x` 为 `0,32,64,...`；本次采到 8 条（对应 block=256 的 8 个 warp）。
- **device-per-thread “全 lane 写出”（stride4）在真实 vLLM 下可跑通**
  - 配置同上但 `..._LANE0_ONLY=0`
  - 结果：`/tmp/bpftime-vllm-thread-dense.jsonl` 里 `lane_id` 覆盖 0..31，`ctaid_x==0` 下 `tid_x` 连续覆盖 0..255（本次采样 kernel 的 block=256）。
- **真实 server + 简单压测可跑通**
  - 例：`VLLM_CONCURRENCY=4`、`VLLM_TOTAL_REQUESTS=20`、`VLLM_STREAM=1` 时，在 lane0-only（MAX_RECORDS=1）下仍可启动/可请求（无 `CUDA illegal memory access`）。
- **全局 detour（不启用采样）也可跑通**
  - `BPFTIME_CUDA_SASS_DETOUR=1` 且不设 filter、不启用 `BPFTIME_CUDA_SASS_SAMPLE`，在 `VLLM_TOTAL_REQUESTS=4` 的小压测下可完成请求。

### 5) vLLM 下仍未覆盖/未命中 detour 的典型情形（当前限制）

即使 trace 里出现 kernel 名称，也可能 **没有可 patch 的 SM120 `.text.*` ELF section**，此时会出现：
- `sass_detour_cubin_verify.error: "no_cubin_dumps"`（dump_dir 为空、或 dump_dir 下无 detoured cubin）
- `BPFTIME_CUDA_SASS_SAMPLE_DUMP_PATH` 只包含 meta，但没有任何 thread/warp/cta 记录

在 Qwen/Qwen3-0.6B 的 vLLM 实测中，下面这类 kernel/来源较容易出现上述情况（需要后续补齐 detour 能力）：
- `flash_*`（flash attention 系列）
- `cublasLt::*`（例如 `splitKreduce_kernel`）

为了把问题“拆开看清楚”，可以打开 unpatched dump：

```bash
BPFTIME_CUDA_SASS_DETOUR_DUMP_UNPATCHED=1
```

它会在 `BPFTIME_CUDA_SASS_DETOUR_DUMP_DIR` 下额外落一批 `*.cubin.raw`，文件名里会带 `reason`（例如 `no .text.* sections matched filter (text/nvinfo/symtab)`），用于离线 `readelf/cuobjdump` 分析。

推测原因（需要进一步用更细日志确认）：
- 通过 `cuLibraryLoadData` 进入的是 **fatbin/PTX/JIT 产物**，不一定是带完整 section headers 的 ELF cubin；
- 或者 ELF 的 EIATTR/section table 形态与当前 `infer_sm_version_from_elf()` / `.text.*` 扫描假设不一致，导致被判定为“不可 patch”。
 - 或者 kernel 名称并不出现在该 ELF 的 `.text.*`/`.nv.info.*`/`.symtab` 字符串里（被 strip/匿名化），导致“按名称 filter”无法命中任何 `.text.*` section（这类情况需要更强的 name→entrypoint 映射能力，或改用更宽的 detour 范围 + 更严格的 runtime gate）。

建议的排查/隔离策略：
- 先从 `BPFTIME_CUDA_TRACE_PATH` 的 JSONL 里找出最后/最频繁的 `launch.name`；
- 用该子串设置 `BPFTIME_CUDA_SASS_DETOUR_FILTER=<substring>`，并打开 `BPFTIME_CUDA_SASS_DETOUR_DUMP_DIR=/tmp/...`，确认是否能 dump 出 detoured cubin；
- 若仍无 dump，则该 kernel 目前不在“可 patch 的 sm120 cubin-only ELF”覆盖范围内，需扩展 detour/解析路径（fatbin/JIT/无 shdr 等）。

## SASS 可观测还差什么？（要做成“能观测 SM/warp/thread”）

目前我们只有“能安全改入口并跑通”的能力。要变成真正的可观测，还缺至少下面几类关键能力：

### A. 在 kernel 内执行“采样代码”（真正的 SASS 插桩）

要实现 SM/Warp/Lane mapping，典型做法是在 kernel 入口或热点位置插入一段指令：
- 读特殊寄存器：`SR_SMID` / `SR_LANEID` / `SR_TID` / `SR_CTAID` 等
- 把样本写入一个全局 buffer（最好是 ring buffer + 原子递增 index）

难点在于 **不能破坏原 kernel 的寄存器/谓词/控制流**：
- 需要寄存器分配（常见要“增加 regcount”并更新 `.nv.info.<func>` 元数据）
- 需要谓词寄存器安全使用
- 需要正确的控制码/调度（sm_120 的指令对齐/控制字段）

### B. “把数据写出来”的通道（GPU → Host）

bpftime 现有 GPU map（如 GPU kernel-shared array map）可以作为输出通道，但要在 cubin-only kernel 内写入，需要解决：
- 如何在 kernel 内拿到 **输出 buffer 指针**
  - 可能要注入常量（`.nv.constant*`）保存指针
  - 或者利用 kernel 参数区（需要稳定 ABI/定位）
- ringbuf/array 的并发写入一致性（warp/lane0 采样 vs 全线程）

### C. 选择性插桩与归因

为了可用性和性能：
- 必须支持按 kernel 名称/正则选择插桩（现在 filter 只是 substring）
- 最终要把样本和 “哪个 kernel / 哪次 launch / 哪个 request” 关联起来
  - request 级归因需要额外 marker（Python/C++ 侧）与 GPU 侧对齐

### D. 多架构与鲁棒性

当前 detour 编码、NOP 模式、控制字段都是 sm_120 观测得来的。
要推广需要：
- per-SM 的指令编码与 control 字段策略（sm_80/sm_90/sm_100/sm_120…）
- code cave 不够时的 section 扩容（ELF 重排/offset 更新/可能的重定位处理）
- 运行时安全开关、失败回退策略（宁可不插桩也不崩）

## “SM/Warp/Lane mapping + vLLM” 的落地建议（下一步）

建议分两阶段把它补全：

1) **SASS snippet 最小采样**（lane0 per warp）
   - 插入点：kernel 入口（已具备 detour 框架）
   - 写入：GPU kernel-shared array/ringbuf
   - 数据：`smid, warpid, laneid` + 时间戳/PC（可选）
   - 目标：在 vLLM 的 cubin-only kernel 上能看到“哪些 SM/warp 在跑”

2) **可观测体系化**
   - 采样开关/采样率/黑白名单
   - request/step marker 对齐（server 侧 + CUDA 事件流）
   - 支持更多观测点：memcpy 语义、KV cache 事件、NCCL（需要额外 hook/CUPTI）

## 可能的影响/风险

- 性能：入口插桩会对高频 kernel 产生额外开销；需要采样/限流
- 正确性：寄存器/谓词/控制码处理不当会导致 silent wrong 或 crash
- 兼容性：不同 CUDA/toolkit/SM 的 SASS 细节差异巨大，必须做版本化与回退
