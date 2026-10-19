# v20.1 group32：正式源码版

## r6 多模态接入

视觉塔及视觉投影使用原生 BF16，不接收语言侧的量化配置。
语言侧保留 r6 Dense FP8 核心、group32 MoE 和 checkpoint 中保存的 QKV scales。
视觉 attention 按 `mm_encoder_attn_backend` 独立选择，语言侧自动选择
SM80 FP8 KV attention。不能为这个部署设置 `--mm-encoder-attn-dtype fp8`。

图文输入不需要 `--language-model-only`；需要纯文本时仍可显式设置该参数。
`serve_group32.sh` 默认支持图文和 MTP3，`MTP_TOKENS=0` 可关闭 MTP。
异步调度使用 vLLM 的自动选择，不额外强制关闭。
MTP 与异步调度同时启用时，vLLM 会自动关闭 cascade attention。

## 2026-10-14 历史发布记录

2026-10-14：将已完成 126 项评测的 native-prepared group32 Marlin 实现
整合到 `v0.29.0-sm80w4a8qkv8`。此次为源码分支发布，尚未为该提交重新
编译或安装 wheel；旧 wheel 不会自动包含本次修改。

## 模型选择与数值语义

保留原始 checkpoint 的 config、FP8 codes 和保存的 scales。SM80 上匹配
Qwen3.5 MoE、INT4 group32 专家权重、FP8 group32 Dense 权重、动态 group32
激活及静态 QKV scheme 时，自动选择 `sm80_group32`。

* routed experts 使用可选 backend `marlin_group32_fp8_qdq`：group32 FP8
  QDQ 后以 BF16 输入原生 Marlin，保留 packed INT4 专家权重。
* FC1 复用 residual RMSNorm 的 prepared QDQ 输出；FC2 前融合训练舍入的
  SwiGLU 与 group32 QDQ。路由概率由原生 Marlin 在 FC2 后乘，使用原生求和。
* 该历史版本的 Dense 权重按保存的 FP32 group32 scale 反量化为 BF16。
  当前 r6 已使用独立 FP8 核心，保留 FP32 scales，不回退到 BF16 QDQ。
* QKV 使用 checkpoint 保存的 FP32 scale；FP8 KV cache 对应真实 E4M3 存储。
* 旧 group128/token-FP8 模型继续使用 `marlin_fp8_qdq_fused`；普通 W4A16
  保留原来的 Marlin 路径。新模型不能显式选旧后端来绕过 group32 QDQ。
* `sm80_qdq_reference` 保留为独立精度参考。它的训练式 FC2 输入概率乘法、
  专家累加与当前生产后端不同；不能宣称完整模型逐 bit 相等。

## 启动

安装从本分支正确构建的 vLLM 后，可用以下最小命令。`MODEL` 指向用户的
原始 v20.1 group32 best-hf-int4fp8 checkpoint。

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve "$MODEL" \
    --dtype bfloat16 \
    --kv-cache-dtype fp8_e4m3 \
    --tensor-parallel-size 2 \
    --max-model-len 262144 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --trust-remote-code
```

通常不需要指定 backend。显式诊断可加：

```bash
--quantization sm80_group32 \
--moe-backend marlin_group32_fp8_qdq \
--attention-backend FLASH_ATTN_QKV_FP8_SM80_FUSED
```

不需要 HF_OVERRIDES，也不要将 `input_activations` 设为 null。
`OMP_NUM_THREADS` 完全由调用者控制，代码和脚本均不强制设置。

`serve_group32.sh` 保留完整 API 示例、GPU 占用检查和可选 MTP 参数。
`ggthink_parser.py` 使用独立名称 `group32_ggthink`，解析 `<ggthink>` 标记；
普通与流式响应继续保留 `reasoning_content` 别名。
`opencompass_group32.py` 提供 base/r/enr 配置。外层 batch_size 和引擎
max_num_seqs、max_num_batched_tokens 的含义不同。

## 本次精度与速度

42 数据集 × base/r/enr，共 126 项、31,989 条请求全部完成。
每个模式的分数为 42 个数据集等权均分，对照是已接受的 R24。

| 模式 | R24 | 本版 | 差值，百分点 |
| --- | ---: | ---: | ---: |
| base | 73.1484 | 73.1109 | -0.0375 |
| r | 75.1771 | 75.0730 | -0.1041 |
| enr | 76.0951 | 75.7123 | -0.3828 |

这是单轮结果，不宣称差异全部来自随机性。相同输入、相同整数路由下，
真实模型的局部融合/原生 Marlin 对照约 23.9 亿 BF16 元素无差异；跨加载
完整模型的 Prefill 差异尚未完全归因，不能宣称全模型 bit 一致。

速度：同一 checkpoint、TP2，四轮 ABBA 引擎加载，每个场景每策略 6 次
计时取中位数；GPU 占用经检查，未强制设置 OMP。KV 预算每卡 24 GiB，
max_num_seqs=256、max_num_batched_tokens=8192、max_model_len=262144。
候选使用 FP8 KV/QKV/QDQ，对照 W4A16 使用 BF16 KV/native FA2。

| 场景：输入 token × batch | 本版 token/s | W4A16 token/s |
| --- | ---: | ---: |
| Prefill 128 × 1 | 3403.36 | 3188.81 |
| Prefill 512 × 4 | 17473.05 | 22926.97 |
| Prefill 8192 × 1 | 28903.91 | 32743.34 |
| Prefill 32768 × 1 | 24360.88 | 27102.79 |
| Decode 1024 × 1 | 126.18 | 151.81 |
| Decode 1024 × 16 | 1332.36 | 1557.83 |
| Decode 1024 × 64 | 3210.66 | 3717.53 |
| Decode 1024 × 256 | 5458.02 | 5995.27 |

评分使用相同采样/逐请求 seed 和历史后处理，每卡 KV 预算 40 GiB。
上述完整速度和评分均关闭 Prefix/MTP，使用默认最高 512 的 Graph 捕获。
功能 smoke 与旧后端回归的本次结果另记于 `release_validation.json`；
smoke 不替代 Prefix/MTP 的完整数据集精度及长期压力测试。

## 范围与后续工作

主要验证对象是 A100 SM80、BF16、TP2、纯文本。EP、sequence-parallel MoE、
PP>1、LoRA 不属于此高速 group32 路径的支持范围。
旧 group128/token-FP8 示例仍在相邻 `sm80_w4a8qkv8` 目录。

按用户要求，本次保留当前已评测实现，不引入新的速度优化。
`deferred_optimizations.json` 保存下一轮的 2048-token Graph、共享专家
prepared 复用与融合、mask/combine 融合、warp 调优、QK-Norm/RoPE/QDQ
完整融合，以及尚未归因的跨加载 Prefill 差异。Cascade 暂不验证。

本次源码评测运行时为 Python 3.10 / Torch 2.13.0+cu129，复用已审计的
R24 原生扩展。它不是当前正式提交新 wheel 的安装验证。新 wheel 必须
重新记录编译环境、文件 SHA256、依赖和安装测试，不能改名复用旧包。
