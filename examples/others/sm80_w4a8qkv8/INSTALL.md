# v0.29.0-sm80w4a8qkv8 安装说明

发布 wheel：`vllm-0.29.0+sm80w4a8qkv8.cu132-cp312-cp312-linux_x86_64.whl`。

验证环境为 Linux x86_64、CPython 3.12.14、PyTorch 2.13.0+cu132、
CUDA 13.2、Triton 3.7.1、A100-SXM4-80GB，驱动 595.91.07，glibc 2.34。
宿主编译参数采用 x86-64-v2。此包的 Python 标签为 cp312；它不是跨 Python
版本或通用 manylinux 包。其他 Python、PyTorch 或 CUDA 组合请按构建脚本重新编译。

## 安装到已有匹配环境

先激活具有上述 CUDA/PyTorch 依赖的虚拟环境，然后执行：

```bash
uv pip install --python .venv/bin/python --no-deps \
  ./vllm-0.29.0+sm80w4a8qkv8.cu132-cp312-cp312-linux_x86_64.whl
.venv/bin/python -c 'import vllm, torch; print(vllm.__version__, vllm.__file__, torch.__version__)'
```

这里 `.venv/bin/python` 应指向目标环境。验证时采用了独立 venv 安装该 wheel，
共享已有依赖；确认加载位置来自该 venv 的 site-packages，并从该安装位置运行模型。
没有将一次共享依赖的安装测试描述成在全新机器上重新下载并安装所有依赖。

## 新建环境

`runtime-cu132-constraints.txt` 记录本次验证环境的直接依赖版本。
先配置能够提供这些版本的包源或离线 wheel 目录，尤其是
`torch==2.13.0+cu132`、`torchvision==0.28.0+cu132`、
`torchaudio==2.11.0+cpu`，再安装：

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  --constraint runtime-cu132-constraints.txt \
  ./vllm-0.29.0+sm80w4a8qkv8.cu132-cp312-cp312-linux_x86_64.whl
```

wheel 未捆绑 PyTorch、CUDA 运行库或模型权重。本模型的 JIT 路径需要可用的 CUDA toolkit 和 `nvcc`，请配置 `CUDA_HOME` 并激活匹配的编译器环境。CUDA/Triton 安装在非标准目录时，
需按该环境设置 CUDA 头文件和库的查找路径；发布启动脚本没有写死构建机器的 Conda 路径。
八个扩展的运行时库查找路径均已改为相对 site-packages 的路径，RECORD 校验记录已重建并核对。

## 启动

使用随包交付的示例目录，或分支中的 `examples/others/sm80_w4a8qkv8`：

```bash
source .venv/bin/activate
export CUDA_HOME=/path/to/cuda-13.2
export PATH="$CUDA_HOME/bin:$PATH"
nvidia-smi --id=0,1
CUDA_VISIBLE_DEVICES=0,1 bash serve.sh /path/to/calibrated_model \
  --reasoning-parser qwen3
```

从示例目录执行上面的 `serve.sh`，并保持 `compilation.json`、
`qwen3_next_quantization.json` 与脚本在同一目录。模型需具有匹配的 W4 权重格式和
校准 Q/K/V scale。示例覆盖的是已验证的 Qwen3-Next 布局，不会自动量化任意 BF16 模型。

默认开启最佳 FP8 attention 与 fused MoE QDQ、exact 2048 graph；关闭 prefix cache
和 cascade。默认 TP 等于可见卡数，最大上下文 262144，最大并发 256，batch budget
2048，显存比例 0.8。脚本和算子均不设置 `OMP_NUM_THREADS`，直接继承用户环境；未设置时沿用原版 vLLM 的线程策略。可通过 `MAX_MODEL_LEN`、`TENSOR_PARALLEL_SIZE`、`PORT` 修改对应配置。

在相同模型与 graph 配置下使用原生 FA2 + Marlin：

```bash
CUDA_VISIBLE_DEVICES=0,1 PROFILE=w4a16 bash serve.sh /path/to/calibrated_model
```

## reasoning_content

配置 reasoning parser 后，`/v1/chat/completions` 的普通消息和流式 delta
在 `reasoning` 非 None 时同时输出相同的 `reasoning_content`。原字段继续保留，
空字符串也会复制；没有 reasoning 时不新增此别名。实现沿用用户提供的 `setdefault`
语义，不覆盖已有的同名 extra 字段。

## 从分支构建

在匹配的 CUDA/PyTorch 构建环境中，进入该分支的仓库根目录：

```bash
VLLM_VERSION_OVERRIDE=0.29.0+sm80w4a8qkv8.cu132 \
  bash examples/others/sm80_w4a8qkv8/build-wheel.sh
```

脚本从源码构建 CUDA 扩展，再使用固定版本的 patchelf 与 wheel 工具修复 RPATH、
重建 RECORD。其输出目录需仅包含本次构建的一个 vLLM wheel；可用 `WHEEL_OUTPUT_DIR`
指定空目录。`MAX_JOBS` 默认 12，`NVCC_THREADS` 默认 1，目标架构默认 SM80。

本轮压力和速度验证为 A100、TP=2；TP=4 及其他模型没有在本轮做完整服务验收。
