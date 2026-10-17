# ComfyUI MOSS Optimized

**MOSS-Image 的独立 ComfyUI 优化插件与可选 Windows CUDA 13 环境工具。**

[English](README_en.md) · [下载实验版](https://github.com/Windy-Sora/ComfyUI-MOSS-Optimized/releases/tag/v0.1.0-alpha.1) · [示例工作流](workflows/moss_optimized.json) · [验证报告](benchmarks/package_validation.json)

> 实验版：当前仅验证 Windows、Python 3.10 和 RTX 5070 Ti Laptop 12GB。这是基于 [OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image) 的社区衍生项目，并非官方版本。模型权重需要自行准备。

## 能做什么

- 用 **MOSS Optimized Loader** 替换原 MOSS 模型加载节点，保持标准 `MODEL` 输出。
- 支持融合 QKV 权重，使用 Comfy Kitchen 的 RMSNorm/RoPE 运算、ComfyAttention 路由与模型预取。
- 可选 SiTU 激活编译；保留原生动态内存管理，不做整个模型的编译。
- 提供配套 Qwen3.5 文本编码器节点、GUI/API 示例工作流。
- 可单独创建 CUDA 13 venv，并从外部目录加载本插件。

插件拥有独立节点 ID，可与原 MOSS 插件共存。节点运行时不会下载模型、安装依赖或覆盖 ComfyUI 核心。

## 快速开始

### 使用已有 ComfyUI 环境

在 ComfyUI 的 `custom_nodes` 目录运行：

```bash
git clone https://github.com/Windy-Sora/ComfyUI-MOSS-Optimized.git
```

也可以下载 ZIP 解压到任意目录，双击根目录的 `Install-Plugin.bat`，输入 **包含 `main.py` 的 ComfyUI 根目录**。安装器只复制本插件及工具；已存在的本插件版本会先备份。

重启 ComfyUI，导入 [`workflows/moss_optimized.json`](workflows/moss_optimized.json)，选择本机的模型文件。已有工作流可只替换模型加载节点。

| 文件 | 放置目录 |
|---|---|
| `moss_image_sft_int8_convrot.safetensors` | `ComfyUI/models/diffusion_models/` |
| `qwen3.5_9b_w4a8_gptq.safetensors` | `ComfyUI/models/text_encoders/` |
| `flux2-vae.safetensors` | `ComfyUI/models/vae/` |

这些是本项目实际测试的文件名，不是对所有 MOSS 权重格式的兼容承诺。模型下载与使用须遵守各自许可证。本插件支持本地 safetensors 组件，暂不提供 HF 整目录或 GGUF 加载器。

### 使用独立 CUDA 13 环境（可选，Windows/NVIDIA）

1. 将本项目放在任意可写目录，双击 `Install-CUDA13.bat`。
2. 输入现有 ComfyUI 根目录。脚本创建 `.runtime` 独立 venv，下载依赖并验证 CUDA 与编译功能。
3. 双击 `Start-CUDA13.bat`。默认访问 `http://127.0.0.1:8189`。

这种方式可以直接加载外部项目源码，不需要复制到原 `custom_nodes`。若已有安装副本，则优先加载安装副本。原 Python 和驱动不会被替换；新环境不借用旧 CUDA 12 DLL。

脚本会识别常见整合包 Python。需要指定 Python、端口或运行时目录时：

```bat
Install-CUDA13.bat -ComfyRoot "D:\ComfyUI" -PythonPath "C:\Python310\python.exe"
Start-CUDA13.bat -Port 8190
Install-CUDA13.bat -ComfyRoot "D:\ComfyUI" -Resume
```

安装需要网络和数 GB 磁盘空间。已有目录不会默认覆盖；`-Resume` 只接受本工具创建的 venv。PyTorch 来自官方 CUDA 13 索引，其他包使用官方 PyPI。

新服务默认仅启用本插件，使用独立的用户配置、输入、输出、临时目录及日志。`Start-CUDA13.bat -AllCustomNodes` 可启用其他插件，但其依赖和 CUDA 扩展须另行验证。多个实例共享显卡时，建议错开生成任务。

## 节点与参数

| 节点 | 用途 |
|---|---|
| `MOSS Optimized Loader` | 加载 MOSS 主模型，选择精度和 `compile_situ` |
| `MOSS Qwen3.5 Text Encoder` | 加载配套编码器，输出标准 `CLIP` |

典型连接：

```text
MOSS Optimized Loader → LoRA（可选）→ ModelSamplingSD3 / SLG → KSampler
```

原 MOSS 文本编码器节点可继续使用。VAE、采样器、SLG 等复用 ComfyUI 已有节点。

`compile_situ` 默认关闭；有可用的 `torch.compile`/Triton 环境时再开启。它只编译无参数的 SiTU 激活模块，量化矩阵乘法和注意力仍走现有算子。首次使用及新形状可能触发编译。不要再叠加整个模型的编译节点。内存较小时，切换选项前先清理模型和节点缓存。

示例设置：1024×1536、50 步、Euler / sgm_uniform、CFG 3、shift 5、SLG 单层 22。示例使用 ComfyUI 自带 **VAEDecodeTiled**，分块 512、重叠 64，以降低解码峰值内存。

## 实测结果

同一台 RTX 5070 Ti Laptop 12GB、32GB RAM 上的记录：

| 配置 | 采样耗时 | 总耗时 |
|---|---:|---:|
| 可运行的融合 QKV 基线 | 约 329 秒 | 341.01 秒 |
| CUDA 12.8 + Triton + 第一轮算子优化 | 约 118 秒 | 121.47 秒 |
| CUDA 13 + RMSNorm/RoPE 融合 | 约 96 秒 | 103.00 秒 |
| 早期版本 + SiTU 编译 | 约 78 秒 | 89.19 秒 |
| **当前独立插件 + 动态内存管理 + SiTU 编译 + 分块解码** | **约 75 秒** | **86.52 秒** |

**这些是单次记录，不是统计平均或通用提速承诺。** 后续运行复用了加载及编译缓存；最终版本采用了分块解码，不能把表格理解为严格控制变量的逐项性能归因。历史基线已经修复融合 QKV 加载问题。

已验证：干净 CUDA 13 venv 安装、`pip check`、编译工具链、完整 50 步生成、SLG、缓存清理、编译/未编译切换后再次生成、安装备份与恢复，以及小型未量化 MOSS 的合成 LoRA 补丁测试。

原始历史参数见 [`benchmarks/results.json`](benchmarks/results.json)，最终验证与失败记录见 [`benchmarks/package_validation.json`](benchmarks/package_validation.json)。

## 环境与限制

已测组合：ComfyUI 提交 `b53f77a74`、Python 3.10.11、PyTorch 2.10.0+cu130、torchvision 0.25.0+cu130、torchaudio 2.10.0+cu130、comfy-kitchen 0.2.37、comfy-aimdo 0.5.6、triton-windows 3.6.0.post26。

ComfyUI 需要提供 Qwen3.5、`ComfyAttention`、`AttentionTensorContainer`、model prefetch 和 `set_torch_compile_wrapper` API。旧版本未适配。环境安装器会锁定关键依赖；新 ComfyUI 若要求冲突版本会停止安装。

- CPU、ROCm、MPS、其他显卡和极低显存场景尚未验证。
- 真实量化 MOSS LoRA 尚未完整验证；合成接口测试不等于所有 LoRA 文件兼容。
- 浮点结果与图片局部细节可能有变化；依赖旧内部 RoPE 布局的第三方补丁需单独验证。
- 普通全图 VAE 解码在本机独立测试中曾发生原生内存分配失败，示例因此使用分块解码。分块会影响解码耗时与局部拼接细节。
- 早期强制普通 ModelPatcher 的候选实现曾在卸载量化权重时退出；最终代码已移除该转换，并通过生成、卸载、切换和再生成测试。

## 回退、复现与打包

原插件仍可使用，接回原加载节点即可。安装器将本插件旧版保存到 `ComfyUI/moss_optimized_backups`；停止服务后可恢复指定备份：

```bat
python tools/install_plugin.py "D:\ComfyUI" --restore "D:\ComfyUI\moss_optimized_backups\BACKUP_NAME"
```

检查与复现：

```bash
python tests/check_package.py
python tests/check_lora.py /path/to/ComfyUI
python tools/benchmark.py --port 8190 --steps 50 --compile --result result.json
python tools/build_release.py
```

运行 LoRA 测试与基准需要在已配置 ComfyUI 依赖的 Python 中执行。打包脚本使用固定白名单，ZIP 内附 SHA256 清单；不会收集模型、环境、用户配置、浏览器数据或原始日志。

## 来源与许可证

基于 [OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image) 的 `comfyui/moss-image`，基础提交为 `c1355e6b1986dfe8fba3c0455783a05899212342`。共享优化算子由 ComfyUI / Comfy Kitchen 提供。

代码采用 [Apache License 2.0](LICENSE)，改动来源见 [NOTICE](NOTICE)。模型权重不随仓库分发，其许可证独立适用。
