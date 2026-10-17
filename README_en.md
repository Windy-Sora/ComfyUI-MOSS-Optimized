# ComfyUI MOSS Optimized (experimental)

[中文文档](README.md) · [Download alpha](https://github.com/Windy-Sora/ComfyUI-MOSS-Optimized/releases/tag/v0.1.0-alpha.1)

Independent MOSS-Image optimization nodes and optional Windows CUDA 13 tools, derived from OpenMOSS/MOSS-Image. This is a community derivative, not an official release. Model weights are not included.

## Installation

Clone or extract this folder into `ComfyUI/custom_nodes/ComfyUI-MOSS-Optimized`, or run `tools/windows/Install-Plugin.bat`. The installer asks for the ComfyUI root, requires a `python` command (or `-PythonPath`), backs up an existing installation, and changes only this plugin folder. Restart ComfyUI. No model weights or Python packages are downloaded by nodes or plugin installation.

ComfyUI must provide Qwen3.5 text encoding, `ComfyAttention`, `AttentionTensorContainer`, model prefetch, and `set_torch_compile_wrapper`. Tested with ComfyUI commit `b53f77a74`, comfy-kitchen 0.2.37 and comfy-aimdo 0.5.6. Older ComfyUI versions are not supported.

## Workflow

Open `workflows/moss_optimized.json`. Select your existing weights:

- Diffusion model: `models/diffusion_models/moss_image_sft_int8_convrot.safetensors`
- Text encoder: `models/text_encoders/qwen3.5_9b_w4a8_gptq.safetensors`
- VAE: `models/vae/flux2-vae.safetensors`

Replace the original MOSS model loader with **MOSS Optimized Loader**. Existing MOSS CLIP loaders can remain; the included **MOSS Qwen3.5 Text Encoder** also allows standalone installation. Node IDs are distinct from the original plugin. Standard MODEL/CLIP outputs are retained. No global monkeypatches or changes to ComfyUI core are installed.

`compile_situ=false` is the default. Enable it with a working torch.compile/Triton toolchain. Only parameter-free SiTU activations are compiled through ModelPatcher; quantized matmul and attention stay eager. First use and new shapes may compile. Compilation retains the native dynamic ModelPatcher and wraps only parameter-free activations. It does not clone or convert model storage. Bypass compilation by disabling this input. Switching this input creates a different loaded model; on memory-limited systems use ComfyUI Free model and node cache before switching between compiled and eager paths. Do not combine with a separate whole-model compilation node.

Example settings: 1024 x 1536, 50 steps, Euler / sgm_uniform, CFG 3, shift 5, SLG single layer 22. The API workflow is included for reproducibility. Models and their licenses must be obtained separately. Current package supports local component safetensors, not HF-directory or GGUF loaders.

## Optional Windows CUDA 13 environment

`tools/windows/Install-CUDA13.bat` creates `.runtime` beside the plugin tools. Supply a Python 3.10 executable with `-PythonPath` to match the tested setup. It downloads the explicitly requested PyTorch CUDA 13 wheels and ComfyUI dependencies; it does not modify the original Python or driver. This is a clean venv, not a copy of your existing custom-node dependencies. Installation is large and needs network access and several GB of disk space. Existing runtime directories are never overwritten; use `-Resume` for the same dedicated venv after an interrupted install, or choose another `-RuntimeDir`.

The script pins the tested torch 2.10.0 / torchvision 0.25.0 / torchaudio 2.10.0 CUDA 13 wheels, comfy-kitchen 0.2.37, comfy-aimdo 0.5.6 and triton-windows 3.6.0.post26, plus the tested NumPy/Transformers/tokenizer/safetensors versions. A newer ComfyUI dependency conflict stops installation instead of changing the original environment. It checks CUDA, required APIs and compilation before saving runtime.json.

Run `Start-CUDA13.bat`. The runtime can load this source folder externally without copying anything into ComfyUI; an installed copy is preferred when present. Default port is 8189, bound to 127.0.0.1, with only this plugin enabled. The runtime has separate user settings and outputs. `-Port 8190` changes the port; `-AllCustomNodes` enables other plugins but their dependencies and CUDA extensions need separate validation. The launch wrapper locates Triton's compiler without importing or borrowing old CUDA 12 DLLs. No driver or system CUDA Toolkit installer is included.

Keep your original launcher for rollback. Two services can stay open, but run GPU jobs on one at a time. To restore an earlier plugin installation, stop its service and run `python tools/install_plugin.py COMFY_ROOT --restore EXACT_BACKUP_PATH`. Backups live outside custom_nodes in `moss_optimized_backups`. If no previous plugin existed, removing this plugin folder and reconnecting the original loader reverts the installation.

## Measurements and limits

Historical same-settings laptop runs: 341.01 s baseline, 121.47 s first optimizations, 103.00 s CUDA 13 fusion, 89.19 s compiled SiTU. See `benchmarks/results.json`. These are individual runs; later runs reused model/compiler caches. This is not a guaranteed speedup or a statistical comparison. The historical baseline already contains the fused-QKV loading fix; the original incompatible loader cannot generate this checkpoint.

Only Windows / RTX 5070 Ti Laptop 12 GB has been exercised. CPU, ROCm, MPS and other GPUs have not been validated. Small floating-point/image differences are expected. External patches that depend on the old internal RoPE representation are not guaranteed compatible. SLG is part of the integration test. Real compatible LoRA weights and low-VRAM edge cases require additional testing before a stable release.

Package validation and fresh-environment installation status are recorded in `benchmarks/package_validation.json` when available. No credentials, personal profiles, logs, model weights, environments or browser data belong in a release.

The bundled example uses standard VAEDecodeTiled (512 px tiles, 64 px overlap) to reduce peak decode memory. Full-frame VAEDecode caused a native allocation failure in standalone 50-step tests on the test machine; tile decoding can change boundary details and runtime. Historical timings used the earlier workflow.

Standalone release candidate: a clean CUDA 13 venv installed successfully and passed pip check/compile checks. The full compiled 1024x1536 50-step workflow with SLG and tiled VAE saved an image in 86.52 s server time. This is one run, not a benchmark average.

The final SiTU option retains dynamic model storage. Compiled generation, cache clearing, switching to eager generation, and cache clearing after a 50-step run passed. An earlier non-dynamic candidate crashed on unload and was removed.
