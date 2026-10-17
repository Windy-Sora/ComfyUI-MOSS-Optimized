import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description="Check CUDA 13, required ComfyUI APIs and optional compilation")
parser.add_argument("comfy_root", type=Path)
parser.add_argument("--compile", action="store_true")
args = parser.parse_args()
root = args.comfy_root.resolve()
if not (root / "main.py").is_file():
    parser.error("Expected a ComfyUI root")
sys.path.insert(0, str(root))
sys.dont_write_bytecode = True
if os.name == "nt" and "CUDA_PATH" not in os.environ:
    spec = importlib.util.find_spec("triton")
    if spec and spec.submodule_search_locations:
        compiler = Path(next(iter(spec.submodule_search_locations))) / "backends/nvidia"
        if (compiler / "include").is_dir():
            os.environ["CUDA_PATH"] = str(compiler)
import comfy.options
comfy.options.enable_args_parsing()
sys.argv = [sys.argv[0], "--use-ck-attention", "--disable-xformers"]
import cuda_malloc
import torch
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention
from comfy.model_prefetch import make_prefetch_queue, prefetch_queue_pop
from comfy_api.torch_helpers import set_torch_compile_wrapper
from comfy.quant_ops import ck
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
if torch.version.cuda is None or int(torch.version.cuda.split('.')[0]) < 13:
    raise SystemExit("This optional runtime tool requires PyTorch CUDA 13 or newer")
result = {"python":sys.version.split()[0], "torch":torch.__version__, "cuda":torch.version.cuda,
          "device":torch.cuda.get_device_name(0),
          "versions":{name:importlib.metadata.version(name) for name in ("comfy-kitchen","comfy-aimdo")},
          "backends":{name:{key:value for key,value in info.items() if key != "capabilities"} for name,info in ck.list_backends().items()}}
if args.compile:
    def activation(x):
        return torch.tanh(x) * torch.sigmoid(x)
    x = torch.randn(1024, device="cuda")
    compiled = torch.compile(activation)
    torch.testing.assert_close(compiled(x), activation(x), rtol=2e-5, atol=2e-5)
    result["compile"] = "passed"
print(json.dumps(result, indent=2, default=str))
