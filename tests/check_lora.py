import argparse
import importlib.util
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description="Check standard LoRA patching on a small unquantized MOSS model")
parser.add_argument("comfy_root", type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(args.comfy_root.resolve()))
sys.dont_write_bytecode = True
sys.argv = [sys.argv[0], "--cpu", "--disable-xformers"]
import comfy.options
comfy.options.enable_args_parsing()
import torch
import comfy.model_patcher
import comfy.sd

torch.set_grad_enabled(False)
spec = importlib.util.spec_from_file_location("moss_optimized_test", root / "__init__.py", submodule_search_locations=[str(root)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
loaders = sys.modules["moss_optimized_test.loaders"]
config = loaders.MOSSImageConfig({"dim":64, "ffn_dim":128, "num_heads":1, "num_layers":1,
    "text_dim":4096, "freq_dim":256, "latent_channels":128, "rope_base":10000.0, "fused_qkv":True})
config.set_inference_dtype(torch.float32, None, device=torch.device("cpu"))
model = config.get_model({}, device=torch.device("cpu"))
key = "diffusion_model.layers.0.proj.o.weight"
weight = model.diffusion_model.layers[0].proj["o"].weight
weight.zero_()
patcher = comfy.model_patcher.ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
up = torch.ones(64, 2) * 0.01
down = torch.ones(2, 64) * 0.02
lora = {key[:-7] + ".lora_up.weight":up, key[:-7] + ".lora_down.weight":down}
patched, clip = comfy.sd.load_lora_for_models(patcher, None, lora, 0.5, 0)
assert key in patched.patches
assert key not in patcher.patches
actual = patched.patch_weight_to_device(key, torch.device("cpu"), return_weight=True)
torch.testing.assert_close(actual, 0.5 * up @ down)
assert torch.count_nonzero(weight).item() == 0
print("PASS: generic LoRA maps to a MOSS layer, clones patcher, applies expected delta and preserves original weights")
print("Synthetic unquantized test only; real quantized MOSS LoRA files still need validation")
del patched, patcher
