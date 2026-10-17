import importlib.util
import os
from pathlib import Path
import runpy
import sys

# This launcher owns runtime setup; model and node modules do not change environments.
root = Path(sys.argv[1]).resolve()
if not (root / "main.py").is_file():
    raise SystemExit("Expected a ComfyUI root containing main.py")
sys.path.insert(0, str(root))
sys.argv = [str(root / "main.py"), *sys.argv[2:]]
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
if os.name == "nt" and "CUDA_PATH" not in os.environ:
    spec = importlib.util.find_spec("triton")
    if spec and spec.submodule_search_locations:
        compiler = Path(next(iter(spec.submodule_search_locations))) / "backends/nvidia"
        if (compiler / "include").is_dir() and (compiler / "lib").is_dir():
            os.environ["CUDA_PATH"] = str(compiler)
import comfy.options
comfy.options.enable_args_parsing()
import cuda_malloc
runpy.run_path(sys.argv[0], run_name="__main__")
