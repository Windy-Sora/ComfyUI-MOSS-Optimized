import hashlib
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
files = [root / name for name in ("__init__.py", "nodes.py", "model.py", "loaders.py", "text_encoder.py", "LICENSE", "NOTICE", "README.md", "README_zh.md", "README_en.md", "pyproject.toml", ".gitignore", ".gitattributes", "Install-Plugin.bat", "Install-CUDA13.bat", "Start-CUDA13.bat")]
for folder, pattern in (("workflows","*.json"),("tools","*.py"),("tools/windows","*.ps1"),("tools/windows","*.bat"),("tests","*.py"),("benchmarks","*.json")):
    files.extend(sorted((root / folder).glob(pattern)))
for file in files:
    if file.is_symlink() or not file.is_file() or not file.resolve().is_relative_to(root):
        raise SystemExit(f"Refusing unexpected release file: {file}")
manifest = {file.relative_to(root).as_posix():hashlib.sha256(file.read_bytes()).hexdigest() for file in files}
output = root / "dist"
output.mkdir(exist_ok=True)
archive = output / "ComfyUI-MOSS-Optimized-0.1.0a1.zip"
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
    for file in files:
        bundle.write(file, "ComfyUI-MOSS-Optimized/" + file.relative_to(root).as_posix())
    bundle.writestr("ComfyUI-MOSS-Optimized/SHA256SUMS.json", json.dumps(manifest, indent=2))
print(archive)
print(f"{len(files)} source files; {archive.stat().st_size} bytes; runtime and local tests excluded")
