import argparse
from datetime import datetime
from pathlib import Path
import shutil

FILES = ("__init__.py", "nodes.py", "model.py", "loaders.py", "text_encoder.py", "pyproject.toml", "LICENSE", "NOTICE", "README.md", "README_zh.md", "README_en.md", "Install-Plugin.bat", "Install-CUDA13.bat", "Start-CUDA13.bat")


def main():
    parser = argparse.ArgumentParser(description="Install only this plugin; existing copies are backed up.")
    parser.add_argument("comfy_root", type=Path)
    parser.add_argument("--restore", type=Path, help="Restore an exact backup path printed by an earlier install")
    args = parser.parse_args()
    comfy = args.comfy_root.resolve()
    if not (comfy / "main.py").is_file() or not (comfy / "comfy/model_patcher.py").is_file():
        parser.error("Expected a ComfyUI root containing main.py and comfy/model_patcher.py")
    source = Path(__file__).resolve().parents[1]
    target = comfy / "custom_nodes/ComfyUI-MOSS-Optimized"
    backups = comfy / "moss_optimized_backups"
    if target.is_symlink() or (target.exists() and target.resolve().parent != (comfy / "custom_nodes").resolve()):
        parser.error("Refusing to replace a symlink installation; manage the link manually")
    restore = args.restore.resolve() if args.restore else None
    if restore and (restore.parent != backups or not (restore / "nodes.py").is_file()):
        parser.error("Restore must point to a plugin backup inside moss_optimized_backups")
    if not restore and target.resolve() == source:
        print("Already installed in custom_nodes. Restart ComfyUI.")
        return
    if target.exists():
        backups.mkdir(exist_ok=True)
        backup = backups / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        target.rename(backup)
        print(f"Previous installation preserved: {backup}")
    if restore:
        shutil.copytree(restore, target)
    else:
        target.mkdir(parents=True)
        for name in FILES:
            shutil.copy2(source / name, target / name)
        shutil.copytree(source / "workflows", target / "workflows")
        shutil.copytree(source / "tools", target / "tools", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    print(f"Installed: {target}")
    print("Restart ComfyUI; replace the MOSS loader with MOSS Optimized Loader.")


if __name__ == "__main__":
    main()
