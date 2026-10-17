import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
for file in [*root.glob("*.py"), *root.glob("tools/*.py"), *root.glob("tests/*.py")]:
    ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
graph = json.loads((root / "workflows/moss_optimized.json").read_text(encoding="utf-8"))
nodes = {node["id"]:node for node in graph["nodes"]}
for key, source, slot, target, input_slot, kind in graph["links"]:
    assert key in nodes[source]["outputs"][slot]["links"]
    assert nodes[target]["inputs"][input_slot]["link"] == key
with tempfile.TemporaryDirectory(prefix="moss-installer-") as temporary:
    comfy = Path(temporary) / "ComfyUI with spaces"
    (comfy / "comfy").mkdir(parents=True)
    (comfy / "main.py").touch()
    (comfy / "comfy/model_patcher.py").touch()
    command = [sys.executable, "-B", str(root / "tools/install_plugin.py"), str(comfy)]
    subprocess.run(command, check=True, capture_output=True)
    installed = comfy / "custom_nodes/ComfyUI-MOSS-Optimized"
    (installed / "old-install-marker.txt").write_text("preserve me")
    subprocess.run(command, check=True, capture_output=True)
    backups = list((comfy / "moss_optimized_backups").iterdir())
    assert len(backups) == 1
    assert (backups[0] / "old-install-marker.txt").read_text() == "preserve me"
    assert not (installed / "old-install-marker.txt").exists()
    subprocess.run(command + ["--restore", str(backups[0])], check=True, capture_output=True)
    assert (installed / "old-install-marker.txt").read_text() == "preserve me"
    rejected = subprocess.run(command + ["--restore", str(comfy)], capture_output=True)
    assert rejected.returncode != 0
print("PASS: Python syntax, workflow links, install, backup, restore and restore path containment")
