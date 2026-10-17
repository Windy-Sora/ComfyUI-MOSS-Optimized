import argparse
import json
from pathlib import Path
import time
import urllib.request

parser = argparse.ArgumentParser(description="Run the fixed public MOSS workflow against a local ComfyUI")
parser.add_argument("--port", type=int, default=8190)
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--compile", action="store_true")
parser.add_argument("--result", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
graph = json.loads((root / "workflows/moss_optimized_api.json").read_text(encoding="utf-8"))
graph["16"]["inputs"]["steps"] = args.steps
graph["21"]["inputs"]["compile_situ"] = args.compile
graph["18"]["inputs"]["filename_prefix"] = "validation/" + ("compiled" if args.compile else "eager") + f"_{args.steps}steps"
base = f"http://127.0.0.1:{args.port}"
def request(path, data=None):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(base + path, data=body, headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)
queue = request("/queue")
if queue["queue_running"] or queue["queue_pending"]:
    raise SystemExit("Server queue is busy; no benchmark submitted")
started = time.monotonic()
job = request("/prompt", {"prompt":graph})
identifier = job["prompt_id"]
print("Submitted", identifier, "compile", args.compile, "steps", args.steps, flush=True)
while time.monotonic() - started < 1800:
    history = request("/history/" + identifier)
    if identifier in history:
        result = history[identifier]
        result["client_elapsed_seconds"] = round(time.monotonic() - started, 3)
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Completed:", result["status"]["status_str"], "elapsed", result["client_elapsed_seconds"], flush=True)
        print(json.dumps(result["outputs"], ensure_ascii=False), flush=True)
        if result["status"]["status_str"] != "success":
            raise SystemExit("Execution failed; see result file")
        break
    time.sleep(2)
else:
    raise SystemExit("Timed out; job may still be running on the server")
