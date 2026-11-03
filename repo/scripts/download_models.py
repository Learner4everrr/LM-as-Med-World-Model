from pathlib import Path
import os, json, hashlib
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
ROOT = Path(__file__).resolve().parents[2]
os.environ["HF_HOME"] = str(ROOT/"data/cache/huggingface")
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import HfApi, snapshot_download
MODELS = ["Qwen/Qwen3-8B", "Qwen/Qwen2.5-7B-Instruct",
          "mistralai/Mistral-7B-Instruct-v0.3", "microsoft/Phi-3.5-mini-instruct"]
PINS=json.loads((ROOT/"repo/configs/models_frozen.json").read_text())
def download(model):
    saved = ROOT/"data/manifests"/(model.replace("/","--")+".json")
    if saved.exists():
        prior=json.loads(saved.read_text())
        if prior["revision"]!=PINS[model]["revision"]:raise RuntimeError("Model revision mismatch: "+model)
        if all((ROOT/prior["path"]/name).exists() for name in prior["files"]):return prior
    info = HfApi().model_info(model,revision=PINS[model]["revision"])
    assert info.sha==PINS[model]["revision"]
    dest = ROOT/"data/models"/model.replace("/", "--")
    path = snapshot_download(model, revision=info.sha, local_dir=dest, max_workers=4,
          allow_patterns=["model*.safetensors", "*.json", "tokenizer.model*", "*.txt", "*.tiktoken", "README.md", "LICENSE*"])
    record = {"model_id": model, "revision": info.sha,
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "license": getattr(info.card_data, "license", None),
              "path": str(dest.relative_to(ROOT)),
              "files": {str(p.relative_to(dest)): {"bytes": p.stat().st_size,
                        "sha256": hashlib.file_digest(p.open("rb"),"sha256").hexdigest()}
                        for p in dest.rglob("*") if p.is_file() and ".cache" not in p.parts}}
    (ROOT/"data/manifests").mkdir(parents=True,exist_ok=True)
    (ROOT/"data/manifests"/(model.replace("/","--")+".json")).write_text(json.dumps(record,indent=2)+"\n")
    print("READY", model, info.sha, flush=True)
    return record
if __name__ == "__main__":
    records = [download(model) for model in MODELS]
    (ROOT/"data/manifests/models.json").write_text(json.dumps(records,indent=2)+"\n")
