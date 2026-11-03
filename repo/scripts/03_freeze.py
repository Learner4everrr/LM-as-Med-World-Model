#!/usr/bin/env python3
"""Create a new internal freeze, or verify the existing one without overwriting."""
from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    dest=ROOT/"data/manifests/study_freeze.json"
    if dest.exists():
        d=json.loads(dest.read_text())
        for p,h in d["frozen_hashes"].items():assert sha(ROOT/p)==h,p
        print("Existing freeze verified:",sha(dest));return
    paths=["repo/configs/study_v1.json","repo/docs/protocol.md","repo/docs/ai_review.md",
           "repo/scripts/05_run_local.py","repo/src/medwm_eval/prompts.py",
           "repo/src/medwm_eval/diagnostics.py","repo/src/medwm_eval/baselines.py",
           "repo/requirements-analysis.lock","repo/requirements-inference.lock"]
    files=[ROOT/p for p in paths]
    for folder in ["data/simulated/benchmark_v1","data/processed/vitaldb_v1"]:
        for split in ["development","test"]:
            for kind in ["inputs","oracles"]:files.append(ROOT/folder/(split+"."+kind+".jsonl"))
    assert all(p.exists() for p in files)
    verification=json.loads((ROOT/"data/results/verification/verification.json").read_text())
    assert verification["contracts"]["returncode"]==0 and verification["dependencies"]["returncode"]==0
    d=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),registration="internal specification, not public preregistration",
           frozen_hashes={str(p.relative_to(ROOT)):sha(p) for p in files},verification=verification)
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(d,indent=2)+"\n")
    print("Created new reproduction freeze:",sha(dest))
if __name__=="__main__":main()
