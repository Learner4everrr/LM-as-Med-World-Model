#!/usr/bin/env python3
from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    dest=ROOT/"data/manifests/interface_memory_freeze.json"
    if dest.exists():raise RuntimeError("Extension already frozen")
    original=json.loads((ROOT/"data/manifests/study_freeze.json").read_text())
    for p,h in original["frozen_hashes"].items():assert sha(ROOT/p)==h,p
    paths=["repo/docs/interface_memory_extension.md","repo/src/medwm_eval/submission_extension.py",
           "repo/scripts/29_run_extension.py","repo/scripts/32_launch_extension.py","repo/scripts/33_freeze_extension.py",
           "repo/tests/test_submission_extension.py","repo/scripts/05_run_local.py","repo/src/medwm_eval/prompts.py",
           "repo/configs/models_frozen.json","repo/requirements-inference.lock"]
    for folder in ["data/simulated/benchmark_v1","data/processed/vitaldb_v1"]:
        paths.extend([folder+"/test.inputs.jsonl",folder+"/test.oracles.jsonl"])
    result=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),post_primary_extension=True,
        annotation_required=False,original_results_already_observed=True,primary_unchanged=True,
        original_primary_sha256=sha(ROOT/"data/results/study_v1/primary.json"),
        original_summary_sha256=sha(ROOT/"data/results/study_v1/summary.json"),
        original_integrity_sha256=sha(ROOT/"data/results/study_v1/integrity.json"),
        original_freeze_sha256=sha(ROOT/"data/manifests/study_freeze.json"),
        frozen_hashes={p:sha(ROOT/p) for p in paths})
    dest.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=="__main__":main()
