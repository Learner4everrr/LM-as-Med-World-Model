#!/usr/bin/env python3
"""Reconstruct all extension prompts, scores and self-fed histories; seal results."""
from pathlib import Path
from copy import deepcopy
from collections import Counter
from datetime import datetime,timezone
import json,hashlib,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"repo/src"))
from medwm_eval.submission_extension import structured_prompt,parse_structured,memory_task,score_memory,VERSION
from medwm_eval.prompts import feedback
MODELS=["Qwen/Qwen3-8B","Qwen/Qwen2.5-7B-Instruct","mistralai/Mistral-7B-Instruct-v0.3","microsoft/Phi-3.5-mini-instruct"]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def loadl(p):return [json.loads(l) for l in p.read_text().splitlines()]
def main():
    freeze=ROOT/"data/manifests/interface_memory_freeze.json";f=json.loads(freeze.read_text())
    for p,h in f["frozen_hashes"].items():assert sha(ROOT/p)==h,p
    for key,path in [("original_primary_sha256","data/results/study_v1/primary.json"),
                     ("original_summary_sha256","data/results/study_v1/summary.json"),
                     ("original_integrity_sha256","data/results/study_v1/integrity.json")]:
        assert sha(ROOT/path)==f[key],path
    old=json.loads((ROOT/"data/manifests/study_freeze.json").read_text())
    for p,h in old["frozen_hashes"].items():assert sha(ROOT/p)==h,p
    cases={};truth={}
    for folder in ["data/simulated/benchmark_v1","data/processed/vitaldb_v1"]:
        cases.update({r["id"]:r for r in loadl(ROOT/folder/"test.inputs.jsonl")})
        truth.update({r["id"]:r["targets"] for r in loadl(ROOT/folder/"test.oracles.jsonl")})
    models=[]
    for model in MODELS:
        out=ROOT/"data/runs/interface_memory_v1/test"/model.replace("/","--")
        manifest=json.loads((out/"manifest.json").read_text())
        assert manifest["freeze_sha256"]==sha(freeze) and manifest["cases"]==660 and manifest["retrieval_tasks"]==1320
        for n,h in manifest["files"].items():assert sha(out/n)==h,n
        raw=loadl(out/"raw.jsonl");cache={r["prompt_hash"]:r for r in raw}
        assert len(raw)==len(cache)
        for r in raw:
            assert r["model_id"]==model and r["prompt_version"]==VERSION
            assert r["timestamp_utc"]>f["timestamp_utc"]
            assert hashlib.sha256(r["prompt"].encode()).hexdigest()==r["prompt_hash"]
        runs=loadl(out/"predictions.jsonl")
        assert len(runs)==660 and {r["id"] for r in runs}==set(cases)
        counts=Counter()
        for run in runs:
            assert run["condition"]=="state_action_named" and run["mode"]=="free"
            c=cases[run["id"]];p=c["payload"];h=deepcopy(p["history"])
            assert 1<=len(run["predictions"])<=12
            for step,r in enumerate(run["predictions"]):
                rawr=cache[r["prompt_hash"]]
                assert r["step"]==step+1
                assert structured_prompt(c,h,step)==rawr["prompt"]
                parsed=parse_structured(rawr["raw"],p["variables"])
                assert all(r.get(k)==v for k,v in parsed.items())
                counts["attempted"]+=1;counts["valid"]+=int(r["valid"]);counts["interval_valid"]+=int(r.get("interval_valid",False))
                if not r["valid"]:
                    assert step==len(run["predictions"])-1;break
                h=feedback(h,r["values"],p["actions"][step],p["step_min"])
            if all(r["valid"] for r in run["predictions"]):assert len(run["predictions"])==12
            counts["complete"]+=int(len(run["predictions"])==12 and all(r["valid"] for r in run["predictions"]))
        memories=json.loads((out/"memory.json").read_text())
        assert len(memories)==1320 and {(r["id"],r["anchor_step"]) for r in memories}=={(cid,a) for cid in cases for a in [0,6]}
        for r in memories:
            prompt,target=memory_task(cases[r["id"]],truth[r["id"]],r["anchor_step"])
            assert r["target"]==target
            record=cache[r["prompt_hash"]]
            assert record["prompt"]==prompt and record["raw"]==r["raw"]
        models.append(dict(model=model,cases=660,retrieval_tasks=1320,counts=dict(counts),scheduled=7920,
            unique_raw_calls=len(raw),input_tokens=sum(r["input_tokens"] for r in raw),
            generated_token_slots_upper_bound=sum(r["generated_token_slots"] for r in raw),
            generation_seconds=sum(r["batch_seconds"]/r["batch_size"] for r in raw),
            prompt_reconstruction_exact=True,raw_parse_exact=True,retrieval_targets_match_displayed_history=True,
            no_future_oracle_in_rollouts=True,paid_calls=0,
            files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    result=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),extension=VERSION,post_primary=True,
                original_primary_and_summary_unchanged=True,freeze_sha256=sha(freeze),models=models,
                verifier_sha256=sha(Path(__file__)))
    dest=ROOT/"data/results/interface_memory_v1";dest.mkdir(parents=True,exist_ok=True)
    (dest/"integrity.json").write_text(json.dumps(result,indent=2))
    print(json.dumps([{k:r[k] for k in ["model","counts","unique_raw_calls"]} for r in models],indent=2))
if __name__=="__main__":main()
