#!/usr/bin/env python3
"""Seal completed primary-condition rollouts while other fixed conditions run."""
from pathlib import Path
from copy import deepcopy
import json,hashlib,sys
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"repo/src"))
from medwm_eval.prompts import make_prompt,parse_prediction,feedback
MODELS=["Qwen/Qwen3-8B","Qwen/Qwen2.5-7B-Instruct","mistralai/Mistral-7B-Instruct-v0.3","microsoft/Phi-3.5-mini-instruct"]
def load(p):return [json.loads(x) for x in p.read_text().splitlines()]
def seal(rows):return hashlib.sha256(json.dumps(sorted(rows,key=lambda x:x["id"]),sort_keys=True).encode()).hexdigest()
def main():
    freeze=json.loads((ROOT/"data/manifests/study_freeze.json").read_text())
    for path,value in freeze["frozen_hashes"].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==value
    cases={}
    for folder in ["data/simulated/benchmark_v1","data/processed/vitaldb_v1"]:
        cases.update({x["id"]:x for x in load(ROOT/folder/"test.inputs.jsonl")})
    reports=[]
    for model in MODELS:
        folder=ROOT/"data/runs/study_v1/test"/model.replace("/","--")
        rows=[r for r in load(folder/"predictions.jsonl") if r["condition"]=="state_action" and r["mode"]=="free"]
        assert len(rows)==len(cases) and {r["id"] for r in rows}==set(cases)
        wanted={p["prompt_hash"] for r in rows for p in r["predictions"]}
        cache={}
        with (folder/"raw.jsonl").open() as handle:
            for line in handle:
                try:r=json.loads(line)
                except json.JSONDecodeError:continue
                if r["prompt_hash"] in wanted:cache[r["prompt_hash"]]=r
                if len(cache)==len(wanted):break
        assert set(cache)==wanted
        valid=0
        for r in rows:
            c=cases[r["id"]];p=c["payload"];h=deepcopy(p["history"])
            assert 1<=len(r["predictions"])<=12
            for k,pred in enumerate(r["predictions"]):
                record=cache[pred["prompt_hash"]]
                assert pred["step"]==k+1
                assert make_prompt(c,h,k,"state_action")==record["prompt"]
                parsed=parse_prediction(record["raw"],len(p["variables"]))
                assert all(pred.get(key)==value for key,value in parsed.items())
                if not pred["valid"]:
                    assert k==len(r["predictions"])-1
                    break
                valid+=1;h=feedback(h,pred["values"],p["actions"][k],p["step_min"])
            else:assert len(r["predictions"])==12
        reports.append(dict(model=model,trajectories=len(rows),valid_points=valid,
                            self_fed_prompts_reconstructed=True,free_rows_sha256=seal(rows)))
    out=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),interim=True,
             scope="Completed state_action/free condition only; other fixed conditions continue; final full-run verification still required.",
             models=reports)
    (ROOT/"data/results/free_rollout_early_audit.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
if __name__=="__main__":main()
