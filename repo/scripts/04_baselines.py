#!/usr/bin/env python3
"""Fit fixed baselines on development only; evaluate held-out identities."""
from pathlib import Path
import sys,json,pickle,hashlib
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"repo/src"))
from medwm_eval.baselines_revised import fit_controls,predict
from medwm_eval.prompts import feedback
import numpy as np
def load(split):
    cases=[];truth={}
    for folder in ["data/simulated/benchmark_v1","data/processed/vitaldb_v1"]:
        cases += [json.loads(x) for x in (ROOT/folder/(split+".inputs.jsonl")).read_text().splitlines()]
        truth.update({x["id"]:x["targets"] for x in [json.loads(l) for l in (ROOT/folder/(split+".oracles.jsonl")).read_text().splitlines()]})
    return cases,truth
def main():
    dev,train=load("development");test,truth=load("test")
    for d in {x["domain"] for x in dev}:
        assert not ({x["patient"] for x in dev if x["domain"]==d}&{x["patient"] for x in test if x["domain"]==d})
    out=ROOT/"data/runs/study_v1/baselines";out.mkdir(parents=True,exist_ok=True)
    rows=[];audits=[]
    for domain in sorted({x["domain"] for x in test}):
        models,audit=fit_controls(dev,train,domain);audits.append(audit)
        (out/(domain+".pkl")).write_bytes(pickle.dumps(models))
        for name in ["persistence","trend","ridge_history","ridge_action","boosted_action"]:
            for mode in ["free","teacher"]:
                for case in [x for x in test if x["domain"]==domain]:
                    p=case["payload"];h=p["history"];pred=[]
                    for step,target in enumerate(truth[case["id"]]):
                        values=predict(name,models,p,h,step)
                        valid=bool(np.isfinite(values).all())
                        pred.append(dict(step=step+1,values=values if valid else None,valid=valid,interval_valid=False))
                        if mode=="free" and not valid:break
                        h=feedback(h,values if mode=="free" else target,p["actions"][step],p["step_min"])
                    rows.append({k:case[k] for k in ["id","group","patient","domain","kind","branch","scenario","split"]}|
                        dict(model_id=name,mode=mode,condition="state_only" if name=="ridge_history" else "state_action",predictions=pred))
            print(domain,name,flush=True)
    (out/"predictions.jsonl").write_text("".join(json.dumps(x,allow_nan=False)+"\n" for x in rows))
    (out/"manifest.json").write_text(json.dumps(dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
         fit_audit=audits,source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/"repo/src/medwm_eval/baselines.py",ROOT/"repo/src/medwm_eval/baselines_revised.py"]},predictions_sha256=hashlib.sha256((out/"predictions.jsonl").read_bytes()).hexdigest()),indent=2))
if __name__=="__main__":main()
