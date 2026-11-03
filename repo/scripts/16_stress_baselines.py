#!/usr/bin/env python3
"""Reuse the original development-trained controls, without stress-data fitting."""
from pathlib import Path
import sys,json,pickle,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"repo/src"))
from medwm_eval.baselines_revised import predict
from medwm_eval.prompts import feedback
def loadl(p):return [json.loads(l) for l in p.read_text().splitlines()]
def main():
    folder=ROOT/"data/simulated/hypoglycemia_stress_v1"
    cases=loadl(folder/"test.inputs.jsonl");truth={x["id"]:x["targets"] for x in loadl(folder/"test.oracles.jsonl")}
    source=ROOT/"data/runs/study_v1/baselines"
    models=pickle.loads((source/"simglucose.pkl").read_bytes())
    out=ROOT/"data/runs/stress_v1/baselines";out.mkdir(parents=True,exist_ok=True);rows=[]
    for name in ["persistence","trend","ridge_history","ridge_action","boosted_action"]:
        for mode in ["free","teacher"]:
            for c in cases:
                p=c["payload"];h=p["history"];pred=[]
                for step,target in enumerate(truth[c["id"]]):
                    values=predict(name,models,p,h,step);valid=bool(np.isfinite(values).all())
                    pred.append(dict(step=step+1,values=values if valid else None,valid=valid,interval_valid=False))
                    if mode=="free" and not valid:break
                    h=feedback(h,values if mode=="free" else target,p["actions"][step],30)
                rows.append({k:c[k] for k in ["id","group","patient","domain","kind","branch","scenario","split"]}|
                     dict(model_id=name,condition="state_only" if name=="ridge_history" else "state_action",mode=mode,predictions=pred))
    (out/"predictions.jsonl").write_text("".join(json.dumps(x,allow_nan=False)+"\n" for x in rows))
    (out/"manifest.json").write_text(json.dumps(dict(no_refitting=True,source_model_sha256=hashlib.sha256((source/"simglucose.pkl").read_bytes()).hexdigest()),indent=2))
    print("stress baseline trajectories",len(rows))
if __name__=="__main__":main()
