#!/usr/bin/env python3
"""Verify the analytical action-insensitive persistence feedback control."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"repo/src"))
from medwm_eval.metrics import cluster_ci
def load(p):return [json.loads(l) for l in p.read_text().splitlines()]
def main():
    cases={c["id"]:c for c in load(ROOT/"data/simulated/benchmark_v1/test.inputs.jsonl")}
    truth={c["id"]:np.array(c["targets"]) for c in load(ROOT/"data/simulated/benchmark_v1/test.oracles.jsonl")}
    runs={(r["id"],r["mode"]):r for r in load(ROOT/"data/runs/study_v1/baselines/predictions.jsonl") if r["model_id"]=="persistence" and r["id"] in cases}
    groups={}
    for c in cases.values():
        if c["kind"]=="dose":groups.setdefault(c["group"],{})[c["branch"]]=c
    rows=[]
    for group,branches in groups.items():
        a,b=branches["A"],branches["B"];delta=truth[a["id"]]-truth[b["id"]]
        for mode in ["free","teacher"]:
            pa=np.array([x["values"] for x in runs[a["id"],mode]["predictions"]])
            pb=np.array([x["values"] for x in runs[b["id"],mode]["predictions"]])
            expected=np.zeros_like(delta) if mode=="free" else np.concatenate([np.zeros_like(delta[:1]),delta[:-1]])
            np.testing.assert_array_equal(pa-pb,expected)
            err=np.abs(expected-delta).mean(0)
            for j,v in enumerate(a["payload"]["variables"]):
                rows.append(dict(domain=a["domain"],variable=v,mode=mode,patient=a["patient"],error=float(err[j])))
    reports=[]
    for d,v in [("simglucose","CGM"),("anesthesia","BIS"),("anesthesia","MAP")]:
        for mode in ["free","teacher"]:
            g=[r for r in rows if r["domain"]==d and r["variable"]==v and r["mode"]==mode]
            reports.append(dict(domain=d,variable=v,mode=mode,**cluster_ci([r["error"] for r in g],[r["patient"] for r in g])))
    out=dict(exact_prediction_identity_verified=True,groups=len(groups),
             identity="Self-fed persistence predicts zero paired effect; teacher-forced persistence predicts the previous reference paired effect.",
             results=reports)
    (ROOT/"data/results/persistence_control_identity.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
if __name__=="__main__":main()
