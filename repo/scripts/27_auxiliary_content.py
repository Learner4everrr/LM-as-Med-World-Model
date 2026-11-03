#!/usr/bin/env python3
"""Post hoc auxiliary content audit; no change to transition scores or rollouts."""
from pathlib import Path
import json,sys,hashlib
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"repo/src"))
from medwm_eval.metrics import cluster_ci
MODELS=["Qwen/Qwen3-8B","Qwen/Qwen2.5-7B-Instruct","mistralai/Mistral-7B-Instruct-v0.3","microsoft/Phi-3.5-mini-instruct"]
def main():
    assert (ROOT/"data/results/study_v1/integrity.json").exists()
    rows=[]
    for model in MODELS:
        p=ROOT/"data/runs/study_v1/test"/model.replace("/","--")/"auxiliary.json"
        for r in json.loads(p.read_text()):
            obj=None;state=False;knowledge=False;decodable=False
            start=r["raw"].find("{")
            if start>=0:
                try:
                    obj,end=json.JSONDecoder().raw_decode(r["raw"][start:])
                    decodable=isinstance(obj,dict)
                except (ValueError,TypeError):pass
            if decodable:
                knowledge=obj.get("direction")==r["expected_direction"]
                try:
                    x=np.asarray(obj["values"]);expected=np.asarray(r["expected_values"])
                    numeric=x.dtype.kind in "fiu" and np.isfinite(x).all()
                    t=obj.get("time_min")
                    state=bool(numeric and x.shape==expected.shape and np.allclose(x,expected,rtol=0,atol=1e-4)
                               and isinstance(t,(int,float)) and not isinstance(t,bool) and t==0)
                except (ValueError,TypeError,KeyError):pass
            rows.append(dict(model=model,domain=r["domain"],patient=r["patient"],id=r["id"],
                             prefix_decodable=float(decodable),state_content=float(state),knowledge_content=float(knowledge)))
    summary=[]
    for model in MODELS:
        for domain in ["simglucose","anesthesia","vitaldb"]:
            g=[r for r in rows if r["model"]==model and r["domain"]==domain]
            for metric in ["prefix_decodable","state_content","knowledge_content"]:
                summary.append(dict(model=model,domain=domain,metric=metric,
                     **cluster_ci([r[metric] for r in g],[r["patient"] for r in g])))
    dest=ROOT/"data/results/study_v1"
    (dest/"auxiliary_content_summary.json").write_text(json.dumps(summary,indent=2))
    (dest/"auxiliary_content_rows.json").write_text(json.dumps(rows,indent=2))
    (dest/"auxiliary_content_manifest.json").write_text(json.dumps(dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        posthoc=True,transition_scores_unchanged=True,
        rule="Decode only the first JSON object beginning at the first opening brace; ignore trailing text; never repair values, search later objects or rescore rollouts.",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
