#!/usr/bin/env python3
"""Descriptive decomposition of the already scored retrieval-time endpoint."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,sys,importlib.util
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"repo/src"))
from medwm_eval.submission_extension import object_parse,numeric
from medwm_eval.metrics import cluster_ci
spec=importlib.util.spec_from_file_location("extension_analysis",ROOT/"repo/scripts/31_analyze_extension.py")
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
OUT=ROOT/"data/results/interface_memory_v1"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    records=[];sources=[]
    for model in mod.MODELS:
        p=ROOT/"data/runs/interface_memory_v1/test"/model.replace("/","--")/"memory.json";sources.append(p)
        for r in json.loads(p.read_text()):
            obj=object_parse(r["raw"],True)
            scores={}
            for key in ["current_time","query_time","last_action_interval_start"]:
                value=obj.get(key) if isinstance(obj,dict) else None
                scores[key]=float(numeric(value) and abs(value-r["target"][key])<=1e-8)
            records.append(dict(model=model,domain=r["domain"],anchor=r["anchor_step"],patient=r["patient"])|scores)
    df=pd.DataFrame(records);df["temporal"]=df[["current_time","query_time","last_action_interval_start"]].all(axis=1).astype(float)
    original=json.loads((OUT/"summary.json").read_text());summary=[]
    for (model,domain,anchor),g in df.groupby(["model","domain","anchor"]):
        for metric in ["current_time","query_time","last_action_interval_start","temporal"]:
            stats=cluster_ci(g[metric],g.patient)
            if metric=="temporal":
                ref=next(r for r in original if all(r.get(k)==v for k,v in dict(type="retrieval",model=model,domain=domain,anchor=anchor,parser="content",metric="temporal").items()))
                assert all(stats[k]==ref[k] for k in stats)
            summary.append(dict(model=model,domain=domain,anchor=int(anchor),metric=metric,**stats))
    rows=[]
    for domain in ["simglucose","anesthesia","vitaldb"]:
        for anchor in [0,6]:
            for model in mod.MODELS:
                vals=[next(r for r in summary if (r["model"],r["domain"],r["anchor"],r["metric"])==(model,domain,anchor,metric)) for metric in ["current_time","query_time","last_action_interval_start"]]
                rows.append([mod.DOM[domain],mod.NAMES[model],str(anchor)]+[mod.fmt(r,True) for r in vals])
    mod.table("retrieval_time_components",["Domain","Model","Anchor",r"Current time (\%)",r"Query time (\%)",r"Interval start (\%)"],rows)
    report=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),post_hoc=True,original_temporal_reproduced=True,
                summary=summary,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources},
                script_sha256=sha(Path(__file__)),protocol_sha256=sha(ROOT/"repo/docs/retrieval_time_audit.md"),
                table_sha256=sha(ROOT/"paper/tables/retrieval_time_components.tex"))
    (OUT/"time_components.json").write_text(json.dumps(report,indent=2))
    print("Temporal decomposition:",len(summary),"statistics; original conjunction unchanged")
if __name__=="__main__":main()
