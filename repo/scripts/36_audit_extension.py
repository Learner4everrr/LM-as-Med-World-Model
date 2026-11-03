#!/usr/bin/env python3
"""Independent extension check from raw prediction arrays and displayed rows."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import json,hashlib,math,re
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"data/results/interface_memory_v1"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lines(p):return [json.loads(x) for x in p.read_text().splitlines()]
def estimate(rows):
    rows=[(c,v) for c,v in rows if math.isfinite(v)]
    clusters=sorted({c for c,v in rows})
    means=np.asarray([np.mean([v for c,v in rows if c==patient]) for patient in clusters])
    if not len(means):return dict(mean=None,lo=None,hi=None,clusters=0,n=0)
    random=np.random.default_rng(20261102)
    boot=means[random.integers(len(means),size=(2000,len(means)))].mean(axis=1)
    return dict(mean=float(means.mean()),lo=float(np.quantile(boot,.025)),hi=float(np.quantile(boot,.975)),clusters=len(means),n=len(rows))
def same(result,reference):
    for k,v in result.items():
        assert (v is None and reference[k] is None) or v is not None and abs(v-reference[k])<1e-10,(k,v,reference)
def main():
    summary=json.loads((OUT/"summary.json").read_text())
    def row(**keys):
        matches=[r for r in summary if all(r.get(k)==v for k,v in keys.items())]
        assert len(matches)==1,keys
        return matches[0]
    cases={};oracles={}
    for directory in ["data/simulated/benchmark_v1","data/processed/vitaldb_v1"]:
        cases.update({r["id"]:r for r in lines(ROOT/directory/"test.inputs.jsonl")})
        oracles.update({r["id"]:np.array(r["targets"]) for r in lines(ROOT/directory/"test.oracles.jsonl")})
    groups=defaultdict(dict)
    for c in cases.values():
        if c["domain"]!="vitaldb":groups[c["group"]][c["branch"]]=c
    verified=[];sources=[]
    for folder in sorted((ROOT/"data/runs/interface_memory_v1/test").iterdir()):
        if not folder.is_dir():continue
        manifest=json.loads((folder/"manifest.json").read_text());model=manifest["model_id"]
        predpath=folder/"predictions.jsonl";rawpath=folder/"raw.jsonl";sources += [predpath,rawpath,folder/"memory.json"]
        def arrays(path,original=False):
            result={}
            for r in lines(path):
                if original and (r["condition"]!="state_action" or r["mode"]!="free"):continue
                x=np.full_like(oracles[r["id"]],np.nan,dtype=float)
                for step in r["predictions"]:
                    if step["valid"]:x[step["step"]-1]=step["values"]
                result[r["id"]]=x
            return result
        new=arrays(predpath)
        oldpath=ROOT/"data/runs/study_v1/test"/folder.name/"predictions.jsonl";sources.append(oldpath)
        old=arrays(oldpath,True)
        comparisons=defaultdict(list)
        for branches in groups.values():
            a,b=branches["A"],branches["B"]
            reference=oracles[a["id"]]-oracles[b["id"]]
            newdiff=new[a["id"]]-new[b["id"]];olddiff=old[a["id"]]-old[b["id"]]
            complete=np.isfinite(newdiff).all();oldcomplete=np.isfinite(olddiff).all()
            for j,var in enumerate(a["payload"]["variables"]):
                key=(a["domain"],var,a["kind"]);patient=a["patient"]
                newerror=float(np.abs(newdiff[:,j]-reference[:,j]).mean()) if complete else float("nan")
                olderror=float(np.abs(olddiff[:,j]-reference[:,j]).mean()) if oldcomplete else float("nan")
                values=dict(complete=float(complete),effect_mae=newerror,
                            delta_zero=newerror-float(np.abs(reference[:,j]).mean()),
                            error_difference=newerror-olderror,completion_difference=float(complete)-float(oldcomplete))
                for metric,value in values.items():comparisons[key+(metric,)].append((patient,value))
        for (domain,var,kind,metric),values in comparisons.items():
            typ="interface_paired" if metric.endswith("difference") else "paired"
            calc=estimate(values);same(calc,row(type=typ,metric=metric,model=model,domain=domain,variable=var,kind=kind))
            verified.append(dict(model=model,type=typ,domain=domain,variable=var,kind=kind,metric=metric,**calc))
        raws={r["prompt_hash"]:r for r in lines(rawpath)}
        memory=defaultdict(list)
        for r in json.loads((folder/"memory.json").read_text()):
            prompt=raws[r["prompt_hash"]]["prompt"]
            displayed=json.JSONDecoder().raw_decode(prompt[prompt.index('{"variables"'):])[0]
            h=displayed["history"];query=next(x for x in h if x["time_min"]==displayed["query_time_min"])
            target=dict(current_time=h[-1]["time_min"],current_values=h[-1]["values"],last_action=h[-1]["past_action"],
                        query_time=query["time_min"],query_action=query["past_action"],last_action_interval_start=h[-2]["time_min"])
            assert target==r["target"]
            for parser in ["strict","content"]:
                text=r["raw"].strip()
                try:
                    if parser=="content":obj=json.JSONDecoder().raw_decode(text[text.index("{"):])[0]
                    else:
                        if text.startswith(chr(96)*3):
                            text=text.split("\n",1)[-1]
                            if text.rstrip().endswith(chr(96)*3):text=text.rstrip()[:-3].strip()
                        obj=json.loads(text)
                except (ValueError,TypeError):obj=None
                scores={}
                for key,want in target.items():
                    tol=1e-4 if key=="current_values" else 1e-6 if key in ["last_action","query_action"] else 1e-8
                    got=obj.get(key) if isinstance(obj,dict) else None
                    if isinstance(want,list):
                        okay=isinstance(got,list) and len(got)==len(want)
                        pairs=zip(got,want) if okay else []
                    else:okay=True;pairs=[(got,want)]
                    scores[key]=okay and all(isinstance(a,(int,float)) and not isinstance(a,bool) and math.isfinite(a) and abs(a-b)<=tol for a,b in pairs)
                fields=dict(json_object=float(isinstance(obj,dict)),state=float(scores["current_values"]),
                            temporal=float(all(scores[x] for x in ["current_time","query_time","last_action_interval_start"])),
                            current_action=float(scores["last_action"]),past_action=float(scores["query_action"]),
                            all_fields=float(all(scores.values())))
                for metric,value in fields.items():memory[(r["domain"],r["anchor_step"],parser,metric)].append((r["patient"],value))
        for (domain,anchor,parser,metric),values in memory.items():
            calc=estimate(values);same(calc,row(type="retrieval",metric=metric,model=model,domain=domain,anchor=anchor,parser=parser))
            verified.append(dict(model=model,type="retrieval",domain=domain,anchor=anchor,parser=parser,metric=metric,**calc))
    sources += [OUT/"summary.json",OUT/"integrity.json",ROOT/"data/manifests/interface_memory_freeze.json"]
    result=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),all_matched=True,
                independent_of_project_metric_and_prompt_modules=True,verified_statistics=len(verified),
                statistics=verified,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources},
                auditor_sha256=sha(Path(__file__)))
    (OUT/"independent_audit.json").write_text(json.dumps(result,indent=2))
    print("Independent extension audit:",len(verified),"statistics matched")
if __name__=="__main__":main()
