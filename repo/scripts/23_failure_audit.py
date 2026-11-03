#!/usr/bin/env python3
"""Descriptive failure taxonomy; never repairs outputs or changes scored rollouts."""
from pathlib import Path
from collections import Counter
import json,sys,hashlib,importlib.util
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"repo/src"))
spec=importlib.util.spec_from_file_location("reports",ROOT/"repo/scripts/20_additional_reports.py")
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)
def classify(raw,n):
    text=raw.strip()
    if text.startswith(chr(96)*3):
        text=text.split("\n",1)[-1]
        if text.rstrip().endswith(chr(96)*3):text=text.rstrip()[:-3].strip()
    try:obj=json.loads(text)
    except (ValueError,TypeError):
        rest=text;pieces=[]
        try:
            while rest:
                obj,end=json.JSONDecoder().raw_decode(rest)
                pieces.append(obj);rest=rest[end:].strip()
            if len(pieces)>1 and all(isinstance(p,dict) for p in pieces):return "multiple_objects"
        except (ValueError,TypeError):pass
        return "malformed_json"
    if not isinstance(obj,dict) or "values" not in obj:return "missing_values"
    vals=obj["values"]
    if n==1 and isinstance(vals,(float,int)) and not isinstance(vals,bool):vals=[vals]
    if not isinstance(vals,list) or len(vals)!=n:return "wrong_cardinality"
    return "invalid_numeric_values"
def main():
    dest=ROOT/"data/results/study_v1"
    assert (dest/"integrity.json").exists()
    cases={}
    for folder in ["data/simulated/benchmark_v1","data/processed/vitaldb_v1"]:
        for line in (ROOT/folder/"test.inputs.jsonl").read_text().splitlines():
            c=json.loads(line);cases[c["id"]]=c
    counts=Counter();examples=[];cost=[]
    for model in report.MODELS:
        folder=ROOT/"data/runs/study_v1/test"/model.replace("/","--")
        raw={r["prompt_hash"]:r for r in [json.loads(x) for x in (folder/"raw.jsonl").read_text().splitlines()]}
        runs=[json.loads(x) for x in (folder/"predictions.jsonl").read_text().splitlines()]
        for run in runs:
            c=cases[run["id"]];n=len(c["payload"]["variables"])
            for p in run["predictions"]:
                category="valid" if p["valid"] else classify(raw[p["prompt_hash"]]["raw"],n)
                counts[(model,run["domain"],run["condition"],run["mode"],category)]+=1
                if not p["valid"] and run["domain"]=="anesthesia" and run["condition"]=="state_action" and run["mode"]=="free" and not any(x["model"]==model for x in examples):
                    examples.append(dict(model=model,case=run["id"],step=p["step"],category=category,raw=raw[p["prompt_hash"]]["raw"]))
        complete=sum(r["condition"]=="state_action" and r["mode"]=="free" and len(r["predictions"])==12 and all(p["valid"] for p in r["predictions"]) for r in runs)
        integrity=next(x for x in json.loads((dest/"integrity.json").read_text())["models"] if x["model"]==model)
        cost.append(dict(model=model,complete_state_action_free_trajectories=complete,
                         total_protocol_generation_seconds=integrity["generation_seconds"],
                         amortized_generation_seconds_per_complete=integrity["generation_seconds"]/complete if complete else None))
    records=[dict(zip(["model","domain","condition","mode","category"],key),count=value) for key,value in counts.items()]
    rows=[]
    for m in report.MODELS:
        for cond,mode in [("state_action","free"),("state_action","teacher"),("state_only","free")]:
            nums=[sum(v for (model,d,c,mo,cat),v in counts.items() if model==m and c==cond and mo==mode and cat==category) for category in ["malformed_json","multiple_objects","wrong_cardinality","missing_values","invalid_numeric_values"]]
            rows.append([report.NAMES[m],("SA" if cond=="state_action" else "S")+("/FR" if mode=="free" else "/TF"),*nums[:3],nums[3]+nums[4],sum(nums)])
    report.table("failures",["Model","Condition","JSON","Multi","Length","Other","Total"],rows)
    report.table("amortized_compute",["Model","Complete SA/FR","All-protocol gen h","Seconds/complete"],
                 [[report.NAMES[r["model"]],r["complete_state_action_free_trajectories"],f'{r["total_protocol_generation_seconds"]/3600:.2f}',f'{r["amortized_generation_seconds_per_complete"]:.1f}'] for r in cost])
    (dest/"failure_taxonomy.json").write_text(json.dumps(dict(posthoc_descriptive=True,does_not_repair_predictions=True,counts=records,first_PAS_failures=examples,compute=cost,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        output_tables={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in report.GENERATED}),indent=2))
if __name__=="__main__":main()
