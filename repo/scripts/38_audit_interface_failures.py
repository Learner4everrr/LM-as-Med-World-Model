#!/usr/bin/env python3
"""Post hoc first-failure descriptions; never repairs or resumes model trajectories."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json,math,hashlib,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"repo/src"))
from medwm_eval.submission_extension import object_parse
OUT=ROOT/"data/results/interface_memory_v1"
def loadl(p):return [json.loads(x) for x in p.read_text().splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def category(raw,variables):
    obj=object_parse(raw)
    if not isinstance(obj,dict):return "invalid_json_or_nonobject"
    try:values=[obj["next_state"][v]["value"] for v in variables]
    except (KeyError,TypeError):return "missing_schema_variable_or_value"
    if any(isinstance(x,bool) for x in values):return "boolean_point_value"
    if any(isinstance(x,str) for x in values):
        try:
            if all(isinstance(x,(str,int,float)) and math.isfinite(float(x)) for x in values):return "quoted_numeric_point_value"
        except (ValueError,TypeError,OverflowError):pass
    if any(not isinstance(x,(int,float)) or not math.isfinite(x) for x in values):return "other_nonnumeric_or_nonfinite_point"
    return "other"
def main():
    cases={}
    for folder in ["data/simulated/benchmark_v1","data/processed/vitaldb_v1"]:
        cases.update({r["id"]:r for r in loadl(ROOT/folder/"test.inputs.jsonl")})
    groups=[];examples=[];sources=[]
    for folder in sorted((ROOT/"data/runs/interface_memory_v1/test").iterdir()):
        if not folder.is_dir():continue
        raw={r["prompt_hash"]:r for r in loadl(folder/"raw.jsonl")}
        counts=defaultdict(Counter);seen=set()
        for run in loadl(folder/"predictions.jsonl"):
            bad=next((p for p in run["predictions"] if not p["valid"]),None)
            if bad is None:continue
            answer=raw[bad["prompt_hash"]]["raw"]
            kind=category(answer,cases[run["id"]]["payload"]["variables"])
            counts[run["domain"]][kind]+=1
            if run["domain"] not in seen:
                seen.add(run["domain"])
                examples.append(dict(model=run["model_id"],domain=run["domain"],id=run["id"],step=bad["step"],
                                     category=kind,raw=answer,prompt_hash=bad["prompt_hash"]))
        model=json.loads((folder/"manifest.json").read_text())["model_id"]
        for domain in ["simglucose","anesthesia","vitaldb"]:
            groups.append(dict(model=model,domain=domain,first_failures=sum(counts[domain].values()),categories={k:counts[domain][k] for k in ["invalid_json_or_nonobject","missing_schema_variable_or_value","quoted_numeric_point_value","boolean_point_value","other_nonnumeric_or_nonfinite_point","other"]}))
        sources += [folder/"raw.jsonl",folder/"predictions.jsonl"]
    base=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),post_hoc=True,no_prediction_repair=True,
              categories=groups,script_sha256=sha(Path(__file__)),
              protocol_sha256=sha(ROOT/"repo/docs/interface_failure_audit.md"),source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources})
    (OUT/"failure_audit_private.json").write_text(json.dumps(base|dict(fixed_order_examples=examples),indent=2))
    (OUT/"failure_summary.json").write_text(json.dumps(base|dict(fixed_order_synthetic_examples=[x for x in examples if x["domain"]!="vitaldb"]),indent=2))
    print(json.dumps(groups,indent=2))
if __name__=="__main__":main()
