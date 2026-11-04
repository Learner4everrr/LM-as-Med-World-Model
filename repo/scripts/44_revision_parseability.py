#!/usr/bin/env python3
"""Post hoc parseability of saved simulator responses; never edits a rollout."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,sys,math,zipfile
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"repo/src"))
from medwm_eval.prompts import parse_prediction
from medwm_eval.submission_extension import parse_structured,object_parse
MODELS=["Qwen/Qwen3-8B","Qwen/Qwen2.5-7B-Instruct","mistralai/Mistral-7B-Instruct-v0.3","microsoft/Phi-3.5-mini-instruct"]
def loadl(p):return [json.loads(l) for l in p.read_text().splitlines()]
def classify(raw,variables,named):
    obj=object_parse(raw)
    if not isinstance(obj,dict):
        first=object_parse(raw,prefix=True)
        return "multiple_objects_or_extra_text" if isinstance(first,dict) else "invalid_json_or_nonobject"
    try:
        vals=[obj["next_state"][v]["value"] for v in variables] if named else obj["values"]
    except (KeyError,TypeError):return "missing_field"
    if not named and len(variables)==1 and isinstance(vals,(int,float)) and not isinstance(vals,bool):vals=[vals]
    if not isinstance(vals,list) or len(vals)!=len(variables):return "array_length_or_type"
    if any(isinstance(x,bool) for x in vals):return "boolean_value"
    if any(isinstance(x,str) for x in vals):return "quoted_or_nonnumeric_string"
    if any(not isinstance(x,(int,float)) or not math.isfinite(x) for x in vals):return "nonnumeric_or_nonfinite"
    return "other_point_failure"
def main():
    cases={r["id"]:r for r in loadl(ROOT/"data/simulated/benchmark_v1/test.inputs.jsonl")}
    summaries=[];sources={}
    def read(study,model,name):
        p=ROOT/"data/runs"/study/"test"/model.replace("/","--")/name
        if p.exists():
            sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest();return loadl(p)
        p=ROOT/"data/releases/synthetic_outputs.zip"
        sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
        with zipfile.ZipFile(p) as z:return [json.loads(l) for l in z.read(study+"/"+model.replace("/","--")+"/"+name).splitlines()]
    for study in ["study_v1","interface_memory_v1"]:
        named=study=="interface_memory_v1"
        condition="state_action_named" if named else "state_action"
        for model in MODELS:
            runs=read(study,model,"predictions.jsonl")
            needed={p["prompt_hash"] for r in runs if r["id"] in cases and r["condition"]==condition and r["mode"]=="free" for p in r["predictions"]}
            raw={r["prompt_hash"]:r["raw"] for r in read(study,model,"raw.jsonl") if r["prompt_hash"] in needed}
            for domain in ["simglucose","anesthesia"]:
                count=Counter();categories=Counter()
                selected=[r for r in runs if r["domain"]==domain and r["condition"]==condition and r["mode"]=="free"]
                assert len(selected)==270
                for r in selected:
                    variables=cases[r["id"]]["payload"]["variables"]
                    parse=(lambda text:parse_structured(text,variables)) if named else (lambda text:parse_prediction(text,len(variables)))
                    bad_seen=False
                    for p in r["predictions"]:
                        answer=raw[p["prompt_hash"]];strict=parse(answer)
                        assert strict["valid"]==p["valid"] and strict["interval_valid"]==p["interval_valid"]
                        obj=object_parse(answer,prefix=True)
                        alt=parse(json.dumps(obj)) if isinstance(obj,dict) else dict(valid=False)
                        count["issued_positions"]+=1
                        count["strict_valid_points"]+=int(strict["valid"])
                        count["first_object_valid_points"]+=int(alt["valid"])
                        if strict["valid"]:
                            assert strict["values"]==p["values"]
                            count["valid_points_invalid_interval"]+=int(not strict["interval_valid"])
                            count["valid_points_out_of_elementary_bounds"]+=int(any((v<=0 if var in ["CGM","MAP"] else not 0<=v<=100) for var,v in zip(variables,strict["values"])))
                        elif not bad_seen:
                            bad_seen=True;count["first_failures"]+=1
                            count["first_failures_with_readable_point"]+=int(alt["valid"])
                            categories[classify(answer,variables,named)]+=1
                    count["strict_complete_trajectories"]+=int(len(r["predictions"])==12 and all(p["valid"] for p in r["predictions"]))
                assert sum(categories.values())==count["first_failures"]
                summaries.append(dict(study=study,model=model,domain=domain,scheduled_trajectories=270,
                    counts={k:count[k] for k in ["issued_positions","strict_valid_points","first_object_valid_points","valid_points_invalid_interval","valid_points_out_of_elementary_bounds","first_failures","first_failures_with_readable_point","strict_complete_trajectories"]},first_failure_categories=dict(categories)))
    out=ROOT/"data/results/revision_v1"
    out.mkdir(parents=True,exist_ok=True)
    result=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),analysis_status="post_hoc_revision",
        scope="Two simulated domains only; original and named interfaces",
        no_prediction_repair=True,no_regenerated_continuations=True,
        per_response_truncation_not_identifiable_from_padded_token_slots=True,
        summaries=summaries,source_hashes=sources,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/"parseability.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(summaries,indent=2))
if __name__=="__main__":main()
