"""Predeclared single-step robustness probes and one-shot horizon control."""
from copy import deepcopy
import json,hashlib
import numpy as np
from .prompts import make_prompt,parse_prediction

VARIANTS=["missing_previous","noisy_history","time_shift_60","pretty_json"]
def diagnostic_prompt(case,variant):
    c=deepcopy(case);p=c["payload"];h=p["history"]
    if variant=="missing_previous":del h[-2]
    if variant=="noisy_history":
        seed=int(hashlib.sha256((case["id"]+":noise").encode()).hexdigest()[:8],16)
        rng=np.random.default_rng(seed)
        sd=[5.] if case["domain"]=="simglucose" else [2.,3.]
        for row in h:row["values"]=(np.array(row["values"])+rng.normal(0,sd)).tolist()
    text=make_prompt(c,h,0)
    split=text.index('{"variables"')
    instruction=text[:split];view,end=json.JSONDecoder().raw_decode(text[split:]);suffix=text[split+end:]
    if variant=="time_shift_60":
        for row in view["history_rows"]:row[0]+=60
        for row in view.get("scheduled_actions",[]):row[0]+=60
        view["predict_time_min"]+=60
    return instruction+json.dumps(view,indent=2 if variant=="pretty_json" else None,separators=None if variant=="pretty_json" else (",",":"))+suffix
def oneshot_prompt(case):
    text=make_prompt(case,case["payload"]["history"],0)
    view,_=json.JSONDecoder().raw_decode(text[text.index('{"variables"'):])
    view.pop("predict_time_min")
    view["predict_times_min"]=[i*case["payload"]["step_min"] for i in [1,3,6,12]]
    return ("Predict the observations at ALL four requested times from this history and the fixed "
            "action schedule, as a patient-dynamics simulator. Do not choose treatment. Return only "
            "JSON with one key values: an array of exactly four numeric arrays, one per requested "
            "time, each in the listed variable order. Include every variable. Use the listed units.\n"
            +json.dumps(view,separators=(",",":")))
def parse_oneshot(raw,n):
    try:
        obj=json.loads(raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
        y=np.array(obj["values"],float)
        if y.shape!=(4,n) or not np.isfinite(y).all():raise ValueError()
        return {"valid":True,"values":y.tolist()}
    except (ValueError,TypeError,KeyError):
        return {"valid":False,"values":None}
def run_diagnostics(engine,cases,out):
    selected={}
    for case in cases:selected.setdefault((case["domain"],case["patient"]),case)
    tasks=[]
    for case in selected.values():
        for variant in VARIANTS:tasks.append((case,variant,diagnostic_prompt(case,variant)))
        if case["domain"]!="vitaldb":tasks.append((case,"one_shot",oneshot_prompt(case)))
    records=[]
    for (case,variant,_),r in zip(tasks,engine.generate([x[2] for x in tasks])):
        result=(parse_oneshot if variant=="one_shot" else parse_prediction)(r["raw"],len(case["payload"]["variables"]))
        records.append({"id":case["id"],"domain":case["domain"],"patient":case["patient"],
                        "variant":variant,"result":result,"prompt_hash":r["prompt_hash"]})
    (out/"diagnostics.json").write_text(json.dumps(records,indent=2))
