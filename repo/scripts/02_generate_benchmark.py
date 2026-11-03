#!/usr/bin/env python3
"""Generate frozen simulator inputs and separate oracles; never calls an LLM."""
from pathlib import Path
from copy import deepcopy
import os,sys,json,hashlib,traceback,argparse
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLBACKEND","Agg")
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"data/cache/matplotlib"))
sys.path.insert(0,str(ROOT/"repo/src"))
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from medwm_eval.envs import simglucose_adapter as glu
from medwm_eval.envs import anesthesia as ane

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def roundlist(x):
    return [round(float(v),6) for v in x]

def split_for(domain,index):
    return "development" if (index<=4 if domain=="simglucose" else index<12) else "test"

def setup(domain,index,scenario):
    seed=20261000+index*100+scenario
    if domain=="simglucose":
        group,index=index//100,index%100
        name=["adult","adolescent","child"][group]+f"#{index:03d}"
        meal=[0,15,30][scenario]
        # Meals are always announced at macro-step boundaries.
        recipe=glu.Recipe(patient=name,seed=seed,warmup_min=180,horizon_min=360,
                          meals=((60,meal),(210,meal),(390,meal)))
        env=glu.build(recipe)
        basal=glu.reference_rate(env)
        past_rate=basal*[.8,1,1.2][scenario]
        hist=[{"time":-180,"values":roundlist([glu.point(env)["glucose_mg_dL"]]),
               "past_action":[past_rate],"past_meal_g":0}]
        for j in range(6):
            rows=glu.advance(env,past_rate,30)
            hist.append({"time":(j+1)*30-180,"values":roundlist([rows[-1]["glucose_mg_dL"]]),
                         "past_action":roundlist([rows[-1]["applied_insulin_U_min"]]),
                         "past_meal_g":sum(r["announced_meal_g"] for r in rows)})
        return env,hist,[basal],{"patient":name,"seed":seed,"scenario":scenario,"recipe":recipe.to_dict()},{}
    rng=np.random.default_rng(817+index)
    profile=[int(rng.integers(25,76)),int(rng.integers(155,186)),
             int(rng.integers(50,101)),int(index%2)]
    env=ane.build_patient(profile,seed)
    base=[profile[2]*.08/60,profile[2]*.1/60]
    past=[base[0]*[.75,1,1.25][scenario],base[1]]
    hist=[{"time":-20,"values":roundlist(ane.observe(env)),"past_action":[0,0],"past_meal_g":0}]
    for j in range(10):
        y=ane.advance(env,past)
        hist.append({"time":(j+1)*2-20,"values":roundlist(y),"past_action":roundlist(past),"past_meal_g":0})
    return env,hist,base,{"patient":f"pas_{index:02d}","seed":seed,"scenario":scenario,"profile":profile},dict(
        age_years=profile[0],height_cm=profile[1],weight_kg=profile[2],sex="male" if profile[3] else "female")

def plan(domain,base,kind,branch,scenario):
    dt=30 if domain=="simglucose" else 2
    actions=[]
    for step in range(12):
        if kind=="dose":
            factor={"A":.75,"R":1.,"B":1.25}[branch]
        else:
            on=0<=step<3 if branch=="A" else 3<=step<6
            factor=1.25 if on else 1.
        vals=[base[0]*factor]+base[1:]
        actions.append({"time":step*dt,"values":vals,
                        "meal_g":[0,15,30][scenario] if domain=="simglucose" and step in [1,7] else 0})
    return actions

def simulate(domain,env,actions):
    labels=[]
    for a in actions:
        if domain=="simglucose":
            rows=glu.advance(env,a["values"][0],30)
            y=[rows[-1]["glucose_mg_dL"]]
            assert abs(sum(r["announced_meal_g"] for r in rows)-a["meal_g"])<1e-9
        else:
            y=ane.advance(env,a["values"])
        labels.append([float(v) for v in y])
    return labels

def build_cases(domain,index,scenario):
    anchor,hist,base,meta,demo=setup(domain,index,scenario)
    state=glu.fingerprint(anchor) if domain=="simglucose" else ane.hidden_state(anchor).copy()
    output=[]
    for kind in ["dose","timing"]:
        groupid=glu.digest({"domain":domain,**meta,"kind":kind})[:24]
        branches=["A","R","B"] if kind=="dose" else ["A","B"]
        exogenous=[]
        for branch in branches:
            actions=plan(domain,base,kind,branch,scenario)
            if domain=="simglucose":
                for a in actions:
                    a["values"]=[glu.validate_rate(anchor,a["values"][0])]
            env=deepcopy(anchor)
            if domain=="simglucose":
                assert glu.fingerprint(env)==state
            else:
                np.testing.assert_array_equal(ane.hidden_state(env),state)
            y=simulate(domain,env,actions)
            # Full replay of every branch; label generation is gated by equality.
            assert simulate(domain,deepcopy(anchor),actions)==y
            if domain=="simglucose":
                exogenous.append(glu.future_noise(env,25))
            case={"id":groupid+"_"+branch,"group":groupid,"patient":meta["patient"],
                  "split":split_for(domain,index%100 if domain=="simglucose" else index),
                  "domain":domain,"kind":kind,"branch":branch,"scenario":scenario,
                  "payload":{"variables":["CGM"] if domain=="simglucose" else ["BIS","MAP"],
                     "units":["mg/dL"] if domain=="simglucose" else ["unitless","mmHg"],
                     "action_names":["basal_insulin_U_min"] if domain=="simglucose" else ["propofol_mg_s","remifentanil_ug_s"],
                     "step_min":30 if domain=="simglucose" else 2,
                     "history":hist,"actions":[{**a,"values":roundlist(a["values"])} for a in actions],
                     "demographics":demo,"action_information":"predefined",
                     "observation_definition":"3-minute mean held CGM" if domain=="simglucose" else "noiseless simulator readout"}}
            oracle={"id":case["id"],"group":groupid,"patient":meta["patient"],"meta":meta,
                    "targets":y,"branch_checks":{"same_anchor":True,"exact_replay":True},
                    "anchor_state_hash":str(state) if domain=="simglucose" else glu.digest(state.tolist())}
            output.append((case,oracle))
        if exogenous:
            assert all(x==exogenous[0] for x in exogenous)
    if domain=="simglucose":
        assert glu.fingerprint(anchor)==state
    else:
        np.testing.assert_array_equal(ane.hidden_state(anchor),state)
    return output

def safe_job(task):
    domain,index,scenario=task
    try:
        return task,build_cases(domain,index,scenario),None
    except Exception:
        return task,[],traceback.format_exc()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--split",choices=["development","test","all"],default="all")
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    dest=ROOT/"data/simulated"/("smoke_v1" if args.smoke else "benchmark_v1")
    dest.mkdir(parents=True,exist_ok=True)
    problems=[]; inputs=[]; labels=[]
    tasks=[]
    for domain in ["simglucose","anesthesia"]:
        indices=[100*g+i for g in range(3) for i in range(1,11)] if domain=="simglucose" else list(range(30))
        if args.smoke:indices=indices[:1]
        for index in indices:
            split=split_for(domain,index%100 if domain=="simglucose" else index)
            if args.split!="all" and split!=args.split:continue
            tasks.extend((domain,index,scenario) for scenario in ([0] if args.smoke else range(3)))
    with ProcessPoolExecutor(max_workers=1 if args.smoke else 8) as pool:
        for (domain,index,scenario),cases,error in pool.map(safe_job,tasks):
            if error:problems.append({"domain":domain,"index":index,"scenario":scenario,"error":error})
            inputs.extend(x[0] for x in cases);labels.extend(x[1] for x in cases)
            print(domain,index,scenario,len(inputs),"branches;",len(problems),"errors",flush=True)
    for split in ["development","test"]:
        if args.split not in ("all",split):
            continue
        selected={x["id"] for x in inputs if x["split"]==split}
        for name,rows in [("inputs",[x for x in inputs if x["id"] in selected]),
                          ("oracles",[x for x in labels if x["id"] in selected])]:
            (dest/(split+"."+name+".jsonl")).write_text("".join(json.dumps(x,allow_nan=False)+"\n" for x in rows))
    report={"timestamp_utc":datetime.now(timezone.utc).isoformat(),"branch_count":len(inputs),
            "pair_groups":len({x["group"] for x in inputs}),"errors":problems,
            "clinical_review":"pending; computational exploratory benchmark",
            "config_sha256":sha(ROOT/"repo/configs/study_v1.json"),
            "artifacts":{str(x.relative_to(ROOT)):sha(x) for x in dest.glob("*.jsonl")}}
    (ROOT/"data/results"/("generation_"+("smoke" if args.smoke else args.split)+".json")).write_text(json.dumps(report,indent=2)+"\n")
    if problems:
        print("WARNING: generation failures preserved; inspect report before inference",flush=True)
if __name__=="__main__":
    main()
