#!/usr/bin/env python3
"""Post-freeze targeted hypoglycemia stress set; primary benchmark unchanged."""
from pathlib import Path
import sys,json,hashlib,traceback
from copy import deepcopy
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"repo/src"))
from medwm_eval.envs import simglucose_adapter as glu
def one(task):
    group,index=task
    name=group+f"#{index:03d}";seed=20262000+["adult","adolescent","child"].index(group)*100+index
    try:
        recipe=glu.Recipe(patient=name,seed=seed,warmup_min=720,horizon_min=360,meals=())
        env=glu.build(recipe);base=glu.reference_rate(env);warm=glu.validate_rate(env,2*base)
        history=[dict(time=0,values=[glu.point(env)["glucose_mg_dL"]],past_action=[0.],past_meal_g=0)]
        anchor=None
        for step in range(1,25):
            rows=glu.advance(env,warm,30);value=rows[-1]["glucose_mg_dL"]
            history.append(dict(time=step*30,values=[value],past_action=[rows[-1]["applied_insulin_U_min"]],past_meal_g=0))
            if step>=6 and 80<=value<=130 and history[-2]["values"][0]-value>=3:
                anchor=step*30;break
        if anchor is None:return [],dict(patient=name,status="excluded",reason="no prespecified falling-state anchor")
        history=history[-7:]
        for h in history:h["time"]-=anchor;h["values"]=[round(v,6) for v in h["values"]];h["past_action"]=[round(v,6) for v in h["past_action"]]
        fingerprint=glu.fingerprint(env);gid=glu.digest(dict(patient=name,seed=seed,anchor=anchor,stress="v1"))[:24]
        output=[];noise=[]
        for branch,factor in [("A",.5),("B",1.)]:
            rate=glu.validate_rate(env,base*factor)
            def simulate(e):
                return [[glu.advance(e,rate,30)[-1]["glucose_mg_dL"]] for _ in range(12)]
            e=deepcopy(env);target=simulate(e);assert target==simulate(deepcopy(env));noise.append(glu.future_noise(e,25))
            c=dict(id=gid+"_"+branch,group=gid,patient=name,split="test",domain="simglucose",kind="hypoglycemia_stress",branch=branch,scenario=0,
                payload=dict(variables=["CGM"],units=["mg/dL"],action_names=["basal_insulin_U_min"],step_min=30,history=history,
                  actions=[dict(time=k*30,values=[round(rate,6)],meal_g=0) for k in range(12)],demographics={},action_information="predefined",observation_definition="3-minute mean held CGM"))
            o=dict(id=c["id"],group=gid,patient=name,targets=target,anchor_state_hash=fingerprint,branch_checks=dict(same_anchor=True,exact_replay=True),
                   meta=dict(seed=seed,anchor_min=anchor,recipe=recipe.to_dict(),warmup_rate_U_min=warm))
            output.append((c,o))
        assert noise[0]==noise[1] and glu.fingerprint(env)==fingerprint
        lows=sum(v[0]<70 for _,o in output for v in o["targets"])
        return output,dict(patient=name,status="included",anchor_min=anchor,low_points=lows)
    except Exception:
        return [],dict(patient=name,status="excluded",reason="simulator termination or error",traceback=traceback.format_exc())
def main():
    dest=ROOT/"data/simulated/hypoglycemia_stress_v1";dest.mkdir(parents=True,exist_ok=True)
    tasks=[(g,i) for g in ["adult","adolescent","child"] for i in range(5,11)]
    inputs=[];oracle=[];audit=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for rows,report in pool.map(one,tasks):
            inputs.extend(x[0] for x in rows);oracle.extend(x[1] for x in rows);audit.append(report)
    for name,rows in [("inputs",inputs),("oracles",oracle)]:
        (dest/("test."+name+".jsonl")).write_text("".join(json.dumps(x,allow_nan=False)+"\n" for x in rows))
    result=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),planned_patients=18,included_patients=len(inputs)//2,
                branches=len(inputs),low_points=sum(r.get("low_points",0) for r in audit),audit=audit,
                status="targeted post-primary-freeze stress analysis, not primary-confirmatory",
                file_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.glob("*.jsonl")})
    (ROOT/"data/results/hypoglycemia_stress_audit.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
