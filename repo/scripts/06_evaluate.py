#!/usr/bin/env python3
"""Patient-clustered analysis. No prompt/model selection or test fitting."""
from pathlib import Path
import sys,json,hashlib,argparse
from collections import defaultdict
from datetime import datetime,timezone
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"repo/src"))
from medwm_eval.metrics import cluster_ci
from medwm_eval.prompts import parse_prediction
def loadl(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
MODELS=["Qwen/Qwen3-8B","Qwen/Qwen2.5-7B-Instruct","mistralai/Mistral-7B-Instruct-v0.3","microsoft/Phi-3.5-mini-instruct"]
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--partial",action="store_true");ap.add_argument("--stress",action="store_true");args=ap.parse_args()
    runname="stress_v1" if args.stress else "study_v1"
    out=ROOT/"data/results"/runname
    if args.partial:out=out/"interim"
    out.mkdir(parents=True,exist_ok=True)
    cases={};truth={}
    for folder in (["data/simulated/hypoglycemia_stress_v1"] if args.stress else ["data/simulated/benchmark_v1","data/processed/vitaldb_v1"]):
        cases.update({x["id"]:x for x in loadl(ROOT/folder/"test.inputs.jsonl")})
        truth.update({x["id"]:np.array(x["targets"],float) for x in loadl(ROOT/folder/"test.oracles.jsonl")})
    files=[ROOT/"data/runs"/runname/"baselines/predictions.jsonl"]
    for model in MODELS:
        folder=ROOT/"data/runs"/runname/"test"/model.replace("/","--")
        if not (folder/"manifest.json").exists():
            if not args.partial:raise RuntimeError("Incomplete run: "+model)
        if (folder/"predictions.jsonl").exists():files.append(folder/"predictions.jsonl")
    runs=[x for file in files for x in loadl(file)]
    points=[];trajectories=[];predarrays={};summary=[]
    def add(kind,frame,keys,values):
        if frame.empty:return
        for key,g in frame.groupby(keys,dropna=False,sort=True):
            if not isinstance(key,tuple):key=(key,)
            meta=dict(zip(keys,key))
            for val in values:
                if val not in g:continue
                summary.append(dict(type=kind,**meta,metric=val,**cluster_ci(g[val],g["patient"])))
    keys=["model","domain","condition","mode","variable"]
    for run in runs:
        case=cases[run["id"]];p=case["payload"];y=truth[case["id"]];pred=np.full_like(y,np.nan)
        lo=np.full_like(y,np.nan);hi=np.full_like(y,np.nan);attempted=np.zeros(12,bool)
        for r in run["predictions"]:
            k=r["step"]-1;attempted[k]=True
            if r["valid"]:pred[k]=r["values"]
            if r.get("interval_valid"):lo[k]=r["lower"];hi[k]=r["upper"]
        complete=bool(np.isfinite(pred).all())
        predarrays[(run["model_id"],run["condition"],run["mode"],case["id"])]=pred
        base=dict(model=run["model_id"],domain=case["domain"],condition=run["condition"],mode=run["mode"],
                  patient=case["patient"],id=case["id"],group=case["group"],kind=case["kind"],branch=case["branch"],scenario=case["scenario"])
        for j,var in enumerate(p["variables"]):
            err=np.abs(pred[:,j]-y[:,j]);valid=np.isfinite(pred[:,j])
            trajectories.append(base|dict(variable=var,complete=float(complete),available_fraction=valid.mean(),
                 mae=float(err.mean()) if complete else np.nan,initial=y[0,j],
                 subgroup=case["patient"].split("#")[0] if case["domain"]=="simglucose" else
                    ("age<50" if p["demographics"].get("age_years",0)<50 else "age>=50")))
            for k in range(12):
                ok=bool(valid[k]);iv=bool(np.isfinite(lo[k,j]) and np.isfinite(hi[k,j]))
                violation=(pred[k,j]<0 or pred[k,j]>100) if var=="BIS" else pred[k,j]<=0
                points.append(base|dict(variable=var,step=k+1,minutes=(k+1)*p["step_min"],truth=y[k,j],pred=pred[k,j],
                    abs_error=err[k],available=float(ok),attempted=float(attempted[k]),complete=complete,
                    interval_valid=float(iv),covered=float(lo[k,j]<=y[k,j]<=hi[k,j]) if iv else np.nan,
                    unconditional_covered=float(iv and lo[k,j]<=y[k,j]<=hi[k,j]),
                    width=float(hi[k,j]-lo[k,j]) if iv else np.nan,violation=float(violation) if ok else np.nan))
    pt=pd.DataFrame(points);tr=pd.DataFrame(trajectories)
    add("trajectory",tr,keys,["mae","complete","available_fraction"])
    add("subgroup",tr,keys+["subgroup"],["mae","complete"])
    add("scenario",tr,keys+["scenario"],["mae","complete"])
    add("step",pt,keys+["step","minutes"],["abs_error","available"])
    add("interval",pt,keys,["interval_valid","covered","unconditional_covered","width","violation"])
    pairrows=[];pairpoint=[];rankrows=[]
    settings=sorted({(x["model_id"],x["condition"],x["mode"]) for x in runs})
    groups=defaultdict(dict)
    for c in cases.values():
        if c["domain"]!="vitaldb":groups[c["group"]][c["branch"]]=c
    for model,condition,mode in settings:
        for group,branches in groups.items():
            a,b=branches["A"],branches["B"]
            ka=(model,condition,mode,a["id"]);kb=(model,condition,mode,b["id"])
            if ka not in predarrays or kb not in predarrays:continue
            pa,pb=predarrays[ka],predarrays[kb];ta,tb=truth[a["id"]],truth[b["id"]]
            pdiff=pa-pb;tdiff=ta-tb;complete=np.isfinite(pdiff).all()
            meta=dict(model=model,condition=condition,mode=mode,domain=a["domain"],kind=a["kind"],
                      group=group,patient=a["patient"],scenario=a["scenario"],
                      subgroup=a["patient"].split("#")[0] if a["domain"]=="simglucose" else ("age<50" if a["payload"]["demographics"].get("age_years",0)<50 else "age>=50"))
            for j,var in enumerate(a["payload"]["variables"]):
                err=np.abs(pdiff[:,j]-tdiff[:,j]);zero=np.abs(tdiff[:,j])
                pairrows.append(meta|dict(variable=var,complete=float(complete),
                    effect_mae=err.mean() if complete else np.nan,
                    zero_effect_mae=zero.mean() if complete else np.nan,
                    delta_vs_persistence=(err-zero).mean() if complete else np.nan))
                for k in range(12):
                    row=meta|dict(variable=var,step=k+1,truth_effect=tdiff[k,j],pred_effect=pdiff[k,j],
                                  effect_error=err[k],available=float(np.isfinite(pdiff[k,j])))
                    for res in [0,1,5]:
                        eligible=abs(tdiff[k,j])>res
                        row["direction_"+str(res)]=float(np.sign(pdiff[k,j])==np.sign(tdiff[k,j])) if eligible and np.isfinite(pdiff[k,j]) else np.nan
                        row["direction_unconditional_"+str(res)]=float(np.isfinite(pdiff[k,j]) and np.sign(pdiff[k,j])==np.sign(tdiff[k,j])) if eligible else np.nan
                    pairpoint.append(row)
                if "R" in branches:
                    r=branches["R"];kr=(model,condition,mode,r["id"])
                    if kr not in predarrays:continue
                    pr=predarrays[kr];tref=truth[r["id"]]
                    diffs=np.stack([ta[:,j]-tref[:,j],tref[:,j]-tb[:,j]],axis=1)
                    predicted=np.stack([pa[:,j]-pr[:,j],pr[:,j]-pb[:,j]],axis=1)
                    eligible=np.all(np.abs(diffs)>1,axis=1)
                    avail=np.all(np.isfinite(predicted),axis=1)
                    correct=np.all(np.sign(diffs)==np.sign(predicted),axis=1)
                    for k in range(12):
                        if eligible[k]:
                            rankrows.append(meta|dict(variable=var,rank_accuracy=float(correct[k]) if avail[k] else np.nan,
                                  rank_unconditional=float(avail[k] and correct[k])))
    pr=pd.DataFrame(pairrows);pp=pd.DataFrame(pairpoint);rk=pd.DataFrame(rankrows)
    add("paired",pr,keys+["kind"],["effect_mae","complete","zero_effect_mae","delta_vs_persistence"])
    add("paired_subgroup",pr,keys+["kind","subgroup"],["effect_mae","complete","delta_vs_persistence"])
    add("paired_scenario",pr,keys+["kind","scenario"],["effect_mae","delta_vs_persistence"])
    add("direction",pp,keys+["kind"],["direction_0","direction_1","direction_5","direction_unconditional_1"])
    add("ranking",rk,keys,["rank_accuracy","rank_unconditional"])
    # Case-matched teacher/free and information contrasts.
    index=["model","domain","variable","id","patient"]
    sa=tr[tr.condition=="state_action"]
    merge=sa[sa["mode"]=="free"].merge(sa[sa["mode"]=="teacher"],on=index,suffixes=("_free","_teacher"))
    merge["free_minus_teacher"]=merge["mae_free"]-merge["mae_teacher"]
    add("feedback_contrast",merge,["model","domain","variable"],["free_minus_teacher"])
    free=tr[tr["mode"]=="free"]
    merge=free[free.condition=="state_action"].merge(free[free.condition=="state_only"],on=index,suffixes=("_action","_only"))
    merge["action_minus_stateonly"]=merge["mae_action"]-merge["mae_only"]
    add("information_contrast",merge,["model","domain","variable"],["action_minus_stateonly"])
    # Same-patient event rates; report positive event-point and patient denominators.
    events=[]
    for var,name,cut,low in [("CGM","low_glucose",70,True),("CGM","high_glucose",180,False),
                            ("BIS","low_BIS",40,True),("BIS","high_BIS",60,False),("MAP","low_MAP",65,True)]:
        df=pt[pt.variable==var].copy()
        df["event"]=df.truth<cut if low else df.truth>cut
        df["detected"]=df.pred<cut if low else df.pred>cut
        positives=df[df.event]
        for key,g in positives.groupby(keys+["patient"]):
            events.append(dict(zip(keys+["patient"],key))|dict(event=name,miss_rate=1-g.detected.mean(),event_points=len(g)))
    ev=pd.DataFrame(events)
    add("event",ev,keys+["event"],["miss_rate"])
    for key,g in ev.groupby(keys+["event"]):
        summary.append(dict(type="event_denominator",**dict(zip(keys+["event"],key)),event_points=int(g.event_points.sum()),patients=len(g)))
    # Any-event-per-trajectory detection, on all scheduled trajectories.
    anyevents=[]
    for var,name,cut,low in [("CGM","low_glucose",70,True),("CGM","high_glucose",180,False),
                            ("BIS","low_BIS",40,True),("BIS","high_BIS",60,False),("MAP","low_MAP",65,True)]:
        df=pt[pt.variable==var]
        for key,g in df.groupby(keys+["patient","id"]):
            actual=(g.truth<cut if low else g.truth>cut).any()
            predicted=(g.pred<cut if low else g.pred>cut).any()
            if actual:
                anyevents.append(dict(zip(keys+["patient","id"],key))|dict(event=name,miss_rate=float(not predicted)))
    add("any_event",pd.DataFrame(anyevents),keys+["event"],["miss_rate"])
    # Matched-case numerical baseline comparisons prevent survivor-set advantages.
    llmtr=tr[tr.model.isin(MODELS)]
    llmpr=pr[pr.model.isin(MODELS)]
    matched=[]
    for baseline in ["persistence","trend","ridge_action","boosted_action"]:
        for kind,df,value,join in [("trajectory",tr,"mae",["domain","mode","variable","id","patient"]),
                                  ("paired",pr,"effect_mae",["domain","mode","variable","group","patient","kind"])]:
            left=df[df.model.isin(MODELS)]
            right=df[(df.model==baseline)&(df.condition=="state_action")][join+[value]]
            merged=left.merge(right,on=join,suffixes=("","_baseline"))
            merged["difference"]=merged[value]-merged[value+"_baseline"]
            merged["baseline"]=baseline
            add("matched_"+kind,merged,keys+["baseline"]+(["kind"] if kind=="paired" else []),["difference"])
    # Fixed engineering sensitivity horizons: consecutive steps from the anchor.
    horizons=[]
    stepdf=pd.DataFrame([x for x in summary if x["type"]=="step"])
    for key,g in stepdf.groupby(keys):
        means=g[g.metric=="abs_error"].set_index("step")
        available=g[g.metric=="available"].set_index("step")
        var=key[-1]
        for tolerance in ([5,10,20] if var=="CGM" else [2,5,10]):
            finalstep=0
            for h in range(1,13):
                if h not in means.index or means.loc[h,"mean"] is None or not np.isfinite(means.loc[h,"mean"]):break
                if means.loc[h,"mean"]>tolerance or available.loc[h,"mean"]<.95:break
                finalstep=h
            horizons.append(dict(zip(keys,key))|dict(tolerance=tolerance,last_step=finalstep,
                minutes=finalstep*(30 if key[1]=="simglucose" else 2 if key[1]=="anesthesia" else 1)))
    pd.DataFrame(horizons).to_csv(out/"engineering_horizons.csv",index=False)
    # Hierarchical patient/scenario resampling of the primary contrast.
    primarycases=pr[(pr.model==MODELS[0])&(pr.domain=="simglucose")&(pr.condition=="state_action")&(pr["mode"]=="free")&(pr.kind=="dose")].dropna(subset=["delta_vs_persistence"])
    hierarchical=None
    if len(primarycases):
        arrs=[g.delta_vs_persistence.to_numpy() for _,g in primarycases.groupby("patient")]
        rng=np.random.default_rng(20261102)
        boots=[]
        for _ in range(2000):
            draw=rng.integers(0,len(arrs),len(arrs))
            boots.append(np.mean([rng.choice(arrs[i],len(arrs[i]),replace=True).mean() for i in draw]))
        hierarchical=dict(mean=float(np.mean([a.mean() for a in arrs])),lo=float(np.quantile(boots,.025)),
                          hi=float(np.quantile(boots,.975)),patients=len(arrs),replicates=2000)
    (out/"primary_hierarchical.json").write_text(json.dumps(hierarchical,indent=2))
    # Auxiliary extraction and knowledge are separate narrow checks.
    auxrows=[];diagrows=[]
    for model in MODELS:
        folder=ROOT/"data/runs"/runname/"test"/model.replace("/","--")
        if (folder/"auxiliary.json").exists():
            for row in json.loads((folder/"auxiliary.json").read_text()):
                try:
                    obj=json.loads(row["raw"].strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
                    copied=np.asarray(obj["values"]); expected=np.asarray(row["expected_values"])
                    valueok=copied.shape==expected.shape and np.allclose(copied,expected,rtol=0,atol=1e-4)
                    timeok=isinstance(obj.get("time_min"),(int,float)) and not isinstance(obj.get("time_min"),bool) and obj["time_min"]==0
                    knowledge=obj.get("direction")==row["expected_direction"]
                except (TypeError,ValueError,KeyError):valueok=timeok=knowledge=False
                auxrows.append(dict(model=model,domain=row["domain"],patient=row["patient"],id=row["id"],
                    state_accuracy=float(valueok and timeok),knowledge_accuracy=float(knowledge)))
        if (folder/"diagnostics.json").exists():
            for row in json.loads((folder/"diagnostics.json").read_text()):
                c=cases[row["id"]];r=row["result"];horizons=[1,3,6,12] if row["variant"]=="one_shot" else [1]
                y=truth[row["id"]]
                for j,var in enumerate(c["payload"]["variables"]):
                    for k,h in enumerate(horizons):
                        pred=r["values"][k][j] if r["valid"] and row["variant"]=="one_shot" else r["values"][j] if r["valid"] else np.nan
                        base=predarrays.get((model,"state_action","free",row["id"]),np.full_like(y,np.nan))[h-1,j]
                        diagrows.append(dict(model=model,domain=row["domain"],patient=row["patient"],variable=var,
                            variant=row["variant"],step=h,available=float(r["valid"]),mae=abs(pred-y[h-1,j]),
                            error_change=abs(pred-y[h-1,j])-abs(base-y[h-1,j])))
    aux=pd.DataFrame(auxrows);diag=pd.DataFrame(diagrows)
    add("auxiliary",aux,["model","domain"],["state_accuracy","knowledge_accuracy"])
    add("diagnostics",diag,["model","domain","variable","variant","step"],["available","mae","error_change"])
    for name,df in [("points",pt),("trajectories",tr),("pairs",pr),("pair_points",pp),("rankings",rk),("events",ev),("auxiliary",aux),("diagnostics",diag)]:
        df.to_parquet(out/(name+".parquet"),index=False)
    (out/"summary.json").write_text(json.dumps(clean(summary),indent=2,allow_nan=False))
    pd.DataFrame(summary).to_csv(out/"summary.csv",index=False)
    primary=[x for x in summary if x["type"]=="paired" and x["model"]==MODELS[0] and x["domain"]=="simglucose" and x["condition"]=="state_action" and x["mode"]=="free" and x["kind"]=="dose"]
    (out/"primary.json").write_text(json.dumps(clean(primary),indent=2,allow_nan=False))
    (out/"analysis_manifest.json").write_text(json.dumps(dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/"repo/src/medwm_eval/metrics.py"]},partial=args.partial,files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},cases=len(cases),run_rows=len(runs),summary_rows=len(summary)),indent=2))
    print(json.dumps(clean(primary),indent=2));print("RESULTS",out,flush=True)
if __name__=="__main__":main()
