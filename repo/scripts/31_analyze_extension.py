#!/usr/bin/env python3
"""Prespecified extension contrasts, cluster statistics and publication outputs."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import json,hashlib,sys,os,shutil,re
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"repo/src"))
from medwm_eval.metrics import cluster_ci
from medwm_eval.figure_style import use_times
from medwm_eval.submission_extension import score_memory
MODELS=["Qwen/Qwen3-8B","Qwen/Qwen2.5-7B-Instruct","mistralai/Mistral-7B-Instruct-v0.3","microsoft/Phi-3.5-mini-instruct"]
NAMES=dict(zip(MODELS,["Qwen3-8B","Qwen2.5-7B","Mistral-7B","Phi-3.5-mini"]))
DOM={"simglucose":"Glucose","anesthesia":"PAS","vitaldb":"VitalDB"}
PANELS=[("simglucose","CGM"),("anesthesia","BIS"),("anesthesia","MAP"),("vitaldb","BIS"),("vitaldb","MAP")]
OUT=ROOT/"data/results/interface_memory_v1";PUB=ROOT/"paper/tables"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def loadl(p):return [json.loads(l) for l in p.read_text().splitlines()]
def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,list):return [clean(v) for v in x]
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def fmt(r,pct=False):
    if r is None or r.get("mean") is None or not np.isfinite(r["mean"]):return "---"
    m=100 if pct else 1
    point=f'{r["mean"]*m:.1f}'
    if r["clusters"]<2:return point
    return point+r" {\scriptsize ["+f'{r["lo"]*m:.1f}, {r["hi"]*m:.1f}'+"]}"
def table(name,headers,rows):
    text=r"\begin{tabular}{l"+"c"*(len(headers)-1)+"}\n"+r"\toprule"+"\n"
    text+=" & ".join(headers)+r" \\"+"\n"+r"\midrule"+"\n"
    text+="\n".join(" & ".join(map(str,row))+r" \\" for row in rows)+"\n"+r"\bottomrule"+"\n"+r"\end{tabular}"+"\n"
    (OUT/"tables").mkdir(exist_ok=True)
    (OUT/"tables"/(name+".tex")).write_text(text);(PUB/(name+".tex")).write_text(text)
def main():
    integrity=json.loads((OUT/"integrity.json").read_text());assert len(integrity["models"])==4
    cases={};truth={}
    for folder in ["data/simulated/benchmark_v1","data/processed/vitaldb_v1"]:
        cases.update({r["id"]:r for r in loadl(ROOT/folder/"test.inputs.jsonl")})
        truth.update({r["id"]:np.array(r["targets"],float) for r in loadl(ROOT/folder/"test.oracles.jsonl")})
    trajectories=[];points=[];pairs=[];memories=[];summary=[]
    def add(typ,df,keys,metrics):
        if df.empty:return
        for key,g in df.groupby(keys,sort=True):
            if not isinstance(key,tuple):key=(key,)
            for metric in metrics:
                summary.append(dict(type=typ,**dict(zip(keys,key)),metric=metric,**cluster_ci(g[metric],g.patient)))
    for model in MODELS:
        path=ROOT/"data/runs/interface_memory_v1/test"/model.replace("/","--")
        arr={}
        for run in loadl(path/"predictions.jsonl"):
            c=cases[run["id"]];y=truth[c["id"]];pred=np.full_like(y,np.nan);lo=pred.copy();hi=pred.copy()
            for r in run["predictions"]:
                k=r["step"]-1
                if r["valid"]:pred[k]=r["values"]
                if r.get("interval_valid"):lo[k]=r["lower"];hi[k]=r["upper"]
            arr[c["id"]]=pred;complete=bool(np.isfinite(pred).all())
            meta=dict(model=model,id=c["id"],domain=c["domain"],patient=c["patient"],group=c["group"],kind=c["kind"])
            for j,v in enumerate(c["payload"]["variables"]):
                trajectories.append(meta|dict(variable=v,complete=float(complete),
                    mae=float(np.abs(pred[:,j]-y[:,j]).mean()) if complete else np.nan))
                for k in range(12):
                    valid=np.isfinite(pred[k,j]);iv=np.isfinite(lo[k,j]) and np.isfinite(hi[k,j])
                    points.append(meta|dict(variable=v,step=k+1,available=float(valid),error=abs(pred[k,j]-y[k,j]),
                        covered=float(lo[k,j]<=y[k,j]<=hi[k,j]) if iv else np.nan,
                        unconditional_covered=float(iv and lo[k,j]<=y[k,j]<=hi[k,j]),interval_valid=float(iv)))
        groups=defaultdict(dict)
        for c in cases.values():
            if c["domain"]!="vitaldb":groups[c["group"]][c["branch"]]=c
        for group,branches in groups.items():
            a,b=branches["A"],branches["B"];diff=arr[a["id"]]-arr[b["id"]];ref=truth[a["id"]]-truth[b["id"]]
            complete=bool(np.isfinite(diff).all())
            for j,v in enumerate(a["payload"]["variables"]):
                errors=np.abs(diff[:,j]-ref[:,j]);zero=np.abs(ref[:,j])
                eligible=np.abs(ref[:,j])>1
                direction=np.sign(diff[:,j])==np.sign(ref[:,j])
                pairs.append(dict(model=model,domain=a["domain"],patient=a["patient"],group=group,kind=a["kind"],variable=v,
                    complete=float(complete),effect_mae=float(errors.mean()) if complete else np.nan,
                    zero_effect_mae=float(zero.mean()) if complete else np.nan,
                    delta_zero=float((errors-zero).mean()) if complete else np.nan,
                    sign_all=float(direction[eligible].mean()) if eligible.any() else np.nan))
        for r in json.loads((path/"memory.json").read_text()):
            for parser,prefix in [("strict",False),("content",True)]:
                memories.append(dict(model=model,domain=r["domain"],patient=r["patient"],id=r["id"],anchor=r["anchor_step"],parser=parser)|
                                score_memory(r["raw"],r["target"],prefix))
    tr=pd.DataFrame(trajectories);pt=pd.DataFrame(points);pr=pd.DataFrame(pairs);mem=pd.DataFrame(memories)
    keys=["model","domain","variable"]
    add("trajectory",tr,keys,["mae","complete"])
    add("step",pt,keys+["step"],["error","available"])
    add("interval",pt,keys,["covered","unconditional_covered","interval_valid"])
    add("paired",pr,keys+["kind"],["effect_mae","complete","zero_effect_mae","delta_zero","sign_all"])
    add("retrieval",mem,["model","domain","anchor","parser"],["json_object","state","temporal","current_action","past_action","all_fields"])
    oldtr=pd.read_parquet(ROOT/"data/results/study_v1/trajectories.parquet")
    oldtr=oldtr[(oldtr.condition=="state_action")&(oldtr["mode"]=="free")]
    merged=tr.merge(oldtr[keys+["id","patient","mae","complete"]],on=keys+["id","patient"],suffixes=("_named","_original"),validate="one_to_one")
    merged["error_difference"]=merged.mae_named-merged.mae_original
    merged["completion_difference"]=merged.complete_named-merged.complete_original
    add("interface_trajectory",merged,keys,["error_difference","completion_difference"])
    oldpr=pd.read_parquet(ROOT/"data/results/study_v1/pairs.parquet")
    oldpr=oldpr[(oldpr.condition=="state_action")&(oldpr["mode"]=="free")]
    paired=pr.merge(oldpr[keys+["group","patient","kind","effect_mae","complete"]],on=keys+["group","patient","kind"],suffixes=("_named","_original"),validate="one_to_one")
    paired["error_difference"]=paired.effect_mae_named-paired.effect_mae_original
    paired["completion_difference"]=paired.complete_named-paired.complete_original
    add("interface_paired",paired,keys+["kind"],["error_difference","completion_difference"])
    # Reference-step equal weighting, matching the original sign estimand.
    # A case-averaged sign is not substituted for that endpoint.
    signpoints=[]
    for model in MODELS:
        arrs={}
        for run in loadl(ROOT/"data/runs/interface_memory_v1/test"/model.replace("/","--")/"predictions.jsonl"):
            y=truth[run["id"]];x=np.full_like(y,np.nan)
            for r in run["predictions"]:
                if r["valid"]:x[r["step"]-1]=r["values"]
            arrs[run["id"]]=x
        for group,branches in groups.items():
            a,b=branches["A"],branches["B"];pred=arrs[a["id"]]-arrs[b["id"]];ref=truth[a["id"]]-truth[b["id"]]
            for j,v in enumerate(a["payload"]["variables"]):
                for k in range(12):
                    if abs(ref[k,j])>1:
                        signpoints.append(dict(model=model,domain=a["domain"],variable=v,kind=a["kind"],patient=a["patient"],
                            correct=float(np.isfinite(pred[k,j]) and np.sign(pred[k,j])==np.sign(ref[k,j]))))
    add("direction",pd.DataFrame(signpoints),keys+["kind"],["correct"])
    def get(typ,metric,**kw):
        r=[x for x in summary if x["type"]==typ and x["metric"]==metric and all(x.get(k)==v for k,v in kw.items())]
        assert len(r)==1,(typ,metric,kw,len(r));return r[0]
    rows=[]
    for d,v in PANELS:
        for m in MODELS:
            kw=dict(model=m,domain=d,variable=v)
            n=tr[(tr.model==m)&(tr.domain==d)&(tr.variable==v)]
            rows.append([DOM[d]+" "+v,NAMES[m],str(int(n.complete.sum()))+"/"+str(len(n)),str(get("interface_trajectory","error_difference",**kw)["n"]),fmt(get("trajectory","mae",**kw)),
                         fmt(get("interface_trajectory","error_difference",**kw)),fmt(get("interface_trajectory","completion_difference",**kw),True)])
    table("interface_trajectory",["Output","Model","Complete","Matched n","Named MAE","MAE change",r"Completion pp"],rows)
    rows=[]
    for d,v in PANELS[:3]:
        for m in MODELS:
            kw=dict(model=m,domain=d,variable=v,kind="dose");g=pr[(pr.model==m)&(pr.domain==d)&(pr.variable==v)&(pr.kind=="dose")]
            rows.append([DOM[d]+" "+v,NAMES[m],str(get("interface_paired","error_difference",**kw)["n"]),fmt(get("paired","effect_mae",**kw)),
                         fmt(get("interface_paired","error_difference",**kw)),fmt(get("paired","delta_zero",**kw))])
    table("interface_paired",["Output","Model","Matched n","Named error",r"\shortstack{Change vs\\original}",r"\shortstack{Excess vs\\zero}"],rows)
    rows=[]
    for d,v in PANELS[:3]:
        for m in MODELS:
            kw=dict(model=m,domain=d,variable=v)
            g=pr[(pr.model==m)&(pr.domain==d)&(pr.variable==v)&(pr.kind=="timing")]
            rows.append([DOM[d]+" "+v,NAMES[m],fmt(get("direction","correct",kind="dose",**kw),True),
                         str(get("interface_paired","error_difference",kind="timing",**kw)["n"]),fmt(get("paired","effect_mae",kind="timing",**kw)),
                         fmt(get("interface_paired","error_difference",kind="timing",**kw))])
    table("interface_action",["Output","Model",r"Dose sign (\%)","Matched n","Timing error","Error change"],rows)
    for anchor in [0,6]:
        rows=[]
        for d in ["simglucose","anesthesia","vitaldb"]:
            for m in MODELS:
                kw=dict(model=m,domain=d,anchor=anchor,parser="content")
                rows.append([DOM[d],NAMES[m]]+[fmt(get("retrieval",metric,**kw),True) for metric in ["state","temporal","current_action","past_action","all_fields"]])
        table("retrieval_"+str(anchor),["Domain","Model",r"State (\%)",r"Time (\%)",r"\shortstack{Current\\action (\%)}",r"\shortstack{Past\\action (\%)}",r"Joint (\%)"],rows)
    rows=[]
    for d in ["simglucose","anesthesia","vitaldb"]:
        for m in MODELS:
            rows.append([DOM[d],NAMES[m]]+[fmt(get("retrieval","all_fields",model=m,domain=d,anchor=a,parser=p),True) for a,p in [(0,"strict"),(0,"content"),(6,"strict"),(6,"content")]])
    table("retrieval_compliance",["Domain","Model",r"\shortstack{Anchor\\strict}",r"\shortstack{Anchor\\content}",r"\shortstack{Step 6\\strict}",r"\shortstack{Step 6\\content}"],rows)
    rows=[]
    for d,v in PANELS:
        for m in MODELS:
            kw=dict(model=m,domain=d,variable=v)
            rows.append([DOM[d]+" "+v,NAMES[m],fmt(get("interval","covered",**kw),True),fmt(get("interval","unconditional_covered",**kw),True)])
    table("interface_intervals",["Output","Model",r"Valid coverage (\%)",r"All coverage (\%)"],rows)
    rows=[]
    for r in integrity["models"]:
        rows.append([NAMES[r["model"]],f'{r["unique_raw_calls"]:,}',f'{r["generation_seconds"]/3600:.2f}',f'{r["input_tokens"]/1e6:.2f}',f'{r["generated_token_slots_upper_bound"]/1e6:.2f}'])
    table("extension_compute",["Model","Calls","Gen h","Input M","Output slots M"],rows)
    for name,df in [("trajectories",tr),("points",pt),("pairs",pr),("retrieval",mem),("trajectory_contrasts",merged),("pair_contrasts",paired)]:
        df.to_parquet(OUT/(name+".parquet"),index=False)
    (OUT/"summary.json").write_text(json.dumps(clean(summary),indent=2));pd.DataFrame(summary).to_csv(OUT/"summary.csv",index=False)
    os.environ["MPLCONFIGDIR"]=str(ROOT/"data/cache/matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    use_times()
    plt.rcParams.update({"font.size":10,"pdf.fonttype":42})
    colors=["#0072B2","#D55E00","#009E73","#CC79A7"]
    fig,axes=plt.subplots(2,3,figsize=(7.2,5.2))
    for col,(d,v) in enumerate(PANELS[:3]):
        for i,m in enumerate(MODELS):
            kw=dict(model=m,domain=d,variable=v,kind="dose")
            for row,typ,metric in [(0,"interface_paired","completion_difference"),(1,"interface_paired","error_difference")]:
                r=get(typ,metric,**kw);factor=100 if row==0 else 1
                if r["mean"] is None:
                    axes[row,col].text(i,.03,"NA",transform=axes[row,col].get_xaxis_transform(),ha="center",fontsize=8)
                    continue
                err=[[factor*(r["mean"]-r["lo"])],[factor*(r["hi"]-r["mean"])]] if r["clusters"]>1 else None
                axes[row,col].errorbar(i,r["mean"]*factor,yerr=err,fmt="o",color=colors[i],capsize=3)
        axes[0,col].set_title(chr(97+col)+") "+DOM[d]+" "+v)
        axes[1,col].set_title(chr(100+col)+")",loc="left",fontweight="bold",fontsize=10)
        for ax in axes[:,col]:
            ax.set_xlim(-.4,3.4);ax.axhline(0,color="#888",ls="--",lw=.8);ax.set_xticks(range(4),[NAMES[m] for m in MODELS],rotation=45,ha="right",rotation_mode="anchor",fontsize=8);ax.grid(axis="y",alpha=.2)
            ax.spines[["top","right"]].set_visible(False)
    axes[0,0].set_ylabel("Pair completion change (pp)")
    axes[1,0].set_ylabel("Matched effect-error change")
    fig.tight_layout()
    (OUT/"figures").mkdir(exist_ok=True)
    for ext in ["pdf","png"]:
        p=OUT/"figures"/("interface_sensitivity."+ext);fig.savefig(p,bbox_inches="tight",dpi=300)
        shutil.copy2(p,ROOT/"paper/figures"/p.name)
    plt.close(fig)
    outputs=list((OUT/"tables").glob("*.tex"))+list((OUT/"figures").glob("*"))
    manifest=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),post_primary=True,partial=False,
        original_primary_unchanged=True,annotation_required=False,source_sha256=sha(Path(__file__)),
        integrity_sha256=sha(OUT/"integrity.json"),summary_sha256=sha(OUT/"summary.json"),
        outputs={str(p.relative_to(ROOT)):sha(p) for p in outputs})
    (OUT/"analysis_manifest.json").write_text(json.dumps(manifest,indent=2))
    print("EXTENSION ANALYSIS COMPLETE",OUT)
if __name__=="__main__":main()
