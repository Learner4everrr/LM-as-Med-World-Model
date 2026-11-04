#!/usr/bin/env python3
"""Post hoc magnitude/horizon sensitivity using saved original synthetic outputs."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, sys, zipfile, os
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2]/"data/cache/matplotlib"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"repo/src"))
from medwm_eval.metrics import cluster_ci
from medwm_eval.figure_style import use_times
MODELS=["Qwen/Qwen3-8B","Qwen/Qwen2.5-7B-Instruct","mistralai/Mistral-7B-Instruct-v0.3","microsoft/Phi-3.5-mini-instruct"]
ORDER=MODELS+["persistence","trend","ridge_history","ridge_action","boosted_action"]
NAMES=dict(zip(ORDER,["Qwen3-8B","Qwen2.5-7B","Mistral-7B","Phi-3.5-mini","Persistence","Linear trend","Ridge history","Ridge actions","Boosted trees"]))
PANELS=[("simglucose","CGM","Glucose CGM","mg/dL"),("anesthesia","BIS","Anesthesia BIS","BIS units"),("anesthesia","MAP","Anesthesia MAP","mmHg")]
OUT=ROOT/"data/results/revision_v1"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def loadl(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def ci(frame, column):
    if frame.empty:return dict(mean=None,lo=None,hi=None,clusters=0,n=0)
    g=frame.groupby(["patient","group"],sort=True)[column].mean().reset_index()
    result=cluster_ci(g[column],g.patient)
    if result["clusters"]<2:result.update(lo=None,hi=None)
    return result
def estimate(frame, scope, **meta):
    valid=frame[frame.available & (frame.complete if scope=="complete_pairs" else True)]
    result=dict(**meta,scope=scope,scheduled_points=len(frame),scheduled_pairs=int(frame.group.nunique()),
                scheduled_clusters=int(frame.patient.nunique()),available_points=len(valid),
                available_pairs=int(valid.group.nunique()),available_clusters=int(valid.patient.nunique()),
                raw_availability=float(len(valid)/len(frame)) if len(frame) else None)
    a=frame.copy();a["usable"]=(a.available & (a.complete if scope=="complete_pairs" else True)).astype(float)
    result["cluster_availability"]=ci(a,"usable")
    for column in ["effect_error","zero_error","excess_error"]:result[column]=ci(valid,column)
    return result
def fmt(r):
    if r["mean"] is None:return "---"
    s=f'{r["mean"]:.1f}'
    if r["lo"] is not None:s+=r" {\scriptsize ["+f'{r["lo"]:.1f}, {r["hi"]:.1f}'+"]}"
    return s
def save(fig,name):
    for folder in [OUT/"figures",ROOT/"paper/figures"]:
        folder.mkdir(parents=True,exist_ok=True)
        fig.savefig(folder/(name+".pdf"),bbox_inches="tight")
        fig.savefig(folder/(name+".png"),bbox_inches="tight",dpi=300)
    plt.close(fig)
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cases=loadl(ROOT/"data/simulated/benchmark_v1/test.inputs.jsonl")
    truth={r["id"]:np.array(r["targets"]) for r in loadl(ROOT/"data/simulated/benchmark_v1/test.oracles.jsonl")}
    groups={}
    for c in cases:
        if c["kind"]=="dose":groups.setdefault(c["group"],{})[c["branch"]]=c
    local=ROOT/"data/runs/study_v1"
    sources=[]
    if (local/"test/Qwen--Qwen3-8B/predictions.jsonl").exists():
        paths=[local/"test"/m.replace("/","--")/"predictions.jsonl" for m in MODELS]+[local/"baselines/predictions.jsonl"]
        runs=[r for p in paths for r in loadl(p) if r["domain"]!="vitaldb"];sources+=paths
    else:
        archive=ROOT/"data/releases/synthetic_outputs.zip";sources.append(archive)
        with zipfile.ZipFile(archive) as z:
            names=["study_v1/"+m.replace("/","--")+"/predictions.jsonl" for m in MODELS]+["study_v1/baselines/predictions.jsonl"]
            runs=[json.loads(line) for n in names for line in z.read(n).splitlines()]
    predictions={}
    for r in runs:
        condition="state_only" if r["model_id"]=="ridge_history" else "state_action"
        if r["kind"]!="dose" or r["mode"]!="free" or r["condition"]!=condition:continue
        values=np.full_like(truth[r["id"]],np.nan,dtype=float)
        for p in r["predictions"]:
            if p["valid"]:values[p["step"]-1]=p["values"]
        predictions[(r["model_id"],r["id"])]=values
    rows=[]
    for model in ORDER:
        for group,branches in groups.items():
            a,b=branches["A"],branches["B"]
            pa,pb=predictions[(model,a["id"])],predictions[(model,b["id"])]
            predicted=pa-pb;reference=truth[a["id"]]-truth[b["id"]]
            complete=bool(np.isfinite(predicted).all())
            for j,var in enumerate(a["payload"]["variables"]):
                for k in range(12):
                    zero=abs(reference[k,j]);error=abs(predicted[k,j]-reference[k,j])
                    rows.append(dict(model=model,domain=a["domain"],variable=var,group=group,
                        patient=a["patient"],step=k+1,truth_effect=reference[k,j],pred_effect=predicted[k,j],
                        zero_error=zero,effect_error=error,excess_error=error-zero,
                        available=bool(np.isfinite(predicted[k,j])),complete=complete))
    points=pd.DataFrame(rows)
    allrows=[];magnitude=[];horizons=[];distributions=[]
    for model in ORDER:
        for domain,var,title,unit in PANELS:
            frame=points[(points.model==model)&(points.domain==domain)&(points.variable==var)]
            meta=dict(model=model,domain=domain,variable=var,unit=unit)
            for scope in ["available_points","complete_pairs"]:
                allrows.append(estimate(frame,scope,**meta))
                for h in range(1,13):horizons.append(estimate(frame[frame.step==h],scope,**meta,step=h))
                edges=[0,5,20,50,np.inf] if var=="CGM" else [0,1,5,10,np.inf]
                for lo,hi in zip(edges[:-1],edges[1:]):
                    mask=(frame.zero_error>=lo)&(frame.zero_error<hi)
                    label=f"[{lo}, {hi})" if np.isfinite(hi) else f"[{lo}, infinity)"
                    magnitude.append(estimate(frame[mask],scope,**meta,bin=label,lower=lo,upper=hi if np.isfinite(hi) else None))
            if model==MODELS[0]:
                distributions.append(dict(domain=domain,variable=var,unit=unit,points=len(frame),pairs=frame.group.nunique(),
                    clusters=frame.patient.nunique(),pooled_quantiles={str(q):float(frame.zero_error.quantile(q)) for q in [0,.25,.5,.75,1]},
                    pooled_fraction_below_first_cut=float((frame.zero_error<edges[1]).mean())))
    original=json.loads((ROOT/"data/results/study_v1/summary.json").read_text())
    matched=0
    for r in allrows:
        if r["scope"]!="complete_pairs":continue
        condition="state_only" if r["model"]=="ridge_history" else "state_action"
        for a,b in [("effect_error","effect_mae"),("zero_error","zero_effect_mae"),("excess_error","delta_vs_persistence")]:
            saved=next(s for s in original if all(s.get(k)==v for k,v in dict(type="paired",model=r["model"],domain=r["domain"],variable=r["variable"],condition=condition,mode="free",kind="dose",metric=b).items()))
            assert abs(r[a]["mean"]-saved["mean"])<1e-10
            if r[a]["clusters"]>1:
                assert abs(r[a]["lo"]-saved["lo"])<1e-10 and abs(r[a]["hi"]-saved["hi"])<1e-10
            matched+=1
    # Independent verification from original point-level analysis, when retained locally.
    oldpath=ROOT/"data/results/study_v1/pair_points.parquet"
    if oldpath.exists():
        old=pd.read_parquet(oldpath)
        key=["model","domain","variable","group","step"]
        old=old[(old.kind=="dose")&(old["mode"]=="free")&((old.condition=="state_action")|((old.model=="ridge_history")&(old.condition=="state_only")))]
        merged=points.merge(old,on=key,suffixes=("_new","_old"),validate="one_to_one")
        assert len(merged)==len(points)
        for name in ["truth_effect","pred_effect"]:
            assert np.allclose(merged[name+"_new"],merged[name+"_old"],atol=1e-12,rtol=0,equal_nan=True)
    # All published rows are simulator-only, including row-level data.
    points.to_csv(OUT/"dose_pair_points.csv",index=False)
    result=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),analysis_status="post_hoc_revision",
                original_primary_unchanged=True,original_statistics_reproduced=matched,
                reference_distributions=distributions,all_horizons=allrows,magnitude_bins=magnitude,horizon=horizons)
    (OUT/"sensitivity.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    # Keep each model on one row and put completion beside its name, without a Pairs column.
    lines=[r"\begin{tabular}{lccc}",r"\toprule",r"Model (CGM; PAS) & CGM error & BIS error & MAP error \\",r"\midrule"]
    for model in ORDER:
        data=[next(r for r in allrows if r["scope"]=="complete_pairs" and r["model"]==model and r["variable"]==v) for v in ["CGM","BIS","MAP"]]
        if model in ["persistence","ridge_history"]:lines.append(r"\midrule")
        label=NAMES[model]+r" {\scriptsize ("+f'{data[0]["available_pairs"]}/54; {data[1]["available_pairs"]}/54'+")}"
        lines.append(" & ".join([label]+[fmt(r["effect_error"]) for r in data])+r" \\")
    lines += [r"\bottomrule",r"\end{tabular}"]
    (ROOT/"paper/tables/paired_completion.tex").write_text("\n".join(lines)+"\n")
    sources += [ROOT/"data/simulated/benchmark_v1/test.inputs.jsonl",ROOT/"data/simulated/benchmark_v1/test.oracles.jsonl",ROOT/"data/results/study_v1/summary.json",ROOT/"repo/docs/revision_analysis.md"]
    (OUT/"analysis_manifest.json").write_text(json.dumps(dict(timestamp_utc=result["timestamp_utc"],
        analysis_status=result["analysis_status"],source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources},
        script_sha256=sha(Path(__file__)),original_statistics_reproduced=matched),indent=2)+"\n")
    if "--no-figures" in sys.argv:
        print(json.dumps(dict(original_statistics_reproduced=matched,figures_skipped=True)));return
    use_times();plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
    colors=dict(zip(MODELS,["#0072B2","#E69F00","#009E73","#CC79A7"]))
    fig,axes=plt.subplots(1,3,figsize=(8.1,3.8),sharey=True)
    for ax,(domain,var,title,unit) in zip(axes,PANELS):
        ax.axvline(0,color="#444",lw=.8,ls="--")
        for i,model in enumerate(ORDER):
            row=next(r for r in allrows if r["scope"]=="complete_pairs" and r["model"]==model and r["variable"]==var)
            r=row["excess_error"];x=r["mean"]
            if x is not None:
                ax.errorbar(x,i,xerr=[[x-r["lo"]],[r["hi"]-x]] if r["lo"] is not None else None,
                    fmt="o",color=colors.get(model,"#777"),ms=4,capsize=2)
            ax.text(.99,i,f'{row["available_pairs"]}/54',transform=ax.get_yaxis_transform(),ha="right",va="center",fontsize=8,
                    fontweight="bold" if row["available_pairs"]<2 else "normal",bbox=dict(facecolor="white",edgecolor="none",pad=.5))
        left,right=ax.get_xlim();ax.set_xlim(left,right+(right-left)*.32)
        ax.set_title(title);ax.set_xlabel("Effect-error excess ("+unit+")");ax.grid(axis="x",alpha=.18)
        ax.text(.99,1.02,"Complete",transform=ax.transAxes,ha="right",fontsize=8)
    axes[0].set_yticks(range(len(ORDER)),[NAMES[m] for m in ORDER]);axes[0].invert_yaxis()
    fig.tight_layout();save(fig,"paired_contrast_revision")
    fig,axes=plt.subplots(3,3,figsize=(8.1,7.6),gridspec_kw={"height_ratios":[1.15,1.0,.55]})
    for col,(domain,var,title,unit) in enumerate(PANELS):
        for model in MODELS:
            rr=[r for r in magnitude if r["scope"]=="available_points" and r["model"]==model and r["variable"]==var]
            for i,r in enumerate(rr):
                e=r["excess_error"]
                if e["mean"] is not None:
                    offset=(MODELS.index(model)-1.5)*.12
                    axes[0,col].errorbar(i+offset,e["mean"],yerr=[[e["mean"]-e["lo"]],[e["hi"]-e["mean"]]] if e["lo"] is not None else None,
                        fmt="o",color=colors[model],ms=3,capsize=1.5)
            hh=[r for r in horizons if r["scope"]=="available_points" and r["model"]==model and r["variable"]==var]
            axes[1,col].plot([r["step"] for r in hh],[r["excess_error"]["mean"] for r in hh],color=colors[model],label=NAMES[model],lw=1.2)
            axes[2,col].plot([r["step"] for r in hh],[r["available_pairs"] for r in hh],color=colors[model],lw=1.2)
        ticks=["0–<5","5–<20","20–<50","≥50"] if var=="CGM" else ["0–<1","1–<5","5–<10","≥10"]
        axes[0,col].set_xticks(range(4),ticks,rotation=25);axes[0,col].set_xlabel("Reference-effect magnitude ("+unit+")")
        axes[0,col].set_title(title)
        axes[1,col].set_xticks([1,3,6,9,12]);axes[1,col].set_xlabel("Rollout step")
        for ax in axes[:2,col]:
            ax.axhline(0,color="#444",lw=.8,ls="--");ax.set_ylabel("Error excess ("+unit+")");ax.grid(axis="y",alpha=.15)
        axes[2,col].set_xticks([1,3,6,9,12]);axes[2,col].set_xlabel("Rollout step")
        axes[2,col].set_ylim(-2,56);axes[2,col].set_yticks([0,27,54]);axes[2,col].set_ylabel("Available pairs / 54")
        axes[2,col].grid(axis="y",alpha=.15)
    handles,labels=axes[1,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=4,frameon=False,bbox_to_anchor=(.5,-.015))
    fig.tight_layout(rect=(0,.05,1,1));save(fig,"effect_sensitivity_revision")
    primary=[r for r in magnitude if r["model"]==MODELS[0] and r["variable"]=="CGM" and r["scope"]=="available_points"]
    print(json.dumps(dict(original_statistics_reproduced=matched,reference_distributions=distributions,primary_magnitude=[dict(bin=r["bin"],points=r["available_points"],pairs=r["available_pairs"],clusters=r["available_clusters"],excess=r["excess_error"]) for r in primary]),indent=2))
if __name__=="__main__":main()
