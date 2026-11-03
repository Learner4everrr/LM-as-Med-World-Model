#!/usr/bin/env python3
"""Publication figures computed from saved results; vector PDF plus PNG."""
from pathlib import Path
import os,json,hashlib,shutil,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"repo/src"))
os.environ["MPLCONFIGDIR"]=str(ROOT/"data/cache/matplotlib")
from medwm_eval.figure_style import use_times
use_times()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
import pandas as pd
import numpy as np
MODELS=["Qwen/Qwen3-8B","Qwen/Qwen2.5-7B-Instruct","mistralai/Mistral-7B-Instruct-v0.3","microsoft/Phi-3.5-mini-instruct"]
NAMES=dict(zip(MODELS,["Qwen3-8B","Qwen2.5-7B","Mistral-7B","Phi-3.5-mini"]))|{"persistence":"Persistence","trend":"Linear trend","ridge_history":"Ridge (history)","ridge_action":"Ridge (actions)","boosted_action":"Boosted trees"}
COLORS=dict(zip(MODELS,["#0072B2","#D55E00","#009E73","#CC79A7"]))|{"persistence":"#333333","boosted_action":"#777777"}
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"ps.fonttype":42,"axes.titlesize":11})
OUT=ROOT/"data/results/study_v1/figures";OUT.mkdir(parents=True,exist_ok=True)
PUB=ROOT/"paper/figures";PUB.mkdir(parents=True,exist_ok=True)
def save(fig,name):
    for ext in ["pdf","png"]:
        p=OUT/(name+"."+ext);fig.savefig(p,bbox_inches="tight",dpi=300)
        shutil.copy2(p,PUB/p.name)
    plt.close(fig)
def schematic():
    fig,ax=plt.subplots(figsize=(7.1,3.6));ax.set_xlim(0,10);ax.set_ylim(0,6);ax.axis("off")
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.08",facecolor=color,edgecolor="#777",linewidth=.7))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=9.5)
    def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=10,color="#555",linewidth=.8))
    ax.text(5,5.85,"C1 state extraction | C2 transitions | C3/C4 paired effects | C5 recurrence",ha="center",fontsize=9)
    box(.1,3.9,2,1.2,"Same snapshot\nVisible history\nFixed events","#E8EEF5")
    box(3,4.65,2.1,.75,"Prescribed\naction A","#E5F3EC")
    box(3,3.25,2.1,.75,"Prescribed\naction B","#F9EBE4")
    box(6,4.65,3.7,.75,"Resettable simulator\nReference A and B","#E8EEF5")
    box(6,3.25,3.7,.75,"Frozen LLM dynamics\nPredicted A and B","#FFF2CC")
    arrow((2.18,4.65),(2.9,5));arrow((2.18,4.3),(2.9,3.7))
    arrow((5.18,5),(5.9,5));arrow((5.18,3.7),(5.9,3.7))
    arrow((4.6,4.65),(6.05,4));arrow((4.6,4),(6.05,4.65))
    box(.1,1.1,4.4,1.25,"12-step recurrence\nSelf-fed: predicted states\nTeacher-forced: reference states","#FFF2CC")
    box(5.25,1.1,4.45,1.25,"VitalDB factual replay\nCurrent action revealed at each step\nNo alternate-action reference","#ECE7F2")
    ax.text(5,2.8,"Score action effects and absolute trajectories; retain failed outputs",ha="center",fontsize=10,fontweight="bold")
    ax.text(5,.35,"CGM: 30 min/step   |   PAS: 2 min/step   |   VitalDB: 1 min/step",ha="center",fontsize=9.5)
    save(fig,"design")
def main():
    source=ROOT/"data/results/study_v1"
    manifest=json.loads((source/"analysis_manifest.json").read_text())
    if manifest["partial"]:raise RuntimeError("Final figures require completed analysis")
    s=pd.DataFrame(json.loads((source/"summary.json").read_text()))
    pr=pd.read_parquet(source/"pairs.parquet");pt=pd.read_parquet(source/"points.parquet")
    schematic()
    order=MODELS+["persistence","trend","ridge_history","ridge_action","boosted_action"]
    fig,axes=plt.subplots(1,3,figsize=(7.2,4.3),sharey=True)
    for ax,(domain,var,title) in zip(axes,[("simglucose","CGM","Glucose (mg/dL)"),("anesthesia","BIS","PAS BIS (units)"),("anesthesia","MAP","PAS MAP (mmHg)")]):
        for i,model in enumerate(order):
            df=pr[(pr.model==model)&(pr.domain==domain)&(pr.variable==var)&(pr["mode"]=="free")]
            if model!="ridge_history":df=df[df.condition=="state_action"]
            df=df[df.kind=="dose"]
            m=s[(s.type=="paired")&(s.model==model)&(s.domain==domain)&(s.variable==var)&(s["mode"]=="free")&(s.kind=="dose")&(s.metric=="effect_mae")]
            m=m[m.condition==("state_only" if model=="ridge_history" else "state_action")]
            if len(m) and pd.notna(m.iloc[0]["mean"]):
                r=m.iloc[0];ax.errorbar(r["mean"],i,xerr=([[r["mean"]-r.lo],[r.hi-r["mean"]]] if r.clusters>1 else None),fmt="o",ms=4,color=COLORS.get(model,"#777"),capsize=2)
            if len(df):ax.text(.98,i,str(int(df.complete.sum()))+"/54",transform=ax.get_yaxis_transform(),ha="right",va="center",fontsize=8,color="#555")
        ax.set_title(chr(97+list(axes).index(ax))+") "+title);ax.set_xlabel("Paired-effect MAE");ax.set_xlim(0,ax.get_xlim()[1]*1.3);ax.grid(axis="x",alpha=.2)
    axes[0].set_yticks(range(len(order)),[NAMES[x] for x in order]);axes[0].invert_yaxis()
    fig.suptitle("Dose-pair fidelity and complete-pair counts",fontsize=11,y=1.01)
    fig.tight_layout();save(fig,"paired_fidelity")
    for name,panels in [("controlled_rollouts",[("simglucose","CGM","Glucose (mg/dL)"),("anesthesia","BIS","PAS BIS"),("anesthesia","MAP","PAS MAP (mmHg)")]),
                        ("factual_rollouts",[("vitaldb","BIS","VitalDB BIS"),("vitaldb","MAP","VitalDB MAP (mmHg)")])]:
        fig,axes=plt.subplots(2,len(panels),figsize=(7.2,4.5),gridspec_kw={"height_ratios":[2.3,1]},squeeze=False)
        for col,(domain,var,title) in enumerate(panels):
            for model in MODELS+["persistence","boosted_action"]:
                for mode,linestyle in [("free","-"),("teacher","--")]:
                    if model not in MODELS and mode=="teacher":continue
                    g=s[(s.type=="step")&(s.model==model)&(s.domain==domain)&(s.variable==var)&(s.condition=="state_action")&(s["mode"]==mode)&(s.metric=="abs_error")].sort_values("step")
                    if len(g):axes[0,col].plot(g.minutes,g["mean"],linestyle,color=COLORS[model],lw=1.2,label=NAMES[model] if mode=="free" else None)
                g=s[(s.type=="step")&(s.model==model)&(s.domain==domain)&(s.variable==var)&(s.condition=="state_action")&(s["mode"]=="free")&(s.metric=="available")].sort_values("step")
                if len(g):axes[1,col].plot(g.minutes,g["mean"]*100,color=COLORS[model],lw=1.2)
            axes[0,col].set_title(chr(97+col)+") "+title);axes[0,col].set_ylim(bottom=0)
            axes[1,col].text(.02,.8,chr(97+len(panels)+col)+")",transform=axes[1,col].transAxes,fontweight="bold",bbox=dict(facecolor="white",edgecolor="none",alpha=.8,pad=1))
            axes[1,col].set_ylim(-3,103);axes[1,col].set_xlabel("Minutes from anchor")
            for ax in axes[:,col]:ax.grid(alpha=.18)
        axes[0,0].set_ylabel("Available-point MAE")
        axes[1,0].set_ylabel("Free outputs (%)")
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc="lower center",ncol=3,fontsize=9,bbox_to_anchor=(.5,-.08))
        fig.suptitle("Solid: self-fed; dashed: teacher-forced. Errors are conditional on available outputs.",fontsize=9)
        fig.tight_layout();save(fig,name)
    # Examples are selected by input order, never by prediction error.
    cases=[]
    for folder in ["data/simulated/benchmark_v1"]:
        cases+=[json.loads(l) for l in (ROOT/folder/"test.inputs.jsonl").read_text().splitlines()]
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.65))
    for ax,(domain,var) in zip(axes,[("simglucose","CGM"),("anesthesia","BIS"),("anesthesia","MAP")]):
        c=next(x for x in cases if x["domain"]==domain and x["kind"]=="dose")
        for branch,color in [("A","#0072B2"),("B","#D55E00")]:
            g=pt[(pt.model==MODELS[0])&(pt.group==c["group"])&(pt.branch==branch)&(pt.variable==var)&(pt.condition=="state_action")&(pt["mode"]=="free")].sort_values("step")
            ax.plot(g.minutes,g.truth,color=color,lw=1.5,label="Oracle "+branch)
            ax.plot(g.minutes,g.pred,color=color,lw=1,ls="--",label=NAMES[MODELS[0]]+" "+branch)
        ax.set_title(chr(97+list(axes).index(ax))+") "+domain+" / "+var);ax.set_xlabel("Minutes");ax.grid(alpha=.2)
    axes[0].set_ylabel("Native units")
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc="lower center",bbox_to_anchor=(.5,-.12),fontsize=9)
    fig.tight_layout();save(fig,"fixed_examples")
    (source/"figure_provenance.json").write_text(json.dumps({"source_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"source_analysis_sha256":hashlib.sha256((source/"summary.json").read_bytes()).hexdigest(),"figures":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in PUB.glob("*.pdf")}},indent=2))
if __name__=="__main__":main()
