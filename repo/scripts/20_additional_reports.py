#!/usr/bin/env python3
"""Additional manuscript diagnostics, generated only from final verified results."""
from pathlib import Path
import json,re,hashlib,sys,os,shutil
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"repo/src"))
from medwm_eval.metrics import cluster_ci
from medwm_eval.figure_style import use_times
os.environ["MPLCONFIGDIR"]=str(ROOT/"data/cache/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
MODELS=["Qwen/Qwen3-8B","Qwen/Qwen2.5-7B-Instruct","mistralai/Mistral-7B-Instruct-v0.3","microsoft/Phi-3.5-mini-instruct"]
NAMES=dict(zip(MODELS,["Qwen3-8B","Qwen2.5-7B","Mistral-7B","Phi-3.5-mini"]))|{"persistence":"Persistence","trend":"Linear trend","ridge_history":"Ridge history","ridge_action":"Ridge actions","boosted_action":"Boosted trees"}
DOM={"simglucose":"Glucose","anesthesia":"PAS","vitaldb":"VitalDB"}
PANELS=[("simglucose","CGM"),("anesthesia","BIS"),("anesthesia","MAP"),("vitaldb","BIS"),("vitaldb","MAP")]
COLORS=["#0072B2","#D55E00","#009E73","#CC79A7"]
SRC=ROOT/"data/results/study_v1";STRESS=ROOT/"data/results/stress_v1"
OUT=SRC/"tables";PUB=ROOT/"paper/tables"
GENERATED=[]
def fmt(r,pct=False,ci=True):
    if r is None or r["mean"] is None or not np.isfinite(r["mean"]):return "---"
    mult=100 if pct else 1
    x=f'{r["mean"]*mult:.1f}'
    if ci and r["clusters"]>1:x+=f' [{r["lo"]*mult:.1f}, {r["hi"]*mult:.1f}]'
    return x
def table(name,headers,rows):
    OUT.mkdir(parents=True,exist_ok=True);PUB.mkdir(parents=True,exist_ok=True)
    def compact(cell):
        return re.sub(r"(-?[\d.]+) \[(-?[\d.]+), (-?[\d.]+)\]",lambda m:m[1]+r" {\scriptsize ["+m[2]+", "+m[3]+"]}",str(cell))
    text=r"\begin{tabular}{l"+"c"*(len(headers)-1)+"}\n"+r"\toprule"+"\n"
    text+=" & ".join(headers)+r" \\"+"\n"+r"\midrule"+"\n"
    text+="\n".join(" & ".join(compact(x) for x in row)+r" \\" for row in rows)
    text+="\n"+r"\bottomrule"+"\n"+r"\end{tabular}"+"\n"
    (OUT/(name+".tex")).write_text(text);(PUB/(name+".tex")).write_text(text)
    GENERATED.append(PUB/(name+".tex"))
def main():
    for p in [SRC,STRESS]:
        assert not json.loads((p/"analysis_manifest.json").read_text())["partial"]
        assert json.loads((p/"integrity.json").read_text())["no_future_oracle_in_free_prompts_verified"]
    OUT.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True)
    summary=json.loads((SRC/"summary.json").read_text())
    stress=json.loads((STRESS/"summary.json").read_text())
    def get(result_type,metric,source=summary,**kw):
        rows=[r for r in source if r["type"]==result_type and r.get("metric")==metric and all(r.get(k)==v for k,v in kw.items())]
        if len(rows)>1:raise ValueError(("ambiguous",result_type,metric,kw))
        return rows[0] if rows else None
    content=json.loads((SRC/"auxiliary_content_summary.json").read_text())
    def cget(metric,model,domain):
        return next(r for r in content if r["metric"]==metric and r["model"]==model and r["domain"]==domain)
    rows=[]
    for m in MODELS:
        for d in ["simglucose","anesthesia","vitaldb"]:
            rows.append([DOM[d],NAMES[m],fmt(get("auxiliary","state_accuracy",model=m,domain=d),True),
                fmt(cget("state_content",m,d),True),fmt(get("auxiliary","knowledge_accuracy",model=m,domain=d),True),
                fmt(cget("knowledge_content",m,d),True)])
    table("auxiliary_content",["Domain","Model","State strict","State content","Direction strict","Direction content"],rows)
    rows=[]
    for d,v in PANELS:
        for m in MODELS:
            kw=dict(model=m,domain=d,variable=v)
            rows.append([DOM[d]+" "+v,NAMES[m],fmt(get("feedback_contrast","free_minus_teacher",**kw)),
                         fmt(get("information_contrast","action_minus_stateonly",**kw))])
    table("contrasts",["Output","Model","FR minus TF","SA minus S"],rows)
    rows=[]
    for d,v in PANELS[:3]:
        for m in MODELS:
            kw=dict(model=m,domain=d,variable=v,condition="state_action",mode="free")
            rows.append([DOM[d]+" "+v,NAMES[m],
                         fmt(get("direction","direction_1",kind="dose",**kw),True),
                         fmt(get("direction","direction_unconditional_1",kind="dose",**kw),True),
                         fmt(get("ranking","rank_unconditional",**kw),True),
                         fmt(get("paired","effect_mae",kind="timing",**kw))])
    table("action_diagnostics",["Output","Model","Sign valid (\\%)","Sign all (\\%)","Rank all (\\%)","Timing error"],rows)
    rows=[]
    tr=pd.read_parquet(STRESS/"trajectories.parquet")
    for m in MODELS+["persistence","trend","ridge_history","ridge_action","boosted_action"]:
        kw=dict(model=m,domain="simglucose",variable="CGM",condition="state_only" if m=="ridge_history" else "state_action",mode="free")
        g=tr[(tr.model==m)&(tr.condition==kw["condition"])&(tr["mode"]=="free")]
        rows.append([NAMES[m],str(int(g.complete.sum()))+"/36",
                     fmt(get("trajectory","mae",source=stress,**kw)),
                     fmt(get("paired","effect_mae",source=stress,kind="hypoglycemia_stress",**kw)),
                     fmt(get("event","miss_rate",source=stress,event="low_glucose",**kw),True)])
    table("stress",["Model","Complete","CGM MAE","Paired error","Low-event miss (\\%)"],rows)
    for tag,panels in [("controlled",PANELS[:3]),("real",PANELS[3:])]:
        rows=[]
        for d,v in panels:
            for m in MODELS:
                vals=[]
                for variant in ["missing_previous","noisy_history","time_shift_60","pretty_json"]:
                    kw=dict(model=m,domain=d,variable=v,variant=variant,step=1)
                    vals.append(fmt(get("diagnostics","error_change",**kw))+" / "+fmt(get("diagnostics","available",**kw),True,False))
                rows.append([DOM[d]+" "+v,NAMES[m]]+vals)
        table("robustness_"+tag,["Output","Model","Missing","Noise","Time shift","JSON spacing"],rows)
    rows=[]
    for d,v in PANELS[:3]:
        for m in MODELS:
            kw=dict(model=m,domain=d,variable=v,variant="one_shot")
            rows.append([DOM[d]+" "+v,NAMES[m],
                fmt(get("diagnostics","available",step=1,**kw),True,False)]+
                [fmt(get("diagnostics","mae",step=h,**kw)) for h in [1,3,6,12]])
    table("oneshot",["Output","Model","Valid (\\%)","Step 1","Step 3","Step 6","Step 12"],rows)
    rows=[]
    for d,v in [("simglucose","CGM"),("anesthesia","BIS"),("vitaldb","BIS")]:
        for m in MODELS:
            kw=dict(model=m,domain=d,variable=v)
            rows.append([DOM[d],NAMES[m],
                fmt(get("trajectory","complete",condition="state_action",mode="free",**kw),True),
                fmt(get("trajectory","available_fraction",condition="state_action",mode="teacher",**kw),True),
                fmt(get("trajectory","complete",condition="state_only",mode="free",**kw),True)])
    table("availability",["Domain","Model",r"SA FR (\%)",r"SA TF (\%)",r"S FR (\%)"],rows)
    # Descriptive decomposition of the already specified sign endpoint.
    pairpoints=pd.read_parquet(SRC/"pair_points.parquet")
    decomposed=[];rows=[]
    for d,v in PANELS[:3]:
        for m in MODELS:
            g=pairpoints[(pairpoints.domain==d)&(pairpoints.variable==v)&(pairpoints.model==m)&
                         (pairpoints.condition=="state_action")&(pairpoints["mode"]=="free")&
                         (pairpoints.kind=="dose")&(pairpoints.truth_effect.abs()>1)].copy()
            valid=np.isfinite(g.pred_effect)
            masks={"correct":valid&(np.sign(g.pred_effect)==np.sign(g.truth_effect)),
                   "zero":valid&(g.pred_effect==0),
                   "reverse":valid&(np.sign(g.pred_effect)!=np.sign(g.truth_effect))&(g.pred_effect!=0),
                   "missing":~valid}
            assert np.array(list(masks.values())).sum(axis=0).tolist()==[1]*len(g)
            r=dict(model=m,domain=d,variable=v)
            for key,values in masks.items():r[key]=cluster_ci(values.astype(float),g.patient)
            decomposed.append(r)
            rows.append([DOM[d]+" "+v,NAMES[m]]+[fmt(r[k],True) for k in ["correct","zero","reverse","missing"]])
    table("sign_decomposition",["Output","Model",r"Correct (\%)",r"Zero (\%)",r"Reverse (\%)",r"Missing (\%)"],rows)
    (SRC/"sign_decomposition.json").write_text(json.dumps(decomposed,indent=2))
    # False-positive and precision descriptions supplement the frozen miss-rate endpoint.
    pt=pd.read_parquet(SRC/"points.parquet");event_records=[]
    names=[("CGM","high_glucose",180,False),("BIS","low_BIS",40,True),("BIS","high_BIS",60,False),("MAP","low_MAP",65,True)]
    for d in ["simglucose","anesthesia","vitaldb"]:
        rows=[]
        for v,name,cut,low in names:
            for m in MODELS+["persistence","boosted_action"]:
                g=pt[(pt.domain==d)&(pt.variable==v)&(pt.model==m)&(pt.condition=="state_action")&(pt["mode"]=="free")].copy()
                if g.empty:continue
                g["event"]=g.truth<cut if low else g.truth>cut
                g["detect"]=g.pred<cut if low else g.pred>cut
                per=[]
                for patient,p in g.groupby("patient"):
                    positive=p[p.event];negative=p[~p.event];predicted=p[p.detect]
                    per.append(dict(patient=patient,miss_rate=1-positive.detect.mean(),
                                    false_positive_rate=negative.detect.mean(),precision=predicted.event.mean(),
                                    reference_points=len(positive),predicted_points=len(predicted)))
                per=pd.DataFrame(per)
                rec=dict(model=m,domain=d,variable=v,event=name,
                         positive_clusters=int((per.reference_points>0).sum()),positive_points=int(per.reference_points.sum()))
                for metric in ["miss_rate","false_positive_rate","precision"]:
                    rec[metric]=cluster_ci(per[metric],per.patient)
                event_records.append(rec)
                rows.append([NAMES[m],name.replace("_",r"\_"),str(rec["positive_points"])+"/"+str(rec["positive_clusters"]),
                             fmt(rec["miss_rate"],True),fmt(rec["false_positive_rate"],True),fmt(rec["precision"],True)])
        table("events_"+d,["Model","Event","Points/subjects","Miss (\\%)","FP (\\%)","Precision (\\%)"],rows)
    (SRC/"event_diagnostics.json").write_text(json.dumps(event_records,indent=2))
    stress_points=pd.read_parquet(STRESS/"points.parquet");rows=[];stress_events=[]
    for m in MODELS+["persistence","trend","ridge_history","ridge_action","boosted_action"]:
        cond="state_only" if m=="ridge_history" else "state_action"
        g=stress_points[(stress_points.model==m)&(stress_points.condition==cond)&(stress_points["mode"]=="free")].copy()
        g["event"]=g.truth<70;g["detect"]=g.pred<70
        per=[]
        for patient,p in g.groupby("patient"):
            per.append(dict(patient=patient,miss_rate=1-p[p.event].detect.mean(),
                            false_positive_rate=p[~p.event].detect.mean(),precision=p[p.detect].event.mean()))
        per=pd.DataFrame(per);r=dict(model=m)
        for metric in ["miss_rate","false_positive_rate","precision"]:r[metric]=cluster_ci(per[metric],per.patient)
        stress_events.append(r)
        rows.append([NAMES[m],fmt(r["miss_rate"],True),fmt(r["false_positive_rate"],True),fmt(r["precision"],True)])
    table("stress_events",["Model","Low-event miss (\\%)","False-positive (\\%)","Precision (\\%)"],rows)
    (STRESS/"event_diagnostics.json").write_text(json.dumps(stress_events,indent=2))
    # Compute and attempted-output audit.
    rows=[]
    for label,folder in [("Main",SRC),("Stress",STRESS)]:
        integrity=json.loads((folder/"integrity.json").read_text())
        for r in integrity["models"]:
            c=r["counts"]
            rows.append([label,NAMES[r["model"]],f'{r["unique_raw_calls"]:,}',
                         f'{r["generation_seconds"]/3600:.2f}',
                         f'{r["input_tokens"]/1e6:.2f}',f'{r["generated_token_slots_upper_bound"]/1e6:.2f}',
                         str(c["valid"])+"/"+str(c["attempted"])])
    table("compute",["Run","Model","Calls","Gen h","Input M","Output M*","Valid/attempted"],rows)
    # Narrow qualitative knowledge versus executed effect direction, without a combined score.
    use_times()
    plt.rcParams.update({"font.size":10,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(2,3,figsize=(7.2,5.2),sharey=True)
    for col,(d,v) in enumerate(PANELS[:3]):
        for i,m in enumerate(MODELS):
            plotted=[(0,get("auxiliary","knowledge_accuracy",model=m,domain=d),-.1,False),
                     (0,cget("knowledge_content",m,d),.1,True),
                     (1,get("direction","direction_unconditional_1",model=m,domain=d,variable=v,condition="state_action",mode="free",kind="dose"),0,True)]
            for row,r,shift,fill in plotted:
                if r and r["mean"] is not None:
                    axes[row,col].errorbar(i+shift,r["mean"]*100,yerr=[[100*(r["mean"]-r["lo"])],[100*(r["hi"]-r["mean"])]],
                                          fmt="o",color=COLORS[i],mfc=COLORS[i] if fill else "white",capsize=3,ms=5)
        axes[0,col].set_title(chr(97+col)+") "+DOM[d]+" "+v)
        axes[1,col].text(.02,.8,chr(100+col)+")",transform=axes[1,col].transAxes,fontweight="bold",bbox=dict(facecolor="white",edgecolor="none",alpha=.8,pad=1))
        for row in range(2):
            axes[row,col].set_ylim(-4,104);axes[row,col].set_xticks(range(4),[NAMES[m] for m in MODELS],rotation=45,ha="right",rotation_mode="anchor",fontsize=8);axes[row,col].grid(axis="y",alpha=.2)
    axes[0,0].set_ylabel("Direction answer (%)")
    axes[1,0].set_ylabel("Executed effect sign (%)")
    from matplotlib.lines import Line2D
    fig.legend([Line2D([],[],marker="o",mfc="white",color="#555",ls=""),Line2D([],[],marker="o",color="#555",ls="")],
               ["Top: strict response","Top: content audit (post hoc)"],loc="lower center",ncol=2,fontsize=9)
    fig.tight_layout(rect=[0,.07,1,1])
    out=SRC/"figures";pub=ROOT/"paper/figures"
    for ext in ["pdf","png"]:
        p=out/("knowledge_action."+ext);fig.savefig(p,bbox_inches="tight",dpi=300);shutil.copy2(p,pub/p.name)
    plt.close(fig)
    (SRC/"additional_report_manifest.json").write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [SRC/"summary.json",STRESS/"summary.json",SRC/"integrity.json",STRESS/"integrity.json",SRC/"auxiliary_content_summary.json"]},
        outputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in GENERATED+[pub/"knowledge_action.pdf",pub/"knowledge_action.png"]}),indent=2))
if __name__=="__main__":main()
