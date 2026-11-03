#!/usr/bin/env python3
"""Generate manuscript tables and numeric macros directly from verified analysis."""
from pathlib import Path
import json,hashlib,shutil,re
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/"data/results/study_v1"
OUT=SRC/"tables";OUT.mkdir(exist_ok=True)
PUB=ROOT/"paper/tables";PUB.mkdir(parents=True,exist_ok=True)
MODELS=["Qwen/Qwen3-8B","Qwen/Qwen2.5-7B-Instruct","mistralai/Mistral-7B-Instruct-v0.3","microsoft/Phi-3.5-mini-instruct"]
NAMES=dict(zip(MODELS,["Qwen3-8B","Qwen2.5-7B","Mistral-7B","Phi-3.5-mini"]))|{"persistence":"Persistence","trend":"Linear trend","ridge_history":"Ridge history","ridge_action":"Ridge actions","boosted_action":"Boosted trees"}
def escape(s):return str(s).replace("_",r"\_").replace("%",r"\%").replace("&",r"\&")
def main():
    if json.loads((SRC/"analysis_manifest.json").read_text())["partial"]:raise RuntimeError("Incomplete analysis")
    s=json.loads((SRC/"summary.json").read_text())
    def get(result_type,metric,**kw):
        found=[r for r in s if r["type"]==result_type and r.get("metric")==metric and all(r.get(k)==v for k,v in kw.items())]
        if len(found)!=1:return None
        return found[0]
    def fmt(row,pct=False,ci=True):
        if row is None or row["mean"] is None:return "---"
        mult=100 if pct else 1
        ans=f'{row["mean"]*mult:.1f}'
        if ci and row["clusters"]>1:ans+=f' [{row["lo"]*mult:.1f}, {row["hi"]*mult:.1f}]'
        return ans
    def table(name,headers,rows,align=None):
        align=align or "l"+"c"*(len(headers)-1)
        def compact(cell):
            return re.sub(r"(-?[\d.]+) \[(-?[\d.]+), (-?[\d.]+)\]",lambda m:m[1]+r" {\scriptsize ["+m[2]+", "+m[3]+"]}",str(cell))
        rows=[[compact(cell) for cell in row] for row in rows]
        text=r"\begin{tabular}{"+align+"}\n"+r"\toprule"+"\n"+" & ".join(headers)+r" \\"+"\n"+r"\midrule"+"\n"
        text+="\n".join(" & ".join(map(str,row))+r" \\" for row in rows)
        text+="\n"+r"\bottomrule"+"\n"+r"\end{tabular}"+"\n"
        p=OUT/(name+".tex");p.write_text(text);shutil.copy2(p,PUB/p.name)
    order=MODELS+["persistence","trend","ridge_history","ridge_action","boosted_action"]
    pairs=pd.read_parquet(SRC/"pairs.parquet")
    rows=[]
    for model in order:
        cond="state_only" if model=="ridge_history" else "state_action"
        vals=[]
        for d,v in [("simglucose","CGM"),("anesthesia","BIS"),("anesthesia","MAP")]:
            vals.append(fmt(get("paired","effect_mae",model=model,domain=d,variable=v,condition=cond,mode="free",kind="dose")))
        counts=[]
        for d in ["simglucose","anesthesia"]:
            df=pairs[(pairs.model==model)&(pairs.domain==d)&(pairs.condition==cond)&(pairs["mode"]=="free")&(pairs.kind=="dose")]
            df=df.drop_duplicates("group")
            counts.append(str(int(df.complete.sum()))+"/54")
        rows.append([NAMES[model],vals[0],vals[1],vals[2]])
    table("paired",["Model","CGM error","BIS error","MAP error"],rows)
    for name,panels in [("trajectory_controlled",[("simglucose","CGM"),("anesthesia","BIS"),("anesthesia","MAP")]),
                        ("trajectory_real",[("vitaldb","BIS"),("vitaldb","MAP")])]:
        rows=[]
        for model in order:
            cond="state_only" if model=="ridge_history" else "state_action"
            vals=[]
            for d,v in panels:
                kw=dict(model=model,domain=d,variable=v,condition=cond,mode="free")
                vals.append(fmt(get("trajectory","mae",**kw))+" / "+fmt(get("trajectory","complete",**kw),True,False)+r"\%")
            rows.append([NAMES[model]]+vals)
        table(name,["Model"]+[({"simglucose":"Glucose","anesthesia":"PAS","vitaldb":"VitalDB"}[d])+" "+v for d,v in panels],rows)
    rows=[]
    for model in MODELS:
        for d in ["simglucose","anesthesia","vitaldb"]:
            state=get("auxiliary","state_accuracy",model=model,domain=d)
            knowledge=get("auxiliary","knowledge_accuracy",model=model,domain=d)
            rows.append([NAMES[model],{"simglucose":"Glucose","anesthesia":"PAS","vitaldb":"VitalDB"}[d],fmt(state,True),fmt(knowledge,True),str(state["clusters"]) if state else "---"])
    table("auxiliary",["Model","Domain","State (\\%)","Knowledge (\\%)","Clusters"],rows)
    rows=[]
    for model in MODELS:
        for d,v in [("simglucose","CGM"),("anesthesia","BIS"),("anesthesia","MAP"),("vitaldb","BIS"),("vitaldb","MAP")]:
            kw=dict(model=model,domain=d,variable=v,condition="state_action",mode="free")
            rows.append([NAMES[model],({"simglucose":"Glucose","anesthesia":"PAS","vitaldb":"VitalDB"}[d])+" "+v,fmt(get("interval","covered",**kw),True),fmt(get("interval","unconditional_covered",**kw),True),fmt(get("interval","width",**kw)),fmt(get("interval","violation",**kw),True,False)])
    table("calibration",["Model","Output","Valid (\\%)","All (\\%)","Width","Viol. (\\%)"],rows)
    for d,v,tag in [("simglucose","CGM","glucose"),("anesthesia","BIS","pas_bis"),("anesthesia","MAP","pas_map"),("vitaldb","BIS","real_bis"),("vitaldb","MAP","real_map")]:
        rows=[]
        for model in order:
            conditions=["state_only"] if model=="ridge_history" else ["state_action","state_only"] if model in MODELS else ["state_action"]
            for condition in conditions:
                for mode in ["free","teacher"]:
                    if condition=="state_only" and model in MODELS and mode=="teacher":continue
                    kw=dict(model=model,domain=d,variable=v,condition=condition,mode=mode)
                    vals=[fmt(get("step","abs_error",**kw,step=h,minutes=h*(30 if d=="simglucose" else 2 if d=="anesthesia" else 1))) for h in [1,3,6,12]]
                    rows.append([NAMES[model],("SA" if condition=="state_action" else "S"),("FR" if mode=="free" else "TF")]+vals)
        table("steps_"+tag,["Model","Input","Mode","Step 1","Step 3","Step 6","Step 12"],rows)
    # Plain-text facts for authoring; every entry retains full metric keys and CI.
    facts={}
    for r in s:
        if r["type"] in ["paired","feedback_contrast","information_contrast","auxiliary","diagnostics","event"] and r.get("model") in MODELS:
            key="|".join(str(r.get(k,"")) for k in ["type","model","domain","variable","condition","mode","kind","metric","variant","step","event"])
            facts[key]=r
    (SRC/"manuscript_facts.json").write_text(json.dumps(facts,indent=2))
    primary=dict(model=MODELS[0],domain="simglucose",variable="CGM",condition="state_action",mode="free",kind="dose")
    macros={
       "PrimaryEffect":fmt(get("paired","effect_mae",**primary)),
       "PrimaryContrast":fmt(get("paired","delta_vs_persistence",**primary)),
       "PrimaryCoverage":fmt(get("paired","complete",**primary),True,False),
       "PrimaryZero":fmt(get("paired","zero_effect_mae",**primary)),
    }
    (ROOT/"paper/numbers.tex").write_text("\n".join(r"\newcommand{\%s}{%s}"%(k,v) for k,v in macros.items())+"\n")
    (SRC/"table_provenance.json").write_text(json.dumps(dict(source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),summary_sha256=hashlib.sha256((SRC/"summary.json").read_bytes()).hexdigest(),tables={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in PUB.glob("*.tex")},macros=macros),indent=2))
if __name__=="__main__":main()
