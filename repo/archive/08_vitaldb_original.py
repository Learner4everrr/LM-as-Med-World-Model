#!/usr/bin/env python3
"""Local-only VitalDB factual replay; raw patient data never sent to hosted APIs."""
from pathlib import Path
import os,sys,json,hashlib,urllib.request,gzip,io,traceback
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[2]
import numpy as np
import pandas as pd
RAW=ROOT/"data/raw/vitaldb"
DEST=ROOT/"data/processed/vitaldb_v1"
TRACKS=["BIS/BIS","BIS/SQI","Solar8000/ART_MBP","Orchestra/PPF20_RATE","Orchestra/RFTN20_RATE"]

def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()

def csv_bytes(path):
    b=path.read_bytes()
    return gzip.decompress(b) if b[:2]==b"\x1f\x8b" else b

def fetch(tid):
    p=RAW/"tracks"/(tid+".csv.gz")
    p.parent.mkdir(exist_ok=True,parents=True)
    if not p.exists():
        b=urllib.request.urlopen("https://api.vitaldb.net/"+tid,timeout=60).read()
        p.write_bytes(b)
    df=pd.read_csv(io.BytesIO(csv_bytes(p)))
    arr=df.iloc[:,:2].apply(pd.to_numeric,errors="coerce").dropna().values
    arr=arr[np.argsort(arr[:,0],kind="stable")]
    return arr

def asof(arr,times,max_age):
    ix=np.searchsorted(arr[:,0],times,side="right")-1
    good=ix>=0
    ix=np.maximum(ix,0)
    age=times-arr[ix,0]
    good &= (age>=0)&(age<=max_age)
    return np.where(good,arr[ix,1],np.nan)

def case_rows(case,trks):
    caseid=int(case.caseid); subject=str(int(case.subjectid))
    chosen=trks[trks.caseid==caseid].set_index("tname")
    with ThreadPoolExecutor(max_workers=5) as pool:
        arrays=dict(zip(TRACKS,pool.map(fetch,[str(chosen.loc[t,"tid"]) for t in TRACKS])))
    end=min(x[-1,0] for x in arrays.values())
    grid=np.arange(0,np.floor(end/60)*60+1,60)
    values={t:asof(a,grid,15 if ("BIS" in t or "MBP" in t) else 90) for t,a in arrays.items()}
    bis=values["BIS/BIS"]; mbp=values["Solar8000/ART_MBP"]
    # Official concentration labels: PPF20=20 mg/mL, RFTN20=20 ug/mL, rates mL/hour.
    acts=np.column_stack([values["Orchestra/PPF20_RATE"]*20/3600,
                          values["Orchestra/RFTN20_RATE"]*20/3600])
    obs=np.column_stack([bis,mbp])
    valid=np.isfinite(obs).all(axis=1)&np.isfinite(acts).all(axis=1)&(values["BIS/SQI"]>=80)
    valid &= (bis>=0)&(bis<=100)&(mbp>=20)&(mbp<=200)&(acts>=0).all(axis=1)
    candidates=[i for i in range(15,len(grid)-12) if valid[i-10:i+13].all() and acts[i,0]>0]
    # Freeze selection using availability and action changes, never future target values beyond artifact bounds.
    change=[i for i in candidates if abs(acts[i,0]-acts[i-1,0])>=.003]
    selected=[]
    for kind,possible in [("action_change",change),("factual",candidates)]:
        for i in possible:
            if all(abs(i-j)>=30 for _,j in selected):
                selected.append((kind,i));break
    if not selected:
        return [],{"caseid":caseid,"subjectid":subject,"status":"excluded","reason":"no complete quality-filtered window"}
    split="development" if int(digest(subject+":split20261102"),16)%3==0 else "test"
    rows=[]
    for kind,i in selected:
        identifier=digest(f"vitaldb:{caseid}:{i}:v1")[:24]
        history=[]
        for j in range(i-10,i+1):
            history.append({"time":int(j-i),"values":obs[j].round(6).tolist(),
                            "past_action":acts[j-1].round(6).tolist(),"past_meal_g":0})
        actions=[{"time":j,"values":acts[i+j].round(6).tolist(),"meal_g":0} for j in range(12)]
        inp={"id":identifier,"group":identifier,"patient":"vital_subject_"+subject,"split":split,
             "domain":"vitaldb","kind":kind,"branch":"factual","scenario":i,
             "payload":{"variables":["BIS","MAP"],"units":["unitless","mmHg"],
                        "action_names":["propofol_mg_s","remifentanil_ug_s"],"step_min":1,
                        "history":history,"actions":actions,
                        "demographics":{"age_years":float(case.age),"weight_kg":float(case.weight),
                                        "height_cm":float(case.height),"sex":str(case.sex)},
                        "action_information":"revealed_at_each_step_only",
                        "observation_definition":"last recorded sample at or before each minute; max age 15 seconds"}}
        lab={"id":identifier,"group":identifier,"patient":inp["patient"],
             "targets":obs[i+1:i+13].tolist(),"meta":{"caseid":caseid,"subjectid":subject,
              "anchor_track_time_sec":float(grid[i]),"source_tracks":{t:str(chosen.loc[t,"tid"]) for t in TRACKS}}}
        rows.append((inp,lab))
    return rows,{"caseid":caseid,"subjectid":subject,"status":"included","split":split,"windows":len(rows)}

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    cases=pd.read_csv(io.BytesIO(csv_bytes(RAW/"cases.csv")),encoding="utf-8-sig")
    trks=pd.read_csv(io.BytesIO(csv_bytes(RAW/"trks.csv")))
    ids=set.intersection(*[set(trks[trks.tname==x].caseid) for x in TRACKS])
    cases=cases[cases.caseid.isin(ids)&(cases.age>=18)].dropna(subset=["subjectid","age","weight","height"])
    cases=cases.sort_values("caseid").drop_duplicates("subjectid")
    cases["order"]=cases.subjectid.map(lambda x:digest(str(int(x))+":sample20261102"))
    cases=cases.sort_values("order")
    inputs=[]; labels=[]; audit=[]
    included=0
    for _,case in cases.iterrows():
        if included>=120:break
        try:
            rows,record=case_rows(case,trks)
            if rows:
                included+=1; inputs.extend(x[0] for x in rows); labels.extend(x[1] for x in rows)
            audit.append(record)
        except Exception:
            audit.append({"caseid":int(case.caseid),"status":"error","traceback":traceback.format_exc()})
        print("included",included,"attempts",len(audit),"windows",len(inputs),flush=True)
    for split in ["development","test"]:
        ids={x["id"] for x in inputs if x["split"]==split}
        for name,rows in [("inputs",inputs),("oracles",labels)]:
            (DEST/(split+"."+name+".jsonl")).write_text("".join(json.dumps(x,allow_nan=False)+"\n" for x in rows if x["id"] in ids))
    report={"timestamp_utc":datetime.now(timezone.utc).isoformat(),"eligible_subjects":len(cases),
            "included_subjects":included,"windows":len(inputs),"audit":audit,
            "tracks":TRACKS,"sampling":"seeded subject hash; at most one case per actual subject",
            "no_counterfactual_truth":True,"external_api_inference":False,
            "future_action_policy":"current recorded pump rate revealed only at each replay step; not known at anchor",
            "license":"Dataset page CC BY-NC-SA 4.0 + DUA; API metadata CC BY 4.0 differs. Apply stricter dataset conditions; local noncommercial analysis, no raw redistribution.",
            "limitations":["selected TIVA subset","unrecorded interventions and confounding",
                           "legacy CSV relative time origin; no surgical timestamps used",
                           "quality filter conditions on future measurement availability, not deployment performance"],
            "files_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in DEST.glob("*.jsonl")}}
    (ROOT/"data/results/vitaldb_audit.json").write_text(json.dumps(report,indent=2)+"\n")
if __name__=="__main__":
    main()
