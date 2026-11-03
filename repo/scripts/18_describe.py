#!/usr/bin/env python3
"""Reproduce cohort summaries and manuscript asset registry from saved inputs."""
from pathlib import Path
from collections import Counter
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def load(p):return [json.loads(x) for x in p.read_text().splitlines()]
def main():
    result={}
    for split in ["development","test"]:
        cs=[];ys={}
        for folder in ["data/simulated/benchmark_v1","data/processed/vitaldb_v1"]:
            cs+=load(ROOT/folder/(split+".inputs.jsonl"))
            ys.update({r["id"]:r["targets"] for r in load(ROOT/folder/(split+".oracles.jsonl"))})
        for domain in ["simglucose","anesthesia","vitaldb"]:
            rows=[r for r in cs if r["domain"]==domain]
            subjects={r["patient"]:r for r in rows}
            y=np.concatenate([ys[r["id"]] for r in rows])
            r=dict(subjects=len(subjects),windows=len(rows),variables=rows[0]["payload"]["variables"],
                   minimum=y.min(0).tolist(),maximum=y.max(0).tolist(),kind_counts=dict(Counter(x["kind"] for x in rows)))
            if domain!="simglucose":
                demos=[x["payload"]["demographics"] for x in subjects.values()]
                r["age_median_iqr"]=np.quantile([d["age_years"] for d in demos],[.5,.25,.75]).tolist()
                r["sex_counts"]=dict(Counter(d["sex"] for d in demos))
            result[split+"_"+domain]=r
    out=ROOT/"data/results/cohort_description.json"
    out.write_text(json.dumps(result,indent=2)+"\n")
    pub=ROOT/"paper/tables";pub.mkdir(exist_ok=True)
    models=json.loads((ROOT/"repo/configs/models_frozen.json").read_text())
    lines=[r"\begin{tabular}{lll}",r"\toprule",r"Model & Revision prefix & License \\",r"\midrule"]
    for m in ["Qwen/Qwen3-8B","Qwen/Qwen2.5-7B-Instruct","mistralai/Mistral-7B-Instruct-v0.3","microsoft/Phi-3.5-mini-instruct"]:
        v=models[m]
        lines.append(m.split("/")[1]+r" & \texttt{"+v["revision"][:12]+"} & "+v["license"]+r" \\")
    lines += [r"\bottomrule",r"\end{tabular}"]
    target=ROOT/"data/results/model_table.tex";target.write_text("\n".join(lines)+"\n")
    (pub/"models.tex").write_bytes(target.read_bytes())
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
