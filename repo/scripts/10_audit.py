#!/usr/bin/env python3
"""Independent structural, split and numerical presentation audit."""
from pathlib import Path
import sys,json,hashlib,importlib.util
from datetime import datetime,timezone
from collections import Counter,defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"repo/src"))
from medwm_eval.envs.anesthesia import verify_pair
def loadl(p):return [json.loads(l) for l in p.read_text().splitlines()]
def main():
    allcases=[];oracle={}
    for folder in ["data/simulated/benchmark_v1","data/processed/vitaldb_v1"]:
        for split in ["development","test"]:
            allcases+=loadl(ROOT/folder/(split+".inputs.jsonl"))
            oracle.update({x["id"]:x for x in loadl(ROOT/folder/(split+".oracles.jsonl"))})
    assert len({x["id"] for x in allcases})==len(allcases)==len(oracle)
    for domain in {x["domain"] for x in allcases}:
        assert not ({x["patient"] for x in allcases if x["domain"]==domain and x["split"]=="development"}&
                    {x["patient"] for x in allcases if x["domain"]==domain and x["split"]=="test"})
    groups=defaultdict(list)
    for c in allcases:
        p=c["payload"];dt=p["step_min"]
        assert p["history"][-1]["time"]==0
        assert all(h["time"]<=0 for h in p["history"])
        assert [a["time"] for a in p["actions"]]==[k*dt for k in range(12)]
        assert np.asarray(oracle[c["id"]]["targets"]).shape==(12,len(p["variables"]))
        assert np.isfinite(oracle[c["id"]]["targets"]).all()
        if c["domain"]!="vitaldb":groups[c["group"]].append(c)
    for rows in groups.values():
        assert {r["branch"] for r in rows}==({"A","R","B"} if rows[0]["kind"]=="dose" else {"A","B"})
        assert all(r["payload"]["history"]==rows[0]["payload"]["history"] for r in rows)
        assert len({oracle[r["id"]]["anchor_state_hash"] for r in rows})==1
        assert all(all(oracle[r["id"]]["branch_checks"].values()) for r in rows)
        assert len({tuple(a["meal_g"] for a in r["payload"]["actions"]) for r in rows})==1
    spec=importlib.util.spec_from_file_location("generator",ROOT/"repo/scripts/02_generate_benchmark.py")
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    quant=[]
    for domain in ["simglucose","anesthesia"]:
        selected=[c for c in allcases if c["domain"]==domain and c["split"]=="development"][:3]
        for c in selected:
            if domain=="simglucose":
                group,num=c["patient"].split("#");index=["adult","adolescent","child"].index(group)*100+int(num)
            else:index=int(c["patient"].split("_")[1])
            env,*_=mod.setup(domain,index,c["scenario"])
            values=mod.simulate(domain,env,c["payload"]["actions"])
            quant.append(dict(id=c["id"],domain=domain,max_abs_difference=np.max(np.abs(np.array(values)-oracle[c["id"]]["targets"]),axis=0).tolist()))
    report=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),cases=len(allcases),groups=len(groups),
        unique_ids=True,patient_split_disjoint=True,finite_targets=True,identical_visible_and_hidden_anchors=True,
        exact_replay_checks_all_true=True,matched_exogenous_meals=True,
        pas_independent_pair_check=verify_pair(),display_quantization_development_check=quant,
        counts={split:dict(Counter(c["domain"] for c in allcases if c["split"]==split)) for split in ["development","test"]},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitation="Quantization sensitivity checked on one development dose group per simulator; not a full-domain bound.")
    (ROOT/"data/results/final_data_audit.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
