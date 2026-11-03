#!/usr/bin/env python3
"""Frozen targeted extension using the main engine, prompts and failure rules."""
from pathlib import Path
from copy import deepcopy
import sys,json,hashlib,argparse,importlib.util
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"repo/src"))
from medwm_eval.prompts import make_prompt,parse_prediction,feedback
spec=importlib.util.spec_from_file_location("main_engine",ROOT/"repo/scripts/05_run_local.py")
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
def loadl(p):return [json.loads(l) for l in p.read_text().splitlines()]
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--model",required=True);args=ap.parse_args()
    for fname in ["study_freeze.json","stress_freeze.json"]:
        freeze=json.loads((ROOT/"data/manifests"/fname).read_text())
        for path,h in freeze["frozen_hashes"].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
    folder=ROOT/"data/simulated/hypoglycemia_stress_v1"
    cases=loadl(folder/"test.inputs.jsonl");truth={x["id"]:x["targets"] for x in loadl(folder/"test.oracles.jsonl")}
    if not cases:raise RuntimeError("No eligible stress cases")
    out=ROOT/"data/runs/stress_v1/test"/args.model.replace("/","--");out.mkdir(parents=True,exist_ok=True)
    engine=mod.Engine(args.model,out,16);rows=[]
    for condition,mode in [("state_action","free"),("state_action","teacher"),("state_only","free")]:
        histories={c["id"]:deepcopy(c["payload"]["history"]) for c in cases}
        active={c["id"]:True for c in cases};pred={c["id"]:[] for c in cases}
        for step in range(12):
            selected=[c for c in cases if active[c["id"]]]
            responses=engine.generate([make_prompt(c,histories[c["id"]],step,condition) for c in selected])
            for c,r in zip(selected,responses):
                cid=c["id"];p=c["payload"]
                parsed=parse_prediction(r["raw"],1);parsed.update(step=step+1,prompt_hash=r["prompt_hash"]);pred[cid].append(parsed)
                if mode=="free" and not parsed["valid"]:active[cid]=False;continue
                values=parsed["values"] if mode=="free" else truth[cid][step]
                histories[cid]=feedback(histories[cid],values,p["actions"][step],30)
            print(condition,mode,step+1,sum(active.values()),flush=True)
        for c in cases:
            rows.append({k:c[k] for k in ["id","group","patient","domain","kind","branch","scenario","split"]}|
                        dict(model_id=args.model,condition=condition,mode=mode,predictions=pred[c["id"]]))
        (out/"predictions.jsonl").write_text("".join(json.dumps(r,allow_nan=False)+"\n" for r in rows))
    info=dict(model_id=args.model,timestamp_utc=datetime.now(timezone.utc).isoformat(),phase="targeted_extension",
              cases=len(cases),unique_prompts=len(engine.cache),batch_size=16,paid_calls=0,
              model_manifest_sha256=hashlib.sha256((ROOT/"data/manifests"/(args.model.replace("/","--")+".json")).read_bytes()).hexdigest(),
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              file_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()})
    (out/"manifest.json").write_text(json.dumps(info,indent=2));print("COMPLETE",out,flush=True)
if __name__=="__main__":main()
