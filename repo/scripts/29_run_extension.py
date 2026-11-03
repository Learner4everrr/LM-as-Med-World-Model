#!/usr/bin/env python3
"""Execute the separately frozen post-primary interface/retrieval extension."""
from pathlib import Path
from copy import deepcopy
import os,sys,json,hashlib,argparse,importlib.util
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"repo/src"))
from medwm_eval.submission_extension import structured_prompt,parse_structured,memory_task,VERSION
from medwm_eval.prompts import feedback
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def loadl(p):return [json.loads(l) for l in p.read_text().splitlines()]
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--model",required=True);args=ap.parse_args()
    frozen=ROOT/"data/manifests/interface_memory_freeze.json"
    f=json.loads(frozen.read_text())
    for p,h in f["frozen_hashes"].items():assert sha(ROOT/p)==h,p
    out=ROOT/"data/runs/interface_memory_v1/test"/args.model.replace("/","--");out.mkdir(parents=True,exist_ok=True)
    if (out/"manifest.json").exists():raise RuntimeError("Already complete; do not overwrite archived runs")
    cases=[];truth={}
    for folder in ["data/simulated/benchmark_v1","data/processed/vitaldb_v1"]:
        cases+=loadl(ROOT/folder/"test.inputs.jsonl")
        truth.update({r["id"]:r["targets"] for r in loadl(ROOT/folder/"test.oracles.jsonl")})
    spec=importlib.util.spec_from_file_location("frozen_engine",ROOT/"repo/scripts/05_run_local.py")
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.PROMPT_VERSION=VERSION
    engine=mod.Engine(args.model,out,16)
    histories={c["id"]:deepcopy(c["payload"]["history"]) for c in cases}
    active={c["id"]:True for c in cases};trajectories={c["id"]:[] for c in cases}
    for step in range(12):
        batch=[c for c in cases if active[c["id"]]]
        responses=engine.generate([structured_prompt(c,histories[c["id"]],step) for c in batch])
        for c,r in zip(batch,responses):
            cid=c["id"];p=c["payload"];parsed=parse_structured(r["raw"],p["variables"])
            parsed.update(step=step+1,prompt_hash=r["prompt_hash"]);trajectories[cid].append(parsed)
            if not parsed["valid"]:active[cid]=False;continue
            histories[cid]=feedback(histories[cid],parsed["values"],p["actions"][step],p["step_min"])
        print(VERSION,args.model,"free step",step+1,"active",sum(active.values()),flush=True)
    outputs=[{k:c[k] for k in ["id","group","patient","domain","kind","branch","scenario","split"]}|
             dict(model_id=args.model,condition="state_action_named",mode="free",predictions=trajectories[c["id"]]) for c in cases]
    (out/"predictions.jsonl").write_text("".join(json.dumps(r,allow_nan=False)+"\n" for r in outputs))
    tasks=[(c,anchor,*memory_task(c,truth[c["id"]],anchor)) for anchor in [0,6] for c in cases]
    responses=engine.generate([t[2] for t in tasks])
    records=[dict(id=c["id"],domain=c["domain"],patient=c["patient"],anchor_step=a,
                  prompt_hash=r["prompt_hash"],target=target,raw=r["raw"]) for (c,a,prompt,target),r in zip(tasks,responses)]
    (out/"memory.json").write_text(json.dumps(records,indent=2))
    info=dict(model_id=args.model,extension=VERSION,freeze_sha256=sha(frozen),timestamp_utc=datetime.now(timezone.utc).isoformat(),
        cases=len(cases),retrieval_tasks=len(tasks),unique_raw_calls=len(engine.cache),batch_size=16,max_new_tokens=160,
        original_engine_unchanged=True,paid_calls=0,
        files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!="launch.log"})
    (out/"manifest.json").write_text(json.dumps(info,indent=2))
    print("COMPLETE",out,flush=True)
if __name__=="__main__":main()
