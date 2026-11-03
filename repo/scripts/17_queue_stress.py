#!/usr/bin/env python3
"""Wait for each main-model process to finish before using its allocated GPU."""
from pathlib import Path
import json,subprocess,os,time
ROOT=Path(__file__).resolve().parents[2]
MODELS=["Qwen/Qwen3-8B","Qwen/Qwen2.5-7B-Instruct","mistralai/Mistral-7B-Instruct-v0.3","microsoft/Phi-3.5-mini-instruct"]
def main():
    pending=dict(enumerate(MODELS));running=[];handles=[]
    while pending:
        for gpu,model in list(pending.items()):
            mainout=ROOT/"data/runs/study_v1/test"/model.replace("/","--")
            if not (mainout/"manifest.json").exists():continue
            out=ROOT/"data/runs/stress_v1/test"/model.replace("/","--");out.mkdir(parents=True,exist_ok=True)
            if (out/"manifest.json").exists():del pending[gpu];continue
            time.sleep(5)
            en=os.environ.copy();en.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4")
            h=(out/"launch.log").open("a");handles.append(h)
            p=subprocess.Popen([str(ROOT/"env/inference/bin/python"),str(ROOT/"repo/scripts/15_run_stress.py"),"--model",model],stdout=h,stderr=subprocess.STDOUT,env=en)
            running.append((model,p));del pending[gpu];print("STARTED",model,p.pid,flush=True)
        if pending:time.sleep(10)
    codes={m:p.wait() for m,p in running}
    for h in handles:h.close()
    print(codes,flush=True)
    if any(codes.values()):raise SystemExit(1)
if __name__=="__main__":main()
