#!/usr/bin/env python3
"""Run one frozen model per GPU and wait for all completion statuses."""
from pathlib import Path
import subprocess,os,json
ROOT=Path(__file__).resolve().parents[2]
MODELS=["Qwen/Qwen3-8B","Qwen/Qwen2.5-7B-Instruct","mistralai/Mistral-7B-Instruct-v0.3","microsoft/Phi-3.5-mini-instruct"]
def main():
    jobs=[];handles=[]
    for gpu,model in enumerate(MODELS):
        out=ROOT/"data/runs/study_v1/test"/model.replace("/","--")
        if (out/"manifest.json").exists():
            print("Already completed:",model,flush=True);continue
        out.mkdir(parents=True,exist_ok=True)
        handle=(out/"launch.log").open("a");handles.append(handle)
        en=os.environ.copy();en.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4")
        proc=subprocess.Popen([str(ROOT/"env/inference/bin/python"),str(ROOT/"repo/scripts/05_run_local.py"),
                               "--model",model,"--batch-size","16"],env=en,stdout=handle,stderr=subprocess.STDOUT)
        jobs.append((model,proc))
    results={model:proc.wait() for model,proc in jobs}
    for h in handles:h.close()
    print(json.dumps(results),flush=True)
    if any(code!=0 for code in results.values()):raise SystemExit(1)
if __name__=="__main__":main()
