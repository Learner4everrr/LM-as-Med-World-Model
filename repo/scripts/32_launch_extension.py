#!/usr/bin/env python3
from pathlib import Path
import subprocess,os,json
ROOT=Path(__file__).resolve().parents[2]
MODELS=["Qwen/Qwen3-8B","Qwen/Qwen2.5-7B-Instruct","mistralai/Mistral-7B-Instruct-v0.3","microsoft/Phi-3.5-mini-instruct"]
def main():
    jobs=[];handles=[]
    for gpu,model in enumerate(MODELS):
        out=ROOT/"data/runs/interface_memory_v1/test"/model.replace("/","--")
        if (out/"manifest.json").exists():continue
        out.mkdir(parents=True,exist_ok=True);h=(out/"launch.log").open("a");handles.append(h)
        en=os.environ.copy();en.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4")
        p=subprocess.Popen([str(ROOT/"env/inference/bin/python"),str(ROOT/"repo/scripts/29_run_extension.py"),"--model",model],
                           cwd=ROOT,env=en,stdout=h,stderr=subprocess.STDOUT)
        jobs.append((model,p));print("STARTED",model,p.pid,flush=True)
    codes={m:p.wait() for m,p in jobs}
    for h in handles:h.close()
    print(json.dumps(codes),flush=True)
    if any(codes.values()):raise SystemExit(1)
if __name__=="__main__":main()
