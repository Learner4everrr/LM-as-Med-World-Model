#!/usr/bin/env python3
"""Run verification and reporting only after every model manifest exists."""
from pathlib import Path
import subprocess,time,sys,os
ROOT=Path(__file__).resolve().parents[2]
MODELS=["Qwen--Qwen3-8B","Qwen--Qwen2.5-7B-Instruct","mistralai--Mistral-7B-Instruct-v0.3","microsoft--Phi-3.5-mini-instruct"]
def call(script,*args):
    print("RUN",script,*args,flush=True)
    en=os.environ.copy();en.update(OPENBLAS_NUM_THREADS="4",OMP_NUM_THREADS="4")
    subprocess.run([str(ROOT/"env/.venv/bin/python"),str(ROOT/"repo/scripts"/script),*args],cwd=ROOT,env=en,check=True)
def main():
    for study,extra in [("study_v1",[]),("stress_v1",["--stress"])]:
        while not all((ROOT/"data/runs"/study/"test"/m/"manifest.json").exists() for m in MODELS):time.sleep(20)
        time.sleep(5)
        call("12_verify_study.py",*extra)
        call("06_evaluate.py",*extra)
        if not extra:
            call("07_figures.py")
            call("11_tables.py")
        print("COMPLETE",study,flush=True)
if __name__=="__main__":main()
