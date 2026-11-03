#!/usr/bin/env python3
"""Finalize the pre-frozen extension once every model has saved its manifest."""
from pathlib import Path
import time,subprocess,os
ROOT=Path(__file__).resolve().parents[2]
MODELS=["Qwen--Qwen3-8B","Qwen--Qwen2.5-7B-Instruct","mistralai--Mistral-7B-Instruct-v0.3","microsoft--Phi-3.5-mini-instruct"]
print("Waiting for four final manifests",flush=True)
while not all((ROOT/"data/runs/interface_memory_v1/test"/m/"manifest.json").exists() for m in MODELS):
    time.sleep(20)
time.sleep(3)
env=os.environ.copy()
for name in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]:env[name]="4"
for name in ["30_verify_extension.py","31_analyze_extension.py"]:
    print("RUN",name,flush=True)
    subprocess.run([str(ROOT/"env/.venv/bin/python"),str(ROOT/"repo/scripts"/name)],cwd=ROOT,env=env,check=True)
print("FINALIZED",flush=True)
