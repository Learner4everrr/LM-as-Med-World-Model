#!/usr/bin/env python3
"""Verify finished inference against frozen inputs and raw prompt/response records."""
from pathlib import Path
import sys,json,hashlib,argparse
from copy import deepcopy
from collections import Counter
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"repo/src"))
from medwm_eval.prompts import make_prompt,parse_prediction,feedback,auxiliary_prompt
from medwm_eval.diagnostics import diagnostic_prompt,oneshot_prompt,parse_oneshot
MODELS=["Qwen/Qwen3-8B","Qwen/Qwen2.5-7B-Instruct","mistralai/Mistral-7B-Instruct-v0.3","microsoft/Phi-3.5-mini-instruct"]
def loadl(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--stress",action="store_true");args=ap.parse_args()
    runname="stress_v1" if args.stress else "study_v1"
    freeze=json.loads((ROOT/"data/manifests/study_freeze.json").read_text())
    for path,value in freeze["frozen_hashes"].items():assert sha(ROOT/path)==value,path
    if args.stress:
        ext=json.loads((ROOT/"data/manifests/stress_freeze.json").read_text())
        for path,value in ext["frozen_hashes"].items():assert sha(ROOT/path)==value,path
    cases={};truth={}
    for folder in (["data/simulated/hypoglycemia_stress_v1"] if args.stress else ["data/simulated/benchmark_v1","data/processed/vitaldb_v1"]):
        cases.update({x["id"]:x for x in loadl(ROOT/folder/"test.inputs.jsonl")})
        truth.update({x["id"]:x["targets"] for x in loadl(ROOT/folder/"test.oracles.jsonl")})
    baseline_root=ROOT/"data/runs"/runname/"baselines"
    baseline_manifest=json.loads((baseline_root/"manifest.json").read_text())
    if not args.stress:
        assert sha(baseline_root/"predictions.jsonl")==baseline_manifest["predictions_sha256"]
        for path,value in baseline_manifest["source_hashes"].items():assert sha(ROOT/path)==value
        assert all(not row["test_used_for_fit"] for row in baseline_manifest["fit_audit"])
    else:
        assert baseline_manifest["no_refitting"]
        assert sha(ROOT/"data/runs/study_v1/baselines/simglucose.pkl")==baseline_manifest["source_model_sha256"]
    baseline_runs=loadl(baseline_root/"predictions.jsonl")
    assert len(baseline_runs)==len(cases)*10
    assert len({(r["id"],r["model_id"],r["mode"]) for r in baseline_runs})==len(baseline_runs)
    assert {r["id"] for r in baseline_runs}==set(cases)
    reports=[]
    for model in MODELS:
        out=ROOT/"data/runs"/runname/"test"/model.replace("/","--")
        manifest=json.loads((out/"manifest.json").read_text())
        for name,value in manifest["file_hashes"].items():
            if name=="launch.log":continue # final COMPLETE line is appended after manifest creation
            assert sha(out/name)==value,(model,name)
        raw=loadl(out/"raw.jsonl");cache={r["prompt_hash"]:r for r in raw};assert len(raw)==len(cache)
        assert all(r["model_id"]==model for r in raw)
        assert all(r["timestamp_utc"]>freeze["timestamp_utc"] for r in raw)
        for row in raw:assert hashlib.sha256(row["prompt"].encode()).hexdigest()==row["prompt_hash"]
        runs=loadl(out/"predictions.jsonl")
        expected={(cid,condition,mode) for cid in cases for condition,mode in [("state_action","free"),("state_action","teacher"),("state_only","free")]}
        actual={(r["id"],r["condition"],r["mode"]) for r in runs}
        assert actual==expected and len(runs)==len(expected)
        early=ROOT/"data/results/free_rollout_early_audit.json"
        if not args.stress and early.exists():
            sealed={x["model"]:x["free_rows_sha256"] for x in json.loads(early.read_text())["models"]}
            free_rows=sorted([r for r in runs if r["condition"]=="state_action" and r["mode"]=="free"],key=lambda x:x["id"])
            assert hashlib.sha256(json.dumps(free_rows,sort_keys=True).encode()).hexdigest()==sealed[model]
        counts=Counter();bydomain=Counter()
        for run in runs:
            c=cases[run["id"]];p=c["payload"];h=deepcopy(p["history"])
            assert len(run["predictions"])<=12
            if run["mode"]=="teacher":assert len(run["predictions"])==12
            for step,r in enumerate(run["predictions"]):
                assert r["step"]==step+1
                record=cache[r["prompt_hash"]]
                assert make_prompt(c,h,step,run["condition"])==record["prompt"]
                parsed=parse_prediction(record["raw"],len(p["variables"]))
                assert all(r.get(k)==v for k,v in parsed.items())
                counts["attempted"]+=1;counts["valid"]+=int(r["valid"])
                counts["scalar_normalized"]+=int(r.get("scalar_normalized",False))
                counts["interval_valid"]+=int(r.get("interval_valid",False))
                bydomain[(run["domain"],"valid" if r["valid"] else "invalid")]+=1
                if run["mode"]=="free" and not r["valid"]:
                    assert step==len(run["predictions"])-1
                    break
                values=r["values"] if run["mode"]=="free" else truth[run["id"]][step]
                h=feedback(h,values,p["actions"][step],p["step_min"])
        lookup={(r["id"],r["condition"],r["mode"]):r for r in runs}
        branch_groups={}
        for c in cases.values():
            if c["domain"]!="vitaldb":branch_groups.setdefault(c["group"],[]).append(c["id"])
        for ids in branch_groups.values():
            anchor=lookup[(ids[0],"state_only","free")]["predictions"]
            assert all(lookup[(cid,"state_only","free")]["predictions"]==anchor for cid in ids)
        for cid in cases:
            assert lookup[(cid,"state_action","free")]["predictions"][0]==lookup[(cid,"state_action","teacher")]["predictions"][0]
        if not args.stress:
            aux=json.loads((out/"auxiliary.json").read_text())
            expected_aux={}
            for c in cases.values():expected_aux.setdefault((c["domain"],c["patient"],c["scenario"]),c["id"])
            assert {r["id"] for r in aux}==set(expected_aux.values()) and len(aux)==len(expected_aux)
            for r in aux:
                c=cases[r["id"]];record=cache[r["prompt_hash"]]
                assert auxiliary_prompt(c)==record["prompt"] and r["raw"]==record["raw"]
                assert r["expected_values"]==c["payload"]["history"][-1]["values"]
                assert r["expected_direction"]==["decrease"]*len(c["payload"]["variables"])
            diag=json.loads((out/"diagnostics.json").read_text())
            expected_diag={}
            for c in cases.values():expected_diag.setdefault((c["domain"],c["patient"]),c["id"])
            expected_keys={(cid,v) for cid in expected_diag.values() for v in
                           (["missing_previous","noisy_history","time_shift_60","pretty_json"]+
                            (["one_shot"] if cases[cid]["domain"]!="vitaldb" else []))}
            assert {(r["id"],r["variant"]) for r in diag}==expected_keys and len(diag)==len(expected_keys)
            for r in diag:
                c=cases[r["id"]];record=cache[r["prompt_hash"]];n=len(c["payload"]["variables"])
                prompt=oneshot_prompt(c) if r["variant"]=="one_shot" else diagnostic_prompt(c,r["variant"])
                assert prompt==record["prompt"]
                parsed=(parse_oneshot if r["variant"]=="one_shot" else parse_prediction)(record["raw"],n)
                assert parsed==r["result"]
        counts["scheduled"]=len(cases)*3*12
        reports.append(dict(model=model,trajectories=len(runs),counts=dict(counts),by_domain={str(k):v for k,v in bydomain.items()},
            state_only_branch_invariance=True,first_step_feedback_equivalence=True,prompt_reconstruction_exact=True,raw_response_parse_exact=True,freeze_hashes_match=True,
            unique_raw_calls=len(raw),input_tokens=sum(r["input_tokens"] for r in raw),
            generated_token_slots_upper_bound=sum(r["generated_token_slots"] for r in raw),
            generation_seconds=sum(r["batch_seconds"]/r["batch_size"] for r in raw),
            paid_calls=0,final_files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    result=dict(verifier_source_sha256=sha(Path(__file__)),study_freeze_sha256=sha(ROOT/"data/manifests/study_freeze.json"),baseline_predictions_sha256=sha(baseline_root/"predictions.jsonl"),baseline_trajectories=len(baseline_runs),timestamp_utc=datetime.now(timezone.utc).isoformat(),models=reports,
                log_hash_note="Run manifests hash the launch log before the final completion line; this final seal hashes the completed log.",
                no_future_oracle_in_free_prompts_verified=True)
    dest=ROOT/"data/results"/runname;dest.mkdir(parents=True,exist_ok=True)
    (dest/"integrity.json").write_text(json.dumps(result,indent=2))
    print(json.dumps([dict(model=x["model"],counts=x["counts"],unique_raw_calls=x["unique_raw_calls"]) for x in reports],indent=2))
if __name__=="__main__":main()
