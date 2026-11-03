#!/usr/bin/env python3
"""Frozen local model inference. Raw outputs and model-visible prompts are retained."""
from pathlib import Path
from copy import deepcopy
import os,sys,json,time,argparse,hashlib,platform
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"repo/src"))
os.environ["HF_HOME"]=str(ROOT/"data/cache/huggingface")
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ.setdefault("TORCH_HOME",str(ROOT/"data/cache/torch"))
os.environ.setdefault("TRITON_CACHE_DIR",str(ROOT/"data/cache/triton"))
from medwm_eval.prompts import make_prompt,parse_prediction,feedback,stable_hash,auxiliary_prompt,PROMPT_VERSION

def loadjsonl(p):
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

class Engine:
    def __init__(self,model_id,out,batch_size):
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        torch.manual_seed(20261102)
        self.torch=torch;self.model_id=model_id;self.out=out;self.batch_size=batch_size
        path=ROOT/"data/models"/model_id.replace("/","--")
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
        self.tokenizer.padding_side="left"
        if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token=self.tokenizer.eos_token
        self.model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.bfloat16,
                      device_map={"":0},attn_implementation="sdpa",local_files_only=True,trust_remote_code=False)
        self.model.eval()
        self.cache={}
        raw_path=out/"raw.jsonl"
        if raw_path.exists():
            for x in loadjsonl(raw_path):self.cache[x["prompt_hash"]]=x
        self.raw=raw_path.open("a",buffering=1)
    def generate(self,prompts):
        keys=[hashlib.sha256(p.encode()).hexdigest() for p in prompts]
        missing={}
        for key,prompt in zip(keys,prompts):
            if key not in self.cache:missing[key]=prompt
        items=list(missing.items())
        for start in range(0,len(items),self.batch_size):
            batch=items[start:start+self.batch_size]
            texts=[self.tokenizer.apply_chat_template([{"role":"user","content":p}],
                    tokenize=False,add_generation_prompt=True,enable_thinking=False) for _,p in batch]
            tok=self.tokenizer(texts,return_tensors="pt",padding=True)
            if tok["input_ids"].shape[1]>8192:
                raise RuntimeError("Prompt exceeds frozen 8192-token limit; no silent truncation")
            tok={k:v.to(self.model.device) for k,v in tok.items()}
            now=time.time()
            with self.torch.inference_mode():
                generated=self.model.generate(**tok,do_sample=False,max_new_tokens=160,
                          pad_token_id=self.tokenizer.pad_token_id,use_cache=True)
            elapsed=time.time()-now
            tails=generated[:,tok["input_ids"].shape[1]:]
            outputs=self.tokenizer.batch_decode(tails,skip_special_tokens=True)
            for j,((key,prompt),raw) in enumerate(zip(batch,outputs)):
                record={"prompt_hash":key,"prompt":prompt,"raw":raw,"model_id":self.model_id,
                        "prompt_version":PROMPT_VERSION,"input_tokens":int(tok["attention_mask"][j].sum()),
                        "generated_token_slots":int(tails.shape[1]),"batch_seconds":elapsed,
                        "batch_size":len(batch),"timestamp_utc":datetime.now(timezone.utc).isoformat()}
                self.raw.write(json.dumps(record,allow_nan=False)+"\n");self.cache[key]=record
            print("generated",len(self.cache),"unique prompts",flush=True)
        return [self.cache[k] for k in keys]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True)
    p.add_argument("--pilot",action="store_true")
    p.add_argument("--pilot-tag",default="pilot_v3")
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--split",choices=["development","test"],default="test")
    p.add_argument("--domains",default="simglucose,anesthesia,vitaldb")
    args=p.parse_args()
    phase=args.pilot_tag if args.pilot else args.split
    out=ROOT/"data/runs/study_v1"/phase/args.model.replace("/","--")
    out.mkdir(parents=True,exist_ok=True)
    split="development" if args.pilot else args.split
    cases=[];truth={}
    roots=[ROOT/"data/simulated/benchmark_v1",ROOT/"data/processed/vitaldb_v1"]
    for root in roots:
        ip=root/(split+".inputs.jsonl");op=root/(split+".oracles.jsonl")
        if not ip.exists():continue
        selected=[x for x in loadjsonl(ip) if x["domain"] in args.domains.split(",")]
        if args.pilot:
            chosen=[]
            for domain in args.domains.split(","):
                # First complete dose group (three branches) or first factual case.
                domaincases=[x for x in selected if x["domain"]==domain]
                if domaincases:
                    group=domaincases[0]["group"]
                    chosen.extend(x for x in domaincases if x["group"]==group)
            selected=chosen
        cases+=selected
        ids={x["id"] for x in selected}
        truth.update({x["id"]:x["targets"] for x in loadjsonl(op) if x["id"] in ids})
    if not cases:raise RuntimeError("No cases")
    # Fail closed if a test split lacks pre-inference protocol freeze.
    frozen=ROOT/"data/manifests/study_freeze.json"
    if not args.pilot:
        if not frozen.exists():raise RuntimeError("Protocol freeze missing")
        record=json.loads(frozen.read_text())
        for path,expected in record["frozen_hashes"].items():
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=expected:
                raise RuntimeError("Frozen artifact changed: "+path)
    engine=Engine(args.model,out,args.batch_size)
    outputs=[]
    for condition,mode in [("state_action","free"),("state_action","teacher"),("state_only","free")]:
        histories={x["id"]:deepcopy(x["payload"]["history"]) for x in cases}
        active={x["id"]:True for x in cases}
        trajectories={x["id"]:[] for x in cases}
        for step in range(12):
            batch=[x for x in cases if active[x["id"]]]
            prompts=[make_prompt(x,histories[x["id"]],step,condition) for x in batch]
            responses=engine.generate(prompts)
            for case,response in zip(batch,responses):
                cid=case["id"];payload=case["payload"]
                result=parse_prediction(response["raw"],len(payload["variables"]))
                result.update(step=step+1,prompt_hash=response["prompt_hash"])
                trajectories[cid].append(result)
                if mode=="free":
                    if not result["valid"]:active[cid]=False;continue
                    values=result["values"]
                else:
                    values=truth[cid][step]
                histories[cid]=feedback(histories[cid],values,payload["actions"][step],payload["step_min"])
            print(args.model,condition,mode,"step",step+1,"active",sum(active.values()),flush=True)
        for case in cases:
            outputs.append({k:case[k] for k in ["id","group","patient","domain","kind","branch","scenario","split"]}|
                           {"model_id":args.model,"condition":condition,"mode":mode,"predictions":trajectories[case["id"]]})
        (out/"predictions.jsonl").write_text("".join(json.dumps(x,allow_nan=False)+"\n" for x in outputs))
    unique={}
    for case in cases:
        key=(case["domain"],case["patient"],case["scenario"])
        unique.setdefault(key,case)
    auxcases=list(unique.values())
    responses=engine.generate([auxiliary_prompt(x) for x in auxcases])
    aux=[{"id":x["id"],"patient":x["patient"],"domain":x["domain"],"scenario":x["scenario"],
          "raw":r["raw"],"prompt_hash":r["prompt_hash"],"expected_values":x["payload"]["history"][-1]["values"],
          "expected_direction":["decrease"]*len(x["payload"]["variables"])} for x,r in zip(auxcases,responses)]
    (out/"auxiliary.json").write_text(json.dumps(aux,indent=2)+"\n")
    from medwm_eval.diagnostics import run_diagnostics
    run_diagnostics(engine,cases,out)
    model_manifest=json.loads((ROOT/"data/manifests"/(args.model.replace("/","--")+".json")).read_text())
    import torch,transformers
    info={"model":model_manifest,"timestamp_utc":datetime.now(timezone.utc).isoformat(),
          "phase":phase,"cases":len(cases),"unique_prompt_count":len(engine.cache),
          "torch":torch.__version__,"transformers":transformers.__version__,"python":platform.python_version(),
          "device":torch.cuda.get_device_name(),"batch_size":args.batch_size,
          "do_sample":False,"max_new_tokens":160,"retries":0,"enable_thinking":False,"paid_calls":0,
          "config_hash":hashlib.sha256((ROOT/"repo/configs/study_v1.json").read_bytes()).hexdigest(),
          "prompt_source_hash":hashlib.sha256((ROOT/"repo/src/medwm_eval/prompts.py").read_bytes()).hexdigest(),
          "file_hashes":{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in out.glob("*") if x.is_file()}}
    (out/"manifest.json").write_text(json.dumps(info,indent=2)+"\n")
    print("COMPLETE",out,flush=True)
if __name__=="__main__":main()
