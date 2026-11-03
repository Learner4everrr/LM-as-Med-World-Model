"""Strict model-visible serialization, parsing and prompt provenance."""
from copy import deepcopy
import json,math,hashlib
PROMPT_VERSION="transition_v3"

def stable_hash(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def make_prompt(case,history,step,condition="state_action"):
    p=case["payload"]; dt=p["step_min"]; n=len(p["variables"])
    if history[-1]["time"]!=step*dt or any(h["time"]>step*dt for h in history):
        raise ValueError("History time does not match current step")
    cols=["time_min"]+p["variables"]
    if condition=="state_action":
        cols+=["past_"+x for x in p["action_names"]]+["past_announced_meal_g"]
    rows=[]
    for h in history:
        values=h["values"]
        if len(values)!=n or not all(math.isfinite(float(x)) for x in values):
            raise ValueError("Invalid observation history")
        row=[h["time"]]+[round(float(x),4) for x in values]
        if condition=="state_action":
            row+=list(h["past_action"])+[h.get("past_meal_g",0)]
        rows.append(row)
    view={"variables":p["variables"],"units":p["units"],"time_unit":"minutes",
          "observation_definition":p["observation_definition"],
          "history_columns":cols,"history_rows":rows,
          "predict_time_min":(step+1)*dt,"demographics":p["demographics"]}
    if condition=="state_action":
        end=step+1 if p["action_information"]=="revealed_at_each_step_only" else len(p["actions"])
        view["action_columns"]=["start_min"]+p["action_names"]+["announced_meal_g_at_start"]
        view["scheduled_actions"]=[[a["time"]]+a["values"]+[a["meal_g"]] for a in p["actions"][step:end]]
        view["action_interval_minutes"]=dt
    instruction=(
        "You are being evaluated as a patient-dynamics simulator, not as a clinician choosing treatment. "
        "Predict the next observation from the visible history. Use the supplied actions when present. "
        "The actions are fixed externally. History past-action columns describe the preceding interval; "
        "scheduled actions start at the stated time and last one action interval. "
        "Meal entries are grams announced at interval start. Do not recommend treatments. "
        "Return only one JSON object with keys values, lower, upper. Each must be a numeric array "
        "in the listed variable order. values are point predictions; lower and upper are your 95% "
        "prediction interval bounds for the next observation, including uncertainty. "
        "Use the listed units, not normalized values. Do not output code, explanations, or extra time points.\n"
    )
    return instruction+f"Each of the three arrays must contain exactly {n} numbers, one for EVERY variable: "+", ".join(p["variables"])+". Never omit a variable.\n"+json.dumps(view,separators=(",",":"),allow_nan=False)+f'\nNow output only the answer JSON with exactly these keys: values, lower, upper. Each array has {n} numbers. Begin with {{"values": and do not repeat the input data.'

def parse_prediction(raw,n):
    text=raw.strip()
    if text.startswith("```"):
        text=text.split("\n",1)[-1]
        if text.rstrip().endswith("```"):text=text.rstrip()[:-3].strip()
    try:
        obj=json.loads(text)
        vals=obj["values"]
        scalar_normalized=n==1 and isinstance(vals,(int,float)) and not isinstance(vals,bool)
        if scalar_normalized:vals=[vals]
        if not isinstance(vals,list) or len(vals)!=n or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in vals):
            raise ValueError("invalid values")
    except (ValueError,KeyError,TypeError):
        return {"valid":False,"reason":"invalid_json_or_values","values":None,"interval_valid":False}
    result={"valid":True,"values":[float(x) for x in vals],"interval_valid":False,"scalar_normalized":scalar_normalized}
    try:
        lower,upper=obj["lower"],obj["upper"]
        if n==1 and isinstance(lower,(int,float)):lower=[lower]
        if n==1 and isinstance(upper,(int,float)):upper=[upper]
        if len(lower)!=n or len(upper)!=n:raise ValueError()
        if not all(isinstance(lo,(float,int)) and isinstance(hi,(float,int)) and not isinstance(lo,bool) and not isinstance(hi,bool) and math.isfinite(lo) and math.isfinite(hi) and lo<=v<=hi for lo,v,hi in zip(lower,vals,upper)):raise ValueError()
        result.update(lower=lower,upper=upper,interval_valid=True)
    except (KeyError,TypeError,ValueError):
        pass
    return result

def feedback(history,values,action,dt):
    h=deepcopy(history)
    h.append({"time":action["time"]+dt,"values":list(values),
              "past_action":list(action["values"]),"past_meal_g":action["meal_g"]})
    return h

def auxiliary_prompt(case):
    p=case["payload"]
    drug="basal insulin" if case["domain"]=="simglucose" else "propofol infusion"
    view={"variables":p["variables"],"units":p["units"],
          "history":[{"time_min":h["time"],"values":h["values"]} for h in p["history"]]}
    return ("Two checks. First copy the last observed time and values. Second, as a general "
            "pharmacology knowledge question, give the usual direction of change for each listed "
            "variable after increasing "+drug+" once its effect develops, other factors held fixed. "
            "This is not a numerical forecast or a treatment recommendation. "
            'Return only JSON: {"time_min": number, "values": [numbers], '
            '"direction": ["decrease" or "increase" or "unchanged", ...]}.\n'+json.dumps(view,separators=(",",":")))
