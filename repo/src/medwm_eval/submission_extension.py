"""Additional named-output interface and temporal treatment retrieval, no human labels."""
from copy import deepcopy
import json,math
from .prompts import make_prompt,feedback

VERSION="interface_memory_v1"
def visible_view(case,history,step):
    text=make_prompt(case,history,step,"state_action")
    return json.JSONDecoder().raw_decode(text[text.index('{"variables"'):])[0]

def structured_prompt(case,history,step):
    view=visible_view(case,history,step)
    skeleton={"next_state":{v:{"value":"number","lower":"number","upper":"number"} for v in case["payload"]["variables"]}}
    return (
        "You are being evaluated as a patient-dynamics simulator, not as a clinician choosing treatment. "
        "Predict the next observation from the visible history. Use the supplied actions. "
        "The actions are fixed externally. History past-action columns describe the preceding interval; "
        "scheduled actions start at the stated time and last one action interval. "
        "Meal entries are grams announced at interval start. Do not recommend treatments. "
        "Return only one JSON object. Use a next_state object keyed by EVERY listed variable. "
        "For each variable, value is its point prediction, and lower and upper are your 95% prediction "
        "interval bounds for the next observation, including uncertainty. Use the listed units, not normalized values. "
        "Do not output code, explanations, or extra time points.\n"
        +"Response structure (replace each number placeholder by a numeric value): "
        +json.dumps(skeleton,separators=(",",":"))+"\nINPUT\n"
        +json.dumps(view,separators=(",",":"),allow_nan=False)
        +'\nNow output only the answer JSON. Begin with {"next_state": and do not repeat the input data.'
    )
def object_parse(raw,prefix=False):
    text=raw.strip()
    if prefix:
        start=text.find("{")
        if start<0:return None
        try:return json.JSONDecoder().raw_decode(text[start:])[0]
        except (ValueError,TypeError):return None
    if text.startswith(chr(96)*3):
        text=text.split("\n",1)[-1]
        if text.rstrip().endswith(chr(96)*3):text=text.rstrip()[:-3].strip()
    try:return json.loads(text)
    except (ValueError,TypeError):return None
def numeric(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def parse_structured(raw,variables):
    obj=object_parse(raw)
    try:
        nodes=[obj["next_state"][v] for v in variables]
        vals=[x["value"] for x in nodes]
        if not all(numeric(v) for v in vals):raise ValueError()
    except (KeyError,TypeError,ValueError):
        return dict(valid=False,reason="invalid_json_or_values",values=None,interval_valid=False)
    result=dict(valid=True,values=[float(x) for x in vals],interval_valid=False,scalar_normalized=False)
    try:
        lo=[x["lower"] for x in nodes];hi=[x["upper"] for x in nodes]
        if not all(numeric(l) and numeric(u) and l<=v<=u for l,v,u in zip(lo,vals,hi)):raise ValueError()
        result.update(interval_valid=True,lower=lo,upper=hi)
    except (KeyError,TypeError,ValueError):pass
    return result
def probe_history(case,targets,anchor):
    h=deepcopy(case["payload"]["history"])
    for k in range(anchor):
        h=feedback(h,targets[k],case["payload"]["actions"][k],case["payload"]["step_min"])
    return h
def memory_task(case,targets,anchor):
    p=case["payload"];h=probe_history(case,targets,anchor)
    rows=[dict(time_min=r["time"],values=[round(float(x),4) for x in r["values"]],
               past_action=list(r["past_action"])) for r in h]
    query=rows[-4]
    view=dict(variables=p["variables"],units=p["units"],action_names=p["action_names"],
              history=rows,query_time_min=query["time_min"])
    target=dict(current_time=rows[-1]["time_min"],current_values=rows[-1]["values"],
                last_action=rows[-1]["past_action"],query_time=query["time_min"],
                query_action=query["past_action"],last_action_interval_start=rows[-2]["time_min"])
    prompt=(
        "Read the visible observation and treatment history. This is a retrieval check, not a forecast or treatment recommendation. "
        "Each past_action vector describes the interval ending at that row's time_min. "
        "Report current_time and current_values from the final row; last_action from the final row; "
        "query_time and query_action from the row whose time_min equals query_time_min; "
        "and last_action_interval_start as the preceding row's time_min. Preserve the listed numeric units and vector order. "
        "Return only one JSON object with keys current_time, current_values, last_action, query_time, query_action, "
        "last_action_interval_start. Numeric vectors must remain arrays. Do not forecast, explain, or repeat the history.\n"
        +json.dumps(view,separators=(",",":"),allow_nan=False))
    return prompt,target
def score_memory(raw,target,prefix=False):
    obj=object_parse(raw,prefix)
    valid=isinstance(obj,dict)
    def same(key,tol):
        if not valid:return False
        x=obj.get(key);y=target[key]
        if isinstance(y,list):
            return isinstance(x,list) and len(x)==len(y) and all(numeric(a) and abs(a-b)<=tol for a,b in zip(x,y))
        return numeric(x) and abs(x-y)<=tol
    state=same("current_values",1e-4)
    temporal=all(same(k,1e-8) for k in ["current_time","query_time","last_action_interval_start"])
    current=same("last_action",1e-6);past=same("query_action",1e-6)
    return dict(json_object=float(valid),state=float(state),temporal=float(temporal),
                current_action=float(current),past_action=float(past),all_fields=float(state and temporal and current and past))
