from copy import deepcopy
import json
from medwm_eval.submission_extension import visible_view,structured_prompt,parse_structured,memory_task,score_memory
from medwm_eval.prompts import make_prompt

def case():
    return dict(payload=dict(variables=["BIS","MAP"],units=["index","mmHg"],action_names=["propofol_mg_s","remifentanil_ug_s"],
        step_min=2,history=[dict(time=t,values=[50.+i,70.+i],past_action=[.1+i*.01,.2]) for i,t in enumerate([-6,-4,-2,0])],
        actions=[dict(time=2*k,values=[.2+k*.01,.3],meal_g=0) for k in range(12)],demographics={},observation_definition="test",action_information="predefined"))
def test_same_input_no_extra_oracle():
    c=case();original=make_prompt(c,c["payload"]["history"],0)
    view=json.JSONDecoder().raw_decode(original[original.index('{"variables"'):])[0]
    new=structured_prompt(c,c["payload"]["history"],0)
    parsed=json.JSONDecoder().raw_decode(new.split("\nINPUT\n",1)[1])[0]
    assert view==parsed==visible_view(c,c["payload"]["history"],0)
def test_named_variable_order_and_interval_failure():
    raw='{"next_state":{"MAP":{"value":70,"lower":60,"upper":80},"BIS":{"value":50,"lower":51,"upper":60}}}'
    r=parse_structured(raw,["BIS","MAP"])
    assert r["valid"] and r["values"]==[50,70] and not r["interval_valid"]
    assert not parse_structured('{"next_state":{"BIS":{"value":true}}}',["BIS"])["valid"]
    assert not parse_structured(raw,["BIS","MAP","CGM"])["valid"]
    assert not parse_structured(raw+" explanation",["BIS","MAP"])["valid"]
def test_retrieval_only_past_and_semantic_temporal_labels():
    c=case();truth=[[40+k,60+k] for k in range(12)]
    prompt,target=memory_task(c,truth,6)
    view=json.loads(prompt.split("\n",1)[1])
    assert view["history"][-1]["values"]==truth[5]
    assert target["current_time"]==12 and target["last_action_interval_start"]==10
    assert target["query_time"]==6 and target["query_action"]==c["payload"]["actions"][2]["values"]
    c2=deepcopy(c);c2["payload"]["actions"][6]["values"]=[999,999]
    truth2=deepcopy(truth);truth2[6]=[9999,9999]
    assert memory_task(c2,truth2,6)==(prompt,target)
def test_retrieval_strict_and_content_denominators():
    _,t=memory_task(case(),[[40,60]]*12,6)
    raw=json.dumps(t)
    assert score_memory(raw,t)["all_fields"]==1
    assert score_memory(raw+" prose",t)["all_fields"]==0
    assert score_memory(raw+" prose",t,True)["all_fields"]==1
    broken=deepcopy(t);broken["last_action"][0]+=1
    assert score_memory(json.dumps(broken),t)["current_action"]==0
    assert score_memory(json.dumps(broken),t)["state"]==1
