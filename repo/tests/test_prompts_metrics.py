from copy import deepcopy
import pytest
import numpy as np
from medwm_eval.prompts import make_prompt,feedback,parse_prediction
from medwm_eval.metrics import paired_effect_mae,trajectory_mae,cluster_ci

def example(domain="anesthesia"):
    return {"domain":domain,"patient":"SECRET_PATIENT","payload":{
        "step_min":2,"variables":["BIS","MAP"],"units":["unitless","mmHg"],
        "action_names":["propofol_mg_s","remifentanil_ug_s"],
        "history":[{"time":0,"values":[50,80],"past_action":[.1,.2],"past_meal_g":0}],
        "actions":[{"time":0,"values":[.1,.2],"meal_g":0},{"time":2,"values":[.3,.2],"meal_g":0}],
        "demographics":{},"observation_definition":"observation",
        "action_information":"revealed_at_each_step_only" if domain=="vitaldb" else "predefined"}}

def test_future_oracle_and_identity_cannot_enter_prompt():
    x=example();h=x["payload"]["history"]
    baseline=make_prompt(x,h,0)
    x.update(targets="FUTURE_ORACLE",seed="SECRET_SEED")
    x["payload"].update(patient_state="HIDDEN_PK",oracle="FUTURE")
    assert make_prompt(x,h,0)==baseline
    assert "SECRET_PATIENT" not in baseline

def test_factual_replay_does_not_reveal_future_actions():
    x=example("vitaldb");h=x["payload"]["history"]
    baseline=make_prompt(x,h,0)
    x["payload"]["actions"][1]["values"]=[999,999]
    assert make_prompt(x,h,0)==baseline

def test_state_only_omits_actions():
    x=example();h=x["payload"]["history"]
    baseline=make_prompt(x,h,0,"state_only")
    x["payload"]["actions"][0]["values"]=[999,999]
    h[0]["past_action"]=[888,888]
    assert make_prompt(x,h,0,"state_only")==baseline

def test_feedback_uses_prediction_and_retains_action():
    x=example();h=x["payload"]["history"]
    next_h=feedback(h,[51,79],x["payload"]["actions"][0],2)
    assert next_h[-1]["values"]==[51,79]
    assert next_h[-1]["past_action"]==[.1,.2]
    assert h[-1]["values"]==[50,80]
    assert "51.0" in make_prompt(x,next_h,1)
    with pytest.raises(ValueError):make_prompt(x,next_h,0)

def test_parser_retains_violations_but_rejects_nonfinite_and_malformed():
    assert parse_prediction('{"values":[-10,300],"lower":[-20,200],"upper":[0,400]}',2)["valid"]
    for text in ['bad','{"values":[true,1]}','{"values":[NaN,1]}','{"values":[1]}']:
        assert not parse_prediction(text,2)["valid"]
    assert not parse_prediction('{"values":[1,2],"lower":[3,4],"upper":[5,6]}',2)["interval_valid"]

def test_metrics_known_perfect_and_swapped():
    a=np.array([[1.],[2.]]);b=np.array([[3.],[4.]])
    assert paired_effect_mae(a,b,a,b)==0
    assert paired_effect_mae(b,a,a,b)==4
    assert trajectory_mae(a,a)==0
    ci=cluster_ci([1,1,3],["p1","p1","p2"],seed=1)
    assert ci["mean"]==2
    assert ci["clusters"]==2

def test_single_variable_scalar_normalization_never_fills_missing_variables():
    r=parse_prediction('{"values":120,"lower":100,"upper":140}',1)
    assert r["valid"] and r["values"]==[120.] and r["scalar_normalized"] and r["interval_valid"]
    assert not parse_prediction('{"values":120}',2)["valid"]
    assert not parse_prediction('{"values":true}',1)["valid"]
