from copy import deepcopy
import json
import pytest
import numpy as np
from medwm_eval.baselines import fit_controls,features
from medwm_eval.diagnostics import diagnostic_prompt,parse_oneshot
from test_prompts_metrics import example

def test_controls_reject_test_training():
    x=example();x["split"]="test"
    with pytest.raises(ValueError):fit_controls([x],{},"anesthesia")

def test_features_no_future_action_or_label():
    x=example("vitaldb");p=x["payload"];h=p["history"]
    f=features(p,h,0)
    p["actions"][1]["values"]=[999,999];p["oracle"]=[999,999]
    np.testing.assert_array_equal(f,features(p,h,0))

def test_time_shift_changes_only_coordinates():
    x=example();x["id"]="unit"
    text=diagnostic_prompt(x,"time_shift_60")
    view,_=json.JSONDecoder().raw_decode(text[text.index('{"variables"'):])
    assert view["predict_time_min"]==62
    assert view["history_rows"][0][0]==60
    assert view["scheduled_actions"][0][0]==60
    assert view["history_rows"][0][1:3]==[50,80]
    assert x["payload"]["history"][0]["time"]==0

def test_oneshot_shape_failure_retained():
    assert not parse_oneshot('{"values":[[1,2]]}',2)["valid"]
    assert parse_oneshot('{"values":[[1,2],[3,4],[5,6],[7,8]]}',2)["valid"]
