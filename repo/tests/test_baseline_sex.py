from medwm_eval.baselines_revised import normalized_payload
from medwm_eval.baselines import features
from test_prompts_metrics import example
import numpy as np

def test_vitaldb_sex_encoding_equivalent_to_simulator_spelling():
    c=example("vitaldb");p=c["payload"];p["demographics"]={"sex":"M"}
    canon=normalized_payload(p)
    assert canon["demographics"]["sex"]=="male" and p["demographics"]["sex"]=="M"
    f=features(canon,p["history"],0)
    p["demographics"]["sex"]="male"
    np.testing.assert_array_equal(f,features(p,p["history"],0))
    p["demographics"]["sex"]="F"
    assert normalized_payload(p)["demographics"]["sex"]=="female"
