from copy import deepcopy
import numpy as np
import pytest
from medwm_eval.envs.anesthesia import build_patient,advance,hidden_state,verify_pair

def test_anesthesia_exact_branching():
    result=verify_pair()
    assert all(result[x] for x in ["identical_anchor","replay_exact","order_independent","source_unchanged","action_effect"])

def test_pas_does_not_mutate_global_rng():
    np.random.seed(381)
    before=np.random.get_state()
    build_patient([50,170,70,1],55)
    after=np.random.get_state()
    assert all(np.array_equal(a,b) for a,b in zip(before,after))

def test_pas_seconds_and_rate_units():
    a=build_patient([50,170,70,1],55)
    b=deepcopy(a)
    advance(a,[.1,.2],120)
    for _ in range(12):
        advance(b,[.1,.2],10)
    np.testing.assert_array_equal(hidden_state(a),hidden_state(b))
    with pytest.raises(ValueError):
        advance(a,[-.1,.2],120)
