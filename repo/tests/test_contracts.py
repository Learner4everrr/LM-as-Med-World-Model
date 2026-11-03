from copy import deepcopy
import json
import numpy as np
import pytest
from medwm_eval.envs.simglucose_adapter import (
    Recipe, advance, build, fingerprint, future_noise, point, reference_rate,
    validate_rate, visible_payload, warmup,
)
from medwm_eval.rollout import free_running, teacher_forced


def test_reset_replays_observations_and_hidden_state():
    e = build(Recipe())
    r = reference_rate(e)
    initial = point(e)
    a = advance(e, r, 180)
    state = e.patient.state.copy()
    e.reset()
    assert point(e) == initial
    assert advance(e, r, 180) == a
    assert np.array_equal(e.patient.state, state)


def test_pairing_branch_order_and_noise():
    e, r, _ = warmup(Recipe())
    anchor = fingerprint(e)
    a, b = deepcopy(e), deepcopy(e)
    assert fingerprint(a) == fingerprint(b) == anchor
    assert future_noise(a, 100) == future_noise(b, 100)
    ra = advance(a, r * .8, 120)
    rb = advance(b, r * 1.2, 120)
    assert fingerprint(e) == anchor
    b2, a2 = deepcopy(e), deepcopy(e)
    assert advance(b2, r * 1.2, 120) == rb
    assert advance(a2, r * .8, 120) == ra
    assert future_noise(a, 100) == future_noise(b, 100)
    assert ra[-1]["glucose_mg_dL"] > rb[-1]["glucose_mg_dL"]
    assert [x["announced_meal_g"] for x in ra] == [x["announced_meal_g"] for x in rb]


def test_units_meal_integration_and_no_silent_clipping():
    e = build(Recipe())
    assert e.sample_time == 3
    rows = advance(e, .01, 60, bolus_rate=.02)
    assert sum(x["announced_meal_g"] for x in rows) == pytest.approx(20)
    assert sum(x["applied_insulin_U_min"] * 3 for x in rows) == pytest.approx(
        (e.pump.basal(.01) + e.pump.bolus(.02)) * 60)
    for value in (-1, float("nan"), float("inf"), 1e6):
        with pytest.raises(ValueError):
            validate_rate(e, value)
    with pytest.raises(ValueError):
        advance(e, .01, 4)


def test_payload_allowlist_and_future_canaries():
    recipe = Recipe()
    e, r, history = warmup(recipe)
    payload = visible_payload(history, recipe, validate_rate(e, r))
    corrupted = deepcopy(history)
    for row in corrupted:
        row.update(patient_state="HIDDEN", future_glucose="FUTURE", seed=999,
                   oracle_bg_mg_dL=-999)
    assert visible_payload(corrupted, recipe, validate_rate(e, r)) == payload
    serialized = json.dumps(payload)
    for forbidden in ("patient_state", "oracle", "seed", "adult#", "split", "label_path"):
        assert forbidden not in serialized
    assert max(x["time_min"] for x in payload["history"]) == 0
    assert payload["preannounced_meals"] == [{"time_min": 30, "carbohydrate_g": 20.0}]


def test_true_self_feeding_and_teacher_forcing():
    seen = []
    def predict(history, actions, target):
        seen.append(deepcopy(history))
        return history[-1]["glucose_mg_dL"] + 1
    history = [{"time_min": 0, "glucose_mg_dL": 100.}]
    actions = [{"basal_U_min": .01}] * 4
    free = free_running(predict, history, actions)
    assert [x["glucose_mg_dL"] for x in free] == [101, 102, 103, 104]
    assert seen[2][-1] == {"time_min": 60, "glucose_mg_dL": 102,
                            "previous_interval_action": {"basal_U_min": .01}}
    teacher = teacher_forced(predict, history, actions, [200, 300, 400, 500])
    assert [x["glucose_mg_dL"] for x in teacher] == [101, 201, 301, 401]
    assert history == [{"time_min": 0, "glucose_mg_dL": 100.}]


def test_invalid_rollout_is_failure():
    with pytest.raises(ValueError):
        free_running(lambda *_: float("nan"),
                     [{"time_min": 0, "glucose_mg_dL": 100}], [{}])

def test_reject_future_history():
    recipe = Recipe()
    e, r, history = warmup(recipe)
    history.append({"t_min": 150, "glucose_mg_dL": 123})
    with pytest.raises(ValueError, match="future"):
        visible_payload(history, recipe, r)


def test_reset_observation_is_not_first_history_entry():
    e = build(Recipe())
    reset = e.reset()
    assert point(e)["glucose_mg_dL"] == reset.observation.CGM
    # Document the upstream double measurement; do not silently equate these.
    assert e.CGM_hist[0] != reset.observation.CGM
