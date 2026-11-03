"""Pinned native simglucose adapter. Research feasibility, not clinical dosing."""
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import hashlib
import json
import math
import pickle

import numpy as np
from simglucose.actuator.pump import InsulinPump
from simglucose.controller.base import Action
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario import CustomScenario


@dataclass(frozen=True)
class Recipe:
    patient: str = "adult#001"
    seed: int = 41
    warmup_min: int = 120
    horizon_min: int = 120
    meals: tuple = ((30, 20.0), (150, 20.0))

    def to_dict(self):
        return asdict(self)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def build(recipe):
    patient = T1DPatient.withName(recipe.patient, random_init_bg=True, seed=recipe.seed)
    sensor = CGMSensor.withName("Dexcom", seed=recipe.seed + 10000)
    pump = InsulinPump.withName("Insulet")
    scenario = CustomScenario(datetime(2020, 1, 1), [
        (timedelta(minutes=t), grams) for t, grams in recipe.meals
    ])
    env = T1DSimEnv(patient, sensor, pump, scenario)
    env.reset()  # Always use one canonical reset path; constructor alone differs.
    return env


def reference_rate(env):
    # Simulator equilibrium input; internal generation only, never expose parameters.
    return float(env.patient._params.u2ss * env.patient._params.BW / 6000)


def validate_rate(env, rate, channel="basal"):
    if not math.isfinite(rate) or rate < 0:
        raise ValueError("Insulin rate must be finite and nonnegative")
    if not float(env.pump._params["min_" + channel]) <= rate <= float(env.pump._params["max_" + channel]):
        raise ValueError("Rate outside pump software bounds")
    return float(getattr(env.pump, channel)(rate))


def point(env):
    return {"t_min": float(env.patient.t), "glucose_mg_dL": float(env.sensor._last_CGM if len(env.CGM_hist) == 1 else env.CGM_hist[-1])}


def advance(env, rate, duration_min, bolus_rate=0.0):
    dt = int(env.sample_time)
    if duration_min <= 0 or duration_min % dt:
        raise ValueError("Duration must be a positive multiple of sensor sample time")
    validate_rate(env, rate)
    validate_rate(env, bolus_rate, "bolus")
    rows = []
    for _ in range(duration_min // dt):
        start = float(env.patient.t)
        result = env.step(Action(basal=rate, bolus=bolus_rate))
        if not env.patient._odesolver.successful():
            raise RuntimeError("Integrator failed")
        row = point(env)
        row.update(interval_start_min=start, requested_basal_U_min=float(rate),
                   requested_bolus_U_min=float(bolus_rate),
                   applied_insulin_U_min=float(env.insulin_hist[-1]),
                   announced_meal_g=float(env.CHO_hist[-1] * dt),
                   oracle_bg_mg_dL=float(result.info["bg"]),
                   done=bool(result.done))
        if not all(math.isfinite(v) for v in row.values()):
            raise RuntimeError("Nonfinite simulation")
        rows.append(row)
        if result.done:
            raise RuntimeError("Simulator termination boundary reached")
    return rows


def warmup(recipe):
    env = build(recipe)
    rate = reference_rate(env)
    initial = point(env)
    rows = advance(env, rate, recipe.warmup_min)
    return env, rate, [initial] + rows


def fingerprint(env):
    # Includes physiology, digestion memory, noise cursor/buffer/RNG, schedule.
    # No new sensor.measure call: that would consume noise at the current time.
    p = env.patient
    noise = env.sensor._noise_generator
    internal = (p.state.copy(), p.t, p.planned_meal, p._last_Qsto,
                p._last_foodtaken, tuple(p._last_action), p.is_eating,
                p.random_state.get_state(), noise, env.sensor._last_CGM,
                env.scenario.scenario, env.scenario.start_time,
                env.CGM_hist, env.insulin_hist, env.CHO_hist)
    return hashlib.sha256(pickle.dumps(internal, protocol=5)).hexdigest()


def future_noise(env, n):
    noise = deepcopy(env.sensor._noise_generator)
    return [float(next(noise)) for _ in range(n)]


def visible_payload(history, recipe, applied_rate):
    # Explicit allowlist: never pass native Step.info or provenance to a model.
    anchor = recipe.warmup_min
    times = [row["t_min"] for row in history]
    if not times or times[-1] != anchor or any(t > anchor for t in times):
        raise ValueError("History must end at the anchor and contain no future")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Observation times must be strictly increasing")
    past = []
    for row in history:
        item = {"time_min": row["t_min"] - anchor,
                "glucose_mg_dL": row["glucose_mg_dL"]}
        if "interval_start_min" in row:
            item.update(previous_interval_start_min=row["interval_start_min"] - anchor,
                        previous_interval_insulin_U_min=row["applied_insulin_U_min"],
                        previous_interval_announced_meal_g=row["announced_meal_g"])
        past.append(item)
    return {
        "glucose_unit": "mg/dL", "time_unit": "minute",
        "history": past,
        "specified_actions": [{"start_min": 0, "end_min": recipe.horizon_min,
                               "basal_U_min": applied_rate, "bolus_U_min": 0.0}],
        "preannounced_meals": [{"time_min": t-anchor, "carbohydrate_g": g}
                              for t, g in recipe.meals if anchor <= t < anchor+recipe.horizon_min],
        "prediction_times_min": list(range(30, recipe.horizon_min + 1, 30)),
    }
