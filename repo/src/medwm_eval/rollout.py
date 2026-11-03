"""Sequential observation updates; caller supplies a frozen transition model."""
from copy import deepcopy
import math


def free_running(predict, history, actions, step_min=30):
    """No oracle argument: subsequent observations are model outputs only."""
    return _run(predict, history, actions, step_min, None)


def teacher_forced(predict, history, actions, oracle, step_min=30):
    if len(oracle) != len(actions):
        raise ValueError("One oracle observation required per action interval")
    return _run(predict, history, actions, step_min, oracle)


def _run(predict, history, actions, step_min, oracle):
    if step_min <= 0 or not history:
        raise ValueError("Positive step and nonempty history required")
    context = deepcopy(history)
    predictions = []
    for i in range(len(actions)):
        target = context[-1]["time_min"] + step_min
        # All future actions are externally fixed; caller gets an isolated copy.
        value = float(predict(deepcopy(context), deepcopy(actions[i:]), target))
        if not math.isfinite(value):
            raise ValueError("Invalid prediction; never impute reference truth")
        predictions.append({"time_min": target, "glucose_mg_dL": value})
        feedback = value if oracle is None else float(oracle[i])
        context.append({"time_min": target, "glucose_mg_dL": feedback,
                        "previous_interval_action": deepcopy(actions[i])})
    return predictions
