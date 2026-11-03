"""Semantic sex-code normalization for numerical controls.

Correction identified by source inspection, not outcome selection. Original frozen
implementation is preserved in baselines.py; this wrapper changes no hyperparameters.
"""
from copy import deepcopy
from . import baselines as original

def normalized_payload(payload):
    p=deepcopy(payload)
    sex=str(p["demographics"].get("sex","")).lower()
    if sex in {"m","male"}:p["demographics"]["sex"]="male"
    elif sex in {"f","female"}:p["demographics"]["sex"]="female"
    return p

def fit_controls(cases,truth,domain):
    copies=[dict(c,payload=normalized_payload(c["payload"])) for c in cases]
    models,audit=original.fit_controls(copies,truth,domain)
    audit["sex_encoding_correction"]="M/m/male -> male; F/f/female -> female"
    return models,audit

def predict(name,models,payload,history,step):
    return original.predict(name,models,normalized_payload(payload),history,step)
