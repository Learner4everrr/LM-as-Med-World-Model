"""Train-only numerical controls with identical visible state and action information."""
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

def features(payload,history,step,with_actions=True):
    y=np.array([h["values"] for h in history],float)
    last=y[-1]
    # Fixed dimension; observed and generated histories share this representation.
    lag=[y[max(0,len(y)-1-k)] for k in range(4)]
    out=np.concatenate(lag+[y[-min(4,len(y)):].mean(0)]).tolist()
    out += [payload["step_min"]]
    demo=payload["demographics"]
    out += [float(demo.get(k,0)) for k in ["age_years","height_cm","weight_kg"]]
    out += [1. if demo.get("sex")=="male" else 0.]
    if with_actions:
        for k in range(4):
            h=history[max(0,len(history)-1-k)]
            out += list(h["past_action"])+[h.get("past_meal_g",0)]
        a=payload["actions"][step]
        out += list(a["values"])+[a["meal_g"]]
    return np.asarray(out,float)

def fit_controls(cases,truth,domain):
    from .prompts import feedback
    rows=[x for x in cases if x["domain"]==domain]
    if not rows or any(x["split"]!="development" for x in rows):
        raise ValueError("Fit requires development cases only")
    xfull=[];xstate=[];delta=[]
    for case in rows:
        p=case["payload"];history=p["history"]
        for step,target in enumerate(truth[case["id"]]):
            xfull.append(features(p,history,step,True))
            xstate.append(features(p,history,step,False))
            delta.append(np.array(target)-history[-1]["values"])
            history=feedback(history,target,p["actions"][step],p["step_min"])
    y=np.asarray(delta)
    models={}
    for name,x in [("ridge_action",xfull),("ridge_history",xstate)]:
        models[name]=make_pipeline(StandardScaler(),Ridge(alpha=10.)).fit(x,y)
    models["boosted_action"]=MultiOutputRegressor(HistGradientBoostingRegressor(
        max_iter=200,max_leaf_nodes=15,min_samples_leaf=10,learning_rate=.05,
        l2_regularization=1.,random_state=20261102)).fit(xfull,y)
    return models,{"domain":domain,"cases":len(rows),"patients":len({x["patient"] for x in rows}),
                   "training_transitions":len(y),"test_used_for_fit":False}

def predict(name,models,payload,history,step):
    last=np.array(history[-1]["values"],float)
    if name=="persistence":return last.tolist()
    if name=="trend":
        return (2*last-np.asarray(history[-2]["values"])).tolist()
    x=features(payload,history,step,name!="ridge_history")
    return (last+models[name].predict(x[None])[0]).tolist()
