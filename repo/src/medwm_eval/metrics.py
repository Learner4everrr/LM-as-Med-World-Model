"""Transparent point/paired errors and patient-cluster uncertainty."""
import numpy as np

def trajectory_mae(pred,truth):
    p,t=np.asarray(pred,float),np.asarray(truth,float)
    if p.shape!=t.shape:raise ValueError("Shape mismatch")
    return np.mean(np.abs(p-t),axis=0)

def paired_effect_mae(pa,pb,ta,tb):
    pa,pb,ta,tb=map(lambda x:np.asarray(x,float),(pa,pb,ta,tb))
    if not(pa.shape==pb.shape==ta.shape==tb.shape):raise ValueError("Shape mismatch")
    return np.mean(np.abs((pa-pb)-(ta-tb)),axis=0)

def cluster_ci(values,clusters,n_boot=2000,seed=20261102):
    values=np.asarray(values,float);clusters=np.asarray(clusters)
    good=np.isfinite(values);values,clusters=values[good],clusters[good]
    means=np.array([values[clusters==c].mean() for c in sorted(set(clusters))])
    if not len(means):return {"mean":None,"lo":None,"hi":None,"clusters":0,"n":0}
    rng=np.random.default_rng(seed)
    boots=means[rng.integers(0,len(means),size=(n_boot,len(means)))].mean(axis=1)
    lo,hi=np.quantile(boots,[.025,.975])
    return {"mean":float(means.mean()),"lo":float(lo),"hi":float(hi),
            "clusters":len(means),"n":len(values)}
