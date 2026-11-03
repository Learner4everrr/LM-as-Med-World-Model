"""PAS 1.0.0 adapter. Simulated pharmacology only."""
from copy import deepcopy
import numpy as np
import python_anesthesia_simulator as pas


def build_patient(profile, seed):
    # Upstream constructor consumes the global RNG even with output noise disabled.
    # Isolate construction; runtime is deterministic with noise=False.
    rng = np.random.get_state()
    try:
        np.random.seed(seed)
        p = pas.Patient(profile, ts=10, random_PK=True, random_PD=True,
                        co_update=False, save_data_bool=False)
    finally:
        np.random.set_state(rng)
    return p


def hidden_state(p):
    return np.concatenate([p.propo_pk.x.ravel(), p.remi_pk.x.ravel(), p.nore_pk.x.ravel()])


def observe(p):
    return [float(np.asarray(p.bis).item()), float(np.asarray(p.map).item())]


def advance(p, rates, seconds=120):
    if seconds <= 0 or seconds % p.ts:
        raise ValueError("Time must align with the PAS integration step")
    if len(rates) != 2 or any(not np.isfinite(x) or x < 0 for x in rates):
        raise ValueError("Need nonnegative finite propofol mg/s and remifentanil ug/s")
    for _ in range(int(seconds/p.ts)):
        p.one_step(u_propo=rates[0], u_remi=rates[1], u_nore=0,
                   blood_rate=0, dist=[0,0,0], noise=False)
    values = observe(p)
    if not np.isfinite(values).all():
        raise RuntimeError("Nonfinite PAS observation")
    return values


def verify_pair(profile=(50,170,70,1), seed=81):
    p = build_patient(profile, seed)
    advance(p,[.1,.1],1200)
    before = hidden_state(p).copy()
    a,b = deepcopy(p),deepcopy(p)
    assert np.array_equal(hidden_state(a),hidden_state(b))
    ra=[advance(a,[.075,.1]) for _ in range(12)]
    rb=[advance(b,[.125,.1]) for _ in range(12)]
    b2,a2=deepcopy(p),deepcopy(p)
    assert rb == [advance(b2,[.125,.1]) for _ in range(12)]
    assert ra == [advance(a2,[.075,.1]) for _ in range(12)]
    assert np.array_equal(before,hidden_state(p))
    rebuilt=build_patient(profile,seed)
    advance(rebuilt,[.1,.1],1200)
    assert np.array_equal(before,hidden_state(rebuilt))
    assert ra == [advance(rebuilt,[.075,.1]) for _ in range(12)]
    assert ra[-1][0]>rb[-1][0]
    return {"identical_anchor":True,"replay_exact":True,"order_independent":True,
            "source_unchanged":True,"action_effect":True,
            "endpoint_A_minus_B":(np.array(ra[-1])-np.array(rb[-1])).tolist()}
