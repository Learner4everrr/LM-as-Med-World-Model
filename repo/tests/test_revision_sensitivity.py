"""Statistical-unit and missingness contracts for post hoc revision summaries."""
import importlib.util
from pathlib import Path
import pandas as pd
import pytest

spec=importlib.util.spec_from_file_location("revision_sensitivity",
    Path(__file__).resolve().parents[1]/"scripts/43_revision_sensitivity.py")
revision=importlib.util.module_from_spec(spec)
spec.loader.exec_module(revision)


def test_equal_identity_weight_after_within_pair_averaging():
    # Unequal point counts must not change equal scenario/identity weighting.
    rows=[dict(patient="a",group="a1",error=0.)]*6
    rows += [dict(patient="a",group="a2",error=10.)]
    rows += [dict(patient="b",group="b1",error=20.)]
    result=revision.ci(pd.DataFrame(rows),"error")
    assert result["mean"] == pytest.approx(12.5)
    assert result["clusters"] == 2 and result["n"] == 3


def test_partial_pair_is_excluded_only_in_complete_pair_sensitivity():
    frame=pd.DataFrame([
        dict(patient="a",group="a1",available=True,complete=True,effect_error=4.,zero_error=2.,excess_error=2.),
        dict(patient="a",group="a1",available=True,complete=True,effect_error=4.,zero_error=2.,excess_error=2.),
        dict(patient="b",group="b1",available=True,complete=False,effect_error=10.,zero_error=3.,excess_error=7.),
        dict(patient="b",group="b1",available=False,complete=False,effect_error=float("nan"),zero_error=3.,excess_error=float("nan")),
    ])
    available=revision.estimate(frame,"available_points")
    complete=revision.estimate(frame,"complete_pairs")
    assert available["excess_error"]["mean"] == pytest.approx(4.5)
    assert available["raw_availability"] == .75
    assert complete["excess_error"]["mean"] == 2.
    assert complete["available_pairs"] == complete["available_clusters"] == 1
    assert complete["excess_error"]["lo"] is None
    assert complete["raw_availability"] == .5
    assert complete["scheduled_points"] == 4


def test_empty_stratum_does_not_invent_zero_error():
    empty=pd.DataFrame(columns=["patient","group","error"])
    result=revision.ci(empty,"error")
    assert result == dict(mean=None,lo=None,hi=None,clusters=0,n=0)
