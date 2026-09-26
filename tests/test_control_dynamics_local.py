"""Information boundaries for the observational local response contrast."""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from identify_control_dynamics_local import design, fit, predict


def batch():
    r=np.random.default_rng(4)
    return SimpleNamespace(history_mask=np.ones((8,26),bool),
        history_state_features=r.normal(size=(8,26,12)),
        history_controls=r.normal(size=(8,26,4)),
        trajectory=SimpleNamespace(controls=r.normal(size=(8,50,4)),truth=np.full((8,51),np.nan)))


def test_history_excludes_current_and_future_commands():
    a=batch();b=copy.deepcopy(a)
    b.history_controls[:,-1]+=100
    b.trajectory.controls+=100
    a0,a1,_=design(a,5);b0,b1,_=design(b,5)
    np.testing.assert_array_equal(a0,b0)
    assert not np.array_equal(a1,b1)


def test_tail_command_changes_do_not_enter_prefix():
    a=batch();b=copy.deepcopy(a)
    b.trajectory.controls[:,5:]=np.nan
    for x,y in zip(design(a,5)[:2],design(b,5)[:2]):np.testing.assert_array_equal(x,y)


def test_whole_flight_weighting_invariant_to_replication():
    r=np.random.default_rng(5);x=r.normal(size=(15,3));y=r.normal(size=(15,2));ids=np.array(['a']*5+['b']*10)
    model=fit(x,y,ids)
    ix=np.r_[np.arange(15),np.arange(5),np.arange(5)]
    replicated=fit(x[ix],y[ix],ids[ix])
    np.testing.assert_allclose(predict(model,x),predict(replicated,x),rtol=1e-12,atol=1e-12)


def test_known_linear_command_response_recovered():
    r=np.random.default_rng(1);x=r.normal(size=(200,3));beta=np.array([[1.,-1],[2.,1.],[-2.,3.]])
    y=x@beta+np.array([1.,4.])
    model=fit(x,y,np.array(['a']*100+['b']*100),penalty=0.)
    np.testing.assert_allclose(predict(model,x),y,rtol=1e-12,atol=1e-12)
