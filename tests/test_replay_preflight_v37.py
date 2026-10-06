import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from replay_preflight_v37 import available_index,infer_cycle_dt,highpass_next_power


def test_causal_availability_checks_publication_and_sample():
    data={'timestamp':np.array([10,20,30]),'timestamp_sample':np.array([9,25,29])}
    assert available_index(data,8) is None
    assert available_index(data,22)==0
    assert available_index(data,25)==1
    assert available_index(data,30)==2


def test_dt_inversion_and_clamp_rejection():
    err=np.array([.2,.5,-.3]);g=np.ones(3)*.1;pre=np.zeros(3)
    post=.003*np.maximum(0,1-(err/np.deg2rad(400))**2)*g*err
    np.testing.assert_allclose(infer_cycle_dt(pre,post,err,g,np.ones(3)),.003)
    assert np.isnan(infer_cycle_dt(np.array([1.]),np.array([1.]),np.array([.2]),np.array([.1]),np.array([1.]))).all()


def test_squared_hpf_does_not_determine_next_state():
    assert .2**2==(-.2)**2
    a=highpass_next_power(.2,0,.1,.8);b=highpass_next_power(-.2,0,.1,.8)
    np.testing.assert_allclose([a,b],[.0576,.0064])
    assert a!=b
