import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('metric_module',HERE/'analyze.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def frame(real,pred):
    n=len(real)
    return pd.DataFrame(dict(pair_uid=[str(i) for i in range(n)],supported=True,real=real,pred=pred,
        increment_error_a=np.asarray(pred)-np.asarray(real),increment_error_b=np.zeros(n),
        absolute_error_a=np.asarray(pred)-np.asarray(real),absolute_error_b=np.zeros(n),causal_real=real))


def test_interpolation_never_extrapolates():
    y,good=m.interpolate([0,.2,.49],np.array([[0],[2],[4.9]]),[0,.1,.49,.5])
    np.testing.assert_allclose(y[:3,0],[0,1,4.9]);assert np.isnan(y[3,0]);assert good.tolist()==[True,True,True,False]


def test_interpolation_rejects_duplicate_times():
    with pytest.raises(AssertionError):m.interpolate([0,0],np.zeros((2,1)),[0])


def test_zero_real_separate_and_hold_abstains():
    d=frame([.01,1,-1],[0,0,0]);r=m.metrics(d,np.ones(3),.1)
    assert r['active_pairs']==2;assert r['direction_accuracy']==0
    assert r['zero_prediction_fraction']==1;assert r['near_zero_fraction']==pytest.approx(1/3)


def test_direction_and_signed_bias():
    d=frame([1,-1],[.5,.5]);r=m.metrics(d,np.ones(2),.1)
    assert r['direction_accuracy']==.5;assert r['bias']==.5;assert r['mae']==1


def test_error_cancellation_does_not_hide_single_trajectory_error():
    d=frame([1],[1]);d['increment_error_a']=2;d['increment_error_b']=2
    d['absolute_error_a']=2;d['absolute_error_b']=2
    r=m.metrics(d,np.ones(1),.1);assert r['rmse']==0;assert r['trajectory_rmse']==2;assert r['cancellation_ratio']==0


def test_flight_and_component_weights_are_distinct():
    p=pd.DataFrame(dict(pair_uid=['0','1','2'],flight_a=['a','a','d'],flight_b=['b','c','e']))
    np.testing.assert_allclose(list(m.weights(p,'flight').values()),[.3,.3,.4])
    np.testing.assert_allclose(list(m.weights(p,'component').values()),[.25,.25,.5])


def test_seed_pooling_is_not_prediction_ensemble():
    d=frame([1,1],[0,2]);d['pair_uid']='same'
    r=m.metrics(d,np.ones(2),.1);assert r['rmse']==1;assert r['bias']==0


def test_swap_pair_sign_preserves_absolute_errors():
    a=m.metrics(frame([1,-2],[.2,-3]),[1,1],.1)
    b=m.metrics(frame([-1,2],[-.2,3]),[1,1],.1)
    assert a['rmse']==b['rmse'];assert a['bias']==-b['bias'];assert a['direction_accuracy']==b['direction_accuracy']
