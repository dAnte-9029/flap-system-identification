import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from reference_error_structure_v9 import decompose,history_features,fit_predict,score_mean_readout


def test_exact_error_decomposition_for_bias_amplitude_and_inverted_shape():
    t=np.tile(np.sin(np.arange(50)[None,:,None]*.5),(3,1,3));p=2*t+np.array([1.,-2.,.3])
    d=decompose(p,t)
    np.testing.assert_allclose(d['total_mse'],d['bias_mse']+d['centered_mse'],atol=1e-14)
    np.testing.assert_allclose(d['centered_mse'],d['amplitude_mse']+d['shape_mse'],atol=1e-14)
    np.testing.assert_allclose(d['shape_mse'],0,atol=1e-14)
    z=decompose(-t,t);np.testing.assert_allclose(z['amplitude_mse'],0,atol=1e-14)
    np.testing.assert_allclose(z['shape_mse'],4*np.var(t,axis=1),atol=1e-14)


def test_decomposition_handles_constant_signals_and_corrected_readout_identity():
    rng=np.random.default_rng(9);t=rng.normal(size=(4,50,3));p=np.zeros_like(t);b=rng.normal(size=(4,3))
    d=decompose(p,t);assert np.isfinite(np.stack(list(d.values()))).all()
    np.testing.assert_allclose(d['shape_mse'],0,atol=1e-14)
    scores=score_mean_readout(p,t,b)
    np.testing.assert_allclose(scores['waveform_error_squared'],np.mean((p-b[:,None]-t)**2,axis=1),atol=1e-14)


def test_history_features_ignore_future_and_query_partition_metadata():
    rng=np.random.default_rng(3)
    x={k:rng.normal(size=(5,26,n)) for k,n in [('history_state',14),('history_reference',5),('history_actuator',4)]}
    x['current_mode']=np.zeros((5,2));x['future_reference']=rng.normal(size=(5,50,5))
    a=history_features(x,'full');x['future_reference'][:]=np.nan;x['future_truth']=np.ones((5,50,3))*999
    np.testing.assert_array_equal(a,history_features(x,'full'))
    assert a.shape==(5,94) and history_features(x,'state').shape==(5,58)


def test_regression_statistics_do_not_use_query_distribution():
    rng=np.random.default_rng(8);x=rng.normal(size=(12,6));y=rng.normal(size=(12,3));q=rng.normal(size=(3,6));logs=np.repeat(['a','b','c'],4)
    _,a=fit_predict(x,y,q,logs,'ridge');_,b=fit_predict(x,y,q+100,logs,'ridge')
    for key in a:np.testing.assert_array_equal(a[key],b[key])


def test_constant_baseline_weights_flights_not_windows():
    x=np.zeros((4,2));y=np.array([[0.],[0.],[0.],[2.]]);logs=np.array(['a','a','a','b'])
    p,_=fit_predict(x,y,x[:2],logs,'constant');np.testing.assert_array_equal(p,np.ones((2,1)))
