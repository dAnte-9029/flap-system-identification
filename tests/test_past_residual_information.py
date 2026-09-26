import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from test_control_dynamics_residual_information import preceding_windows,fit_gain


def test_old_forecast_ends_at_current_origin():
    origins=pd.DataFrame(dict(window_id=['a','b'],start_sample_in_segment=[25,80],state_sample_count=[51,51],
        origin_timestamp_us=[100,200],end_timestamp_us=[300,400],history_span_s=[.5,.5]))
    old=preceding_windows(origins)
    np.testing.assert_array_equal(old.start_sample_in_segment+old.state_sample_count-1,origins.start_sample_in_segment)
    assert not any('timestamp' in c for c in old)
    assert origins.start_sample_in_segment.iloc[0]==25


def test_no_past_cross_segment_boundary():
    with pytest.raises(ValueError,match='causal past'):
        preceding_windows(pd.DataFrame(dict(window_id=['x'],start_sample_in_segment=[3],state_sample_count=[51])))


def test_persistent_and_anti_persistent_gains_distinguished():
    x=np.array([[-2.,1.],[-1.,3.],[1.,-1.],[2.,-3.]])
    y=x*np.array([.5,-.7])
    gain=fit_gain(x,y,np.array(['a','a','b','b']))
    np.testing.assert_allclose(gain,np.array([.5,-.7])/1.01)
    assert gain.clip(0,1)[1]==0


def test_gain_fit_equal_flights_not_equal_samples():
    x=np.array([[1.],[2.],[3.],[4.]])
    y=np.array([[.2],[.7],[2.],[3.]])
    ids=np.array(['a','a','b','b'])
    ix=[0,1,2,3,0,1,0,1]
    np.testing.assert_allclose(fit_gain(x,y,ids),fit_gain(x[ix],y[ix],ids[ix]))
