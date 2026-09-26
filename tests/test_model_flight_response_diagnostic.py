import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pandas as pd
import run_model_flight_response_diagnostic as r

def test_coordinates():
    np.testing.assert_allclose(r.coordinates(np.array([.6,.4,-.2,.1])),[.6,.1,.3,.1])
def test_activity_constant_and_pure_common():
    u=np.zeros((2,50,4));u[1,1:25,1:3]=.5
    a,s=r.activity(u,np.ones(4))
    assert np.all(a[0]==0) and a[1,1]>0 and a[1,2]==0
    np.testing.assert_array_equal(u[0],0)
def test_local_identity_bounds():
    o=pd.DataFrame(dict(window_id=['a'],start_sample_in_segment=[25],state_sample_count=[51],origin_timestamp_us=[2]))
    w=r.local_windows(o)
    np.testing.assert_array_equal(w.start_sample_in_segment,25+np.arange(0,50,5))
    assert (w.state_sample_count==6).all() and w.start_sample_in_segment.max()+5==75
    assert o.state_sample_count.iloc[0]==51 and 'origin_timestamp_us' not in w

def test_native_weight():
    np.testing.assert_allclose(r.weighted_mean(np.array([[[2.],[4.]]]),np.array([[1.,3.]])),[[3.5]])
