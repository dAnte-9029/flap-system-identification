import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from run_existing_response_candidates import hold_flags,nearest_opposite,weighted

def test_hold_requires_target_and_other_channels():
    first=np.array([1.,-1.,1.]);late=np.array([[1,1],[-1,-1],[1,-1.]])
    other=np.array([[0,0],[1,0],[0,0]])
    np.testing.assert_array_equal(hold_flags(first,late,other,np.array([.1,.1])),[True,False,False])
def test_opposite_matching_uses_only_features_and_flight():
    x=np.array([[0.],[.1],[2.]]);s=np.array([1,-1,-1]);f=np.array(['a','a','b'])
    assert nearest_opposite(x,s,f,True)[0][:2]==(0,1)
    assert nearest_opposite(x,s,f,False)[0][:2]==(0,2)
def test_native_weighted_mean():
    np.testing.assert_allclose(weighted(np.array([[[1.],[3.]]]),np.array([[.1,.3]])),[[2.5]])
