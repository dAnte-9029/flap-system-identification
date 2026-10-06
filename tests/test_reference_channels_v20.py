import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from reference_channels_v20 import masked_reference,MASKS,support,quantiles


def test_mask_preserves_initial_and_only_changes_planned_channels():
    r=torch.arange(2*50*5,dtype=torch.float32).reshape(2,50,5);old=r.clone()
    for name,mask in MASKS.items():
        a=masked_reference(r,mask);assert torch.equal(a[:,0],r[:,0])
        for j in range(5):assert torch.equal(a[:,:,j],r[:,:,j] if mask&(1<<j) else r[:,:1,j].expand(-1,50))
    assert torch.equal(r,old)


def test_flight_weighted_quantiles_do_not_weight_long_flight_more():
    np.testing.assert_array_equal(quantiles(np.array([0.,0.,0.,10.]),np.array(['a','a','a','b']),[.25,.75]),[0.,10.])


def test_support_excludes_same_flight_calibration_and_query_statistics():
    f=np.array([[0.],[0.],[1.],[1.] ]);logs=np.array(['a','a','b','b'])
    a=support(f,np.array([[.5],[2.]]),logs);b=support(f,np.array([[999.]]),logs)
    np.testing.assert_allclose(a['calibration'],2.)
    np.testing.assert_array_equal(a['mean'],b['mean']);np.testing.assert_array_equal(a['scale'],b['scale'])
    assert a['threshold']==b['threshold']==2.
    assert np.isfinite(a['distance']).all()
