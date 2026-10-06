import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from controller_history_v24 import history_block,infer_scaling,fit_history_stats


def test_two_topics_hold_independently_and_ignore_future_rows():
    s={'timestamp':np.array([10,20,40]),**{f'{a}speed_integ':np.array([1.,2.,99.]) for a in ['roll','pitch','yaw']}}
    g={'timestamp':np.array([8,25,50]),**{f'compression_gains[{i}]':np.array([.8,.6,.3]) for i in range(3)}}
    x,si,gi=history_block(s,g,np.array([20,30]))
    np.testing.assert_array_equal(si,[1,1]);np.testing.assert_array_equal(gi,[0,1])
    np.testing.assert_allclose(x[:,6:],[ [0,12e-6],[10e-6,5e-6]],rtol=1e-6)
    s['rollspeed_integ'][2]=9999
    np.testing.assert_array_equal(history_block(s,g,np.array([20,30]))[0],x)
    with pytest.raises(ValueError):history_block(s,g,np.array([5]))


def test_scaling_inversion_requires_nonzero_feedforward_and_uses_matching_rate():
    rates=np.array([[.1,-.2,.3],[0.,0.,0.]])
    sp=np.array([[.4,.5,-.6],[0.,0.,0.]])
    p=np.array([.04,.06,.05]);ff=np.array([.8,.4,.3]);scale=1.25
    selected,ratio,used=infer_scaling((sp-rates)*p,sp*ff/scale,rates,p,ff)
    np.testing.assert_allclose(selected[0],scale);assert np.isnan(selected[1])
    np.testing.assert_allclose(ratio[0],scale);np.testing.assert_allclose(used,sp)


def test_fit_stats_constant_channels_have_finite_floor():
    mean,scale=fit_history_stats(np.ones((5,8)))
    np.testing.assert_array_equal(mean,np.ones(8));np.testing.assert_array_equal(scale,np.full(8,1e-4))
    with pytest.raises(ValueError):fit_history_stats(np.ones((5,7)))
