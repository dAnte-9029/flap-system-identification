import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from reference_increment_v16 import history_spectrum, clipped_increment, safe_corr, metrics


def test_clamp_and_increment_error_identity():
    raw=np.array([[-8., 2., 10.]])
    delta=clipped_increment(raw, .02, [20.,20.,20.])
    np.testing.assert_allclose(delta, [[-2.4,.8,2.4]])
    origin=np.array([[.2,-.5,1.]])
    truth=np.array([[1.,.5,-1.]])
    np.testing.assert_allclose((origin+delta)-(origin+truth), delta-truth)


def test_history_frequency_and_offset_invariance():
    t=np.arange(26)*.02
    h=np.stack([np.sin(2*np.pi*2*t), np.sin(2*np.pi*12*t)], axis=-1)[None]
    rms, share=history_spectrum(h)
    assert share[0,0]<.01 and share[0,1]>.98
    np.testing.assert_allclose(history_spectrum(h+23)[0], rms, atol=1e-12)
    np.testing.assert_allclose(history_spectrum(h+23)[1], share, atol=1e-12)


def test_spectrum_parseval_and_constant():
    h=np.random.default_rng(2).normal(size=(3,26,3))
    rms,share=history_spectrum(h,cutoff=-1)
    win=np.hanning(26)
    expected=np.sum(((h-h.mean(1,keepdims=True))*win[None,:,None])**2,axis=1)/sum(win**2)
    np.testing.assert_allclose(rms**2, expected)
    np.testing.assert_allclose(share,1)
    assert np.all(history_spectrum(np.ones((2,26,3)))[0]==0)
    with pytest.raises(ValueError):history_spectrum(np.zeros((1,50,3)))


def test_sign_denominator_and_limits():
    g=pd.DataFrame(dict(truth_delta=[1.,-1.,.001,4.],pred_delta=[.001,-.4,0.,2.4],
        raw_head=[.0025,-1.,0.,6.],limit=[2.4]*4,rollout_saturation=[0.]*4,
        hf_rms=[1.,2.,3.,4.],hf_share=[.5]*4))
    m=metrics(g)
    assert m['sign_eligible']==3
    assert m['sign_agreement']==pytest.approx(2/3)
    assert m['prediction_near_zero']==pytest.approx(1/3)
    assert m['truth_exceeds_limit']==.25 and m['first_saturation']==.25
    assert m['clamp_oracle_rmse']==pytest.approx(.8)
    assert np.isnan(safe_corr([1,1,1],[2,3,4]))
