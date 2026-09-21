import numpy as np
import pandas as pd
import pytest
from system_identification.evaluation.control_conditioned import (
    activity,control_bins,control_coordinates,attitude_rotation_vector_error,
    bootstrap_weights,paired_statistics,
)


def test_control_coordinates_match_frozen_tail_transform():
    from system_identification.models.trajectory_main_v2 import transform_tail_controls
    import torch
    u=np.array([[[.7,.8,-.2,.3],[.6,-.1,.5,-.4]]])
    x=control_coordinates(u)
    np.testing.assert_allclose(x[...,1:],transform_tail_controls(torch.tensor(u)).numpy())
    np.testing.assert_allclose(x[0,0],[.7,.3,.5,.3])


def test_large_constant_controls_are_magnitude_not_excitation():
    u=np.tile([.9,.8,.8,.7],(2,25,1))
    a=activity(u,np.zeros(4),np.ones(4))
    assert np.any(a['magnitude']>0)
    assert np.all(a['delta']==0) and np.all(a['tv']==0) and np.all(a['total']==0)
    labels,top,stats=control_bins(a['total'])
    assert set(labels)=={'Low'} and not top.any() and stats['high']==0


def test_activity_formulas_reference_and_channel_scaling():
    u=np.zeros((1,4,4));u[0,:,0]=[.1,.3,.2,.4]
    a=activity(u,[.1,0,0,0],[.1,1,1,1])
    np.testing.assert_allclose(a['magnitude'][0,0],np.sqrt(.14/4))
    np.testing.assert_allclose(a['delta'][0,0],.3)
    np.testing.assert_allclose(a['tv'][0,0],.5)
    np.testing.assert_allclose(a['total'],1.5)
    with pytest.raises(ValueError):activity(u,np.zeros(4),[0,1,1,1])


def test_bins_ties_are_preserved_without_error_or_origin_ranking():
    score=np.r_[np.zeros(90),np.ones(10)]
    labels,top,th=control_bins(score)
    assert np.all(labels[:90]=='Low') and np.all(labels[90:]=='High')
    assert np.sum(top)==10 and th['q80']==0
    perm=np.random.default_rng(7).permutation(100)
    other=control_bins(score[perm])
    np.testing.assert_array_equal(other[0],labels[perm])
    np.testing.assert_array_equal(other[1],top[perm])


def test_rotation_vector_components_are_sign_safe_and_body_frame():
    truth=np.array([[1.,0,0,0]])
    pred=np.array([[np.sqrt(.5),0,np.sqrt(.5),0]])
    vec=attitude_rotation_vector_error(pred,truth)
    np.testing.assert_allclose(vec,[[0,90,0]],atol=1e-12)
    np.testing.assert_allclose(attitude_rotation_vector_error(-pred,truth),vec)
    np.testing.assert_allclose(attitude_rotation_vector_error(truth,truth),0.)


def test_bootstrap_preserves_cohort_counts_and_equal_flight_weights():
    w=bootstrap_weights(['Sep7','Sep7','Sep17'],draws=500)
    np.testing.assert_allclose(w.sum(1),1.)
    np.testing.assert_allclose(w[:,2],1/3)
    np.testing.assert_allclose(w[:,:2].sum(1),2/3)


def test_pair_sign_flight_estimand_and_pseudoreplication():
    metadata=pd.DataFrame(dict(log_id=['a','b'],cohort=['Sep7','Sep7']))
    b2=np.array([[2.],[2.]]);b3=np.array([[1.],[5.]])
    stats,flights=paired_statistics(b2,b3,metadata,['m'],draws=2000)
    assert stats.iloc[0].flight_equal_mean_RMSE_difference==1.
    assert stats.iloc[0].B3_flight_wins==1 and stats.iloc[0].B2_flight_wins==1
    assert stats.iloc[0].origin_pooled_B3_win_fraction==.5
    # Replicating the first flight's identical origins must not tighten flight CI.
    indices=np.r_[np.zeros(100,dtype=int),1]
    other,_=paired_statistics(b2[indices],b3[indices],metadata.iloc[indices].reset_index(drop=True),['m'],draws=2000)
    for col in ['flight_equal_mean_RMSE_difference','RMSE_difference_ci95_low','RMSE_difference_ci95_high',
                'flight_equal_mean_paired_difference','mean_paired_ci95_low','mean_paired_ci95_high']:
        assert other.iloc[0][col]==stats.iloc[0][col]
    assert other.iloc[0].origin_pooled_mean_difference!=stats.iloc[0].origin_pooled_mean_difference


def test_pairing_rejects_mismatched_or_nonfinite_arrays():
    meta=pd.DataFrame(dict(log_id=['a'],cohort=['Sep7']))
    with pytest.raises(ValueError):paired_statistics(np.zeros((2,1)),np.zeros((1,1)),meta,['m'])
    with pytest.raises(ValueError):paired_statistics(np.array([[np.nan]]),np.zeros((1,1)),meta,['m'])
