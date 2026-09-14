import numpy as np
import pandas as pd
import pytest

from system_identification.evaluation.trajectory_rollout_diagnostics import (
    ERROR_NAMES, error_curve, origin_metadata, prediction_errors, shifted_local_windows,
)
from system_identification.models.trajectory import ConstantTwistPredictor, TrajectoryPrediction
from system_identification.training.trajectory_main_v1 import assemble_history_trajectory_windows


def fixture():
    n=310
    s=pd.DataFrame({"log_id":["flight"]*n,"split":["validation"]*n,
        "sample_in_log":np.arange(n),"sample_in_segment":np.arange(n),
        "segment_id":np.zeros(n,dtype=int),"timestamp_us":np.arange(n)*20000,
        "valid_core":True,"valid_logged_phase":True,"logged_flap_phase_rad":.2,
        "relative_flap_phase_rad":np.arange(n)*.3,"flap_frequency_hz":4.})
    for prefix in ["position_ned_m_","velocity_ned_m_s_","angular_velocity_body_rad_s_"]:
        for axis in "xyz":s[prefix+axis]=0.
    s["position_ned_m_x"]=np.arange(n)*.04
    s["velocity_ned_m_s_x"]=2.
    for component in "wxyz":s["attitude_q_"+component]=float(component=="w")
    for name in ["flap_motor","left_elevon","right_elevon","rudder"]:
        s["control_"+name+"_normalized"]=.1
    w=pd.DataFrame([{"window_id":"parent","log_id":"flight","segment_id":0,
        "start_sample_in_log":30,"end_sample_in_log":280,"start_sample_in_segment":30,
        "end_sample_in_segment":280,"start_timestamp_us":600000,"end_timestamp_us":5600000,
        "state_sample_count":251,"control_step_count":250,"horizon_s":5.,"available_history_s":.6}])
    return s,w


def test_reset_moves_true_origin_and_history_but_keeps_parent_identity():
    s,w=fixture(); local=shifted_local_windows(s,w,50)
    assert local.parent_window_id.iloc[0]=="parent"
    assert local.start_sample_in_log.iloc[0]==80
    batch=assemble_history_trajectory_windows(s,local,history_steps=26)
    assert batch.history_mask.all()
    assert batch.trajectory.truth.position_n[0,0,0]==pytest.approx(3.2)
    original=batch.history_state_features.copy()
    s.loc[81:,"velocity_ned_m_s_x"]=99
    altered=assemble_history_trajectory_windows(s,local,history_steps=26)
    np.testing.assert_array_equal(original,altered.history_state_features)


def test_reset_cannot_cross_parent_or_segment_boundary():
    s,w=fixture()
    with pytest.raises(ValueError,match="exceeds parent"):shifted_local_windows(s,w,250)
    s.loc[100:,"segment_id"]=1
    with pytest.raises(ValueError,match="boundary"):shifted_local_windows(s,w,50)


def test_identical_first_second_in_reset_and_continuous_modes():
    s,w=fixture(); full=assemble_history_trajectory_windows(s,w,history_steps=26)
    local=assemble_history_trajectory_windows(s,shifted_local_windows(s,w,0),history_steps=26)
    predictor=ConstantTwistPredictor()
    a=predictor.rollout(full.trajectory.initial_state(),full.trajectory.controls,full.trajectory.dt_s)
    b=predictor.rollout(local.trajectory.initial_state(),local.trajectory.controls,local.trajectory.dt_s)
    for name in vars(a):np.testing.assert_allclose(getattr(a,name)[:,:51],getattr(b,name))


def test_nonfinite_prediction_is_retained_as_failure():
    s,w=fixture(); batch=assemble_history_trajectory_windows(s,shifted_local_windows(s,w,0),history_steps=26)
    truth=batch.trajectory.truth
    pred=TrajectoryPrediction(**{k:v.copy() for k,v in vars(truth).items()})
    pred.position_n[:,20]=np.nan
    errors=prediction_errors(pred,truth)
    assert np.isinf(errors[ERROR_NAMES[0]][0,20])
    curve=error_curve(errors,batch.trajectory.log_ids,batch.trajectory.dt_s,"dummy","reset_1s",1)
    row=curve.loc[np.isclose(curve.local_horizon_s,.4)].iloc[0]
    assert row.nonfinite_fraction==1 and row.global_horizon_s==pytest.approx(1.4)


def test_phase_groups_are_metadata_with_explicit_missing_group():
    s,w=fixture();s.loc[30,"valid_logged_phase"]=False
    result=origin_metadata(s,w)
    assert result.logged_phase_bin_at_origin.iloc[0]=="missing"
    assert result.window_id.iloc[0]=="parent"
