import copy
import numpy as np
import pandas as pd
import pytest
import torch

from test_trajectory_main_v1 import _samples,_windows
from test_main_v2_simulator import make_simulator
from system_identification.training.trajectory_main_v1 import assemble_history_trajectory_windows
from system_identification.evaluation.main_v2_free_running import select_windows,endpoint_errors,stability,PHYSICAL_FIELDS,aggregate
from scripts.run_main_v2_free_running import make_initial,rollout


def test_window_selection_preserves_segments_history_and_native_timestamps():
    rows=[]
    for log_id,n in [('a',326),('b',275)]:
        rows.append(pd.DataFrame(dict(log_id=log_id,segment_id=0,sample_in_segment=np.arange(n),
                   timestamp_us=np.arange(n)*20001,valid_core=True)))
    windows,coverage=select_windows(pd.concat(rows))
    assert windows.start_sample_in_segment.tolist()==[25,75]
    assert set(windows.log_id)=={'a'}
    assert coverage.iloc[1].excluded_reason
    assert windows.start_timestamp_us.iloc[0]==25*20001
    broken=pd.concat(rows).copy();broken.loc[3,'timestamp_us']=0
    with pytest.raises(ValueError): select_windows(broken)


def test_simulator_pipeline_is_independent_of_poisoned_future_truth():
    batch=assemble_history_trajectory_windows(_samples(),_windows(),history_steps=3)
    poison=copy.deepcopy(batch)
    for name in PHYSICAL_FIELDS: getattr(poison.trajectory.truth,name)[:,1:]=np.nan
    sim=make_simulator();idx=np.arange(2)
    cmd=torch.tensor(batch.trajectory.controls,dtype=torch.float32)
    dt=torch.tensor(batch.trajectory.dt_s,dtype=torch.float32)
    with torch.inference_mode():
        a,_=rollout(sim,make_initial(sim,batch,idx,'warm26','cpu'),cmd,dt)
        b,_=rollout(sim,make_initial(sim,poison,idx,'warm26','cpu'),cmd,dt)
    for name in a: np.testing.assert_array_equal(a[name],b[name])


def test_phase_wrap_and_quaternion_sign_do_not_create_false_errors():
    batch=assemble_history_trajectory_windows(_samples(),_windows(),history_steps=3)
    pred={n:getattr(batch.trajectory.truth,n).copy() for n in PHYSICAL_FIELDS}
    pred['relative_phase_rad']+=2*np.pi-.01
    pred['quaternion_nb']*=-1
    errors=endpoint_errors(pred,batch.trajectory.truth,2)
    np.testing.assert_allclose(errors['phase_rad'],.01)
    np.testing.assert_allclose(errors['attitude_deg'],0,atol=1e-6)


def test_finite_unsupported_and_clipped_paths_fail_and_remain_in_statistics():
    batch=assemble_history_trajectory_windows(_samples(),_windows(),history_steps=3)
    pred={n:getattr(batch.trajectory.truth,n).copy() for n in PHYSICAL_FIELDS}
    dt=batch.trajectory.dt_s
    diag=dict(acceleration_b=np.zeros((2,2,3)),angular_acceleration_b=np.zeros((2,2,3)),
              **{n:np.zeros((2,2)) for n in ('frequency_clipped','derivative_clipped','drive_clipped','residual_clipped')})
    env={n:dict(min=0.,max=100.,p1=0.,p99=100.) for n in ('speed','body_rate','frequency','acceleration','angular_acceleration')}
    env['frequency']['max']=3.
    _,flags=stability(pred,diag,dt,env)
    assert flags['failed'].all() and not flags['numerical'].any()
    env['frequency']['max']=10.
    diag['derivative_clipped'][0,0]=1
    _,flags=stability(pred,diag,dt,env)
    assert flags['failed'][0,0] and not flags['failed'][1].any()
    frame=pd.DataFrame(dict(log_id=['a','b'],horizon_s=[1.,1.],failed=[True,False],numerical_failed=[False,False],error=[10.,0.]))
    result=aggregate(frame,['horizon_s'],['error']).iloc[0]
    assert result.n_failed==1 and result.error_mean==5.
    assert result.error_rmse==np.sqrt(50.)
