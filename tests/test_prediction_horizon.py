from dataclasses import fields
import copy
import numpy as np
import torch
from test_main_v2_simulator import make_simulator,make_inputs,initial
from system_identification.evaluation.prediction_horizon import PHYSICAL,teacher_refresh_rollout,kinematic_hold,variation_ratio,error_statistics


def test_teacher_refresh_first_step_equal_and_history_does_not_read_future():
    sim,x=make_simulator(),make_inputs();s=initial(sim,x)
    p={key:getattr(s,key)[:,None].repeat((1,51)+( (1,) if getattr(s,key).ndim==2 else ())) for key in PHYSICAL}
    hc=x['history_controls'][:,:1].repeat(1,51,1)
    cmd=x['future_controls'][:,:5];dt=x['dt_s'][:,:5]
    with torch.no_grad():
        pred=teacher_refresh_rollout(sim,s,p,hc,cmd,dt)
        first,_=sim.step(s,cmd[:,0],dt[:,0])
        for key in PHYSICAL:np.testing.assert_array_equal(pred[key][:,1],getattr(first,key).numpy())
        poison={key:value.clone() for key,value in p.items()}
        for value in poison.values():value[:,30:]=float('nan')
        bad=hc.clone();bad[:,30:]=float('nan')
        second=teacher_refresh_rollout(sim,s,poison,bad,cmd,dt)
        for key in pred:np.testing.assert_array_equal(pred[key],second[key])


def test_autonomous_pipeline_ignores_poisoned_future_labels():
    from test_trajectory_main_v1 import _samples,_windows
    from scripts.run_main_v2_free_running import make_initial,rollout
    from system_identification.training.trajectory_main_v1 import assemble_history_trajectory_windows
    batch=assemble_history_trajectory_windows(_samples(),_windows(),history_steps=26);poison=copy.deepcopy(batch)
    for key in PHYSICAL:getattr(poison.trajectory.truth,key)[:,1:]=np.nan
    sim=make_simulator();outputs=[]
    with torch.inference_mode():
        for data in [batch,poison]:
            s=make_initial(sim,data,np.arange(2),'warm26','cpu')
            result,_=rollout(sim,s,torch.tensor(data.trajectory.controls[:,:5],dtype=torch.float32),torch.tensor(data.trajectory.dt_s[:,:5],dtype=torch.float32));outputs.append(result)
    for key in outputs[0]:np.testing.assert_array_equal(outputs[0][key],outputs[1][key])


def test_hold_integrates_real_dt_and_variation_one_step_is_increment_ratio():
    sim,x=make_simulator(),make_inputs();s=initial(sim,x);dt=torch.tensor([[.01,.03],[.02,.025]])
    hold=kinematic_hold(s,dt)
    np.testing.assert_allclose(hold['position_n'][:,-1],(s.position_n+s.velocity_n*dt.sum(1)[:,None]).numpy(),atol=1e-6)
    assert np.allclose(np.linalg.norm(hold['quaternion_nb'],axis=-1),1,atol=1e-6)
    truth=np.array([[[0.,0,0],[2.,0,0]],[[0.,0,0],[0.,0,0]]]);pred=truth*.25
    ratio,_=variation_ratio(pred,truth,1);assert ratio[0]==.25 and np.isnan(ratio[1])
    stats=error_statistics(np.array([1.,1.,3.]),np.array(['a','a','b']))
    assert stats['equal_flight_rmse']==2 and stats['pooled_rmse']!=2
