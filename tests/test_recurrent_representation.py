from dataclasses import fields, replace
import io
import torch
from test_main_v2_simulator import make_simulator, make_inputs, initial
from system_identification.models.rolling_history_simulator import RollingHistorySimulator,RollingHistoryState,state_features
from system_identification.evaluation.recurrent_representation import teacher_recurrent_step,PHYSICAL


def test_teacher_recurrence_uses_true_next_feature_and_keeps_proxies_anchor():
    sim,x=make_simulator(),make_inputs()
    with torch.no_grad():
        s=initial(sim,x);u=x['future_controls'][:,0];dt=x['dt_s'][:,0]
        predicted,_=sim.step(s,u,dt)
        truth_now={k:getattr(s,k) for k in PHYSICAL}
        next_true={k:getattr(predicted,k).clone() for k in PHYSICAL}
        next_true['angular_velocity_b']+=1
        next_true['flap_frequency_hz']+=.5
        state,pred,_=teacher_recurrent_step(sim,s,u,dt,truth_now,next_true)
        expected=sim.model.base_model.recurrent_cell(sim.model.base_model._model_input(state_features(state),u),s.gru_hidden)
        torch.testing.assert_close(state.gru_hidden,expected,rtol=0,atol=0)
        assert not torch.allclose(state.gru_hidden,pred.gru_hidden)
        for name in ('drive_state','tail_state','phase_anchor'):
            torch.testing.assert_close(getattr(state,name),getattr(predicted,name),rtol=0,atol=0)
        torch.testing.assert_close(pred.angular_velocity_b,predicted.angular_velocity_b,rtol=0,atol=0)


def test_predicted_history_resume_is_bitwise_equal_all_buffers():
    sim,x=make_simulator(),make_inputs()
    with torch.inference_mode():
        for k in (13,26):
            rolling=RollingHistorySimulator(sim,k)
            s=rolling.reset(initial(sim,x),x['history_state_features'],x['history_controls'],x['history_mask'])
            uninterrupted=[]
            for j in range(250):
                s,_=rolling.step(s,x['future_controls'][:,j],x['dt_s'][:,j]);uninterrupted.append(s)
            stream=io.BytesIO();torch.save(uninterrupted[99].snapshot(),stream);stream.seek(0)
            restored=RollingHistoryState.from_snapshot(torch.load(stream,weights_only=True))
            for j in range(100,250):
                restored,_=rolling.step(restored,x['future_controls'][:,j],x['dt_s'][:,j])
                for f in fields(restored.physical):
                    torch.testing.assert_close(getattr(restored.physical,f.name),getattr(uninterrupted[j].physical,f.name),rtol=0,atol=0)
                for name in ('features','controls','mask'):
                    torch.testing.assert_close(getattr(restored,name),getattr(uninterrupted[j],name),rtol=0,atol=0)


def test_rolling_only_appends_its_own_prediction_and_keeps_phase_anchor():
    sim,x=make_simulator(),make_inputs()
    with torch.no_grad():
        s=initial(sim,x);r=RollingHistorySimulator(sim,26)
        state=r.reset(s,x['history_state_features'],x['history_controls'],x['history_mask'])
        original=state.features.clone()
        for j in range(30):
            state,_=r.step(state,x['future_controls'][:,j],x['dt_s'][:,j])
            torch.testing.assert_close(state.features[:,-1],state_features(state.physical),rtol=0,atol=0)
            torch.testing.assert_close(state.physical.phase_anchor,s.phase_anchor,rtol=0,atol=0)
        assert torch.isfinite(state.features).all()
        assert not torch.equal(state.features,original)


def test_predicted_reencode_pipeline_ignores_poisoned_future_labels():
    import copy
    import numpy as np
    from test_trajectory_main_v1 import _samples,_windows
    from scripts.run_main_v2_free_running import make_initial
    from system_identification.training.trajectory_main_v1 import assemble_history_trajectory_windows
    batch=assemble_history_trajectory_windows(_samples(),_windows(),history_steps=26)
    poison=copy.deepcopy(batch)
    for name in PHYSICAL: getattr(poison.trajectory.truth,name)[:,1:]=np.nan
    sim=make_simulator()
    with torch.inference_mode():
        for k in (13,26):
            r=RollingHistorySimulator(sim,k);outputs=[]
            for data in (batch,poison):
                s=r.reset(make_initial(sim,data,np.arange(2),'warm26','cpu'),
                    torch.tensor(data.history_state_features,dtype=torch.float32),
                    torch.tensor(data.history_controls,dtype=torch.float32),torch.tensor(data.history_mask))
                for j in range(data.trajectory.controls.shape[1]):
                    s,_=r.step(s,torch.tensor(data.trajectory.controls[:,j],dtype=torch.float32),torch.tensor(data.trajectory.dt_s[:,j],dtype=torch.float32))
                outputs.append(s)
            for f in fields(outputs[0].physical):
                torch.testing.assert_close(getattr(outputs[0].physical,f.name),getattr(outputs[1].physical,f.name),rtol=0,atol=0)
            torch.testing.assert_close(outputs[0].features,outputs[1].features,rtol=0,atol=0)
