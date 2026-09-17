from dataclasses import fields, replace
import io

import numpy as np
import pytest
import torch

from test_trajectory_main_v2 import _base_model, _inputs
from system_identification.models.main_v2_simulator import MainV2Simulator, SimulatorState
from system_identification.models.trajectory_main_v2 import ActuatorAwareTrajectoryModel


def make_simulator():
    return MainV2Simulator(ActuatorAwareTrajectoryModel(
        base_model=_base_model(), use_drive=True, use_tail=True, gated_tail=True,
        tail_mean=np.zeros(3), tail_std=np.ones(3)).eval())


def make_inputs():
    x = _inputs(2, 250)
    torch.manual_seed(81)
    x['history_state_features'] = torch.randn(2,26,12) * 0.2
    x['history_controls'] = torch.rand(2,26,4) * 0.5
    x['history_mask'] = torch.ones(2,26,dtype=torch.bool)
    x['history_mask'][0,:5] = False
    x['relative_phase_rad'] = torch.tensor([5.9, 2.3])
    x['future_controls'] = torch.rand(2,250,4) * 0.7
    x['dt_s'] = 0.019 + torch.rand(2,250) * 0.002
    return x


def initial(sim, x):
    return sim.reset(**{k:v for k,v in x.items() if k not in {'future_controls','dt_s'}})


def test_250_steps_equal_legacy_and_serialized_resume_all_states():
    sim, x = make_simulator(), make_inputs()
    with torch.inference_mode():
        legacy = sim.model(**x)
        state = initial(sim, x)
        all_states = [state]
        for k in range(250):
            state, _ = sim.step(state, x['future_controls'][:,k], x['dt_s'][:,k])
            all_states.append(state)
        for name in ('position_n','velocity_n','quaternion_nb','angular_velocity_b',
                     'relative_phase_rad','flap_frequency_hz'):
            torch.testing.assert_close(torch.stack([getattr(s,name) for s in all_states],1),
                                       getattr(legacy,name), rtol=0, atol=0)
        stream = io.BytesIO()
        torch.save(all_states[100].snapshot(), stream)
        stream.seek(0)
        restored = sim.reset(state=SimulatorState.from_snapshot(torch.load(stream, weights_only=True)))
        # Restoring must not encode history or depend on legacy regularization cache.
        sim.model.base_model._encode_history = lambda *a: (_ for _ in ()).throw(AssertionError('history read'))
        sim.model._last_control_residuals = [torch.full((2,7), float('nan'))]
        for k in range(100,250):
            restored, _ = sim.step(restored, x['future_controls'][:,k], x['dt_s'][:,k])
            for f in fields(restored):
                torch.testing.assert_close(getattr(restored,f.name),getattr(all_states[k+1],f.name),rtol=0,atol=0)


def test_explicit_reset_copies_without_reanchoring_and_validates():
    sim, x = make_simulator(), make_inputs()
    state = initial(sim,x)
    restored = sim.reset(state=state)
    restored.gru_hidden.add_(1)
    assert not torch.equal(restored.gru_hidden,state.gru_hidden)
    with pytest.raises(ValueError):
        sim.reset(state=state, history_mask=x['history_mask'])
    with pytest.raises(ValueError):
        sim.reset(state=replace(state, quaternion_nb=state.quaternion_nb * 2))
    with pytest.raises(ValueError):
        SimulatorState.from_snapshot({'position_n':state.position_n})
    with pytest.raises(ValueError):
        sim.step(state,x['future_controls'][:,0],torch.zeros(2))


@pytest.mark.parametrize('strategy',['single','zero','repeated26'])
def test_cold_reset_uses_only_t0_and_has_declared_proxy_state(strategy):
    sim,x = make_simulator(),make_inputs()
    physical = {k:v for k,v in x.items() if not k.startswith('history_') and k not in {'future_controls','dt_s'}}
    command = x['future_controls'][:,0]
    state = sim.cold_reset(strategy=strategy,command=command,**physical)
    torch.testing.assert_close(state.drive_state,sim.model._normalized_motor(command))
    torch.testing.assert_close(state.tail_state,sim.model._normalized_tail(command))
    if strategy == 'zero':
        assert not state.gru_hidden.any()


def test_clip_diagnostics_are_actual_preclamp_events():
    sim,x = make_simulator(),make_inputs()
    with torch.no_grad():
        sim.model.drive_head[-1].bias.fill_(100)
        sim.model.base_model.derivative_head[-1].bias.fill_(100)
    state = initial(sim,x)
    state = replace(state,flap_frequency_hz=torch.full((2,),19.99))
    _,d = sim.step(state,x['future_controls'][:,0],x['dt_s'][:,0])
    assert (d.drive_clipped == 1).all()
    assert (d.derivative_clipped == 7).all()
    assert (d.frequency_clipped == 1).all()
