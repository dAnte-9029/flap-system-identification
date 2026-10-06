import io
from dataclasses import replace
import numpy as np
import pytest
import torch

from system_identification.models.actuator_command_gru import ActuatorCommandGRU
from system_identification.models.actuator_command_step import ActuatorCommandStepper


def fixture():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    stats = {}
    for name, width in [('state', 14), ('reference', 5), ('actuator', 4)]:
        stats[name+'_mean'] = np.zeros(width)
        stats[name+'_scale'] = np.ones(width)
    stats.update(rate_scale=np.ones(3), velocity_scale=np.ones(3))
    model = ActuatorCommandGRU(stats).eval()
    with torch.no_grad():
        model.head[-1].weight.normal_(0, .03)
    inputs = dict(history_state=torch.randn(2, 26, 14),
        history_command=torch.randn(2, 26, 4), history_actuator=torch.randn(2, 26, 4),
        initial_quaternion=torch.tensor([[1., 0, 0, 0]]).repeat(2, 1),
        initial_velocity_n=torch.randn(2, 3), initial_rates_b=torch.randn(2, 3),
        current_mode=torch.tensor([[1., 0], [0., 1]]))
    return model, inputs, torch.randn(2, 50, 4), torch.full((2, 50), .02)


def test_exact_native_parity_and_horizon():
    model, inputs, commands, dt = fixture()
    adapter = ActuatorCommandStepper(model)
    state = adapter.initialize(**inputs)
    with torch.no_grad():
        expected = model(**inputs, future_command=commands, dt_s=dt)
    for k in range(50):
        state, output = adapter.step(state, commands[:, k], dt[:, k])
        for name in expected:
            torch.testing.assert_close(output[name], expected[name][:, k], rtol=0, atol=0)
    with pytest.raises(ValueError, match='horizon'):
        adapter.step(state, commands[:, 0], dt[:, 0])


def test_saved_restore_and_independent_interleaving():
    model, inputs, commands, dt = fixture()
    adapter = ActuatorCommandStepper(model)
    a = adapter.initialize(**inputs)
    b = adapter.initialize(**inputs)
    altered = commands.clone()
    altered[:, 10:] += .25
    with torch.no_grad():
        expected = model(**inputs, future_command=altered, dt_s=dt)
    differences = []
    for k in range(50):
        b, out_b = adapter.step(b, altered[:, k], dt[:, k])
        a, out_a = adapter.step(a, commands[:, k], dt[:, k])
        if k == 16:
            buffer = io.BytesIO()
            torch.save(adapter.snapshot(a), buffer)
            buffer.seek(0)
            # A fresh adapter can continue the same session with identical weights.
            a = ActuatorCommandStepper(model).restore(torch.load(buffer, weights_only=True))
        for name in expected:
            torch.testing.assert_close(out_b[name], expected[name][:, k], rtol=0, atol=0)
            if k < 10:
                torch.testing.assert_close(out_a[name], out_b[name], rtol=0, atol=0)
        differences.append(not torch.equal(out_a['rates_b'], out_b['rates_b']))
    assert any(differences[10:])
    with torch.no_grad():
        original = model(**inputs, future_command=commands, dt_s=dt)
    torch.testing.assert_close(a.rates_b, original['rates_b'][:, -1], rtol=0, atol=0)


def test_snapshot_input_and_output_ownership():
    model, inputs, commands, dt = fixture()
    adapter = ActuatorCommandStepper(model)
    state = adapter.initialize(**inputs)
    snapshot = adapter.snapshot(state)
    restored = adapter.restore(snapshot)
    snapshot['hidden'].fill_(999)
    assert not torch.equal(snapshot['hidden'], restored.hidden)
    inputs['history_state'].fill_(999)
    inputs['initial_velocity_n'].fill_(999)
    inputs['current_mode'].fill_(999)
    assert not (state.velocity_n == 999).any() and not (state.air == 999).any()
    before = adapter.snapshot(state)
    next_state, output = adapter.step(state, commands[:, 0], dt[:, 0])
    output['rates_b'].fill_(999)
    assert not (next_state.rates_b == 999).any()
    for name, value in before.items():
        actual = adapter.snapshot(state)[name]
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual, value, rtol=0, atol=0)
    # Cloned weights isolate an active session from caller-side model changes.
    with torch.no_grad():
        model.head[-1].bias.add_(100)
    repeated, _ = adapter.step(state, commands[:, 0], dt[:, 0])
    torch.testing.assert_close(repeated.rates_b, next_state.rates_b, rtol=0, atol=0)


def test_invalid_state_model_time_and_input_rejected():
    model, inputs, commands, dt = fixture()
    adapter = ActuatorCommandStepper(model)
    state = adapter.initialize(**inputs)
    snapshot = adapter.snapshot(state)
    with torch.no_grad():
        model.state_mean[0] += 1
    with pytest.raises(ValueError, match='contract'):
        ActuatorCommandStepper(model).restore(snapshot)
    incomplete = dict(snapshot)
    del incomplete['air']
    with pytest.raises(ValueError):
        adapter.restore(incomplete)
    with pytest.raises(ValueError):
        adapter.step(state, commands[:, 0], torch.full((2,), .0025))
    with pytest.raises(ValueError):
        adapter.step(state, torch.full((2, 4), float('nan')), dt[:, 0])
    with pytest.raises(ValueError):
        adapter.step(replace(state, hidden=torch.zeros(2, 3)), commands[:, 0], dt[:, 0])
    with pytest.raises(ValueError):
        adapter.initialize(**dict(inputs, history_state=inputs['history_state'][:, :25]))
