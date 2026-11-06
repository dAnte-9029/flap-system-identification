"""CPU directed architecture, causal and numerical contracts; no model training."""
import copy
from dataclasses import replace
import math
import time
import pytest
import torch
from system_identification.models.pacld import PACLD, Config, ControlTape, phase_carrier
from system_identification.models.pacld.types import initial_state
from system_identification.models.pahbd import Config as SyntheticConfig
from system_identification.models.pahbd import geometry as so3
from system_identification.data.pahbd_synthetic import controlled_periodic_window
from system_identification.training.pahbd import trajectory_loss


torch.set_num_threads(1)


def fixture():
    torch.manual_seed(17)
    model = PACLD(Config(width=32)).double()
    w, truth = controlled_periodic_window(SyntheticConfig(), .2)
    return model, w, truth


def test_shared_encoder_and_complete_native_forward_backward():
    m, w, _ = fixture()
    start = time.perf_counter()
    b, pair = m.predict_pair(w)
    times = sorted({e.measurement for e in w.targets})
    loss = sum(trajectory_loss(pred, w.targets, times)[0] for pred in pair.values())
    loss.backward()
    assert b.z_s.shape == (12,) and b.c.shape == (6,)
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    for module in (m.belief.shared, m.belief.slow_head, m.belief.fast_head,
                   m.flow.dynamics.slow, m.flow.dynamics.equilibrium,
                   m.flow.dynamics.relaxation, m.flow.dynamics.coupling, m.flow.dynamics.decoder):
        assert sum(p.grad.abs().sum() for p in module.parameters()) > 0
    for states in pair.values():
        assert len(states) == len(times)
        for s, t in zip(states, times):
            assert s.time == t and s.last_measurement <= 0 and s.last_availability <= 0
            for v in (s.x, s.z_s, s.c, s.f, s.phi, s.variance):
                assert torch.isfinite(v).all()
            assert abs(s.x[6:10].norm()-1) < 1e-12
            assert torch.linalg.eigvalsh(s.phase_covariance).min() > 0
    print(f'synthetic pair forward/backward: {time.perf_counter()-start:.3f}s, loss={float(loss.detach()):.6g}')


def test_phase_rotations_wrap_and_gradient():
    c = torch.tensor([1., 0., 1., 0., 1., 0.], dtype=torch.float64)
    phi = torch.tensor([math.pi/2], dtype=c.dtype, requires_grad=True)
    assert torch.allclose(phase_carrier(phi, c), c.new_tensor([0, 1, -1, 0, 0, -1]), atol=1e-14)
    assert torch.allclose(phase_carrier(phi+2*math.pi, c), phase_carrier(phi, c), atol=1e-14)
    assert (phase_carrier(phi*0+2*math.pi-1e-8, c)-phase_carrier(phi*0+2*math.pi+1e-8, c)).abs().max() < 7e-8
    assert torch.autograd.gradcheck(phase_carrier, (phi, c.requires_grad_()))
    assert phase_carrier(phi.expand(2, 1), c.expand(2, 6)).shape == (2, 6)


def test_one_decoder_base_full_nested_phase_and_position_contract():
    m, w, _ = fixture()
    s = initial_state(w, m.config).with_(c=torch.linspace(-.6, .8, 6, dtype=torch.float64))
    u = w.controls.values[0]
    d = m.flow.dynamics
    calls = []
    hook = d.decoder.register_forward_hook(lambda module, args, out: calls.append(id(module)))
    full = d.decode(s, u, 'FULL')
    base = d.decode(s, u, 'BASE')
    hook.remove()
    assert calls == [id(d.decoder)]*2
    assert not torch.allclose(full, d.decode(s.with_(phi=s.phi+.7), u, 'FULL'))
    assert torch.equal(base, d.decode(s.with_(phi=s.phi+.7), u, 'BASE'))
    for mode in ('BASE', 'FULL'):
        moved = s.x.clone(); moved[:3] += 1000
        assert torch.equal(d.decode(s, u, mode), d.decode(s.with_(x=moved), u, mode))
    zero = s.with_(c=s.c*0)
    assert torch.equal(d.decode(zero, u, 'BASE'), d.decode(zero, u, 'FULL'))
    no_reference = s.with_(phase_quality=s.phase_quality*0)
    assert torch.equal(d.decode(no_reference, u, 'BASE'), d.decode(no_reference, u, 'FULL'))
    base_a = m.flow.propagate(s, w.start, -.15, w.controls, 'BASE')
    base_b = m.flow.propagate(s.with_(phi=s.phi+.7), w.start, -.15, w.controls, 'BASE')
    assert torch.equal(base_a.x, base_b.x) and torch.equal(base_a.c, base_b.c)


def test_control_gradients_and_exact_actuator_step():
    m, w, _ = fixture()
    s = initial_state(w, m.config)
    u = w.controls.values[0].clone().requires_grad_()
    u = u + u.new_tensor([.1, .2, -.1, .3])
    def transition(control):
        end = m.flow.propagate(s, w.start, w.start+.03, ControlTape([w.start], control[None]))
        return torch.cat((end.x, end.z_s, end.c, end.f, end.phi, end.actuator))
    jac = torch.autograd.functional.jacobian(transition, u)
    assert torch.isfinite(jac).all() and (jac.abs().sum(0) > 0).all()
    # Direct motor pathways must exist even holding f fixed.
    d = m.flow.dynamics
    for fn in (lambda control: d.decode(s, control), lambda control: d.latent_rates(s, control)[0],
               lambda control: d.latent_rates(s, control)[1]):
        j = torch.autograd.functional.jacobian(fn, u)
        assert j[..., 0].abs().sum() > 0
    end = m.flow.step(s, u, .005)
    expected = u[1:]+(s.actuator-u[1:])*math.exp(-.005/.06)
    assert torch.allclose(end.actuator, expected, atol=1e-14)


def test_future_observation_and_target_poison_cannot_change_belief_or_rollout():
    m, w, _ = fixture()
    a, pa = m(w)
    poisoned = copy.deepcopy(w)
    event = w.history[-1]
    poisoned.history.append(replace(event, measurement=-.1, availability=.001,
                                    value=torch.full_like(event.value, float('nan'))))
    poisoned.targets = [replace(e, value=torch.full_like(e.value, float('nan'))) for e in w.targets]
    # Also prove PA-HBD's latent field is not reused as per-window inference.
    poisoned.initial.h.fill_(999)
    b, pb = m(poisoned)
    for sa, sb in [(a, b), *zip(pa, pb)]:
        for name in ('x', 'z_s', 'c', 'f', 'phi', 'variance'):
            assert torch.equal(getattr(sa, name), getattr(sb, name))


def test_hall_anchor_circular_innovation_masks_and_delayed_replay():
    m, w, _ = fixture()
    s = initial_state(w, m.config).with_(phase_quality=torch.zeros(1, dtype=torch.float64))
    e = w.history[0]
    value = e.value.clone(); value[14] = 2*math.pi-.01
    mask = torch.zeros_like(e.mask); mask[13] = 1
    hall = replace(e, measurement=w.start, availability=w.start+.02, value=value, mask=mask, source='wing_phase')
    u = w.controls.values[0]
    bad = replace(hall, quality=torch.zeros_like(hall.quality))
    masked = m.belief.update(s, bad, 0., u)
    assert torch.equal(masked.phi, s.phi) and torch.equal(masked.z_s, s.z_s)
    a = m.belief.update(s, hall, 0., u)
    assert torch.equal(a.phi, value[14:15]) and a.phase_quality.item() == 1.
    value2 = value.clone(); value2[14] = .01
    b = m.belief.update(a, replace(hall, value=value2), 0., u)
    assert 0 <= float(b.phi-a.phi) < .021
    late = replace(hall, measurement=w.start+.04, availability=w.start+.12)
    early = replace(hall, measurement=w.start+.08, availability=w.start+.09)
    a = m.belief(m.flow, s, w.start, [late, early], w.controls)
    b = m.belief(m.flow, s, w.start, [early, late], w.controls)
    assert torch.equal(a.x, b.x) and a.last_measurement == early.measurement
    with pytest.raises(ValueError):
        m.belief(m.flow, s, w.start, [replace(hall, measurement=.01, availability=0.)], w.controls)


def test_native_times_zoh_and_time_guards():
    m, w, _ = fixture()
    s = initial_state(w, m.config)
    constant = ControlTape([w.start], w.controls.values[:1])
    split = ControlTape([w.start, -.21, -.134, -.01], w.controls.values[:1].expand(4, -1).clone())
    a = m.flow.propagate(s, w.start, .123, constant)
    b = m.flow.propagate(s, w.start, .123, split)
    assert torch.equal(a.x, b.x) and torch.equal(a.c, b.c)
    queries = [-.19, -.083, .017, .123]
    states = m.flow.rollout(s, queries, constant, start=w.start)
    assert [p.time for p in states] == queries
    assert torch.allclose(a.x, states[-1].x, atol=2e-5, rtol=1e-5)
    # A command at an endpoint does not affect integration before that endpoint.
    jump = ControlTape([w.start, -.1], torch.stack((w.controls.values[0], w.controls.values[0]+.4)))
    assert torch.equal(m.flow.propagate(s, w.start, -.1, jump).x,
                       m.flow.propagate(s, w.start, -.1, constant).x)
    for start, end in [(w.start, w.start-.01), (0., .1), (w.start, float('nan'))]:
        with pytest.raises(ValueError): m.flow.propagate(s, start, end, constant)
    with pytest.raises(ValueError): m.flow.rollout(s, [-.1, -.2], constant, start=w.start)
    with pytest.raises(ValueError): m.flow.propagate(s, w.start, .1, ControlTape([0.], constant.values))


def test_lie_midpoint_against_independent_dop853():
    m, w, truth = fixture()
    s = initial_state(w, m.config)
    # Replace only vector field with the existing independent synthetic ODE.
    def vector(state, u, mode):
        p = state.phi[0]; a = state.actuator
        body = torch.stack((.2+.5*a[0]+.4*torch.cos(p), .3*a[1]+.1*a[2]*torch.sin(p),
                            -9.80665+.25*torch.sin(p)+.3*a[2]))
        alpha = torch.stack((.2*torch.sin(p)+.6*a[0]*torch.cos(p), .1*a[1], .2*a[2]))
        return so3.rotate(state.x[6:10], body)+body.new_tensor([0, 0, 9.80665]), alpha, state.z_s*0, state.c*0
    m.flow.dynamics.forward = vector
    errors = []
    for dt in (.01, .005, .0025):
        m.config.max_step = dt
        out = m.flow.propagate(s, w.start, .2, w.controls)
        errors.append(float((out.x-torch.tensor(truth(.2)[:13])).norm()))
        assert abs(out.x[6:10].norm()-1) < 1e-12
    assert errors[1] < .35*errors[0] and errors[2] < .35*errors[1]
    assert errors[-1] < 2e-5
    print('DOP853 convergence errors:', errors)


def test_config_and_checkpoint_state_dict_roundtrip():
    m, w, _ = fixture()
    restored = PACLD(m.config).double()
    restored.load_state_dict(m.state_dict())
    assert torch.equal(m(w, [.03])[1][0].x, restored(w, [.03])[1][0].x)
    for args in ({'fast_dim': 8}, {'max_step': 0}, {'actuator_tau': float('nan')}, {'lambda_max': .1}):
        with pytest.raises(ValueError): Config(**args)


def test_latent_equations_bounds_phase_exclusion_and_so3_observation():
    m, w, _ = fixture()
    s = initial_state(w, m.config)
    u = w.controls.values[0]+.2
    d = m.flow.dynamics
    context = d.context(s, u)
    eq = 2*torch.tanh(d.equilibrium(context))
    rate = .5+19.5*torch.sigmoid(d.relaxation(context))
    K = torch.tanh(d.coupling(context)).reshape(6, 3)
    dz, dc = d.latent_rates(s, u)
    assert torch.allclose(dc, rate*(eq-s.c)+K@((u[1:]-s.actuator)/.06), atol=1e-14)
    assert (eq.abs() <= 2).all() and ((rate >= .5) & (rate <= 20)).all()
    assert (K.abs() <= 1).all() and (dz.abs() <= 10).all()
    other = d.latent_rates(s.with_(phi=s.phi+.9), u)
    assert torch.equal(dz, other[0]) and torch.equal(dc, other[1])
    e = next(e for e in w.history if e.source == 'vehicle_attitude')
    at = m.flow.propagate(s, w.start, e.measurement, w.controls)
    flipped = e.value.clone(); flipped[6:10] *= -1
    a = m.belief.update(at, e, e.measurement-w.start, u)
    b = m.belief.update(at, replace(e, value=flipped), e.measurement-w.start, u)
    assert torch.allclose(a.x, b.x, atol=1e-14) and torch.allclose(a.z_s, b.z_s, atol=1e-14)
    # Numerical check of the full propagated physical-state control derivative.
    u = u.detach().requires_grad_()
    def output(control):
        return m.flow.propagate(s, w.start, w.start+.011, ControlTape([w.start], control[None])).x
    assert torch.autograd.gradcheck(output, (u,), atol=1e-5, rtol=1e-3)
