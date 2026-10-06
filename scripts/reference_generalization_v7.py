"""History-only phase candidates and frozen-weight rollout diagnostics."""
import numpy as np
import torch
from diagnose_reference_predictability_v6 import fit_ridge
from system_identification.models.reference_gru_unit_gravity import unit_state_features
from system_identification.models.trajectory_main_v1 import (
    _normalize_quaternion, _rotation_body_to_ned, _delta_quaternion, _quaternion_multiply)

CANDIDATES = ['hold', 'frequency_mean5', 'phase_slope6', 'phase_slope11',
              'phase_slope26', 'damped_trend', 'ridge10', 'ridge100']


def history_phase(history):
    p = np.unwrap(np.arctan2(history[:, :, 9], history[:, :, 10]), axis=1)
    return p - p[:, -1:]


def slope(values):
    t = (np.arange(values.shape[1]) - values.shape[1] + 1) * .02
    t -= t.mean()
    return (values @ t) / (t @ t)


def phase_features(history):
    phase = history_phase(history)
    freq = history[:, :, 11]
    return np.column_stack([freq[:, -1], freq[:, -5:].mean(1), freq.mean(1),
        slope(freq[:, -11:]), freq.std(1),
        *[slope(phase[:, -n:]) / (2*np.pi) for n in [6, 11, 26]],
        np.linalg.norm(history[:, -1, :3], axis=1),
        history[:, -1, 3:6], history[:, -1, 6:9]])


def phase_prediction(name, history, train_history=None, train_phase=None, train_logs=None):
    """Only learned candidates accept labeled *training* futures; query has history only."""
    time = np.arange(1, 51) * .02
    freq = history[:, :, 11]
    baseline = 2*np.pi*freq[:, -1, None]*time
    params = {}
    if name == 'hold':
        return baseline, params
    if name == 'frequency_mean5':
        return 2*np.pi*freq[:, -5:].mean(1)[:, None]*time, params
    if name.startswith('phase_slope'):
        n = int(name[len('phase_slope'):])
        return slope(history_phase(history)[:, -n:])[:, None]*time, params
    if name == 'damped_trend':
        drift = slope(freq[:, -11:])
        return baseline + 2*np.pi*drift[:, None]*.15*(time-.15*(1-np.exp(-time/.15))), params
    if name not in ['ridge10', 'ridge100']:
        raise ValueError(name)
    # The relative phase source wraps; unwrap with the known origin first.
    phase = np.unwrap(np.column_stack([np.zeros(len(train_phase)), train_phase]), axis=1)[:, 1:]
    train_base = 2*np.pi*train_history[:, -1, 11, None]*time
    residual, params = fit_ridge(phase_features(train_history), phase-train_base,
        phase_features(history), train_logs, float(name[5:]))
    return baseline + residual, params


def phase_loss(prediction, truth):
    """Bounded harmonic chord loss: per-window, equal first/second harmonics and time."""
    delta = prediction-truth
    return np.mean(np.stack([2-2*np.cos(k*delta) for k in [1, 2]]), axis=(0, 2))


def choose_candidate(history, phase, logs, candidates=CANDIDATES):
    """LOO selection; the caller must pass training flights only."""
    scores = {}
    for name in candidates:
        flight_scores = []
        for log in sorted(set(logs)):
            fit = logs != log
            query = ~fit
            pred, _ = phase_prediction(name, history[query], history[fit], phase[fit], logs[fit])
            flight_scores.append(float(np.sqrt(phase_loss(pred, phase[query]).mean())))
        scores[name] = float(np.mean(flight_scores))
    return min(scores, key=scores.get), scores


@torch.no_grad()
def scheduled_rollout(model, x, phase_start):
    """Diagnostic replay with an explicit start-of-step phase schedule.

    All other state/context evolution matches the frozen integrated model.
    Caller marks oracle schedules nondeployable; no optimizer is constructed.
    """
    hs = x['history_state']
    history = torch.cat([(hs-model.state_mean)/model.state_scale,
        (x['history_reference']-model.reference_mean)/model.reference_scale,
        (x['history_actuator']-model.actuator_mean)/model.actuator_scale,
        x['current_mode'][:, None, :].expand(-1, hs.shape[1], -1)], dim=-1)
    _, h = model.encoder(history)
    h = h[0]
    quat = _normalize_quaternion(x['initial_quaternion'])
    vel, rate = x['initial_velocity_n'], x['initial_rates_b']
    freq = hs[:, -1, 11]
    anchor = torch.zeros_like(freq)
    air = hs[:, -1, 12:14]
    result = {k: [] for k in ['velocity_n', 'quaternion', 'rates_b']}
    for k in range(x['future_reference'].shape[1]):
        features = torch.cat([unit_state_features(vel, quat, rate, phase_start[:, k], anchor, freq), air], -1)
        ref = (x['future_reference'][:, k]-model.reference_mean)/model.reference_scale
        inp = torch.cat([(features-model.state_mean)/model.state_scale, ref, x['current_mode']], -1)
        h = model.cell(inp, h)
        derivative = model.head(torch.cat([h, inp], -1)).clamp(-6., 6.)*model.derivative_scale
        dt = x['dt_s'][:, k, None]
        vel = vel + torch.einsum('bij,bj->bi', _rotation_body_to_ned(quat), derivative[:, :3])*dt
        newrate = rate + derivative[:, 3:]*dt
        quat = _normalize_quaternion(_quaternion_multiply(quat, _delta_quaternion(.5*(rate+newrate)*dt)))
        rate = newrate
        for key, value in [('velocity_n', vel), ('quaternion', quat), ('rates_b', rate)]:
            result[key].append(value)
    return {k: torch.stack(v, 1) for k, v in result.items()}


def held_start_schedule(x):
    # Same recurrence/order as the original decoder, including float32 rounding.
    freq = x['history_state'][:, -1, 11]
    phase = torch.zeros_like(freq)
    result = []
    for k in range(x['dt_s'].shape[1]):
        result.append(phase)
        phase = phase + 2*torch.pi*freq*x['dt_s'][:, k]
    return torch.stack(result, 1)
