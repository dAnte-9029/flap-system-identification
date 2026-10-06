"""Outcome-independent features and offline waveform/kinematic diagnostics."""
import numpy as np
import torch
from system_identification.models.trajectory_main_v1 import (
    _normalize_quaternion, _quaternion_multiply, _delta_quaternion)


def modal_commands(commands):
    a = np.asarray(commands, dtype=float)
    return np.stack([a[..., 0], .5*(a[..., 1]+a[..., 2]),
                     .5*(a[..., 2]-a[..., 1]), a[..., 3]], axis=-1)


def features(history):
    a = np.asarray(history, dtype=float)
    return np.concatenate([a.mean(1), a.std(1), a[:, -1],
                           np.sqrt(np.mean(np.diff(a, axis=1)**2, axis=1))], axis=1)


def integrate_rates(initial_quaternion, initial_rates, rates, dt):
    """Same midpoint kinematics as model; future-truth substitutions are oracle only."""
    q = _normalize_quaternion(torch.as_tensor(initial_quaternion, dtype=torch.float32))
    previous = torch.as_tensor(initial_rates, dtype=torch.float32)
    rates = torch.as_tensor(rates, dtype=torch.float32)
    dt = torch.as_tensor(dt, dtype=torch.float32)
    out = []
    for k in range(rates.shape[1]):
        q = _normalize_quaternion(_quaternion_multiply(q,
            _delta_quaternion(.5*(previous+rates[:, k])*dt[:, k, None])))
        out.append(q); previous = rates[:, k]
    return torch.stack(out, 1).numpy()


def angle_squared(pred, truth):
    # Preserve v30's metric, including its float64 dot and sign invariance.
    dot = np.abs((np.asarray(pred, dtype=float)*np.asarray(truth, dtype=float)).sum(-1)).clip(0, 1)
    return (2*np.arccos(dot))**2


def lag_errors(pred, truth):
    """Fixed 40-sample support for every lag; centered MSE, descriptive only.

    Positive lag aligns later predictions with earlier truth. Minimum after
    seeing truth is an oracle waveform diagnostic, not measured actuator delay.
    """
    p = np.asarray(pred, dtype=float); t = np.asarray(truth, dtype=float)
    target = t[:, 5:45]; target = target-target.mean(1, keepdims=True)
    result = {}
    for lag in range(-5, 6):
        shifted = p[:, 5+lag:45+lag]
        shifted = shifted-shifted.mean(1, keepdims=True)
        result[lag] = np.mean((shifted-target)**2, axis=1)
    return result
