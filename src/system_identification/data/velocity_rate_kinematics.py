"""Observation consistency diagnostics, never training labels or physical truth."""
import numpy as np

from system_identification.data.velocity_rate_dataset import TARGET


def quaternion_product(a, b):
    return np.column_stack((
        a[:, 0]*b[:, 0] - np.sum(a[:, 1:]*b[:, 1:], axis=1),
        a[:, :1]*b[:, 1:] + b[:, :1]*a[:, 1:] + np.cross(a[:, 1:], b[:, 1:]),
    ))


def endpoint_residual(samples, windows):
    """Trapezoidal v and midpoint body-rate exp, on recorded measurement dt.

    The paired q/omega observations have their own timestamps. Their timing
    offsets, estimator updates, and this quadrature convention all contribute
    to these residuals; they are not physical trajectory errors.
    """
    truth = samples[TARGET].to_numpy(dtype=float)
    t = samples.state_measurement_us.to_numpy(dtype=np.int64)
    position = np.empty(len(windows))
    attitude = np.empty(len(windows))
    for steps in np.unique(windows.future_count):
        selected = np.flatnonzero(windows.future_count.to_numpy() == steps)
        origin = windows.origin_sample.to_numpy(dtype=int)[selected]
        idx = origin[:, None] + np.arange(int(steps)+1)
        state = truth[idx]
        dt = np.diff(t[idx], axis=1)*1e-6
        displacement = np.sum(.5*(state[:, :-1, 3:6]+state[:, 1:, 3:6])*dt[..., None], axis=1)
        position[selected] = np.linalg.norm(state[:, -1, :3]-state[:, 0, :3]-displacement, axis=1)
        q = state[:, 0, 6:10].copy()
        q /= np.linalg.norm(q, axis=1)[:, None]
        for k in range(int(steps)):
            rotvec = .5*(state[:, k, 10:13]+state[:, k+1, 10:13])*dt[:, k, None]
            theta = np.linalg.norm(rotvec, axis=1)
            scale = .5*np.sinc(theta/(2*np.pi))
            dq = np.column_stack((np.cos(theta/2), rotvec*scale[:, None]))
            q = quaternion_product(q, dq)
            q /= np.linalg.norm(q, axis=1)[:, None]
        end = state[:, -1, 6:10]
        end = end/np.linalg.norm(end, axis=1)[:, None]
        dot = np.clip(np.abs(np.sum(q*end, axis=1)), 0, 1)
        attitude[selected] = np.degrees(2*np.arccos(dot))
    return position, attitude
