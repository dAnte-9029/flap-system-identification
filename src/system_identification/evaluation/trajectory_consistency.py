"""Read-only vector integration diagnostics; never modifies trajectory labels."""
from __future__ import annotations
import numpy as np
import pandas as pd


def integration_residual(position, velocity, dt_s):
    position, velocity, dt_s = map(np.asarray, (position, velocity, dt_s))
    if position.shape != velocity.shape or position.shape[-1] != 3:
        raise ValueError('position and velocity must share (..., time, 3) shape')
    if dt_s.shape != position.shape[:-2] + (position.shape[-2] - 1,):
        raise ValueError('dt shape mismatch')
    if not np.isfinite(dt_s).all() or (dt_s <= 0).any():
        raise ValueError('dt must be finite and positive')
    return np.diff(position, axis=-2) - .5 * (velocity[..., 1:, :] + velocity[..., :-1, :]) * dt_s[..., None]


def metric_tables(vectors, log_ids, steps):
    """Per-log vector/axis RMSE and separately named equal-log/pooled estimates.

    Each input has shape (window, interval, 3); step=50 selects index 49.
    Nonfinite values become infinity, never silently dropped.
    """
    log_ids = np.asarray(log_ids).astype(str)
    rows = []
    for name, values in vectors.items():
        values = np.asarray(values)
        if values.ndim != 3 or values.shape[0] != len(log_ids) or values.shape[2] != 3:
            raise ValueError('expected (window, interval, 3)')
        for step in steps:
            selected = values[:, step-1]
            for log in np.unique(log_ids):
                x = selected[log_ids == log]
                bad = ~np.isfinite(x).all(axis=1)
                square = np.where(np.isfinite(x), x*x, np.inf)
                rows.append(dict(metric=name, step=step, nominal_horizon_s=step/50,
                    log_id=log, n_windows=len(x), nonfinite_windows=int(bad.sum()),
                    vector_rmse=float(np.sqrt(np.mean(square.sum(axis=1)))),
                    north_rmse=float(np.sqrt(np.mean(square[:, 0]))),
                    east_rmse=float(np.sqrt(np.mean(square[:, 1]))),
                    down_rmse=float(np.sqrt(np.mean(square[:, 2])))))
    per = pd.DataFrame(rows)
    aggregate = []
    cols = ['vector_rmse', 'north_rmse', 'east_rmse', 'down_rmse']
    for (metric, step), g in per.groupby(['metric', 'step'], sort=False):
        for kind in ['equal_log', 'pooled_windows']:
            row = dict(metric=metric, step=step, nominal_horizon_s=step/50,
                       aggregation=kind, n_logs=len(g), n_windows=int(g.n_windows.sum()),
                       nonfinite_windows=int(g.nonfinite_windows.sum()))
            for col in cols:
                x = g[col].to_numpy()
                row[col] = float(np.mean(x) if kind == 'equal_log' else
                                 np.sqrt(np.average(x*x, weights=g.n_windows)))
            aggregate.append(row)
    return per, pd.DataFrame(aggregate)


def shifted_velocity_residual(position, velocity, times_s, lag_s, margin_s=.11):
    """Diagnostic only: common interior span, linear interpolation at t+lag.

    Endpoints are cropped equally for every lag; do not compare this span with a
    full horizon or promote the fitted lag to a production timestamp correction.
    """
    residual = []
    duration = []
    for p, v, t in zip(position, velocity, times_s, strict=True):
        if np.any(np.diff(t) <= 0) or abs(lag_s) > margin_s:
            raise ValueError('invalid lag or time')
        keep = (t >= t[0]+margin_s) & (t <= t[-1]-margin_s)
        target = t[keep]
        if len(target) < 2:
            raise ValueError('insufficient common interior')
        shifted = np.column_stack([np.interp(target+lag_s, t, v[:, axis]) for axis in range(3)])
        residual.append(integration_residual(p[keep], shifted, np.diff(target)).sum(axis=0))
        duration.append(target[-1]-target[0])
    return np.asarray(residual), np.asarray(duration)
