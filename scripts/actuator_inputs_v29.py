"""Train-only actuator input replacement and unique-row normalization."""
import numpy as np


def replace_inputs(parent, grid):
    if grid.shape != (len(parent['input_history_state']), 77, 4):
        raise ValueError('expected N x 77 x 4 command grid')
    if not np.isfinite(grid).all():
        raise ValueError('nonfinite command')
    np.testing.assert_array_equal(grid[:, :26], parent['input_history_actuator'])
    result = {k: v.copy() for k, v in parent.items()
              if k not in ['input_history_reference', 'input_future_reference']}
    result['input_history_command'] = grid[:, 1:27].copy()
    result['input_future_command'] = grid[:, 26:76].copy()
    return result


def unique_fit_commands(grid, query_us, log_ids, fit_indices):
    """Unique (flight, query time), origin-25..+49. Excludes final audit row."""
    rows = {}
    for i in fit_indices:
        for t, value in zip(query_us[i, 1:76], grid[i, 1:76]):
            key = (str(log_ids[i]), int(t))
            if key in rows:
                np.testing.assert_array_equal(rows[key], value)
            rows[key] = value
    if not rows:
        raise ValueError('empty fit set')
    keys = sorted(rows)
    return np.stack([rows[k] for k in keys]), keys


def command_stats(values):
    if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all():
        raise ValueError('finite four-channel command matrix required')
    mean = np.r_[values.astype(float).mean(0), 0.]
    scale = np.r_[values.astype(float).std(0).clip(1e-4), 1.]
    return mean, scale
