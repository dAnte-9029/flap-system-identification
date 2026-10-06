import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from actuator_inputs_v29 import replace_inputs, unique_fit_commands, command_stats
from system_identification.models.actuator_command_gru import ActuatorCommandGRU
from system_identification.models.reference_gru_unit_gravity import ReferenceGRUUnitGravity


def test_time_alignment_and_audit_row_exclusion():
    grid = np.arange(2*77*4, dtype=np.float32).reshape(2, 77, 4)
    parent = dict(input_history_state=np.zeros((2, 26, 14)),
                  input_history_actuator=grid[:, :26].copy(),
                  input_history_reference=np.ones((2, 26, 5)),
                  input_future_reference=np.ones((2, 50, 5)),
                  label_rates_b=np.ones((2, 50, 3)))
    x = replace_inputs(parent, grid)
    np.testing.assert_array_equal(x['input_history_command'][:, -1], grid[:, 26])
    np.testing.assert_array_equal(x['input_future_command'][:, 0], grid[:, 26])
    np.testing.assert_array_equal(x['input_future_command'][:, -1], grid[:, 75])
    changed = grid.copy(); changed[:, 76] += 999
    for k, v in replace_inputs(parent, changed).items():
        np.testing.assert_array_equal(v, x[k])
    assert not any('reference' in k for k in x)
    np.testing.assert_array_equal(x['label_rates_b'], parent['label_rates_b'])


def test_fit_only_unique_rows_and_padding_normalization():
    grid = np.arange(3*77*4, dtype=np.float32).reshape(3, 77, 4)
    grid[1] = grid[0]
    times = np.tile(np.arange(77), (3, 1)); logs = np.array(['fit', 'fit', 'query'])
    values, keys = unique_fit_commands(grid, times, logs, [0, 1])
    assert len(keys) == 75 and {k[0] for k in keys} == {'fit'}
    before = command_stats(values)
    grid[2] = 1e8; grid[:2, 76] = -1e8
    after = command_stats(unique_fit_commands(grid, times, logs, [0, 1])[0])
    for a, b in zip(before, after): np.testing.assert_array_equal(a, b)
    assert before[0][4] == 0 and before[1][4] == 1
    grid[1, 1, 0] += 1
    with pytest.raises(AssertionError): unique_fit_commands(grid, times, logs, [0, 1])


def test_capacity_padding_equivalence_and_nonzero_response_causality():
    torch.set_num_threads(1); torch.manual_seed(17)
    stats = {}
    for name, width in [('state', 14), ('reference', 5), ('actuator', 4)]:
        stats[name+'_mean'] = np.zeros(width); stats[name+'_scale'] = np.ones(width)
    stats.update(rate_scale=np.ones(3), velocity_scale=np.ones(3))
    model = ActuatorCommandGRU(stats)
    assert sum(p.numel() for p in model.parameters()) == 40070
    # Zero-initialized output heads would make a causality test vacuous.
    with torch.no_grad(): model.head[-1].weight.normal_(0, .01)
    baseline = ReferenceGRUUnitGravity(stats); baseline.load_state_dict(model.state_dict())
    x = dict(history_state=torch.randn(2, 26, 14), history_command=torch.randn(2, 26, 4),
             history_actuator=torch.randn(2, 26, 4), future_command=torch.randn(2, 50, 4),
             initial_quaternion=torch.tensor([[1., 0, 0, 0]]).repeat(2, 1),
             initial_velocity_n=torch.zeros(2, 3), initial_rates_b=torch.zeros(2, 3),
             current_mode=torch.tensor([[1., 0]]).repeat(2, 1), dt_s=torch.full((2, 50), .02))
    legacy = {k: v for k, v in x.items() if k not in ['history_command', 'future_command']}
    legacy.update(history_reference=model.padded(x['history_command']),
                  future_reference=model.padded(x['future_command']))
    y = model(**x); old = baseline(**legacy)
    for k in y: torch.testing.assert_close(y[k], old[k], rtol=0, atol=0)
    changed = dict(x); changed['future_command'] = x['future_command'].clone()
    changed['future_command'][:, 10:] += .25
    after = model(**changed)
    for k in y: torch.testing.assert_close(y[k][:, :10], after[k][:, :10], rtol=0, atol=0)
    assert not torch.equal(y['rates_b'][:, 10:], after['rates_b'][:, 10:])
    x['future_command'].requires_grad_()
    model(**x)['rates_b'].square().sum().backward()
    grad = x['future_command'].grad
    assert torch.isfinite(grad).all() and (grad.abs().sum((0, 1)) > 0).all()
    with pytest.raises(ValueError): model.padded(torch.zeros(2, 5))
    stats['reference_mean'][4] = 1
    with pytest.raises(ValueError): ActuatorCommandGRU(stats)
