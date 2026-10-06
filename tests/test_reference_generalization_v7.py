import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from reference_generalization_v7 import (phase_prediction, phase_loss, scheduled_rollout,
    held_start_schedule, history_phase, choose_candidate)
from test_reference_capacity_v6 import models


def synthetic_history(n=12):
    rng = np.random.default_rng(8)
    h = rng.normal(size=(n, 26, 14))*.05
    f = np.linspace(3, 4, n)
    phi = f[:, None]*2*np.pi*(np.arange(26)-25)*.02
    h[:, :, 9] = np.sin(phi)
    h[:, :, 10] = np.cos(phi)
    h[:, :, 11] = f[:, None]
    return h


def test_phase_unwrap_and_slope_include_origin_without_wrap_jump():
    h = synthetic_history()
    np.testing.assert_allclose(history_phase(h)[:, -1], 0)
    hold, _ = phase_prediction('hold', h)
    for name in ['phase_slope6', 'phase_slope11', 'phase_slope26', 'damped_trend']:
        p, _ = phase_prediction(name, h)
        np.testing.assert_allclose(p, hold, atol=1e-12)
    assert np.max(phase_loss(hold+2*np.pi, hold)) < 1e-14


def test_fitted_phase_parameters_ignore_query_and_unwrap_training_labels():
    h = synthetic_history(); y, _ = phase_prediction('hold', h)
    logs = np.repeat(['a', 'b', 'c'], 4)
    p, params = phase_prediction('ridge10', h[:2], h, (y+np.pi)%(2*np.pi)-np.pi, logs)
    _, other = phase_prediction('ridge10', h[:2]+.01, h, y, logs)
    for key in params:
        np.testing.assert_allclose(params[key], other[key], atol=1e-11)
    np.testing.assert_allclose(p, y[:2], atol=1e-10)


def test_schedule_reproduces_frozen_decoder_and_respects_prefix():
    model, _, x = models(); torch.nn.init.normal_(model.head[-1].weight, std=.01)
    schedule = held_start_schedule(x)
    with torch.no_grad(): expected = model(**x)
    actual = scheduled_rollout(model, x, schedule)
    for k in actual: torch.testing.assert_close(actual[k], expected[k], rtol=0, atol=0)
    changed = {k: v.clone() for k, v in x.items()}
    changed['future_reference'][:, 5:] += 10
    schedule[:, 5:] += 2
    later = scheduled_rollout(model, changed, schedule)
    for k in actual: torch.testing.assert_close(actual[k][:, :5], later[k][:, :5], rtol=0, atol=0)


def test_selection_is_deterministic_on_training_flights():
    h = synthetic_history(); y, _ = phase_prediction('hold', h)
    logs = np.repeat(['a', 'b', 'c'], 4)
    name, scores = choose_candidate(h, y, logs, ['hold', 'frequency_mean5'])
    assert name == 'hold'
    assert scores['hold'] < 1e-7
