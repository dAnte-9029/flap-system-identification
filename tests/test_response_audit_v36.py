import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from response_audit_v36 import anchors, response_trace, response_metrics, fixed_support_lags


def test_anchor_uses_correct_previous_commands_and_step_time():
    h = np.zeros((1, 26, 4)); f = np.zeros((1, 50, 4)); f[:, 9:, 1] = -2
    k, value, contrast = anchors(h, f)
    assert k[0, 1] == 9 and value[0, 1] == -2
    np.testing.assert_array_equal(value[0, [0, 2, 3]], 0)
    # Current origin appears once, not in the previous-command baseline.
    h[:, -1, 0] = 3; f[:, :, 0] = 3
    k, value, _ = anchors(h, f)
    assert k[0, 0] == 0 and value[0, 0] == 3


def test_response_retains_opposite_and_zero_outcomes():
    h = np.full((26, 3), 4.); f = np.full((50, 3), 6.)
    truth = response_trace(h, f, 0)
    np.testing.assert_array_equal(truth, 2)
    m = response_metrics(-truth, truth)
    np.testing.assert_array_equal(m['direction_match'], 0)
    np.testing.assert_array_equal(m['response_mse'], 16)
    m = response_metrics(truth, np.zeros_like(truth))
    np.testing.assert_array_equal(m['truth_neutral'], 1)
    assert np.isnan(m['shape_cosine']).all()


def test_lag_sign_and_common_support():
    t = np.column_stack([np.sin(np.arange(25)*.7)]*3)
    p = np.zeros_like(t); p[2:] = t[:-2]
    lags = fixed_support_lags(p, t)
    assert min(lags, key=lambda k: lags[k].sum()) == 2
    np.testing.assert_allclose(lags[2], 0, atol=1e-30)
