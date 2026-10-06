"""Input-only anchors and descriptive observed-response metrics (not causal gains)."""
import numpy as np


def anchors(history, future):
    """Largest absolute before/after5-command contrast in first25 origins.

    history ends at origin, future starts at origin. One anchor per parent/channel;
    no output or model information enters this retrospective event definition.
    """
    h, f = np.asarray(history, float), np.asarray(future, float)
    before = np.concatenate([h[:, -6:-1], f], axis=1)
    contrast = np.stack([f[:, k:k+5].mean(1)-before[:, k:k+5].mean(1)
                         for k in range(25)], axis=1)
    offsets = np.abs(contrast).argmax(1)
    selected = np.take_along_axis(contrast, offsets[:, None], axis=1)[:, 0]
    return offsets, selected, contrast


def response_trace(history_rates, future_rates, offset):
    """Post25 curve minus its own pre5 mean. Retains weak/opposite responses."""
    h, f = np.asarray(history_rates, float), np.asarray(future_rates, float)
    joined = np.concatenate([h[-5:], f], axis=0)
    return f[offset:offset+25]-joined[offset:offset+5].mean(0)


def response_metrics(pred, truth):
    p, t = np.asarray(pred, float), np.asarray(truth, float)
    assert p.shape == t.shape == (25, 3)
    ps, ts = p[5:].mean(0), t[5:].mean(0)
    # Numerical zero only, no response-strength exclusion or tuned noise cutoff.
    sign = lambda a: np.where(abs(a) <= 1e-12, 0, np.sign(a))
    pc, tc = p-p.mean(0), t-t.mean(0)
    den = np.sqrt((pc**2).mean(0)*(tc**2).mean(0))
    cosine = np.divide((pc*tc).mean(0), den, out=np.full(3, np.nan), where=den > 1e-12)
    return dict(response_mse=((p-t)**2).mean(0), predicted_power=(p**2).mean(0),
        truth_power=(t**2).mean(0), predicted_sustained=ps, truth_sustained=ts,
        sustained_error=ps-ts, sustained_error_squared=(ps-ts)**2,
        direction_match=(sign(ps) == sign(ts)).astype(float), truth_neutral=(sign(ts) == 0).astype(float),
        shape_cosine=cosine, predicted_area=.02*p.sum(0), truth_area=.02*t.sum(0))


def fixed_support_lags(pred, truth):
    """15 common target samples, +/-100ms oracle alignment, NOT actuator delay."""
    target = truth[5:20]-truth[5:20].mean(0)
    return {lag: ((pred[5+lag:20+lag]-pred[5+lag:20+lag].mean(0)-target)**2).mean(0)
            for lag in range(-5, 6)}
