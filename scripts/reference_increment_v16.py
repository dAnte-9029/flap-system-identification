"""Pure diagnostics of original-origin increments and history-only spectrum."""
import numpy as np


def history_spectrum(history_rates, dt=.02, cutoff=8.):
    """26-point mean-removed Hann periodogram; coarse diagnostic, not IMU truth."""
    h = np.asarray(history_rates, dtype=float)
    if h.ndim != 3 or h.shape[1] != 26 or not np.isfinite(h).all():
        raise ValueError('expected finite N x 26 x axes observed history')
    win = np.hanning(h.shape[1])
    z = (h - h.mean(axis=1, keepdims=True)) * win[None, :, None]
    power = abs(np.fft.rfft(z, axis=1)) ** 2
    power[:, 1:-1] *= 2
    power /= len(win) * np.sum(win ** 2)
    freq = np.fft.rfftfreq(len(win), dt)
    total = power.sum(axis=1)
    high = power[:, freq > cutoff].sum(axis=1)
    return np.sqrt(high), np.divide(high, total, out=np.zeros_like(high), where=total > 1e-20)


def safe_corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or min(a.std(), b.std()) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def clipped_increment(raw_rate_head, dt, scale):
    return np.clip(raw_rate_head, -6., 6.) * np.asarray(scale) * np.asarray(dt)


def metrics(g, deadband=.01):
    """One flight, one model, one axis; sign denominator retains near-zero predictions."""
    t = g.truth_delta.to_numpy(dtype=float)
    p = g.pred_delta.to_numpy(dtype=float)
    e = p-t
    eligible = abs(t) > deadband
    correct = (np.sign(t) == np.sign(p)) & (abs(p) > deadband)
    exceed = abs(t) > g.limit.to_numpy()
    rms_t, rms_p = np.sqrt(np.mean(t*t)), np.sqrt(np.mean(p*p))
    std_t, std_p = t.std(), p.std()
    result = dict(n=len(g), rmse=np.sqrt(np.mean(e*e)), bias=e.mean(),
        truth_rms=rms_t, pred_rms=rms_p, truth_std=std_t, pred_std=std_p,
        rms_ratio=rms_p/rms_t if rms_t > 1e-12 else np.nan,
        std_ratio=std_p/std_t if std_t > 1e-12 else np.nan,
        correlation=safe_corr(p, t),
        pred_on_truth_slope=np.mean((t-t.mean())*(p-p.mean()))/(std_t**2) if std_t > 1e-12 else np.nan,
        hold_gain=1-np.sqrt(np.mean(e*e))/rms_t if rms_t > 1e-12 else np.nan,
        sign_eligible=int(eligible.sum()),
        sign_agreement=correct[eligible].mean() if eligible.any() else np.nan,
        prediction_near_zero=(abs(p[eligible]) <= deadband).mean() if eligible.any() else np.nan,
        first_saturation=(abs(g.raw_head.to_numpy()) >= 6).mean(),
        rollout_saturation=g.rollout_saturation.mean(), truth_exceeds_limit=exceed.mean(),
        exceed_error_share=np.sum(e[exceed]**2)/np.sum(e**2) if np.sum(e**2)>1e-20 else 0.,
        # Pointwise best possible clamp-compatible prediction, not an attainable model.
        clamp_oracle_rmse=np.sqrt(np.mean(np.maximum(abs(t)-g.limit.to_numpy(), 0)**2)),
        hf_rms=g.hf_rms.mean(), hf_share=g.hf_share.mean(),
        hf_error_spearman=safe_corr(g.hf_rms.rank(), abs(g.pred_delta-g.truth_delta).rank()),
        hf_truth_spearman=safe_corr(g.hf_rms.rank(), abs(g.truth_delta).rank()))
    for name, a in [('truth', t), ('pred', p)]:
        for q in [.05, .5, .95]:
            result[f'{name}_q{int(q*100):02}'] = np.quantile(a, q)
    return result
