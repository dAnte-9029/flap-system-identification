"""Leakage-safe diagnostics for control observability in trajectory data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.stats import wasserstein_distance


def _standardization(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    return mean, np.where(std > 1.0e-8, std, 1.0)


def fit_standardized_ridge(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    evaluation_features: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    """Fit on train only and predict evaluation rows with train normalization."""
    x = np.asarray(train_features, dtype=float)
    y = np.asarray(train_targets, dtype=float)
    x_eval = np.asarray(evaluation_features, dtype=float)
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    if x.ndim != 2 or y.ndim != 2 or x_eval.ndim != 2 or len(x) != len(y):
        raise ValueError("ridge arrays must be two-dimensional with equal train rows")
    if x.shape[1] != x_eval.shape[1] or len(x) == 0:
        raise ValueError("evaluation feature width must match nonempty train features")
    if not all(np.isfinite(array).all() for array in (x, y, x_eval)):
        raise ValueError("ridge arrays must be finite")
    feature_mean, feature_std = _standardization(x)
    target_mean, target_std = _standardization(y)
    x_scaled = (x - feature_mean) / feature_std
    y_scaled = (y - target_mean) / target_std
    gram = x_scaled.T @ x_scaled + float(alpha) * np.eye(x.shape[1])
    coefficients = np.linalg.solve(gram, x_scaled.T @ y_scaled)
    intercept = np.mean(y_scaled - x_scaled @ coefficients, axis=0)
    prediction = (((x_eval - feature_mean) / feature_std) @ coefficients + intercept)
    prediction = prediction * target_std + target_mean
    return prediction, {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "coefficients": coefficients,
        "intercept": intercept,
        "alpha": float(alpha),
    }


def predict_standardized_ridge(
    features: np.ndarray, fit: Mapping[str, np.ndarray | float]
) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    return (
        ((x - np.asarray(fit["feature_mean"])) / np.asarray(fit["feature_std"]))
        @ np.asarray(fit["coefficients"])
        + np.asarray(fit["intercept"])
    ) * np.asarray(fit["target_std"]) + np.asarray(fit["target_mean"])


def control_summary_features(
    controls: np.ndarray,
    dt_s: np.ndarray,
    *,
    steps: int,
    channel_names: Sequence[str],
    time_constants_s: Sequence[float] = (0.05, 0.15, 0.40),
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Summarize only the future commands available up to the selected horizon."""
    values = np.asarray(controls, dtype=float)
    intervals = np.asarray(dt_s, dtype=float)
    if values.ndim != 3 or intervals.shape != values.shape[:2]:
        raise ValueError("controls must be [window, step, channel] with matching dt_s")
    if steps < 1 or steps > values.shape[1] or len(channel_names) != values.shape[2]:
        raise ValueError("invalid step count or channel names")
    selected = values[:, :steps]
    selected_dt = intervals[:, :steps]
    blocks = [
        selected[:, 0],
        selected[:, -1],
        np.mean(selected, axis=1),
        np.std(selected, axis=1),
        np.sum(np.abs(np.diff(selected, axis=1)), axis=1),
    ]
    statistic_names = ["first", "last", "mean", "std", "total_variation"]
    for tau_s in time_constants_s:
        if tau_s <= 0.0:
            raise ValueError("time constants must be positive")
        state = selected[:, 0].copy()
        for index in range(1, steps):
            gain = 1.0 - np.exp(-selected_dt[:, index - 1] / float(tau_s))
            state += gain[:, None] * (selected[:, index] - state)
        blocks.append(state)
        statistic_names.append(f"lpf_tau_{tau_s:.2f}".replace(".", "p"))
    matrix = np.concatenate(blocks, axis=1)
    names = tuple(
        f"{channel}_{statistic}"
        for statistic in statistic_names
        for channel in channel_names
    )
    return matrix, names


def quaternion_relative_rotation_vector(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """Return the shortest body-frame rotation vector from q0 to q1 (wxyz)."""
    first = np.asarray(q0, dtype=float)
    second = np.asarray(q1, dtype=float)
    if first.shape != second.shape or first.shape[-1] != 4:
        raise ValueError("quaternion arrays must have equal [...,4] shapes")
    first = first / np.linalg.norm(first, axis=-1, keepdims=True)
    second = second / np.linalg.norm(second, axis=-1, keepdims=True)
    w0, x0, y0, z0 = np.moveaxis(first, -1, 0)
    w1, x1, y1, z1 = np.moveaxis(second, -1, 0)
    relative = np.stack(
        (
            w0 * w1 + x0 * x1 + y0 * y1 + z0 * z1,
            w0 * x1 - x0 * w1 - y0 * z1 + z0 * y1,
            w0 * y1 + x0 * z1 - y0 * w1 - z0 * x1,
            w0 * z1 - x0 * y1 + y0 * x1 - z0 * w1,
        ),
        axis=-1,
    )
    relative = np.where(relative[..., :1] < 0.0, -relative, relative)
    vector_norm = np.linalg.norm(relative[..., 1:], axis=-1)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(relative[..., 0], 0.0, None))
    scale = np.divide(angle, vector_norm, out=np.full_like(angle, 2.0), where=vector_norm > 1e-10)
    return relative[..., 1:] * scale[..., None]


def paired_log_bootstrap(
    reference_by_log: Mapping[str, float],
    candidate_by_log: Mapping[str, float],
    *,
    seed: int,
    draws: int,
) -> dict[str, float | int]:
    """Bootstrap the mean paired error reduction with flight logs as units."""
    log_ids = sorted(set(reference_by_log) & set(candidate_by_log))
    if not log_ids or draws < 1:
        raise ValueError("paired bootstrap needs shared logs and positive draws")
    difference = np.array(
        [float(reference_by_log[key]) - float(candidate_by_log[key]) for key in log_ids]
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(log_ids), size=(draws, len(log_ids)))
    bootstrap = np.mean(difference[indices], axis=1)
    return {
        "log_count": len(log_ids),
        "logs_improved": int(np.sum(difference > 0.0)),
        "mean_gain": float(np.mean(difference)),
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
    }


def distribution_shift_summary(
    train_values: np.ndarray, validation_values: np.ndarray
) -> dict[str, float]:
    train = np.asarray(train_values, dtype=float)
    validation = np.asarray(validation_values, dtype=float)
    train = train[np.isfinite(train)]
    validation = validation[np.isfinite(validation)]
    if not len(train) or not len(validation):
        raise ValueError("distribution shift requires finite train and validation values")
    train_std = max(float(np.std(train)), 1.0e-8)
    lower, upper = np.quantile(train, [0.01, 0.99])
    return {
        "train_mean": float(np.mean(train)),
        "train_std": float(np.std(train)),
        "validation_mean": float(np.mean(validation)),
        "validation_std": float(np.std(validation)),
        "standardized_mean_shift": float((np.mean(validation) - np.mean(train)) / train_std),
        "normalized_wasserstein_distance": float(wasserstein_distance(train, validation) / train_std),
        "validation_outside_train_p01_p99_fraction": float(
            np.mean((validation < lower) | (validation > upper))
        ),
        "train_p01": float(lower),
        "train_p99": float(upper),
    }
