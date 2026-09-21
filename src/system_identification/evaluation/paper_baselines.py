"""Shared short-horizon metrics with flights, not windows, as reporting units."""
from __future__ import annotations

import numpy as np
import pandas as pd

from system_identification.evaluation.trajectory import evaluate_trajectory_predictions

HORIZONS = {0.1: 5, 0.2: 10, 0.5: 25, 1.0: 50}
METRIC_MAP = {
    "position_error_m": "position_rmse_m",
    "velocity_error_m_s": "velocity_rmse_m_s",
    "attitude_error_deg": "attitude_error_deg",
    "body_rate_error_rad_s": "body_rate_rmse_rad_s",
    "delta_v_error_m_s": "delta_v_rmse_m_s",
    "delta_omega_error_rad_s": "delta_omega_rmse_rad_s",
    **{f"velocity_{a}_error_m_s": f"velocity_{a}_rmse_m_s" for a in "xyz"},
    **{f"body_rate_{a}_error_rad_s": f"body_rate_{a}_rmse_rad_s" for a in "pqr"},
}


def cohort(log_id):
    if log_id.startswith("2026.9.7/"):
        return "Sep7"
    if log_id.startswith("9.17数据/"):
        return "Sep17"
    raise ValueError(f"flight is outside the declared open validation cohorts: {log_id}")


def endpoint_metrics(prediction, batch, *, model, seed=17):
    """Endpoint vector RMSE inputs; attitude is geodesic degrees, sign invariant."""
    truth = batch.truth
    if (not np.isfinite(batch.dt_s).all() or np.any(batch.dt_s <= 0)
            or np.any(batch.dt_s > .05)):
        raise ValueError("invalid native timestamps")
    for key, values in vars(prediction).items():
        expected = getattr(truth, key)
        if values.shape != expected.shape or not np.isfinite(values).all():
            raise ValueError(f"invalid prediction shape/values: {model}/{key}")
        if not np.allclose(values[:, 0], expected[:, 0], rtol=1e-6, atol=2e-5):
            raise ValueError(f"initial state mismatch: {model}/{key}")
    norm_error = np.max(np.abs(np.linalg.norm(prediction.quaternion_nb, axis=-1) - 1))
    if norm_error > 1e-5:
        raise ValueError(f"non-unit prediction quaternion: {norm_error}")
    frame = evaluate_trajectory_predictions(
        prediction, truth, model_name=model, split="validation",
        window_ids=batch.window_ids, log_ids=batch.log_ids,
        segment_ids=batch.segment_ids, horizon_steps=HORIZONS, dt_s=batch.dt_s,
    )
    frame["seed"] = seed
    frame["cohort"] = frame.log_id.map(cohort)
    for horizon, step in HORIZONS.items():
        selected = frame.horizon_s == horizon
        if not np.allclose(frame.loc[selected, "observed_horizon_s"],
                           batch.dt_s[:, :step].sum(axis=1), atol=1e-12, rtol=0):
            raise ValueError("native timestamps do not match integrated horizons")
        for field, stem, axes, unit in [
            ("velocity_n", "velocity", "xyz", "m_s"),
            ("angular_velocity_b", "body_rate", "pqr", "rad_s"),
        ]:
            p, t = getattr(prediction, field), getattr(truth, field)
            error = p[:, step] - t[:, step]
            # Both increments start at the SAME observed t0, as in the protocol.
            delta_error = (p[:, step] - t[:, 0]) - (t[:, step] - t[:, 0])
            delta_name = "delta_v_error_m_s" if stem == "velocity" else "delta_omega_error_rad_s"
            frame.loc[selected, delta_name] = np.linalg.norm(delta_error, axis=1)
            if not np.allclose(delta_error, error, atol=1e-12, rtol=1e-12):
                raise ValueError("increment/endpoint identity failed")
            for j, axis in enumerate(axes):
                frame.loc[selected, f"{stem}_{axis}_error_{unit}"] = np.abs(error[:, j])
    if not np.isfinite(frame[list(METRIC_MAP)].to_numpy()).all():
        raise ValueError("nonfinite metric")
    return frame


def aggregate_flights(rows):
    """Per-flight RMS, then equal-flight means and sample SD (ddof=1).

    Single-seed pilot only: deliberately no seed SD or window-based SE/CI.
    """
    keys = ["model", "seed", "cohort", "log_id", "horizon_s"]
    per_flight = rows.groupby(keys, sort=True)[list(METRIC_MAP)].agg(
        lambda x: float(np.sqrt(np.mean(np.square(x.to_numpy()))))
    ).rename(columns=METRIC_MAP).reset_index()
    counts = rows.groupby(keys).size().rename("n_windows").reset_index()
    per_flight = per_flight.merge(counts, on=keys, validate="one_to_one")
    expanded = pd.concat([per_flight, per_flight.assign(cohort="ALL")], ignore_index=True)
    groups = expanded.groupby(["model", "seed", "cohort", "horizon_s"], sort=True)
    metrics = list(METRIC_MAP.values())
    summary = groups[metrics].mean().reset_index()
    sizes = groups.agg(n_flights=("log_id", "nunique"), n_windows=("n_windows", "sum")).reset_index()
    summary = summary.merge(sizes, validate="one_to_one")
    aggregate = groups[metrics].agg(["mean", "std"])
    aggregate.columns = [f"{metric}_{stat}_across_flights" for metric, stat in aggregate.columns]
    aggregate = aggregate.reset_index().merge(sizes, validate="one_to_one")
    return per_flight, summary, aggregate
