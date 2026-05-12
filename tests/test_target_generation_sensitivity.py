from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.diagnose_target_generation_sensitivity import (
    compute_variant_metrics,
    rewrite_force_labels_for_variant,
)


def _metadata() -> dict:
    return {
        "mass_properties": {
            "mass_kg": {"value": 1.0},
            "inertia_b_kg_m2": {"value": np.eye(3).tolist()},
        },
        "label_definition": {"gravity_m_s2": 9.81},
    }


def _frame() -> pd.DataFrame:
    time_s = np.arange(80, dtype=float) * 0.01
    vy = np.sin(2.0 * np.pi * 1.0 * time_s)
    ay = np.gradient(vy, time_s)
    frame = pd.DataFrame(
        {
            "time_s": time_s,
            "log_id": "log_a",
            "vehicle_local_position.ax": np.zeros(len(time_s)),
            "vehicle_local_position.ay": ay,
            "vehicle_local_position.az": np.full(len(time_s), 9.81),
            "vehicle_local_position.vx": np.zeros(len(time_s)),
            "vehicle_local_position.vy": vy,
            "vehicle_local_position.vz": np.zeros(len(time_s)),
            "vehicle_attitude.q[0]": np.ones(len(time_s)),
            "vehicle_attitude.q[1]": np.zeros(len(time_s)),
            "vehicle_attitude.q[2]": np.zeros(len(time_s)),
            "vehicle_attitude.q[3]": np.zeros(len(time_s)),
            "vehicle_angular_velocity.xyz[0]": np.zeros(len(time_s)),
            "vehicle_angular_velocity.xyz[1]": np.zeros(len(time_s)),
            "vehicle_angular_velocity.xyz[2]": np.zeros(len(time_s)),
            "vehicle_angular_velocity.xyz_derivative[0]": np.zeros(len(time_s)),
            "vehicle_angular_velocity.xyz_derivative[1]": np.zeros(len(time_s)),
            "vehicle_angular_velocity.xyz_derivative[2]": np.zeros(len(time_s)),
            "fy_b": ay,
        }
    )
    return frame


def test_rewrite_force_labels_for_variant_changes_fyb_when_smoothing_window_changes():
    frame = _frame()

    raw = rewrite_force_labels_for_variant(frame, _metadata(), variant_name="raw", window_s=None)
    smooth = rewrite_force_labels_for_variant(frame, _metadata(), variant_name="smooth_0p12", window_s=0.12)

    assert len(raw) == len(frame)
    assert len(smooth) == len(frame)
    assert "fy_b_raw" in raw.columns
    assert "fy_b_smooth_0p12" in smooth.columns
    assert not np.allclose(raw["fy_b_raw"], smooth["fy_b_smooth_0p12"])


def test_compute_variant_metrics_reports_similarity_and_spike_stability():
    frame = _frame()
    variant = rewrite_force_labels_for_variant(frame, _metadata(), variant_name="smooth_0p12", window_s=0.12)

    metrics = compute_variant_metrics(
        frame,
        variant,
        reference_column="fy_b",
        variant_column="fy_b_smooth_0p12",
        split_name="test",
    )

    assert metrics["split"] == "test"
    assert metrics["variant"] == "smooth_0p12"
    assert metrics["sample_count"] == len(frame)
    assert "corr_with_raw" in metrics
    assert "top1pct_overlap_fraction" in metrics
