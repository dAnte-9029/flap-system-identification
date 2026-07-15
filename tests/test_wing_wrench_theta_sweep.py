from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import system_identification.analysis.wing_wrench_theta_sweep as sweep


def _canonical_segment(*, log_id: str = "log_a", segment_id: int = 1, samples: int = 1200) -> pd.DataFrame:
    time_s = np.arange(samples, dtype=float) * 0.01
    phase = np.mod(2.0 * np.pi * 5.0 * time_s, 2.0 * np.pi)
    cycle = np.floor(5.0 * time_s).astype(int)
    frame = pd.DataFrame(
        {
            "log_id": log_id,
            "segment_id": segment_id,
            "time_s": time_s,
            "timestamp_us": (time_s * 1.0e6).astype(int),
            "mechanical_phase_rad": phase,
            "phase_valid": True,
            "cycle_id": cycle,
            "cycle_valid": True,
            "flap_frequency_hz": 5.0,
            "airspeed_validated.true_airspeed_m_s": 8.0 + 0.1 * np.sin(0.2 * time_s),
            "airspeed_validated.pitch_filtered": 0.0,
            "vehicle_air_data.rho": 1.15,
            "vehicle_land_detected.landed": False,
            "vehicle_angular_velocity.xyz[1]": 0.02 * np.sin(time_s),
            "vehicle_angular_velocity.xyz[2]": 0.01 * np.cos(time_s),
            "label_valid": True,
        }
    )
    for index, target in enumerate(sweep.TARGETS):
        frame[target] = np.sin(phase + index * 0.1)
    return frame


def _aligned() -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for theta in (0.0, 5.0):
        for sample in range(8):
            phase = 2.0 * np.pi * sample / 8.0
            row: dict[str, float | str | int] = {
                "window_id": "w01",
                "theta_tip_deg": theta,
                "cycle_id": sample // 4,
                "mechanical_phase_rad": phase,
            }
            for target_index, target in enumerate(sweep.TARGETS):
                true = np.sin(phase + 0.1 * target_index)
                row[f"true_{target}"] = true
                row[f"pred_{target}"] = true + 0.01 * theta
            rows.append(row)
    return pd.DataFrame(rows)


def test_metrics_handle_nan_and_constant_signal_correlation() -> None:
    result = sweep._metric_row(
        np.array([1.0, 1.0, np.nan, 1.0]),
        np.array([1.5, 0.5, 4.0, 1.0]),
    )
    assert result["valid_sample_count"] == 3
    assert np.isclose(result["rmse"], np.sqrt((0.25 + 0.25) / 3.0))
    assert np.isclose(result["mae"], 1.0 / 3.0)
    assert np.isclose(result["bias"], 0.0)
    assert np.isnan(result["correlation"])
    assert result["correlation_reason"] == "constant_label"


def test_phase_binning_outputs_empty_bins_with_zero_count() -> None:
    result = sweep.compute_phase_binned_curves(_aligned(), phase_bins=16)
    assert len(result) == 1 * 2 * 6 * 16
    empty = result.loc[result["phase_bin"] == 1]
    assert (empty["label_count"] == 0).all()
    assert empty["label_mean"].isna().all()
    occupied = result.loc[result["phase_bin"] == 0]
    assert (occupied["label_count"] == 1).all()


def test_raw_and_cycle_mean_metrics_do_not_mix_theta_values() -> None:
    aligned = _aligned()
    raw = sweep.compute_window_metrics(aligned)
    cycle_metrics, cycle_means = sweep.compute_cycle_mean_metrics(aligned)
    assert len(raw) == 1 * 2 * 6
    assert len(cycle_metrics) == 1 * 2 * 6
    assert len(cycle_means) == 1 * 2 * 2
    zero_rmse = raw.loc[raw["theta_tip_deg"] == 0.0, "rmse"]
    five_rmse = raw.loc[raw["theta_tip_deg"] == 5.0, "rmse"]
    np.testing.assert_allclose(zero_rmse, 0.0, atol=1.0e-15)
    np.testing.assert_allclose(five_rmse, 0.05, atol=1.0e-15)


def test_window_validation_rejects_gap_and_segment_mismatch() -> None:
    samples = _canonical_segment(samples=600)
    manifest = pd.DataFrame(
        {
            "window_id": ["w01"],
            "log_id": ["log_a"],
            "segment_id": [1],
            "t_start_s": [0.0],
            "t_end_s": [4.0],
            "description": ["stable"],
        }
    )
    sweep.validate_windows(samples, manifest)
    broken = samples.loc[~samples.index.isin(range(150, 300))].copy()
    try:
        sweep.validate_windows(broken, manifest)
    except ValueError as exc:
        assert "gap" in str(exc)
    else:
        raise AssertionError("large gap was not rejected")
    wrong_segment = manifest.assign(segment_id=2)
    try:
        sweep.validate_windows(samples, wrong_segment)
    except ValueError as exc:
        assert "no samples" in str(exc)
    else:
        raise AssertionError("unknown segment was not rejected")


def test_full_segment_is_evaluated_before_window_crop(monkeypatch, tmp_path: Path) -> None:
    samples = _canonical_segment(samples=800)
    manifest = pd.DataFrame(
        {
            "window_id": ["w01"],
            "log_id": ["log_a"],
            "segment_id": [1],
            "t_start_s": [2.0],
            "t_end_s": [6.0],
            "description": ["stable"],
        }
    )
    lengths: list[int] = []

    def fake_evaluate(segment, *, theta_tip_deg, geometry_path, config, phase_acceleration_mode):
        lengths.append(len(segment))
        outputs = []
        for theta in theta_tip_deg:
            result = segment[["log_id", "segment_id", "time_s", "mechanical_phase_rad", "cycle_id"]].copy()
            result["theta_tip_deg"] = theta
            for target in sweep.TARGETS:
                result[f"true_{target}"] = segment[target].to_numpy()
                result[f"pred_{target}"] = segment[target].to_numpy()
            outputs.append(result)
        return pd.concat(outputs, ignore_index=True)

    monkeypatch.setattr(sweep, "evaluate_wing_only_delaurier_segment", fake_evaluate)
    result = sweep.evaluate_selected_windows(
        samples,
        manifest,
        theta_tip_deg=[0.0, 5.0],
        geometry_path=tmp_path / "unused.csv",
    )
    assert lengths == [len(samples)]
    expected_window_rows = len(samples.loc[(samples["time_s"] >= 2.0) & (samples["time_s"] <= 6.0)])
    assert len(result) == 2 * expected_window_rows
    assert set(result["theta_tip_deg"]) == {0.0, 5.0}


def test_deterministic_auto_selection_produces_nonoverlapping_valid_windows() -> None:
    samples = pd.concat(
        [
            _canonical_segment(log_id="log_a", segment_id=1),
            _canonical_segment(log_id="log_b", segment_id=2).assign(
                **{"airspeed_validated.true_airspeed_m_s": 11.0}
            ),
        ],
        ignore_index=True,
    )
    first = sweep.select_representative_windows(samples, window_count=4)
    second = sweep.select_representative_windows(samples, window_count=4)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 4
    assert first["window_id"].is_unique
    assert set(first["selection_mode"]) == {"deterministic_auto"}
