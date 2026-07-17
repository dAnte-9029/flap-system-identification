from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from system_identification.analysis.component_attribution import symmetric_finite_difference
from system_identification.analysis.force_discrepancy_attribution import (
    AuditConfig,
    _date_from_log_id,
    decompose_cycle_residuals,
    diagnostic_history_probes,
    equal_cycle_sample_weights,
    estimate_circular_phase_shift,
    half_stroke_attribution,
    harmonic_cycle_summary,
    keyed_align_label_and_prior,
    macro_log_rmse,
    normalize_partitions,
    select_complete_cycles,
    summarize_phase_alignment,
)
from system_identification.physics.baselines.wing_only import (
    WingOnlyBaselineConfig,
    evaluate_wing_only_delaurier_segment,
)
from system_identification.conventions.phase import wrap_to_2pi


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_wrap_to_2pi_supports_declared_scalar_input() -> None:
    assert float(wrap_to_2pi(2.0 * math.pi)) == pytest.approx(0.0)


def _aligned_cycle_frame(*, logs: tuple[str, ...] = ("log_0_2026-4-12-10-00-00",), partitions: tuple[str, ...] = ("train",)) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    timestamp = 1_000_000
    for partition in partitions:
        for log_index, log_id in enumerate(logs):
            for cycle_id in range(2):
                phase = np.linspace(0.0, 2.0 * math.pi, 25, endpoint=True)
                for sample, phi in enumerate(phase):
                    label_fx = 2.0 + 1.5 * math.cos(phi) + 0.2 * math.sin(2.0 * phi)
                    prior_fx = 1.5 + 1.0 * math.cos(phi - 0.2)
                    label_fz = -8.0 + 2.0 * math.sin(phi) + 0.5 * math.cos(2.0 * phi)
                    prior_fz = -7.0 + 1.3 * math.sin(phi - 0.1)
                    rows.append(
                        {
                            "partition": partition,
                            "split": partition,
                            "log_id": log_id,
                            "segment_id": 0,
                            "cycle_id": cycle_id,
                            "cycle_valid": True,
                            "time_s": cycle_id + sample / 24.0 / 4.0,
                            "timestamp_us": timestamp,
                            "phase_corrected_rad": phi,
                            "mechanical_phase_rad": phi % (2.0 * math.pi),
                            "flap_frequency_hz": 4.0 + 0.1 * log_index,
                            "condition_frequency_hz": 4.0 + 0.1 * log_index,
                            "condition_airspeed_m_s": 8.0 + log_index,
                            "condition_alpha_rad": 0.1,
                            "condition_dynamic_pressure_pa": 40.0,
                            "condition_reduced_frequency": 0.04,
                            "label_fx_b": label_fx,
                            "prior_fx_b": prior_fx,
                            "residual_fx_b": label_fx - prior_fx,
                            "label_fz_b": label_fz,
                            "prior_fz_b": prior_fz,
                            "residual_fz_b": label_fz - prior_fz,
                        }
                    )
                    timestamp += 10_000
    return pd.DataFrame(rows)


def _sample_prior_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = pd.DataFrame(
        {
            "log_id": ["a", "a", "b"],
            "timestamp_us": [1, 2, 3],
            "split": ["train"] * 3,
            "fx_b": [1.0, 2.0, 3.0],
            "fz_b": [-1.0, -2.0, -3.0],
        }
    )
    prior = pd.DataFrame(
        {
            "log_id": ["a", "a", "b"],
            "timestamp_us": [1, 2, 3],
            "fx_b": [0.5, 1.0, 1.5],
            "fz_b": [-0.5, -1.0, -1.5],
        }
    )
    return samples, prior


def test_keyed_alignment_does_not_depend_on_row_order() -> None:
    samples, prior = _sample_prior_frames()
    result = keyed_align_label_and_prior(samples, prior.sample(frac=1.0, random_state=4), partition="train")
    assert result.aligned["prior_fx_b"].tolist() == [0.5, 1.0, 1.5]
    assert result.report["row_order_used"] is False


def test_keyed_alignment_duplicate_key_fails() -> None:
    samples, prior = _sample_prior_frames()
    with pytest.raises(ValueError, match="Duplicate alignment keys"):
        keyed_align_label_and_prior(samples, pd.concat([prior, prior.iloc[[0]]]), partition="train")


def test_keyed_alignment_missing_key_is_reported() -> None:
    samples, prior = _sample_prior_frames()
    result = keyed_align_label_and_prior(samples, prior.iloc[:2], partition="train", maximum_missing_fraction=0.34)
    assert result.report["missing_prior_rows"] == 1
    assert result.mismatches["mismatch_type"].tolist() == ["missing_prior"]


def test_keyed_alignment_timestamp_unit_mismatch_fails() -> None:
    samples, prior = _sample_prior_frames()
    prior["timestamp_us"] *= 1_000
    with pytest.raises(ValueError, match="timestamp unit mismatch"):
        keyed_align_label_and_prior(samples, prior, partition="train")


def test_keyed_alignment_orphan_prior_is_reported() -> None:
    samples, prior = _sample_prior_frames()
    orphan = prior.iloc[[0]].copy()
    orphan["timestamp_us"] = 4
    result = keyed_align_label_and_prior(
        samples,
        pd.concat([prior, orphan], ignore_index=True),
        partition="train",
    )
    assert result.report["orphan_prior_rows"] == 1
    assert "orphan_prior" in set(result.mismatches["mismatch_type"])


def test_partition_identity_is_preserved() -> None:
    samples, prior = _sample_prior_frames()
    result = keyed_align_label_and_prior(samples, prior, partition="train")
    assert set(result.aligned["partition"]) == {"train"}


def test_test_partition_cannot_enter_default_audit() -> None:
    with pytest.raises(ValueError, match="must not load the test"):
        normalize_partitions(("train", "test"))


def test_cycle_grouping_never_crosses_log() -> None:
    frame = _aligned_cycle_frame(logs=("a", "b"))
    selection = select_complete_cycles(frame, AuditConfig(minimum_cycle_samples=8))
    assert selection.quality.groupby(["log_id", "cycle_id"]).size().eq(1).all()


def test_cycle_grouping_never_crosses_partition() -> None:
    frame = _aligned_cycle_frame(partitions=("train", "val"))
    selection = select_complete_cycles(frame, AuditConfig(minimum_cycle_samples=8))
    assert selection.quality.groupby(["partition", "cycle_id"]).size().eq(1).all()


def test_duplicate_phase_endpoint_is_removed_and_recorded() -> None:
    selection = select_complete_cycles(_aligned_cycle_frame(), AuditConfig(minimum_cycle_samples=8))
    assert selection.quality["endpoint_action"].eq("removed_last_duplicate").all()
    assert selection.accepted_rows.groupby("cycle_id").size().eq(24).all()
    for _, group in selection.accepted_rows.groupby("cycle_id"):
        assert not np.isclose(group["phase_corrected_rad"].iloc[0], group["phase_corrected_rad"].iloc[-1])


def test_incomplete_cycle_is_rejected_with_reason() -> None:
    frame = _aligned_cycle_frame()
    frame = frame.loc[frame["phase_corrected_rad"] < math.pi]
    selection = select_complete_cycles(frame, AuditConfig(minimum_cycle_samples=8))
    assert not selection.quality["accepted"].any()
    assert selection.quality["rejection_reasons"].str.contains("incomplete_phase_coverage").all()


def test_synthetic_fixed_phase_offset_is_recovered() -> None:
    phase = np.linspace(0.0, 2.0 * math.pi, 144, endpoint=False)
    offset = 0.35
    label = np.sin(phase)
    prior = np.sin(phase - offset)
    estimate = estimate_circular_phase_shift(label, prior, phase_grid=phase)
    assert abs(abs(float(estimate["shift_rad"])) - offset) < 0.03


def test_synthetic_fixed_delay_across_frequencies_is_recovered() -> None:
    tau = 0.012
    rows = []
    for frequency in (3.0, 4.0, 5.0, 6.0):
        rows.append(
            {
                "component": "fx_b",
                "log_id": f"f{frequency}",
                "status": "ok",
                "shift_rad": 2.0 * math.pi * frequency * tau,
                "frequency_hz": frequency,
                "equivalent_delay_s": tau,
                "max_correlation": 1.0,
                "grid_step_rad": 0.01,
            }
        )
    _, summary = summarize_phase_alignment(pd.DataFrame(rows))
    assert summary["fx_b"]["H2_fixed_delay_s"] == pytest.approx(tau, abs=1.0e-8)


def test_constant_signal_does_not_create_phase_shift() -> None:
    constant = np.ones(64)
    estimate = estimate_circular_phase_shift(constant, constant)
    assert estimate["status"] == "low_variance"
    assert math.isnan(float(estimate["shift_rad"]))


def test_circular_boundary_shift_is_recovered() -> None:
    phase = np.linspace(0.0, 2.0 * math.pi, 180, endpoint=False)
    offset = math.pi - 0.08
    estimate = estimate_circular_phase_shift(np.cos(phase), np.cos(phase - offset), phase_grid=phase)
    assert abs(abs(float(estimate["shift_rad"])) - offset) < 0.04


def test_mean_plus_zero_mean_reconstructs_residual() -> None:
    selected = select_complete_cycles(_aligned_cycle_frame(), AuditConfig(minimum_cycle_samples=8)).accepted_rows
    means, samples = decompose_cycle_residuals(selected)
    reconstructed = samples["cycle_mean_residual_fx_b"] + samples["waveform_residual_fx_b"]
    assert np.allclose(reconstructed, samples["residual_fx_b"])
    assert means["decomposition_max_abs_error_fx_b"].max() < 1.0e-12


def test_zero_mean_waveform_has_zero_cycle_mean() -> None:
    selected = select_complete_cycles(_aligned_cycle_frame(), AuditConfig(minimum_cycle_samples=8)).accepted_rows
    _, samples = decompose_cycle_residuals(selected)
    assert samples.groupby("cycle_id")["waveform_residual_fz_b"].mean().abs().max() < 1.0e-12


def test_half_stroke_uses_authoritative_phase_direction() -> None:
    selected = select_complete_cycles(_aligned_cycle_frame(), AuditConfig(minimum_cycle_samples=8)).accepted_rows
    _, samples = decompose_cycle_residuals(selected)
    half, _ = half_stroke_attribution(samples, AuditConfig(minimum_cycle_samples=8))
    assert set(half["half_stroke"]) == {"upstroke", "downstroke"}
    up = half.loc[half["half_stroke"] == "upstroke"]
    assert up["phase_interval"].eq("[3pi/2,2pi) U [0,pi/2)").all()
    assert half["direction_contract_match_fraction"].eq(1.0).all()


def test_harmonic_coefficient_recovers_synthetic_signal() -> None:
    frame = _aligned_cycle_frame()
    frame["residual_fx_b"] = 3.0 * np.cos(2.0 * frame["phase_corrected_rad"] + 0.4)
    frame["label_fx_b"] = frame["prior_fx_b"] + frame["residual_fx_b"]
    selected = select_complete_cycles(frame, AuditConfig(minimum_cycle_samples=8)).accepted_rows
    _, samples = decompose_cycle_residuals(selected)
    harmonics, _ = harmonic_cycle_summary(samples, AuditConfig(phase_bins=144, minimum_cycle_samples=8))
    second = harmonics.loc[(harmonics["component"] == "fx_b") & (harmonics["harmonic_order"] == 2)]
    assert second["amplitude"].mean() == pytest.approx(3.0, rel=0.03)


def test_short_history_probe_preserves_validation_rows_after_sorting() -> None:
    frame = _aligned_cycle_frame(
        logs=("log_0_2026-4-12-10-00-00",),
        partitions=("train", "val"),
    )
    selected = select_complete_cycles(frame, AuditConfig(minimum_cycle_samples=8)).accepted_rows
    _, samples = decompose_cycle_residuals(selected)
    probes = diagnostic_history_probes(
        samples,
        AuditConfig(minimum_cycle_samples=8, harmonic_max_order=2, ridge_alphas=(1.0,), history_lengths=(1,)),
    )
    history = probes.loc[probes["history_samples"] == 1]
    assert history["validation_sample_count"].gt(0).all()
    assert np.isfinite(history["validation_equal_log_macro_rmse"]).all()


def _write_geometry(path: Path) -> None:
    path.write_text(
        "x_mid_m,c_m,dhat\n0.01,0.03,0\n0.03,0.028,0\n0.05,0.025,0\n0.07,0.02,0\n",
        encoding="utf-8",
    )


def _physics_frame() -> pd.DataFrame:
    phase = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
    return pd.DataFrame(
        {
            "log_id": "log",
            "segment_id": 0,
            "time_s": np.arange(len(phase)) * 0.01,
            "timestamp_us": np.arange(len(phase)) * 10_000,
            "mechanical_phase_rad": phase,
            "phase_corrected_rad": phase,
            "flap_frequency_hz": 4.0,
            "vehicle_air_data.rho": 1.2,
            "airspeed_validated.true_airspeed_m_s": 8.0,
            "airspeed_validated.pitch_filtered": 0.0,
            "cycle_id": 0,
            "cycle_valid": True,
            "label_valid": True,
            "fx_b": 0.0,
            "fy_b": 0.0,
            "fz_b": 0.0,
            "mx_b": 0.0,
            "my_b": 0.0,
            "mz_b": 0.0,
        }
    )


def test_component_sum_equals_total_prior(tmp_path: Path) -> None:
    geometry = tmp_path / "wing.csv"
    _write_geometry(geometry)
    output = evaluate_wing_only_delaurier_segment(
        _physics_frame(),
        theta_tip_deg=(10.0,),
        geometry_path=geometry,
        config=WingOnlyBaselineConfig(num_strips=6),
        spanwise_regions={"root": (0.0, 1 / 3), "mid": (1 / 3, 2 / 3), "tip": (2 / 3, 1.0)},
        include_detailed_diagnostics=True,
    )
    for force in ("fx_b", "fz_b"):
        component_sum = sum(output[f"component_{name}_{force}"] for name in ("dN_c", "dN_a", "dT_s", "dD_camber", "dD_f"))
        assert np.allclose(component_sum, output[f"pred_{force}"], atol=1.0e-10)


def test_left_right_and_spanwise_aggregation_conserve_force(tmp_path: Path) -> None:
    geometry = tmp_path / "wing.csv"
    _write_geometry(geometry)
    output = evaluate_wing_only_delaurier_segment(
        _physics_frame(),
        theta_tip_deg=(10.0,),
        geometry_path=geometry,
        config=WingOnlyBaselineConfig(num_strips=6),
        spanwise_regions={"root": (0.0, 1 / 3), "mid": (1 / 3, 2 / 3), "tip": (2 / 3, 1.0)},
        include_detailed_diagnostics=True,
    )
    for name in ("dN_c", "dN_a", "dT_s", "dD_camber", "dD_f"):
        assert np.allclose(
            output[f"component_{name}_left_fx_b"] + output[f"component_{name}_right_fx_b"],
            output[f"component_{name}_fx_b"],
        )
        assert np.allclose(
            sum(output[f"span_{region}_component_{name}_fx_b"] for region in ("root", "mid", "tip")),
            output[f"component_{name}_fx_b"],
        )


def test_symmetric_finite_difference_is_step_stable() -> None:
    x = np.array([-1.0, 0.0, 2.0])
    for step in (1.0e-2, 5.0e-3):
        derivative = symmetric_finite_difference(np.square(x + step), np.square(x - step), step)
        assert np.allclose(derivative, 2.0 * x, atol=1.0e-10)


def test_symmetric_finite_difference_rejects_zero_step() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        symmetric_finite_difference(np.ones(2), np.ones(2), 0.0)


def test_equal_log_macro_not_dominated_by_long_log() -> None:
    frame = pd.DataFrame({"log_id": ["long"] * 100 + ["short"]})
    errors = np.array([1.0] * 100 + [3.0])
    assert macro_log_rmse(frame, errors) == pytest.approx(2.0)


def test_equal_cycle_weights_assign_equal_total_weight_per_log_and_cycle() -> None:
    frame = pd.DataFrame(
        {
            "partition": "train",
            "log_id": ["a"] * 6 + ["b"] * 4,
            "segment_id": 0,
            "cycle_id": [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        }
    )
    frame["weight"] = equal_cycle_sample_weights(frame)
    log_total = frame.groupby("log_id")["weight"].sum()
    assert log_total["a"] == pytest.approx(log_total["b"])
    cycle_total = frame.loc[frame["log_id"] == "a"].groupby("cycle_id")["weight"].sum()
    assert cycle_total.iloc[0] == pytest.approx(cycle_total.iloc[1])


def test_date_identity_is_preserved() -> None:
    assert _date_from_log_id("log_5_2026-4-12-17-51-44") == "2026-04-12"
    assert _date_from_log_id("unknown") == "unknown"


def test_detailed_diagnostics_do_not_change_production_prior_values(tmp_path: Path) -> None:
    geometry = tmp_path / "wing.csv"
    _write_geometry(geometry)
    baseline = evaluate_wing_only_delaurier_segment(
        _physics_frame(), theta_tip_deg=(10.0,), geometry_path=geometry, config=WingOnlyBaselineConfig(num_strips=6)
    )
    detailed = evaluate_wing_only_delaurier_segment(
        _physics_frame(),
        theta_tip_deg=(10.0,),
        geometry_path=geometry,
        config=WingOnlyBaselineConfig(num_strips=6),
        spanwise_regions={"root": (0.0, 1 / 3), "mid": (1 / 3, 2 / 3), "tip": (2 / 3, 1.0)},
        include_detailed_diagnostics=True,
    )
    assert np.array_equal(baseline[["pred_fx_b", "pred_fz_b"]], detailed[["pred_fx_b", "pred_fz_b"]])


def _write_cli_fixture(root: Path) -> tuple[Path, Path]:
    dataset = root / "dataset"
    prior_root = root / "prior"
    dataset.mkdir()
    prior_root.mkdir()
    (dataset / "dataset_manifest.json").write_text(json.dumps({"dataset_id": "synthetic"}), encoding="utf-8")
    (prior_root / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_id": "synthetic_active_prior",
                "lifecycle_status": "active",
                "partitions": ["train", "validation"],
                "test_partition_loaded": False,
                "physics_source": {"commit": "synthetic"},
                "airflow_mode": "legacy_scalar_true_airspeed",
                "contracts": {
                    "frame_contract": "synthetic_body_frame",
                    "airflow_contract": "legacy_scalar_true_airspeed",
                    "phase_contract": "synthetic_phase",
                },
            }
        ),
        encoding="utf-8",
    )
    for partition, log_id in (("train", "log_0_2026-4-12-00-00-00"), ("val", "log_1_2026-4-16-00-00-00")):
        aligned = _aligned_cycle_frame(logs=(log_id,), partitions=(partition,))
        sample = pd.DataFrame(
            {
                "log_id": aligned["log_id"],
                "timestamp_us": aligned["timestamp_us"],
                "split": partition,
                "segment_id": aligned["segment_id"],
                "cycle_id": aligned["cycle_id"],
                "cycle_valid": True,
                "time_s": aligned["time_s"],
                "phase_corrected_rad": aligned["phase_corrected_rad"],
                "mechanical_phase_rad": aligned["mechanical_phase_rad"],
                "flap_frequency_hz": aligned["flap_frequency_hz"],
                "cycle_flap_frequency_hz": aligned["condition_frequency_hz"],
                "airspeed_validated.true_airspeed_m_s": aligned["condition_airspeed_m_s"],
                "vehicle_air_data.rho": 1.2,
                "fx_b": aligned["label_fx_b"],
                "fz_b": aligned["label_fz_b"],
            }
        )
        prior = sample[["log_id", "timestamp_us", "segment_id", "time_s"]].copy()
        prior["fx_b"] = aligned["prior_fx_b"].to_numpy()
        prior["fz_b"] = aligned["prior_fz_b"].to_numpy()
        sample.to_parquet(dataset / f"{partition}_samples.parquet", index=False)
        prior.to_parquet(prior_root / f"{partition}_predictions.parquet", index=False)
    return dataset, prior_root


def test_cli_headless_smoke_manifest_report_and_input_immutability(tmp_path: Path) -> None:
    dataset, prior = _write_cli_fixture(tmp_path)
    output = tmp_path / "outputs"
    report = tmp_path / "report.md"
    input_hash = (dataset / "train_samples.parquet").read_bytes()
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/audit_force_discrepancy_attribution.py"),
        "--dataset-root",
        str(dataset),
        "--prior-root",
        str(prior),
        "--output-root",
        str(output),
        "--partitions",
        "train",
        "validation",
        "--run-id",
        "smoke",
        "--report-path",
        str(report),
        "--no-label-variants",
        "--skip-component-diagnostics",
        "--skip-physical-sensitivity",
        "--phase-bins",
        "24",
        "--minimum-cycle-samples",
        "8",
        "--ridge-alphas",
        "1",
        "--history-lengths",
        "1",
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    manifest = json.loads((output / "smoke/manifest.json").read_text(encoding="utf-8"))
    required = {
        "branch",
        "git_commit",
        "dataset_path",
        "split_identity",
        "used_partitions",
        "used_log_ids",
        "prior_artifact",
        "alignment_keys",
        "cycle_quality_thresholds",
        "phase_grid",
        "random_seed",
        "output_schema_version",
    }
    assert required.issubset(manifest)
    assert manifest["test_rows_loaded"] == 0
    assert report.is_file()
    assert "test partition 未读取" in report.read_text(encoding="utf-8")
    assert (dataset / "train_samples.parquet").read_bytes() == input_hash
    assert len(list((output / "smoke/figures").glob("*.png"))) == 16


def test_cli_completed_strict_failure_returns_two(tmp_path: Path) -> None:
    dataset, prior = _write_cli_fixture(tmp_path)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/audit_force_discrepancy_attribution.py"),
        "--dataset-root",
        str(dataset),
        "--prior-root",
        str(prior),
        "--output-root",
        str(tmp_path / "strict"),
        "--run-id",
        "strict",
        "--report-path",
        str(tmp_path / "strict_report.md"),
        "--no-label-variants",
        "--strict-require-label-variants",
        "--skip-component-diagnostics",
        "--skip-physical-sensitivity",
        "--phase-bins",
        "24",
        "--minimum-cycle-samples",
        "8",
        "--ridge-alphas",
        "1",
        "--history-lengths",
        "1",
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    assert completed.returncode == 2, completed.stderr + completed.stdout
    manifest = json.loads((tmp_path / "strict/strict/manifest.json").read_text(encoding="utf-8"))
    assert manifest["strict_failure_count"] == 1
