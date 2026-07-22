from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from system_identification.artifacts.static_correction_bundle import load_static_bundle, save_static_bundle
from system_identification.models.correction.prediction import predict_total
from system_identification.models.correction.specifications import StaticCorrectionSpec
from system_identification.training.correction.fitting import fit_candidate


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    cycles = pd.DataFrame(
        {
            "cycle_id": ["c0", "c1", "c2"],
            "partition": ["train"] * 3,
            "log_id": ["l0", "l1", "l2"],
            "flight_date": ["2026-04-12"] * 3,
            "prior_fx_mean_n": [0.0, 1.0, 2.0],
            "label_fx_mean_n": [1.0, 1.5, 2.0],
            "alpha_mean_std": [-1.0, 0.0, 1.0],
            "flapping_frequency_mean_std": [0.0, 0.0, 0.0],
            "weight_equal_cycle": [1.0] * 3,
            "weight_equal_log": [1.0] * 3,
            "weight_equal_date": [1 / 3] * 3,
        }
    )
    rows: list[dict[str, object]] = []
    for cycle_index, cycle_id in enumerate(cycles["cycle_id"]):
        for i, phase in enumerate(np.arange(8) * 2 * np.pi / 8):
            rows.append(
                {
                    "cycle_id": cycle_id,
                    "partition": "train",
                    "log_id": f"l{cycle_index}",
                    "flight_date": "2026-04-12",
                    "timestamp_us": cycle_index * 100 + i,
                    "prior_fx_n": float(cycle_index) + np.sin(phase),
                    "label_fx_n": 1.0 + 0.5 * cycle_index + 0.5 * np.sin(phase) + np.cos(phase),
                    "prior_fx_mean_n": float(cycle_index),
                    "label_fx_mean_n": 1.0 + 0.5 * cycle_index,
                    "prior_fx_waveform_n": np.sin(phase),
                    "label_fx_waveform_n": 0.5 * np.sin(phase) + np.cos(phase),
                    "alpha_mean_std": float(cycle_index - 1),
                    "flapping_frequency_mean_std": 0.0,
                    "sin_1_phase_centered": np.sin(phase),
                    "cos_1_phase_centered": np.cos(phase),
                    "weight_equal_cycle_sample": 1 / 8,
                    "weight_equal_log_sample": 1 / 8,
                    "weight_equal_date_sample": 1 / 24,
                }
            )
    return cycles, pd.DataFrame(rows)


def _provenance() -> dict[str, object]:
    return {
        "correction_ready_artifact_id": "longitudinal_mean_wb_ratio8_synthetic",
        "correction_ready_manifest_hash": "0" * 64,
        "dataset_id": "canonical_v0.4_training_ready_split_measured_massprops_ratio8_phasefix_v3",
        "dataset_hash": "1" * 64,
        "prior_id": "delaurier_attitude_aware_3b5d4ec_ratio8_phasefix_trainval_v4",
        "prior_hash": "2" * 64,
        "ratio_contract": "ratio8_v1",
        "phase_contract": "hall_indexed_mechanical_phase_ratio8_v1",
        "included_partitions": ["train"],
        "git_commit": "a" * 40,
        "git_dirty": False,
    }


def _bundle(status: str = "candidate"):
    cycle, waveform = _inputs()
    spec = StaticCorrectionSpec(
        model_type="shaped_prior_mean_wb",
        force_component="fx",
        harmonic_order=1,
        condition_set="alpha",
        mean_prior_retention=0.5,
        waveform_prior_retention=0.5,
        ridge_lambda_mean=1e-6,
        ridge_lambda_waveform=1e-6,
        mean_weighting="equal_cycle",
        waveform_weighting="equal_cycle",
        fit_intercept=True,
    )
    normalization = {
        "alpha_mean_rad": {"mean": 0.2, "std": 0.1, "source_partition": "train"},
        "flapping_frequency_mean_hz": {"mean": 4.0, "std": 0.5, "source_partition": "train"},
    }
    return fit_candidate(spec, cycle, waveform, normalization, _provenance(), status=status), cycle, waveform


def test_save_load_round_trip_preserves_prediction_schema_and_normalization(tmp_path: Path) -> None:
    bundle, cycle, waveform = _bundle()
    destination = tmp_path / "bundle"
    save_static_bundle(bundle, destination)
    loaded = load_static_bundle(destination)
    np.testing.assert_allclose(
        predict_total(bundle, cycle, waveform)["prediction_n"],
        predict_total(loaded, cycle, waveform)["prediction_n"],
    )
    assert loaded.mean_solution.feature_names == bundle.mean_solution.feature_names
    assert loaded.waveform_solution.feature_names == bundle.waveform_solution.feature_names
    assert loaded.normalization == bundle.normalization


def test_bundle_files_and_manifest_provenance_are_complete(tmp_path: Path) -> None:
    bundle, _, _ = _bundle(status="smoke_test")
    destination = save_static_bundle(bundle, tmp_path / "bundle")
    required = {
        "bundle_manifest.json",
        "model_spec.json",
        "mean_coefficients.json",
        "waveform_coefficients.json",
        "feature_schema.json",
        "normalization.json",
        "training_provenance.json",
        "fit_diagnostics.json",
    }
    assert required == {path.name for path in destination.iterdir()}
    manifest = json.loads((destination / "bundle_manifest.json").read_text())
    assert manifest["status"] == "smoke_test"
    assert manifest["included_partitions"] == ["train"]
    assert manifest["ratio_contract"] == "ratio8_v1"
    assert manifest["phase_contract"] == "hall_indexed_mechanical_phase_ratio8_v1"
    assert manifest["mean_prior_retention"] == 0.5
    assert manifest["waveform_prior_retention"] == 0.5


def test_bundle_hash_is_deterministic_for_same_inputs() -> None:
    first, _, _ = _bundle()
    second, _, _ = _bundle()
    assert first.bundle_hash == second.bundle_hash


@pytest.mark.parametrize("status", ["selected", "approved", "production", "final"])
def test_forbidden_candidate_status_is_rejected(status: str) -> None:
    with pytest.raises(ValueError, match="status"):
        _bundle(status=status)


def test_bundle_refuses_overwrite_and_detects_tampering(tmp_path: Path) -> None:
    bundle, _, _ = _bundle()
    destination = save_static_bundle(bundle, tmp_path / "bundle")
    with pytest.raises(FileExistsError):
        save_static_bundle(bundle, destination)
    spec_path = destination / "model_spec.json"
    spec_path.write_text(spec_path.read_text().replace('"fx"', '"fz"', 1))
    with pytest.raises(ValueError, match="hash"):
        load_static_bundle(destination)
