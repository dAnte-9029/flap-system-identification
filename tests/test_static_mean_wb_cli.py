from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.smoke_fit_static_mean_wb_models import main
from system_identification.artifacts.static_correction_data import load_static_correction_training_data
from system_identification.models.correction.specifications import parse_model_family_config
from system_identification.training.correction.smoke import run_static_correction_smoke


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _artifact(tmp_path: Path, **manifest_overrides: object) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "c1"
    root.mkdir()
    cycle_rows: list[dict[str, object]] = []
    waveform_rows: list[dict[str, object]] = []
    for cycle_index in range(6):
        cycle_id = f"c{cycle_index}"
        alpha = (cycle_index - 2.5) / 2.5
        frequency = np.cos(cycle_index)
        prior_fx_mean = 0.25 * cycle_index
        prior_fz_mean = -1.0 - 0.3 * cycle_index
        cycle_rows.append(
            {
                "cycle_id": cycle_id,
                "partition": "train",
                "log_id": f"l{cycle_index % 2}",
                "flight_date": "2026-04-12",
                "alpha_mean_std": alpha,
                "flapping_frequency_mean_std": frequency,
                "prior_fx_mean_n": prior_fx_mean,
                "label_fx_mean_n": 0.5 * prior_fx_mean + 1.0 + 0.2 * alpha,
                "prior_fz_mean_n": prior_fz_mean,
                "label_fz_mean_n": prior_fz_mean + 2.0 - 0.1 * frequency,
                "weight_equal_cycle": 1.0,
                "weight_equal_log": 1 / 3,
                "weight_equal_date": 1 / 6,
            }
        )
        for sample_index, phase in enumerate(np.arange(16) * 2 * np.pi / 16):
            fx_prior_wave = np.sin(phase)
            fz_prior_wave = -2.0 * np.cos(phase)
            row: dict[str, object] = {
                "cycle_id": cycle_id,
                "partition": "train",
                "log_id": f"l{cycle_index % 2}",
                "flight_date": "2026-04-12",
                "timestamp_us": cycle_index * 1000 + sample_index,
                "alpha_mean_std": alpha,
                "flapping_frequency_mean_std": frequency,
                "prior_fx_mean_n": prior_fx_mean,
                "label_fx_mean_n": cycle_rows[-1]["label_fx_mean_n"],
                "prior_fx_waveform_n": fx_prior_wave,
                "label_fx_waveform_n": 0.5 * fx_prior_wave + 0.3 * np.cos(phase),
                "prior_fx_n": prior_fx_mean + fx_prior_wave,
                "label_fx_n": cycle_rows[-1]["label_fx_mean_n"] + 0.5 * fx_prior_wave + 0.3 * np.cos(phase),
                "prior_fz_mean_n": prior_fz_mean,
                "label_fz_mean_n": cycle_rows[-1]["label_fz_mean_n"],
                "prior_fz_waveform_n": fz_prior_wave,
                "label_fz_waveform_n": 0.8 * fz_prior_wave + 0.4 * np.sin(phase),
                "prior_fz_n": prior_fz_mean + fz_prior_wave,
                "label_fz_n": cycle_rows[-1]["label_fz_mean_n"] + 0.8 * fz_prior_wave + 0.4 * np.sin(phase),
                "weight_equal_cycle_sample": 1 / 16,
                "weight_equal_log_sample": 1 / 48,
                "weight_equal_date_sample": 1 / 96,
            }
            for harmonic in range(1, 5):
                row[f"sin_{harmonic}_phase_centered"] = np.sin(harmonic * phase)
                row[f"cos_{harmonic}_phase_centered"] = np.cos(harmonic * phase)
            waveform_rows.append(row)
    pd.DataFrame(cycle_rows).to_parquet(root / "cycle_table.parquet", index=False)
    pd.DataFrame(waveform_rows).to_parquet(root / "waveform_table.parquet", index=False)
    manifest: dict[str, object] = {
        "artifact_id": "synthetic_ratio8_c1",
        "schema_version": "longitudinal_correction_ready_v1",
        "dataset_id": "dataset_v4",
        "dataset_manifest_hash": "1" * 64,
        "resolved_prior_id": "prior_v4",
        "prior_artifact_hash": "2" * 64,
        "ratio_contract_version": "ratio8_v1",
        "phase_contract_version": "phase_ratio8_v1",
        "frequency_contract_version": "frequency_ratio8_v1",
        "wing_transmission_ratio": 8.0,
        "prior_lifecycle_status": "active",
        "git_dirty": False,
        "test_labels_loaded": False,
        "included_partitions": ["train"],
        "excluded_partitions": ["test"],
    }
    manifest.update(manifest_overrides)
    _json(root / "manifest.json", manifest)
    _json(
        root / "normalization.json",
        {
            "alpha_mean_rad": {"mean": 0.0, "std": 1.0, "source_partition": "train"},
            "flapping_frequency_mean_hz": {"mean": 4.0, "std": 1.0, "source_partition": "train"},
        },
    )
    _json(
        root / "quality_checks.json",
        {"strict_failures": [], "checks": {"test_label_not_loaded": {"passed": True}}},
    )
    manifest_hash = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
    config: dict[str, object] = {
        "schema_version": "static_mean_wb_family_v1",
        "force_components": ["fx", "fz"],
        "model_types": [
            "raw_prior",
            "gain_bias",
            "physical_component_scale",
            "fixed_prior_mean_wb",
            "shaped_prior_mean_wb",
            "no_prior_mean_wb",
        ],
        "harmonic_orders": [1, 2, 3, 4],
        "condition_sets": ["none", "alpha", "frequency", "alpha_frequency"],
        "prior_retention_values": [0.0, 0.25, 0.5, 0.75, 1.0],
        "ridge_values_for_future_c3": [1e-6, 1e-4, 1e-2, 1.0, 100.0],
        "allowed_fit_partitions": ["train"],
        "forbidden_features": ["airspeed", "dynamic_pressure", "history", "future_state"],
        "authority": {
            "correction_ready_manifest_sha256": manifest_hash,
            "dataset_id": "dataset_v4",
            "dataset_manifest_sha256": "1" * 64,
            "prior_id": "prior_v4",
            "prior_manifest_sha256": "2" * 64,
            "ratio_contract_version": "ratio8_v1",
            "phase_contract_version": "phase_ratio8_v1",
            "frequency_contract_version": "frequency_ratio8_v1",
        },
        "smoke_defaults": {
            "harmonic_order": 2,
            "condition_set": "alpha_frequency",
            "mean_prior_retention": 0.5,
            "waveform_prior_retention": 0.5,
            "ridge_lambda_mean": 1e-4,
            "ridge_lambda_waveform": 1e-4,
            "mean_weighting": "equal_log",
            "waveform_weighting": "equal_log",
        },
    }
    return root, config


def test_train_only_headless_smoke_writes_complete_immutable_outputs(tmp_path: Path) -> None:
    root, config_value = _artifact(tmp_path)
    config = parse_model_family_config(config_value)
    data = load_static_correction_training_data(root, authority=config.authority)
    before = {path: path.stat().st_mtime_ns for path in root.iterdir()}
    summary = run_static_correction_smoke(data, config, tmp_path / "smoke", project_root=Path(__file__).parents[1])
    after = {path: path.stat().st_mtime_ns for path in root.iterdir()}
    assert before == after
    assert summary["candidate_count"] == 10
    assert summary["selection_performed"] is False
    assert summary["validation_labels_loaded"] is False
    assert summary["test_labels_loaded"] is False
    assert summary["physical_component_availability"]["physical_component_scale_fz"] == "unavailable"
    assert len({row["train_rmse_n"] for row in summary["candidate_metrics"]}) > 1
    for row in summary["candidate_metrics"]:
        bundle_root = Path(row["bundle_path"])
        assert (bundle_root / "bundle_manifest.json").is_file()
        assert (bundle_root / "fit_diagnostics.json").is_file()


@pytest.mark.parametrize("partition", ["validation", "test", "train validation"])
def test_cli_rejects_non_train_partition_before_reading_inputs(partition: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="train-only"):
        main(
            [
                "--correction-ready-root",
                str(tmp_path / "missing"),
                "--config",
                str(tmp_path / "missing.yaml"),
                "--partition",
                partition,
                "--output-root",
                str(tmp_path / "output"),
            ]
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"wing_transmission_ratio": 7.5}, "ratio 8.0"),
        ({"test_labels_loaded": True}, "test labels"),
        ({"included_partitions": ["train", "test"]}, "includes test"),
        ({"git_dirty": True}, "Dirty"),
    ],
)
def test_artifact_provenance_failures_are_closed(
    tmp_path: Path, override: dict[str, object], message: str
) -> None:
    root, config_value = _artifact(tmp_path, **override)
    config = parse_model_family_config(config_value)
    with pytest.raises(ValueError, match=message):
        load_static_correction_training_data(root, authority=config.authority)
