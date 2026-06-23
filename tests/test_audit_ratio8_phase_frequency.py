import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_ratio8_phase_frequency import audit_ratio8_phase_frequency


def _write_split(root: Path, split: str, log_id: str) -> None:
    t = np.arange(40, dtype=float) * 0.01
    frequency_hz = 5.0
    phase = np.mod(2.0 * np.pi * frequency_hz * t, 2.0 * np.pi)
    phase_offset = 0.3
    frame = pd.DataFrame(
        {
            "log_id": log_id,
            "time_s": t,
            "encoder_phase_unwrapped_rad": 8.0 * 2.0 * np.pi * frequency_hz * t,
            "wing_phase.phase_rad": np.mod(phase + phase_offset, 2.0 * np.pi),
            "flap_frequency_hz": np.full(len(t), frequency_hz),
            "flap_frequency_hz_source": "encoder_rpm_est_metadata_ratio",
            "flap_frequency_topic_hz": np.full(len(t), frequency_hz * 8.0 / 7.5),
            "encoder_rpm_est": np.full(len(t), frequency_hz * 60.0 * 8.0),
        }
    )
    frame.to_parquet(root / f"{split}_samples.parquet", index=False)


def _write_split_with_legacy_encoder_rpm(root: Path, split: str, log_id: str) -> None:
    t = np.arange(40, dtype=float) * 0.01
    frequency_hz = 5.0
    phase = np.mod(2.0 * np.pi * frequency_hz * t, 2.0 * np.pi)
    frame = pd.DataFrame(
        {
            "log_id": log_id,
            "time_s": t,
            "encoder_phase_unwrapped_rad": 8.0 * 2.0 * np.pi * frequency_hz * t,
            "wing_phase.phase_rad": phase,
            "flap_frequency_hz": np.full(len(t), frequency_hz),
            "flap_frequency_hz_source": "wing_phase.flap_frequency_hz",
            "flap_frequency_topic_hz": np.full(len(t), frequency_hz),
            # Legacy PX4 logs stored this estimator with the old 7.5 ratio semantics.
            "encoder_rpm_est": np.full(len(t), frequency_hz * 60.0 * 7.5),
        }
    )
    frame.to_parquet(root / f"{split}_samples.parquet", index=False)


def test_audit_ratio8_phase_frequency_writes_summary_and_tables(tmp_path: Path):
    split_root = tmp_path / "split"
    output_root = tmp_path / "audit"
    split_root.mkdir()
    _write_split(split_root, "train", "log_train")
    _write_split(split_root, "val", "log_val")
    _write_split(split_root, "test", "log_test")

    summary = audit_ratio8_phase_frequency(split_root=split_root, ratio=8.0, output_root=output_root)

    assert summary["pass"] is True
    assert summary["median_phase_resultant_R"] > 0.99
    assert summary["median_phase_rmse_rad"] < 1.0e-6
    assert summary["median_canonical_to_encoder_frequency_ratio"] == 1.0
    assert (output_root / "phase_offset_by_log.csv").exists()
    assert (output_root / "frequency_consistency.csv").exists()
    written = json.loads((output_root / "summary.json").read_text())
    assert written["ratio"] == 8.0


def test_audit_accepts_wing_phase_frequency_when_encoder_rpm_has_legacy_ratio(tmp_path: Path):
    split_root = tmp_path / "split"
    output_root = tmp_path / "audit"
    split_root.mkdir()
    _write_split_with_legacy_encoder_rpm(split_root, "train", "log_train")
    _write_split_with_legacy_encoder_rpm(split_root, "val", "log_val")
    _write_split_with_legacy_encoder_rpm(split_root, "test", "log_test")

    summary = audit_ratio8_phase_frequency(split_root=split_root, ratio=8.0, output_root=output_root)

    assert summary["pass"] is True
    assert abs(summary["median_canonical_to_wing_frequency_ratio"] - 1.0) < 1.0e-12
    assert abs(summary["median_canonical_to_encoder_frequency_ratio"] - 8.0 / 7.5) < 1.0e-12
    assert summary["encoder_rpm_est_matches_legacy_ratio_7p5"] is True
