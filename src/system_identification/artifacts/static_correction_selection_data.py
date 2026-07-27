"""Stage-separated C3 access to the frozen correction-ready tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd
import pyarrow.dataset as ds

from system_identification.artifacts.io import sha256_file
from system_identification.artifacts.static_correction_data import (
    StaticCorrectionTrainingData,
    load_static_correction_training_data,
)


@dataclass(frozen=True)
class StaticCorrectionValidationData:
    train: StaticCorrectionTrainingData
    validation_cycle_frame: pd.DataFrame
    validation_waveform_frame: pd.DataFrame
    input_hashes: Mapping[str, str]
    validation_labels_loaded: bool = True
    test_labels_loaded: bool = False


def _scan_partition(path: Path, partition: str) -> pd.DataFrame:
    dataset = ds.dataset(path, format="parquet")
    if "partition" not in dataset.schema.names:
        raise ValueError(f"Selection table has no partition column: {path}")
    frame = dataset.to_table(filter=ds.field("partition") == partition).to_pandas()
    if len(frame) == 0 or set(frame["partition"].astype(str).unique()) != {partition}:
        raise ValueError(f"Partition scan returned invalid rows for {partition}: {path}")
    return frame


def load_static_correction_validation_data(
    root: str | Path,
    *,
    authority: Mapping[str, object],
    project_root: str | Path,
    partition: str = "validation",
) -> StaticCorrectionValidationData:
    """Load train for fitting and validation for frozen-finalist evaluation only."""

    if partition != "validation":
        raise ValueError("C3 Stage B only accepts partition='validation'; test is forbidden")
    artifact_root = Path(root).resolve()
    train = load_static_correction_training_data(
        artifact_root,
        authority=authority,
        partition="train",
        project_root=project_root,
    )
    cycle_path = artifact_root / "cycle_table.parquet"
    waveform_path = artifact_root / "waveform_table.parquet"
    before = {str(path): sha256_file(path) for path in (cycle_path, waveform_path)}
    cycle = _scan_partition(cycle_path, "validation")
    waveform = _scan_partition(waveform_path, "validation")
    if cycle["cycle_id"].duplicated().any():
        raise ValueError("Validation cycle table contains duplicate cycle_id")
    if waveform[["cycle_id", "timestamp_us"]].duplicated().any():
        raise ValueError("Validation waveform table contains duplicate stable keys")
    if set(waveform["cycle_id"]) - set(cycle["cycle_id"]):
        raise ValueError("Validation waveform rows have missing cycle entries")
    after = {str(path): sha256_file(path) for path in (cycle_path, waveform_path)}
    if before != after:
        raise ValueError("Correction-ready input changed while Stage B loaded validation")
    return StaticCorrectionValidationData(
        train=train,
        validation_cycle_frame=cycle,
        validation_waveform_frame=waveform,
        input_hashes=before,
    )
