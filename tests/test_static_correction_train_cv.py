from __future__ import annotations

import pandas as pd
import pytest

from system_identification.training.correction.grouped_cv import build_date_aware_grouped_folds


def _cycles() -> pd.DataFrame:
    rows = []
    for index in range(20):
        for cycle in range(index % 5 + 1):
            rows.append(
                {
                    "partition": "train",
                    "log_id": f"log_{index:02d}",
                    "flight_date": f"2026-04-{12 + index % 4:02d}",
                    "cycle_id": f"{index}_{cycle}",
                }
            )
    return pd.DataFrame(rows)


def test_log_never_crosses_folds() -> None:
    manifest = build_date_aware_grouped_folds(_cycles())
    ids = [log_id for fold in manifest.folds for log_id in fold["log_ids"]]
    assert len(ids) == len(set(ids)) == 20


def test_validation_rows_are_rejected_from_fold_builder() -> None:
    frame = _cycles()
    frame.loc[0, "partition"] = "validation"
    with pytest.raises(ValueError, match="train rows only"):
        build_date_aware_grouped_folds(frame)


def test_fold_assignment_is_deterministic() -> None:
    first = build_date_aware_grouped_folds(_cycles())
    second = build_date_aware_grouped_folds(_cycles())
    assert first.to_dict() == second.to_dict()


def test_row_shuffle_does_not_change_fold_assignment() -> None:
    original = build_date_aware_grouped_folds(_cycles())
    shuffled = build_date_aware_grouped_folds(_cycles().sample(frac=1.0, random_state=7))
    assert original.assignment_hash == shuffled.assignment_hash
    assert original.folds == shuffled.folds


def test_assignment_hash_changes_when_log_identity_changes() -> None:
    first = build_date_aware_grouped_folds(_cycles())
    frame = _cycles()
    frame.loc[frame["log_id"] == "log_00", "log_id"] = "renamed"
    second = build_date_aware_grouped_folds(frame)
    assert first.assignment_hash != second.assignment_hash


def test_five_folds_cover_all_flight_dates() -> None:
    manifest = build_date_aware_grouped_folds(_cycles())
    assert len(manifest.folds) == 5
    assert sum(int(fold["cycle_count"]) for fold in manifest.folds) == len(_cycles())
