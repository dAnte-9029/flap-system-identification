from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_delaurier_residual_kfold_splits import (
    assign_balanced_log_folds,
    build_residual_kfold_splits,
    split_date_coverage,
)


def _write_split(root: Path, prior_root: Path, split: str, logs: list[tuple[str, int]]) -> None:
    rows = []
    priors = []
    for log_id, count in logs:
        for index in range(count):
            value = float(len(rows) + 1)
            rows.append(
                {
                    "dataset_id": "dataset_a",
                    "log_id": log_id,
                    "segment_id": 0,
                    "time_s": float(index) * 0.01,
                    "split": split,
                    "fx_b": value,
                    "fy_b": value + 1.0,
                    "fz_b": value + 2.0,
                    "mx_b": value + 3.0,
                    "my_b": value + 4.0,
                    "mz_b": value + 5.0,
                }
            )
            priors.append(
                {
                    "fx_b": 0.5,
                    "fy_b": 1.5,
                    "fz_b": 2.5,
                    "mx_b": 3.5,
                    "my_b": 4.5,
                    "mz_b": 5.5,
                }
            )
    pd.DataFrame(rows).to_parquet(root / f"{split}_samples.parquet", index=False)
    pd.DataFrame(priors).to_parquet(prior_root / f"{split}_predictions.parquet", index=False)


def test_assign_balanced_log_folds_is_deterministic_and_covers_logs() -> None:
    logs = pd.DataFrame(
        {
            "dataset_id": ["dataset_a"] * 5,
            "log_id": [f"log_{index}" for index in range(5)],
            "valid_sample_count": [100, 90, 80, 70, 60],
        }
    )

    assigned = assign_balanced_log_folds(logs, n_folds=3, seed=7)

    assert sorted(assigned["fold"].unique().tolist()) == [0, 1, 2]
    assert assigned[["dataset_id", "log_id"]].drop_duplicates().shape[0] == 5
    assert assigned.equals(assign_balanced_log_folds(logs, n_folds=3, seed=7))


def test_assign_balanced_log_folds_spreads_each_date_when_possible() -> None:
    logs = pd.DataFrame(
        {
            "dataset_id": ["dataset_a"] * 9,
            "log_id": [
                "log_0_2026-4-12-10-00-00",
                "log_1_2026-4-12-10-10-00",
                "log_2_2026-4-12-10-20-00",
                "log_0_2026-4-13-10-00-00",
                "log_1_2026-4-13-10-10-00",
                "log_2_2026-4-13-10-20-00",
                "log_3_2026-4-13-10-30-00",
                "log_4_2026-4-13-10-40-00",
                "log_5_2026-4-13-10-50-00",
            ],
            "valid_sample_count": [10, 11, 12, 20, 21, 22, 23, 24, 25],
        }
    )

    assigned = assign_balanced_log_folds(logs, n_folds=5, seed=11)

    counts = assigned.groupby("date")["fold"].nunique().to_dict()
    assert counts["2026-4-12"] == 3
    assert counts["2026-4-13"] == 5
    assert assigned["fold"].nunique() == 5


def test_build_residual_kfold_splits_materializes_log_level_residuals(tmp_path: Path) -> None:
    split_root = tmp_path / "split"
    prior_root = tmp_path / "prior"
    output_root = tmp_path / "kfold"
    split_root.mkdir()
    prior_root.mkdir()
    all_logs = [
        ("log_a", 3),
        ("log_b", 4),
        ("log_c", 5),
        ("log_d", 6),
        ("log_e", 7),
        ("log_f", 8),
    ]
    _write_split(split_root, prior_root, "train", all_logs[:2])
    _write_split(split_root, prior_root, "val", all_logs[2:4])
    _write_split(split_root, prior_root, "test", all_logs[4:])
    pd.DataFrame(
        {
            "dataset_id": ["dataset_a"] * len(all_logs),
            "log_id": [log_id for log_id, _ in all_logs],
            "valid_sample_count": [count for _, count in all_logs],
        }
    ).to_csv(split_root / "all_logs.csv", index=False)

    manifest = build_residual_kfold_splits(
        split_root=split_root,
        prior_root=prior_root,
        output_root=output_root,
        n_folds=3,
        seed=3,
        prior_name="unit_prior",
    )

    assert manifest["n_folds"] == 3
    seen_test_logs: set[str] = set()
    for fold_index in range(3):
        fold_root = output_root / f"fold_{fold_index:02d}"
        train = pd.read_parquet(fold_root / "train_samples.parquet")
        val = pd.read_parquet(fold_root / "val_samples.parquet")
        test = pd.read_parquet(fold_root / "test_samples.parquet")
        coverage = pd.read_csv(fold_root / "date_coverage.csv")
        assert set(train["log_id"]).isdisjoint(set(val["log_id"]))
        assert set(train["log_id"]).isdisjoint(set(test["log_id"]))
        assert set(val["log_id"]).isdisjoint(set(test["log_id"]))
        seen_test_logs.update(test["log_id"].unique().tolist())
        assert np.allclose(test["fx_b"], test["label_fx_b"] - test["prior_fx_b"])
        assert (fold_root / "residual_manifest.json").exists()
        assert set(coverage["split"]) == {"train", "val", "test"}

    assert seen_test_logs == {log_id for log_id, _ in all_logs}


def test_split_date_coverage_counts_logs_and_samples() -> None:
    logs = pd.DataFrame(
        {
            "split": ["train", "train", "test"],
            "date": ["2026-4-12", "2026-4-12", "2026-4-13"],
            "dataset_id": ["dataset_a", "dataset_a", "dataset_a"],
            "log_id": ["log_a", "log_b", "log_c"],
            "valid_sample_count": [3, 5, 7],
        }
    )

    coverage = split_date_coverage(logs)

    train_row = coverage.loc[(coverage["split"] == "train") & (coverage["date"] == "2026-4-12")].iloc[0]
    assert train_row["log_count"] == 2
    assert train_row["sample_count"] == 8
