"""Deterministic date-aware grouped folds for C3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from system_identification.training.correction.selection_specs import canonical_hash


@dataclass(frozen=True)
class GroupedFoldManifest:
    schema_version: str
    fold_count: int
    group_column: str
    assignment_rule: str
    random_seed: int
    folds: tuple[Mapping[str, object], ...]
    assignment_hash: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "fold_count": self.fold_count,
            "group_column": self.group_column,
            "assignment_rule": self.assignment_rule,
            "random_seed": self.random_seed,
            "folds": [dict(fold) for fold in self.folds],
            "assignment_hash": self.assignment_hash,
        }


def build_date_aware_grouped_folds(
    cycle_frame: pd.DataFrame,
    *,
    fold_count: int = 5,
    random_seed: int = 20260722,
) -> GroupedFoldManifest:
    required = {"partition", "log_id", "flight_date", "cycle_id"}
    missing = sorted(required - set(cycle_frame.columns))
    if missing:
        raise ValueError(f"Fold input missing columns: {missing}")
    if set(cycle_frame["partition"].astype(str).unique()) != {"train"}:
        raise ValueError("Grouped folds accept train rows only")
    if fold_count < 2:
        raise ValueError("fold_count must be at least two")
    log_table = (
        cycle_frame.groupby(["flight_date", "log_id"], as_index=False, sort=True)
        .agg(cycle_count=("cycle_id", "size"))
        .sort_values(["flight_date", "cycle_count", "log_id"], ascending=[True, False, True], kind="stable")
    )
    fold_cycles = [0] * fold_count
    fold_logs: list[list[dict[str, object]]] = [[] for _ in range(fold_count)]
    for row in log_table.itertuples(index=False):
        target = min(range(fold_count), key=lambda index: (fold_cycles[index], index))
        entry = {
            "log_id": str(row.log_id),
            "flight_date": str(row.flight_date),
            "cycle_count": int(row.cycle_count),
        }
        fold_logs[target].append(entry)
        fold_cycles[target] += int(row.cycle_count)
    folds = tuple(
        {
            "fold_id": index,
            "cycle_count": fold_cycles[index],
            "log_count": len(fold_logs[index]),
            "flight_dates": sorted({str(item["flight_date"]) for item in fold_logs[index]}),
            "logs": sorted(fold_logs[index], key=lambda item: str(item["log_id"])),
            "log_ids": sorted(str(item["log_id"]) for item in fold_logs[index]),
        }
        for index in range(fold_count)
    )
    all_ids = [log_id for fold in folds for log_id in fold["log_ids"]]
    expected_ids = sorted(log_table["log_id"].astype(str).tolist())
    if sorted(all_ids) != expected_ids or len(all_ids) != len(set(all_ids)):
        raise ValueError("Grouped fold construction lost or duplicated a log")
    hash_payload = {
        "assignment_rule": "date_grouped_descending_cycle_count_greedy_min_fold_cycles_tie_fold_id",
        "folds": [{"fold_id": fold["fold_id"], "log_ids": fold["log_ids"]} for fold in folds],
    }
    return GroupedFoldManifest(
        schema_version="static_correction_train_cv_folds_v1",
        fold_count=fold_count,
        group_column="log_id",
        assignment_rule=str(hash_payload["assignment_rule"]),
        random_seed=random_seed,
        folds=folds,
        assignment_hash=canonical_hash(hash_payload),
    )
