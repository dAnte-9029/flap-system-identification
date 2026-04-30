from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


VALID_ROW_COLUMNS = ("label_valid", "cycle_valid", "flap_active", "cycle_id")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _valid_row_mask(samples: pd.DataFrame) -> pd.Series:
    _require_columns(samples, VALID_ROW_COLUMNS)
    mask = (
        samples["label_valid"].astype(bool)
        & samples["cycle_valid"].astype(bool)
        & samples["flap_active"].astype(bool)
        & samples["cycle_id"].notna()
        & (samples["cycle_id"].astype(int) >= 0)
    )

    landed_column = "vehicle_land_detected.landed"
    if landed_column in samples.columns:
        landed = pd.to_numeric(samples[landed_column], errors="coerce")
        mask &= landed.fillna(1.0) < 0.5

    return mask


def _empty_blocks_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "block_id",
            "dataset_id",
            "log_id",
            "source_samples_path",
            "cycle_start",
            "cycle_end",
            "cycle_count",
            "sample_count",
        ]
    )


def extract_cycle_blocks(
    samples: pd.DataFrame,
    *,
    dataset_id: str,
    log_id: str,
    source_samples_path: str,
    block_size_cycles: int = 60,
) -> pd.DataFrame:
    if block_size_cycles <= 0:
        raise ValueError("block_size_cycles must be positive")

    valid_samples = samples.loc[_valid_row_mask(samples)].copy()
    if valid_samples.empty:
        return _empty_blocks_frame()

    valid_samples["cycle_id"] = valid_samples["cycle_id"].astype(int)
    cycle_ids = np.sort(valid_samples["cycle_id"].unique())
    if len(cycle_ids) == 0:
        return _empty_blocks_frame()

    rows: list[dict[str, object]] = []
    run_start = 0
    block_index = 0

    for idx in range(1, len(cycle_ids) + 1):
        is_run_end = idx == len(cycle_ids) or cycle_ids[idx] != cycle_ids[idx - 1] + 1
        if not is_run_end:
            continue

        run_cycle_ids = cycle_ids[run_start:idx]
        for chunk_start in range(0, len(run_cycle_ids), block_size_cycles):
            chunk_cycle_ids = run_cycle_ids[chunk_start : chunk_start + block_size_cycles]
            cycle_start = int(chunk_cycle_ids[0])
            cycle_end = int(chunk_cycle_ids[-1])
            sample_count = int(valid_samples["cycle_id"].between(cycle_start, cycle_end).sum())
            rows.append(
                {
                    "block_id": f"{dataset_id}:{log_id}:block_{block_index:04d}",
                    "dataset_id": dataset_id,
                    "log_id": log_id,
                    "source_samples_path": str(source_samples_path),
                    "cycle_start": cycle_start,
                    "cycle_end": cycle_end,
                    "cycle_count": int(len(chunk_cycle_ids)),
                    "sample_count": sample_count,
                }
            )
            block_index += 1
        run_start = idx

    return pd.DataFrame(rows)


def _split_block_counts(
    total_blocks: int,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, int]:
    ratios = {"train": float(train_ratio), "val": float(val_ratio), "test": float(test_ratio)}
    if any(value < 0.0 for value in ratios.values()):
        raise ValueError("split ratios must be non-negative")
    ratio_sum = sum(ratios.values())
    if ratio_sum <= 0.0:
        raise ValueError("at least one split ratio must be positive")

    normalized = {name: value / ratio_sum for name, value in ratios.items()}
    raw = {name: normalized[name] * total_blocks for name in normalized}
    counts = {name: int(np.floor(raw[name])) for name in raw}

    remainder = total_blocks - sum(counts.values())
    order = sorted(
        normalized.keys(),
        key=lambda name: (raw[name] - counts[name], normalized[name]),
        reverse=True,
    )
    for idx in range(remainder):
        counts[order[idx % len(order)]] += 1

    positive_splits = [name for name, value in normalized.items() if value > 0.0]
    if total_blocks >= len(positive_splits):
        for split_name in positive_splits:
            if counts[split_name] > 0:
                continue
            donor = max(
                (name for name in counts if counts[name] > 1),
                key=lambda name: (counts[name], normalized[name]),
                default=None,
            )
            if donor is None:
                break
            counts[donor] -= 1
            counts[split_name] += 1

    return counts


def assign_cycle_block_splits(
    blocks: pd.DataFrame,
    *,
    seed: int = 0,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> pd.DataFrame:
    if blocks.empty:
        return blocks.assign(split=pd.Series(dtype=object))

    assigned = blocks.reset_index(drop=True).copy()
    counts = _split_block_counts(
        len(assigned),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    permutation = np.random.default_rng(seed).permutation(len(assigned))
    split_labels = np.empty(len(assigned), dtype=object)

    cursor = 0
    for split_name in ("train", "val", "test"):
        next_cursor = cursor + counts[split_name]
        split_labels[permutation[cursor:next_cursor]] = split_name
        cursor = next_cursor

    assigned["split"] = split_labels
    return assigned


def assign_log_splits(
    logs: pd.DataFrame,
    *,
    seed: int = 0,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> pd.DataFrame:
    if logs.empty:
        return logs.assign(split=pd.Series(dtype=object))

    assigned = logs.reset_index(drop=True).copy()
    counts = _split_block_counts(
        len(assigned),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    permutation = np.random.default_rng(seed).permutation(len(assigned))
    split_labels = np.empty(len(assigned), dtype=object)

    cursor = 0
    for split_name in ("train", "val", "test"):
        next_cursor = cursor + counts[split_name]
        split_labels[permutation[cursor:next_cursor]] = split_name
        cursor = next_cursor

    assigned["split"] = split_labels
    return assigned


def build_train_purge_intervals(assigned_blocks: pd.DataFrame, *, purge_cycles: int = 8) -> pd.DataFrame:
    if assigned_blocks.empty:
        return pd.DataFrame(columns=["dataset_id", "log_id", "cycle_start", "cycle_end"])

    if purge_cycles < 0:
        raise ValueError("purge_cycles must be non-negative")

    protected = assigned_blocks.loc[assigned_blocks["split"].isin(["val", "test"])].copy()
    if protected.empty:
        return pd.DataFrame(columns=["dataset_id", "log_id", "cycle_start", "cycle_end"])

    protected["cycle_start"] = (protected["cycle_start"].astype(int) - purge_cycles).clip(lower=0)
    protected["cycle_end"] = protected["cycle_end"].astype(int) + purge_cycles
    protected = protected.sort_values(["dataset_id", "log_id", "cycle_start", "cycle_end"]).reset_index(drop=True)

    merged_rows: list[dict[str, object]] = []

    for (dataset_id, log_id), group in protected.groupby(["dataset_id", "log_id"], sort=False):
        current_start: int | None = None
        current_end: int | None = None

        for row in group.itertuples(index=False):
            start = int(row.cycle_start)
            end = int(row.cycle_end)
            if current_start is None:
                current_start = start
                current_end = end
                continue
            if start <= current_end + 1:
                current_end = max(current_end, end)
                continue
            merged_rows.append(
                {
                    "dataset_id": dataset_id,
                    "log_id": log_id,
                    "cycle_start": current_start,
                    "cycle_end": current_end,
                }
            )
            current_start = start
            current_end = end

        if current_start is not None and current_end is not None:
            merged_rows.append(
                {
                    "dataset_id": dataset_id,
                    "log_id": log_id,
                    "cycle_start": current_start,
                    "cycle_end": current_end,
                }
            )

    return pd.DataFrame(merged_rows)


def _load_log_records_from_manifest(manifest_path: str | Path) -> list[dict[str, str]]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    dataset_id = manifest["dataset_id"]
    accepted_logs_path = Path(manifest["accepted_logs_json"])
    accepted_logs = json.loads(accepted_logs_path.read_text(encoding="utf-8"))

    records: list[dict[str, str]] = []
    for item in accepted_logs:
        records.append(
            {
                "dataset_id": dataset_id,
                "manifest_path": str(manifest_path),
                "log_id": item["log_id"],
                "source_samples_path": item["samples_path"],
            }
        )
    return records


def _cycle_to_block_map(blocks_for_split: pd.DataFrame) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for row in blocks_for_split.itertuples(index=False):
        for cycle_id in range(int(row.cycle_start), int(row.cycle_end) + 1):
            mapping[cycle_id] = str(row.block_id)
    return mapping


def _apply_purge_mask(frame: pd.DataFrame, purge_intervals: pd.DataFrame) -> pd.Series:
    if frame.empty or purge_intervals.empty:
        return pd.Series(True, index=frame.index)

    keep = pd.Series(True, index=frame.index)
    cycle_ids = frame["cycle_id"].astype(int)
    for row in purge_intervals.itertuples(index=False):
        keep &= ~cycle_ids.between(int(row.cycle_start), int(row.cycle_end))
    return keep


def _split_rows_for_log(
    samples: pd.DataFrame,
    *,
    dataset_id: str,
    log_id: str,
    source_samples_path: str,
    assigned_blocks_for_log: pd.DataFrame,
    purge_intervals_for_log: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    valid_samples = samples.loc[_valid_row_mask(samples)].copy()
    blocks_for_split = assigned_blocks_for_log.loc[assigned_blocks_for_log["split"] == split_name].copy()

    if valid_samples.empty or blocks_for_split.empty:
        return pd.DataFrame(columns=list(samples.columns) + ["dataset_id", "log_id", "source_samples_path", "block_id", "split"])

    cycle_to_block = _cycle_to_block_map(blocks_for_split)
    selected = valid_samples["cycle_id"].astype(int).map(cycle_to_block)
    output = valid_samples.loc[selected.notna()].copy()
    output["block_id"] = selected[selected.notna()].to_numpy()

    if split_name == "train":
        keep_mask = _apply_purge_mask(output, purge_intervals_for_log)
        output = output.loc[keep_mask].copy()

    output["dataset_id"] = dataset_id
    output["log_id"] = log_id
    output["source_samples_path"] = str(source_samples_path)
    output["split"] = split_name
    return output


def materialize_cycle_block_split(
    *,
    manifest_paths: list[str | Path],
    output_root: str | Path,
    block_size_cycles: int = 60,
    purge_cycles: int = 8,
    seed: int = 0,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, object]:
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_records: list[dict[str, str]] = []
    for manifest_path in manifest_paths:
        log_records.extend(_load_log_records_from_manifest(manifest_path))

    all_blocks: list[pd.DataFrame] = []
    for record in log_records:
        block_columns = ["cycle_id", "label_valid", "cycle_valid", "flap_active"]
        samples = pd.read_parquet(record["source_samples_path"], columns=block_columns)
        blocks = extract_cycle_blocks(
            samples,
            dataset_id=record["dataset_id"],
            log_id=record["log_id"],
            source_samples_path=record["source_samples_path"],
            block_size_cycles=block_size_cycles,
        )
        if not blocks.empty:
            all_blocks.append(blocks)

    if not all_blocks:
        raise ValueError("No usable cycle blocks found in the provided manifests")

    all_blocks_df = pd.concat(all_blocks, ignore_index=True)
    assigned_blocks = assign_cycle_block_splits(
        all_blocks_df,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    purge_intervals = build_train_purge_intervals(assigned_blocks, purge_cycles=purge_cycles)

    split_frames: dict[str, list[pd.DataFrame]] = {"train": [], "val": [], "test": []}

    for record in log_records:
        dataset_id = record["dataset_id"]
        log_id = record["log_id"]
        source_samples_path = record["source_samples_path"]
        samples = pd.read_parquet(source_samples_path)

        assigned_for_log = assigned_blocks.loc[
            (assigned_blocks["dataset_id"] == dataset_id) & (assigned_blocks["log_id"] == log_id)
        ].copy()
        purge_for_log = purge_intervals.loc[
            (purge_intervals["dataset_id"] == dataset_id) & (purge_intervals["log_id"] == log_id)
        ].copy()

        for split_name in ("train", "val", "test"):
            split_rows = _split_rows_for_log(
                samples,
                dataset_id=dataset_id,
                log_id=log_id,
                source_samples_path=source_samples_path,
                assigned_blocks_for_log=assigned_for_log,
                purge_intervals_for_log=purge_for_log,
                split_name=split_name,
            )
            if not split_rows.empty:
                split_frames[split_name].append(split_rows)

    split_tables: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "val", "test"):
        if split_frames[split_name]:
            split_tables[split_name] = pd.concat(split_frames[split_name], ignore_index=True)
        else:
            split_tables[split_name] = pd.DataFrame()

    all_blocks_path = output_dir / "all_blocks.csv"
    train_blocks_path = output_dir / "train_blocks.csv"
    val_blocks_path = output_dir / "val_blocks.csv"
    test_blocks_path = output_dir / "test_blocks.csv"
    train_samples_path = output_dir / "train_samples.parquet"
    val_samples_path = output_dir / "val_samples.parquet"
    test_samples_path = output_dir / "test_samples.parquet"
    dataset_manifest_path = output_dir / "dataset_manifest.json"

    assigned_blocks.sort_values(["dataset_id", "log_id", "cycle_start"]).to_csv(all_blocks_path, index=False)
    assigned_blocks.loc[assigned_blocks["split"] == "train"].to_csv(train_blocks_path, index=False)
    assigned_blocks.loc[assigned_blocks["split"] == "val"].to_csv(val_blocks_path, index=False)
    assigned_blocks.loc[assigned_blocks["split"] == "test"].to_csv(test_blocks_path, index=False)

    split_tables["train"].to_parquet(train_samples_path, index=False)
    split_tables["val"].to_parquet(val_samples_path, index=False)
    split_tables["test"].to_parquet(test_samples_path, index=False)

    manifest = {
        "dataset_id": output_dir.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_dir),
        "source_manifest_paths": [str(Path(path)) for path in manifest_paths],
        "source_dataset_ids": sorted({record["dataset_id"] for record in log_records}),
        "input_log_count": len(log_records),
        "input_block_count": int(len(assigned_blocks)),
        "block_size_cycles": int(block_size_cycles),
        "purge_cycles": int(purge_cycles),
        "seed": int(seed),
        "split_ratios": {
            "train": float(train_ratio),
            "val": float(val_ratio),
            "test": float(test_ratio),
        },
        "split_block_counts": {
            split_name: int((assigned_blocks["split"] == split_name).sum())
            for split_name in ("train", "val", "test")
        },
        "split_sample_counts": {
            split_name: int(len(split_tables[split_name]))
            for split_name in ("train", "val", "test")
        },
        "split_cycle_counts": {
            split_name: int(split_tables[split_name]["cycle_id"].nunique()) if not split_tables[split_name].empty else 0
            for split_name in ("train", "val", "test")
        },
        "all_blocks_csv": str(all_blocks_path),
        "train_blocks_csv": str(train_blocks_path),
        "val_blocks_csv": str(val_blocks_path),
        "test_blocks_csv": str(test_blocks_path),
        "train_samples_parquet": str(train_samples_path),
        "val_samples_parquet": str(val_samples_path),
        "test_samples_parquet": str(test_samples_path),
    }
    dataset_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "dataset_manifest_path": str(dataset_manifest_path),
        "all_blocks_path": str(all_blocks_path),
        "train_blocks_path": str(train_blocks_path),
        "val_blocks_path": str(val_blocks_path),
        "test_blocks_path": str(test_blocks_path),
        "train_samples_path": str(train_samples_path),
        "val_samples_path": str(val_samples_path),
        "test_samples_path": str(test_samples_path),
    }


def materialize_log_split(
    *,
    manifest_paths: list[str | Path],
    output_root: str | Path,
    seed: int = 0,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, object]:
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_records: list[dict[str, str]] = []
    for manifest_path in manifest_paths:
        log_records.extend(_load_log_records_from_manifest(manifest_path))

    if not log_records:
        raise ValueError("No accepted logs found in the provided manifests")

    logs = pd.DataFrame(log_records)
    assigned_logs = assign_log_splits(
        logs,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    split_frames: dict[str, list[pd.DataFrame]] = {"train": [], "val": [], "test": []}
    valid_sample_counts: list[int] = []

    for row in assigned_logs.itertuples(index=False):
        samples = pd.read_parquet(row.source_samples_path)
        valid_samples = samples.loc[_valid_row_mask(samples)].copy()
        valid_sample_counts.append(int(len(valid_samples)))
        if valid_samples.empty:
            continue
        valid_samples["dataset_id"] = str(row.dataset_id)
        valid_samples["log_id"] = str(row.log_id)
        valid_samples["source_samples_path"] = str(row.source_samples_path)
        valid_samples["split"] = str(row.split)
        split_frames[str(row.split)].append(valid_samples)

    assigned_logs["valid_sample_count"] = valid_sample_counts

    split_tables: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "val", "test"):
        if split_frames[split_name]:
            split_tables[split_name] = pd.concat(split_frames[split_name], ignore_index=True)
        else:
            split_tables[split_name] = pd.DataFrame()

    all_logs_path = output_dir / "all_logs.csv"
    train_logs_path = output_dir / "train_logs.csv"
    val_logs_path = output_dir / "val_logs.csv"
    test_logs_path = output_dir / "test_logs.csv"
    train_samples_path = output_dir / "train_samples.parquet"
    val_samples_path = output_dir / "val_samples.parquet"
    test_samples_path = output_dir / "test_samples.parquet"
    dataset_manifest_path = output_dir / "dataset_manifest.json"

    assigned_logs.sort_values(["split", "dataset_id", "log_id"]).to_csv(all_logs_path, index=False)
    assigned_logs.loc[assigned_logs["split"] == "train"].to_csv(train_logs_path, index=False)
    assigned_logs.loc[assigned_logs["split"] == "val"].to_csv(val_logs_path, index=False)
    assigned_logs.loc[assigned_logs["split"] == "test"].to_csv(test_logs_path, index=False)

    split_tables["train"].to_parquet(train_samples_path, index=False)
    split_tables["val"].to_parquet(val_samples_path, index=False)
    split_tables["test"].to_parquet(test_samples_path, index=False)

    manifest = {
        "dataset_id": output_dir.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_dir),
        "split_policy": "whole_log",
        "source_manifest_paths": [str(Path(path)) for path in manifest_paths],
        "source_dataset_ids": sorted({record["dataset_id"] for record in log_records}),
        "input_log_count": int(len(assigned_logs)),
        "seed": int(seed),
        "split_ratios": {
            "train": float(train_ratio),
            "val": float(val_ratio),
            "test": float(test_ratio),
        },
        "split_log_counts": {
            split_name: int((assigned_logs["split"] == split_name).sum())
            for split_name in ("train", "val", "test")
        },
        "split_sample_counts": {
            split_name: int(len(split_tables[split_name]))
            for split_name in ("train", "val", "test")
        },
        "all_logs_csv": str(all_logs_path),
        "train_logs_csv": str(train_logs_path),
        "val_logs_csv": str(val_logs_path),
        "test_logs_csv": str(test_logs_path),
        "train_samples_parquet": str(train_samples_path),
        "val_samples_parquet": str(val_samples_path),
        "test_samples_parquet": str(test_samples_path),
    }
    dataset_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "dataset_manifest_path": str(dataset_manifest_path),
        "all_logs_path": str(all_logs_path),
        "train_logs_path": str(train_logs_path),
        "val_logs_path": str(val_logs_path),
        "test_logs_path": str(test_logs_path),
        "train_samples_path": str(train_samples_path),
        "val_samples_path": str(val_samples_path),
        "test_samples_path": str(test_samples_path),
    }
