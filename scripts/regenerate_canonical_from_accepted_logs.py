#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from system_identification.pipeline import run_ulog_to_canonical


WRENCH_COLUMNS = ["fx_b", "fy_b", "fz_b", "mx_b", "my_b", "mz_b"]


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def _finite_ratio(frame: pd.DataFrame, columns: list[str]) -> float:
    if frame.empty or any(column not in frame.columns for column in columns):
        return 0.0
    return float(frame[columns].notna().all(axis=1).mean())


def _source_log_size_mb(path: Path) -> float:
    return float(path.stat().st_size / (1024.0 * 1024.0))


def build_accepted_log_record(
    *,
    input_row: pd.Series,
    dataset_id: str,
    outputs: dict[str, Path],
    source_log_path: Path,
) -> dict[str, Any]:
    samples = pd.read_parquet(outputs["samples_path"])
    segments = pd.read_parquet(outputs["segments_path"]) if outputs["segments_path"].exists() else pd.DataFrame()

    record = input_row.to_dict()
    record.update(
        {
            "source_dataset_id": dataset_id,
            "source_log_path": str(source_log_path),
            "source_size_mb": _source_log_size_mb(source_log_path),
            "source_manifest_path": str(outputs["manifest_path"]),
            "samples_path": str(outputs["samples_path"]),
            "segments_path": str(outputs["segments_path"]),
            "report_path": str(outputs["report_path"]),
            "sample_count": int(len(samples)),
            "segment_count": int(len(segments)),
            "label_valid_ratio": _finite_ratio(samples, WRENCH_COLUMNS),
            "fx_finite_ratio": _finite_ratio(samples, ["fx_b"]),
            "mx_finite_ratio": _finite_ratio(samples, ["mx_b"]),
            "valid_duration_s": float(samples["time_s"].max() - samples["time_s"].min()) if "time_s" in samples and len(samples) else 0.0,
        }
    )
    return record


def regenerate_from_accepted_logs(
    *,
    accepted_logs_csv: str | Path,
    metadata_path: str | Path,
    output_root: str | Path,
    rate_hz: float = 100.0,
    limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    accepted_path = _resolve_path(accepted_logs_csv)
    metadata = _resolve_path(metadata_path)
    output = _resolve_path(output_root)

    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output root already exists and is not empty: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    accepted = pd.read_csv(accepted_path)
    if "source_log_path" not in accepted.columns:
        raise ValueError(f"{accepted_path} must contain a source_log_path column")

    if limit is not None:
        accepted = accepted.head(int(limit)).copy()

    dataset_id = output.name
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for row in accepted.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        source_log_path = _resolve_path(row_series["source_log_path"])
        try:
            outputs = run_ulog_to_canonical(
                ulg_path=source_log_path,
                metadata_path=metadata,
                output_root=output,
                rate_hz=rate_hz,
            )
        except Exception as exc:  # pragma: no cover - exercised in real batch runs
            failures.append(
                {
                    "log_id": str(row_series.get("log_id", source_log_path.stem)),
                    "source_log_path": str(source_log_path),
                    "error": repr(exc),
                }
            )
            continue

        records.append(
            build_accepted_log_record(
                input_row=row_series,
                dataset_id=dataset_id,
                outputs=outputs,
                source_log_path=source_log_path,
            )
        )

    accepted_logs_csv_path = output / "accepted_logs.csv"
    accepted_logs_json_path = output / "accepted_logs.json"
    excluded_logs_csv_path = output / "excluded_logs.csv"
    excluded_logs_json_path = output / "excluded_logs.json"
    manifest_path = output / "dataset_manifest.json"

    accepted_frame = pd.DataFrame(records)
    accepted_frame.to_csv(accepted_logs_csv_path, index=False)
    accepted_logs_json_path.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")

    excluded_frame = pd.DataFrame(failures, columns=["log_id", "source_log_path", "error"])
    excluded_frame.to_csv(excluded_logs_csv_path, index=False)
    excluded_logs_json_path.write_text(json.dumps(failures, indent=2, sort_keys=True), encoding="utf-8")

    manifest = {
        "accepted_log_count": int(len(records)),
        "accepted_logs_csv": str(accepted_logs_csv_path),
        "accepted_logs_json": str(accepted_logs_json_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "excluded_log_count": int(len(failures)),
        "excluded_logs_csv": str(excluded_logs_csv_path),
        "excluded_logs_json": str(excluded_logs_json_path),
        "label_status": "effective wrench regenerated with measured mass properties metadata",
        "metadata_path": str(metadata),
        "output_root": str(output),
        "pipeline_version": "ulog_to_canonical_v0.2_measured_massprops_batch",
        "rate_hz": float(rate_hz),
        "source_accepted_logs_csv": str(accepted_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "dataset_manifest_path": manifest_path,
        "accepted_logs_csv_path": accepted_logs_csv_path,
        "accepted_logs_json_path": accepted_logs_json_path,
        "excluded_logs_csv_path": excluded_logs_csv_path,
        "excluded_logs_json_path": excluded_logs_json_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate canonical parquet logs from an accepted_logs.csv file")
    parser.add_argument("--accepted-logs-csv", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rate-hz", type=float, default=100.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = regenerate_from_accepted_logs(
        accepted_logs_csv=args.accepted_logs_csv,
        metadata_path=args.metadata,
        output_root=args.output,
        rate_hz=args.rate_hz,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
