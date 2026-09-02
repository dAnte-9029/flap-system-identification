#!/usr/bin/env python3
"""Build an immutable train/validation dataset with three-body wrench labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from system_identification.labels.multibody_effective_wrench import (  # noqa: E402
    load_multibody_label_config,
    reconstruct_multibody_labels_from_samples,
)


PARTITION_FILENAMES = {"train": "train_samples.parquet", "validation": "val_samples.parquet"}
TARGET_COLUMNS = ("fx_b", "fy_b", "fz_b", "mx_b", "my_b", "mz_b")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_identity() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()

    return {
        "branch": run("branch", "--show-current"),
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--short")),
    }


def _load_registered_source(registry_path: Path, source_dataset_id: str) -> tuple[Path, dict[str, Any], str]:
    with registry_path.open("r", encoding="utf-8") as handle:
        registry = yaml.safe_load(handle)
    if registry.get("default_dataset_id") != source_dataset_id:
        raise ValueError("Source dataset must be the registered default canonical dataset.")
    entry = registry.get("datasets", {}).get(source_dataset_id)
    if not isinstance(entry, dict) or entry.get("lifecycle_status") != "active":
        raise ValueError("Source dataset must have an active registry entry.")
    source_root = PROJECT_ROOT / entry["repository_relative_root"]
    manifest_path = PROJECT_ROOT / entry["manifest_path"]
    manifest_hash = _sha256(manifest_path)
    if manifest_hash != entry.get("manifest_sha256"):
        raise ValueError("Registered source manifest hash mismatch.")
    for partition, filename in PARTITION_FILENAMES.items():
        expected = entry.get("artifact_sha256", {}).get(filename)
        if expected is None or _sha256(source_root / filename) != expected:
            raise ValueError(f"Registered {partition} artifact hash mismatch.")
    return source_root, entry, manifest_hash


def _key_digest(frame: pd.DataFrame) -> str:
    if frame[["log_id", "timestamp_us"]].duplicated().any():
        raise ValueError("Sample keys (log_id, timestamp_us) must be unique.")
    digest = hashlib.sha256()
    for log_id, timestamp_us in frame[["log_id", "timestamp_us"]].itertuples(index=False, name=None):
        digest.update(str(log_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(timestamp_us)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _rewrite_partition(
    frame: pd.DataFrame,
    *,
    dataset_id: str,
    config: dict[str, Any],
    model: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    phase = config["phase_contract"]
    labels = reconstruct_multibody_labels_from_samples(
        frame,
        model=model,
        amplitude_rad=float(phase["amplitude_rad"]),
        phase_derivative_window_s=float(phase["derivative_window_s"]),
        phase_derivative_polyorder=int(phase["derivative_polyorder"]),
    )
    rewritten = frame.copy()
    for target in TARGET_COLUMNS:
        rewritten[f"{target}_rigid_v01"] = rewritten[target].to_numpy(dtype=float)
    for column in labels.columns:
        rewritten[column] = labels[column].to_numpy()
    for target in TARGET_COLUMNS:
        rewritten[target] = rewritten[f"{target}_multibody"].to_numpy(dtype=float)
    rewritten["label_valid"] = rewritten["multibody_label_valid"].to_numpy(dtype=bool)
    rewritten["label_reconstruction_valid"] = rewritten["multibody_label_valid"].to_numpy(dtype=bool)
    rewritten["label_variant"] = str(config["label_model_id"])
    rewritten["dataset_id"] = dataset_id

    valid = rewritten["multibody_label_valid"].to_numpy(dtype=bool)
    summary: dict[str, Any] = {
        "row_count": int(len(rewritten)),
        "valid_row_count": int(valid.sum()),
        "valid_ratio": float(valid.mean()),
        "key_sha256": _key_digest(rewritten),
    }
    for target in TARGET_COLUMNS:
        delta = rewritten.loc[valid, f"delta_{target}_multibody_minus_rigid"].to_numpy(dtype=float)
        summary[f"delta_{target}"] = {
            "mean": float(np.mean(delta)),
            "rms": float(np.sqrt(np.mean(delta**2))),
            "p95_abs": float(np.percentile(np.abs(delta), 95.0)),
            "max_abs": float(np.max(np.abs(delta))),
        }
    phase_error = rewritten.loc[valid, "position_phase_rate_minus_logged_frequency_rad_s"].to_numpy(dtype=float)
    summary["position_phase_rate_crosscheck_rad_s"] = {
        "mean": float(np.mean(phase_error)),
        "rms": float(np.sqrt(np.mean(phase_error**2))),
        "p95_abs": float(np.percentile(np.abs(phase_error), 95.0)),
    }
    for axis in "xyz":
        values = rewritten.loc[valid, f"dynamic_cg_{axis}_frd_m"].to_numpy(dtype=float)
        summary[f"dynamic_cg_{axis}_frd_m"] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "peak_to_peak": float(np.ptp(values)),
        }
    return rewritten, summary


def build_multibody_wrench_dataset(
    *,
    registry_path: str | Path,
    source_dataset_id: str,
    config_path: str | Path,
    output_root: str | Path,
) -> dict[str, Path]:
    registry = Path(registry_path).resolve()
    config_file = Path(config_path).resolve()
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable dataset root: {output}")

    source_root, source_entry, source_manifest_hash = _load_registered_source(registry, source_dataset_id)
    config, model = load_multibody_label_config(config_file)
    source_contract = config.get("source_contract", {})
    if source_contract.get("canonical_dataset_id") != source_dataset_id:
        raise ValueError("Label configuration source dataset does not match the requested source.")
    if source_contract.get("partitions") != ["train", "validation"] or source_contract.get("test_locked") is not True:
        raise ValueError("Label configuration must explicitly lock test and request train/validation only.")

    output.mkdir(parents=True, exist_ok=False)
    artifact_hashes: dict[str, str] = {}
    summaries: dict[str, Any] = {}
    source_artifact_hashes: dict[str, str] = {}
    try:
        for partition, filename in PARTITION_FILENAMES.items():
            source_path = source_root / filename
            frame = pd.read_parquet(source_path)
            source_key_hash = _key_digest(frame)
            rewritten, summary = _rewrite_partition(
                frame,
                dataset_id=output.name,
                config=config,
                model=model,
            )
            if _key_digest(rewritten) != source_key_hash:
                raise ValueError(f"{partition} sample identity changed during reconstruction.")
            output_path = output / filename
            rewritten.to_parquet(output_path, index=False)
            artifact_hashes[filename] = _sha256(output_path)
            source_artifact_hashes[filename] = _sha256(source_path)
            summaries[partition] = summary

        for filename in ("train_logs.csv", "val_logs.csv"):
            source_path = source_root / filename
            if source_path.exists():
                output_path = output / filename
                shutil.copy2(source_path, output_path)
                artifact_hashes[filename] = _sha256(output_path)

        summary_path = output / "multibody_label_summary.json"
        summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")
        artifact_hashes[summary_path.name] = _sha256(summary_path)
        content_hash = hashlib.sha256(
            "\n".join(f"{name}:{digest}" for name, digest in sorted(artifact_hashes.items())).encode("utf-8")
        ).hexdigest()
        manifest = {
            "schema_version": "canonical_multibody_wrench_candidate_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_id": output.name,
            "lifecycle_status": "candidate_train_validation_only",
            "included_partitions": ["train", "validation"],
            "excluded_partitions": ["test"],
            "test_labels_loaded": False,
            "source_dataset": {
                "dataset_id": source_dataset_id,
                "repository_relative_root": source_entry["repository_relative_root"],
                "manifest_sha256": source_manifest_hash,
                "artifact_sha256": source_artifact_hashes,
                "phase_contract": source_entry["phase_contract"],
                "frequency_contract": source_entry["frequency_contract"],
            },
            "label_config": {
                "path": str(config_file.relative_to(PROJECT_ROOT)),
                "sha256": _sha256(config_file),
                "label_model_id": config["label_model_id"],
            },
            "label_contract": config["frame_contract"],
            "phase_kinematics_contract": config["phase_contract"],
            "mass_property_provenance": config["mass_property_provenance"],
            "partition_summary": summaries,
            "artifact_sha256": artifact_hashes,
            "dataset_content_sha256": content_hash,
            "git": _git_identity(),
        }
        manifest_path = output / "dataset_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        shutil.rmtree(output)
        raise

    return {
        "output_root": output,
        "manifest_path": output / "dataset_manifest.json",
        "summary_path": output / "multibody_label_summary.json",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=PROJECT_ROOT / "configs/data/canonical_dataset_registry.yaml")
    parser.add_argument(
        "--source-dataset-id",
        default="canonical_v0.4_training_ready_split_measured_massprops_ratio8_phasefix_v3",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/data/multibody_wrench_label_v1.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_multibody_wrench_dataset(
        registry_path=args.registry,
        source_dataset_id=args.source_dataset_id,
        config_path=args.config,
        output_root=args.output,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
