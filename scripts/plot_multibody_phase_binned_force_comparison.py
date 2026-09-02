#!/usr/bin/env python3
"""Plot phase-binned DeLaurier, rigid, and multibody Fx/Fz curves."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from system_identification.artifacts.io import sha256_file, write_json, write_table
from system_identification.artifacts.prior_registry import resolve_delaurier_prior


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_REGISTRY = PROJECT_ROOT / "configs/data/canonical_dataset_registry.yaml"
DEFAULT_PRIOR_REGISTRY = PROJECT_ROOT / "configs/physics/delaurier_prior_registry.yaml"
DEFAULT_CANDIDATE_ROOT = PROJECT_ROOT / "dataset/canonical_v0.5_multibody_wrench_candidate_trainval_v2"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts/20260813_multibody_phase_binned_force_comparison_v2"
PARTITIONS = ("train", "validation")
SAMPLE_FILES = {"train": "train_samples.parquet", "validation": "val_samples.parquet"}
PREDICTION_FILES = {"train": "train_predictions.parquet", "validation": "val_predictions.parquet"}
KEY_COLUMNS = ("log_id", "timestamp_us")
COMPONENTS = ("fx_b", "fz_b")
CURVES = ("delaurier", "rigid_inverse_v04", "multibody_inverse_v05")
CURVE_STYLE = {
    "delaurier": {"label": "DeLaurier prior", "color": "#333333", "linestyle": "--", "linewidth": 1.8},
    "rigid_inverse_v04": {
        "label": "Rigid inverse (v0.4)",
        "color": "#0072B2",
        "linestyle": "-",
        "linewidth": 1.8,
    },
    "multibody_inverse_v05": {
        "label": "Multibody inverse (v0.5-v2)",
        "color": "#D55E00",
        "linestyle": "-",
        "linewidth": 2.0,
    },
}


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def _absolute(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _assert_hash(path: Path, expected: object, description: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != str(expected):
        raise ValueError(f"{description} hash mismatch: expected {expected}, got {actual}")
    return actual


def resolve_inputs(
    *,
    dataset_registry_path: Path,
    prior_registry_path: Path,
    candidate_root: Path,
    partitions: Sequence[str] = PARTITIONS,
) -> dict[str, Any]:
    requested = tuple(str(value) for value in partitions)
    if requested != PARTITIONS:
        raise ValueError("This pre-C7 comparison requires exactly train and validation; test remains sealed")

    dataset_registry = _read_mapping(dataset_registry_path.resolve())
    dataset_id = str(dataset_registry.get("default_dataset_id", ""))
    dataset_entry = dataset_registry.get("datasets", {}).get(dataset_id)
    if not isinstance(dataset_entry, Mapping) or dataset_entry.get("lifecycle_status") != "active":
        raise ValueError("Canonical registry default is missing or not active")
    dataset_root = _absolute(dataset_entry["repository_relative_root"])
    dataset_manifest_path = _absolute(dataset_entry["manifest_path"])
    dataset_manifest_hash = _assert_hash(
        dataset_manifest_path, dataset_entry["manifest_sha256"], "Canonical dataset manifest"
    )
    dataset_hashes: dict[str, str] = {}
    for partition in requested:
        sample_path = dataset_root / SAMPLE_FILES[partition]
        dataset_hashes[partition] = _assert_hash(
            sample_path,
            dataset_entry["artifact_sha256"][sample_path.name],
            f"Canonical {partition} samples",
        )

    prior = resolve_delaurier_prior(
        registry_path=prior_registry_path.resolve(), requested_partitions=requested
    )
    prior_manifest_path = prior.artifact_root / "manifest.json"
    prior_manifest = _read_mapping(prior_manifest_path)
    prior_manifest_hash = _assert_hash(
        prior_manifest_path,
        prior.registry_entry["artifact_manifest_sha256"],
        "DeLaurier prior manifest",
    )
    if prior.manifest.get("test_partition_loaded", True):
        raise ValueError("DeLaurier prior manifest reports test access")
    if str(prior.manifest.get("dataset_id")) != dataset_id:
        raise ValueError("DeLaurier prior and canonical dataset IDs disagree")
    prior_hashes: dict[str, str] = {}
    for partition in requested:
        prediction_path = prior.artifact_root / PREDICTION_FILES[partition]
        prior_hashes[partition] = _assert_hash(
            prediction_path,
            prior_manifest["prediction_sha256"][partition],
            f"DeLaurier {partition} predictions",
        )

    candidate_root = candidate_root.resolve()
    candidate_manifest_path = candidate_root / "dataset_manifest.json"
    candidate_manifest = _read_mapping(candidate_manifest_path)
    if candidate_manifest.get("lifecycle_status") != "candidate_train_validation_only":
        raise ValueError("Multibody dataset is not a train/validation-only candidate")
    if tuple(candidate_manifest.get("included_partitions", [])) != PARTITIONS:
        raise ValueError("Multibody candidate does not contain exactly train and validation")
    if candidate_manifest.get("test_labels_loaded", True):
        raise ValueError("Multibody candidate reports test-label access")
    source = candidate_manifest.get("source_dataset")
    if not isinstance(source, Mapping) or source.get("dataset_id") != dataset_id:
        raise ValueError("Multibody candidate source dataset identity mismatch")
    if source.get("manifest_sha256") != dataset_manifest_hash:
        raise ValueError("Multibody candidate source manifest hash mismatch")
    candidate_hashes: dict[str, str] = {}
    for partition in requested:
        sample_path = candidate_root / SAMPLE_FILES[partition]
        candidate_hashes[partition] = _assert_hash(
            sample_path,
            candidate_manifest["artifact_sha256"][sample_path.name],
            f"Multibody {partition} samples",
        )

    return {
        "dataset_id": dataset_id,
        "dataset_root": dataset_root,
        "dataset_manifest_path": dataset_manifest_path,
        "dataset_manifest_hash": dataset_manifest_hash,
        "dataset_hashes": dataset_hashes,
        "dataset_entry": dataset_entry,
        "prior_id": prior.prior_id,
        "prior_root": prior.artifact_root,
        "prior_manifest_path": prior_manifest_path,
        "prior_manifest_hash": prior_manifest_hash,
        "prior_hashes": prior_hashes,
        "prior_resolution": prior,
        "candidate_root": candidate_root,
        "candidate_manifest_path": candidate_manifest_path,
        "candidate_manifest_hash": sha256_file(candidate_manifest_path),
        "candidate_hashes": candidate_hashes,
        "candidate_manifest": candidate_manifest,
    }


def _require_unique_keys(frame: pd.DataFrame, source: str) -> None:
    missing = [column for column in KEY_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing key columns: {missing}")
    duplicate = frame.duplicated(list(KEY_COLUMNS), keep=False)
    if duplicate.any():
        raise ValueError(f"{source} has {int(duplicate.sum())} rows with duplicate alignment keys")


def _outer_keyed_merge(left: pd.DataFrame, right: pd.DataFrame, *, source: str) -> pd.DataFrame:
    merged = left.merge(right, on=list(KEY_COLUMNS), how="outer", validate="one_to_one", indicator=True)
    mismatch = merged["_merge"].ne("both")
    if mismatch.any():
        counts = merged.loc[mismatch, "_merge"].value_counts().to_dict()
        raise ValueError(f"{source} alignment key mismatch: {counts}")
    return merged.drop(columns="_merge")


def align_partition(old: pd.DataFrame, corrected: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
    """Align three force sources by stable keys and retain only valid shared rows."""

    for frame, source in ((old, "canonical v0.4"), (corrected, "multibody v0.5-v2"), (prior, "DeLaurier")):
        _require_unique_keys(frame, source)

    old_required = {
        "mechanical_phase_rad",
        "fx_b",
        "fz_b",
        "phase_valid",
        "label_reconstruction_valid",
    }
    corrected_required = {
        "mechanical_phase_rad",
        "fx_b",
        "fz_b",
        "fx_b_rigid_v01",
        "fz_b_rigid_v01",
        "multibody_label_valid",
    }
    prior_required = set(COMPONENTS)
    for frame, required, source in (
        (old, old_required, "canonical v0.4"),
        (corrected, corrected_required, "multibody v0.5-v2"),
        (prior, prior_required, "DeLaurier"),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{source} is missing required columns: {missing}")

    old_view = old.loc[:, [*KEY_COLUMNS, *sorted(old_required)]].rename(
        columns={
            "mechanical_phase_rad": "phase_old_rad",
            "fx_b": "rigid_inverse_fx_b",
            "fz_b": "rigid_inverse_fz_b",
        }
    )
    corrected_view = corrected.loc[:, [*KEY_COLUMNS, *sorted(corrected_required)]].rename(
        columns={
            "mechanical_phase_rad": "phase_corrected_rad",
            "fx_b": "multibody_inverse_fx_b",
            "fz_b": "multibody_inverse_fz_b",
            "fx_b_rigid_v01": "candidate_backup_fx_b",
            "fz_b_rigid_v01": "candidate_backup_fz_b",
        }
    )
    prior_view = prior.loc[:, [*KEY_COLUMNS, *COMPONENTS]].rename(
        columns={"fx_b": "delaurier_fx_b", "fz_b": "delaurier_fz_b"}
    )
    aligned = _outer_keyed_merge(old_view, corrected_view, source="v0.4 versus v0.5-v2")
    aligned = _outer_keyed_merge(aligned, prior_view, source="labels versus DeLaurier")

    if not np.allclose(aligned["phase_old_rad"], aligned["phase_corrected_rad"], rtol=0.0, atol=1e-12):
        raise ValueError("v0.4 and v0.5-v2 mechanical phase columns disagree")
    for component in COMPONENTS:
        if not np.allclose(
            aligned[f"rigid_inverse_{component}"],
            aligned[f"candidate_backup_{component}"],
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError(f"v0.5-v2 did not preserve the v0.4 {component} backup exactly")

    numeric = [
        "phase_old_rad",
        *(f"delaurier_{component}" for component in COMPONENTS),
        *(f"rigid_inverse_{component}" for component in COMPONENTS),
        *(f"multibody_inverse_{component}" for component in COMPONENTS),
    ]
    finite = np.isfinite(aligned[numeric].to_numpy(dtype=float)).all(axis=1)
    valid = (
        aligned["phase_valid"].astype(bool).to_numpy()
        & aligned["label_reconstruction_valid"].astype(bool).to_numpy()
        & aligned["multibody_label_valid"].astype(bool).to_numpy()
        & finite
    )
    return aligned.loc[valid].reset_index(drop=True)


def equal_log_phase_bins(frame: pd.DataFrame, *, partition: str, phase_bins: int = 72) -> pd.DataFrame:
    """Compute per-log bin means followed by an equal-log macro mean."""

    if phase_bins < 2:
        raise ValueError("phase_bins must be at least 2")
    phase = np.mod(frame["phase_old_rad"].to_numpy(dtype=float), 2.0 * math.pi)
    bin_index = np.floor(phase / (2.0 * math.pi) * int(phase_bins)).astype(int)
    work = frame.loc[:, ["log_id"]].copy()
    work["phase_bin"] = np.clip(bin_index, 0, int(phase_bins) - 1)
    value_columns: list[str] = []
    for component in COMPONENTS:
        for curve in CURVES:
            column = f"{curve}_{component}"
            source_column = {
                "delaurier": f"delaurier_{component}",
                "rigid_inverse_v04": f"rigid_inverse_{component}",
                "multibody_inverse_v05": f"multibody_inverse_{component}",
            }[curve]
            work[column] = frame[source_column].to_numpy(dtype=float)
            value_columns.append(column)
    per_log = work.groupby(["log_id", "phase_bin"], sort=True, as_index=False)[value_columns].mean()
    samples_per_log_bin = work.groupby(["log_id", "phase_bin"], sort=True).size().rename("sample_count")

    rows: list[dict[str, Any]] = []
    bin_width = 2.0 * math.pi / int(phase_bins)
    for phase_bin in range(int(phase_bins)):
        group = per_log.loc[per_log["phase_bin"].eq(phase_bin)]
        sample_count = int(samples_per_log_bin.xs(phase_bin, level="phase_bin").sum()) if not group.empty else 0
        for component in COMPONENTS:
            for curve in CURVES:
                values = group[f"{curve}_{component}"].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                std = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
                rows.append(
                    {
                        "partition": partition,
                        "component": component,
                        "curve": curve,
                        "phase_bin": phase_bin,
                        "phase_center_rad": (phase_bin + 0.5) * bin_width,
                        "force_mean_n": float(np.mean(values)) if len(values) else np.nan,
                        "force_std_across_logs_n": std,
                        "force_sem_across_logs_n": std / math.sqrt(len(values)) if len(values) > 1 else np.nan,
                        "log_count": int(len(values)),
                        "sample_count": sample_count,
                    }
                )
    return pd.DataFrame(rows)


def summarize_curves(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (partition, component), group in table.groupby(["partition", "component"], sort=True):
        pivot = group.pivot(index="phase_bin", columns="curve", values="force_mean_n").sort_index()
        old = pivot["rigid_inverse_v04"].to_numpy(dtype=float)
        corrected = pivot["multibody_inverse_v05"].to_numpy(dtype=float)
        prior = pivot["delaurier"].to_numpy(dtype=float)
        delta = corrected - old
        phase = (
            group.loc[group["curve"].eq("delaurier")]
            .sort_values("phase_bin")["phase_center_rad"]
            .to_numpy(dtype=float)
        )
        peak_index = int(np.nanargmax(np.abs(delta)))
        rows.append(
            {
                "partition": partition,
                "component": component,
                "multibody_minus_rigid_mean_n": float(np.nanmean(delta)),
                "multibody_minus_rigid_waveform_rms_n": float(np.sqrt(np.nanmean(delta * delta))),
                "multibody_minus_rigid_peak_abs_n": float(abs(delta[peak_index])),
                "multibody_minus_rigid_peak_phase_rad": float(phase[peak_index]),
                "delaurier_vs_rigid_waveform_rmse_n": float(np.sqrt(np.nanmean((prior - old) ** 2))),
                "delaurier_vs_multibody_waveform_rmse_n": float(np.sqrt(np.nanmean((prior - corrected) ** 2))),
                "delaurier_vs_rigid_waveform_corr": float(np.corrcoef(prior, old)[0, 1]),
                "delaurier_vs_multibody_waveform_corr": float(np.corrcoef(prior, corrected)[0, 1]),
            }
        )
    return pd.DataFrame(rows)


def plot_phase_curves(table: pd.DataFrame, output_path: Path) -> None:
    if output_path.exists() or output_path.with_suffix(".pdf").exists():
        raise FileExistsError(f"Refusing to overwrite figure: {output_path}")
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.0), sharex=True)
    component_labels = {"fx_b": r"$F_x$", "fz_b": r"$F_z$"}
    partition_labels = {"train": "Train", "validation": "Validation"}
    for row, partition in enumerate(PARTITIONS):
        for column, component in enumerate(COMPONENTS):
            axis = axes[row, column]
            group = table.loc[(table["partition"] == partition) & (table["component"] == component)]
            for curve in CURVES:
                curve_frame = group.loc[group["curve"].eq(curve)].sort_values("phase_bin")
                style = CURVE_STYLE[curve]
                axis.plot(
                    curve_frame["phase_center_rad"],
                    curve_frame["force_mean_n"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    label=style["label"],
                    zorder=3,
                )
            log_count = int(group["log_count"].max())
            axis.axhline(0.0, color="#777777", linewidth=0.7, zorder=1)
            axis.grid(True, color="#E5E7EB", linewidth=0.7, zorder=0)
            axis.set_xlim(0.0, 2.0 * math.pi)
            axis.set_ylabel(f"{component_labels[component]} (N)")
            axis.set_title(f"{partition_labels[partition]}: {component_labels[component]} ({log_count} logs)")
    ticks = [0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi, 2.0 * math.pi]
    tick_labels = ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
    for axis in axes[-1, :]:
        axis.set_xticks(ticks, tick_labels)
        axis.set_xlabel("Mechanical wingbeat phase (rad)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=3, frameon=False)
    fig.suptitle("Phase-binned longitudinal force comparison (equal-log macro mean)", fontsize=13, y=0.925)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88), h_pad=1.4, w_pad=1.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(
    *,
    dataset_registry_path: Path = DEFAULT_DATASET_REGISTRY,
    prior_registry_path: Path = DEFAULT_PRIOR_REGISTRY,
    candidate_root: Path = DEFAULT_CANDIDATE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    phase_bins: int = 72,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_root}")
    resolved = resolve_inputs(
        dataset_registry_path=dataset_registry_path,
        prior_registry_path=prior_registry_path,
        candidate_root=candidate_root,
    )
    tables: list[pd.DataFrame] = []
    row_counts: dict[str, dict[str, int]] = {}
    for partition in PARTITIONS:
        old = pd.read_parquet(
            resolved["dataset_root"] / SAMPLE_FILES[partition],
            columns=[*KEY_COLUMNS, "mechanical_phase_rad", "fx_b", "fz_b", "phase_valid", "label_reconstruction_valid"],
        )
        corrected = pd.read_parquet(
            resolved["candidate_root"] / SAMPLE_FILES[partition],
            columns=[
                *KEY_COLUMNS,
                "mechanical_phase_rad",
                "fx_b",
                "fz_b",
                "fx_b_rigid_v01",
                "fz_b_rigid_v01",
                "multibody_label_valid",
            ],
        )
        prior = pd.read_parquet(
            resolved["prior_root"] / PREDICTION_FILES[partition],
            columns=[*KEY_COLUMNS, *COMPONENTS],
        )
        aligned = align_partition(old, corrected, prior)
        row_counts[partition] = {"source": int(len(old)), "valid_aligned": int(len(aligned))}
        tables.append(equal_log_phase_bins(aligned, partition=partition, phase_bins=phase_bins))

    phase_table = pd.concat(tables, ignore_index=True)
    summary = summarize_curves(phase_table)
    output_root.mkdir(parents=True, exist_ok=False)
    write_table(output_root / "phase_binned_equal_log_curves.csv", phase_table)
    write_table(output_root / "phase_binned_curve_summary.csv", summary)
    figure_path = output_root / "phase_binned_fx_fz_three_way_train_validation.png"
    plot_phase_curves(phase_table, figure_path)
    manifest = {
        "schema_version": "multibody_phase_binned_force_comparison_v1",
        "partitions": list(PARTITIONS),
        "test_partition_loaded": False,
        "phase_bins": int(phase_bins),
        "aggregation": "per-log phase-bin mean, followed by equal-log macro mean",
        "phase_contract": resolved["dataset_entry"]["phase_contract"],
        "frequency_contract": resolved["dataset_entry"]["frequency_contract"],
        "alignment_keys": list(KEY_COLUMNS),
        "row_counts": row_counts,
        "canonical_dataset": {
            "dataset_id": resolved["dataset_id"],
            "root": str(resolved["dataset_root"]),
            "manifest_path": str(resolved["dataset_manifest_path"]),
            "manifest_sha256": resolved["dataset_manifest_hash"],
            "sample_sha256": resolved["dataset_hashes"],
        },
        "delaurier_prior": {
            "prior_id": resolved["prior_id"],
            "root": str(resolved["prior_root"]),
            "lifecycle_status": resolved["prior_resolution"].lifecycle_status,
            "physics_source_commit": resolved["prior_resolution"].physics_source_commit,
            "frame_contract": resolved["prior_resolution"].frame_contract,
            "airflow_contract": resolved["prior_resolution"].airflow_contract,
            "phase_contract": resolved["prior_resolution"].phase_contract,
            "manifest_sha256": resolved["prior_manifest_hash"],
            "prediction_sha256": resolved["prior_hashes"],
        },
        "multibody_candidate": {
            "dataset_id": resolved["candidate_manifest"]["dataset_id"],
            "root": str(resolved["candidate_root"]),
            "manifest_path": str(resolved["candidate_manifest_path"]),
            "manifest_sha256": resolved["candidate_manifest_hash"],
            "sample_sha256": resolved["candidate_hashes"],
            "label_contract": resolved["candidate_manifest"]["label_contract"],
        },
        "outputs": {
            "phase_curves_csv": "phase_binned_equal_log_curves.csv",
            "summary_csv": "phase_binned_curve_summary.csv",
            "figure_png": figure_path.name,
            "figure_pdf": figure_path.with_suffix(".pdf").name,
        },
    }
    write_json(output_root / "manifest.json", manifest)
    return {"output_root": output_root, "figure_path": figure_path, "summary": summary, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-registry", type=Path, default=DEFAULT_DATASET_REGISTRY)
    parser.add_argument("--prior-registry", type=Path, default=DEFAULT_PRIOR_REGISTRY)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--phase-bins", type=int, default=72)
    args = parser.parse_args()
    result = run(
        dataset_registry_path=args.dataset_registry,
        prior_registry_path=args.prior_registry,
        candidate_root=args.candidate_root,
        output_root=args.output_root,
        phase_bins=args.phase_bins,
    )
    print(json.dumps({"output_root": str(result["output_root"]), "figure": str(result["figure_path"])}, indent=2))


if __name__ == "__main__":
    main()
