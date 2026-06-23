#!/usr/bin/env python3
"""Audit ratio-8 flapping phase and frequency consistency in canonical splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SPLITS = ("train", "val", "test")


def _wrap_pi(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def _circular_offset_stats(reference_phase: np.ndarray, observed_phase: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference_phase, dtype=float)
    observed = np.asarray(observed_phase, dtype=float)
    mask = np.isfinite(reference) & np.isfinite(observed)
    if mask.sum() == 0:
        return {
            "offset_rad": np.nan,
            "offset_deg": np.nan,
            "resultant_R": np.nan,
            "median_abs_res_rad": np.nan,
            "p95_abs_res_rad": np.nan,
            "rmse_rad": np.nan,
        }
    delta = _wrap_pi(observed[mask] - reference[mask])
    mean_vector = np.mean(np.exp(1j * delta))
    offset = float(np.angle(mean_vector))
    residual = _wrap_pi(delta - offset)
    return {
        "offset_rad": offset,
        "offset_deg": float(np.degrees(offset)),
        "resultant_R": float(np.abs(mean_vector)),
        "median_abs_res_rad": float(np.median(np.abs(residual))),
        "p95_abs_res_rad": float(np.percentile(np.abs(residual), 95.0)),
        "rmse_rad": float(np.sqrt(np.mean(residual * residual))),
    }


def _median_derivative_frequency(time_s: np.ndarray, unwrapped_phase: np.ndarray) -> float:
    time = np.asarray(time_s, dtype=float)
    phase = np.asarray(unwrapped_phase, dtype=float)
    mask = np.isfinite(time) & np.isfinite(phase)
    time = time[mask]
    phase = phase[mask]
    if len(time) < 2:
        return np.nan
    dt = np.diff(time)
    dphi = np.diff(phase)
    good = (dt > 0.0) & np.isfinite(dt) & np.isfinite(dphi)
    if not np.any(good):
        return np.nan
    return float(np.median(dphi[good] / dt[good] / (2.0 * np.pi)))


def _read_split_frames(split_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split in SPLITS:
        path = split_root / f"{split}_samples.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        frame["split"] = split
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No split sample parquets found under {split_root}")
    return pd.concat(frames, ignore_index=True)


def audit_ratio8_phase_frequency(*, split_root: Path, ratio: float, output_root: Path) -> dict[str, Any]:
    samples = _read_split_frames(split_root)
    output_root.mkdir(parents=True, exist_ok=True)

    required = {"log_id", "time_s", "encoder_phase_unwrapped_rad", "wing_phase.phase_rad", "flap_frequency_hz"}
    missing = sorted(required - set(samples.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    phase_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []

    for (split, log_id), group in samples.groupby(["split", "log_id"], sort=True):
        group = group.sort_values("time_s")
        encoder_phase = group["encoder_phase_unwrapped_rad"].to_numpy(dtype=float) / float(ratio)
        wing_phase = group["wing_phase.phase_rad"].to_numpy(dtype=float)
        phase_stats = _circular_offset_stats(np.mod(encoder_phase, 2.0 * np.pi), wing_phase)
        phase_rows.append({"split": split, "log_id": log_id, "n": int(len(group)), **phase_stats})

        wing_frequency = _median_derivative_frequency(
            group["time_s"].to_numpy(dtype=float),
            np.unwrap(wing_phase),
        )
        canonical_frequency = float(np.nanmedian(group["flap_frequency_hz"].to_numpy(dtype=float)))
        encoder_frequency = np.nan
        if "encoder_rpm_est" in group.columns:
            encoder_frequency = float(np.nanmedian(np.abs(group["encoder_rpm_est"].to_numpy(dtype=float)) / (60.0 * float(ratio))))
        topic_frequency = np.nan
        if "flap_frequency_topic_hz" in group.columns:
            topic_frequency = float(np.nanmedian(group["flap_frequency_topic_hz"].to_numpy(dtype=float)))
        frequency_source = "missing"
        if "flap_frequency_hz_source" in group.columns and not group["flap_frequency_hz_source"].empty:
            frequency_source = str(group["flap_frequency_hz_source"].mode(dropna=True).iloc[0])

        frequency_rows.append(
            {
                "split": split,
                "log_id": log_id,
                "n": int(len(group)),
                "canonical_flap_frequency_hz": canonical_frequency,
                "self_unwrapped_wing_frequency_hz": wing_frequency,
                "encoder_ratio_frequency_hz": encoder_frequency,
                "topic_flap_frequency_hz": topic_frequency,
                "canonical_to_wing_frequency_ratio": canonical_frequency / wing_frequency if np.isfinite(wing_frequency) and abs(wing_frequency) > 1e-9 else np.nan,
                "canonical_to_encoder_frequency_ratio": canonical_frequency / encoder_frequency if np.isfinite(encoder_frequency) and abs(encoder_frequency) > 1e-9 else np.nan,
                "topic_to_encoder_frequency_ratio": topic_frequency / encoder_frequency if np.isfinite(topic_frequency) and np.isfinite(encoder_frequency) and abs(encoder_frequency) > 1e-9 else np.nan,
                "flap_frequency_hz_source": frequency_source,
            }
        )

    phase_table = pd.DataFrame(phase_rows)
    frequency_table = pd.DataFrame(frequency_rows)
    phase_table.to_csv(output_root / "phase_offset_by_log.csv", index=False)
    frequency_table.to_csv(output_root / "frequency_consistency.csv", index=False)

    unverified_source_count = int(
        (frequency_table["flap_frequency_hz_source"] == "flap_frequency_topic_fallback_unverified").sum()
    )
    summary: dict[str, Any] = {
        "split_root": str(split_root),
        "output_root": str(output_root),
        "ratio": float(ratio),
        "num_logs": int(len(phase_table)),
        "median_phase_resultant_R": float(np.nanmedian(phase_table["resultant_R"].to_numpy(dtype=float))),
        "median_phase_rmse_rad": float(np.nanmedian(phase_table["rmse_rad"].to_numpy(dtype=float))),
        "median_phase_abs_res_rad": float(np.nanmedian(phase_table["median_abs_res_rad"].to_numpy(dtype=float))),
        "median_canonical_to_wing_frequency_ratio": float(
            np.nanmedian(frequency_table["canonical_to_wing_frequency_ratio"].to_numpy(dtype=float))
        ),
        "median_canonical_to_encoder_frequency_ratio": float(
            np.nanmedian(frequency_table["canonical_to_encoder_frequency_ratio"].to_numpy(dtype=float))
        ),
        "median_topic_to_encoder_frequency_ratio": float(
            np.nanmedian(frequency_table["topic_to_encoder_frequency_ratio"].to_numpy(dtype=float))
        ),
        "unverified_topic_frequency_log_count": unverified_source_count,
    }
    legacy_encoder_ratio = float(ratio) / 7.5
    summary["encoder_rpm_est_matches_legacy_ratio_7p5"] = bool(
        np.isfinite(summary["median_canonical_to_encoder_frequency_ratio"])
        and abs(summary["median_canonical_to_encoder_frequency_ratio"] - legacy_encoder_ratio) < 0.01
    )
    summary["pass"] = bool(
        summary["median_phase_resultant_R"] > 0.98
        and summary["median_phase_rmse_rad"] < 0.20
        and 0.98 <= summary["median_canonical_to_wing_frequency_ratio"] <= 1.02
        and unverified_source_count == 0
    )

    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--ratio", type=float, default=8.0)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    summary = audit_ratio8_phase_frequency(split_root=args.split_root, ratio=args.ratio, output_root=args.output_root)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
