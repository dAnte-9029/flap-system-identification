#!/usr/bin/env python3
"""Sweep internal DeLaurier exporter parameters against fx/fz flight-log labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_fx_fz_correction import TARGETS, _array, _load_split, _metrics_rows

SPLITS = ("train", "val", "test")


def _parse_csv_floats(text: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def _candidate_name(stage: str, params: dict[str, float | bool]) -> str:
    parts = [stage]
    for key in ("theta_w_deg", "twist_eta_max_deg", "eta_s", "cd_f"):
        value = params[key]
        text = str(value).replace("-", "m").replace(".", "p")
        parts.append(f"{key}_{text}")
    return "__".join(parts)


def _export_prior(
    *,
    python_exe: Path,
    exporter: Path,
    split_root: Path,
    metadata: Path,
    output_root: Path,
    params: dict[str, float | bool],
    chunk_size: int,
    device: str,
    reuse_existing: bool,
) -> None:
    if reuse_existing and all((output_root / f"{split}_predictions.parquet").exists() for split in SPLITS):
        return
    cmd = [
        str(python_exe),
        str(exporter),
        "--split-root",
        str(split_root),
        "--metadata",
        str(metadata),
        "--output-root",
        str(output_root),
        "--overwrite",
        "--chunk-size",
        str(int(chunk_size)),
        "--device",
        device,
        "--theta-w-deg",
        str(float(params["theta_w_deg"])),
        "--twist-eta-max-deg",
        str(float(params["twist_eta_max_deg"])),
        "--twist-eta-limit-deg",
        str(float(params["twist_eta_max_deg"])),
        "--eta-s",
        str(float(params["eta_s"])),
        "--cd-f",
        str(float(params["cd_f"])),
    ]
    subprocess.run(cmd, check=True, cwd=exporter.parents[2])


def _evaluate_prior(*, split_root: Path, prior_root: Path, model_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in SPLITS:
        frame = _load_split(split_root, prior_root, split)
        pred = _array(frame, tuple(f"prior_{target}" for target in TARGETS))
        rows.extend(_metrics_rows(frame, pred, split=split, model=model_name))
    return rows


def _val_mean_rmse(rows: list[dict[str, object]], model_name: str) -> float:
    for row in rows:
        if row["split"] == "val" and row["model"] == model_name and row["target"] == "fx_fz_mean":
            return float(row["rmse"])
    raise ValueError(f"missing val fx_fz_mean for {model_name}")


def _run_candidate(
    *,
    python_exe: Path,
    exporter: Path,
    split_root: Path,
    metadata: Path,
    prior_root: Path,
    params: dict[str, float | bool],
    chunk_size: int,
    device: str,
    reuse_existing: bool,
    model_name: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    _export_prior(
        python_exe=python_exe,
        exporter=exporter,
        split_root=split_root,
        metadata=metadata,
        output_root=prior_root,
        params=params,
        chunk_size=chunk_size,
        device=device,
        reuse_existing=reuse_existing,
    )
    rows = _evaluate_prior(split_root=split_root, prior_root=prior_root, model_name=model_name)
    summary = {
        "model": model_name,
        "prior_root": str(prior_root),
        "val_rmse": _val_mean_rmse(rows, model_name),
        **params,
    }
    return rows, summary


def run(
    *,
    split_root: Path,
    metadata: Path,
    output_root: Path,
    exporter: Path,
    python_exe: Path,
    theta_values: tuple[float, ...],
    twist_values: tuple[float, ...],
    eta_values: tuple[float, ...],
    cd_f_values: tuple[float, ...],
    chunk_size: int,
    device: str,
    reuse_existing: bool,
) -> dict[str, object]:
    split_root = split_root.resolve()
    metadata = metadata.resolve()
    output_root = output_root.resolve()
    exporter = exporter.resolve()
    python_exe = python_exe.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    prior_parent = output_root / "priors"
    prior_parent.mkdir(exist_ok=True)
    all_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []

    base_params: dict[str, float | bool] = {
        "theta_w_deg": 0.0,
        "twist_eta_max_deg": 10.0,
        "eta_s": 0.65,
        "cd_f": 0.028,
    }

    # Stage 1: incidence/twist sweep with nominal thrust and friction parameters.
    seen: set[tuple[float, float, float, float]] = set()
    for theta in theta_values:
        for twist in twist_values:
            params = {**base_params, "theta_w_deg": float(theta), "twist_eta_max_deg": float(twist)}
            key = (
                float(params["theta_w_deg"]),
                float(params["twist_eta_max_deg"]),
                float(params["eta_s"]),
                float(params["cd_f"]),
            )
            if key in seen:
                continue
            seen.add(key)
            name = _candidate_name("stage1", params)
            rows, summary = _run_candidate(
                python_exe=python_exe,
                exporter=exporter,
                split_root=split_root,
                metadata=metadata,
                prior_root=prior_parent / name,
                params=params,
                chunk_size=chunk_size,
                device=device,
                reuse_existing=reuse_existing,
                model_name=name,
            )
            all_rows.extend(rows)
            candidate_rows.append({"stage": "stage1_theta_twist", **summary})
            pd.DataFrame(candidate_rows).to_csv(output_root / "internal_parameter_sweep_candidates.csv", index=False)
            pd.DataFrame(all_rows).to_csv(output_root / "internal_parameter_sweep_metrics.csv", index=False)

    stage1 = pd.DataFrame(candidate_rows).sort_values("val_rmse").iloc[0].to_dict()
    best_theta = float(stage1["theta_w_deg"])
    best_twist = float(stage1["twist_eta_max_deg"])

    # Stage 2: thrust/drag sweep around the best stage-1 incidence and twist.
    for eta in eta_values:
        for cd_f in cd_f_values:
            params = {
                **base_params,
                "theta_w_deg": best_theta,
                "twist_eta_max_deg": best_twist,
                "eta_s": float(eta),
                "cd_f": float(cd_f),
            }
            key = (
                float(params["theta_w_deg"]),
                float(params["twist_eta_max_deg"]),
                float(params["eta_s"]),
                float(params["cd_f"]),
            )
            if key in seen:
                continue
            seen.add(key)
            name = _candidate_name("stage2", params)
            rows, summary = _run_candidate(
                python_exe=python_exe,
                exporter=exporter,
                split_root=split_root,
                metadata=metadata,
                prior_root=prior_parent / name,
                params=params,
                chunk_size=chunk_size,
                device=device,
                reuse_existing=reuse_existing,
                model_name=name,
            )
            all_rows.extend(rows)
            candidate_rows.append({"stage": "stage2_eta_cd_f", **summary})
            pd.DataFrame(candidate_rows).to_csv(output_root / "internal_parameter_sweep_candidates.csv", index=False)
            pd.DataFrame(all_rows).to_csv(output_root / "internal_parameter_sweep_metrics.csv", index=False)

    candidates = pd.DataFrame(candidate_rows).sort_values("val_rmse").reset_index(drop=True)
    metrics = pd.DataFrame(all_rows)
    selected = candidates.iloc[0].to_dict()
    metrics["is_selected"] = metrics["model"].eq(str(selected["model"]))
    candidates["is_selected"] = candidates["model"].eq(str(selected["model"]))
    candidates.to_csv(output_root / "internal_parameter_sweep_candidates.csv", index=False)
    metrics.to_csv(output_root / "internal_parameter_sweep_metrics.csv", index=False)
    manifest = {
        "split_root": str(split_root),
        "metadata": str(metadata),
        "output_root": str(output_root),
        "exporter": str(exporter),
        "python_exe": str(python_exe),
        "targets": list(TARGETS),
        "selection_metric": "validation fx_fz_mean RMSE",
        "selected": selected,
        "stage1_best": stage1,
        "scope_note": "Each candidate re-exports the DeLaurier prior with modified internal exporter parameters.",
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--exporter", type=Path, default=Path("/home/zn/IsaacLab/scripts/flapping_px4/export_delaurier_prior_predictions.py"))
    parser.add_argument("--python-exe", type=Path, default=Path("/home/zn/anaconda3/envs/env_isaaclab/bin/python"))
    parser.add_argument("--theta-values", default="-12,-8,-4,0,4,8,12")
    parser.add_argument("--twist-values", default="0,5,10,15,20")
    parser.add_argument("--eta-values", default="0.25,0.5,0.65,0.85,1.1")
    parser.add_argument("--cd-f-values", default="0.0,0.015,0.028,0.05,0.08")
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                split_root=args.split_root,
                metadata=args.metadata,
                output_root=args.output_root,
                exporter=args.exporter,
                python_exe=args.python_exe,
                theta_values=_parse_csv_floats(args.theta_values),
                twist_values=_parse_csv_floats(args.twist_values),
                eta_values=_parse_csv_floats(args.eta_values),
                cd_f_values=_parse_csv_floats(args.cd_f_values),
                chunk_size=args.chunk_size,
                device=args.device,
                reuse_existing=args.reuse_existing,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
