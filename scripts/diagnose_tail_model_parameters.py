#!/usr/bin/env python3
"""Diagnose whether flight logs support simple tail-model parameter changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_delaurier_residual_split import align_prior_to_samples
from scripts.run_component_prior_ablation import (
    ComponentConfig,
    TARGET_COLUMNS,
    _body_drag_prior,
    _metric_row,
    _tail_prior,
)

SPLITS = ("train", "val", "test")
TAIL_FOCUS = ("fy_b", "mx_b", "my_b", "mz_b")


def _parse_csv_floats(text: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in text.split(",") if item.strip())


def _load(split_root: Path, wing_prior_root: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = pd.read_parquet(split_root / f"{split}_samples.parquet").reset_index(drop=True)
    wing_raw = pd.read_parquet(wing_prior_root / f"{split}_predictions.parquet")
    wing, _ = align_prior_to_samples(samples, wing_raw, allow_row_order_fallback=False)
    return samples, wing.loc[:, TARGET_COLUMNS].reset_index(drop=True)


def _signed_samples(samples: pd.DataFrame, *, elevon_sign: float, rudder_sign: float, swap_elevons: bool) -> pd.DataFrame:
    out = samples.copy()
    left = samples["servo_left_elevon"].to_numpy(dtype=float)
    right = samples["servo_right_elevon"].to_numpy(dtype=float)
    if swap_elevons:
        left, right = right, left
    out["servo_left_elevon"] = float(elevon_sign) * left
    out["servo_right_elevon"] = float(elevon_sign) * right
    out["servo_rudder"] = float(rudder_sign) * samples["servo_rudder"].to_numpy(dtype=float)
    return out


def _residual_target(samples: pd.DataFrame, wing: pd.DataFrame, body: pd.DataFrame) -> pd.DataFrame:
    values = samples.loc[:, TARGET_COLUMNS].to_numpy(dtype=float) - wing.to_numpy(dtype=float) - body.to_numpy(dtype=float)
    return pd.DataFrame(values, columns=TARGET_COLUMNS)


def _corr(y: np.ndarray, x: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(x)
    if int(mask.sum()) < 3 or np.std(y[mask]) <= 1.0e-12 or np.std(x[mask]) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(y[mask], x[mask])[0, 1])


def _fit_tail_gain_bias(train_residual: pd.DataFrame, train_tail: pd.DataFrame) -> dict[str, tuple[float, float]]:
    params: dict[str, tuple[float, float]] = {}
    for target in TARGET_COLUMNS:
        x = train_tail[target].to_numpy(dtype=float)
        y = train_residual[target].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if int(mask.sum()) < 3 or np.std(x[mask]) <= 1.0e-12:
            params[target] = (0.0, float(np.nanmean(y[mask])) if int(mask.sum()) else 0.0)
            continue
        design = np.column_stack([x[mask], np.ones(int(mask.sum()))])
        gain, bias = np.linalg.lstsq(design, y[mask], rcond=None)[0]
        params[target] = (float(gain), float(bias))
    return params


def _apply_tail_gain_bias(tail: pd.DataFrame, params: dict[str, tuple[float, float]]) -> pd.DataFrame:
    out = pd.DataFrame(index=tail.index)
    for target in TARGET_COLUMNS:
        gain, bias = params[target]
        out[target] = gain * tail[target].to_numpy(dtype=float) + bias
    return out.loc[:, TARGET_COLUMNS]


def run(
    *,
    split_root: Path,
    wing_prior_root: Path,
    output_root: Path,
    elevon_max_values: tuple[float, ...],
    rudder_max_values: tuple[float, ...],
    fixed_effectiveness_values: tuple[float, ...],
    elevon_effectiveness_values: tuple[float, ...],
    cda_m2: float,
    base_com_x_values: tuple[float, ...],
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    loaded = {split: _load(split_root, wing_prior_root, split) for split in SPLITS}
    body = {split: _body_drag_prior(samples, cda_m2=cda_m2) for split, (samples, _) in loaded.items()}
    residual = {split: _residual_target(samples, wing, body[split]) for split, (samples, wing) in loaded.items()}

    rows: list[dict[str, object]] = []
    param_rows: list[dict[str, object]] = []
    case_id = 0
    for elevon_sign in (-1.0, 1.0):
        for rudder_sign in (-1.0, 1.0):
            for swap_elevons in (False, True):
                for elevon_max_deg in elevon_max_values:
                    for rudder_max_deg in rudder_max_values:
                        for fixed_eff in fixed_effectiveness_values:
                            for elevon_eff in elevon_effectiveness_values:
                                for base_com_x in base_com_x_values:
                                    case_id += 1
                                    cfg = ComponentConfig(
                                        elevon_max_deg=elevon_max_deg,
                                        rudder_max_deg=rudder_max_deg,
                                        nominal_body_cda_m2=cda_m2,
                                        base_com_pos_frd_m=(base_com_x, 0.0, 0.0),
                                        tail_fixed_horizontal_effectiveness=fixed_eff,
                                        tail_elevon_effectiveness=elevon_eff,
                                        tail_elevon_alpha_limit_deg=25.0,
                                    )
                                    tails: dict[str, pd.DataFrame] = {}
                                    for split, (samples, _) in loaded.items():
                                        signed = _signed_samples(
                                            samples,
                                            elevon_sign=elevon_sign,
                                            rudder_sign=rudder_sign,
                                            swap_elevons=swap_elevons,
                                        )
                                        tails[split] = _tail_prior(signed, cfg)
                                    params = _fit_tail_gain_bias(residual["train"], tails["train"])
                                    for target, (gain, bias) in params.items():
                                        param_rows.append(
                                            {
                                                "case_id": case_id,
                                                "target": target,
                                                "gain": gain,
                                                "bias": bias,
                                            }
                                        )
                                    case = {
                                        "case_id": case_id,
                                        "elevon_sign": elevon_sign,
                                        "rudder_sign": rudder_sign,
                                        "swap_elevons": bool(swap_elevons),
                                        "elevon_max_deg": elevon_max_deg,
                                        "rudder_max_deg": rudder_max_deg,
                                        "fixed_horizontal_effectiveness": fixed_eff,
                                        "elevon_effectiveness": elevon_eff,
                                        "base_com_x_m": base_com_x,
                                    }
                                    for split in SPLITS:
                                        calibrated_tail = _apply_tail_gain_bias(tails[split], params)
                                        for target in TARGET_COLUMNS:
                                            y = residual[split][target].to_numpy(dtype=float)
                                            raw = tails[split][target].to_numpy(dtype=float)
                                            cal = calibrated_tail[target].to_numpy(dtype=float)
                                            raw_metrics = _metric_row(y, raw)
                                            cal_metrics = _metric_row(y, cal)
                                            rows.append(
                                                {
                                                    **case,
                                                    "split": split,
                                                    "target": target,
                                                    "raw_tail_rmse": raw_metrics["rmse"],
                                                    "raw_tail_r2": raw_metrics["r2"],
                                                    "raw_tail_corr": _corr(y, raw),
                                                    "calibrated_tail_rmse": cal_metrics["rmse"],
                                                    "calibrated_tail_r2": cal_metrics["r2"],
                                                    "calibrated_tail_corr": _corr(y, cal),
                                                }
                                            )

    table = pd.DataFrame(rows)
    table.to_csv(output_root / "tail_parameter_sweep_metrics.csv", index=False)
    pd.DataFrame(param_rows).to_csv(output_root / "tail_parameter_sweep_fit_params.csv", index=False)

    focus = table.loc[table["target"].isin(TAIL_FOCUS)].copy()
    aggregate = (
        focus.groupby(
            [
                "case_id",
                "split",
                "elevon_sign",
                "rudder_sign",
                "swap_elevons",
                "elevon_max_deg",
                "rudder_max_deg",
                "fixed_horizontal_effectiveness",
                "elevon_effectiveness",
                "base_com_x_m",
            ],
            observed=True,
        )
        .agg(
            raw_tail_focus_rmse=("raw_tail_rmse", "mean"),
            raw_tail_focus_r2=("raw_tail_r2", "mean"),
            raw_tail_focus_abs_corr=("raw_tail_corr", lambda x: float(np.nanmean(np.abs(x)))),
            calibrated_tail_focus_rmse=("calibrated_tail_rmse", "mean"),
            calibrated_tail_focus_r2=("calibrated_tail_r2", "mean"),
            calibrated_tail_focus_abs_corr=("calibrated_tail_corr", lambda x: float(np.nanmean(np.abs(x)))),
        )
        .reset_index()
    )
    aggregate.to_csv(output_root / "tail_parameter_sweep_focus_summary.csv", index=False)
    test_rank = aggregate.loc[aggregate["split"].eq("test")].sort_values(
        ["calibrated_tail_focus_r2", "calibrated_tail_focus_abs_corr"], ascending=False
    )
    test_rank.head(30).to_csv(output_root / "top_tail_parameter_cases_test.csv", index=False)

    manifest = {
        "split_root": str(split_root),
        "wing_prior_root": str(wing_prior_root),
        "output_root": str(output_root),
        "cda_m2": cda_m2,
        "case_count": int(case_id),
        "target_residual": "effective_wrench - wing_prior - nominal_body_drag",
        "interpretation_boundary": "Diagnostic only; tail components are model-based priors, not measured isolated tail loads.",
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--wing-prior-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--elevon-max-values", default="25,41")
    parser.add_argument("--rudder-max-values", default="15,25,35")
    parser.add_argument("--fixed-effectiveness-values", default="0.25,0.5,1.0")
    parser.add_argument("--elevon-effectiveness-values", default="0.6,1.2,2.0")
    parser.add_argument("--base-com-x-values", default="-0.15,-0.10,-0.05")
    parser.add_argument("--cda-m2", type=float, default=0.02)
    args = parser.parse_args()
    manifest = run(
        split_root=args.split_root,
        wing_prior_root=args.wing_prior_root,
        output_root=args.output_root,
        elevon_max_values=_parse_csv_floats(args.elevon_max_values),
        rudder_max_values=_parse_csv_floats(args.rudder_max_values),
        fixed_effectiveness_values=_parse_csv_floats(args.fixed_effectiveness_values),
        elevon_effectiveness_values=_parse_csv_floats(args.elevon_effectiveness_values),
        cda_m2=float(args.cda_m2),
        base_com_x_values=_parse_csv_floats(args.base_com_x_values),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
