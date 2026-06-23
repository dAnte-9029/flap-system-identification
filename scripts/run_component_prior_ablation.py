#!/usr/bin/env python3
"""Generate wing/tail/body component priors and evaluate whole-vehicle ablations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Mapping

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ISAAC_FLAPPING_SOURCE = Path("/home/zn/IsaacLab/source/flapping_bot")
if str(ISAAC_FLAPPING_SOURCE) not in sys.path:
    sys.path.insert(0, str(ISAAC_FLAPPING_SOURCE))

from flapping_bot.physics.tail_aero import TailAeroCfg, TailAeroModel
from scripts.build_delaurier_residual_split import align_prior_to_samples

TARGET_COLUMNS = ("fx_b", "fy_b", "fz_b", "mx_b", "my_b", "mz_b")
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class ComponentConfig:
    elevon_max_deg: float
    rudder_max_deg: float
    nominal_body_cda_m2: float
    base_com_pos_frd_m: tuple[float, float, float]
    tail_fixed_horizontal_effectiveness: float
    tail_elevon_effectiveness: float
    tail_elevon_alpha_limit_deg: float


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), float(default), dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(float(default)).to_numpy(dtype=float)


def _body_air_velocity_frd(frame: pd.DataFrame) -> np.ndarray:
    required = [f"vehicle_attitude.q[{idx}]" for idx in range(4)] + [
        "vehicle_local_position.vx",
        "vehicle_local_position.vy",
        "vehicle_local_position.vz",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        tas = _numeric(frame, "airspeed_validated.true_airspeed_m_s", 0.0)
        return np.column_stack([tas, np.zeros(len(frame)), np.zeros(len(frame))])

    q0 = _numeric(frame, "vehicle_attitude.q[0]")
    q1 = _numeric(frame, "vehicle_attitude.q[1]")
    q2 = _numeric(frame, "vehicle_attitude.q[2]")
    q3 = _numeric(frame, "vehicle_attitude.q[3]")
    vn = _numeric(frame, "vehicle_local_position.vx") - _numeric(frame, "wind.windspeed_north")
    ve = _numeric(frame, "vehicle_local_position.vy") - _numeric(frame, "wind.windspeed_east")
    vd = _numeric(frame, "vehicle_local_position.vz")

    r00 = 1.0 - 2.0 * (q2 * q2 + q3 * q3)
    r01 = 2.0 * (q1 * q2 - q0 * q3)
    r02 = 2.0 * (q1 * q3 + q0 * q2)
    r10 = 2.0 * (q1 * q2 + q0 * q3)
    r11 = 1.0 - 2.0 * (q1 * q1 + q3 * q3)
    r12 = 2.0 * (q2 * q3 - q0 * q1)
    r20 = 2.0 * (q1 * q3 - q0 * q2)
    r21 = 2.0 * (q2 * q3 + q0 * q1)
    r22 = 1.0 - 2.0 * (q1 * q1 + q2 * q2)

    u_b = r00 * vn + r10 * ve + r20 * vd
    v_b = r01 * vn + r11 * ve + r21 * vd
    w_b = r02 * vn + r12 * ve + r22 * vd
    return np.column_stack([u_b, v_b, w_b])


def _frd_to_flu(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    out[:, 1] *= -1.0
    out[:, 2] *= -1.0
    return out


def _flu_to_frd(values: np.ndarray) -> np.ndarray:
    return _frd_to_flu(values)


def _tail_prior(samples: pd.DataFrame, cfg: ComponentConfig) -> pd.DataFrame:
    v_air_frd = _body_air_velocity_frd(samples)
    w_frd = np.column_stack(
        [
            _numeric(samples, "vehicle_angular_velocity.xyz[0]"),
            _numeric(samples, "vehicle_angular_velocity.xyz[1]"),
            _numeric(samples, "vehicle_angular_velocity.xyz[2]"),
        ]
    )
    v_air_flu = _frd_to_flu(v_air_frd)
    w_flu = _frd_to_flu(w_frd)
    base_com_flu = _frd_to_flu(np.asarray(cfg.base_com_pos_frd_m, dtype=float).reshape(1, 3))

    tail_cfg = TailAeroCfg(
        fixed_horizontal_effectiveness=float(cfg.tail_fixed_horizontal_effectiveness),
        elevon_effectiveness=float(cfg.tail_elevon_effectiveness),
        elevon_alpha_limit_deg=float(cfg.tail_elevon_alpha_limit_deg),
    )
    model = TailAeroModel(tail_cfg, "cpu")
    elevon_max = math.radians(float(cfg.elevon_max_deg))
    rudder_max = math.radians(float(cfg.rudder_max_deg))
    left_rad = np.clip(_numeric(samples, "servo_left_elevon") * elevon_max, -elevon_max, elevon_max)
    right_rad = np.clip(_numeric(samples, "servo_right_elevon") * elevon_max, -elevon_max, elevon_max)
    rudder_rad = np.clip(_numeric(samples, "servo_rudder") * rudder_max, -rudder_max, rudder_max)

    with torch.no_grad():
        force_flu, torque_flu = model.compute_wrench(
            root_lin_vel_b=torch.as_tensor(v_air_flu, dtype=torch.float32),
            root_ang_vel_b=torch.as_tensor(w_flu, dtype=torch.float32),
            left_elevon_rad=torch.as_tensor(left_rad, dtype=torch.float32),
            right_elevon_rad=torch.as_tensor(right_rad, dtype=torch.float32),
            rudder_rad=torch.as_tensor(rudder_rad, dtype=torch.float32),
            base_com_pos_b=torch.as_tensor(np.repeat(base_com_flu, len(samples), axis=0), dtype=torch.float32),
        )
    force_frd = _flu_to_frd(force_flu.cpu().numpy())
    torque_frd = _flu_to_frd(torque_flu.cpu().numpy())
    values = np.column_stack([force_frd, torque_frd])
    return pd.DataFrame(values, columns=TARGET_COLUMNS)


def _body_drag_prior(samples: pd.DataFrame, *, cda_m2: float) -> pd.DataFrame:
    v_air_frd = _body_air_velocity_frd(samples)
    rho = _numeric(samples, "vehicle_air_data.rho", 1.225)
    speed = np.linalg.norm(v_air_frd, axis=1)
    force = -0.5 * rho[:, None] * float(cda_m2) * speed[:, None] * v_air_frd
    values = np.column_stack([force, np.zeros((len(samples), 3), dtype=float)])
    return pd.DataFrame(values, columns=TARGET_COLUMNS)


def _zero_prior(n: int) -> pd.DataFrame:
    return pd.DataFrame(np.zeros((n, len(TARGET_COLUMNS)), dtype=float), columns=TARGET_COLUMNS)


def _sum_priors(*frames: pd.DataFrame) -> pd.DataFrame:
    total = np.zeros((len(frames[0]), len(TARGET_COLUMNS)), dtype=float)
    for frame in frames:
        total += frame.loc[:, TARGET_COLUMNS].to_numpy(dtype=float)
    return pd.DataFrame(total, columns=TARGET_COLUMNS)


def _metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) == 0:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan, "r2": np.nan, "corr": np.nan}
    err = y_pred[mask] - y_true[mask]
    centered = y_true[mask] - float(np.mean(y_true[mask]))
    ss_tot = float(np.sum(centered * centered))
    ss_res = float(np.sum(err * err))
    corr = np.nan
    if int(mask.sum()) > 2 and np.std(y_true[mask]) > 1.0e-12 and np.std(y_pred[mask]) > 1.0e-12:
        corr = float(np.corrcoef(y_true[mask], y_pred[mask])[0, 1])
    return {
        "n": int(mask.sum()),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else np.nan,
        "corr": corr,
    }


def _metrics(samples: pd.DataFrame, pred: pd.DataFrame, *, split: str, variant: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for target in TARGET_COLUMNS:
        row = _metric_row(samples[target].to_numpy(dtype=float), pred[target].to_numpy(dtype=float))
        rows.append({"split": split, "variant": variant, "target": target, **row})
    for group_name, group_targets in {
        "force_mean": ("fx_b", "fy_b", "fz_b"),
        "moment_mean": ("mx_b", "my_b", "mz_b"),
        "all_mean": TARGET_COLUMNS,
    }.items():
        subset = [row for row in rows if row["target"] in group_targets]
        rows.append(
            {
                "split": split,
                "variant": variant,
                "target": group_name,
                "n": min(int(row["n"]) for row in subset),
                "rmse": float(np.nanmean([float(row["rmse"]) for row in subset])),
                "mae": float(np.nanmean([float(row["mae"]) for row in subset])),
                "bias": float(np.nanmean([float(row["bias"]) for row in subset])),
                "r2": float(np.nanmean([float(row["r2"]) for row in subset])),
                "corr": float(np.nanmean([float(row["corr"]) for row in subset])),
            }
        )
    return rows


def _fit_component_model(train_samples: pd.DataFrame, train_components: Mapping[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    params: dict[str, np.ndarray] = {}
    for target in TARGET_COLUMNS:
        columns = [
            train_components["wing"][target].to_numpy(dtype=float),
            train_components["tail"][target].to_numpy(dtype=float),
            train_components["body"][target].to_numpy(dtype=float),
            np.ones(len(train_samples), dtype=float),
        ]
        x = np.column_stack(columns)
        y = train_samples[target].to_numpy(dtype=float)
        mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
        params[target] = np.linalg.lstsq(x[mask], y[mask], rcond=None)[0]
    return params


def _apply_component_model(components: Mapping[str, pd.DataFrame], params: Mapping[str, np.ndarray]) -> pd.DataFrame:
    output = pd.DataFrame(index=components["wing"].index)
    for target in TARGET_COLUMNS:
        coeff = np.asarray(params[target], dtype=float)
        output[target] = (
            coeff[0] * components["wing"][target].to_numpy(dtype=float)
            + coeff[1] * components["tail"][target].to_numpy(dtype=float)
            + coeff[2] * components["body"][target].to_numpy(dtype=float)
            + coeff[3]
        )
    return output.loc[:, TARGET_COLUMNS]


def _metadata(samples: pd.DataFrame) -> pd.DataFrame:
    keep = [column for column in ("dataset_id", "log_id", "segment_id", "time_s", "timestamp_us", "split") if column in samples.columns]
    return samples.loc[:, keep].reset_index(drop=True)


def run(
    *,
    split_root: Path,
    wing_prior_root: Path,
    output_root: Path,
    cfg: ComponentConfig,
    allow_row_order_fallback: bool,
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    prediction_root = output_root / "component_prediction_parquets"
    prediction_root.mkdir(exist_ok=True)

    split_components: dict[str, dict[str, pd.DataFrame]] = {}
    split_samples: dict[str, pd.DataFrame] = {}
    alignment: dict[str, object] = {}
    for split in SPLITS:
        samples = pd.read_parquet(split_root / f"{split}_samples.parquet")
        raw_wing = pd.read_parquet(wing_prior_root / f"{split}_predictions.parquet")
        wing, info = align_prior_to_samples(samples, raw_wing, allow_row_order_fallback=allow_row_order_fallback)
        wing = wing.loc[:, TARGET_COLUMNS].reset_index(drop=True)
        tail = _tail_prior(samples, cfg).reset_index(drop=True)
        body = _body_drag_prior(samples, cda_m2=cfg.nominal_body_cda_m2).reset_index(drop=True)
        split_samples[split] = samples.reset_index(drop=True)
        split_components[split] = {"wing": wing, "tail": tail, "body": body}
        alignment[split] = info

        out = _metadata(samples)
        for name, frame in split_components[split].items():
            for target in TARGET_COLUMNS:
                out[f"{name}_{target}"] = frame[target].to_numpy(dtype=float)
        variants = {
            "wing_only": wing,
            "wing_plus_body": _sum_priors(wing, body),
            "wing_plus_tail": _sum_priors(wing, tail),
            "wing_plus_tail_plus_body": _sum_priors(wing, tail, body),
        }
        for name, frame in variants.items():
            for target in TARGET_COLUMNS:
                out[f"{name}_{target}"] = frame[target].to_numpy(dtype=float)
        out.to_parquet(prediction_root / f"{split}_component_priors.parquet", index=False)

    params = _fit_component_model(split_samples["train"], split_components["train"])
    parameter_rows = []
    for target, coeff in params.items():
        parameter_rows.append(
            {
                "target": target,
                "wing_gain": float(coeff[0]),
                "tail_gain": float(coeff[1]),
                "body_gain": float(coeff[2]),
                "bias": float(coeff[3]),
            }
        )
    pd.DataFrame(parameter_rows).to_csv(output_root / "calibrated_component_parameters.csv", index=False)

    metric_rows: list[dict[str, object]] = []
    for split in SPLITS:
        samples = split_samples[split]
        comps = split_components[split]
        variants = {
            "zero": _zero_prior(len(samples)),
            "wing_only": comps["wing"],
            "wing_plus_body": _sum_priors(comps["wing"], comps["body"]),
            "wing_plus_tail": _sum_priors(comps["wing"], comps["tail"]),
            "wing_plus_tail_plus_body": _sum_priors(comps["wing"], comps["tail"], comps["body"]),
            "calibrated_whole_vehicle_prior": _apply_component_model(comps, params),
        }
        for variant, pred in variants.items():
            metric_rows.extend(_metrics(samples, pred, split=split, variant=variant))
        calibrated = _apply_component_model(comps, params)
        calibrated_out = _metadata(samples)
        for target in TARGET_COLUMNS:
            calibrated_out[target] = calibrated[target].to_numpy(dtype=float)
        calibrated_out.to_parquet(prediction_root / f"{split}_calibrated_whole_vehicle_prior.parquet", index=False)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_root / "component_ablation_metrics.csv", index=False)
    test_summary = metrics.loc[
        (metrics["split"] == "test") & (metrics["target"].isin([*TARGET_COLUMNS, "force_mean", "moment_mean", "all_mean"])),
        ["variant", "target", "n", "rmse", "mae", "bias", "r2", "corr"],
    ].copy()
    test_summary.to_csv(output_root / "test_component_ablation_summary.csv", index=False)

    manifest = {
        "split_root": str(split_root),
        "wing_prior_root": str(wing_prior_root),
        "output_root": str(output_root),
        "config": {
            "elevon_max_deg": cfg.elevon_max_deg,
            "rudder_max_deg": cfg.rudder_max_deg,
            "nominal_body_cda_m2": cfg.nominal_body_cda_m2,
            "base_com_pos_frd_m": list(cfg.base_com_pos_frd_m),
            "tail_fixed_horizontal_effectiveness": cfg.tail_fixed_horizontal_effectiveness,
            "tail_elevon_effectiveness": cfg.tail_elevon_effectiveness,
            "tail_elevon_alpha_limit_deg": cfg.tail_elevon_alpha_limit_deg,
        },
        "frame_convention": {
            "samples_and_outputs": "body FRD, CG-reference effective wrench columns",
            "tail_model_internal": "body FLU; converted to/from FRD with diag(1,-1,-1)",
        },
        "servo_interpretation": "servo_* columns treated as normalized deflection commands and mapped by elevon/rudder limits",
        "alignment": alignment,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_root / "README.md").write_text(
        "# Component Prior Ablation\n\n"
        "This artifact evaluates whole-vehicle component priors against log-derived effective-wrench labels.\n"
        "It does not claim isolated wing, tail, or body load measurements; component terms are model-based priors.\n\n"
        "Key files:\n"
        "- `component_ablation_metrics.csv`: train/val/test channel metrics.\n"
        "- `test_component_ablation_summary.csv`: held-out test summary.\n"
        "- `calibrated_component_parameters.csv`: train-only low-dimensional component coefficients.\n"
        "- `component_prediction_parquets/`: aligned component prior outputs.\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--wing-prior-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--elevon-max-deg", type=float, default=41.0)
    parser.add_argument("--rudder-max-deg", type=float, default=25.0)
    parser.add_argument("--nominal-body-cda-m2", type=float, default=0.02)
    parser.add_argument("--base-com-pos-frd-m", type=float, nargs=3, default=(-0.10, 0.0, 0.0))
    parser.add_argument("--tail-fixed-horizontal-effectiveness", type=float, default=0.5)
    parser.add_argument("--tail-elevon-effectiveness", type=float, default=1.2)
    parser.add_argument("--tail-elevon-alpha-limit-deg", type=float, default=25.0)
    parser.add_argument("--allow-row-order-fallback", action="store_true")
    args = parser.parse_args()

    cfg = ComponentConfig(
        elevon_max_deg=float(args.elevon_max_deg),
        rudder_max_deg=float(args.rudder_max_deg),
        nominal_body_cda_m2=float(args.nominal_body_cda_m2),
        base_com_pos_frd_m=tuple(float(v) for v in args.base_com_pos_frd_m),
        tail_fixed_horizontal_effectiveness=float(args.tail_fixed_horizontal_effectiveness),
        tail_elevon_effectiveness=float(args.tail_elevon_effectiveness),
        tail_elevon_alpha_limit_deg=float(args.tail_elevon_alpha_limit_deg),
    )
    manifest = run(
        split_root=args.split_root,
        wing_prior_root=args.wing_prior_root,
        output_root=args.output_root,
        cfg=cfg,
        allow_row_order_fallback=bool(args.allow_row_order_fallback),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
