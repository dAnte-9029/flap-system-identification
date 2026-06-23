#!/usr/bin/env python3
"""Sweep bounded tail aerodynamic-center geometry for moment-residual alignment."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

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
from scripts.run_component_prior_ablation import (
    TARGET_COLUMNS,
    _body_air_velocity_frd,
    _body_drag_prior,
    _frd_to_flu,
    _metric_row,
    _numeric,
)

MOMENT_FOCUS = ("mx_b", "my_b", "mz_b")
TAIL_FOCUS = ("fy_b", "mx_b", "my_b", "mz_b")


def _parse_csv_floats(text: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in text.split(",") if item.strip())


def _parse_vec3(text: str) -> tuple[float, float, float]:
    values = _parse_csv_floats(text)
    if len(values) != 3:
        raise ValueError(f"Expected three comma-separated values, got {text!r}.")
    return (float(values[0]), float(values[1]), float(values[2]))


def _corr(y: np.ndarray, x: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(x)
    if int(mask.sum()) < 3 or np.std(y[mask]) <= 1.0e-12 or np.std(x[mask]) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(y[mask], x[mask])[0, 1])


def _offset_x(surface, offset_m: float):
    arm = tuple(float(v) for v in surface.lever_arm_body)
    return replace(surface, lever_arm_body=(arm[0] + float(offset_m), arm[1], arm[2]))


def _scale_y(surface, scale: float):
    arm = tuple(float(v) for v in surface.lever_arm_body)
    return replace(surface, lever_arm_body=(arm[0], arm[1] * float(scale), arm[2]))


def _scale_z(surface, scale: float):
    arm = tuple(float(v) for v in surface.lever_arm_body)
    return replace(surface, lever_arm_body=(arm[0], arm[1], arm[2] * float(scale)))


def _make_tail_cfg(
    *,
    horizontal_x_offset_m: float,
    elevon_x_offset_m: float,
    elevon_y_scale: float,
    vertical_x_offset_m: float,
    vertical_z_scale: float,
    fixed_horizontal_effectiveness: float,
    elevon_effectiveness: float,
    elevon_alpha_limit_deg: float,
    horizontal_tail_q_scale: float,
    horizontal_tail_incidence_bias_deg: float,
) -> TailAeroCfg:
    base = TailAeroCfg(
        fixed_horizontal_effectiveness=float(fixed_horizontal_effectiveness),
        elevon_effectiveness=float(elevon_effectiveness),
        elevon_alpha_limit_deg=float(elevon_alpha_limit_deg),
        horizontal_tail_q_scale=float(horizontal_tail_q_scale),
        horizontal_tail_incidence_bias_deg=float(horizontal_tail_incidence_bias_deg),
    )
    fixed_horizontal = _offset_x(base.fixed_horizontal, horizontal_x_offset_m)
    left_elevon = _scale_y(_offset_x(base.left_elevon, elevon_x_offset_m), elevon_y_scale)
    right_elevon = _scale_y(_offset_x(base.right_elevon, elevon_x_offset_m), elevon_y_scale)
    fixed_vertical = _scale_z(_offset_x(base.fixed_vertical, vertical_x_offset_m), vertical_z_scale)
    rudder = _scale_z(_offset_x(base.rudder, vertical_x_offset_m), vertical_z_scale)
    return replace(
        base,
        fixed_horizontal=fixed_horizontal,
        left_elevon=left_elevon,
        right_elevon=right_elevon,
        fixed_vertical=fixed_vertical,
        rudder=rudder,
    )


def _tail_prior_with_cfg(
    samples: pd.DataFrame,
    *,
    tail_cfg: TailAeroCfg,
    base_com_pos_frd_m: tuple[float, float, float],
    elevon_max_deg: float,
    rudder_max_deg: float,
) -> pd.DataFrame:
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
    base_com_flu = _frd_to_flu(np.asarray(base_com_pos_frd_m, dtype=float).reshape(1, 3))

    elevon_max = math.radians(float(elevon_max_deg))
    rudder_max = math.radians(float(rudder_max_deg))
    left_rad = np.clip(_numeric(samples, "servo_left_elevon") * elevon_max, -elevon_max, elevon_max)
    right_rad = np.clip(_numeric(samples, "servo_right_elevon") * elevon_max, -elevon_max, elevon_max)
    rudder_rad = np.clip(_numeric(samples, "servo_rudder") * rudder_max, -rudder_max, rudder_max)

    model = TailAeroModel(tail_cfg, "cpu")
    with torch.no_grad():
        force_flu, torque_flu = model.compute_wrench(
            root_lin_vel_b=torch.as_tensor(v_air_flu, dtype=torch.float32),
            root_ang_vel_b=torch.as_tensor(w_flu, dtype=torch.float32),
            left_elevon_rad=torch.as_tensor(left_rad, dtype=torch.float32),
            right_elevon_rad=torch.as_tensor(right_rad, dtype=torch.float32),
            rudder_rad=torch.as_tensor(rudder_rad, dtype=torch.float32),
            base_com_pos_b=torch.as_tensor(np.repeat(base_com_flu, len(samples), axis=0), dtype=torch.float32),
        )
    force_frd = _frd_to_flu(force_flu.cpu().numpy())
    torque_frd = _frd_to_flu(torque_flu.cpu().numpy())
    return pd.DataFrame(np.column_stack([force_frd, torque_frd]), columns=TARGET_COLUMNS)


def _load_split(split_root: Path, wing_prior_root: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    samples = pd.read_parquet(split_root / f"{split}_samples.parquet").reset_index(drop=True)
    raw_wing = pd.read_parquet(wing_prior_root / f"{split}_predictions.parquet")
    wing, _ = align_prior_to_samples(samples, raw_wing, allow_row_order_fallback=False)
    wing = wing.loc[:, TARGET_COLUMNS].reset_index(drop=True)
    body = _body_drag_prior(samples, cda_m2=0.02).reset_index(drop=True)
    residual = samples.loc[:, TARGET_COLUMNS].reset_index(drop=True) - wing - body
    return samples, wing, residual


def _metric_rows(
    *,
    residual: pd.DataFrame,
    pred: pd.DataFrame,
    split: str,
    case: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for target in TARGET_COLUMNS:
        y = residual[target].to_numpy(dtype=float)
        x = pred[target].to_numpy(dtype=float)
        metric = _metric_row(y, x)
        rows.append({"split": split, "target": target, **case, **metric, "corr": _corr(y, x)})
    for group_name, group_targets in {
        "moment_mean": MOMENT_FOCUS,
        "tail_focus_mean": TAIL_FOCUS,
    }.items():
        subset = [row for row in rows if row["target"] in group_targets]
        rows.append(
            {
                "split": split,
                "target": group_name,
                **case,
                "n": min(int(row["n"]) for row in subset),
                "rmse": float(np.nanmean([float(row["rmse"]) for row in subset])),
                "mae": float(np.nanmean([float(row["mae"]) for row in subset])),
                "bias": float(np.nanmean([float(row["bias"]) for row in subset])),
                "r2": float(np.nanmean([float(row["r2"]) for row in subset])),
                "corr": float(np.nanmean([abs(float(row["corr"])) for row in subset])),
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, object]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    splits = tuple(item.strip() for item in args.splits.split(",") if item.strip())
    loaded = {split: _load_split(Path(args.split_root), Path(args.wing_prior_root), split) for split in splits}
    base_com = _parse_vec3(args.base_com_pos_frd_m)

    rows: list[dict[str, object]] = []
    surface_rows: list[dict[str, object]] = []
    case_id = 0
    for h_x in _parse_csv_floats(args.horizontal_x_offsets_m):
        for e_x in _parse_csv_floats(args.elevon_x_offsets_m):
            for e_y in _parse_csv_floats(args.elevon_y_scales):
                for v_x in _parse_csv_floats(args.vertical_x_offsets_m):
                    for v_z in _parse_csv_floats(args.vertical_z_scales):
                        case_id += 1
                        tail_cfg = _make_tail_cfg(
                            horizontal_x_offset_m=h_x,
                            elevon_x_offset_m=e_x,
                            elevon_y_scale=e_y,
                            vertical_x_offset_m=v_x,
                            vertical_z_scale=v_z,
                            fixed_horizontal_effectiveness=float(args.fixed_horizontal_effectiveness),
                            elevon_effectiveness=float(args.elevon_effectiveness),
                            elevon_alpha_limit_deg=float(args.elevon_alpha_limit_deg),
                            horizontal_tail_q_scale=float(args.horizontal_tail_q_scale),
                            horizontal_tail_incidence_bias_deg=float(args.horizontal_tail_incidence_bias_deg),
                        )
                        case = {
                            "case_id": case_id,
                            "horizontal_x_offset_m": h_x,
                            "elevon_x_offset_m": e_x,
                            "elevon_y_scale": e_y,
                            "vertical_x_offset_m": v_x,
                            "vertical_z_scale": v_z,
                        }
                        for surface_name in ("fixed_horizontal", "left_elevon", "right_elevon", "fixed_vertical", "rudder"):
                            surface = getattr(tail_cfg, surface_name)
                            surface_rows.append(
                                {
                                    **case,
                                    "surface": surface_name,
                                    "lever_x_m": surface.lever_arm_body[0],
                                    "lever_y_m": surface.lever_arm_body[1],
                                    "lever_z_m": surface.lever_arm_body[2],
                                    "area_m2": surface.area,
                                    "cl_alpha_per_rad": surface.cl_alpha_per_rad,
                                }
                            )
                        for split, (samples, _, residual) in loaded.items():
                            pred = _tail_prior_with_cfg(
                                samples,
                                tail_cfg=tail_cfg,
                                base_com_pos_frd_m=base_com,
                                elevon_max_deg=float(args.elevon_max_deg),
                                rudder_max_deg=float(args.rudder_max_deg),
                            )
                            rows.extend(_metric_rows(residual=residual, pred=pred, split=split, case=case))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_root / "tail_geometry_sweep_metrics.csv", index=False)
    pd.DataFrame(surface_rows).to_csv(output_root / "tail_geometry_sweep_surface_geometry.csv", index=False)

    summary = metrics.loc[metrics["target"].isin(["my_b", "moment_mean", "tail_focus_mean"])].copy()
    summary.to_csv(output_root / "tail_geometry_sweep_focus_summary.csv", index=False)
    for split in splits:
        split_summary = summary.loc[summary["split"].eq(split)].copy()
        split_summary.sort_values(["target", "corr", "r2"], ascending=[True, False, False]).to_csv(
            output_root / f"top_geometry_candidates_{split}_by_target.csv", index=False
        )
        split_summary.loc[split_summary["target"].eq("my_b")].sort_values(["corr", "r2"], ascending=False).head(30).to_csv(
            output_root / f"top_my_b_geometry_candidates_{split}.csv", index=False
        )

    manifest = {
        "split_root": str(args.split_root),
        "wing_prior_root": str(args.wing_prior_root),
        "output_root": str(output_root),
        "splits": splits,
        "base_com_pos_frd_m": base_com,
        "elevon_max_deg": float(args.elevon_max_deg),
        "rudder_max_deg": float(args.rudder_max_deg),
        "fixed_aero_parameters": {
            "fixed_horizontal_effectiveness": float(args.fixed_horizontal_effectiveness),
            "elevon_effectiveness": float(args.elevon_effectiveness),
            "elevon_alpha_limit_deg": float(args.elevon_alpha_limit_deg),
            "horizontal_tail_q_scale": float(args.horizontal_tail_q_scale),
            "horizontal_tail_incidence_bias_deg": float(args.horizontal_tail_incidence_bias_deg),
        },
        "case_count": case_id,
        "target_residual": "effective_wrench - wing_prior - nominal_body_drag",
        "interpretation_boundary": "Geometry diagnostic only; aerodynamic-center offsets are bounded prior-shaping parameters, not measured tail loads.",
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--wing-prior-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--base-com-pos-frd-m", default="-0.12154,0.00541,-0.04298")
    parser.add_argument("--horizontal-x-offsets-m", default="-0.04,-0.02,0,0.02,0.04")
    parser.add_argument("--elevon-x-offsets-m", default="-0.04,-0.02,0,0.02,0.04")
    parser.add_argument("--elevon-y-scales", default="0.8,1.0,1.2")
    parser.add_argument("--vertical-x-offsets-m", default="-0.03,0,0.03")
    parser.add_argument("--vertical-z-scales", default="0.8,1.0,1.2")
    parser.add_argument("--elevon-max-deg", type=float, default=41.0)
    parser.add_argument("--rudder-max-deg", type=float, default=25.0)
    parser.add_argument("--fixed-horizontal-effectiveness", type=float, default=0.5)
    parser.add_argument("--elevon-effectiveness", type=float, default=1.2)
    parser.add_argument("--elevon-alpha-limit-deg", type=float, default=25.0)
    parser.add_argument("--horizontal-tail-q-scale", type=float, default=1.0)
    parser.add_argument("--horizontal-tail-incidence-bias-deg", type=float, default=0.0)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
