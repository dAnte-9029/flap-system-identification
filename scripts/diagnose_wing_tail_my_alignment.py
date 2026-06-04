#!/usr/bin/env python3
"""Diagnose wing-plus-tail structure for effective pitch moment.

This script treats pitch moment as a whole-vehicle effective quantity.  It fixes
the wing application point from an area-weighted quarter-chord prior, sweeps a
simple tail prior over broad interpretable parameters, and fits only train-split
global alignment coefficients for

    my ~= a_w * my_wing_fixed_ac + a_t * my_tail_prior + b.

The fitted parameters are diagnostics, not isolated component load estimates.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ISAAC_FLAPPING_SOURCE = Path("/home/zn/IsaacLab/source/flapping_bot")
if str(ISAAC_FLAPPING_SOURCE) not in sys.path:
    sys.path.insert(0, str(ISAAC_FLAPPING_SOURCE))

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional in tests
    plt = None

def _load_tail_aero_backend():
    try:
        import torch
        from flapping_bot.physics.tail_aero import TailAeroCfg, TailAeroModel
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Tail sweep requires torch and IsaacLab's flapping_bot package. "
            "Run with `conda run -n env_isaaclab ...` on this machine."
        ) from exc
    return torch, TailAeroCfg, TailAeroModel


SPLITS = ("train", "val", "test")


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), float(default), dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(float(default)).to_numpy(dtype=float)


def _frd_to_flu(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    out[:, 1] *= -1.0
    out[:, 2] *= -1.0
    return out


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


@dataclass(frozen=True)
class WingJointParams:
    left_origin_flu_m: tuple[float, float, float] = (0.0, 0.056, 0.0)
    right_origin_flu_m: tuple[float, float, float] = (0.0, -0.056, 0.0)
    left_roll_offset_rad: float = -0.019391
    right_roll_offset_rad: float = 0.019391


@dataclass(frozen=True)
class TailCandidate:
    case_id: int
    source_stage: str
    horizontal_x_offset_m: float = 0.0
    elevon_x_offset_m: float = 0.0
    vertical_x_offset_m: float = 0.0
    elevon_y_scale: float = 1.0
    vertical_z_scale: float = 1.0
    fixed_horizontal_effectiveness: float = 0.5
    elevon_effectiveness: float = 1.2
    horizontal_tail_q_scale: float = 1.0
    horizontal_tail_incidence_bias_deg: float = 0.0
    elevon_alpha_limit_deg: float = 25.0
    elevon_max_deg: float = 41.0
    rudder_max_deg: float = 25.0
    elevon_sign: float = 1.0
    rudder_sign: float = 1.0
    swap_elevons: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "source_stage": self.source_stage,
            "horizontal_x_offset_m": self.horizontal_x_offset_m,
            "elevon_x_offset_m": self.elevon_x_offset_m,
            "vertical_x_offset_m": self.vertical_x_offset_m,
            "elevon_y_scale": self.elevon_y_scale,
            "vertical_z_scale": self.vertical_z_scale,
            "fixed_horizontal_effectiveness": self.fixed_horizontal_effectiveness,
            "elevon_effectiveness": self.elevon_effectiveness,
            "horizontal_tail_q_scale": self.horizontal_tail_q_scale,
            "horizontal_tail_incidence_bias_deg": self.horizontal_tail_incidence_bias_deg,
            "elevon_alpha_limit_deg": self.elevon_alpha_limit_deg,
            "elevon_max_deg": self.elevon_max_deg,
            "rudder_max_deg": self.rudder_max_deg,
            "elevon_sign": self.elevon_sign,
            "rudder_sign": self.rudder_sign,
            "swap_elevons": bool(self.swap_elevons),
        }


def _rotation_x(angle_rad: np.ndarray | float) -> np.ndarray:
    angle = np.asarray(angle_rad, dtype=float)
    ca = np.cos(angle)
    sa = np.sin(angle)
    out = np.zeros(angle.shape + (3, 3), dtype=float)
    out[..., 0, 0] = 1.0
    out[..., 1, 1] = ca
    out[..., 1, 2] = -sa
    out[..., 2, 1] = sa
    out[..., 2, 2] = ca
    return out


def pitch_moment_from_force(arm_frd_m: np.ndarray, force_frd_n: np.ndarray) -> np.ndarray:
    """Return FRD pitch moment from row-wise arm and force.

    FRD cross product gives ``M_y = r_z F_x - r_x F_z``.
    """
    arm = np.asarray(arm_frd_m, dtype=float)
    force = np.asarray(force_frd_n, dtype=float)
    if arm.shape[-1] != 3 or force.shape[-1] != 3:
        raise ValueError("arm and force must have last dimension 3")
    return arm[..., 2] * force[..., 0] - arm[..., 0] * force[..., 2]


def load_area_weighted_quarter_chord_link_point(csv_path: Path) -> dict[str, float]:
    geom = pd.read_csv(csv_path)
    required = {"x_mid_m", "c_m", "dhat"}
    if not required.issubset(geom.columns):
        raise ValueError(f"{csv_path} must contain {sorted(required)}")
    x = geom["x_mid_m"].to_numpy(dtype=float)
    c = geom["c_m"].to_numpy(dtype=float)
    dhat = geom["dhat"].to_numpy(dtype=float)
    if len(x) < 2:
        raise ValueError("wing geometry CSV must contain at least two rows")
    dx = float(np.median(np.diff(np.sort(x))))
    weights = c * dx
    weights = weights / np.clip(np.sum(weights), 1.0e-12, None)
    span_center_m = float(np.sum(weights * x))
    # Current IsaacLab convention: link +x is chordwise; dhat=0.25 makes
    # the area-weighted quarter-chord point zero relative to the pitching axis.
    quarter_chord_x_m = float(np.sum(weights * ((dhat - 0.25) * c)))
    return {
        "span_center_m": span_center_m,
        "quarter_chord_x_m": quarter_chord_x_m,
        "area_m2_one_wing": float(np.sum(c * dx)),
        "dx_m": dx,
    }


def _flu_to_frd_np(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    out[..., 1] *= -1.0
    out[..., 2] *= -1.0
    return out


def fixed_wing_ac_arm_frd(
    samples: pd.DataFrame,
    *,
    cg_frd_m: np.ndarray,
    wing_geom_csv: Path,
    joint_params: WingJointParams,
) -> np.ndarray:
    q = pd.to_numeric(samples["wing_stroke_angle_rad"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    geom = load_area_weighted_quarter_chord_link_point(wing_geom_csv)
    span_center = geom["span_center_m"]
    qcx = geom["quarter_chord_x_m"]

    left_origin = np.asarray(joint_params.left_origin_flu_m, dtype=float)
    right_origin = np.asarray(joint_params.right_origin_flu_m, dtype=float)
    left_link = np.array([qcx, span_center, 0.0], dtype=float)
    right_link = np.array([qcx, -span_center, 0.0], dtype=float)

    left_rot = _rotation_x(q + float(joint_params.left_roll_offset_rad))
    right_rot = _rotation_x(q + float(joint_params.right_roll_offset_rad))
    left_point = left_origin.reshape(1, 3) + np.einsum("nij,j->ni", left_rot, left_link)
    right_point = right_origin.reshape(1, 3) + np.einsum("nij,j->ni", right_rot, right_link)
    equivalent_point_flu = 0.5 * (left_point + right_point)
    equivalent_point_frd = _flu_to_frd_np(equivalent_point_flu)
    return equivalent_point_frd - np.asarray(cg_frd_m, dtype=float).reshape(1, 3)


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan, "r2": np.nan, "corr": np.nan}
    err = pred[mask] - y[mask]
    ss_tot = float(np.sum((y[mask] - float(np.mean(y[mask]))) ** 2))
    ss_res = float(np.sum(err * err))
    corr = np.nan
    if n > 2 and np.std(y[mask]) > 1.0e-12 and np.std(pred[mask]) > 1.0e-12:
        corr = float(np.corrcoef(y[mask], pred[mask])[0, 1])
    return {
        "n": n,
        "rmse": float(np.sqrt(np.mean(err * err))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else np.nan,
        "corr": corr,
    }


def fit_train_gain_bias(train_y: np.ndarray, train_x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(train_x, dtype=float)
    y = np.asarray(train_y, dtype=float)
    design = np.column_stack([x, np.ones(len(x), dtype=float)])
    mask = np.isfinite(y) & np.all(np.isfinite(design), axis=1)
    if int(mask.sum()) < 2:
        return (0.0, float(np.nanmean(y)))
    gain, bias = np.linalg.lstsq(design[mask], y[mask], rcond=None)[0]
    return (float(gain), float(bias))


def _fit_joint(train_y: np.ndarray, train_wing: np.ndarray, train_tail: np.ndarray, alpha: float) -> np.ndarray:
    y = np.asarray(train_y, dtype=float)
    design = np.column_stack([train_wing, train_tail, np.ones(len(y), dtype=float)])
    mask = np.isfinite(y) & np.all(np.isfinite(design), axis=1)
    x = design[mask]
    yy = y[mask]
    if float(alpha) <= 0.0:
        return np.linalg.lstsq(x, yy, rcond=None)[0]
    reg = np.diag([float(alpha), float(alpha), 0.0])
    return np.linalg.solve(x.T @ x + reg, x.T @ yy)


def _offset_x(surface, offset_m: float):
    arm = tuple(float(v) for v in surface.lever_arm_body)
    return replace(surface, lever_arm_body=(arm[0] + float(offset_m), arm[1], arm[2]))


def _scale_y(surface, scale: float):
    arm = tuple(float(v) for v in surface.lever_arm_body)
    return replace(surface, lever_arm_body=(arm[0], arm[1] * float(scale), arm[2]))


def _scale_z(surface, scale: float):
    arm = tuple(float(v) for v in surface.lever_arm_body)
    return replace(surface, lever_arm_body=(arm[0], arm[1], arm[2] * float(scale)))


def _make_tail_cfg(candidate: TailCandidate) -> TailAeroCfg:
    _torch, TailAeroCfg, _TailAeroModel = _load_tail_aero_backend()
    base = TailAeroCfg(
        fixed_horizontal_effectiveness=float(candidate.fixed_horizontal_effectiveness),
        elevon_effectiveness=float(candidate.elevon_effectiveness),
        elevon_alpha_limit_deg=float(candidate.elevon_alpha_limit_deg),
        horizontal_tail_q_scale=float(candidate.horizontal_tail_q_scale),
        horizontal_tail_incidence_bias_deg=float(candidate.horizontal_tail_incidence_bias_deg),
    )
    fixed_horizontal = _offset_x(base.fixed_horizontal, candidate.horizontal_x_offset_m)
    left_elevon = _scale_y(_offset_x(base.left_elevon, candidate.elevon_x_offset_m), candidate.elevon_y_scale)
    right_elevon = _scale_y(_offset_x(base.right_elevon, candidate.elevon_x_offset_m), candidate.elevon_y_scale)
    fixed_vertical = _scale_z(_offset_x(base.fixed_vertical, candidate.vertical_x_offset_m), candidate.vertical_z_scale)
    rudder = _scale_z(_offset_x(base.rudder, candidate.vertical_x_offset_m), candidate.vertical_z_scale)
    return replace(
        base,
        fixed_horizontal=fixed_horizontal,
        left_elevon=left_elevon,
        right_elevon=right_elevon,
        fixed_vertical=fixed_vertical,
        rudder=rudder,
    )


def _tail_my_prior(samples: pd.DataFrame, candidate: TailCandidate, cg_frd_m: tuple[float, float, float]) -> np.ndarray:
    torch, _TailAeroCfg, TailAeroModel = _load_tail_aero_backend()
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
    base_com_flu = _frd_to_flu(np.asarray(cg_frd_m, dtype=float).reshape(1, 3))

    elevon_max = math.radians(float(candidate.elevon_max_deg))
    rudder_max = math.radians(float(candidate.rudder_max_deg))
    left_cmd = _numeric(samples, "servo_left_elevon")
    right_cmd = _numeric(samples, "servo_right_elevon")
    if candidate.swap_elevons:
        left_cmd, right_cmd = right_cmd, left_cmd
    left_rad = np.clip(float(candidate.elevon_sign) * left_cmd * elevon_max, -elevon_max, elevon_max)
    right_rad = np.clip(float(candidate.elevon_sign) * right_cmd * elevon_max, -elevon_max, elevon_max)
    rudder_rad = np.clip(float(candidate.rudder_sign) * _numeric(samples, "servo_rudder") * rudder_max, -rudder_max, rudder_max)

    model = TailAeroModel(_make_tail_cfg(candidate), "cpu")
    with torch.no_grad():
        _force_flu, torque_flu = model.compute_wrench(
            root_lin_vel_b=torch.as_tensor(v_air_flu, dtype=torch.float32),
            root_ang_vel_b=torch.as_tensor(w_flu, dtype=torch.float32),
            left_elevon_rad=torch.as_tensor(left_rad, dtype=torch.float32),
            right_elevon_rad=torch.as_tensor(right_rad, dtype=torch.float32),
            rudder_rad=torch.as_tensor(rudder_rad, dtype=torch.float32),
            base_com_pos_b=torch.as_tensor(np.repeat(base_com_flu, len(samples), axis=0), dtype=torch.float32),
        )
    torque_frd = _frd_to_flu(torque_flu.cpu().numpy())
    return torque_frd[:, 1]


def _read_split_inputs(
    *,
    split_root: Path,
    force_correction_root: Path,
    split: str,
    wing_geom_csv: Path,
    cg_frd_m: np.ndarray,
    joint_params: WingJointParams,
) -> pd.DataFrame:
    samples = pd.read_parquet(split_root / f"{split}_samples.parquet").reset_index(drop=True)
    force = pd.read_parquet(force_correction_root / f"{split}_selected_predictions.parquet").reset_index(drop=True)
    if len(samples) != len(force):
        raise ValueError(f"{split}: samples/predictions row mismatch {len(samples)} != {len(force)}")
    arm = fixed_wing_ac_arm_frd(samples, cg_frd_m=cg_frd_m, wing_geom_csv=wing_geom_csv, joint_params=joint_params)
    force_frd = np.column_stack(
        [
            pd.to_numeric(force["force_v2_fx_b"], errors="coerce").to_numpy(dtype=float),
            np.zeros(len(force), dtype=float),
            pd.to_numeric(force["force_v2_fz_b"], errors="coerce").to_numpy(dtype=float),
        ]
    )
    out = samples.copy()
    out["fx_corr"] = force_frd[:, 0]
    out["fz_corr"] = force_frd[:, 2]
    out["wing_fixed_ac_rx_m"] = arm[:, 0]
    out["wing_fixed_ac_ry_m"] = arm[:, 1]
    out["wing_fixed_ac_rz_m"] = arm[:, 2]
    out["my_wing_fixed_ac"] = pitch_moment_from_force(arm, force_frd)
    return out


def _parse_vec3(text: str) -> tuple[float, float, float]:
    values = [float(v.strip()) for v in text.split(",") if v.strip()]
    if len(values) != 3:
        raise ValueError(f"expected 3 comma-separated values, got {text!r}")
    return (values[0], values[1], values[2])


def _downsample(frame: pd.DataFrame, stride: int) -> pd.DataFrame:
    if int(stride) <= 1:
        return frame
    if "log_id" not in frame.columns:
        return frame.iloc[:: int(stride)].reset_index(drop=True)
    parts = [group.iloc[:: int(stride)] for _, group in frame.groupby("log_id", sort=False)]
    return pd.concat(parts, axis=0).reset_index(drop=True)


def _parameter_ranges() -> dict[str, list[object]]:
    return {
        "horizontal_x_offset_m": [-0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12],
        "elevon_x_offset_m": [-0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12],
        "vertical_x_offset_m": [-0.10, -0.05, 0.0, 0.05, 0.10],
        "elevon_y_scale": [0.5, 0.75, 1.0, 1.25, 1.5],
        "vertical_z_scale": [0.5, 0.75, 1.0, 1.25, 1.5],
        "fixed_horizontal_effectiveness": [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0],
        "elevon_effectiveness": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
        "horizontal_tail_q_scale": [0.25, 0.5, 1.0, 1.5, 2.0, 3.0],
        "horizontal_tail_incidence_bias_deg": [-20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0],
        "elevon_alpha_limit_deg": [15.0, 20.0, 25.0, 35.0, 45.0],
        "elevon_max_deg": [25.0, 35.0, 41.0, 50.0, 60.0],
        "rudder_max_deg": [15.0, 25.0, 35.0, 45.0],
        "elevon_sign": [-1.0, 1.0],
        "rudder_sign": [-1.0, 1.0],
        "swap_elevons": [False, True],
    }


def _candidate_from_updates(case_id: int, stage: str, updates: dict[str, object]) -> TailCandidate:
    return TailCandidate(case_id=case_id, source_stage=stage, **updates)


def generate_tail_candidates(search_stage: str, max_random_cases: int, seed: int) -> list[TailCandidate]:
    ranges = _parameter_ranges()
    candidates: list[TailCandidate] = []
    seen: set[tuple[tuple[str, object], ...]] = set()

    def add(stage: str, updates: dict[str, object]) -> None:
        key = tuple(sorted(updates.items()))
        if key in seen:
            return
        seen.add(key)
        candidates.append(_candidate_from_updates(len(candidates) + 1, stage, updates))

    add("nominal", {})
    for name, values in ranges.items():
        for value in values:
            add(f"one_at_a_time_{name}", {name: value})

    if search_stage == "smoke":
        return candidates[:80]

    rng = np.random.default_rng(int(seed))
    names = list(ranges.keys())
    random_count = int(max_random_cases)
    if search_stage == "quick":
        random_count = min(random_count, 300)
    for _ in range(random_count):
        updates = {name: ranges[name][int(rng.integers(0, len(ranges[name])))] for name in names}
        add("random_broad", updates)
    return candidates


def _evaluate_baselines(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    train_mean = float(np.nanmean(frames["train"]["my_b"].to_numpy(dtype=float)))
    wing_gain, wing_bias = fit_train_gain_bias(frames["train"]["my_b"], frames["train"]["my_wing_fixed_ac"])
    for split, frame in frames.items():
        y = frame["my_b"].to_numpy(dtype=float)
        variants = {
            "bias_only_train_mean": np.full(len(frame), train_mean, dtype=float),
            "wing_fixed_ac_raw": frame["my_wing_fixed_ac"].to_numpy(dtype=float),
            "wing_fixed_ac_train_gain_bias": wing_gain * frame["my_wing_fixed_ac"].to_numpy(dtype=float) + wing_bias,
        }
        for variant, pred in variants.items():
            rows.append({"split": split, "case_id": 0, "variant": variant, "target": "my_b", **metric_row(y, pred)})
    return pd.DataFrame(rows)


def _candidate_metrics(
    *,
    candidate: TailCandidate,
    frames: dict[str, pd.DataFrame],
    cg_frd_m: tuple[float, float, float],
    alphas: Iterable[float],
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    tail = {split: _tail_my_prior(frame, candidate, cg_frd_m) for split, frame in frames.items()}
    train = frames["train"]
    y_train = train["my_b"].to_numpy(dtype=float)
    wing_train = train["my_wing_fixed_ac"].to_numpy(dtype=float)
    tail_train = tail["train"]

    rows: list[dict[str, object]] = []
    predictions: dict[str, np.ndarray] = {}
    cand = candidate.to_dict()
    for alpha in alphas:
        coeff = _fit_joint(y_train, wing_train, tail_train, float(alpha))
        model_name = f"joint_wing_tail_alpha_{float(alpha):g}"
        for split, frame in frames.items():
            pred = coeff[0] * frame["my_wing_fixed_ac"].to_numpy(dtype=float) + coeff[1] * tail[split] + coeff[2]
            rows.append(
                {
                    **cand,
                    "split": split,
                    "variant": model_name,
                    "target": "my_b",
                    "alpha": float(alpha),
                    "a_w": float(coeff[0]),
                    "a_t": float(coeff[1]),
                    "bias_Nm": float(coeff[2]),
                    **metric_row(frame["my_b"].to_numpy(dtype=float), pred),
                }
            )
            if split == "test" and float(alpha) == 0.0:
                predictions["test_joint_alpha_0"] = pred

    tail_gain, tail_bias = fit_train_gain_bias(y_train, tail_train)
    for split, frame in frames.items():
        y = frame["my_b"].to_numpy(dtype=float)
        variants = {
            "tail_raw": tail[split],
            "tail_train_gain_bias": tail_gain * tail[split] + tail_bias,
            "wing_plus_tail_raw": frame["my_wing_fixed_ac"].to_numpy(dtype=float) + tail[split],
        }
        for variant, pred in variants.items():
            rows.append({**cand, "split": split, "variant": variant, "target": "my_b", **metric_row(y, pred)})
    return rows, tail


def _select_top(metrics: pd.DataFrame, top_k: int) -> pd.DataFrame:
    val = metrics.query("split == 'val' and variant.str.startswith('joint_wing_tail')", engine="python").copy()
    if val.empty:
        raise ValueError("no validation joint metrics found")
    val["coeff_sane"] = (val["a_w"].abs() <= 3.0) & (val["a_t"].abs() <= 10.0)
    sane = val[val["coeff_sane"]].copy()
    ranked = sane if not sane.empty else val
    ranked = ranked.sort_values(["rmse", "corr", "case_id"], ascending=[True, False, True])
    ranked = ranked.drop_duplicates(subset=["case_id"], keep="first").head(int(top_k))
    return ranked.reset_index(drop=True)


def _write_figures(output_root: Path, selected_predictions: pd.DataFrame) -> None:
    if plt is None:
        return
    fig_dir = output_root / "figures"
    fig_dir.mkdir(exist_ok=True)
    test = selected_predictions[selected_predictions["split"].eq("test")].copy()
    if test.empty:
        return
    log_id = test["log_id"].iloc[0] if "log_id" in test.columns else None
    sub = test[test["log_id"].eq(log_id)].copy() if log_id is not None else test.copy()
    sub = sub.iloc[: min(3000, len(sub))]
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(sub["time_s"], sub["my_b"], label="label my_b", lw=1.2)
    ax.plot(sub["time_s"], sub["my_wing_fixed_ac"], label="wing fixed AC", lw=1.0)
    ax.plot(sub["time_s"], sub["my_tail_selected"], label="tail prior", lw=1.0)
    ax.plot(sub["time_s"], sub["my_joint_selected"], label="selected wing+tail", lw=1.2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("My (N m)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "test_my_timeseries_selected.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    ax.scatter(test["my_b"], test["my_joint_selected"], s=2, alpha=0.25)
    ax.set_xlabel("label my_b (N m)")
    ax.set_ylabel("selected prediction (N m)")
    fig.tight_layout()
    fig.savefig(fig_dir / "test_my_scatter_selected.png", dpi=180)
    plt.close(fig)


def _write_summary_figure(output_root: Path, baseline: pd.DataFrame, full_metrics: pd.DataFrame) -> None:
    if plt is None:
        return
    fig_dir = output_root / "figures"
    fig_dir.mkdir(exist_ok=True)
    test_baseline = baseline.query("split == 'test'").copy()
    test_joint = full_metrics.query("split == 'test' and variant.str.startswith('joint_wing_tail')", engine="python").copy()
    rows = []
    for variant in ("bias_only_train_mean", "wing_fixed_ac_raw", "wing_fixed_ac_train_gain_bias"):
        subset = test_baseline[test_baseline["variant"].eq(variant)]
        if not subset.empty:
            rows.append((variant, float(subset["rmse"].iloc[0]), float(subset["corr"].iloc[0]) if pd.notna(subset["corr"].iloc[0]) else 0.0))
    if not test_joint.empty:
        best = test_joint.sort_values(["rmse", "corr"], ascending=[True, False]).iloc[0]
        rows.append(("selected wing+tail", float(best["rmse"]), float(best["corr"])))
    if not rows:
        return
    labels = [row[0] for row in rows]
    rmse = [row[1] for row in rows]
    corr = [row[2] for row in rows]
    x = np.arange(len(rows))
    fig, ax1 = plt.subplots(figsize=(8.5, 3.8))
    ax1.bar(x - 0.18, rmse, width=0.36, label="RMSE", color="#4C78A8")
    ax1.set_ylabel("RMSE (N m)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, corr, width=0.36, label="corr", color="#F58518")
    ax2.set_ylabel("corr")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "parameter_sensitivity_top_candidates.png", dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, object]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cg_frd_m = np.asarray(_parse_vec3(args.cg_frd_m), dtype=float)
    joint_params = WingJointParams()

    full_frames = {
        split: _read_split_inputs(
            split_root=Path(args.split_root),
            force_correction_root=Path(args.force_correction_root),
            split=split,
            wing_geom_csv=Path(args.wing_geom_csv),
            cg_frd_m=cg_frd_m,
            joint_params=joint_params,
        )
        for split in SPLITS
    }
    search_frames = {split: _downsample(frame, int(args.search_stride)) for split, frame in full_frames.items()}

    baseline_metrics = _evaluate_baselines(full_frames)
    baseline_metrics.to_csv(output_root / "baseline_metrics.csv", index=False)

    candidates = generate_tail_candidates(args.search_stage, int(args.max_random_cases), int(args.seed))
    alphas = [float(v) for v in args.alphas.split(",") if v.strip()]
    search_rows: list[dict[str, object]] = []
    for idx, candidate in enumerate(candidates, start=1):
        rows, _tail = _candidate_metrics(candidate=candidate, frames=search_frames, cg_frd_m=tuple(cg_frd_m), alphas=alphas)
        search_rows.extend(rows)
        if idx % 100 == 0:
            print(f"evaluated {idx}/{len(candidates)} search candidates", flush=True)
    search_metrics = pd.DataFrame(search_rows)
    search_metrics.to_csv(output_root / "wing_tail_my_alignment_search_metrics.csv", index=False)

    top = _select_top(search_metrics, int(args.top_k_full))
    top.to_csv(output_root / "top_val_candidates.csv", index=False)

    full_rows: list[dict[str, object]] = []
    selected_predictions: pd.DataFrame | None = None
    for _, row in top.iterrows():
        case_id = int(row["case_id"])
        candidate = next(c for c in candidates if c.case_id == case_id)
        rows, tail = _candidate_metrics(candidate=candidate, frames=full_frames, cg_frd_m=tuple(cg_frd_m), alphas=alphas)
        full_rows.extend(rows)
        if selected_predictions is None:
            alpha = float(row["alpha"])
            coeff = _fit_joint(
                full_frames["train"]["my_b"].to_numpy(dtype=float),
                full_frames["train"]["my_wing_fixed_ac"].to_numpy(dtype=float),
                tail["train"],
                alpha,
            )
            outs = []
            for split, frame in full_frames.items():
                out = frame.loc[:, [c for c in ("time_s", "timestamp_us", "log_id", "segment_id", "split") if c in frame.columns]].copy()
                out["split"] = split
                out["my_b"] = frame["my_b"].to_numpy(dtype=float)
                out["my_wing_fixed_ac"] = frame["my_wing_fixed_ac"].to_numpy(dtype=float)
                out["my_tail_selected"] = tail[split]
                out["my_joint_selected"] = coeff[0] * out["my_wing_fixed_ac"].to_numpy(dtype=float) + coeff[1] * tail[split] + coeff[2]
                out["selected_a_w"] = float(coeff[0])
                out["selected_a_t"] = float(coeff[1])
                out["selected_bias_Nm"] = float(coeff[2])
                out["selected_case_id"] = case_id
                outs.append(out)
            selected_predictions = pd.concat(outs, axis=0).reset_index(drop=True)

    full_metrics = pd.DataFrame(full_rows)
    full_metrics.to_csv(output_root / "wing_tail_my_alignment_metrics.csv", index=False)
    top_test = full_metrics.query("split == 'test'").sort_values(["rmse", "corr"], ascending=[True, False])
    if "case_id" in top_test.columns:
        top_test = top_test.drop_duplicates(subset=["case_id"], keep="first")
    top_test.to_csv(output_root / "top_test_candidates.csv", index=False)
    if selected_predictions is not None:
        selected_predictions.to_parquet(output_root / "selected_wing_tail_my_predictions.parquet", index=False)
        for split, frame in selected_predictions.groupby("split", sort=False):
            frame.to_parquet(output_root / f"{split}_selected_wing_tail_my_predictions.parquet", index=False)
        _write_figures(output_root, selected_predictions)
    _write_summary_figure(output_root, baseline_metrics, full_metrics)

    manifest = {
        "split_root": str(args.split_root),
        "force_correction_root": str(args.force_correction_root),
        "output_root": str(output_root),
        "wing_geom_csv": str(args.wing_geom_csv),
        "cg_frd_m": cg_frd_m.tolist(),
        "search_stage": args.search_stage,
        "search_stride": int(args.search_stride),
        "candidate_count": len(candidates),
        "top_k_full": int(args.top_k_full),
        "alphas": alphas,
        "selection_rule": "validation my_b RMSE, sane coefficient filter if available, corr tie-break",
        "claim_boundary": "diagnostic whole-vehicle effective pitch moment alignment; not isolated wing or tail load measurement",
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_readme(output_root, manifest, baseline_metrics, full_metrics)
    return manifest


def _write_readme(output_root: Path, manifest: dict[str, object], baseline: pd.DataFrame, full_metrics: pd.DataFrame) -> None:
    test_baseline = baseline.query("split == 'test'")[["variant", "rmse", "r2", "corr", "bias"]]
    test_top = full_metrics.query("split == 'test' and variant.str.startswith('joint_wing_tail')", engine="python").sort_values(
        ["rmse", "corr"], ascending=[True, False]
    )[["case_id", "variant", "rmse", "r2", "corr", "bias", "a_w", "a_t", "bias_Nm"]].head(10)
    lines = [
        "# Wing-Tail My Alignment Sweep",
        "",
        "This artifact diagnoses whether a fixed quarter-chord wing moment prior plus a swept simple tail prior can explain held-out effective pitch moment.",
        "",
        "Interpretation boundary: component terms are model priors and train-only alignment diagnostics, not isolated measured wing or tail loads.",
        "",
        "## Manifest",
        "",
        "```json",
        json.dumps(manifest, indent=2),
        "```",
        "",
        "## Test Baselines",
        "",
        "```text",
        test_baseline.to_string(index=False),
        "```",
        "",
        "## Top Test Joint Candidates",
        "",
        "```text",
        test_top.to_string(index=False),
        "```",
        "",
    ]
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--force-correction-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--wing-geom-csv", type=Path, default=Path("/home/zn/IsaacLab/outputs_DeLaurier/right_wing_te_fit_poly5_gap50.csv"))
    parser.add_argument("--cg-frd-m", default="-0.12154,0.00541,-0.04298")
    parser.add_argument("--search-stage", choices=("smoke", "quick", "full"), default="quick")
    parser.add_argument("--search-stride", type=int, default=20)
    parser.add_argument("--max-random-cases", type=int, default=1200)
    parser.add_argument("--top-k-full", type=int, default=20)
    parser.add_argument("--alphas", default="0,0.01,0.1,1,10")
    parser.add_argument("--seed", type=int, default=20260604)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
