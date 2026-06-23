#!/usr/bin/env python3
"""Compare structured fx/fz corrections for a raw DeLaurier force prior."""

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

from scripts.train_deployable_wrench_correction_v2 import (
    CONTROL_FEATURES,
    INTERACTION_FEATURES,
    LATERAL_FEATURES,
    PHASE_METADATA_COLUMNS,
    RATE_FEATURES,
    _fit_ridge_frame,
    build_v2_feature_frame,
)
from scripts.train_fx_fz_correction import (
    SPLITS,
    TARGETS,
    _array,
    _feature_groups,
    _load_split,
    _metrics_rows,
)


COMPACT_COLUMNS = (
    "phase_sin_1",
    "phase_cos_1",
    "phase_sin_2",
    "phase_cos_2",
    "alpha_rad",
    "flap_frequency_hz",
    "true_airspeed_m_s",
    "dynamic_pressure_pa",
)

PHASE_FREQ_COLUMNS = (
    "phase_sin_1",
    "phase_cos_1",
    "phase_sin_2",
    "phase_cos_2",
    "flap_frequency_hz",
    "flap_frequency_hz_x_phase_sin_1",
    "flap_frequency_hz_x_phase_cos_1",
)

PHASE_FREQ_Q_COLUMNS = (
    "phase_sin_1",
    "phase_cos_1",
    "phase_sin_2",
    "phase_cos_2",
    "flap_frequency_hz",
    "flap_frequency_hz_x_phase_sin_1",
    "flap_frequency_hz_x_phase_cos_1",
    "body_rate_q",
    "body_rate_q_x_phase_sin_1",
    "body_rate_q_x_phase_cos_1",
)

PITCH_COLUMNS = (
    "phase_sin_1",
    "phase_cos_1",
    "phase_sin_2",
    "phase_cos_2",
    "alpha_rad",
    "flap_frequency_hz",
    "true_airspeed_m_s",
    "dynamic_pressure_pa",
    "body_rate_q",
    "elevon_sum_proxy",
)


def _present_columns(features: pd.DataFrame, columns: tuple[str, ...]) -> list[str]:
    return [column for column in columns if column in features.columns]


def _with_intercept(features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = pd.DataFrame({"one": np.ones(len(features), dtype=float)}, index=features.index)
    if columns:
        out = pd.concat([out, features.loc[:, columns].copy()], axis=1)
    return out


def _gain_bias_design(phi: pd.DataFrame, prior: np.ndarray, target_index: int) -> pd.DataFrame:
    """Build y = a(phi) * prior_i + b(phi) design for one force channel."""
    prior_i = prior[:, target_index]
    gain = {f"gain_{column}": prior_i * phi[column].to_numpy(dtype=float) for column in phi.columns}
    bias = {f"bias_{column}": phi[column].to_numpy(dtype=float) for column in phi.columns}
    return pd.DataFrame({**gain, **bias}, index=phi.index)


def structured_family_specs(
    features: pd.DataFrame,
    requested_families: tuple[str, ...] | None = None,
) -> dict[str, list[str]]:
    requested = set(requested_families or ())
    specs = {
        "phase_freq_gain_bias": _present_columns(features, PHASE_FREQ_COLUMNS),
        "phase_freq_q_gain_bias": _present_columns(features, PHASE_FREQ_Q_COLUMNS),
        "phase_gain_bias_compact": _present_columns(features, COMPACT_COLUMNS),
        "phase_gain_bias_pitch": _present_columns(features, PITCH_COLUMNS),
    }
    if requested:
        known = set(specs) | {"constant_affine", "dense_deployable_affine"}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"unknown structured families: {unknown}")
        specs = {family: columns for family, columns in specs.items() if family in requested}
    return specs


def _dense_design(features: pd.DataFrame, prior: np.ndarray, columns: list[str]) -> pd.DataFrame:
    base = features.loc[:, columns].copy()
    interactions: dict[str, np.ndarray] = {}
    for idx, target in enumerate(TARGETS):
        for column in columns:
            interactions[f"prior_{target}_x_{column}"] = prior[:, idx] * base[column].to_numpy(dtype=float)
    if interactions:
        base = pd.concat([base, pd.DataFrame(interactions, index=base.index)], axis=1)
    return base


def _fit_direct_model(
    train_designs: list[pd.DataFrame],
    y_train: np.ndarray,
    alpha: float,
) -> list[object]:
    models = []
    for idx, design in enumerate(train_designs):
        models.append(_fit_ridge_frame(design, y_train[:, [idx]], float(alpha)))
    return models


def _predict_direct(models: list[object], designs: list[pd.DataFrame]) -> np.ndarray:
    columns = []
    for model, design in zip(models, designs):
        columns.append(model.predict(design)[:, 0])
    return np.column_stack(columns)


def _fit_residual_model(train_design: pd.DataFrame, y_train: np.ndarray, prior_train: np.ndarray, alpha: float) -> object:
    return _fit_ridge_frame(train_design, y_train - prior_train, float(alpha))


def _evaluate_candidate(
    *,
    frames: dict[str, pd.DataFrame],
    name: str,
    predictions: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split, frame in frames.items():
        rows.extend(_metrics_rows(frame, predictions[split], split=split, model=name))
    return rows


def _selection_score(metrics_rows: list[dict[str, object]], model_name: str) -> float:
    for row in metrics_rows:
        if row["split"] == "val" and row["model"] == model_name and row["target"] == "fx_fz_mean":
            return float(row["rmse"])
    raise ValueError(f"missing validation score for {model_name}")


def run(
    *,
    split_root: Path,
    prior_root: Path,
    output_root: Path,
    alphas: tuple[float, ...],
    families: tuple[str, ...] | None = None,
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    pred_root = output_root / "prediction_parquets"
    pred_root.mkdir(exist_ok=True)

    frames = {split: _load_split(split_root, prior_root, split) for split in SPLITS}
    feature_outputs = {split: build_v2_feature_frame(frame) for split, frame in frames.items()}
    features = {split: output[0] for split, output in feature_outputs.items()}
    feature_specs = {split: output[1] for split, output in feature_outputs.items()}
    y_train = _array(frames["train"], tuple(f"label_{target}" for target in TARGETS))
    prior_train = _array(frames["train"], tuple(f"prior_{target}" for target in TARGETS))

    all_rows: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    prediction_bank: dict[str, dict[str, np.ndarray]] = {}

    raw_predictions = {
        split: _array(frame, tuple(f"prior_{target}" for target in TARGETS)) for split, frame in frames.items()
    }
    raw_rows = _evaluate_candidate(frames=frames, name="raw_prior", predictions=raw_predictions)
    all_rows.extend(raw_rows)
    candidates.append({"model": "raw_prior", "family": "baseline", "alpha": np.nan, "n_features": 0, "val_rmse": _selection_score(raw_rows, "raw_prior")})
    prediction_bank["raw_prior"] = raw_predictions
    requested = set(families or ())
    run_all_families = not requested

    # Constant affine calibration: y_i = a_i f_i^prior + b_i.
    if run_all_families or "constant_affine" in requested:
        phi_constant = {split: _with_intercept(features[split], []) for split in SPLITS}
        for alpha in alphas:
            train_designs = [_gain_bias_design(phi_constant["train"], prior_train, idx) for idx in range(len(TARGETS))]
            models = _fit_direct_model(train_designs, y_train, float(alpha))
            predictions = {}
            for split, frame in frames.items():
                prior = _array(frame, tuple(f"prior_{target}" for target in TARGETS))
                designs = [_gain_bias_design(phi_constant[split], prior, idx) for idx in range(len(TARGETS))]
                predictions[split] = _predict_direct(models, designs)
            name = f"constant_affine_alpha_{float(alpha):g}"
            rows = _evaluate_candidate(frames=frames, name=name, predictions=predictions)
            all_rows.extend(rows)
            candidates.append(
                {
                    "model": name,
                    "family": "constant_affine",
                    "alpha": float(alpha),
                    "n_features": int(sum(len(design.columns) for design in train_designs)),
                    "val_rmse": _selection_score(rows, name),
                }
            )
            prediction_bank[name] = predictions

    structured_specs = structured_family_specs(features["train"], families)
    for family, columns in structured_specs.items():
        phi = {split: _with_intercept(features[split], columns) for split in SPLITS}
        for alpha in alphas:
            train_designs = [_gain_bias_design(phi["train"], prior_train, idx) for idx in range(len(TARGETS))]
            models = _fit_direct_model(train_designs, y_train, float(alpha))
            predictions = {}
            for split, frame in frames.items():
                prior = _array(frame, tuple(f"prior_{target}" for target in TARGETS))
                designs = [_gain_bias_design(phi[split], prior, idx) for idx in range(len(TARGETS))]
                predictions[split] = _predict_direct(models, designs)
            name = f"{family}_alpha_{float(alpha):g}"
            rows = _evaluate_candidate(frames=frames, name=name, predictions=predictions)
            all_rows.extend(rows)
            candidates.append(
                {
                    "model": name,
                    "family": family,
                    "alpha": float(alpha),
                    "n_features": int(sum(len(design.columns) for design in train_designs)),
                    "val_rmse": _selection_score(rows, name),
                    "columns": ",".join(columns),
                }
            )
            prediction_bank[name] = predictions

    dense_groups = _feature_groups(features["train"].columns.tolist())
    dense_columns = dense_groups.get("base+rates+controls+lateral+interactions", [])
    if dense_columns and (run_all_families or "dense_deployable_affine" in requested):
        for alpha in alphas:
            train_design = _dense_design(features["train"], prior_train, dense_columns)
            model = _fit_residual_model(train_design, y_train, prior_train, float(alpha))
            predictions = {}
            for split, frame in frames.items():
                prior = _array(frame, tuple(f"prior_{target}" for target in TARGETS))
                design = _dense_design(features[split], prior, dense_columns)
                predictions[split] = prior + model.predict(design)
            name = f"dense_deployable_affine_alpha_{float(alpha):g}"
            rows = _evaluate_candidate(frames=frames, name=name, predictions=predictions)
            all_rows.extend(rows)
            candidates.append(
                {
                    "model": name,
                    "family": "dense_deployable_affine",
                    "alpha": float(alpha),
                    "n_features": int(len(train_design.columns)),
                    "val_rmse": _selection_score(rows, name),
                    "columns": ",".join(dense_columns),
                }
            )
            prediction_bank[name] = predictions

    metrics = pd.DataFrame(all_rows)
    selection = pd.DataFrame(candidates).sort_values(["val_rmse", "n_features", "model"]).reset_index(drop=True)
    selected_name = str(selection.iloc[0]["model"])
    metrics["is_selected"] = metrics["model"].eq(selected_name)
    metrics.to_csv(output_root / "structured_fx_fz_metrics.csv", index=False)
    selection.to_csv(output_root / "structured_fx_fz_model_selection.csv", index=False)

    def save_predictions(model_name: str, suffix: str) -> None:
        for split, frame in frames.items():
            prior = raw_predictions[split]
            pred = prediction_bank[model_name][split]
            out_cols = [
                column
                for column in (
                    "timestamp_us",
                    "time_s",
                    "log_id",
                    "segment_id",
                    "cycle_id",
                    *PHASE_METADATA_COLUMNS,
                    "split",
                )
                if column in frame.columns
            ]
            out = frame.loc[:, out_cols].copy()
            for idx, target in enumerate(TARGETS):
                out[f"label_{target}"] = frame[f"label_{target}"].to_numpy(dtype=float)
                out[f"prior_{target}"] = prior[:, idx]
                out[f"force_v2_{target}"] = pred[:, idx]
                out[f"force_v2_residual_{target}"] = out[f"label_{target}"] - out[f"force_v2_{target}"]
            out.to_parquet(pred_root / f"{split}_{suffix}_predictions.parquet", index=False)

    save_predictions(selected_name, "selected")
    output_families = (
        "constant_affine",
        "phase_freq_gain_bias",
        "phase_freq_q_gain_bias",
        "phase_gain_bias_compact",
        "phase_gain_bias_pitch",
        "dense_deployable_affine",
    )
    for family in output_families:
        rows = selection[selection["family"].eq(family)]
        if len(rows) > 0:
            save_predictions(str(rows.iloc[0]["model"]), family)

    manifest = {
        "split_root": str(split_root),
        "prior_root": str(prior_root),
        "output_root": str(output_root),
        "targets": list(TARGETS),
        "selected_model": selected_name,
        "selected": selection.iloc[0].replace({np.nan: None}).to_dict(),
        "families": sorted(selection["family"].dropna().unique().tolist()),
        "requested_families": sorted(requested) if requested else None,
        "feature_spec": feature_specs["train"],
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _parse_alphas(text: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--alphas", default="0,0.001,0.01,0.1,1,10,100,1000")
    parser.add_argument("--families", nargs="*", default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                split_root=args.split_root,
                prior_root=args.prior_root,
                output_root=args.output_root,
                alphas=_parse_alphas(args.alphas),
                families=tuple(args.families) if args.families else None,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
