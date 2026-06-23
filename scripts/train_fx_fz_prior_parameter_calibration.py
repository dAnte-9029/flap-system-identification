#!/usr/bin/env python3
"""Evaluate low-dimensional parameter-style calibrations of fx/fz DeLaurier priors.

The fitted models operate on exported DeLaurier force predictions and phase/frequency
terms. They are intended as a fast diagnostic for whether low-dimensional prior
calibration can explain the flight-log force labels before re-exporting IsaacLab
with modified internal DeLaurier constants.
"""

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

from scripts.train_deployable_wrench_correction_v2 import PHASE_METADATA_COLUMNS, _fit_ridge_frame, build_v2_feature_frame
from scripts.train_fx_fz_correction import SPLITS, TARGETS, _array, _load_split, _metrics_rows

PHASE_FREQ_COLUMNS = (
    "phase_sin_1",
    "phase_cos_1",
    "phase_sin_2",
    "phase_cos_2",
    "flap_frequency_hz",
    "flap_frequency_hz_x_phase_sin_1",
    "flap_frequency_hz_x_phase_cos_1",
)


def _safe_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def _constant_design(prior: np.ndarray, index: int, n: int) -> pd.DataFrame:
    """y_i = a_i * prior_i + b_i."""
    return pd.DataFrame(
        {
            "prior": prior[:, index],
            "bias": np.ones(n, dtype=float),
        }
    )


def _phase_harmonic_design(prior: np.ndarray, features: pd.DataFrame, index: int) -> pd.DataFrame:
    """Parameter-style harmonic calibration.

    This approximates low-dimensional scale/bias/phase-shape calibration:
    y_i = a0 f_prior + b0 + c1 sin(phi) + c2 cos(phi)
          + c3 sin(2phi) + c4 cos(2phi).
    """
    cols = _safe_columns(features, ("phase_sin_1", "phase_cos_1", "phase_sin_2", "phase_cos_2"))
    out = _constant_design(prior, index, len(features))
    for column in cols:
        out[f"bias_{column}"] = features[column].to_numpy(dtype=float)
    return out


def _phase_freq_harmonic_design(prior: np.ndarray, features: pd.DataFrame, index: int) -> pd.DataFrame:
    """Add flapping-frequency modulation to the harmonic calibration."""
    cols = _safe_columns(features, PHASE_FREQ_COLUMNS)
    out = _constant_design(prior, index, len(features))
    for column in cols:
        out[f"bias_{column}"] = features[column].to_numpy(dtype=float)
    return out


def _phase_freq_gain_bias_design(prior: np.ndarray, features: pd.DataFrame, index: int) -> pd.DataFrame:
    """y_i = a_i(phase,freq) f_prior_i + b_i(phase,freq)."""
    cols = _safe_columns(features, PHASE_FREQ_COLUMNS)
    phi = pd.DataFrame({"one": np.ones(len(features), dtype=float)}, index=features.index)
    if cols:
        phi = pd.concat([phi, features.loc[:, cols].copy()], axis=1)
    p = prior[:, index]
    data: dict[str, np.ndarray] = {}
    for column in phi.columns:
        values = phi[column].to_numpy(dtype=float)
        data[f"gain_{column}"] = p * values
        data[f"bias_{column}"] = values
    return pd.DataFrame(data, index=features.index)


def _fit_channel_models(
    train_designs: list[pd.DataFrame],
    y_train: np.ndarray,
    alpha: float,
) -> list[object]:
    return [_fit_ridge_frame(design, y_train[:, [idx]], alpha) for idx, design in enumerate(train_designs)]


def _predict_channel_models(models: list[object], designs: list[pd.DataFrame]) -> np.ndarray:
    return np.column_stack([model.predict(design)[:, 0] for model, design in zip(models, designs)])


def _evaluate(frames: dict[str, pd.DataFrame], model_name: str, predictions: dict[str, np.ndarray]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split, frame in frames.items():
        rows.extend(_metrics_rows(frame, predictions[split], split=split, model=model_name))
    return rows


def _val_rmse(rows: list[dict[str, object]], model_name: str) -> float:
    for row in rows:
        if row["split"] == "val" and row["model"] == model_name and row["target"] == "fx_fz_mean":
            return float(row["rmse"])
    raise ValueError(f"missing validation score for {model_name}")


def _save_predictions(
    *,
    path: Path,
    frames: dict[str, pd.DataFrame],
    raw_prior: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for split, frame in frames.items():
        out_cols = [
            column
            for column in ("timestamp_us", "time_s", "log_id", "segment_id", "cycle_id", *PHASE_METADATA_COLUMNS, "split")
            if column in frame.columns
        ]
        out = frame.loc[:, out_cols].copy()
        for idx, target in enumerate(TARGETS):
            out[f"label_{target}"] = frame[f"label_{target}"].to_numpy(dtype=float)
            out[f"prior_{target}"] = raw_prior[split][:, idx]
            out[f"force_v2_{target}"] = predictions[split][:, idx]
            out[f"force_v2_residual_{target}"] = out[f"label_{target}"] - out[f"force_v2_{target}"]
        out.to_parquet(path / f"{split}_predictions.parquet", index=False)


def run(*, split_root: Path, prior_root: Path, output_root: Path, alphas: tuple[float, ...]) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    frames = {split: _load_split(split_root, prior_root, split) for split in SPLITS}
    features = {split: build_v2_feature_frame(frame)[0] for split, frame in frames.items()}
    y_train = _array(frames["train"], tuple(f"label_{target}" for target in TARGETS))
    prior = {split: _array(frame, tuple(f"prior_{target}" for target in TARGETS)) for split, frame in frames.items()}

    design_builders = {
        "constant_prior_scale_bias": _constant_design,
        "harmonic_bias_calibration": _phase_harmonic_design,
        "phase_freq_bias_calibration": _phase_freq_harmonic_design,
        "phase_freq_gain_bias_calibration": _phase_freq_gain_bias_design,
    }

    all_rows: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    prediction_bank: dict[str, dict[str, np.ndarray]] = {}

    raw_predictions = prior
    raw_rows = _evaluate(frames, "raw_prior", raw_predictions)
    all_rows.extend(raw_rows)
    prediction_bank["raw_prior"] = raw_predictions
    candidates.append({"model": "raw_prior", "family": "baseline", "alpha": np.nan, "n_features": 0, "val_rmse": _val_rmse(raw_rows, "raw_prior")})

    for family, builder in design_builders.items():
        for alpha in alphas:
            train_designs = [builder(prior["train"], features["train"], idx) if family != "constant_prior_scale_bias" else builder(prior["train"], idx, len(features["train"])) for idx in range(len(TARGETS))]
            models = _fit_channel_models(train_designs, y_train, float(alpha))
            predictions: dict[str, np.ndarray] = {}
            for split in SPLITS:
                if family == "constant_prior_scale_bias":
                    designs = [builder(prior[split], idx, len(features[split])) for idx in range(len(TARGETS))]
                else:
                    designs = [builder(prior[split], features[split], idx) for idx in range(len(TARGETS))]
                predictions[split] = _predict_channel_models(models, designs)
            name = f"{family}_alpha_{float(alpha):g}"
            rows = _evaluate(frames, name, predictions)
            all_rows.extend(rows)
            candidates.append(
                {
                    "model": name,
                    "family": family,
                    "alpha": float(alpha),
                    "n_features": int(sum(len(design.columns) for design in train_designs)),
                    "val_rmse": _val_rmse(rows, name),
                }
            )
            prediction_bank[name] = predictions

    metrics = pd.DataFrame(all_rows)
    selection = pd.DataFrame(candidates).sort_values(["val_rmse", "n_features", "model"]).reset_index(drop=True)
    selected_name = str(selection.iloc[0]["model"])
    metrics["is_selected"] = metrics["model"].eq(selected_name)
    metrics.to_csv(output_root / "prior_parameter_calibration_metrics.csv", index=False)
    selection.to_csv(output_root / "prior_parameter_calibration_model_selection.csv", index=False)

    pred_root = output_root / "prediction_parquets"
    _save_predictions(path=pred_root / "selected", frames=frames, raw_prior=prior, predictions=prediction_bank[selected_name])
    for family in design_builders:
        rows = selection.loc[selection["family"].eq(family)]
        if len(rows) > 0:
            _save_predictions(
                path=pred_root / family,
                frames=frames,
                raw_prior=prior,
                predictions=prediction_bank[str(rows.iloc[0]["model"])],
            )

    manifest = {
        "split_root": str(split_root),
        "prior_root": str(prior_root),
        "output_root": str(output_root),
        "targets": list(TARGETS),
        "selected_model": selected_name,
        "selected": selection.iloc[0].replace({np.nan: None}).to_dict(),
        "scope_note": (
            "These models calibrate exported prior predictions with low-dimensional scale, bias, "
            "and phase/frequency terms. They are not full IsaacLab internal DeLaurier parameter re-exports."
        ),
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
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                split_root=args.split_root,
                prior_root=args.prior_root,
                output_root=args.output_root,
                alphas=_parse_alphas(args.alphas),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
