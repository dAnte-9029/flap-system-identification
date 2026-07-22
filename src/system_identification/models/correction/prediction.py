"""Keyed, row-order-preserving inference for static correction bundles."""

from __future__ import annotations

import numpy as np
import pandas as pd

from system_identification.models.correction.bundles import StaticCorrectionBundle
from system_identification.models.correction.features import build_mean_design, build_waveform_design
from system_identification.models.correction.specifications import MEAN_WB_TYPES
from system_identification.models.correction.static_models import RidgeSolution


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Prediction schema mismatch; missing columns: {missing}")


def _predict_solution(solution: RidgeSolution, values: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    if tuple(names) != solution.feature_names:
        raise ValueError(
            f"Prediction feature schema mismatch; expected={solution.feature_names}, received={tuple(names)}"
        )
    prediction = values @ solution.coefficients
    if not np.isfinite(prediction).all():
        raise ValueError("Prediction contains non-finite values")
    return prediction


def _base_output(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, ["cycle_id"])
    columns = ["cycle_id"]
    if "timestamp_us" in frame.columns:
        columns.append("timestamp_us")
    return frame.loc[:, columns].reset_index(drop=True).copy()


def _validate_cycle_frame(cycle_frame: pd.DataFrame) -> None:
    _require_columns(cycle_frame, ["cycle_id"])
    if cycle_frame["cycle_id"].duplicated().any():
        raise ValueError("cycle_frame contains duplicate cycle_id values")


def predict_cycle_mean(bundle: StaticCorrectionBundle, cycle_frame: pd.DataFrame) -> pd.DataFrame:
    _validate_cycle_frame(cycle_frame)
    spec = bundle.spec
    component = spec.force_component
    output = _base_output(cycle_frame)
    if spec.model_type == "raw_prior":
        _require_columns(cycle_frame, [f"prior_{component}_mean_n"])
        retained = cycle_frame[f"prior_{component}_mean_n"].to_numpy(dtype=np.float64, copy=False)
        correction = np.zeros(len(cycle_frame), dtype=np.float64)
    elif spec.model_type in MEAN_WB_TYPES:
        if bundle.mean_solution is None:
            raise ValueError("Bundle has no mean solution")
        design = build_mean_design(cycle_frame, spec)
        correction = _predict_solution(bundle.mean_solution, design.values, design.feature_names)
        if spec.mean_prior_retention == 0.0:
            retained = np.zeros(len(cycle_frame), dtype=np.float64)
        else:
            _require_columns(cycle_frame, [f"prior_{component}_mean_n"])
            retained = float(spec.mean_prior_retention) * cycle_frame[f"prior_{component}_mean_n"].to_numpy(
                dtype=np.float64, copy=False
            )
    else:
        raise ValueError(f"Direct cycle-mean prediction is not defined for {spec.model_type}")
    output["predicted_mean_n"] = retained + correction
    output["prior_mean_retained_n"] = retained
    output["mean_correction_n"] = correction
    return output


def predict_waveform(bundle: StaticCorrectionBundle, waveform_frame: pd.DataFrame) -> pd.DataFrame:
    spec = bundle.spec
    component = spec.force_component
    output = _base_output(waveform_frame)
    if spec.model_type == "raw_prior":
        _require_columns(waveform_frame, [f"prior_{component}_waveform_n"])
        retained = waveform_frame[f"prior_{component}_waveform_n"].to_numpy(dtype=np.float64, copy=False)
        correction = np.zeros(len(waveform_frame), dtype=np.float64)
    elif spec.model_type in MEAN_WB_TYPES:
        if bundle.waveform_solution is None:
            raise ValueError("Bundle has no waveform solution")
        design = build_waveform_design(waveform_frame, spec)
        correction = _predict_solution(bundle.waveform_solution, design.values, design.feature_names)
        if spec.waveform_prior_retention == 0.0:
            retained = np.zeros(len(waveform_frame), dtype=np.float64)
        else:
            _require_columns(waveform_frame, [f"prior_{component}_waveform_n"])
            retained = float(spec.waveform_prior_retention) * waveform_frame[
                f"prior_{component}_waveform_n"
            ].to_numpy(dtype=np.float64, copy=False)
    else:
        raise ValueError(f"Direct waveform prediction is not defined for {spec.model_type}")
    output["predicted_waveform_n"] = retained + correction
    output["prior_waveform_retained_n"] = retained
    output["waveform_correction_n"] = correction
    cycle_means = output.groupby("cycle_id", sort=False)["predicted_waveform_n"].mean()
    if len(cycle_means) and float(cycle_means.abs().max()) > 1e-8:
        raise ValueError("Predicted waveform violates the per-cycle zero-mean contract")
    return output


def _validate_mean_join(cycle_frame: pd.DataFrame, waveform_frame: pd.DataFrame) -> None:
    _validate_cycle_frame(cycle_frame)
    _require_columns(waveform_frame, ["cycle_id"])
    available = set(cycle_frame["cycle_id"].astype(str))
    requested = set(waveform_frame["cycle_id"].astype(str))
    missing = sorted(requested - available)
    if missing:
        raise ValueError(f"Waveform rows have missing cycle mean entries: {missing[:5]}")


def _decompose_total(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    prior_retained: np.ndarray,
) -> pd.DataFrame:
    output = _base_output(frame)
    temporary = pd.DataFrame(
        {"cycle_id": frame["cycle_id"].to_numpy(), "prediction": prediction, "prior": prior_retained}
    )
    predicted_mean_map = temporary.groupby("cycle_id", sort=False)["prediction"].mean().to_dict()
    prior_mean_map = temporary.groupby("cycle_id", sort=False)["prior"].mean().to_dict()
    predicted_mean = frame["cycle_id"].map(predicted_mean_map).to_numpy(dtype=np.float64)
    prior_mean = frame["cycle_id"].map(prior_mean_map).to_numpy(dtype=np.float64)
    predicted_wave = prediction - predicted_mean
    prior_wave = prior_retained - prior_mean
    output["prediction_n"] = prediction
    output["predicted_mean_n"] = predicted_mean
    output["predicted_waveform_n"] = predicted_wave
    output["prior_mean_retained_n"] = prior_mean
    output["prior_waveform_retained_n"] = prior_wave
    output["mean_correction_n"] = predicted_mean - prior_mean
    output["waveform_correction_n"] = predicted_wave - prior_wave
    return output


def predict_total(
    bundle: StaticCorrectionBundle,
    cycle_frame: pd.DataFrame,
    waveform_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Predict total force with a strict keyed cycle join and preserved waveform row order."""

    _validate_mean_join(cycle_frame, waveform_frame)
    spec = bundle.spec
    component = spec.force_component
    if spec.model_type == "gain_bias":
        if bundle.waveform_solution is None:
            raise ValueError("gain_bias bundle has no coefficients")
        _require_columns(waveform_frame, [f"prior_{component}_n"])
        prior = waveform_frame[f"prior_{component}_n"].to_numpy(dtype=np.float64, copy=False)
        coefficients = bundle.waveform_solution.coefficients
        prediction = coefficients[0] * prior + coefficients[1]
        return _decompose_total(waveform_frame, prediction, coefficients[0] * prior)
    if spec.model_type == "physical_component_scale":
        if bundle.component_scale is None:
            raise ValueError("physical component bundle has no constrained scale")
        _require_columns(
            waveform_frame,
            [f"prior_{component}_n", f"prior_{component}_normal_component_n"],
        )
        prior = waveform_frame[f"prior_{component}_n"].to_numpy(dtype=np.float64, copy=False)
        normal = waveform_frame[f"prior_{component}_normal_component_n"].to_numpy(dtype=np.float64, copy=False)
        prediction = prior + (bundle.component_scale - 1.0) * normal
        return _decompose_total(waveform_frame, prediction, prior)
    if spec.model_type == "raw_prior":
        _require_columns(waveform_frame, [f"prior_{component}_n"])
        prior = waveform_frame[f"prior_{component}_n"].to_numpy(dtype=np.float64, copy=False)
        return _decompose_total(waveform_frame, prior.copy(), prior)

    mean = predict_cycle_mean(bundle, cycle_frame)
    waveform = predict_waveform(bundle, waveform_frame)
    mean_map = mean.set_index("cycle_id")[
        ["predicted_mean_n", "prior_mean_retained_n", "mean_correction_n"]
    ].to_dict("index")
    output = waveform.copy()
    output["predicted_mean_n"] = waveform_frame["cycle_id"].map(
        {key: value["predicted_mean_n"] for key, value in mean_map.items()}
    ).to_numpy(dtype=np.float64)
    output["prior_mean_retained_n"] = waveform_frame["cycle_id"].map(
        {key: value["prior_mean_retained_n"] for key, value in mean_map.items()}
    ).to_numpy(dtype=np.float64)
    output["mean_correction_n"] = waveform_frame["cycle_id"].map(
        {key: value["mean_correction_n"] for key, value in mean_map.items()}
    ).to_numpy(dtype=np.float64)
    output["prediction_n"] = output["predicted_mean_n"] + output["predicted_waveform_n"]
    ordered = [
        "cycle_id",
        *( ["timestamp_us"] if "timestamp_us" in output.columns else [] ),
        "prediction_n",
        "predicted_mean_n",
        "predicted_waveform_n",
        "prior_mean_retained_n",
        "prior_waveform_retained_n",
        "mean_correction_n",
        "waveform_correction_n",
    ]
    return output.loc[:, ordered]
