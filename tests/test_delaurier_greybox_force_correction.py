import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.train_delaurier_greybox_force_correction import (
    FORCE_TARGETS,
    CorrectionModel,
    FeatureSpec,
    build_feature_frame,
    dataframe_to_markdown,
)


def test_build_feature_frame_derives_harmonics_conditions_and_interactions() -> None:
    frame = pd.DataFrame(
        {
            "phase_corrected_rad": [0.0, np.pi / 2.0],
            "cycle_flap_frequency_hz": [4.0, 6.0],
            "airspeed_validated.true_airspeed_m_s": [10.0, 20.0],
            "vehicle_air_data.rho": [1.2, 1.0],
            "vehicle_attitude.q[0]": [1.0, 1.0],
            "vehicle_attitude.q[1]": [0.0, 0.0],
            "vehicle_attitude.q[2]": [0.0, 0.0],
            "vehicle_attitude.q[3]": [0.0, 0.0],
            "vehicle_local_position.vx": [10.0, 20.0],
            "vehicle_local_position.vy": [0.0, 0.0],
            "vehicle_local_position.vz": [-1.0, 2.0],
            "wind.windspeed_north": [0.0, 0.0],
            "wind.windspeed_east": [0.0, 0.0],
        }
    )

    features, spec = build_feature_frame(frame)

    assert isinstance(spec, FeatureSpec)
    assert spec.phase_column == "phase_corrected_rad"
    assert spec.aoa_source == "body_air_relative_velocity"
    assert features["phase_sin_1"].to_numpy().tolist() == [0.0, 1.0]
    assert np.allclose(features["phase_cos_1"].to_numpy(), [1.0, 0.0])
    assert np.allclose(features["dynamic_pressure_pa"].to_numpy(), [60.0, 200.0])
    assert np.allclose(features["alpha_rad"].to_numpy(), np.arctan2([1.0, -2.0], [10.0, 20.0]))
    assert "alpha_rad_x_phase_sin_1" in features.columns
    assert "flap_frequency_hz_x_phase_cos_1" in features.columns


def test_build_feature_frame_falls_back_to_pitch_proxy_for_alpha() -> None:
    frame = pd.DataFrame(
        {
            "drive_phase_rad": [0.0],
            "flap_frequency_hz": [5.0],
            "airspeed_validated.true_airspeed_m_s": [12.0],
            "vehicle_air_data.rho": [1.25],
            "airspeed_validated.pitch_filtered": [0.2],
        }
    )

    features, spec = build_feature_frame(frame)

    assert spec.phase_column == "drive_phase_rad"
    assert spec.frequency_column == "flap_frequency_hz"
    assert spec.aoa_source == "airspeed_validated.pitch_filtered"
    assert features.loc[0, "alpha_rad"] == 0.2


def test_correction_model_applies_additive_multiplicative_and_affine_math() -> None:
    prior = np.array([[2.0, -4.0, 10.0], [3.0, 5.0, -6.0]])
    base = np.full((2, len(FORCE_TARGETS)), 0.5)
    gain = np.full((2, len(FORCE_TARGETS)), 0.1)
    model = CorrectionModel(
        name="affine",
        targets=FORCE_TARGETS,
        feature_columns=["constant"],
        feature_mean=np.array([0.0]),
        feature_scale=np.array([1.0]),
        coefficients=np.zeros((1, len(FORCE_TARGETS))),
        intercept=np.zeros(len(FORCE_TARGETS)),
        alpha=0.0,
    )

    assert np.allclose(model.apply_correction(prior, base, variant="additive"), prior + base)
    assert np.allclose(model.apply_correction(prior, gain, variant="multiplicative"), prior * 1.1)
    assert np.allclose(model.apply_correction(prior, base + gain * prior, variant="affine"), prior + base + gain * prior)


def test_cli_execution_context_can_import_diagnostic_modules() -> None:
    script_dir = Path(__file__).resolve().parents[1] / "scripts"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import train_delaurier_greybox_force_correction; "
                "from scripts.analyze_delaurier_residual_phase import phase_bin_table; "
                "print(phase_bin_table.__name__)"
            ),
        ],
        cwd=script_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "phase_bin_table" in result.stdout


def test_dataframe_to_markdown_does_not_require_optional_tabulate() -> None:
    table = dataframe_to_markdown(pd.DataFrame({"name": ["prior"], "rmse": [1.23456]}))

    assert "| name | rmse |" in table
    assert "| prior | 1.23456 |" in table
