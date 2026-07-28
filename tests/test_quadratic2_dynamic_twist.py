from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from system_identification.physics.baselines.wing_only import (
    WingOnlyBaselineConfig,
    evaluate_wing_only_delaurier_segment,
)
from system_identification.physics.delaurier.dynamic_twist import (
    compute_delaurier_dynamic_twist,
)


ETA = np.array([0.0, 0.25, 0.5, 0.75, 1.0])


def _twist(
    phase: np.ndarray,
    *,
    rate: np.ndarray | float = 7.0,
    acceleration: np.ndarray | float = 2.5,
    profile_name: str = "quadratic2_phase",
    kappa: float = 0.375,
    offset: float = 0.2,
):
    return compute_delaurier_dynamic_twist(
        strip_span_m=ETA,
        strip_width_m=np.full(len(ETA), 0.1),
        semi_span_m=1.0,
        mean_pitch_rad=0.11,
        tip_twist_amplitude_rad=0.4,
        phase_rad=np.asarray(phase, dtype=float),
        phase_rate_rad_s=np.asarray(rate, dtype=float),
        phase_acceleration_rad_s2=np.asarray(acceleration, dtype=float),
        enabled=True,
        profile_name=profile_name,
        kappa=kappa,
        phase_offset_rad=offset,
    )


@pytest.mark.parametrize("phase_deg", [0.0, 90.0, 180.0, 270.0])
def test_quadratic2_analytic_derivatives_match_centered_finite_differences(
    phase_deg: float,
) -> None:
    phase0 = math.radians(phase_deg)
    rate0 = 7.0
    acceleration = 2.5
    step = 1.0e-5

    def delta_at(time_s: float) -> np.ndarray:
        phase = phase0 + rate0 * time_s + 0.5 * acceleration * time_s**2
        rate = rate0 + acceleration * time_s
        return _twist(
            np.array([phase]),
            rate=np.array([rate]),
            acceleration=np.array([acceleration]),
        ).delta_theta[0]

    center = delta_at(0.0)
    plus = delta_at(step)
    minus = delta_at(-step)
    finite_first = (plus - minus) / (2.0 * step)
    finite_second = (plus - 2.0 * center + minus) / step**2
    analytic = _twist(
        np.array([phase0]),
        rate=np.array([rate0]),
        acceleration=np.array([acceleration]),
    )

    # Float64 centered differences at h=1e-5 balance O(h^2) truncation with
    # cancellation. First and second derivatives have different conditioning.
    np.testing.assert_allclose(analytic.delta_theta_dot[0], finite_first, atol=2.0e-9, rtol=2.0e-9)
    np.testing.assert_allclose(
        analytic.delta_theta_ddot[0],
        finite_second,
        atol=2.0e-6,
        rtol=2.0e-7,
    )


def test_quadratic2_uses_requested_eta_grid_and_tip_amplitude() -> None:
    phase = np.array([math.pi / 2.0 + 0.2])
    result = _twist(phase, rate=np.array([3.0]), acceleration=np.array([1.0]))
    expected_shape = (1.0 - 0.375) * ETA + 0.375 * ETA**2
    np.testing.assert_array_equal(result.span_fraction[0], ETA)
    np.testing.assert_allclose(result.delta_theta[0], -0.4 * expected_shape, atol=1.0e-14)
    assert result.delta_theta[0, 0] == 0.0
    assert result.delta_theta[0, -1] == pytest.approx(-0.4)


def test_quadratic2_kappa_zero_phase_zero_matches_legacy_pointwise() -> None:
    common = dict(
        strip_span_m=np.array([0.05, 0.2, 0.55, 0.95]),
        strip_width_m=np.full(4, 0.1),
        semi_span_m=1.0,
        mean_pitch_rad=np.array([0.1, -0.03]),
        tip_twist_amplitude_rad=math.radians(17.0),
        phase_rad=np.array([0.3, 2.1]),
        phase_rate_rad_s=np.array([23.0, 31.0]),
        phase_acceleration_rad_s2=np.array([4.0, -3.0]),
        enabled=True,
    )
    legacy = compute_delaurier_dynamic_twist(**common, profile_name="legacy_linear")
    quadratic = compute_delaurier_dynamic_twist(
        **common,
        profile_name="quadratic2_phase",
        kappa=0.0,
        phase_offset_rad=0.0,
    )
    for field in (
        "theta",
        "theta_dot",
        "theta_ddot",
        "delta_theta",
        "delta_theta_dot",
        "delta_theta_ddot",
    ):
        np.testing.assert_array_equal(getattr(quadratic, field), getattr(legacy, field))


def test_signed_left_right_span_produces_identical_physical_washout() -> None:
    common = dict(
        strip_width_m=np.full(4, 0.1),
        semi_span_m=1.0,
        mean_pitch_rad=0.0,
        tip_twist_amplitude_rad=0.3,
        phase_rad=np.array([0.7]),
        phase_rate_rad_s=np.array([20.0]),
        phase_acceleration_rad_s2=np.array([2.0]),
        enabled=True,
        profile_name="quadratic2_phase",
        kappa=-0.5,
        phase_offset_rad=0.25,
    )
    left = compute_delaurier_dynamic_twist(
        **common, strip_span_m=np.array([0.1, 0.3, 0.6, 0.9])
    )
    right = compute_delaurier_dynamic_twist(
        **common, strip_span_m=-np.array([0.1, 0.3, 0.6, 0.9])
    )
    for field in ("theta", "theta_dot", "theta_ddot", "delta_theta"):
        np.testing.assert_array_equal(getattr(left, field), getattr(right, field))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"profile_name": "unknown"}, "Unsupported twist profile"),
        ({"profile_name": "quadratic2_phase", "kappa": -1.01}, "within"),
        ({"profile_name": "quadratic2_phase", "kappa": 1.01}, "within"),
        ({"profile_name": "quadratic2_phase", "offset": np.nan}, "finite"),
        ({"profile_name": "legacy_linear", "kappa": 0.1}, "requires"),
    ],
)
def test_invalid_profile_parameters_fail_closed(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _twist(np.array([0.2]), **kwargs)


def _segment() -> pd.DataFrame:
    count = 32
    phase = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    frame = pd.DataFrame(
        {
            "log_id": "log",
            "segment_id": 1,
            "time_s": np.arange(count) * 0.01,
            "timestamp_us": np.arange(count) * 10_000,
            "mechanical_phase_rad": phase,
            "flap_frequency_hz": 5.0,
            "vehicle_air_data.rho": 1.15,
            "vehicle_local_position.vx": 8.0,
            "vehicle_local_position.vy": 0.0,
            "vehicle_local_position.vz": 0.0,
            "wind.windspeed_north": 0.0,
            "wind.windspeed_east": 0.0,
            "vehicle_attitude.q[0]": 1.0,
            "vehicle_attitude.q[1]": 0.0,
            "vehicle_attitude.q[2]": 0.0,
            "vehicle_attitude.q[3]": 0.0,
        }
    )
    return frame


def test_legacy_parity_reaches_strip_force_and_integrated_fx_fz() -> None:
    geometry = (
        Path(__file__).parents[1]
        / "metadata"
        / "aircraft"
        / "flapper_01"
        / "wing_geometry_isaaclab_3b5d4ec.csv"
    )
    base = WingOnlyBaselineConfig(
        airflow_mode="attitude_ground_wind_3d",
        chunk_size=11,
    )
    legacy = evaluate_wing_only_delaurier_segment(
        _segment(),
        theta_tip_deg=[17.0],
        geometry_path=geometry,
        config=base,
        include_detailed_diagnostics=True,
    )
    quadratic = evaluate_wing_only_delaurier_segment(
        _segment(),
        theta_tip_deg=[17.0],
        geometry_path=geometry,
        config=replace(
            base,
            twist_profile_name="quadratic2_phase",
            twist_kappa=0.0,
            twist_phase_offset_rad=0.0,
        ),
        include_detailed_diagnostics=True,
    )
    compare = [
        column
        for column in legacy.columns
        if column.startswith(("pred_", "component_", "span_"))
        and column not in {"pred_left_fy_b", "pred_right_fy_b"}
    ]
    for column in compare:
        np.testing.assert_array_equal(quadratic[column], legacy[column])
    np.testing.assert_array_equal(quadratic["pred_fx_b"], legacy["pred_fx_b"])
    np.testing.assert_array_equal(quadratic["pred_fz_b"], legacy["pred_fz_b"])


def test_old_default_config_behavior_is_legacy_linear() -> None:
    config = WingOnlyBaselineConfig()
    assert config.twist_profile_name == "legacy_linear"
    assert config.twist_kappa == 0.0
    assert config.twist_phase_offset_rad == 0.0
