import numpy as np

from system_identification.phase import (
    compute_drive_phase_rad,
    compute_wing_stroke_angle_rad,
    compute_wing_stroke_direction,
    wrap_to_2pi,
)


def test_wrap_to_2pi_maps_negative_angles_into_expected_interval():
    values = np.array([-0.1, 0.0, 2.0 * np.pi + 0.2])

    wrapped = wrap_to_2pi(values)

    assert np.all(wrapped >= 0.0)
    assert np.all(wrapped < 2.0 * np.pi)
    assert np.isclose(wrapped[0], 2.0 * np.pi - 0.1)
    assert np.isclose(wrapped[1], 0.0)
    assert np.isclose(wrapped[2], 0.2)


def test_compute_drive_phase_rad_uses_ratio_sign_and_offset():
    encoder_phase_unwrapped = np.array([0.0, 7.5 * np.pi / 2.0, 7.5 * np.pi])

    drive_unwrapped, drive_wrapped = compute_drive_phase_rad(
        encoder_phase_unwrapped_rad=encoder_phase_unwrapped,
        encoder_to_drive_ratio=7.5,
        encoder_to_drive_sign=1.0,
        drive_phase_zero_offset_rad=0.0,
    )

    assert np.allclose(drive_unwrapped, np.array([0.0, np.pi / 2.0, np.pi]))
    assert np.allclose(drive_wrapped, np.array([0.0, np.pi / 2.0, np.pi]))


def test_compute_wing_stroke_angle_and_direction_follow_user_convention():
    drive_phase = np.array([0.0, np.pi / 4.0, np.pi, 7.0 * np.pi / 4.0])

    stroke = compute_wing_stroke_angle_rad(
        drive_phase_rad=drive_phase,
        wing_stroke_amplitude_rad=np.deg2rad(30.0),
        wing_stroke_phase_offset_rad=0.0,
    )
    direction = compute_wing_stroke_direction(
        drive_phase_rad=drive_phase,
        wing_stroke_phase_offset_rad=0.0,
    )

    assert np.isclose(stroke[0], 0.0)
    assert stroke[1] > 0.0
    assert np.isclose(stroke[2], 0.0, atol=1e-6)
    assert stroke[3] < 0.0
    assert list(direction) == ["upstroke", "upstroke", "downstroke", "upstroke"]
