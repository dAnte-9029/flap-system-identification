from pathlib import Path


PX4_ROOT = Path("/home/zn/PX4-Autopilot")


def test_px4_default_flap_ratio_is_current_aircraft_ratio():
    rpm_params = (PX4_ROOT / "src/modules/rpm_pid/rpm_pid_params.c").read_text()
    as5600_hpp = (PX4_ROOT / "src/drivers/encoder/as5600/AS5600.hpp").read_text()
    wing_phase_cpp = (PX4_ROOT / "src/modules/wing_phase/WingPhase.cpp").read_text()

    assert "PARAM_DEFINE_FLOAT(FLAP_RATIO, 8.0f);" in rpm_params
    assert "_flap_ratio{8.0f}" in as5600_hpp
    assert "_counts_per_cycle{kCountsPerRevolution * 8.0f}" in wing_phase_cpp
