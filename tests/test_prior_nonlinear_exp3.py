from pathlib import Path

from scripts.run_nested_prior_nonlinear_calibration_exp3 import (
    PARAMETER_SPECS,
    _export_prior_command,
    _prior_name,
)


def test_exp3_parameter_specs_include_independent_negative_stall_bound():
    names = [spec.name for spec in PARAMETER_SPECS]

    assert names == [
        "alpha0_deg",
        "alpha_stall_min_deg",
        "alpha_stall_max_deg",
        "cd_cf",
        "xi",
    ]


def test_exp3_prior_name_and_export_command_include_alpha_stall_min():
    params = {
        "twist_eta_max_deg": 10.0,
        "alpha0_deg": 7.25,
        "eta_s": 0.65,
        "cd_f": 0.0,
        "enable_separation": True,
        "alpha_stall_min_deg": -11.5,
        "alpha_stall_max_deg": 26.0,
        "cd_cf": 0.66,
        "xi": 1.3,
    }

    name = _prior_name("candidate", params)
    cmd = _export_prior_command(
        python_exe=Path("/env/python"),
        exporter=Path("/repo/export.py"),
        split_root=Path("/split"),
        metadata=Path("/meta.yaml"),
        output_root=Path("/out"),
        params=params,
        chunk_size=123,
        device="cuda",
    )

    assert "alpha_stall_min_deg_m11p5" in name
    assert "--alpha-stall-min-deg" in cmd
    assert cmd[cmd.index("--alpha-stall-min-deg") + 1] == "-11.5"
    assert "--enable-separation" in cmd
