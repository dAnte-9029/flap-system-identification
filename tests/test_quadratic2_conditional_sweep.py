from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from system_identification.analysis.quadratic2_conditional_sweep import (
    _signed_delta_deg,
    conditional_wide,
    conditional_candidates,
    conditional_dry_run_summary,
)
from system_identification.analysis.quadratic2_twist_sweep import resolve_experiment


PROJECT_ROOT = Path(__file__).parents[1]
CONFIG = (
    PROJECT_ROOT / "configs" / "delaurier" / "quadratic2_conditional_sweep_v2.yaml"
)


def test_conditional_grid_is_exact_unique_and_test_sealed() -> None:
    resolved = resolve_experiment(CONFIG, project_root=PROJECT_ROOT)
    candidates = conditional_candidates(resolved)
    assert len(candidates) == 4 * 9 * 15 == 540
    assert len({candidate.parameter_hash for candidate in candidates}) == 540
    assert {candidate.A_tip_deg for candidate in candidates} == {20.0, 30.0, 35.0, 40.0}
    assert min(candidate.kappa for candidate in candidates) == -1.0
    assert max(candidate.kappa for candidate in candidates) == 1.0
    assert min(candidate.psi_theta_deg for candidate in candidates) == -60.0
    assert max(candidate.psi_theta_deg for candidate in candidates) == 10.0
    summary = conditional_dry_run_summary(resolved)
    assert summary["expected_reused_unique"] == 187
    assert summary["expected_new_unique"] == 353
    assert summary["test_partition_used"] is False
    assert summary["sealed_test"] is True
    assert summary["boundary_diagnostic_enabled"] is False


def test_signed_phase_delta_wraps_across_cycle_boundary() -> None:
    import pandas as pd

    model = pd.Series(np.radians([10.0, 350.0, 180.0]))
    data = pd.Series(np.radians([350.0, 10.0, 0.0]))
    assert np.allclose(_signed_delta_deg(model, data), [20.0, -20.0, -180.0])


def test_conditional_wide_normalizes_xcorr_lag_to_signed_degrees() -> None:
    import pandas as pd

    rows = []
    for partition in ("train", "validation"):
        for component in ("fx", "fz"):
            rows.append(
                {
                    "parameter_hash": "candidate",
                    "profile_name": "quadratic2_phase",
                    "A_tip_deg": 35.0,
                    "kappa": 0.0,
                    "psi_theta_deg": -25.0,
                    "static_twist_offset_deg": 0.0,
                    "stage": "conditional",
                    "family": "test",
                    "airflow_mode": "attitude_ground_wind_3d",
                    "partition": partition,
                    "component": component,
                    "model_first_harmonic_phase_rad": np.radians(350.0),
                    "data_first_harmonic_phase_rad": np.radians(10.0),
                    "model_primary_peak_phase_deg": 212.5,
                    "data_primary_peak_phase_deg": 217.5,
                    "model_primary_peak_phase_smooth_deg": 212.5,
                    "data_primary_peak_phase_smooth_deg": 217.5,
                    "circular_xcorr_lag_deg": 355.0,
                }
            )
    wide = conditional_wide(pd.DataFrame(rows))
    assert wide["validation_fx_xcorr_signed_lag_deg"].iloc[0] == -5.0


def test_conditional_cli_dry_run_reports_contract_and_counts() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_quadratic2_conditional_sweep.py"),
            "--config",
            str(CONFIG),
            "--stage",
            "conditional",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["conditional_parameter_combinations"] == 540
    assert payload["expected_reused_unique"] == 187
    assert payload["expected_new_unique"] == 353
    assert payload["test_partition_used"] is False
