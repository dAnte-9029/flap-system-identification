from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from system_identification.analysis.quadratic2_twist_sweep import (
    TwistCandidate,
    _curve_metrics,
    _pareto_mask,
    _read_jsonl,
    _wide_metrics,
    build_shortlists,
    coarse_candidates,
    dry_run_summary,
    oat_candidates,
    resolve_experiment,
)


PROJECT_ROOT = Path(__file__).parents[1]
CONFIG = PROJECT_ROOT / "configs" / "delaurier" / "quadratic2_twist_sweep_v1.yaml"


def test_config_round_trip_resolves_active_train_validation_contract() -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert yaml.safe_load(yaml.safe_dump(raw, sort_keys=True)) == raw
    resolved = resolve_experiment(CONFIG, project_root=PROJECT_ROOT)
    summary = dry_run_summary(resolved)
    assert summary["partitions"] == ["train", "validation"]
    assert summary["sealed_test"] is True
    assert summary["test_partition_used"] is False
    assert summary["oat_parameter_combinations"] == 36
    assert summary["coarse_parameter_combinations"] == 9 * 9 * 13
    assert "+Fz down" in summary["force_convention"]


def test_parameter_hash_is_stable_and_physical_parameter_specific() -> None:
    candidate = TwistCandidate(
        profile_name="quadratic2_phase",
        A_tip_deg=10.0,
        kappa=0.25,
        psi_theta_deg=-15.0,
        stage="coarse",
        family="grid",
    )
    same = TwistCandidate(**candidate.__dict__)
    other_stage = TwistCandidate(**{**candidate.__dict__, "stage": "refine"})
    assert candidate.parameter_hash == same.parameter_hash
    assert candidate.parameter_hash == other_stage.parameter_hash
    assert len(candidate.parameter_hash) == 16


def test_compact_resume_log_keeps_latest_complete_unique_hash(tmp_path: Path) -> None:
    path = tmp_path / "resume.jsonl"
    alpha = {
        "profile_name": "quadratic2_phase",
        "A_tip_deg": 10.0,
        "kappa": 0.0,
        "psi_theta_deg": 0.0,
        "static_twist_offset_deg": 0.0,
        "stage": "coarse",
        "family": "grid",
    }
    beta = {**alpha, "A_tip_deg": 20.0}
    path.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {"parameter_hash": "old-alpha", "candidate": alpha, "metrics": [], "value": 1},
                {"parameter_hash": "old-beta", "candidate": beta, "metrics": [], "value": 2},
                {"parameter_hash": "other-old-alpha", "candidate": alpha, "metrics": [], "value": 3},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    records = _read_jsonl(path)
    alpha_hash = TwistCandidate(**alpha).parameter_hash
    beta_hash = TwistCandidate(**beta).parameter_hash
    assert set(records) == {alpha_hash, beta_hash}
    assert records[alpha_hash]["value"] == 3


def test_oat_and_coarse_grids_stay_inside_authorized_ranges() -> None:
    resolved = resolve_experiment(CONFIG, project_root=PROJECT_ROOT)
    oat = oat_candidates(resolved)
    coarse = coarse_candidates(resolved)
    assert len(oat) == 36
    assert len(coarse) == 1053
    assert {candidate.family for candidate in oat} == {
        "A_tip_deg",
        "kappa",
        "psi_theta_deg",
        "static_twist_offset_deg",
    }
    assert min(candidate.kappa for candidate in coarse) == -1.0
    assert max(candidate.kappa for candidate in coarse) == 1.0


def test_curve_metrics_reports_primary_peak_harmonic_and_circular_lag() -> None:
    phase = (np.arange(72) + 0.5) * 2.0 * np.pi / 72
    data_peak = math.radians(217.5)
    model_peak = math.radians(197.5)
    data = 3.0 + 2.0 * np.cos(phase - data_peak)
    model = 3.0 + 2.0 * np.cos(phase - model_peak)
    metrics = _curve_metrics(
        phase,
        data,
        model,
        component="fx",
        smooth_window=7,
        smooth_order=3,
        fx_interval_rad=(np.pi, 1.5 * np.pi),
    )
    assert np.isclose(metrics["data_primary_peak_phase_deg"], 217.5)
    assert np.isclose(metrics["model_primary_peak_phase_deg"], 197.5)
    assert np.isclose(metrics["primary_peak_phase_error_deg"], 20.0)
    assert np.isclose(math.degrees(metrics["first_harmonic_phase_error_rad"]), 20.0)
    assert np.isfinite(metrics["circular_xcorr_lag_deg"])
    assert np.isfinite(metrics["model_peak_half_height_width_rad"])
    assert "model_integral_90_180_n_rad" in metrics
    assert "model_integral_180_270_n_rad" in metrics


def test_pareto_mask_keeps_only_nondominated_points() -> None:
    mask = _pareto_mask(
        np.array([1.0, 2.0, 1.5, 3.0]),
        np.array([3.0, 1.0, 2.0, 3.0]),
    )
    assert mask.tolist() == [True, True, True, False]


def test_shortlist_gate_rejects_fz_amplitude_degradation_and_narrow_spike() -> None:
    resolved = resolve_experiment(CONFIG, project_root=PROJECT_ROOT)
    rows = []
    for candidate_hash, fz_error, width_deg in (
        ("baseline", 10.0, 90.0),
        ("good", 10.5, 90.0),
        ("bad_fz", 13.0, 90.0),
        ("spike", 10.5, 5.0),
    ):
        for partition in ("train", "validation"):
            for component in ("fx", "fz"):
                rows.append(
                    {
                        "parameter_hash": candidate_hash,
                        "profile_name": "legacy_linear" if candidate_hash == "baseline" else "quadratic2_phase",
                        "A_tip_deg": 0.0 if candidate_hash == "baseline" else 10.0,
                        "kappa": 0.0,
                        "psi_theta_deg": 0.0,
                        "static_twist_offset_deg": 0.0,
                        "stage": "baseline" if candidate_hash == "baseline" else "coarse",
                        "family": "test",
                        "airflow_mode": "attitude_ground_wind_3d",
                        "partition": partition,
                        "component": component,
                        "rmse": 3.0 if component == "fx" else 7.0,
                        "mae": 2.0,
                        "pearson_r": 0.8,
                        "primary_peak_phase_error_deg": 20.0 if component == "fx" else np.nan,
                        "model_peak_half_height_width_rad": math.radians(width_deg) if component == "fx" else np.nan,
                        "minimum_amplitude_error_abs": fz_error if component == "fz" else np.nan,
                    }
                )
    metrics = pd.DataFrame(rows)
    baseline = _wide_metrics(metrics.loc[metrics.parameter_hash == "baseline"]).iloc[0]
    with patch(
        "system_identification.analysis.quadratic2_twist_sweep._baseline_wide",
        return_value=baseline,
    ):
        fx_phase, _, _ = build_shortlists(
            resolved, metrics.loc[metrics.parameter_hash != "baseline"]
        )
    assert set(fx_phase.parameter_hash) == {"good"}
    assert "validation_fx_model_peak_half_height_width_rad" in _wide_metrics(metrics).columns


def test_cli_dry_run_reports_sealed_test_and_combination_counts() -> None:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_quadratic2_twist_sweep.py"),
        "--config",
        str(CONFIG),
        "--stage",
        "coarse",
        "--dry-run",
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["test_partition_used"] is False
    assert payload["sealed_test"] is True
    assert payload["coarse_parameter_combinations"] == 1053
    assert payload["wind_mode"] == "attitude_ground_wind_3d"
