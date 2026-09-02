from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.plot_multibody_phase_binned_force_comparison import align_partition, equal_log_phase_bins


def _sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    old = pd.DataFrame(
        {
            "log_id": ["short", "long", "long", "long"],
            "timestamp_us": [1, 1, 2, 3],
            "mechanical_phase_rad": [0.1, 0.1, 0.2, 0.3],
            "fx_b": [10.0, 0.0, 0.0, 0.0],
            "fz_b": [20.0, 2.0, 2.0, 2.0],
            "phase_valid": True,
            "label_reconstruction_valid": True,
        }
    )
    corrected = old[["log_id", "timestamp_us", "mechanical_phase_rad"]].copy()
    corrected["fx_b"] = old["fx_b"] + 2.0
    corrected["fz_b"] = old["fz_b"] - 4.0
    corrected["fx_b_rigid_v01"] = old["fx_b"]
    corrected["fz_b_rigid_v01"] = old["fz_b"]
    corrected["multibody_label_valid"] = True
    prior = old[["log_id", "timestamp_us"]].copy()
    prior["fx_b"] = 1.0
    prior["fz_b"] = -1.0
    return old, corrected, prior


def test_equal_log_phase_bins_do_not_overweight_long_logs() -> None:
    aligned = align_partition(*_sources())
    table = equal_log_phase_bins(aligned, partition="validation", phase_bins=2)

    rigid_fx = table.loc[
        (table["component"] == "fx_b")
        & (table["curve"] == "rigid_inverse_v04")
        & (table["phase_bin"] == 0)
    ].iloc[0]
    corrected_fz = table.loc[
        (table["component"] == "fz_b")
        & (table["curve"] == "multibody_inverse_v05")
        & (table["phase_bin"] == 0)
    ].iloc[0]

    assert rigid_fx["force_mean_n"] == pytest.approx(5.0)
    assert rigid_fx["log_count"] == 2
    assert rigid_fx["sample_count"] == 4
    assert corrected_fz["force_mean_n"] == pytest.approx(7.0)


def test_alignment_refuses_missing_keys() -> None:
    old, corrected, prior = _sources()
    prior = prior.iloc[:-1].copy()

    with pytest.raises(ValueError, match="alignment key mismatch"):
        align_partition(old, corrected, prior)


def test_alignment_refuses_modified_rigid_backup() -> None:
    old, corrected, prior = _sources()
    corrected.loc[0, "fx_b_rigid_v01"] += 0.1

    with pytest.raises(ValueError, match="preserve the v0.4 fx_b backup"):
        align_partition(old, corrected, prior)
