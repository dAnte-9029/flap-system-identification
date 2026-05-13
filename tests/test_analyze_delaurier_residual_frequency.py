from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analyze_delaurier_residual_frequency import frequency_residual_energy_table


def test_frequency_residual_energy_table_separates_structured_bands() -> None:
    sample_rate_hz = 100.0
    time_s = np.arange(1000, dtype=float) / sample_rate_hz
    low = 2.0 * np.sin(2.0 * np.pi * 0.5 * time_s)
    flap = 3.0 * np.sin(2.0 * np.pi * 5.0 * time_s)
    high = 1.0 * np.sin(2.0 * np.pi * 18.0 * time_s)
    true_residual = low + flap + high
    pred_residual = low + flap

    frame = pd.DataFrame(
        {
            "log_id": "a",
            "segment_id": 0,
            "time_s": time_s,
            "cycle_flap_frequency_hz": 5.0,
            "label_fx_b": true_residual,
            "prior_fx_b": np.zeros_like(true_residual),
            "pred_fx_b": pred_residual,
        }
    )

    table = frequency_residual_energy_table(frame, targets=("fx_b",))

    low_row = table.loc[(table["target"] == "fx_b") & (table["component"] == "low_0_1hz")].iloc[0]
    flap_row = table.loc[(table["target"] == "fx_b") & (table["component"] == "flap_main")].iloc[0]
    high_row = table.loc[
        (table["target"] == "fx_b")
        & (table["component"] == "broadband_high_8_25hz_excl_structured")
    ].iloc[0]

    assert low_row["true_energy_fraction"] > 0.25
    assert flap_row["true_energy_fraction"] > 0.55
    assert high_row["true_energy_fraction"] > 0.05
    assert high_row["remaining_energy_fraction_of_true"] > 0.95
    assert low_row["remaining_energy_fraction_of_true"] < 1e-10
    assert flap_row["remaining_energy_fraction_of_true"] < 1e-10
    assert low_row["energy_reduction_fraction"] > 0.999
    assert flap_row["energy_reduction_fraction"] > 0.999
    assert high_row["energy_reduction_fraction"] < 0.05
