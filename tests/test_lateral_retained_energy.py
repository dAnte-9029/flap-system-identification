from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analyze_lateral_retained_energy import compute_retained_energy_summary


def test_retained_energy_summary_separates_structured_and_broadband_components():
    sample_rate_hz = 100.0
    time_s = np.arange(2000, dtype=float) / sample_rate_hz
    low = 1.2 * np.sin(2.0 * np.pi * 0.5 * time_s)
    mid = 0.7 * np.sin(2.0 * np.pi * 2.0 * time_s)
    flap = 0.6 * np.sin(2.0 * np.pi * 5.0 * time_s)
    harmonic_2f = 0.4 * np.sin(2.0 * np.pi * 10.0 * time_s)
    harmonic_3f = 0.3 * np.sin(2.0 * np.pi * 15.0 * time_s)
    broadband_high = 0.5 * np.sin(2.0 * np.pi * 18.0 * time_s)
    raw = low + mid + flap + harmonic_2f + harmonic_3f + broadband_high
    smooth = low + mid + flap + 0.5 * harmonic_2f + 0.2 * harmonic_3f

    base = {
        "time_s": time_s,
        "log_id": "log_a",
        "segment_id": 0,
        "cycle_flap_frequency_hz": 5.0,
    }
    raw_frame = pd.DataFrame({**base, "fy_b": raw})
    smooth_frame = pd.DataFrame({**base, "fy_b": smooth})

    _, summary = compute_retained_energy_summary(
        raw_frame,
        {"sg_test": smooth_frame},
        targets=("fy_b",),
        high_band=(8.0, 25.0),
        edge_trim_s=0.0,
    )

    def ratio_for(band: str) -> float:
        row = summary.loc[(summary["variant"] == "sg_test") & (summary["target"] == "fy_b") & (summary["band"] == band)]
        return float(row["retained_energy_ratio"].iloc[0])

    assert ratio_for("low_0_1hz") > 0.95
    assert ratio_for("mid_1_3hz") > 0.95
    assert ratio_for("flap_main") > 0.95
    assert 0.20 < ratio_for("harmonic_2f") < 0.35
    assert ratio_for("harmonic_3f") < 0.10

    high_row = summary.loc[
        (summary["variant"] == "sg_test") & (summary["target"] == "fy_b") & (summary["band"] == "broadband_high_8_25hz_excl_structured")
    ].iloc[0]
    assert high_row["raw_energy_fraction"] > 0.08
    assert high_row["removed_energy_ratio"] > 0.95
