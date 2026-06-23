from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.train_fx_fz_structured_correction import structured_family_specs


def test_structured_family_specs_can_select_exact_phase_freq_q_family() -> None:
    features = pd.DataFrame(
        {
            "phase_sin_1": np.zeros(2),
            "phase_cos_1": np.ones(2),
            "phase_sin_2": np.zeros(2),
            "phase_cos_2": np.ones(2),
            "flap_frequency_hz": np.full(2, 4.5),
            "flap_frequency_hz_x_phase_sin_1": np.zeros(2),
            "flap_frequency_hz_x_phase_cos_1": np.full(2, 4.5),
            "body_rate_q": np.full(2, 0.1),
            "body_rate_q_x_phase_sin_1": np.zeros(2),
            "body_rate_q_x_phase_cos_1": np.full(2, 0.1),
            "alpha_rad": np.zeros(2),
        }
    )

    specs = structured_family_specs(features, requested_families=("phase_freq_q_gain_bias",))

    assert list(specs) == ["phase_freq_q_gain_bias"]
    assert specs["phase_freq_q_gain_bias"] == [
        "phase_sin_1",
        "phase_cos_1",
        "phase_sin_2",
        "phase_cos_2",
        "flap_frequency_hz",
        "flap_frequency_hz_x_phase_sin_1",
        "flap_frequency_hz_x_phase_cos_1",
        "body_rate_q",
        "body_rate_q_x_phase_sin_1",
        "body_rate_q_x_phase_cos_1",
    ]
