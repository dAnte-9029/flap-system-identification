from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.diagnose_wing_tail_my_alignment import (
    WingJointParams,
    fixed_wing_ac_arm_frd,
    pitch_moment_from_force,
)


def test_pitch_moment_from_arm_and_force_frd() -> None:
    arm = np.array([[0.12, 0.0, 0.05]])
    force = np.array([[3.0, 0.0, -8.0]])

    my = pitch_moment_from_force(arm, force)

    assert np.allclose(my, 0.05 * 3.0 - 0.12 * (-8.0))


def test_fixed_wing_ac_arm_returns_finite_rx_rz(tmp_path) -> None:
    geom_csv = tmp_path / "wing_geom.csv"
    geom_csv.write_text(
        "x_mid_m,c_m,dhat\n"
        "0.1,0.2,0.25\n"
        "0.2,0.2,0.25\n"
        "0.3,0.2,0.25\n",
        encoding="utf-8",
    )
    samples = pd.DataFrame({"wing_stroke_angle_rad": [0.0, 0.1, -0.1]})
    cg_frd_m = np.array([-0.12, 0.0, -0.04])

    arm = fixed_wing_ac_arm_frd(
        samples,
        cg_frd_m=cg_frd_m,
        wing_geom_csv=geom_csv,
        joint_params=WingJointParams(),
    )

    assert arm.shape == (3, 3)
    assert np.isfinite(arm[:, 0]).all()
    assert np.isfinite(arm[:, 2]).all()
