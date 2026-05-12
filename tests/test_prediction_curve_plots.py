from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.plot_prediction_curves import split_frame_on_plot_breaks, write_prediction_curve_plots


def test_write_prediction_curve_plots_creates_overview_and_zoom(tmp_path: Path):
    time_s = np.arange(40, dtype=float) * 0.02
    aligned = pd.DataFrame(
        {
            "log_id": ["log_a"] * len(time_s),
            "time_s": time_s,
            "true_fx_b": np.sin(time_s),
            "pred_fx_b": np.sin(time_s) + 0.05,
            "true_fy_b": np.cos(time_s),
            "pred_fy_b": np.cos(time_s) - 0.03,
        }
    )

    manifest = write_prediction_curve_plots(
        aligned,
        output_dir=tmp_path,
        split_name="test",
        targets=("fx_b", "fy_b"),
        zoom_samples=12,
        max_overview_points=20,
    )

    assert {"overview", "zoom_start", "zoom_worst_residual"} == set(manifest["view"])
    assert set(manifest["log_id"]) == {"log_a"}
    for plot_path in manifest["plot_path"]:
        path = Path(plot_path)
        assert path.exists()
        assert path.stat().st_size > 0


def test_split_frame_on_plot_breaks_splits_segment_changes_and_time_gaps():
    frame = pd.DataFrame(
        {
            "time_s": [0.00, 0.01, 0.02, 0.50, 0.51, 0.52],
            "segment_id": [1, 1, 1, 1, 2, 2],
            "true_fx_b": [0, 1, 2, 3, 4, 5],
            "pred_fx_b": [0, 1, 2, 3, 4, 5],
        }
    )

    parts = split_frame_on_plot_breaks(frame)

    assert [len(part) for part in parts] == [3, 1, 2]
