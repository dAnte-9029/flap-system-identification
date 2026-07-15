"""Public time-series helpers shared by artifact plotting scripts."""

from __future__ import annotations

import numpy as np
import pandas as pd


def split_frame_on_plot_breaks(
    frame: pd.DataFrame,
    *,
    time_column: str = "time_s",
    segment_column: str = "segment_id",
    gap_multiplier: float = 8.0,
) -> list[pd.DataFrame]:
    """Split a plotted time series at segment changes and large time gaps."""

    if len(frame) <= 1:
        return [frame.copy()]
    break_before = np.zeros(len(frame), dtype=bool)
    if segment_column in frame.columns:
        segment_values = frame[segment_column].to_numpy()
        break_before[1:] |= segment_values[1:] != segment_values[:-1]
    if time_column in frame.columns:
        time_values = frame[time_column].to_numpy(dtype=float)
        delta = np.diff(time_values)
        valid = delta[np.isfinite(delta) & (delta > 0.0)]
        if len(valid):
            nominal = float(np.nanmedian(valid))
            if np.isfinite(nominal) and nominal > 0.0:
                break_before[1:] |= delta > float(gap_multiplier) * nominal
    split_indices = np.flatnonzero(break_before)
    starts = [0, *split_indices.tolist()]
    ends = [*split_indices.tolist(), len(frame)]
    return [frame.iloc[start:end].copy() for start, end in zip(starts, ends) if end > start]
