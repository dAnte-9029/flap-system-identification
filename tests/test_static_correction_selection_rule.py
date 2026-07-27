from __future__ import annotations

import pytest

from system_identification.training.correction.selection_rules import (
    leave_one_log_out_selection,
    select_one_se,
)


def _candidate(
    candidate_id: str,
    errors: list[float],
    complexity: list[float],
    **secondary: float,
) -> dict[str, object]:
    logs = {f"log_{index}": value for index, value in enumerate(errors)}
    defaults = {
        "macro_waveform_rmse": sum(errors) / len(errors),
        "worst_log_total_rmse": max(errors),
        "phase_bin_waveform_rmse": sum(errors) / len(errors),
        "peak_magnitude_error": 0.0,
        "macro_mean_rmse": sum(errors) / len(errors),
        "downstroke_integral_error_abs": 0.0,
    }
    defaults.update(secondary)
    return {
        "candidate_id": candidate_id,
        "macro_total_rmse": sum(errors) / len(errors),
        "per_log_rmse": errors,
        "per_log_rmse_by_log": logs,
        "complexity": complexity,
        **defaults,
    }


def test_one_se_selects_simpler_near_equivalent_model() -> None:
    complex_best = _candidate("complex", [0.8, 1.0, 1.2, 1.0, 1.0], [12, 3, 2, 1, 2, 1])
    simple = _candidate("simple", [0.9, 1.1, 1.2, 1.0, 1.0], [4, 1, 0, 0, 1, 1])
    assert select_one_se([complex_best, simple], component="fx")["selected_candidate_id"] == "simple"


def test_model_outside_one_se_cannot_win_on_complexity() -> None:
    best = _candidate("best", [1.0] * 5, [10, 4, 2, 1, 2, 1])
    too_bad = _candidate("too_bad", [2.0] * 5, [1, 1, 0, 0, 0, 1])
    assert select_one_se([best, too_bad], component="fz")["selected_candidate_id"] == "best"


def test_fx_secondary_breaks_equal_complexity() -> None:
    a = _candidate("a", [1.0] * 5, [5, 2, 1, 0, 1, 1], macro_waveform_rmse=0.9)
    b = _candidate("b", [1.0] * 5, [5, 2, 1, 0, 1, 1], macro_waveform_rmse=0.8)
    assert select_one_se([a, b], component="fx")["selected_candidate_id"] == "b"


def test_fz_secondary_breaks_equal_complexity() -> None:
    a = _candidate("a", [1.0] * 5, [5, 2, 1, 0, 1, 1], macro_mean_rmse=0.9)
    b = _candidate("b", [1.0] * 5, [5, 2, 1, 0, 1, 1], macro_mean_rmse=0.7)
    assert select_one_se([a, b], component="fz")["selected_candidate_id"] == "b"


def test_leave_one_log_out_does_not_retrain() -> None:
    candidates = [
        _candidate("a", [1.0, 1.0, 1.0, 1.0, 1.0], [2, 1, 0, 0, 1, 1]),
        _candidate("b", [0.9, 1.1, 0.9, 1.1, 0.9], [3, 1, 0, 0, 1, 1]),
    ]
    result = leave_one_log_out_selection(candidates, component="fx")
    assert len(result["leave_one_log_out"]) == 5
    assert sum(result["selection_counts"].values()) == 5


def test_empty_selection_fails() -> None:
    with pytest.raises(ValueError, match="empty"):
        select_one_se([], component="fx")
