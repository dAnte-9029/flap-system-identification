"""Validation-only best-epoch selection using the legacy comparison rule."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BestEpochSelection:
    state_dict: dict[str, Any]
    val_loss: float
    epoch: int


def update_best_epoch_selection(
    selection: BestEpochSelection,
    *,
    val_loss: float,
    epoch: int,
    state_dict: dict[str, Any],
    tolerance: float = 1e-8,
) -> tuple[BestEpochSelection, bool]:
    improved = val_loss < selection.val_loss - tolerance
    if not improved:
        return selection, False
    return (
        BestEpochSelection(
            state_dict=copy.deepcopy(state_dict),
            val_loss=val_loss,
            epoch=epoch,
        ),
        True,
    )
