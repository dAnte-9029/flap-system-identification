"""Early-stopping patience state with the legacy stop timing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EarlyStoppingState:
    epochs_without_improvement: int = 0


def update_early_stopping(
    state: EarlyStoppingState,
    *,
    improved: bool,
    patience: int,
) -> tuple[EarlyStoppingState, bool]:
    if improved:
        return EarlyStoppingState(epochs_without_improvement=0), False
    updated = EarlyStoppingState(epochs_without_improvement=state.epochs_without_improvement + 1)
    return updated, updated.epochs_without_improvement >= patience
