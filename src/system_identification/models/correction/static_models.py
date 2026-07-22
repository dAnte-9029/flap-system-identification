"""Pure immutable coefficient and numerical-diagnostic structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class RidgeDiagnostics:
    row_count: int
    coefficient_count: int
    matrix_rank: int
    condition_number: float
    weighted_residual_norm: float
    effective_weight_sum: float
    rank_deficient: bool
    finite_checks: bool
    solver: str = "weighted_augmented_lstsq"

    def to_dict(self) -> dict[str, object]:
        return {
            "row_count": self.row_count,
            "coefficient_count": self.coefficient_count,
            "matrix_rank": self.matrix_rank,
            "condition_number": self.condition_number if np.isfinite(self.condition_number) else None,
            "weighted_residual_norm": self.weighted_residual_norm,
            "effective_weight_sum": self.effective_weight_sum,
            "rank_deficient": self.rank_deficient,
            "finite_checks": self.finite_checks,
            "solver": self.solver,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "RidgeDiagnostics":
        return cls(
            row_count=int(value["row_count"]),
            coefficient_count=int(value["coefficient_count"]),
            matrix_rank=int(value["matrix_rank"]),
            condition_number=(
                float("inf") if value.get("condition_number") is None else float(value["condition_number"])
            ),
            weighted_residual_norm=float(value["weighted_residual_norm"]),
            effective_weight_sum=float(value["effective_weight_sum"]),
            rank_deficient=bool(value["rank_deficient"]),
            finite_checks=bool(value["finite_checks"]),
            solver=str(value.get("solver", "weighted_augmented_lstsq")),
        )


@dataclass(frozen=True)
class RidgeSolution:
    coefficients: np.ndarray
    feature_names: tuple[str, ...]
    ridge_lambda: float
    penalize_mask: tuple[bool, ...]
    diagnostics: RidgeDiagnostics

    def to_dict(self) -> dict[str, object]:
        return {
            "coefficients": [float(value) for value in self.coefficients],
            "feature_names": list(self.feature_names),
            "ridge_lambda": float(self.ridge_lambda),
            "penalize_mask": list(self.penalize_mask),
            "intercept_penalty": "unpenalized" if "intercept" in self.feature_names else "not_applicable",
            "diagnostics": self.diagnostics.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "RidgeSolution":
        return cls(
            coefficients=np.asarray(value["coefficients"], dtype=np.float64),
            feature_names=tuple(str(item) for item in value["feature_names"]),
            ridge_lambda=float(value["ridge_lambda"]),
            penalize_mask=tuple(bool(item) for item in value["penalize_mask"]),
            diagnostics=RidgeDiagnostics.from_dict(value["diagnostics"]),
        )
