"""Manifest-backed DeLaurier prior materialization APIs."""

from system_identification.physics.priors.export import (
    AUTHORITATIVE_PRIOR_ID,
    AUTHORITATIVE_THETA_TIP_DEG,
    materialize_authoritative_delaurier_prior,
)

__all__ = [
    "AUTHORITATIVE_PRIOR_ID",
    "AUTHORITATIVE_THETA_TIP_DEG",
    "materialize_authoritative_delaurier_prior",
]
