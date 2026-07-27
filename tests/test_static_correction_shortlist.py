from __future__ import annotations

import copy

import pytest

from system_identification.training.correction.selection_rules import (
    seal_shortlist,
    verify_sealed_shortlist,
)


def _payload() -> dict[str, object]:
    return {
        "schema_version": "static_correction_train_cv_shortlist_v1",
        "source_config_hash": "config",
        "source_artifact_hash": "artifact",
        "finalists": {
            "fx": [{"candidate_id": "a", "model_spec": {"model_type": "gain_bias"}}],
            "fz": [{"candidate_id": "b", "model_spec": {"model_type": "gain_bias"}}],
        },
    }


def test_sealed_hash_is_deterministic() -> None:
    assert seal_shortlist(_payload())["shortlist_hash"] == seal_shortlist(_payload())["shortlist_hash"]


def test_sealed_shortlist_verifies() -> None:
    value = seal_shortlist(_payload())
    verify_sealed_shortlist(value, expected_config_hash="config", expected_artifact_hash="artifact")


def test_candidate_change_invalidates_seal() -> None:
    value = seal_shortlist(_payload())
    value["finalists"]["fx"][0]["candidate_id"] = "changed"
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_sealed_shortlist(value, expected_config_hash="config", expected_artifact_hash="artifact")


def test_config_change_invalidates_shortlist() -> None:
    value = seal_shortlist(_payload())
    with pytest.raises(ValueError, match="config hash"):
        verify_sealed_shortlist(value, expected_config_hash="different", expected_artifact_hash="artifact")


def test_artifact_change_invalidates_shortlist() -> None:
    value = seal_shortlist(_payload())
    with pytest.raises(ValueError, match="artifact hash"):
        verify_sealed_shortlist(value, expected_config_hash="config", expected_artifact_hash="different")


def test_resealing_does_not_hash_old_hash_field() -> None:
    first = seal_shortlist(_payload())
    second = seal_shortlist(copy.deepcopy(first))
    assert first == second
