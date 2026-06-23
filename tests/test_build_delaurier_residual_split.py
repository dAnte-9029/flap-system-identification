from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_delaurier_residual_split import (
    TARGET_COLUMNS,
    align_prior_to_samples,
    build_residual_frame,
    build_residual_split,
)


def _samples() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "log_id": ["a", "a"],
            "time_s": [0.0, 0.01],
            "phase_corrected_rad": [0.1, 0.2],
            "fx_b": [10.0, 12.0],
            "fy_b": [1.0, 2.0],
            "fz_b": [-3.0, -4.0],
            "mx_b": [0.1, 0.2],
            "my_b": [0.3, 0.4],
            "mz_b": [0.5, 0.6],
        }
    )


def _prior(offset: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fx_b": [8.0 + offset, 9.0 + offset],
            "fy_b": [0.5, 1.5],
            "fz_b": [-2.0, -1.0],
            "mx_b": [0.0, 0.1],
            "my_b": [0.1, 0.2],
            "mz_b": [0.4, 0.4],
        }
    )


def _keyed_prior(offset: float = 0.0) -> pd.DataFrame:
    prior = _prior(offset=offset)
    prior.insert(0, "segment_id", [0, 0])
    prior.insert(0, "time_s", [0.0, 0.01])
    prior.insert(0, "log_id", ["a", "a"])
    return prior


def test_build_residual_frame_replaces_targets_and_preserves_true_and_prior() -> None:
    residual = build_residual_frame(_samples(), _prior(), prior_name="delaurier_physical")

    assert list(residual["log_id"]) == ["a", "a"]
    assert residual["phase_corrected_rad"].tolist() == [0.1, 0.2]
    np.testing.assert_allclose(residual["fx_b"], [2.0, 3.0])
    np.testing.assert_allclose(residual["fz_b"], [-1.0, -3.0])
    np.testing.assert_allclose(residual["true_fx_b"], [10.0, 12.0])
    np.testing.assert_allclose(residual["label_fx_b"], [10.0, 12.0])
    np.testing.assert_allclose(residual["prior_fx_b"], [8.0, 9.0])
    assert set(TARGET_COLUMNS).issubset(residual.columns)
    assert residual.attrs["residual_prior_name"] == "delaurier_physical"


def test_align_prior_to_samples_uses_keys_when_prior_rows_are_shuffled() -> None:
    samples = _samples()
    shuffled_prior = _keyed_prior().iloc[[1, 0]].reset_index(drop=True)

    aligned, info = align_prior_to_samples(samples, shuffled_prior)

    assert info["alignment_mode"] == "key_merge"
    np.testing.assert_allclose(aligned["fx_b"], [8.0, 9.0])
    np.testing.assert_allclose(aligned["fz_b"], [-2.0, -1.0])


def test_align_prior_to_samples_rejects_unkeyed_prior_without_explicit_fallback() -> None:
    with np.testing.assert_raises_regex(ValueError, "stable alignment keys"):
        align_prior_to_samples(_samples(), _prior())


def test_build_residual_split_writes_all_split_files(tmp_path: Path) -> None:
    split_root = tmp_path / "split"
    prior_root = tmp_path / "prior"
    output_root = tmp_path / "out"
    split_root.mkdir()
    prior_root.mkdir()
    for split in ("train", "val", "test"):
        _samples().to_parquet(split_root / f"{split}_samples.parquet", index=False)
        _prior(offset=1.0).to_parquet(prior_root / f"{split}_predictions.parquet", index=False)

    build_residual_split(
        split_root,
        prior_root,
        output_root,
        prior_name="delaurier_physical",
        allow_row_order_fallback=True,
    )

    rewritten = pd.read_parquet(output_root / "train_samples.parquet")
    manifest = pd.read_json(output_root / "residual_manifest.json", typ="series")
    np.testing.assert_allclose(rewritten["fx_b"], [1.0, 2.0])
    assert manifest["prior_name"] == "delaurier_physical"
    assert manifest["alignment"]["train"]["alignment_mode"] == "row_order_fallback"
