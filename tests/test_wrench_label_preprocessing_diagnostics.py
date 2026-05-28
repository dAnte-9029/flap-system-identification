from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.diagnose_wrench_label_preprocessing import (
    compute_channel_metrics,
    run_diagnostics,
)


def _frame(scale: float = 1.0) -> pd.DataFrame:
    time_s = np.arange(120, dtype=float) * 0.01
    base = np.sin(2.0 * np.pi * 1.0 * time_s)
    spike = np.zeros_like(base)
    spike[30] = 10.0
    values = scale * (base + spike)
    return pd.DataFrame(
        {
            "time_s": time_s,
            "log_id": "log_a",
            "fx_b": values,
            "fy_b": 0.5 * values,
            "fz_b": 2.0 * values,
            "mx_b": 0.01 * values,
            "my_b": 0.02 * values,
            "mz_b": 0.03 * values,
        }
    )


def _write_split(root: Path, frame: pd.DataFrame) -> None:
    root.mkdir()
    for split in ("train", "val", "test"):
        frame.to_parquet(root / f"{split}_samples.parquet", index=False)


def test_compute_channel_metrics_reports_spike_and_similarity():
    raw = _frame()
    clean = _frame()
    clean["fx_b"] = np.sin(2.0 * np.pi * 1.0 * clean["time_s"].to_numpy())

    metrics = compute_channel_metrics(raw, clean, split="test", channel="fx_b")

    assert metrics["split"] == "test"
    assert metrics["channel"] == "fx_b"
    assert metrics["sample_count"] == len(raw)
    assert metrics["p99_abs_clean"] < metrics["p99_abs_raw"]
    assert "highpass_energy_fraction_raw_8p0hz" in metrics


def test_run_diagnostics_writes_split_and_log_metrics(tmp_path: Path):
    raw_root = tmp_path / "raw"
    clean_root = tmp_path / "clean"
    output_dir = tmp_path / "diag"
    raw = _frame()
    clean = _frame()
    clean["fx_b"] = np.sin(2.0 * np.pi * 1.0 * clean["time_s"].to_numpy())
    _write_split(raw_root, raw)
    _write_split(clean_root, clean)

    outputs = run_diagnostics(raw_split_root=raw_root, clean_split_root=clean_root, output_dir=output_dir)

    assert Path(outputs["split_metrics"]).exists()
    assert Path(outputs["log_metrics"]).exists()
    assert Path(outputs["report"]).exists()
    metrics = pd.read_csv(outputs["split_metrics"])
    assert set(metrics["channel"]) == {"fx_b", "fy_b", "fz_b", "mx_b", "my_b", "mz_b"}
