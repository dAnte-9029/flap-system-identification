"""Shared target schema, deterministic seeding, and split-frame preparation."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DEFAULT_TARGET_COLUMNS = ["fx_b", "fy_b", "fz_b", "mx_b", "my_b", "mz_b"]


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_split_frame(split_root: str | Path, split_name: str, max_samples: int | None, sample_seed: int) -> pd.DataFrame:
    path = Path(split_root) / f"{split_name}_samples.parquet"
    frame = pd.read_parquet(path)
    if max_samples is not None and len(frame) > max_samples:
        frame = frame.sample(n=max_samples, random_state=sample_seed).reset_index(drop=True)
    return frame
