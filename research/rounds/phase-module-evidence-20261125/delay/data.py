"""Causal history sampled at fixed time, held-frequency or measured cycle lags."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROUND = Path(__file__).resolve().parents[1]
PREVIOUS = ROUND.parent / "periodic-motion-20261125" / "readout"
sys.path.insert(0, str(PREVIOUS))
from model import FlightPredictor

DATA = Path("/home/zn/flap-system-identification/outputs")
CYCLES = np.array([.25, .5, .75, 1.])
CLOCKS = ("fixed", "held", "measured")


def taps(history, main, clock):
    """Interpolate motion in the past. Each stream uses the same valid rows in all arms."""
    signal = history[:, ::-1]
    angle = np.arctan2(signal[..., 13], signal[..., 14])
    lag = -np.unwrap(angle.astype(float), axis=1) / (2 * np.pi)
    lag = lag - lag[:, :1]
    progress = np.diff(lag, axis=1)
    frequency = signal[:, 0, 15]
    good_clock = ((signal[..., 21] > 0).all(1) & (signal[:, 0, 22] > 0)
                  & (progress > 0).all(1) & (progress < .45).all(1)
                  & (lag[:, -1] >= CYCLES[-1]) & (frequency * .5 >= CYCLES[-1]))
    # Out-of-support samples get a common zero + unavailable indicator.
    clock_grid = {"measured": lag, "held": .02 * np.arange(26) * frequency[:, None],
                  "fixed": np.broadcast_to(.02 * np.arange(26) * 3.8, lag.shape)}[clock]
    left = (clock_grid[..., None] < CYCLES).sum(1).clip(1, 25) - 1
    row = np.arange(len(signal))[:, None]
    first = clock_grid[row, left]
    fraction = ((CYCLES - first) /
                np.maximum(clock_grid[row, left + 1] - first, 1e-8)).clip(0, 1)
    values = signal[row, left, :6] + fraction[..., None] * (
        signal[row, left + 1, :6] - signal[row, left, :6])
    velocity = good_clock & main & (signal[..., 16] > 0).all(1) & (signal[..., 18] > 0).all(1)
    rate = good_clock & (signal[..., 17] > 0).all(1)
    flags = np.column_stack([velocity, rate])
    values *= np.repeat(flags, 3, axis=1)[:, None, :]
    return np.column_stack([values.reshape(len(values), -1), flags]).astype("float32")


def load(split, clock):
    assert split in ("train", "validation") and clock in CLOCKS
    cache = DATA / "autopilot_flight_dynamics_20261124"
    history = DATA / "history_slow_fast_20261119_cache"
    with np.load(cache / f"{split}.npz") as files:
        data = {k: files[k] for k in files.files}
    lookup = pd.read_parquet(history / f"{split}_windows.parquet", columns=["window_id", "sampling_group", "flight_id"])
    lookup = lookup.set_index("window_id", verify_integrity=True)
    ix = lookup.index.get_indexer(data["window"])
    assert (ix >= 0).all() and np.array_equal(lookup.flight_id.to_numpy()[ix], data["flight"])
    h = np.load(history / f"{split}_history.npy", mmap_mode="r")
    main = lookup.sampling_group.to_numpy()[ix] == "main_50hz"
    lag = []
    for i in range(0, len(ix), 2048):
        s = slice(i, i + 2048)
        raw = h[ix[s]]
        np.testing.assert_array_equal(raw[:, ::5].reshape(len(raw), -1), data["hist"][s])
        lag.append(taps(raw, main[s], clock))
    data["taps"] = np.concatenate(lag)
    extra = np.load(DATA / "autopilot_periodic_motion_20261125" / f"{split}_features.npy")
    data["hist"] = np.column_stack([data["hist"], extra])
    return data


@torch.inference_mode()
def corrected(data, seed):
    filename = PREVIOUS / "original_runs" / f"incumbent{seed}" / "corrected.pt"
    state = torch.load(filename, map_location="cpu", weights_only=True)
    model = FlightPredictor(state["mean"], state["std"])
    model.load_state_dict(state)
    model.eval()
    pred = []
    for i in range(0, len(data["hist"]), 1024):
        inputs = [torch.from_numpy(np.ascontiguousarray(data[k][i:i + 1024]))
                  for k in ("hist", "pooled", "u0", "time")]
        pred.append(model(*inputs).numpy())
    return np.concatenate(pred)
