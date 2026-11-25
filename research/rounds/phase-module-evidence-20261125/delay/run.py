"""Fit direct multi-horizon NARX and the identical residual readout on the retained predictor."""
import json
import os
from pathlib import Path
import sys

for library in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[library] = "5"

import numpy as np
import torch

from data import CLOCKS, corrected, load, taps
from arm import CLOCK

WIDTH = 256
RIDGE = .001
QUERIES = (1, 3, 7)
NAMES = ("v05", "w05", "v1", "w1", "v2", "w2")


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def normalise(raw, center, scale):
    # Presence columns and near-constant sensors retain bounded scale.
    return (raw - center) / scale


def design(standard, seed):
    rng = np.random.default_rng(seed)
    projection = rng.standard_normal((standard.shape[1], WIDTH), dtype=np.float32)
    projection /= np.sqrt(standard.shape[1])
    bias = rng.standard_normal(WIDTH, dtype=np.float32)
    hidden = np.tanh(standard @ projection + bias)
    return np.column_stack([np.ones(len(standard), np.float32), standard, hidden])


def inputs(data, horizon, stream):
    return np.column_stack([data["hist"], data["taps"], data["u0"],
                            data["pooled"][:, horizon, stream], data["time"][:, horizon, stream]])


def fit(seed, filename):
    d = load("train", CLOCK)
    assert len(set(d["flight"])) == 121
    anchor = corrected(d, seed)
    centers, scales, weights, residuals = [], [], [], []
    for j in range(8):
        for c in range(2):
            valid = d["mask"][:, j, c].astype(bool)
            raw = inputs(d, j, c)[valid]
            center = raw.mean(0, dtype=np.float64).astype(np.float32)
            scale = raw.std(0, dtype=np.float64).clip(.025).astype(np.float32)
            z = design(normalise(raw, center, scale), seed).astype(np.float64)
            y = d["y"][valid, j, c]
            targets = np.column_stack([y, y - anchor[valid, j, c]]).astype(np.float64)
            normal, rhs = z.T @ z, z.T @ targets
            normal.flat[::len(normal) + 1] += len(y) * RIDGE
            normal[0, 0] -= len(y) * RIDGE
            w = np.linalg.solve(normal, rhs)
            residuals.append(float(np.max(np.abs(normal @ w - rhs)) / np.max(np.abs(rhs))))
            centers.append(center)
            scales.append(scale)
            weights.append(w)
    assert max(residuals) < 1e-9
    with Path(filename).open("wb") as output:
        np.savez_compressed(output, clock=CLOCK, seed=seed, center=centers, scale=scales,
                            weights=weights)
    write_json(os.environ["RA_TRAIN_INFO"], dict(steps=16, samples=int(d["mask"].sum()),
               max_stationarity=max(residuals), windows=len(d["hist"]),
               taps_available_velocity_rate=d["taps"][:, -2:].sum(0).tolist()))


def evaluate(seed, filename):
    with np.load(filename) as f:
        clock, trained_seed = f["clock"].item(), f["seed"].item()
        centers, scales, weights = [f[k] for k in ("center", "scale", "weights")]
    assert clock in CLOCKS and trained_seed == seed
    d = load("validation", clock)
    ids = np.unique(d["flight"])
    assert len(ids) == 39
    anchor = corrected(d, seed)
    flight = {tag: {} for tag in ("direct", "residual", "corrected")}
    metrics = {}
    for qi, j in enumerate(QUERIES):
        for c in range(2):
            index = 2 * j + c
            z = design(normalise(inputs(d, j, c), centers[index], scales[index]), seed)
            pred = z @ weights[index]
            ref = d["y"][:, j, c]
            old = anchor[:, j, c]
            key = NAMES[2 * qi + c]
            for tag, estimate in (("direct", pred[:, :3]), ("residual", old + pred[:, 3:]), ("corrected", old)):
                sq = ((estimate - ref) ** 2).sum(1)
                scores = {fid: float(sq[(d["flight"] == fid) & d["mask"][:, j, c].astype(bool)].mean()) for fid in ids}
                flight[tag][key] = scores
                if tag == "residual":
                    metrics[key] = float(np.sqrt(np.mean(list(scores.values()))))
    write_json(Path(os.environ["RA_OUTPUT"]) / "flight_mse.json", flight)
    write_json(Path(os.environ["RA_OUTPUT"]) / "coverage.json", {
        "windows": len(d["hist"]), "available_velocity_rate": d["taps"][:, -2:].sum(0).tolist()})
    write_json(os.environ["RA_METRICS"], metrics)


def preflight():
    # With constant 3.8 Hz the three arms must be identical.
    h = np.ones((2, 26, 30), np.float32)
    t = .02 * (np.arange(26) - 25)
    h[:, :, :6] = t[None, :, None]
    theta = 2 * np.pi * 3.8 * t
    h[:, :, 13:16] = np.stack([np.sin(theta), np.cos(theta), np.full_like(t, 3.8)], axis=1)
    for arm in CLOCKS:
        out = taps(h, np.array([True, False]), arm)
        np.testing.assert_allclose(out[0, :24:6], -np.arange(1, 5) / (4 * 3.8), atol=2e-6)
        np.testing.assert_array_equal(out[1, -2:], [0, 1])
    h[:, -1, 21] = 0
    np.testing.assert_array_equal(taps(h, np.ones(2, bool), CLOCK), np.zeros((2, 26)))


if __name__ == "__main__":
    torch.set_num_threads(5)
    stage = sys.argv[1]
    if stage == "preflight":
        preflight()
    else:
        seed = int(os.environ["RA_SEED"])
        {"train": fit, "evaluate": evaluate}[stage](seed, os.environ["RA_MODEL"])
