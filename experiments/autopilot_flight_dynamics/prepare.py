"""Attach eight native observation times and their causal command integrals to the frozen flight windows."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import yaml

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
PREVIOUS = HERE.parent / "autopilot_command_reference"
sys.path.insert(0, str(PREVIOUS))
from prepare import HISTORY, ROOT, TAPES, digest, pool_commands

CACHE = ROOT / "outputs/autopilot_flight_dynamics_20261124"
HORIZONS = np.arange(1, 9) * 0.25


def prepare_split(part, prior):
    old = prior["split"][part]
    assert digest(Path(old["file"])) == old["sha256"]
    windows = pd.read_parquet(HISTORY / f"{part}_windows.parquet").set_index("window_id", verify_integrity=True)
    selected = pd.read_parquet(TAPES / f"{part}_ids.parquet").window_id
    indices = windows.index.get_indexer(selected)
    assert selected.is_unique and (indices >= 0).all()
    groups = windows.iloc[indices]
    init, times, outputs, valid = [
        np.load(HISTORY / f"{part}_{key}.npy", mmap_mode="r") for key in ("initial", "t", "y", "valid")
    ]
    commands = np.load(TAPES / f"{part}_commands.npy", mmap_mode="r")
    count = len(indices)
    assert count == len(commands) == old["windows"]
    dense = dict(time=np.empty((count, 8, 2), np.float32), y=np.empty((count, 8, 2, 3), np.float32),
                 mask=np.empty((count, 8, 2), bool), pooled=np.empty((count, 8, 2, 16), np.float32))
    for start in range(0, count, 384):
        batch = slice(start, min(start + 384, count))
        ix = indices[batch]
        row = np.arange(len(ix))
        initial, yt, is_valid, y = init[ix], times[ix], valid[ix], outputs[ix]
        tolerance = np.where(groups.sampling_group.iloc[batch] == "main_50hz", 0.035, 0.105)
        rotation = Rotation.from_quat(initial[:, [7, 8, 9, 6]])
        for step, horizon in enumerate(HORIZONS):
            for stream, channel in enumerate((0, 2)):
                distance = np.where(is_valid[:, :, channel], abs(yt[:, :, channel] - horizon), np.inf)
                nearest = distance.argmin(axis=1)
                ok = distance[row, nearest] <= tolerance
                t = np.where(ok, yt[row, nearest, channel], horizon)
                target = (rotation.apply(y[row, nearest, 3:6] - initial[:, 3:6], inverse=True)
                          if stream == 0 else y[row, nearest, 10:13] - initial[:, 10:13])
                dense["time"][batch, step, stream] = t
                dense["mask"][batch, step, stream] = ok
                dense["y"][batch, step, stream] = np.where(ok[:, None], target, 0)
        t = dense["time"][batch]
        tape = commands[batch]
        assert (np.diff(t, axis=1) > 0).all()
        assert (t.max((1, 2)) <= tape[..., 4].sum((1, 2)) * 1e-6 + 1e-6).all()
        dense["pooled"][batch] = pool_commands(tape, t.reshape(len(ix), -1)).reshape(len(ix), 8, 2, 16)
    with np.load(old["file"], allow_pickle=False) as before:
        np.testing.assert_array_equal(before["window"], selected.to_numpy(dtype=str))
        np.testing.assert_array_equal(before["flight"], groups.flight_id.to_numpy(dtype=str))
        for key in dense:
            actual = dense[key][:, [1, 3, 7]].reshape(before[key].shape)
            np.testing.assert_allclose(actual, before[key], atol=2e-6, rtol=0)
        path = CACHE / f"{part}.npz"
        if path.exists():
            raise FileExistsError(path)
        np.savez(path, **dense, hist=before["hist"], u0=before["u0"], flight=before["flight"], window=before["window"])
    return dict(file=str(path), sha256=digest(path), windows=count, flights=int(groups.flight_id.nunique()),
                native_counts=dense["mask"].sum(0).tolist())


if __name__ == "__main__":
    record = json.loads((PREVIOUS / "data.json").read_text())
    registry = yaml.safe_load((PROJECT / "configs/data/velocity_rate_dataset_registry.yaml").read_text())
    selected = registry["datasets"][record["dataset"]]
    assert digest(ROOT / selected["manifest_path"]) == selected["manifest_sha256"] == record["manifest_sha256"]
    CACHE.mkdir(exist_ok=True)
    splits = {part: prepare_split(part, record) for part in ("train", "validation")}
    with np.load(splits["train"]["file"]) as a, np.load(splits["validation"]["file"]) as b:
        assert set(a["flight"]).isdisjoint(b["flight"])
    (HERE / "data.json").write_text(json.dumps(dict(dataset=record["dataset"], manifest_sha256=record["manifest_sha256"],
                                                    split=splits), indent=2) + "\n")
