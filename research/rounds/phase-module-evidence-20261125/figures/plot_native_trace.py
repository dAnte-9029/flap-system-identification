"""Plot the previously designated native pitch-rate window with and without wave transport."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
SOURCE = Path("/home/zn/flap-system-identification")
sys.path.insert(0, str(ROUND / "waveform"))
from experiment import FlightPredictor, PRIOR, extract, pool_commands
sys.path.insert(0, str(ROUND / "waveform/coherence"))
from inference import HarmonicTransport, causal_conditions


@torch.inference_mode()
def example():
    diagnostics = SOURCE / "research/rounds/periodic-motion-20261125/diagnostics"
    chosen = json.loads((diagnostics / "original_component_example.json").read_text())
    wi = chosen["cache_window_index"]
    with np.load(diagnostics / "reference.npz") as z:
        sel = np.flatnonzero((z["window_idx"] == wi) & (z["ch"] == 1))
        sel = sel[np.argsort(z["time"][sel])]
        times, blocks, measured = z["time"][sel], z["block"][sel], z["data"][sel, 1]
        assert np.all(z["flight_ids"][z["flight_index"][sel]] == chosen["flight"])
    with np.load(diagnostics / "original_corrected17_points.npz") as p:
        old = p["slow_abs"][sel, 1] + p["fast_abs"][sel, 1]
    data = SOURCE / "outputs"
    with np.load(data / "autopilot_flight_dynamics_20261124/validation.npz") as z:
        key = z["window"][wi]
        hist, u = z["hist"][wi], torch.tensor(z["u0"][wi:wi+1])
    past = data / "history_slow_fast_20261119_cache"
    meta = pd.read_parquet(past / "validation_windows.parquet", columns=["window_id", "sampling_group"])
    row = meta.index[meta.window_id == key].item()
    raw = np.load(past / "validation_history.npy", mmap_mode="r")[row:row+1]
    coef, _ = extract(raw, np.array([meta.iloc[row].sampling_group == "main_50hz"]))
    extra = np.load(data / "autopilot_periodic_motion_20261125/validation_features.npy", mmap_mode="r")[wi]
    h = torch.tensor(np.r_[hist, extra][None])
    tape = np.load(data / "controlled_slow_fast_20261119_tapes/validation_commands.npy", mmap_mode="r")[wi:wi+1]
    t = torch.tensor(times[None, :, None], dtype=torch.float32).expand(-1, -1, 2)
    pool = torch.tensor(pool_commands(tape, times[None]))[:, :, None, :].expand(-1, -1, 2, -1)
    state = torch.load(PRIOR / "readout/original_runs/incumbent17/corrected.pt",
                       weights_only=True, map_location="cpu")
    network = FlightPredictor(state["mean"], state["std"])
    network.load_state_dict(state)
    theta = network.clocks(h, pool, u, t)[..., -1]
    head = HarmonicTransport.from_files(17)
    delta = head(torch.from_numpy(coef), t, theta, causal_conditions(h, pool, u, t))[0, :, 1, 1].numpy()
    return times, blocks, measured, old, delta


if __name__ == "__main__":
    t, blocks, obs, old, change = example()
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Noto Sans CJK SC","Noto Sans CJK JP","DejaVu Sans"],
                         "axes.spines.top":False,"axes.spines.right":False})
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9.2,4.8), layout="constrained",
                             gridspec_kw={"height_ratios":[2,1]})
    for ax in axes:
        ax.grid(axis="y", alpha=.19)
    for j, group in enumerate(np.split(np.arange(len(t)), np.flatnonzero(np.diff(t) > .045) + 1)):
        label = lambda s: s if j == 0 else None
        axes[0].plot(t[group], obs[group], "o-", color="#30373b", markersize=2.7, linewidth=1.2, label=label("原生测量"))
        axes[0].plot(t[group], old[group], "--", color="#bb792c", linewidth=1.4, label=label("原一阶模型"))
        axes[0].plot(t[group], (old + change)[group], color="#117a75", linewidth=1.6, label=label("加入二、三阶输运"))
        axes[1].plot(t[group], change[group], color="#117a75", linewidth=1.4)
    axes[1].axhline(0, color="gray", linewidth=.6)
    axes[0].set(ylabel="俯仰角速度 (rad/s)", title="在上一轮预选的同一实飞窗口预测拍内运动")
    axes[1].set(ylabel="高阶预测增量 (rad/s)", xlabel="距预测起点的时间 (s)")
    axes[0].legend(ncols=3,frameon=False)
    fig.savefig(HERE / "native_pitch_trace.png", dpi=165)
