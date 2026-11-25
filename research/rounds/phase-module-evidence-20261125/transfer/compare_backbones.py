"""Aggregate one paired seed across all three backbones on identical native targets."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
BASES = ("recurrent", "tcn", "ode")
LABELS = ("已有递推动力学", "时序卷积网络", "受控 Neural ODE")
METRICS = ("v05", "w05", "v1", "w1", "v2", "w2")


def summarize():
    source = pd.read_csv(ROUND / "waveform/coherence/flight_mse.csv")
    source = source[source.seed == 17].set_index(["arm", "flight"])
    ids = sorted(source.loc["corrected"].index)
    assert len(ids) == 39
    draws = np.random.default_rng(17).integers(0, len(ids), (10000, len(ids)))
    reports = {}
    external = {key: json.loads((HERE / f"{key}_result.json").read_text())["flight_mse"]
                for key in BASES[1:]}
    for base in BASES:
        reports[base] = {}
        for metric in METRICS:
            if base == "recurrent":
                old, new = (source.loc[k].loc[ids, metric].to_numpy() for k in ("corrected", "conditional"))
            else:
                pair = [external[base][tag][metric] for tag in (base, "second_round")]
                assert all(sorted(group) == ids for group in pair)
                old, new = (np.array([group[f] for f in ids]) for group in pair)
            rmse_old, rmse_new = np.sqrt(old.mean()), np.sqrt(new.mean())
            gain = np.sqrt(old[draws].mean(1)) - np.sqrt(new[draws].mean(1))
            reports[base][metric] = dict(base=rmse_old, with_head=rmse_new, gain=rmse_old-rmse_new,
                    gain_pct=100*(rmse_old-rmse_new)/rmse_old, gain_ci95=np.quantile(gain, [.025, .975]).tolist(),
                    wins=int(np.count_nonzero(old > new)))
    return reports


def plot(scores):
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Noto Sans CJK SC","Noto Sans CJK JP","DejaVu Sans"],
                         "axes.spines.top":False,"axes.spines.right":False,"axes.unicode_minus":False})
    fig, ax = plt.subplots(figsize=(7.6,3.8),layout="constrained")
    ax.axhline(0, color="#77838a", lw=.8)
    for i,(base, color) in enumerate(zip(BASES, ["#136f63", "#31549a", "#ad591c"])):
        vals = [scores[base][m] for m in ("w05","w1","w2")]
        y = np.array([v["gain"] for v in vals])
        bounds = np.array([v["gain_ci95"] for v in vals])
        ax.errorbar(np.arange(3)+(i-1)*.12,y,yerr=[y-bounds[:,0], bounds[:,1]-y],
                    lw=1.5, fmt="o", capsize=4, color=color, label=LABELS[i])
    ax.set(xticks=np.arange(3),xticklabels=["0.5 秒","1 秒","2 秒"],
           ylabel="机体角速度 RMSE 改善 (rad/s)",xlabel="预测时域")
    ax.legend(ncols=3,frameon=False)
    ax.grid(axis="y",alpha=.18)
    fig.savefig(HERE / "backbone_gain.png",dpi=170)


if __name__ == "__main__":
    scores = summarize()
    (HERE / "backbone_comparison.json").write_text(json.dumps(scores, indent=2) + "\n")
    plot(scores)
