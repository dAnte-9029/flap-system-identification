"""Replot the first H4 waveform experiment from its published flight-level MSE tables."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WAVE = HERE.parent / "waveform"
SEEDS = [17, 29, 43]
ARMS = ["corrected", "nominal", "sham", "origin"]
NAVY, TEAL, GRAY = "#284266", "#007b70", "#737b84"


def paired(table, reference, metric, archive):
    pivot = table[table.arm.isin([reference, "full"])].pivot(
        index=["arm", "seed"], columns="flight", values=metric
    ).sort_index(axis=1)
    assert not pivot.isna().any().any()
    old, full = (pivot.loc[arm].loc[SEEDS].to_numpy() for arm in (reference, "full"))
    gain = np.sqrt(old.mean(1)) - np.sqrt(full.mean(1))
    flights = old.shape[1]
    draws = np.random.default_rng(17).integers(0, flights, (10000, flights))
    sampled = (np.sqrt(old[:, draws].mean(2)) - np.sqrt(full[:, draws].mean(2))).mean(0)
    lo, hi = np.quantile(sampled, [0.025, 0.975])
    saved = next(row for row in archive if row["reference"] == reference and row["metric"] == metric)
    np.testing.assert_allclose([gain.mean(), lo, hi], [saved["gain"], saved["lo"], saved["hi"]], atol=1e-12)
    assert saved["flights"] == flights
    return gain, lo, hi


def draw_point(ax, y, gain, lo, hi, color, with_seeds=False):
    mean = gain.mean()
    ax.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]], fmt="o", color=color,
                markersize=6, elinewidth=1.8, capsize=4, zorder=3)
    if with_seeds:
        ax.scatter(gain, [y-.23]*len(gain), s=18, marker="|", color=GRAY, zorder=2)


def format_axis(ax, title, ticks, fmt):
    ax.set_title(title, fontweight="bold", pad=14, fontsize=15)
    ax.axvline(0, color="#687786", linewidth=1.1, linestyle=(0,(3,3)))
    ax.grid(axis="x", color="#e2e6ea", linewidth=.7)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter(fmt))
    ax.tick_params(axis="both", length=0, pad=5)
    ax.spines[["left","top","right"]].set_visible(False)
    ax.spines["bottom"].set_color("#a1a9b2")


def endpoints(results):
    table = pd.read_csv(WAVE / "flight_mse.csv")
    fig, axes = plt.subplots(1,3,figsize=(11.9,4.7),sharey=True)
    y = np.arange(3,-1,-1)
    settings = [
        ("w05","0.5 秒",[-.005,0,.005,.010,.015,.020],(-.007,.022)),
        ("w1","1 秒",[0,.003,.006,.009],(-.003,.0115)),
        ("w2","2 秒",[-.0015,-.001,-.0005,0,.0005],(-.0017,.0008))
    ]
    for ax,(metric,title,ticks,limits) in zip(axes,settings):
        for arm, ypos in zip(ARMS,y):
            points = paired(table, arm, metric, results["endpoints"]["contrasts"])
            draw_point(ax, ypos, *points, color=NAVY if arm=="corrected" else TEAL, with_seeds=True)
        format_axis(ax, title, ticks, "%.4f" if metric=="w2" else "%.3f")
        ax.set_xlim(*limits)
    axes[0].set_yticks(y,["上轮一阶模型","恒频 3.8 Hz 重训","错位相位重训","仅拟合起点"])
    axes[0].set_ylim(-.65,3.55)
    fig.subplots_adjust(left=.16,right=.985,top=.76,bottom=.30,wspace=.15)
    fig.suptitle("二、三阶波形输运对角速度端点预测的贡献", y=.975, fontsize=18, fontweight="bold")
    fig.text(.53,.16,"角速度 RMSE 改善量（对照 − 完整模型，rad/s）",ha="center",fontsize=12)
    fig.text(.025,.035,"H4 第一轮  ·  39 个开发架次  ·  圆点为三个种子的平均，灰色短线为单种子结果。\n误差线为 10,000 次架次配对重采样的 95% 区间。各时域的横轴刻度独立。",fontsize=10,color="#475361")
    fig.savefig(HERE/"h4_endpoint_gain.png", dpi=190)
    plt.close(fig)


def native(results):
    table = pd.read_csv(WAVE / "native_flight_mse.csv")
    table = table[table.ch == 1]
    rows = [("measurement","corrected"),("measurement","nominal"),("higher","corrected"),("higher","nominal")]
    y = [3.3,2.6,1.1,.4]
    settings = [
        ("early","0–0.5 秒",[-.02,0,.04,.08,.12,.16],(-.024,.152)),
        ("late","1–2 秒",[0,.004,.008,.012],(-.002,.014)),
    ]
    fig,axes = plt.subplots(1,2,figsize=(10.5,4.9),sharey=True)
    for ax,(period,title,ticks,limits) in zip(axes,settings):
        cov = next(r for r in results["native_coverage"] if r["ch"]==1 and r["period"]==period)
        for (metric,arm),ypos in zip(rows,y):
            selected = table[(table.period==period)&(table.metric==metric)].rename(columns={"mse":"value"})
            selected[metric] = selected["value"]
            saved = results["native"][f"1_{period}_{metric}"]["contrasts"]
            points = paired(selected,arm,metric,saved)
            draw_point(ax, ypos, *points,color=NAVY if metric=="measurement" else TEAL)
        ax.axhline(1.85,color="#e2e6ea",linewidth=.8)
        format_axis(ax, f'{title}（{cov["flights"]} 架，{cov["retained_beats"]} 个拍窗）', ticks, "%.03f" if period == "late" else "%.02f")
        ax.set_xlim(*limits)
    axes[0].set_yticks(y,["实测误差 · 对上轮一阶","实测误差 · 对恒频重训","高阶投影 · 对上轮一阶","高阶投影 · 对恒频重训"])
    axes[0].set_ylim(-.1,3.85)
    fig.subplots_adjust(left=.215,right=.985,top=.75,bottom=.30,wspace=.13)
    fig.suptitle("完整目标翼拍上的角速度误差", y=.98, fontsize=18, fontweight="bold")
    fig.text(.55,.165,"角速度 RMSE 改善量（对照 − 完整模型，rad/s）",ha="center",fontsize=12)
    fig.text(.025,.03,"H4 第一轮  ·  高阶投影先去除线性趋势和一阶谐波，再测二、三阶空间内的误差。\n圆点为三个种子的平均，误差线为 10,000 次架次配对重采样的 95% 区间。两列横轴刻度独立。",fontsize=10,color="#475361")
    fig.savefig(HERE/"h4_native_gain.png",dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    for family in ["Noto Sans CJK SC","Noto Sans CJK JP","Droid Sans Fallback"]:
        if family in fonts:
            plt.rcParams["font.sans-serif"] = [family,"DejaVu Sans"]
            break
    plt.rcParams.update({"font.size":11,"axes.unicode_minus":False,"figure.facecolor":"white"})
    results = json.loads((WAVE/"results.json").read_text())
    endpoints(results)
    native(results)
