"""Chinese report and standalone figures for paired training-horizon experiments."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_velocity_rate_horizons import OUT, OLD, LONG
from evaluate_velocity_rate_horizon_training import METRICS, label
from report_velocity_rate_learning import table

LABELS = ["位置误差 (m)", "NED速度误差 (m/s)", "姿态测地角误差 (°)", "FRD角速度误差 (rad/s)"]
COLORS = {0: "#777777", 200: "#2878b5", 500: "#dd7622", 1000: "#24834c"}
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False, "font.size": 10})


def figures():
    folder = OUT / "figures"
    folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row, horizon in enumerate((500, 1000)):
        for col, model in enumerate(("GRU-NoControl", "GRU-Control")):
            ax = axes[row, col]
            for seed, color in zip((17, 29, 43), ("#2878b5", "#dd7622", "#24834c")):
                records = pd.read_json(OUT / f"train_{horizon}ms/{model}/seed_{seed}/training.jsonl", lines=True)
                ax.plot(records.epoch, records.validation_flight_loss, color=color, label=f"seed{seed} 验证")
                ax.plot(records.epoch, records.train_window_loss, color=color, ls="--", alpha=.6)
            ax.set(title=f"{horizon}ms · {model}", xlabel="epoch", ylabel="归一化全步损失")
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("实线：验证航次等权；虚线：训练窗口等权；不同训练时域的损失不直接作性能排名")
    fig.tight_layout()
    fig.savefig(folder / "learning_curves.png", dpi=160)
    plt.close(fig)
    native = pd.read_csv(OUT / "native_steps_per_flight.csv")
    flight = pd.read_csv(OUT / "per_flight.csv")
    for model in ("GRU-NoControl", "GRU-Control"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, metric, title in zip(axes.flat, METRICS, LABELS):
            for horizon in (0, 200, 500, 1000):
                condition = "Kinematic Baseline" if horizon == 0 else label(horizon, model)
                g = native[native.condition == condition].groupby(["flight_id", "native_step"])[[metric, "actual_time_s"]].mean().reset_index()
                stats = g.groupby("native_step")[metric].agg(["mean", "std"])
                t = g.groupby("native_step").actual_time_s.mean()
                ax.plot(t, stats["mean"], color=COLORS[horizon],
                        label="Baseline" if horizon == 0 else f"训练 {horizon} ms")
                ax.fill_between(t, np.maximum(0, stats["mean"]-stats["std"]), stats["mean"]+stats["std"],
                                color=COLORS[horizon], alpha=.09)
            for x in (.5, 1):
                ax.axvline(x, color="black", alpha=.25, ls="--", lw=.8)
            ax.set(xlabel="自主递推时间 (s)", ylabel=title)
            ax.grid(alpha=.2)
        axes[0, 0].legend()
        fig.suptitle(f"{model}：航次等权均值 ± 航次间样本标准差；三个种子先平均")
        fig.tight_layout()
        fig.savefig(folder / f"error_growth_{model}.png", dpi=160)
        plt.close(fig)
    conditions = ["Kinematic Baseline"] + [label(h, m) for h in (200, 500, 1000) for m in ("GRU-NoControl", "GRU-Control")]
    fig, axes = plt.subplots(3, 4, figsize=(17, 11))
    for row, horizon in enumerate((500, 1000, 2000)):
        for col, (metric, title) in enumerate(zip(METRICS, LABELS)):
            ax = axes[row, col]
            vals = [flight[(flight.condition == c) & (flight.horizon_ms == horizon)].groupby("flight_id")[metric].mean().to_numpy()
                    for c in conditions]
            ax.boxplot(vals, tick_labels=["Base", "200N", "200C", "500N", "500C", "1sN", "1sC"])
            for i, v in enumerate(vals):
                ax.scatter(np.full(len(v), i+1)+np.linspace(-.12, .12, len(v)), v, s=5, color="#2878b5", alpha=.5)
            ax.set(title=f"{horizon/1000:g}s · {title}")
            ax.tick_params(axis="x", labelsize=8)
            ax.grid(axis="y", alpha=.2)
    fig.suptitle("27条未见航次的端点误差分布；N=NoControl，C=Control，前缀为训练时域")
    fig.tight_layout()
    fig.savefig(folder / "flight_distributions.png", dpi=160)
    plt.close(fig)
    old = np.load(LONG / "trajectory_examples.npz")
    meta = pd.read_csv(LONG / "example_windows.csv")
    for horizon in (500, 1000):
        ex = np.load(OUT / f"trajectory_examples_{horizon}ms.npz")
        fig = plt.figure(figsize=(14, 9))
        for i in range(6):
            ax = fig.add_subplot(2, 3, i+1, projection="3d")
            valid = ex["future_mask"][i]
            start = ex["initial"][i, :3]
            paths = [
                ("观测", ex["target"][i, valid, :3], "black"),
                ("Baseline", old["Kinematic_Baseline_seed_-1"][i, valid, :3], "#777777"),
                ("NoControl", ex[f"H{horizon}-GRU-NoControl_seed_17"][i, valid, :3], "#2878b5"),
                ("Control", ex[f"H{horizon}-GRU-Control_seed_17"][i, valid, :3], "#dd7622"),
            ]
            for name, p, color in paths:
                p = np.vstack((start, p))
                ax.plot(p[:, 1], p[:, 0], -p[:, 2], color=color, label=name, lw=1.8)
            ax.set(xlabel="东 (m)", ylabel="北 (m)", zlabel="上 (m)", title=f"航次 {meta.flight_id.iloc[i][:10]}")
        fig.axes[0].legend(fontsize=8)
        fig.suptitle(f"训练{horizon}ms → 连续预测2s；固定六航次中间窗口，种子17，无真值重置")
        fig.tight_layout()
        fig.savefig(folder / f"trajectories_train_{horizon}ms.png", dpi=160)
        plt.close(fig)
    seed = pd.read_csv(OUT / "per_seed.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric, title in zip(axes.flat, METRICS, LABELS):
        for horizon in (200, 500, 1000):
            for s in (17, 29, 43):
                a = seed[(seed.condition == label(horizon, "GRU-NoControl")) & (seed.seed == s)].set_index("horizon_ms")[metric]
                b = seed[(seed.condition == label(horizon, "GRU-Control")) & (seed.seed == s)].set_index("horizon_ms")[metric]
                ax.plot(a.index/1000, 100*(a-b)/a, color=COLORS[horizon], alpha=.6,
                        marker={17: "o", 29: "s", 43: "^"}[s],
                        label=f"{horizon}ms seed{s}")
        ax.axhline(0, color="black", lw=.8)
        ax.set(xlabel="预测时域 (s)", ylabel="Control相对改善 (%)", title=title)
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(folder / "control_gain_by_seed.png", dpi=160)
    plt.close(fig)


def report():
    summary = pd.read_csv(OUT / "summary.csv")
    budget = pd.read_csv(OUT / "training_budget.csv")
    pairs = pd.read_csv(OUT / "paired_comparisons.csv")
    norm = json.loads((OUT / "loss_scale_audit.json").read_text())
    interpretation = (OUT / "interpretation_zh.md").read_text() if (OUT / "interpretation_zh.md").exists() else ""
    rows = []
    order = ["Kinematic Baseline"] + [label(h, m) for h in (200, 500, 1000) for m in ("GRU-NoControl", "GRU-Control")]
    for h in (500, 1000, 2000):
        for condition in order:
            f = summary[(summary.condition == condition) & (summary.horizon_ms == h)].set_index("metric")
            rows.append([h/1000, condition] + [f'{f.loc[m, "flight_mean"]:.4f} ± {f.loc[m, "flight_std"]:.4f}' for m in METRICS])
    lines = [
        "# GRU64：500ms与1s多步损失训练，验证至2s", "",
        interpretation, "",
        "## 实际完成的实验", "",
        "分别训练500ms、1s两组模型；每组包含GRU-NoControl与GRU-Control，固定种子17、29、43，共12次正式GPU训练。每次从相同随机初始化规则开始，并非从200ms checkpoint微调。测试集持续封存；没有MPC、PX4修改或控制可用性认证。", "",
        "训练从过去500ms历史编码出隐藏状态，之后连续自主递推。500ms组在约25步、1s组在约50步的**全部有效未来步**计算损失；原生dt和偶发26/51步保留。没有teacher forcing、分段真值重置或未来相位/频率输入。2s仅用于验证分析，不用于checkpoint选择。", "",
        "## 数据、输入与公平性", "",
        "数据按configs/data/velocity_rate_dataset_registry.yaml解析velocity_rate_multistep_v1/main_50hz，manifest及授权train/validation样本、窗口、执行器事件哈希已验证。见data_audit.json和run_manifest.json。", "",
        "- 两个新训练时域共享107,920个窗口、94条训练飞行。验证选择共享36,332个窗口、27条飞行。它们是1s注册起点与500ms注册起点的交集，保留100ms起点步长。",
        "- 共享缓存和train/validation_windows.parquet保留1s注册窗口元数据；count_500ms和count_1000ms记录两个实际裁剪端点。500ms训练在送入GPU前按count_500ms裁剪目标、dt与事件序列并设置mask，不读取后半段控制作为输入。",
        "- 最终各模型使用冻结长时域评估中的35,946个共同起点、27条验证飞行，统一评估500ms/1s/2s。旧200ms模型和基线沿用已校验的同起点结果。",
        "- 两模型都使用历史速度、角速度、姿态、有效扑翼观测、mask及实际dt；姿态采用旋转矩阵特征，无四元数符号二义性。位置仅作为积分初值。",
        "- Control额外使用历史四路命令和未来原生控制事件序列，保留区间内命令变化。NoControl不读历史或未来控制。",
        "- 未来频率保持起点有效值，相位按它因果外推；无效观测掩码保留。未来真实运动状态只用于训练目标或验证误差。",
        "- 新500ms和1s组的输入/输出归一化均沿用原训练集统计；共用重新从1s训练窗口计算的损失尺度。旧200ms训练使用110,863窗口及原损失尺度，因此旧模型只是历史参照，不是严格单变量时域消融。", "",
        "## 损失与训练预算", "",
        "每步L = (||Δp||/s_p)^2 + (||Δv||/s_v)^2 + (SO(3)测地角/s_R)^2 + (||Δω||/s_ω)^2。先对窗口内有效步等权平均，再对batch窗口平均；位置为相对起点位移。四项权重均1。", "",
        f"共同尺度依次为{[round(v, 6) for v in norm['loss_scales']]}，单位为m、m/s、rad、rad/s；由共同1s训练集的恒定运动学基线残差RMS设定，未用验证集。精确值见normalization.json。", "",
        table(pd.DataFrame([{"训练时域ms": h, **dict(zip(['L_p', 'L_v', 'L_R', 'L_omega'], v))} for h, v in norm["train_baseline_mean_scaled_square"].items()]).round(6)), "",
        "上表是两时域训练基线的各项平均无量纲平方损失，便于审查实际相对权重。500ms与1s使用同一尺度，但不同跨度的分量大小仍会不同；早期步并未因损失尺度而获得单独加权。", "",
        "GRU64/事件编码GRU16，两种模型均分配37,590参数。训练日志按“梯度非None的参数张量”计数，NoControl为36,486、Control为37,590；此计数仍包括NoControl中连接恒零控制输入的权重。NoControl排除不执行的事件GRU（1,104参数）、历史控制输入列（768）和递推事件输入列（3,072）后，功能上参与预测的参数为32,646。两者GRU主体和宽度相同，但有效容量并非严格相等。相同seed、相同起点随机顺序、AdamW(lr=0.001, weight_decay=0.0001)、batch1024、梯度裁剪5、24epoch上限、patience5、最小改善1e-4及余弦学习率。按各自训练时域的验证航次等权多步损失选最佳checkpoint。", "",
        "每个epoch的新数据窗口数相同；1s组每次更新的递推与反传长度约为500ms组两倍，所以更新次数预算一致不等于计算量一致。Control与NoControl的事件编码开销亦不同。完整预算如下。", "",
        table(budget[["training_horizon_ms", "model", "seed", "epochs", "updates", "best_epoch", "seconds", "peak_allocated_bytes", "stop_reason"]].round(2)), "",
        "## 端点误差", "",
        "**均值 ± 航次间样本标准差**：先在每条飞行内平均窗口，再平均种子，最后27条飞行等权。不是逐轴RMSE。重叠窗口不作为独立飞行；窗口等权均值、标准差、P95/P99及种子间标准差见summary.csv。", "",
        table(pd.DataFrame(rows, columns=["验证时域s", "条件", *LABELS])), "",
        "## 配对比较", "",
        "先在每条飞行内平均三个种子，再按飞行进行配对bootstrap（5000次）的95%区间；同日期飞行仍可能相关。负差值表示候选模型误差更小。没有按窗口数量夸大独立样本量。", "",
    ]
    selected = pairs[((pairs.reference.str.replace("NoControl", "Control") == pairs.candidate) & (~pairs.reference.str.startswith("H200"))) |
                     ((pairs.reference == "H500-GRU-Control") & (pairs.candidate == "H1000-GRU-Control"))].copy()
    selected["区间"] = selected.apply(lambda r: f"[{r.bootstrap_low:.4f}, {r.bootstrap_high:.4f}]", axis=1)
    lines += [table(selected[["horizon_ms", "reference", "candidate", "metric", "improvement_pct", "better_flights", "区间"]].round(3)), "",
        "## 航次、工况和误差来源", "",
        "group_metrics.csv给出REVIEW、高动态、模式变化和控制变化幅度分组的航次等权结果。控制变化幅度使用历史+2s命令范围向量的L2范数，按共同验证窗口三分位进行描述性分组；阈值见group_definition.json，从不进入训练或checkpoint选择。高动态/REVIEW沿用冻结2s窗口元数据。", "",
        "观测积分参照沿用同一批起点：将真实未来v/ω送入相同梯形位置及SO(3)积分，再与EKF观测比较。它只用于事后诊断，不送入模型；不是严格预测误差下界，也不能用两个误差模长的差直接做贡献分解。模型相对该积分轨迹的差见dynamics_integral_500ms.csv及dynamics_integral_1000ms.csv。", "",
    ]
    groups = pd.read_csv(OUT / "group_metrics.csv")
    groups = groups[(groups.condition == "H1000-GRU-Control") & (groups.horizon_ms == 2000)
                    & (groups.grouping != "mode_change")]
    observations = pd.read_csv(OUT / "observation_group_metrics.csv")
    observations = observations[observations.horizon_ms == 2000].set_index(["grouping", "group", "metric"])
    rows = []
    for (grouping, value), f in groups.groupby(["grouping", "group"]):
        f = f.set_index("metric")
        o = observations.loc[(grouping, value, "position_m")]
        rows.append([grouping, value, int(f.n_flights.iloc[0]), int(f.n_windows.iloc[0]),
                     f'{f.loc["position_m", "flight_mean"]:.3f} ± {f.loc["position_m", "flight_std"]:.3f}',
                     f'{f.loc["attitude_deg", "flight_mean"]:.3f} ± {f.loc["attitude_deg", "flight_std"]:.3f}',
                     f'{o.flight_mean:.3f} ± {o.flight_std:.3f}'])
    lines += ["以下为1s训练Control在2s处的分组结果：均值 ± 组内航次间样本标准差。不同组的飞行构成和窗口不同，不是受控工况因果实验。", "",
              table(pd.DataFrame(rows, columns=["分组", "组", "航次", "窗口", "位置m", "姿态°", "观测积分位置残差m"])), ""]
    obs = pd.read_csv(LONG / "observation_per_flight.csv")
    obs = obs[obs.horizon_ms.isin([500, 1000, 2000])]
    rows = []
    for h, f in obs.groupby("horizon_ms"):
        rows.append([h, f"{f.position_m.mean():.4f} ± {f.position_m.std():.4f}",
                     f"{f.attitude_deg.mean():.4f} ± {f.attitude_deg.std():.4f}"])
    lines += [table(pd.DataFrame(rows, columns=["时域ms", "观测积分位置残差m", "观测积分姿态残差°"])), "",
        "未来控制是闭环日志中的已记录命令，可能包含控制器对未来观测的反馈信息。Control对NoControl检验的是历史与未来命令的总体条件预测信息，不能单独证明控制反事实响应、独立干预有效性或闭环稳定性。", "",
        "## 图表", "",
        "![训练曲线](figures/learning_curves.png)", "",
        "![NoControl误差增长](figures/error_growth_GRU-NoControl.png)", "",
        "![Control误差增长](figures/error_growth_GRU-Control.png)", "",
        "![航次分布](figures/flight_distributions.png)", "",
        "![500ms训练轨迹](figures/trajectories_train_500ms.png)", "",
        "![1s训练轨迹](figures/trajectories_train_1000ms.png)", "",
        "![控制增益种子分布](figures/control_gain_by_seed.png)", "",
        "轨迹例子固定为与旧评估相同的六条航次中间窗口、种子17，不按预测好坏挑选。逐步曲线覆盖约20ms至2s，按原生步的实际平均时间绘制；支持量保存在native_steps_per_flight.csv，末步少量99步窗口不会补真值。", "",
        "## 复现与归档", "",
        "使用/home/zn/anaconda3/envs/flap-train-gpu/bin/python。顺序为：train_velocity_rate_horizons.py prepare → normalize → train --horizon {500,1000}（自动先smoke，也可单独运行smoke阶段）；evaluate_velocity_rate_horizon_training.py evaluate --horizon {500,1000} → summarize → verify；运行本报告脚本，提交实现代码后执行evaluate_velocity_rate_horizon_training.py archive，再提交实验归档。所有实验根目录只创建一次，历史结果不覆盖。", "",
        "训练配置、源码SHA256、数据哈希、checkpoint SHA256、日志与峰值显存分别记录在run_manifest.json、data_audit.json、各训练目录training_provenance.json、summary.json及training.jsonl。逐窗口端点在metrics/；逐航次、种子、配对和工况统计均以CSV保存。大体积原生步逐窗口误差及训练缓存保留在忽略目录artifacts/velocity_rate_gru64_horizon_training_v1，归档清单记录哈希。", "",
        "测试包括长递推因果前缀、NoControl隔离、可变步数裁剪、全步损失与padding、1s梯度传播；GPU smoke验证有限梯度/四元数/输入隔离。新checkpoint在128个分布于全验证集的窗口上用SciPy独立复算全部原生步指标；逐窗口聚合另行复算，并核验历史实验归档不变。详见verification.json、evaluation_*.json和testing.json。", "",
    ]
    (OUT / "report_zh.md").write_text("\n".join(lines))


if __name__ == "__main__":
    figures()
    report()
