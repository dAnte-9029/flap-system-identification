# Main V2 stateful simulator：0.2–5 s validation 结果

2026-09-14。HEAD `1482b874757f157cba67c9987146168729535644`；冻结 Main V2 权重未改变。
本轮完成 stateful 包装、续跑一致性验证与正式 validation benchmark；未训练、未改网络/损失、未打开 sealed test。

## 七个明确回答

**Q1：能否包装成真正 stateful simulator？YES。** 最小充分状态已经显式化，固定权重/配置下，state + command + dt 唯一决定下一状态。warm reset、显式恢复、CPU tensor snapshot、step 均已实现。五条真实验证日志起点的连续 250 步与 100+150 步保存/恢复结果，所有物理状态、GRU、drive、tail、phase anchor 逐位一致；与原 forward 的六类物理输出最大差为 0。未来标签全替换 NaN 后结果逐位不变。

**Q2：5 s 是否数值稳定？YES，限本次 722 条 warm26 验证轨迹。** 非有限值和四元数范数 gate 失败均为 0；frequency、derivative、drive、residual 四类裁剪的事件数均为 0。不能外推到任意新控制策略或更长时域。

**Q3：5 s 动力学是否可信？NO，作为可供 RL 使用的动力学模型尚不可信。** 5 s 姿态 RMSE 45.53°、速度 RMSE 5.05 m/s，姿态 p95 88.65°；所有起点末段角速度变化均触发过度衰减诊断。722 条中 255 条（35.32%）超出训练 min/max 支持域，benchmark 明确 FAIL（退出码 2），并不是 NaN 爆炸。

**Q4：哪个 horizon 开始明显恶化？** 0.5→1→2 s 位置 RMSE 0.204→0.571→1.936 m，速度 0.691→1.156→2.096 m/s，姿态 6.43→10.51→18.01°；1–2 s 已出现明显累计恶化，2–5 s 扩大到位置 11.81 m、速度 5.05 m/s、姿态 45.53°。速度和姿态基本持续增长，不能声称存在一个已证实的突然发散时刻。没有给定绝对精度验收阈值，因此不能把 1 s 或 2 s 宣布为安全时域。

**Q5：最先坏掉的状态？** 若“坏”指匹配实飞动态的衰减，短至 0.2 s 已能看到速度变化不足（prefix std 比中位数 0.346），到 0.5 s 角速度 prefix std 比降至 0.450，1 s 为 0.348，末 1 s 仅 0.074。若按硬支持门槛，最早的失败状态/时间见下表，主要是角速度和加速度低于训练最小值，完全不是速度/角速度向上爆炸。频率 RMSE 在 0.5–5 s 约 0.19–0.20 Hz，不是首要增长源。相位 circular RMSE 从 0.226 增到 1.626 rad，说明周期同步也逐渐丢失。以上不构成某个状态“导致”另一状态漂移的因果证明。

**Q6：最差工况？** 使用连续变量分桶而不擅自命名机动：高 motor-command 组（86 起点）的 5 s 速度/姿态 RMSE 为 6.59 m/s / 59.69°；高 tail-command-norm 组（220 起点）为 6.01 / 55.51°；高控制变化组（241 起点）为 6.00 / 54.88°。较大 |roll| 组也较差（5.50 / 49.90°）。最差整条日志为 log_22（140 起点），6.54 m/s / 58.39°。这些是相关分层，存在 flight/控制/姿态共线，不能解释为单个控制通道的因果效应。

**Q7：下一步优先级：A > D > E > C > B。**

1. **A，保持架构做训练目标的受控对照**：当前权重只以 1 s 多步目标训练，1–5 s 的刚体漂移与角速度幅度压低突出。后续优先比较多 horizon 目标及角运动动态保真；本轮不训练，不能保证改 loss 后改善。
2. **D，检查/改进 aerodynamic 或 state-transition 建模**：频率已稳定但刚体误差持续积累，需区分导数偏差、周期动态损失与姿态反馈旋转误差；不直接跳到 Main V3。
3. **E，针对高 motor/tail、控制变化与跨日志分布补证据/数据**：这些组误差较大，且本次 replay 未验证新策略反事实动作响应；本轮不给未经验证的激励幅值。
4. **C，执行器模型不是首个改动**：频率误差没有随 horizon 明显增加，改 actuator 是否能修复刚体误差没有证据。
5. **B，不优先加长历史或新增 latent**：13 点与26点已很接近，冷启动差距随 horizon 缩小，而5 s大漂移仍在。

## 状态/API 与一致性

代码：`src/system_identification/models/main_v2_simulator.py`。
状态：p_n、v_n、q_nb、ω_b、frequency、relative phase、64维 GRU hidden、标量 drive proxy、三维 tail proxy、phase anchor。不重复存 body velocity、重力投影、相位 sin/cos 或导数。tail proxy 顺序为 symmetric/differential/rudder，不能称作实测舵角。

```python
sim = MainV2Simulator(frozen_model.eval())
state = sim.reset(**warm_inputs)       # history through t0 only
state, diagnostics = sim.step(state, command, dt)  # batched command [B,4], dt [B]
torch.save(state.snapshot(), "state.pt")
restored = SimulatorState.from_snapshot(
    torch.load("state.pt", weights_only=True), device=device)
state = sim.reset(state=restored)      # no history encoding or re-anchoring
```

Snapshot 需配合同一模型权重、归一化及固定时间常数。simulator 外部负责时钟；模型输出也包含观察状态，不需要单独复制 observation。旧 forward、训练缓存语义、checkpoint schema 均未修改。

## 样本与时间契约

722 个起点，五条 validation flight，完整26点历史，同一 segment 内250个未来控制/251个状态；每50个原生样本滚动一次。所有初始化方式用完全相同起点。不随机切分，不跨段，不插值。训练数据只用于观测范围/分桶，没有更新归一化或权重。

实际 dt min/max = 0.009855/0.029945 s，p1/p99 = 0.009858/0.029940，median = 0.019959，mean = 0.020000129。可见局部约10/30 ms，并非每步严格20 ms。
250步实际时长 4.989626–5.010254 s；历史实际约 0.488983–0.509019 s。原 warm history filter 固定0.02 s保持不变，以维持旧评估语义。

已有八月2秒报告是3920个起点；本轮是满足完整历史与5秒未来的新722个共同起点。因此不能直接把两张表的微小差异解释为模型改善。本轮所有模型比较都使用这722个相同起点。重叠起点不是722次独立飞行。

## 误差曲线和尾部

以下为先每日志 RMSE、再等权平均；失败但有限的轨迹仍包含在统计中。

| horizon s | position m | velocity m/s | attitude deg | body rate rad/s | frequency Hz | phase rad |
| --- | --- | --- | --- | --- | --- | --- |
| 0.2000 | 0.0875 | 0.5038 | 3.2274 | 0.7074 | 0.1925 | 0.2262 |
| 0.5000 | 0.2038 | 0.6912 | 6.4270 | 0.7423 | 0.1870 | 0.3366 |
| 1.0000 | 0.5707 | 1.1559 | 10.5139 | 0.7495 | 0.1969 | 0.5899 |
| 2.0000 | 1.9358 | 2.0956 | 18.0142 | 0.7633 | 0.2024 | 0.9143 |
| 3.0000 | 4.2692 | 3.2490 | 27.1578 | 0.7742 | 0.2039 | 1.1810 |
| 5.0000 | 11.8073 | 5.0515 | 45.5318 | 0.7879 | 0.2016 | 1.6259 |

5 s 逐起点分布（pooled，不是日志等权）：

| metric | mean | median | p90 | p95 | max |
| --- | --- | --- | --- | --- | --- |
| position_m | 10.0499 | 8.4699 | 19.0026 | 22.0317 | 45.5621 |
| velocity_m_s | 4.2467 | 3.6151 | 8.2742 | 10.2036 | 17.4135 |
| attitude_deg | 37.4183 | 29.8070 | 74.9330 | 88.6549 | 167.6658 |
| body_rate_rad_s | 0.7181 | 0.6962 | 1.1485 | 1.3306 | 2.0961 |
| frequency_hz | 0.1656 | 0.1436 | 0.3349 | 0.3797 | 0.5410 |
| phase_rad | 1.3915 | 1.3338 | 2.6296 | 2.9284 | 3.1145 |

各 horizon 的 mean/median/p90/p95/max、horizontal/vertical position、vx/vy/vz、p/q/r 与各自 RMSE 均在 `per_horizon.csv`、`per_flight.csv`、`per_rollout.csv`。姿态为 geodesic，phase 为 atan2(sin Δφ, cos Δφ) 的绝对 circular error。

原 constant-twist 参考在5 s位置/速度/姿态/角速度/频率等权RMSE为18.73 m / 6.58 m/s /126.79° /1.152 rad/s /0.434 Hz。Main V2在全部horizon均优于这个基线，但这不等于满足绝对保真需求。

## Gate 与“有限但假”

| horizon | valid | failed | failure_pct | numeric_failure_pct | truth_support_failure_pct |
| --- | --- | --- | --- | --- | --- |
| 0.2 | 720 | 2 | 0.2770 | 0.0000 | 0.0000 |
| 0.5 | 705 | 17 | 2.3546 | 0.0000 | 0.0000 |
| 1.0 | 682 | 40 | 5.5402 | 0.0000 | 0.0000 |
| 2.0 | 630 | 92 | 12.7424 | 0.0000 | 0.0000 |
| 3.0 | 569 | 153 | 21.1911 | 0.0000 | 0.0000 |
| 5.0 | 467 | 255 | 35.3186 | 0.0000 | 0.1385 |

支持域门槛取训练 min/max，不是适航安全范围；p1/p99越界单独记录。真实 validation 自身5 s越界比例为0.139%（1条），模型35.32%。本轮超域均来自低角速度/低加速度：

| state | below_train_min_rollouts | above_train_max_rollouts | earliest_low_time_s | median_first_low_time_s | predicted_min | predicted_max | train_min | train_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| body_rate | 135 | 0 | 0.2994 | 3.0039 | 0.0014 | 1.6709 | 0.0177 | 4.9103 |
| acceleration | 181 | 0 | 0.1774 | 2.4050 | 0.0472 | 7.8201 | 0.1980 | 62.7389 |

低于原生差分加速度的最小值本身并不证明违反物理定律，噪声会抬高差分统计；所以这里使用“支持域失败”，不称“飞机不可能”。没有任何超过训练最高速度、最高角速度、最高频率/加速度/角加速度的模型路径。裁剪事件全部为0，因此不能归因于 clamp 勉强维持有限。

训练与validation实际观测范围：

| partition | signal | min | max | p1 | p99 |
| --- | --- | --- | --- | --- | --- |
| train | speed | 0.1900 | 15.3511 | 4.6096 | 11.3533 |
| train | body_rate | 0.0177 | 4.9103 | 0.1734 | 2.4289 |
| train | frequency | 0.8369 | 5.3702 | 2.2118 | 5.1064 |
| train | acceleration | 0.1980 | 62.7389 | 1.4873 | 22.8077 |
| train | angular_acceleration | 0.0000 | 283.6162 | 2.7039 | 108.8369 |
| validation | speed | 0.4895 | 12.1669 | 4.7857 | 9.4439 |
| validation | body_rate | 0.0183 | 4.2011 | 0.1490 | 1.6731 |
| validation | frequency | 0.5694 | 5.0309 | 2.3626 | 4.5378 |
| validation | acceleration | 0.2181 | 72.1578 | 1.4981 | 21.1179 |
| validation | angular_acceleration | 0.0000 | 235.6867 | 2.6116 | 70.0701 |

末1秒 predicted/truth variation ratio 中位数：velocity 0.370、body rate 0.074、frequency 0.327。角速度衰减 suspect 722/722；unsupported equilibrium suspect 2/722；低控制变化下的 energy-growth suspect 11/722。

这些为事先声明的描述性筛查：不把 speed²/rate² 代理当作真实机械能，也不把实测信号的所有高频变化都当作真实气动振荡（可能包含估计/采样噪声）。两个 equilibrium suspect 仅指末段近稳态且落在训练支持域外，没有证明异常吸引子；11个energy-growth suspect没有伴随上包线或裁剪失败，不能单凭低command variation断言无能量输入。连续比值、漂移、能量代理增长与具体window ID见 `trajectory_diagnostics.csv`。

角速度 prefix variation 中位数逐渐减少，表明存在实测动态被抹平的系统性迹象，而非仅少数异常窗口：

| horizon_s | state | ratio_median | ratio_p10 | ratio_p90 | below_half_count | count |
| --- | --- | --- | --- | --- | --- | --- |
| 0.2000 | body_rate | 0.5846 | 0.3360 | 0.8791 | 241 | 722 |
| 0.5000 | body_rate | 0.4500 | 0.2655 | 0.7001 | 429 | 722 |
| 1.0000 | body_rate | 0.3480 | 0.2166 | 0.5687 | 595 | 722 |
| 2.0000 | body_rate | 0.2874 | 0.1974 | 0.4500 | 690 | 722 |
| 3.0000 | body_rate | 0.2726 | 0.1888 | 0.4033 | 709 | 722 |
| 5.0000 | body_rate | 0.2458 | 0.1718 | 0.3606 | 720 | 722 |

## 历史依赖与冷启动

| initialization | horizon_s | position_m_equal_log_rmse | velocity_m_s_equal_log_rmse | attitude_deg_equal_log_rmse | body_rate_rad_s_equal_log_rmse |
| --- | --- | --- | --- | --- | --- |
| cold_repeated26 | 0.2000 | 0.0988 | 0.6145 | 3.7358 | 0.8201 |
| cold_repeated26 | 0.5000 | 0.2439 | 0.7351 | 6.8037 | 0.7762 |
| cold_repeated26 | 1.0000 | 0.6346 | 1.2179 | 10.9110 | 0.7584 |
| cold_repeated26 | 2.0000 | 2.0850 | 2.2087 | 18.8994 | 0.7604 |
| cold_single | 0.2000 | 0.0959 | 0.5659 | 3.6330 | 0.7590 |
| cold_single | 0.5000 | 0.2287 | 0.6946 | 6.4400 | 0.7567 |
| cold_single | 1.0000 | 0.5906 | 1.1472 | 10.4835 | 0.7487 |
| cold_single | 2.0000 | 1.9536 | 2.1093 | 18.1715 | 0.7600 |
| cold_zero | 0.2000 | 0.0979 | 0.5723 | 3.7718 | 0.7486 |
| cold_zero | 0.5000 | 0.2331 | 0.6973 | 6.4720 | 0.7577 |
| cold_zero | 1.0000 | 0.5952 | 1.1473 | 10.5030 | 0.7486 |
| cold_zero | 2.0000 | 1.9532 | 2.1038 | 18.1600 | 0.7602 |
| warm13 | 0.2000 | 0.0874 | 0.5048 | 3.2140 | 0.7186 |
| warm13 | 0.5000 | 0.2051 | 0.6937 | 6.4318 | 0.7437 |
| warm13 | 1.0000 | 0.5759 | 1.1663 | 10.6019 | 0.7508 |
| warm13 | 2.0000 | 1.9609 | 2.1167 | 18.2170 | 0.7637 |
| warm26 | 0.2000 | 0.0875 | 0.5038 | 3.2274 | 0.7074 |
| warm26 | 0.5000 | 0.2038 | 0.6912 | 6.4270 | 0.7423 |
| warm26 | 1.0000 | 0.5707 | 1.1559 | 10.5139 | 0.7495 |
| warm26 | 2.0000 | 1.9358 | 2.0956 | 18.0142 | 0.7633 |
| warm5 | 0.2000 | 0.0938 | 0.5396 | 3.4323 | 0.7349 |
| warm5 | 0.5000 | 0.2218 | 0.6814 | 6.4335 | 0.7471 |
| warm5 | 1.0000 | 0.5841 | 1.1495 | 10.5212 | 0.7523 |
| warm5 | 2.0000 | 1.9551 | 2.1181 | 18.1852 | 0.7614 |

13点覆盖约0.24 s；在0.2/0.5/1/2 s全部核心等权RMSE与26点的差异不超过1.6%。5点（约0.08 s）在0.2 s位置/速度/姿态退化约7.2%/7.1%/6.4%；single-point cold分别退化约9.6%/12.3%/12.6%。cold-zero短期更差，重复26次t0不能代替真实历史（0.2 s速度退化约22%）。

因此，在本checkpoint/五条验证日志上，**约0.24 s真实历史已接近0.5 s历史，历史主要改善最初0.2–0.5 s**；不能称0.5 s是必需。2 s后single/zero与warm的主要指标多在约1%内，但这是共享的长期误差主导，绝不是冷启动已解决动力学。标准benchmark仍采用warm26；未来RL reset可考虑burn-in，但本轮尚不启动RL。没有评估>26点历史，因此不宣称全局最优历史长度。

## Flight / 连续状态分桶

| log_id | n_rollouts | failure_pct | position_m_equal_log_rmse | velocity_m_s_equal_log_rmse | attitude_deg_equal_log_rmse | body_rate_rad_s_equal_log_rmse |
| --- | --- | --- | --- | --- | --- | --- |
| 2026.8.10-8.20/log_15_2026-8-20-06-08-16.ulg | 145 | 33.7931 | 11.4850 | 4.6110 | 42.1676 | 0.7632 |
| 2026.8.10-8.20/log_16_2026-8-20-06-20-52.ulg | 135 | 42.2222 | 10.0612 | 4.4397 | 40.2986 | 0.7016 |
| 2026.8.10-8.20/log_17_2026-8-20-06-29-34.ulg | 170 | 38.2353 | 10.7904 | 4.8402 | 44.2534 | 0.7638 |
| 2026.8.10-8.20/log_19_2026-8-20-06-51-18.ulg | 132 | 16.6667 | 11.4432 | 4.8277 | 42.5495 | 0.7793 |
| 2026.8.10-8.20/log_22_2026-8-20-07-09-54.ulg | 140 | 44.2857 | 15.2565 | 6.5389 | 58.3897 | 0.9316 |

训练三分位边界（bin0低、bin1中、bin2高）：

```json
{
  "vertical_speed": [
    -0.4917822579542796,
    0.30235158403714496
  ],
  "abs_body_yaw_rate": [
    0.14150231579939523,
    0.31481460730234784
  ],
  "abs_roll_deg": [
    6.242118525196147,
    13.641610403555285
  ],
  "pitch_deg": [
    12.581461855761564,
    17.579768048931676
  ],
  "motor_command": [
    0.6498799920082092,
    0.7378357450167338
  ],
  "tail_command_magnitude": [
    0.31867464710654,
    0.5263807801666808
  ]
}
```

NED vertical speed为正向下。|body yaw rate|不是严格航迹转弯率，因此不把它直接命名为turning。5 s分桶结果如下；0.5/1/2 s对照保存在 `regime_bins.csv`。

| variable | bin | n_rollouts | velocity_m_s_equal_log_rmse | attitude_deg_equal_log_rmse |
| --- | --- | --- | --- | --- |
| vertical_speed | 0 | 234 | 5.2077 | 47.6687 |
| vertical_speed | 1 | 265 | 5.0847 | 46.2569 |
| vertical_speed | 2 | 223 | 4.7912 | 41.3408 |
| abs_body_yaw_rate | 0 | 283 | 5.0237 | 44.8942 |
| abs_body_yaw_rate | 1 | 239 | 5.0632 | 46.1636 |
| abs_body_yaw_rate | 2 | 200 | 5.0979 | 45.7792 |
| abs_roll_deg | 0 | 280 | 4.7143 | 42.3349 |
| abs_roll_deg | 1 | 267 | 5.1039 | 46.0753 |
| abs_roll_deg | 2 | 175 | 5.4989 | 49.8954 |
| pitch_deg | 0 | 117 | 4.9929 | 43.4348 |
| pitch_deg | 1 | 298 | 4.8155 | 42.8682 |
| pitch_deg | 2 | 307 | 5.3398 | 49.1398 |
| motor_command | 0 | 368 | 4.6383 | 40.4068 |
| motor_command | 1 | 268 | 4.9550 | 45.6982 |
| motor_command | 2 | 86 | 6.5897 | 59.6930 |
| tail_command_magnitude | 0 | 188 | 4.4140 | 38.9748 |
| tail_command_magnitude | 1 | 314 | 4.8126 | 42.4792 |
| tail_command_magnitude | 2 | 220 | 6.0147 | 55.5141 |

控制变化组按future command tape总变差离线划分，阈值记录在summary；这些未来量仅用于离线分析，绝不进入初始化/模型特征。相关分桶不能隔离因果效应或保证新控制策略安全。

## Readiness

- **Simulator structural readiness: 100%**，仅按本轮十项stateful API清单计分，不代表生产级RL系统完整度。
- **Dynamics fidelity readiness: 60%**，按实验前声明的十项证据清单计分：数值通过、无裁剪、四类状态全horizon优于constant-twist共六项通过；全路径支持域、无有限但假suspect、外部绝对精度验收、反事实动作验证四项未通过/缺失。
- **RL-ready: NO。** 分数不是安全概率；绝对精度目标与反事实验证不能由相对基线收益代替。

## 复现、验证与产物

协议：`docs/contracts/2026-09-14_main_v2_stateful_benchmark.md`。结果目录：`/home/zn/flap-system-identification/docs/analysis/results/main_v2_free_running_5s`。
完整warm轨迹与pause snapshot在summary的trajectory_files中，带SHA256；旧checkpoint/历史结果保持不变。
复现到新目录（benchmark预期以2退出表示门槛FAIL，结果已落盘；其他非零码应排查）：

```bash
/home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/run_main_v2_free_running.py \
  --output-root /tmp/main-v2-reproduce/results --trajectory-root /tmp/main-v2-reproduce/traces
/home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest -q \
  tests/test_main_v2_simulator.py tests/test_main_v2_free_running.py \
  tests/test_trajectory_main_v1.py tests/test_trajectory_main_v2.py
```

默认完整基准会自动生成结果目录内的report.md；独立report脚本仅供已有且尚无review产物的完整结果使用。
本机CUDA不可用，实际在指定环境CPU执行；没有安装依赖或换解释器。相关trajectory/september回归56项通过；git diff --check另行记录于交付检查。

图：五项 `error_vs_horizon_*.png`；每flight的 `review/heatmap_*.png`显式保留segment间白色空档（优先看review版本）；`review/worst_velocity_trace.png`显示最差速度起点的预测与实测时序，选择规则固定为5 s最大velocity error。
