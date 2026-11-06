# PACLD v1 冻结模型综合评价

代码基线 `669c50042f9f50fdbea0ad6e8b875bd40c116b86`。Seed 17，mixed precision，Global Batch 32，342 updates / 10,944 训练窗口。本次未训练、未改权重/损失/数据、未访问 sealed test、未启动闭环仿真。

## 冻结身份与复现

工作树 `/home/zn/flap-system-identification/.worktrees/pacld-gpu-batched-v1`。模型 `runs/pacld_frozen_eval_seed17/model.pt`；SHA256 `9d82640692c0baec476e276696be4f89fd89a58854361bbcfdf3dbb84cb466c9`（此前复制及复核，本次只读）。完整来源、配置、数据身份和固定 validation 成员见同目录 `manifest.json`。本次从 checkpoint 的 validation_window_ids 取出固定 156 窗口，未重新抽样。

运行 `/home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/evaluate_pacld_frozen.py`，再运行 `scripts/report_pacld_frozen.py`。完整 JSON、CSV、PNG 位于 `/home/zn/flap-system-identification/.worktrees/pacld-gpu-batched-v1/runs/pacld_frozen_eval_seed17/results`，不提交模型及大型结果。

## 11 节点验证与训练充分性

| step | full | macro | full_delta | macro_delta |
| --- | --- | --- | --- | --- |
| 32 | 54.30733 | 7.74028 | nan | nan |
| 64 | 51.05680 | 7.74078 | -3.25053 | 0.00050 |
| 96 | 47.93009 | 7.21105 | -3.12670 | -0.52973 |
| 128 | 47.63276 | 7.20869 | -0.29733 | -0.00236 |
| 160 | 46.80371 | 7.11366 | -0.82905 | -0.09503 |
| 192 | 48.01960 | 7.53585 | 1.21589 | 0.42219 |
| 224 | 44.13966 | 6.85603 | -3.87994 | -0.67982 |
| 256 | 42.87934 | 6.88262 | -1.26032 | 0.02659 |
| 288 | 41.70407 | 6.84207 | -1.17527 | -0.04055 |
| 320 | 41.17672 | 6.88410 | -0.52735 | 0.04203 |
| 342 | 39.84822 | 6.77555 | -1.32850 | -0.10854 |

FULL 最佳点为 342 步 39.848219，Macro 为 6.775553。192 步出现反弹，随后改善。最后 22 步 FULL 仍下降 1.328499，现有验证曲线不支持已达到平台期；继续训练可能改善验证目标，但不保证物理/控制收益。本次不继续训练。没有同口径训练曲线，不能认定过拟合。图：`results/validation_curve.png`。

## 四时域物理预测

NED 位置/速度，FRD 角速度，Hamilton wxyz 姿态。姿态采用 SO(3) 测地误差（表中度）；其余依次 m、m/s、rad/s。向量指标是三维误差平方模长先平均再开根号。每个请求时刻取 ±30 ms 内最近原生有效观测，不插值；不同指标覆盖独立报告。窗口先在航次/窗口时域内平均，再对时域组平均，最后航次等权。native_observations 是所选原生目标计数，跨窗口可能重复，不能视为独立样本数。

| model | component | horizon_s | rmse | flights | windows | native_observations |
| --- | --- | --- | --- | --- | --- | --- |
| BASE | R | 0.20000 | 3.96291 | 39 | 152 | 152 |
| BASE | R | 0.50000 | 7.07356 | 39 | 110 | 110 |
| BASE | R | 1.00000 | 12.70444 | 39 | 69 | 69 |
| BASE | R | 2.00000 | 21.89404 | 31 | 31 | 31 |
| BASE | omega | 0.20000 | 0.77232 | 39 | 156 | 156 |
| BASE | omega | 0.50000 | 0.71538 | 39 | 117 | 117 |
| BASE | omega | 1.00000 | 0.69808 | 39 | 78 | 78 |
| BASE | omega | 2.00000 | 0.82868 | 39 | 39 | 39 |
| BASE | p | 0.20000 | 0.13178 | 39 | 156 | 156 |
| BASE | p | 0.50000 | 0.29221 | 39 | 117 | 117 |
| BASE | p | 1.00000 | 0.75662 | 39 | 78 | 78 |
| BASE | p | 2.00000 | 2.96942 | 39 | 39 | 39 |
| BASE | v | 0.20000 | 0.57108 | 39 | 156 | 156 |
| BASE | v | 0.50000 | 0.81848 | 39 | 117 | 117 |
| BASE | v | 1.00000 | 1.35856 | 39 | 78 | 78 |
| BASE | v | 2.00000 | 3.45190 | 39 | 39 | 39 |
| FULL | R | 0.20000 | 3.85437 | 39 | 152 | 152 |
| FULL | R | 0.50000 | 6.76276 | 39 | 110 | 110 |
| FULL | R | 1.00000 | 12.20834 | 39 | 69 | 69 |
| FULL | R | 2.00000 | 21.53842 | 31 | 31 | 31 |
| FULL | omega | 0.20000 | 0.71442 | 39 | 156 | 156 |
| FULL | omega | 0.50000 | 0.68115 | 39 | 117 | 117 |
| FULL | omega | 1.00000 | 0.68182 | 39 | 78 | 78 |
| FULL | omega | 2.00000 | 0.82080 | 39 | 39 | 39 |
| FULL | p | 0.20000 | 0.13094 | 39 | 156 | 156 |
| FULL | p | 0.50000 | 0.29186 | 39 | 117 | 117 |
| FULL | p | 1.00000 | 0.73951 | 39 | 78 | 78 |
| FULL | p | 2.00000 | 2.91799 | 39 | 39 | 39 |
| FULL | v | 0.20000 | 0.55564 | 39 | 156 | 156 |
| FULL | v | 0.50000 | 0.80861 | 39 | 117 | 117 |
| FULL | v | 1.00000 | 1.30369 | 39 | 78 | 78 |
| FULL | v | 2.00000 | 3.42667 | 39 | 39 | 39 |
| kinematic | R | 0.20000 | 4.05049 | 39 | 152 | 152 |
| kinematic | R | 0.50000 | 7.68894 | 39 | 110 | 110 |
| kinematic | R | 1.00000 | 15.15247 | 39 | 69 | 69 |
| kinematic | R | 2.00000 | 28.46858 | 31 | 31 | 31 |
| kinematic | omega | 0.20000 | 0.77593 | 39 | 156 | 156 |
| kinematic | omega | 0.50000 | 0.72708 | 39 | 117 | 117 |
| kinematic | omega | 1.00000 | 0.76840 | 39 | 78 | 78 |
| kinematic | omega | 2.00000 | 0.83780 | 39 | 39 | 39 |
| kinematic | p | 0.20000 | 0.13669 | 39 | 156 | 156 |
| kinematic | p | 0.50000 | 0.36319 | 39 | 117 | 117 |
| kinematic | p | 1.00000 | 1.12351 | 39 | 78 | 78 |
| kinematic | p | 2.00000 | 4.01236 | 39 | 39 | 39 |
| kinematic | v | 0.20000 | 0.64488 | 39 | 156 | 156 |
| kinematic | v | 0.50000 | 1.16002 | 39 | 117 | 117 |
| kinematic | v | 1.00000 | 2.12648 | 39 | 78 | 78 |
| kinematic | v | 2.00000 | 4.11446 | 39 | 39 | 39 |

FULL、BASE、运动学基线使用相同因果 Belief、相同窗口及目标。运动学基线保持初始惯性速度和机体角速度，以 SO(3) 积分姿态。BASE 仅禁用显式相位载波，仍保留包络、隐状态、控制和全部共享参数。图：`results/physical_rmse.png`。

GRU 未纳入公平比较：当前评价接口没有经核验的同一 156 窗口、四时域、相同因果输入/物理状态输出合同的冻结 GRU adapter。既有 FrozenRollResponse 的 GRU 对齐目标和来源合同不同，不能抄旧表拼接排名。因此“竞争力”结论仅相对本次同口径运动学基线，不涵盖 GRU。

## 拍内 FULL/BASE

| sampling_group | variable | regime | full_rmse | base_rmse | base_minus_full_mse | flights | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| main_50hz | q | control_change | 0.43984 | 0.49321 | 0.04980 | 27 | native sample prediction error; no interpolated high-frequency truth |
| main_50hz | q | stable | 0.41604 | 0.47324 | 0.05086 | 27 | native sample prediction error; no interpolated high-frequency truth |
| main_50hz | vD | control_change | 0.58363 | 0.59507 | 0.01348 | 27 | native sample prediction error; no interpolated high-frequency truth |
| main_50hz | vD | stable | 0.64805 | 0.65934 | 0.01476 | 27 | native sample prediction error; no interpolated high-frequency truth |
| sparse_10hz | q | control_change | 0.41456 | 0.45353 | 0.03383 | 12 | native sample prediction error; no interpolated high-frequency truth |
| sparse_10hz | q | stable | 0.42845 | 0.47128 | 0.03854 | 12 | native sample prediction error; no interpolated high-frequency truth |
| sparse_10hz | vD | stable | 0.56920 | 0.57864 | 0.01083 | 12 | native sample prediction error; no interpolated high-frequency truth |

按原生 q、vD 观测计算；control_change/stable 使用既有 diagnostics 阈值与回看区间。不是重新筛选控制事件。相位仅用于离线残差分组，从未进入 Encoder 或未来传播。图 `results/phase_residuals.png` 为 16 个机械相位箱内先航次/时域聚合、再航次等权的有符号残差；详细相位箱、航次、窗口见 `phase_residuals.csv`。相位相关只说明描述性结构，不代表高频真值重建。

## 跨拍净运动

| mode | component | rmse | pairs | windows | flights |
| --- | --- | --- | --- | --- | --- |
| BASE | R | 4.75429 | 3785 | 117 | 39 |
| BASE | omega | 0.50009 | 7857 | 117 | 39 |
| BASE | p | 0.35517 | 3467 | 117 | 39 |
| BASE | v | 0.52056 | 3467 | 117 | 39 |
| FULL | R | 4.59133 | 3785 | 117 | 39 |
| FULL | omega | 0.49995 | 7857 | 117 | 39 |
| FULL | p | 0.35318 | 3467 | 117 | 39 |
| FULL | v | 0.51545 | 3467 | 117 | 39 |

严格复用 valid_pairs 的真实跨拍配对；位置/速度/角速度比较物理增量，姿态比较相对旋转增量。RMSE 航次等权，配对高度重叠；pairs 是计数而非独立样本量，独立信息以 flights 为主。没有伪造 Macro 配对。

## 隐状态功能诊断

以下是跨窗口中位数，不是隐变量真值标签：

```json
{
  "slow_std_median": [
    0.5908520880783136,
    1.088664164040631,
    0.6846391497256439,
    0.36157750221458795,
    0.09605534602768931,
    0.462442911737752,
    0.18421402629487743,
    0.46665735803821,
    0.2142938692163735,
    0.3005721890230671,
    0.7259778179365202,
    0.41369176361279747
  ],
  "envelope_rms_median": [
    2.7898619468893733,
    2.806129499749719,
    2.7758413773323594
  ],
  "envelope_phase_slope_median": [
    -0.0009827696127249598,
    -7.047700663338564e-06,
    0.0002463274520286163
  ],
  "explicit_decoder_rms_median": [
    0.23464403640838868,
    0.12391397746813387,
    0.390377828481982,
    0.6974154583188068,
    2.652792136410395,
    0.35075749277058765
  ]
}
```

slow_std 为每窗口 12 维 Slow 沿未来轨迹的标准差；三组 envelope_rms 为二维包络模长 RMS；envelope_phase_slope 是包络角对自主传播相位的斜率，完全反向抵消参考值依次 -1/-2/-3。显式解码贡献在同一个 FULL 状态上计算 decode(FULL)-decode(BASE)，前三维为解码体轴加速度、后三维为角加速度。与 BASE 自主轨迹的差异不是同一概念。包络 RMS 中位数约 2.78–2.81，接近分量限幅 2 所对应的模长上限 2√2，提示包络接近饱和值得关注，但不能据此诊断物理塌缩。包络角斜率中位数接近 0，未显示整体 -1/-2/-3 反向抵消。完整逐窗口摘要见 evaluation.json；平滑、非零振幅或相位斜率均不足以证明物理快慢解耦或排除局部塌缩。

## 冻结控制响应

```json
{
  "validation": {
    "pairs": 1,
    "flights": 2,
    "components": 1,
    "missing_windows": 1
  },
  "train": {
    "pairs": 48,
    "flights": 46,
    "components": 11,
    "missing_windows": 26
  }
}
```

保留原 FrozenRollResponse 主配对、航次、依赖分组；不重筛选事件。只有所有窗口身份均存在才运行；缺失明确记为 missing_windows。本次 validation 唯一配对缺少 1 个窗口，train 48 对缺少 26 个窗口，因此两者均未计算响应指标；不是模型响应失败，而是当前 PACLD admission index 与历史配对窗口不完全兼容。缺失身份见 response_missing_windows.json。零配对或缺失窗口时不输出伪造指标。查询时刻保持原协议的 decision-relative shift，严格检查 actual_horizon_s，超出支持直接拒绝。历史仅包含当时已发布观测，未来使用真实 ZOH 控制。

该协议自身使用既有原生轨迹插值对齐，这与上面的原生端点误差不同；不是额外构造原生真值，也不是反事实控制真值。Train 描述性结果与 validation 分开，不能作为独立泛化证据。仅一个 validation 配对即使可计算，也不足以支持可靠泛化结论。

## 正确性与证据边界

定向测试 `tests/test_pacld_frozen_eval.py`：CPU/GPU 两项通过。检查无 Macro 配对的强制 BASE、异步任意合法查询时刻、共享因果 Belief、超时域拒绝；批量 FULL/BASE 对单窗口 FP64 参考使用 rtol=1e-8、atol=2e-10。本次实际评价沿用训练 mixed precision；该测试证明 FP64 执行定义一致，不把 mixed 与 FP64 宣称为逐位相同。

## 科研判断

- **VERIFIED：预测精度。** FULL 在 16/16 个物理通道/时域组合优于运动学基线；在 16/16 个组合优于 BASE。具体优势与不足以四时域表为准，不能据此声称优于 GRU。
- **SUPPORTED / UNRESOLVED：结构分工。** 所有 7 个可计算的 q/vD 分组均有 FULL 收益：q RMSE 降低约 8.6%–12.1%，vD 约 1.6%–1.9%。Fast 提供可观测额外预测价值，但垂向收益较小。跨拍 BASE 相比 FULL 的位置/速度/姿态/角速度 RMSE 分别约高 0.56%/0.99%/3.55%/0.03%，说明在这些冻结配对上大部分跨拍运动保留。是否保留跨拍运动应看相对误差，而非隐状态外观。不存在真实慢变标签，BASE 不是真实慢动力学，非零 Fast 贡献也不等于额外预测价值。
- **UNRESOLVED：控制有效性。** 冻结配对覆盖及支持检查决定可计算范围；有限 validation 配对不足以确认可靠控制响应泛化。观察性配对不能提供反事实真值。
- **SUPPORTED：训练尚有改善迹象。** 最后节点仍改善，未证实平台。下一阶段最关键的问题是物理预测收益能否转化为独立航次上的控制响应证据，而非增加隐状态可视化或宣称物理解耦。
