# Phase Harmonic Encoding Result

日期：2026-05-08

## 结论

`wingbeat phase-aware harmonic encoding` 已实现并完成对比。结果是：短跑 validation screen 中 `harmonic3` 最好，但 full-data final/test 中没有超过现有 Transformer phase baseline。当前不建议把 `harmonic3` 设为默认模型；它可以作为已测试过的 periodic inductive bias 写进方法探索和消融结果。

英文表述：

> A wingbeat phase-aware harmonic encoding was tested as an explicit periodic inductive bias. Although it improved the validation-screen result, it did not improve the locked full-data test result over the existing phase-actuator Transformer baseline.

## 协议

固定协议：

```text
split_root: dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1
split_policy: whole-log split
inputs: no acceleration inputs, no past wrench targets
backbone: causal Transformer
history: 128
d_model: 64
layers: 2
heads: 4
dropout: 0.05
dim_feedforward: 128
```

screen 阶段只看 validation，不输出 test 指标。final 阶段只跑锁定候选 `harmonic3` 和现有 `sin_cos/phase_actuator` 对照，并打开 test eval。

## 实现内容

新增 derived features：

```text
phase_corrected_h2_sin = sin(2 phi)
phase_corrected_h2_cos = cos(2 phi)
phase_corrected_h3_sin = sin(3 phi)
phase_corrected_h3_cos = cos(3 phi)
```

新增 recipe / sequence modes：

```text
no_phase_actuator_airdata
raw_phase_actuator_airdata
phase_harmonic_actuator_airdata
causal_transformer_paper_no_accel_v2_phase_harmonic_airdata
```

注意：这里的 `sin_cos` 对照实际使用现有 `phase_actuator_airdata`，它包含 `phase_corrected_rad`、`wing_stroke_angle_rad`、`phase_corrected_sin/cos`、flap frequency、actuator 和 airdata。报告时不要把它误写成“只有 sin/cos”。

## Validation Screen

运行目录：

```text
artifacts/20260508_phase_harmonic_screen/
```

| config | val RMSE | val R2 | fx R2 | fy R2 | fz R2 | roll mx R2 | pitch my R2 | yaw mz R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| harmonic3 | 1.030528 | 0.649209 | 0.955959 | 0.278724 | 0.952747 | 0.460852 | 0.869214 | 0.377757 |
| sin_cos / existing phase | 1.058631 | 0.635574 | 0.952479 | 0.258561 | 0.950012 | 0.425251 | 0.864722 | 0.362421 |
| raw_phase | 1.071835 | 0.641730 | 0.951136 | 0.263924 | 0.948307 | 0.431611 | 0.864884 | 0.390519 |
| no_phase | 1.172649 | 0.612849 | 0.944816 | 0.234484 | 0.934871 | 0.382714 | 0.833107 | 0.347104 |

screen 结论：phase 信息很有用；`no_phase` 明显更差。`harmonic3` 在抽样 validation 上优于现有 phase 表达，因此进入 final。

## Full-Data Final/Test

运行目录：

```text
artifacts/20260508_phase_harmonic_final/
```

| config | val RMSE | val R2 | test RMSE | test R2 | fx R2 | fy R2 | fz R2 | roll mx R2 | pitch my R2 | yaw mz R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sin_cos / existing phase | 0.935145 | 0.680652 | 0.892095 | 0.736392 | 0.968832 | 0.343493 | 0.968979 | 0.620899 | 0.910379 | 0.605771 |
| harmonic3 | 0.941037 | 0.672479 | 0.893017 | 0.730513 | 0.968444 | 0.356869 | 0.968647 | 0.594013 | 0.903498 | 0.591606 |

final 结论：

```text
test RMSE: harmonic3 略差，0.893017 vs 0.892095
test R2: harmonic3 略差，0.730513 vs 0.736392
fy_b: harmonic3 略好，0.356869 vs 0.343493
roll/yaw: harmonic3 更差，mx 0.594 vs 0.621, mz 0.592 vs 0.606
```

从控制角度，`fy_b` 最不关键，roll/yaw 比 side force 更重要。因此即使 harmonic3 稍微改善 fy，也不应该牺牲 roll/yaw 后把它设为默认模型。

## Suspect Log Diagnostics

诊断目录：

```text
artifacts/20260508_phase_harmonic_final/diagnostics/
```

### Existing phase baseline

| case | samples | logs | lateral RMSE mean | lateral R2 mean | fy R2 | mx R2 | mz R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 60478 | 4 | 0.390713 | 0.523388 | 0.343493 | 0.620899 | 0.605771 |
| without_suspect | 42757 | 3 | 0.342510 | 0.627032 | 0.494437 | 0.714431 | 0.672227 |
| suspect_only | 17721 | 1 | 0.487779 | 0.283430 | -0.018422 | 0.445226 | 0.423486 |

### Harmonic3

| case | samples | logs | lateral RMSE mean | lateral R2 mean | fy R2 | mx R2 | mz R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 60478 | 4 | 0.386745 | 0.514163 | 0.356869 | 0.594013 | 0.591606 |
| without_suspect | 42757 | 3 | 0.332476 | 0.634524 | 0.523706 | 0.705126 | 0.674740 |
| suspect_only | 17721 | 1 | 0.493704 | 0.235261 | -0.043140 | 0.385320 | 0.363601 |

诊断解释：

```text
without_suspect 时 harmonic3 的 lateral RMSE mean 更低，fy_b 更好；
但 all-test 的 lateral R2 mean 更低；
suspect_only 上 harmonic3 明显更差；
roll/yaw moment 在 final test 上不如 existing phase baseline。
```

因此，`harmonic3` 更像是对 side-force 局部有帮助，但不是全局更稳的默认模型。

## 可写进论文/组会的判断

中文：

```text
显式加入高阶扑翼相位谐波可以提升抽样 validation 表现，说明网络确实能利用周期性先验。但在锁定协议的 full-data test 上，高阶谐波没有超过现有 phase-actuator Transformer baseline，并且会轻微损害 roll/yaw 力矩预测。考虑到控制任务中 roll/yaw 比 side force 更关键，当前默认模型仍保留现有 phase-actuator Transformer。
```

英文：

> Higher-order wingbeat harmonic features provide a plausible periodic inductive bias and improve the validation-screen result. However, under the locked full-data test protocol, they do not outperform the existing phase-actuator Transformer baseline and slightly degrade the roll/yaw moment predictions. Therefore, the harmonic encoding is reported as an ablated design choice rather than adopted as the default model.

## 下一步建议

短期不继续扩大 harmonic order。更值得做的是：

```text
1. 保持现有 Transformer phase baseline 作为默认模型。
2. 对 roll/yaw 做 targeted improvement，而不是只优化 overall RMSE 或 fy。
3. 如果继续研究 phase novelty，优先尝试 phase-conditioned modulation，而不是继续堆更高阶 harmonic。
```
