# Phase-Conditioned FiLM No-Suspect Result

日期：2026-05-08

## 结论

`Transformer + head/output FiLM` 在 no-suspect whole-log split 上优于现有 phase-actuator Transformer baseline，建议作为当前默认候选继续推进。`input-sequence FiLM` 没有赢，不建议采用。

核心结果：

```text
baseline test RMSE = 0.818145
head FiLM test RMSE = 0.792811
relative RMSE improvement = 3.10%
```

从控制相关通道看，Head FiLM 提升了 `fx_b/fz_b/my_b/mz_b/fy_b`，但 `mx_b` 有小幅下降：

```text
mx_b R2: 0.705137 vs baseline 0.714431
mz_b R2: 0.688185 vs baseline 0.672227
```

因此结论是：**Head FiLM 是有效改进，但后续应重点确认 roll moment 小幅下降是否可接受。**

英文表述：

> Phase-conditioned head FiLM improves the no-suspect full-data test result under the locked whole-log protocol, suggesting that wingbeat phase is better treated as a conditioning variable than only as a concatenated input.

## 协议

主比较使用 no-suspect split：

```text
source split: dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1
filtered split: dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1_no_suspect_log
removed log: log_4_2026-4-12-17-43-30
split_policy: whole_log
feature_set: paper_no_accel_v2
no acceleration inputs: verified from model bundles
```

过滤后的样本数：

```text
train: 326318, removed 0
val: 83836, removed 0
test: 44027, removed 18483
```

注意：parquet 里可以保留原始加速度列用于追溯，但训练输入列没有加速度。三组 final bundle 均验证：

```text
feature_set paper_no_accel_v2
bad_accel []
```

## 方法

对比三组：

```text
baseline:
  causal_transformer_paper_no_accel_v2_phase_actuator_airdata

head/output FiLM:
  causal_transformer_head_film_paper_no_accel_v2_phase_actuator_airdata

input-sequence FiLM:
  causal_transformer_input_film_paper_no_accel_v2_phase_actuator_airdata
```

共享超参：

```text
history: 128
d_model: 64
layers: 2
heads: 4
dropout: 0.05
dim_feedforward: 128
loss: Huber, delta 1.5
```

FiLM conditioner：

```text
phase_corrected_sin
phase_corrected_cos
film_hidden_size: 32
film_scale: 0.1
final FiLM layer zero initialized
```

## Validation Screen

运行目录：

```text
artifacts/20260508_phase_film_screen_no_suspect/
```

screen 阶段没有 test 指标列。

| config | val RMSE | val R2 | fx R2 | fy R2 | fz R2 | mx R2 | my R2 | mz R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 1.058631 | 0.635574 | 0.952479 | 0.258561 | 0.950012 | 0.425251 | 0.864722 | 0.362421 |
| head FiLM | 1.060726 | 0.633633 | 0.954792 | 0.248243 | 0.949293 | 0.420096 | 0.866309 | 0.363065 |
| input FiLM | 1.066312 | 0.637078 | 0.954471 | 0.249341 | 0.948530 | 0.429396 | 0.863307 | 0.377422 |

screen 排名没有直接支持 FiLM：baseline 的 validation RMSE 最低。但由于只有三组，final 阶段仍按计划全部跑完整数据。

## Full-Data No-Suspect Test

运行目录：

```text
artifacts/20260508_phase_film_final_no_suspect/
```

| config | test RMSE | test R2 | fx R2 | fy R2 | fz R2 | mx R2 | my R2 | mz R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| head FiLM | 0.792811 | 0.800301 | 0.976369 | 0.515267 | 0.973509 | 0.705137 | 0.943336 | 0.688185 |
| baseline | 0.818145 | 0.794247 | 0.974439 | 0.494437 | 0.971633 | 0.714431 | 0.938315 | 0.672227 |
| input FiLM | 0.849590 | 0.786635 | 0.972319 | 0.493896 | 0.968473 | 0.692570 | 0.936342 | 0.656211 |

解释：

```text
Head FiLM:
  overall RMSE 明显最好
  overall R2 最好
  fx/fz/my 三个纵向/俯仰主通道最好
  mz yaw 最好
  fy side force 最好
  mx roll 比 baseline 小幅下降

Input FiLM:
  overall 最差
  roll/yaw 都不如 baseline
```

默认选择：

```text
adopt Head FiLM as current best candidate
reject Input FiLM
```

## Lateral Diagnostics

no-suspect test split 上 per-log 横侧向结果：

### Baseline

| log_id | samples | lateral R2 mean | fy R2 | mx R2 | mz R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| log_18_2026-4-15-12-56-08 | 14497 | 0.554536 | 0.393877 | 0.684058 | 0.585674 |
| log_4_2026-4-14-12-30-12 | 14518 | 0.677967 | 0.571725 | 0.742624 | 0.719553 |
| log_34_2026-4-16-19-13-30 | 13742 | 0.635363 | 0.486868 | 0.705581 | 0.713640 |

### Head FiLM

| log_id | samples | lateral R2 mean | fy R2 | mx R2 | mz R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| log_18_2026-4-15-12-56-08 | 14497 | 0.572324 | 0.429872 | 0.669006 | 0.618093 |
| log_4_2026-4-14-12-30-12 | 14518 | 0.692489 | 0.593459 | 0.751767 | 0.732242 |
| log_34_2026-4-16-19-13-30 | 13742 | 0.619682 | 0.480760 | 0.667633 | 0.710653 |

### Input FiLM

| log_id | samples | lateral R2 mean | fy R2 | mx R2 | mz R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| log_18_2026-4-15-12-56-08 | 14497 | 0.538451 | 0.387587 | 0.661264 | 0.566501 |
| log_4_2026-4-14-12-30-12 | 14518 | 0.673775 | 0.581164 | 0.730655 | 0.709506 |
| log_34_2026-4-16-19-13-30 | 13742 | 0.611388 | 0.478374 | 0.664849 | 0.690943 |

Head FiLM 的 lateral mean R2 在两条 log 上提升，一条 log 上下降。下降主要来自 `log_34` 的 roll moment。

## Original Split Robustness

原始 split 只作为诊断，不用于模型选择。坏 log 放回去后，三组模型都被明显拉低。

| config | case | lateral R2 mean | fy R2 | mx R2 | mz R2 |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | all | 0.523388 | 0.343493 | 0.620899 | 0.605771 |
| baseline | without_suspect | 0.627032 | 0.494437 | 0.714431 | 0.672227 |
| baseline | suspect_only | 0.283430 | -0.018422 | 0.445226 | 0.423486 |
| head FiLM | all | 0.522062 | 0.363950 | 0.618339 | 0.583896 |
| head FiLM | without_suspect | 0.636196 | 0.515267 | 0.705137 | 0.688185 |
| head FiLM | suspect_only | 0.251454 | 0.001144 | 0.455315 | 0.297902 |
| input FiLM | all | 0.514451 | 0.344366 | 0.607079 | 0.591908 |
| input FiLM | without_suspect | 0.614226 | 0.493896 | 0.692570 | 0.656211 |
| input FiLM | suspect_only | 0.282623 | -0.014158 | 0.446509 | 0.415518 |

这支持继续把 suspect log 当成数据问题单独处理，而不是混进模型选择。

## 论文/组会表述

中文：

```text
在移除已标记的 suspect log 后，我们比较了普通 phase-actuator Transformer、head/output FiLM 和 input-sequence FiLM。Head FiLM 在 full-data no-suspect test 上取得最低 RMSE 和最高 overall R2，说明将扑翼相位作为调制条件比简单拼接输入更有效。Input FiLM 没有带来提升，可能因为逐时刻调制增加了优化难度。
```

英文：

> Under the no-suspect whole-log protocol, phase-conditioned head FiLM achieves the best full-data test performance, indicating that wingbeat phase is more effective when used to modulate the output representation than when it is only concatenated as an input feature.

谨慎表述：

> The improvement is not uniform across all moment channels; the roll moment R2 slightly decreases, so subsequent work should verify whether this trade-off is acceptable for control-oriented use.

## 下一步

建议：

```text
1. 把 Head FiLM 作为当前 best candidate。
2. 不继续 input-sequence FiLM，除非后续重新设计正则或 layer-wise 位置。
3. 针对 mx_b roll 小幅下降做 focused check：
   - per-log roll error
   - roll/yaw control-weighted score
   - short-horizon control-relevant rollout
4. suspect log 单独走数据质量调查，不再用于模型选择。
```
