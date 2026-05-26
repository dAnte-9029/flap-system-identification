# Component Residual Attribution Diagnostic

本次诊断把 DeLaurier/grey-box 模型解释不了的部分写成显式 residual，然后用观测变量做分桶、相关性排序和 held-out-log 小模型验证。定位是找 `candidate mismatch sources` 和下一步修正方向，不做严格气动因果归因。

## 输入与产物

- Plan: `docs/plans/2026-05-26-component-residual-attribution-diagnostic-plan.md`
- Script: `scripts/analyze_component_residual_attribution.py`
- Tests: `tests/test_component_residual_attribution.py`
- Artifacts: `artifacts/20260526_component_residual_attribution_v1`
- Main tables:
  - `residual_variable_rankings.csv`
  - `per_log_residual_variable_rankings.csv`
  - `residual_feature_group_ablation.csv`
  - `per_log_residual_feature_group_ablation.csv`
  - `residual_variable_bins.csv`

残差定义：

```text
force_prior_residual     = label_force - prior_force
force_corrected_residual = label_force - corrected_force
moment_prior_residual    = label_moment - prior_moment
moment_current_residual  = label_moment - current_moment_prediction
```

当前重点看：

```text
fy_b: force_corrected_residual
mx_b/my_b/mz_b: moment_current_residual
```

## 关键结果

### 1. 当前 grey-box / dynamic-arm 后 residual 已经比原始 prior 小很多

Test residual RMSE:

| Channel | Prior residual RMSE | Current residual RMSE |
| --- | ---: | ---: |
| fx_b | 5.4203 | 1.8173 |
| fy_b | 3.5236 | 1.3990 |
| fz_b | 13.1050 | 3.2575 |
| mx_b | 0.07142 | 0.00294 |
| my_b | 0.60173 | 0.00217 |
| mz_b | 0.10190 | 0.000713 |

这说明当前 pipeline 已经把非常大的 physics-prior mismatch 压下来了。下面 residual attribution 分析的是“剩余误差”，不是原始 DeLaurier prior 的全部误差。

### 2. fy_b 剩余误差和 yaw/侧向-方向变量有关，但解释量很小

`fy_b` 的 top association 包括：

```text
body_rate_r
q_dyn_x_body_rate_r
q_dyn_x_servo_rudder
elevon_diff_proxy
servo_rudder
```

Held-out feature-group ablation:

| Feature group | Test RMSE reduction | Residual R2 |
| --- | ---: | ---: |
| all_candidate | 1.84% | 0.036 |
| body_rates | 1.23% | 0.025 |
| lateral_tail | 0.12% | 0.002 |
| tail_controls | 0.09% | 0.002 |

解释：`fy_b` residual 里确实有和 yaw rate / rudder / elevon 差动相关的结构，但简单的观测变量线性残差模型只能解释很小一部分。更合理的说法是：

> The lateral-force residual is weakly but consistently associated with lateral-directional motion and control variables, suggesting remaining lateral-directional model mismatch and/or low-confidence target components.

不能说 `fy_b` 误差就是 rudder 或某个单一变量导致的。

### 3. mx_b 的 residual 最清楚地和 roll-rate 相关

`mx_b` top association:

```text
body_rate_p
q_dyn_x_body_rate_p
phase_sin_1 / phase_sin_2
```

Held-out ablation:

| Feature group | Test RMSE reduction | Residual R2 |
| --- | ---: | ---: |
| all_candidate | 8.12% | 0.156 |
| body_rates | 5.74% | 0.111 |
| tail_controls | 0.09% | 0.002 |

解释：roll moment 剩余误差里最明显的结构来自 roll-rate / roll damping 相关项，而不是简单尾舵/舵面静态项。这支持后续优先检查：

```text
roll damping
body/wing interaction damping
dynamic moment arm
rate-dependent residual
```

### 4. my_b 和 mz_b 也有 body-rate 结构，但解释量弱于 mx_b

`my_b`:

```text
body_rate_q
q_dyn_x_body_rate_q
phase terms
```

best held-out reduction:

```text
all_candidate: 2.17%
body_rates:   0.44%
```

`mz_b`:

```text
body_rate_r
body_rate_p
q_dyn_x_body_rate_p/r
phase terms
```

best held-out reduction:

```text
all_candidate: 2.84%
body_rates:   1.18%
```

解释：pitch/yaw moment residual 中仍有 rate-dependent 结构，但用当前这组 proxy 和线性小模型只能解释有限部分。它们可以作为后续修正方向，但不应该被写成已经完成物理归因。

## Per-log 稳定性

`all_candidate` 的 test per-log residual RMSE reduction 范围：

| Target | Mean | Min | Max |
| --- | ---: | ---: | ---: |
| fy_b | 1.86% | 1.30% | 2.27% |
| mx_b | 7.93% | 6.24% | 10.78% |
| my_b | 2.00% | 0.97% | 2.87% |
| mz_b | 2.86% | 1.71% | 4.73% |

这说明 `mx_b` 的可解释残差结构最稳定；`fy_b/my_b/mz_b` 有弱结构，但增益较小。

## 对模型修正的含义

优先级建议：

1. `mx_b`: 先做 rate-dependent roll moment correction，尤其是 `p` 和 `q_dyn * p`。这是当前最站得住的 residual attribution 方向。
2. `fy_b`: 不建议直接认为是尾翼模型问题。可以尝试 yaw-rate/rudder/lateral-directional residual，但预期提升有限；更重要的是继续检查 label quality、未观测侧风/侧滑和高频低置信度成分。
3. `my_b/mz_b`: 可以把 body-rate 作为辅助 residual feature，但当前证据强度弱于 `mx_b`。
4. 不建议只靠一个简单 linear residual head 作为最终方法。它适合作为诊断工具；如果做模型修正，可以发展成 bounded grey-box correction 或小型 residual module，并且必须保留 held-out-log 验证。

## 论文表述边界

可以写：

> Residual attribution revealed structured, held-out-log-consistent mismatch associated with body-rate and lateral-directional variables. The roll-moment residual showed the clearest rate-dependent structure, while the remaining lateral-force residual was only weakly explainable by measured yaw-rate and control proxies.

不要写：

> The residual is caused by rudder deflection.

更稳的中文表述：

> 这些结果说明，当前模型剩余误差并非完全随机，其中一部分与机体角速度和横航向控制变量有关；但由于该分析基于观测飞行日志，变量耦合、label noise、proxy 不准和 baseline 选择都会影响结论，因此它只能作为模型修正优先级的依据，而不能作为严格气动因果归因。

## 验证

已运行：

```bash
python -m pytest tests/test_component_residual_attribution.py -q
python -m pytest tests/test_component_residual_attribution.py tests/test_delaurier_greybox_force_correction.py tests/test_dynamic_arm_moment_head.py tests/test_analyze_delaurier_residual_conditions.py -q
python scripts/analyze_component_residual_attribution.py --split-root dataset/canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1 --force-prediction-root artifacts/20260525_delaurier_greybox_force_correction_v1 --moment-prediction-root artifacts/20260525_dynamic_arm_moment_head_v1 --prior-root artifacts/delaurier_physical_prior_v1 --output-root artifacts/20260526_component_residual_attribution_v1 --quantile-bins 10 --min-samples 500
```

测试结果：

```text
7 passed
20 passed
```

完整诊断已生成所有计划内 CSV、README 和 figures。

