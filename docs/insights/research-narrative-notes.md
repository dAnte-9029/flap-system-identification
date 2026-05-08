# Research Narrative Notes

这个文件集中放后续论文、组会、开题/中期汇报里可复用的判断。原则：中文为主，方便快速阅读；关键英文句子保留为可直接改写进 paper 的表述。

## 1. 数据泄露与协议固定

早期高指标不能直接相信，因为随机 sample split 或相邻时间片混到 train/test 会产生严重 temporal leakage。扑翼数据是连续时间序列，相邻样本共享相位、速度、姿态、舵面和气动力状态，模型可能只是在插值同一条日志，而不是泛化到新 flight log。

现在应固定为：

```text
whole-log split
no acceleration inputs
per-target metrics
test set only for final confirmation
```

可用英文：

> To avoid temporal leakage, all reported results use a whole-log split rather than random sample-level splitting.

## 2. Baseline 定位

MLP baseline 是合理起点，因为它检验的是“当前特征能否静态解释 wrench”。但扑翼系统有明显相位记忆和时序效应，MLP 不应被当作最终模型，而是作为 leakage-resistant reference。

目前更合理的叙事：

```text
MLP: static feature baseline
GRU/LSTM/TCN/Transformer: causal temporal backbones
Transformer: 当前综合结果最好
TCN+GRU: 有潜力，但 focused sweep 后不如 Transformer
```

可用英文：

> The MLP baseline serves as a leakage-resistant static reference, while causal sequence models are used to test whether temporal history improves aerodynamic wrench estimation.

## 3. CT-SubNet 暂时舍弃的理由

CT-SubNet 论文适合有可测 output 序列的动态系统辨识，因为 latent state 可以由 past input/output 估计。但这里的 output 是 aerodynamic wrench，本身就是我们想估计的量，并不是飞行中直接可测的传感器输出。因此完整采用 CT-SubNet 会遇到概念矛盾：训练时可用 label 估 latent，部署时却没有真实 wrench。

可以说：

> CT-SubNet-style latent reconstruction is less directly applicable here because the aerodynamic wrench is not an online measurable output, but the target quantity to be inferred.

## 4. 当前模型能力边界

当前 Transformer 不是失败。它已经很好地解释了主导纵向/俯仰动力学：

```text
fx_b R2 ≈ 0.97
fz_b R2 ≈ 0.97
my_b R2 ≈ 0.91
```

这说明模型确实学到了 phase、frequency、airspeed、actuator 与主要气动力/力矩之间的关系。

但它还不是完整六维 wrench 的高保真预测器。横侧向尤其 `fy_b` 明显弱：

```text
mx_b roll: all ≈ 0.62, without suspect ≈ 0.71
mz_b yaw : all ≈ 0.61, without suspect ≈ 0.67
fy_b side: all ≈ 0.34, without suspect ≈ 0.49
```

可用英文：

> The learned model captures the dominant longitudinal and pitch dynamics with high accuracy, but remains less reliable in the lateral-directional channels, especially the side-force component.

## 5. 控制重要性重排

从控制角度看，所有 target 不应等权解释。更合理的优先级：

```text
第一优先级：fx_b, fz_b, my_b
第二优先级：mx_b
第三优先级：mz_b
最低优先级：fy_b
```

因为前进、升力和 pitch 是主要纵向控制；转弯主要靠 roll；yaw 更多是配合 roll 维持稳定；side force 更像扰动项。

所以当前结果可以表述为：

```text
control-relevant dominant channels are strong
roll/yaw are moderate
side force is the main weak channel
```

可用英文：

> From a control perspective, the weakest channel is also the least directly actuated one, while the dominant longitudinal and pitch channels are already well captured.

## 6. 横侧向诊断结论

横侧向差很大程度被一个 suspect log 拖累：

```text
suspect log: log_4_2026-4-12-17-43-30

all test:
  lateral mean R2 = 0.523
  fy_b R2 = 0.343

without suspect:
  lateral mean R2 = 0.627
  fy_b R2 = 0.494

suspect only:
  lateral mean R2 = 0.283
  fy_b R2 = -0.018
```

但这不是全部原因。去掉 suspect log 后，`fy_b` 仍明显低于纵向通道。因此应报告：

```text
test_all
test_without_suspect
suspect_only
```

不要直接静默删除该日志。更好的说法是将其标记为 hard/suspect case，并检查 airspeed 缺失、merge、label 构造。

可用英文：

> A single suspect log substantially depresses the lateral-directional metrics; therefore, we report both all-test and without-suspect results rather than silently removing it.

## 7. 工况分桶解释

最差的 bins 包括：

```text
phase [0, pi/2)
phase [3pi/2, 2pi)
servo_rudder positive
elevon_diff negative
```

但这些不能直接解释成“这些工况必然模型差”。其中 positive rudder 和 negative elevon_diff 与 suspect log 高度混杂。去掉 suspect log 后这些 bin 会明显变好。

更准确的说法：

```text
regime degradation exists, but is confounded with log-level distribution shift
```

可用英文：

> The apparent regime-dependent degradation is partially confounded with log-level distribution shift, especially in positive-rudder and negative-elevon-difference conditions.

## 8. Phase/Lag 与 residual correlation

Batch 2 诊断显示，整体不是明显相位滞后问题。只有 suspect log 的 `fy_b` 有 1 sample lag，RMSE 改善很小：

```text
fy_b suspect log:
  zero-lag RMSE = 1.460
  best-lag RMSE = 1.425
```

Residual correlation 最大也只有约 0.086，说明不是某个简单特征漏掉就能解决。

可用英文：

> Lag analysis shows no global phase-shift failure, and residual correlations are weak, suggesting that the remaining errors are not explained by a single missing scalar feature.

## 9. 长时间 rollout 风险

单步 wrench estimation 和长时间 rollout 是两个难度等级。当前模型可以作为局部气动力/力矩估计器，但不应直接包装成高精度 long-horizon simulator。

原因：

```text
one-step error 在 rollout 中会积分累积
横侧向小偏差会导致 attitude/position drift
roll/yaw 中等误差可能影响转弯长期轨迹
fy_b 噪声虽控制重要性低，但会影响完整状态预测
```

论文/组会里的定位可以改成：

```text
short-term aerodynamic wrench estimator
control-relevant channel analysis
long-horizon rollout as future work
```

可用英文：

> The current model is better viewed as a local aerodynamic wrench estimator than as a standalone long-horizon predictive simulator.

> Long-horizon rollout requires dynamically consistent and low-bias predictions across all force and moment channels; otherwise small one-step residuals accumulate into trajectory drift.

## 10. 未来工作叙事

短期：

```text
固定 leakage-resistant protocol
报告 per-target 和 control-weighted metrics
加入 with/without suspect log 结果
重点提升 roll/yaw，而不是只盯 fy_b
```

中期：

```text
检查 suspect log 的 airspeed/label pipeline
做 control-weighted loss 或 per-target weighting
做 short-horizon rollout，而不是马上长时间 rollout
```

长期：

```text
physics-informed constraints
closed-loop residual correction
hybrid model: learned wrench + simplified flight dynamics
uncertainty-aware controller
```

可用英文：

> These results motivate a staged roadmap: first validate leakage-resistant one-step wrench estimation, then evaluate short-horizon consistency, and finally integrate the learned model into a control-oriented rollout framework.

## 11. Phase-aware harmonic encoding 消融

已测试 `wingbeat phase-aware harmonic encoding`：

```text
sin(phi),  cos(phi)
sin(2phi), cos(2phi)
sin(3phi), cos(3phi)
```

validation screen 中 `harmonic3` 最好：

```text
harmonic3 val RMSE = 1.0305
existing phase val RMSE = 1.0586
no phase val RMSE = 1.1726
```

这说明 phase 信息确实重要，高阶相位谐波也能作为周期性先验被网络利用。

但 full-data final/test 中，`harmonic3` 没有超过现有 phase-actuator Transformer：

```text
existing phase:
  test RMSE = 0.8921
  test R2 = 0.7364
  mx_b R2 = 0.6209
  mz_b R2 = 0.6058

harmonic3:
  test RMSE = 0.8930
  test R2 = 0.7305
  mx_b R2 = 0.5940
  mz_b R2 = 0.5916
```

`harmonic3` 对 `fy_b` 有轻微帮助：

```text
fy_b R2: 0.3569 vs 0.3435
without suspect fy_b R2: 0.5237 vs 0.4944
```

但从控制重要性看，roll/yaw 比 side force 更关键，因此当前不应把 `harmonic3` 设为默认模型。

可用英文：

> Higher-order wingbeat harmonic features provide a plausible periodic inductive bias and improve the validation-screen result. However, under the locked full-data test protocol, they do not outperform the existing phase-actuator Transformer baseline and slightly degrade the roll/yaw moment predictions.

后续如果继续做 phase novelty，更建议尝试：

```text
phase-conditioned modulation
wingbeat-conditioned feature modulation
```

而不是继续简单堆更高阶 harmonic。

## 12. Phase-conditioned FiLM no-suspect 结果

后续模型选择改用 no-suspect whole-log split：

```text
removed suspect log: log_4_2026-4-12-17-43-30
split_policy: whole_log
feature_set: paper_no_accel_v2
no acceleration inputs
```

在这个协议下，比较了：

```text
baseline Transformer
Transformer + head/output FiLM
Transformer + input-sequence FiLM
```

最终 no-suspect test：

```text
baseline:
  RMSE = 0.8181
  R2 = 0.7942
  fy_b R2 = 0.4944
  mx_b R2 = 0.7144
  mz_b R2 = 0.6722

head FiLM:
  RMSE = 0.7928
  R2 = 0.8003
  fy_b R2 = 0.5153
  mx_b R2 = 0.7051
  mz_b R2 = 0.6882

input FiLM:
  RMSE = 0.8496
  R2 = 0.7866
  fy_b R2 = 0.4939
  mx_b R2 = 0.6926
  mz_b R2 = 0.6562
```

判断：

```text
Head/output FiLM 是当前 best candidate。
Input-sequence FiLM 不采用。
Head FiLM 的整体 RMSE 提升约 3.1%，纵向/俯仰和 yaw 都提升；
但 roll mx_b R2 小幅下降，需要后续 focused check。
```

可用英文：

> Under the no-suspect whole-log protocol, phase-conditioned head FiLM achieves the best full-data test performance, indicating that wingbeat phase is more effective when used to modulate the output representation than when it is only concatenated as an input feature.

谨慎句：

> The improvement is not uniform across all moment channels; the roll moment R2 slightly decreases, so subsequent work should verify whether this trade-off is acceptable for control-oriented use.
