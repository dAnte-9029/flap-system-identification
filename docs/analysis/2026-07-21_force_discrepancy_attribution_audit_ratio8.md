# EDA0 — DeLaurier 纵向力模型误差归因审查

> 本报告基于 RATIO=8 的修正 phase contract。
> 旧 ratio=7.5 EDA0 仅保留用于历史复现，不得用于后续 correction 结构选择。

日期：2026-07-21T13:56:33.653263+00:00

状态：analysis-only；未训练正式 correction model；test partition 未读取

run ID：`20260721T135633Z_09b4bb6`

Git：`09b4bb6e497a677b5da871e3b600f4ebd6e1dd39`

## 1. 执行摘要

1. **Fx 能量结构：**cycle-mean 占 per-log macro residual energy 的 14.0%，zero-mean wingbeat 占 86.0%；对应 `cycle_mean_residuals.csv` 与 Figure 3–4。
2. **Fz 能量结构：**cycle-mean 占 33.5%，zero-mean wingbeat 占 66.5%；对应 `cycle_mean_residuals.csv` 与 Figure 3–5。
3. **Fx phase：**fixed phase offset=-0.554 rad；fixed-delay estimate=-21.675 ms；H1/H2 circular RMSE=0.068/0.058 rad。
4. **Fz phase：**fixed phase offset=-0.159 rad；fixed-delay estimate=-6.382 ms；H1/H2 circular RMSE=0.106/0.100 rad。
5. **输入修复判定：**phase convention first=`no`，fixed delay first=`no`；Fx/Fz offset 的 component 间一致性是全局输入修复的必要条件；这只是只读假设审查，没有改 label 或 prior。
6. **Mean correction：**Fx=`no`，Fz=`yes`。
7. **Phase correction：**Fx=`yes`，Fz=`yes`；建议 harmonic order range=[2, 4]；fx_b K1–K4 cumulative coverage=29.2%/84.9%/95.8%/98.1%；fz_b K1–K4 cumulative coverage=89.1%/94.5%/96.8%/99.0%。
8. **Condition dependence：**phase-conditioned 相对 phase-only 的 validation equal-log macro gain 为 Fx=23.3%、Fz=9.5%；建议变量=['angle_of_attack', 'flapping_frequency']。
9. **Short history：**最佳预定义短历史相对 static gain 为 Fx=9.7%、Fz=15.8%；dynamic model=`yes`，TCN=`no`。
10. **Label robustness：**label variant 的 phase-bin 平均 1-SD 为 0.822 N；decision=no；Fx optimal shift 对已有 variant 的变化=0.127 rad；Fz residual waveform variant correlation=0.984；对应 `label_robustness.csv`、`label_uncertainty_phase.csv` 与 Figure 13。
11. **DeLaurier prior value：**matched-capacity gain 为 Fx=-15.5%、Fz=-78.3%；incremental value=`no`。
12. **Trim：**WT1 sensitivity artifact 缺失，不能定量给出 trim frequency shift；仅报告 cycle-mean Fz discrepancy=4.694 N。

上述结论的 residual 均定义为 total reconstructed effective force 减去 wing-only DeLaurier prior；不能把它直接称为纯 wing aerodynamic error。

## 2. 数据与模型 contract

- Label：body FRD 下、whole-aircraft CG contract 对应的 total reconstructed effective force，包含未建模或未分离的 tail、body、interaction、disturbance、asymmetry 与 reconstruction error。
- Prior：`/home/zn/flap-system-identification/artifacts/20260721_delaurier_attitude_aware_3b5d4ec_ratio8_phasefix_trainval_v4` 的 wing-only DeLaurier force；artifact ID=`delaurier_attitude_aware_3b5d4ec_ratio8_phasefix_trainval_v4`，lifecycle=`active`，physics commit=`3b5d4ec1d28f1384cf042402992ad7ea59995f49`。
- Primary residual：`F_rec_effective - F_D_wing-only`，仅分析 Fx/Fz。
- 使用 partition：`['train', 'val']`；test 未加载，manifest 的 `test_rows_loaded=0`。
- Tail-subtracted residual：本次没有使用；没有把 IsaacLab tail runtime 复制到本仓库，也没有校准 tail。
- 对齐 key：`['log_id', 'timestamp_us']`，不依赖行顺序或 DataFrame index。
- 复用：`['conventions.phase wrap/direction', 'physics.delaurier airflow/dynamic-twist/strip-wrench', 'physics.baselines.wing_only aggregate component diagnostics', 'canonical dataset phase/cycle/sample identity']`。
- 从旧 script-only 逻辑迁移到 package：`['phase/condition diagnostic patterns into authoritative analysis package', 'component residual attribution away from script-only orchestration']`。
- 新增：`['active/legacy prior registry with fail-closed resolution', 'attitude-aware train/validation keyed prior materialization', 'strict keyed audit alignment', 'phase-delay hypotheses', 'mean/WB decomposition', 'authoritative half-stroke, harmonic, repeatability, probes, decision matrix']`。
- 本次未清理的 legacy 重复：`['scripts/analyze_delaurier_residual_phase.py', 'scripts/analyze_delaurier_residual_frequency.py', 'scripts/analyze_delaurier_residual_conditions.py', 'scripts/analyze_component_residual_attribution.py', 'legacy label diagnostic scripts']`；保留旧 CLI 语义与兼容入口。

## 3. 数据质量与覆盖

- 使用 25 个 logs：`['log_0_2026-4-14-10-21-28', 'log_0_2026-4-14-11-50-14', 'log_10_2026-4-15-11-37-24', 'log_12_2026-4-15-11-57-08', 'log_13_2026-4-15-12-04-42', 'log_17_2026-4-15-12-47-22', 'log_19_2026-4-15-13-02-32', 'log_1_2026-4-12-16-33-22', 'log_1_2026-4-14-12-01-42', 'log_20_2026-4-15-13-12-34', 'log_24_2026-4-16-09-49-52', 'log_25_2026-4-16-09-57-28', 'log_26_2026-4-16-10-09-34', 'log_27_2026-4-16-10-17-56', 'log_30_2026-4-16-10-36-20', 'log_31_2026-4-16-18-41-26', 'log_32_2026-4-16-18-53-24', 'log_36_2026-4-16-19-24-20', 'log_38_2026-4-16-19-37-00', 'log_3_2026-4-14-10-52-38', 'log_3_2026-4-14-12-19-22', 'log_5_2026-4-12-17-51-44', 'log_5_2026-4-15-10-30-38', 'log_6_2026-4-15-10-48-46', 'log_8_2026-4-15-11-23-44']`。
- 日期：`['2026-04-12', '2026-04-14', '2026-04-15', '2026-04-16']`。
- cycle 总数 15711，接受 15526，拒绝比例 1.18%。
- 主要 rejection reasons：`{'too_few_samples;incomplete_phase_coverage': 94, 'incomplete_phase_coverage': 72, 'too_few_samples;non_monotonic_timestamp;incomplete_phase_coverage': 18}`；没有静默丢弃 cycle。
- Alignment evidence：见 `alignment_report.json` 与 `alignment_mismatches.csv`。
- Cycle evidence：见 `cycle_quality.csv` 与 `cycle_rejection_reasons.csv`。
- Condition envelope 与各 log 覆盖：见 `condition_dependence.csv`、Figure 10 和 `date_level_summary.csv`。

## 4. Fx 误差归因

Fx 的 per-log macro cycle-mean residual 为 -1.147 N，mean/WB energy fraction 为 14.0%/86.0%。固定 offset 与 fixed delay 的相对拟合见 Figure 2；单凭 shift 后 RMSE 下降不足以修改输入。绝对积分最大的 physical half-stroke 为 downstroke，equal-log macro 积分为 -2.309 N rad，跨 log 同号比例=100.0%。Joint condition probe gain 为 23.3%；单变量 U/AoA/f gain=1.2%/1.8%/17.7%。Cross-log 证据见 Figure 11–12，label-variant band 见 Figure 13。

## 5. Fz 误差归因

Fz 的 per-log macro cycle-mean residual 为 4.740 N，mean/WB energy fraction 为 33.5%/66.5%。绝对积分最大的 physical half-stroke 为 downstroke，equal-log macro 积分为 15.148 N rad，跨 log 同号比例=100.0%。该 half-stroke 指标来自 zero-mean waveform，因此半拍正负面积不会被误算为 cycle mean；4.740 N 的 cycle-mean discrepancy 是独立 branch，而不是由 zero-mean 半拍峰值代数产生。该 half-stroke identity 来自 canonical `q=A sin(phi)` direction helper，不使用未经验证的 `[0,pi)` 命名。Joint condition gain=9.5%；单变量 U/AoA/f gain=2.4%/5.7%/1.3%。WT1 sensitivity artifact 缺失，不能定量给出 trim frequency shift；仅报告 cycle-mean Fz discrepancy=4.694 N。 过大的半拍究竟是 localized peak、reversal 邻域还是整拍偏差，量化见 `half_stroke_residuals.csv` 与 Figure 5。

## 6. 物理机制证据

| 候选机制 | 状态 | 证据与边界 |
|---|---|---|
| phase convention | no | Fx/Fz shift 的方向与 component 间一致性；Figure 2 |
| logging/filter fixed delay | no | H1/H2 circular RMSE；不能与真实动态滞后完全区分 |
| circulatory normal force | 证据不足 | component shape association 仅作候选，Figure 6/8 |
| apparent-mass force | 证据不足 | dN_a 与 local sensitivity，不代表已校准参数 |
| chordwise force | 证据不足 | dT_s/dD components 与 Fx shape association |
| dynamic twist | 证据不足 | symmetric local sensitivity；未优化 twist |
| attached-flow assumption | 证据不足 | active primary 与 frozen diagnostic 均为 attached flow；condition/shape association 尚不能唯一归因于该假设 |
| tail/body contamination | 部分支持 | label/prior scope mismatch 确认存在，但未分离量化 |
| label reconstruction | 部分支持 | label variant 的 phase-bin 平均 1-SD 为 0.822 N；decision=no |
| unsteady/flexible-wing memory | 部分支持 | linear short-history probe，不能推出 TCN 必要性 |

Component diagnostic：
- fx_b: 与 residual waveform shape correlation 绝对值最大的是 `dD_f` (equal-log mean r=-0.679)；这是候选关联，不是唯一物理因果。
- fz_b: 与 residual waveform shape correlation 绝对值最大的是 `dN_c` (equal-log mean r=-0.799)；这是候选关联，不是唯一物理因果。

primary-vs-diagnostic prior RMSE: Fx=0.000 N, Fz=0.000 N；非零 mismatch 时 component 只作 hypothesis probe。Local physical sensitivity：
- fx_b: 最大 absolute macro sensitivity similarity 为 `phase_offset_rad` (r=-0.826)；仍不是参数校准证据。
- fz_b: 最大 absolute macro sensitivity similarity 为 `normal_force_scale` (r=-0.908)；仍不是参数校准证据。

进入 decision matrix 的 physical parameter candidates：`['normal_force_scale']`（由可配置 similarity 与 step-stability thresholds 生成）。

## 7. Residual 可重复性

- fx_b: per-log vs macro r=0.990；same-date different-log r=0.991；cross-date r=0.975。
- fz_b: per-log vs macro r=0.989；same-date different-log r=0.988；cross-date r=0.972。

Within-log cycle variance、cross-log matrix、same-date/different-date 与 matched-condition distance 分别见 `waveform_repeatability.csv`、`log_waveform_correlations.csv` 和 `date_level_summary.csv`。主要结论使用 equal-log macro；pooled sample 只作补充，长日志不会自动获得更大结论权重。

## 8. 对 correction 结构的建议

| 项目 | 决策 |
|---|---|
| 输入 phase convention 修复 | no |
| fixed delay 修复 | no |
| Fx cycle-mean branch | no |
| Fz cycle-mean branch | yes |
| Fx phase harmonic | yes |
| Fz phase harmonic | yes |
| condition interaction | ['angle_of_attack', 'flapping_frequency'] |
| harmonic order range | [2, 4] |
| cyclic spline | 暂不需要；先用 K=1–4 evidence 判断低阶 basis 是否足够 |
| dynamic residual | yes |
| TCN | no；本 audit 不训练 TCN |

推荐顺序是：保留现有 input phase/timestamp contract（当前不支持全局 offset 或 fixed-delay 修复）；实现独立 cycle-mean 与低阶 phase/condition branch；只有 static/history validation macro gain 足够大且不能由 fixed lag 解释时，才进入 dynamic residual 设计。Figure 16 汇总该决策链。

## 9. DeLaurier prior 的作用

Matched-capacity comparison 使用相同 harmonic order、condition features、ridge grid、train/validation logs 与 sample weighting。Overall validation equal-log macro gain 为 Fx=-15.5%、Fz=-78.3%；cycle-mean RMSE gain 为 Fx=-33.1%、Fz=-161.3%。因此结构化 overall verdict=`no`：Fx/Fz overall 与 cycle mean 均未显示稳定增量价值；complete-log validation 之外没有独立 condition-extrapolation 证据。见 `matched_capacity_prior_probe.csv` 与 Figure 15。论文中宜描述为“在本 flight envelope 与 matched low-capacity basis 下，DeLaurier prior 未提供稳定 overall incremental information”，而不是预设 prior 必然有效。

## 10. 限制

1. Total effective label 与 wing-only prior 存在 scope mismatch；residual 不是纯 wing aerodynamic error。
2. Tail 尚未最终冻结，本次没有 tail-subtracted residual；body force 也未单独建模。
3. Moment 不在 EDA0 范围。
4. Label reconstruction uncertainty 只使用已有 variants；没有随意生成新 labels。
5. Current flight envelope 仅覆盖 manifest 中 train/validation logs 与 dates。
6. Test partition 尚未使用，也不能用来选择 phase offset、delay、K 或 condition features。
7. Component/spanwise/local sensitivity 来自 opt-in frozen offline diagnostic；若其 total 与 primary prior 不同，结果只能用于候选机制排序。
8. WT1 artifact 缺失时，trim shift 不能定量计算。

## 11. 下一阶段实施建议

1. 不先修改全局 phase/timestamp；若后续获得独立同步证据，再作为单独 input-contract 变更并回归量化。
2. 若 mean branch verdict 为 yes，先实现低容量 per-cycle/slow-condition mean correction；保持 raw prior 可选。
3. 依据 Figure 9 的 energy coverage 实现建议范围内的 Fourier harmonic residual；只在 Figure 10/condition probe 显示稳定 gain 时加入 U/alpha/f interactions。
4. 使用 complete-log train/validation protocol 比较 correction candidates；test 保持锁定。
5. 以 opt-in linear short-history branch 复核 4-sample gain；它是 dynamic residual 候选而非最终模型。本结果不支持直接启动 TCN 训练。

## 12. Artifact 索引

- Output directory：`/home/zn/flap-system-identification/outputs/force_discrepancy_attribution/20260721T135633Z_09b4bb6`
- Git commit：`09b4bb6e497a677b5da871e3b600f4ebd6e1dd39`
- Dataset：`/home/zn/flap-system-identification/dataset/canonical_v0.4_training_ready_split_measured_massprops_ratio8_phasefix_v3`
- Split identity：`canonical_v0.4_training_ready_split_measured_massprops_ratio8_phasefix_v3`
- Prior：`/home/zn/flap-system-identification/artifacts/20260721_delaurier_attitude_aware_3b5d4ec_ratio8_phasefix_trainval_v4`（`delaurier_attitude_aware_3b5d4ec_ratio8_phasefix_trainval_v4`，`active`）
- 关键 CSV：`phase_alignment_cycles.csv`、`cycle_mean_residuals.csv`、`half_stroke_residuals.csv`、`harmonic_cycle_summary.csv`、`condition_dependence.csv`、`component_residual_similarity.csv`、`matched_capacity_prior_probe.csv`
- 关键 figures：`figures/figure_01_phase_curves.png` 至 `figures/figure_16_decision_summary.png`
- Decision：`decision_summary.json`
- 复现命令：

```bash
python scripts/audit_force_discrepancy_attribution.py --dataset-root dataset/canonical_v0.4_training_ready_split_measured_massprops_ratio8_phasefix_v3 --split-manifest dataset/canonical_v0.4_training_ready_split_measured_massprops_ratio8_phasefix_v3/dataset_manifest.json --prior-root artifacts/20260721_delaurier_attitude_aware_3b5d4ec_ratio8_phasefix_trainval_v4 --prior-registry configs/physics/delaurier_prior_registry.yaml --partitions train validation --output-root outputs/force_discrepancy_attribution --report-path docs/analysis/2026-07-21_force_discrepancy_attribution_audit_ratio8.md --strict-require-label-variants --strict-require-component-diagnostics
```
