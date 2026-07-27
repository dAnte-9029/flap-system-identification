# C3 — 静态纵向力修正模型正式选择

日期：2026-07-27

状态：`READY FOR C4`

## 1. 执行摘要

C3 在 20 个 train logs 上完成了 deterministic 5-fold grouped CV，并在读取
validation 之前封存 shortlist。Stage B 只评估该 shortlist，在 5 个 validation
logs 上按 per-log macro total-force RMSE 和 one-standard-error rule 分别选择 Fx
与 Fz。Test label 与 test prediction 均未读取。

Fx 最终选择 `complete_7809de4d75fb`：no-prior、K=4、mean 与 waveform
均只使用 frequency condition，18 个 learned coefficients。它的 validation
macro RMSE 为 1.0375 N；绝对最低值为 0.9928 N，但需要 27 个 coefficients。
最低值的 one-SE 阈值为 1.0510 N，因此选择 18-coefficient 模型。

Fz 最终选择 `complete_61ead6598f5c`：no-prior、intercept-only mean branch、
frequency-conditioned K=4 waveform branch，共 17 个 coefficients。它的 validation
macro RMSE 为 1.8665 N；绝对最低值为 1.8465 N，one-SE 阈值为 1.9650 N。
所选模型牺牲 0.0200 N macro RMSE，换取更少 coefficient 和更简单 mean branch。

两分量的 leave-one-validation-log-out 选择均为 5/5 不变，没有被单个 validation
log 主导。Matched-capacity fixed/shaped prior 在 Fx/Fz 上都没有展示稳定增量预测
价值，正式 verdict 均为
`No stable incremental predictive value demonstrated`。

## 2. Branch 与 commits

- Branch：`feat/static-correction-model-selection`。
- 起始基线：`dd4b44b55deda8d6d9585873714fbe61edcb9f2a`，与当时
  `origin/main` 一致。
- 初始 implementation commit：`0a145ca1869eae6c789fffbe5fd9fc23cf11f9ae`。
- Validation schema follow-up：`a41c184b204b8f3d1aadd01941cc08dcf21cb552`。
- Stage manifest follow-up：`bd85b7ac9fc7a0bb80c6349922ff64a47f5e8ccc`。
- 正式运行绑定的 clean implementation tip：
  `b49d61bb00895c2d8c3ab653882b28bd10a17334`。
- Commit B：本报告、selected specs 与小型 CSV 所在的
  `docs(analysis): report static correction selection` 提交。

两次 Stage B fail-closed 尝试分别暴露了 normalized waveform label 路由和 immutable
manifest lifecycle 问题；均在读取 test 前停止。失败/被 supersede 的 selected bundles
保留为 `static_correction_selected_train_failed_a41c184` 与
`static_correction_selected_train_superseded_bd85b7a`，不作为正式证据。

## 3. Authoritative dataset、prior 与 artifact

- Canonical dataset：
  `canonical_v0.4_training_ready_split_measured_massprops_ratio8_phasefix_v3`；
  manifest SHA-256=`aa12aa66f762390ab1a356b94916694f5ed9689af670f313544aeb57a250cc07`。
- Active DeLaurier prior：
  `delaurier_attitude_aware_3b5d4ec_ratio8_phasefix_trainval_v4`；
  manifest SHA-256=`c86b34ea10328207b1867b117d44656eacf54b751e883a1ee99de0656695200c`。
- Correction-ready artifact：
  `longitudinal_mean_wb_ratio8_20260721T140238Z_09b4bb6`；
  manifest SHA-256=`5298c1c3ef8aabaa96fc439335883d333b4d9c60608e60c6a2194f779bfa0ad2`。
- Ratio/phase/frequency：
  `ratio8_v1` /
  `hall_indexed_mechanical_phase_ratio8_v1` /
  `flap_frequency_ratio8_v1`。
- Frame/unit/sign：body FRD、N、`+Fx` forward、`+Fz` down。
- Target 仍是 provisional whole-aircraft effective longitudinal force；tail/body
  未扣除，moment 不在范围。

独立末次 audit 重新计算的 C1 table hashes 为：

- `cycle_table.parquet`：
  `eb2e07219afbb813c24f2be1cb0bb7836cacd321569bc40bc07a5e0f4f0cfcbf`；
- `waveform_table.parquet`：
  `d154b357ed8b659dca381fd2112fa99877e1d6b0668d69c0be2aae2cd887ae8e`。

## 4. Search space 与阶段隔离

冻结配置是
`configs/correction/static_correction_model_selection_v1.yaml`，config semantic
hash=`ad368816cb7db2976156e0dc95910f0f4f016172fa46003f319dac932a646e84`。

Mean first pass 对每个分量搜索 5 retentions × 4 conditions × 5 ridge，共
100 candidates；两个分量共 200。Waveform first pass 对每个分量搜索
5 retentions × 4 harmonic orders × 4 conditions × 5 ridge，共 400 candidates；
两个分量共 800。Weighting 不进入第一轮全组合，只对 branch shortlist 比较
equal-cycle、equal-log 与 equal-date。

Stage A CLI：

```text
scripts/build_static_correction_train_cv_shortlist.py
```

Stage B CLI：

```text
scripts/evaluate_static_correction_validation_finalists.py
```

最终 run 为 `20260727T080823Z_b49d61b`。Stage A 只调用 train loader；sealed
shortlist hash=`0964e12bca25cf6b9efd01585f8b0f1e8e6524404d9d474d4ba71faf959f22a1`。
Stage B 在扫描 validation 前验证 shortlist、config 与 artifact hash，不能增加 candidate。

Normalization 固定为 C1 full-train statistics：

```text
normalization_source = full_train_partition_from_C1
validation_participated = false
```

## 5. Train grouped-CV design 与 fold composition

Fold assignment 使用 flight date 内按 cycle count 降序、再贪心分配到当前总 cycle
数最少的 fold，固定 fold-index tie break。20 个 train logs 各自完整落入一个 fold，
没有 sample/cycle random split。Assignment
hash=`3b57b4686bbd04126d4771676667c8dc17f3d7b0a4de4bda5c9324a600767ab2`。

| Fold | Cycles | Logs | Dates |
|---:|---:|---|---|
| 0 | 2314 | log_10_2026-4-15-11-37-24; log_1_2026-4-12-16-33-22; log_20_2026-4-15-13-12-34; log_31_2026-4-16-18-41-26 | 04-12, 04-15, 04-16 |
| 1 | 2414 | log_1_2026-4-14-12-01-42; log_26_2026-4-16-10-09-34; log_5_2026-4-15-10-30-38; log_6_2026-4-15-10-48-46 | 04-14, 04-15, 04-16 |
| 2 | 2489 | log_0_2026-4-14-11-50-14; log_25_2026-4-16-09-57-28; log_32_2026-4-16-18-53-24; log_8_2026-4-15-11-23-44 | 04-14, 04-15, 04-16 |
| 3 | 2472 | log_17_2026-4-15-12-47-22; log_24_2026-4-16-09-49-52; log_27_2026-4-16-10-17-56; log_3_2026-4-14-12-19-22 | 04-14, 04-15, 04-16 |
| 4 | 2479 | log_0_2026-4-14-10-21-28; log_12_2026-4-15-11-57-08; log_30_2026-4-16-10-36-20; log_38_2026-4-16-19-37-00 | 04-14, 04-15, 04-16 |

## 6. Mean branch results

Stage A 产生 200 条 initial 与 18 条 weighting-refinement mean records。

| Component | Role | Retention | Condition | Ridge | Weight | Macro RMSE (N) | Coefficients |
|---|---|---:|---|---:|---|---:|---:|
| Fx | best | 0.00 | alpha_frequency | 1e-6 | equal_cycle | 0.4844 | 3 |
| Fx | one-SE simplest | 0.00 | frequency | 1 | equal_cycle | 0.5196 | 2 |
| Fx | best prior-retaining | 0.25 | frequency | 1e-6 | equal_cycle | 0.5252 | 2 |
| Fz | best | 0.00 | frequency | 1e-6 | equal_cycle | 0.7597 | 2 |
| Fz | one-SE simplest | 0.00 | none | 100 | equal_log | 0.7655 | 1 |
| Fz | best prior-retaining | 0.25 | alpha_frequency | 1e-6 | equal_log | 0.7727 | 3 |

Fx mean 的 alpha+frequency 最低，但 frequency-only 在 one-SE 内，用少一个
condition coefficient。Fz mean 的 intercept-only 模型与最低值仅差 0.0058 N，
说明在当前 envelope 内，trim-relevant mean correction 主要表现为稳定 offset，
没有足够证据保留 condition dependence。

## 7. Waveform branch results

Stage A 产生 800 条 initial 与 15 条 weighting-refinement waveform records。

| Component | Role | Retention | K | Condition | Ridge | Weight | Macro RMSE (N) | Coefficients |
|---|---|---:|---:|---|---:|---|---:|---:|
| Fx | best | 0.00 | 4 | alpha_frequency | 0.01 | equal_cycle | 0.8171 | 24 |
| Fx | one-SE simplest | 0.00 | 4 | frequency | 0.01 | equal_cycle | 0.8271 | 16 |
| Fx | best prior-retaining | 0.25 | 4 | alpha_frequency | 0.01 | equal_cycle | 0.8362 | 24 |
| Fz | best | 0.00 | 4 | frequency | 0.01 | equal_log | 1.4287 | 16 |
| Fz | one-SE simplest | 0.00 | 4 | frequency | 0.01 | equal_cycle | 1.4322 | 16 |
| Fz | best prior-retaining | 0.25 | 4 | alpha_frequency | 0.01 | equal_log | 1.5130 | 24 |

两分量都需要 K=4 才进入 branch shortlist；这与 EDA0 的 phase-localized discrepancy
一致。Frequency interaction 保留下来；AoA interaction 的增益不足以抵消额外 8 个
waveform coefficients。没有增加 K>4，也没有根据 validation 扩大空间。

## 8. Weighting refinement 与完整 candidates

正式结果共 218 mean records、815 waveform records，以及 26 complete candidates
（Fx/Fz 各 13）。Complete composition 包含 raw prior、gain-bias、no-prior、
fixed-prior 与 shaped-prior。Physical component-scale 保持 `unavailable`，因为
active prior 缺少可按 stable key 对齐的 row-level authoritative component artifact。

Selected Fx 的 train-CV complete macro total/mean/waveform RMSE 分别为
1.0110/0.5196/0.8271 N。Selected Fz 分别为 1.6609/0.7655/1.4322 N。

进入 validation 的 Fx 为 5 selectable + raw，Fz 为 6 selectable + raw。
Raw baseline 不占 selectable limit。Shortlist 同时保留了 matched-capacity
no/fixed/shaped comparison。

## 9. Validation leaderboard

完整小型 CSV 见
`docs/analysis/results/2026-07-27_static_correction_validation_leaderboard.csv`。

### Fx

| Rank | Candidate | Type | Total macro | Mean macro | WB macro | Worst log | Coeff. |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | complete_96f7b37445d7 | no-prior | 0.9928 | 0.4590 | 0.8650 | 1.2048 | 27 |
| 2 | complete_7ecd182bd659 | shaped | 1.0062 | 0.4590 | 0.8794 | 1.2197 | 27 |
| 3 | complete_7809de4d75fb | no-prior, selected | 1.0375 | 0.5249 | 0.8789 | 1.3294 | 18 |
| 4 | complete_a97786b91877 | fixed | 1.2110 | 0.6716 | 0.9830 | 1.4475 | 27 |
| 5 | complete_2fbd66b2a155 | gain-bias | 3.3463 | 0.6576 | 3.2549 | 3.7130 | 2 |
| 6 | complete_aebd9a0e75ed | raw prior | 3.8540 | 1.3317 | 3.5893 | 4.1898 | 0 |

### Fz

| Rank | Candidate | Type | Total macro | Mean macro | WB macro | Worst log | Coeff. |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | complete_c5a65afb9957 | shaped | 1.8465 | 0.7895 | 1.6326 | 2.1655 | 19 |
| 2 | complete_61ead6598f5c | no-prior, selected | 1.8665 | 0.8372 | 1.6285 | 2.1829 | 17 |
| 3 | complete_b7cabb1110b0 | no-prior | 1.8672 | 0.8290 | 1.6326 | 2.2024 | 18 |
| 4 | complete_3bd562d21e06 | shaped | 2.3716 | 1.2882 | 1.9133 | 2.8208 | 18 |
| 5 | complete_049b684bdff9 | gain-bias | 3.3341 | 1.3712 | 2.8749 | 3.6777 | 2 |
| 6 | complete_a25b38b82d75 | fixed | 3.6258 | 2.5584 | 2.4296 | 4.3988 | 18 |
| 7 | complete_c5a74dce2f0e | raw prior | 8.4880 | 5.1027 | 6.6723 | 9.2847 | 0 |

## 10. Fx selection 与 one-SE reason

Validation minimum 0.992762 N 的 per-log SE 为 0.058268 N，one-SE threshold
为 1.051031 N。`complete_7809de4d75fb` 的 1.037476 N 位于该范围内。它将
mean/WB conditions 从 alpha+frequency 简化为 frequency-only，并把 learned
coefficient count 从 27 降到 18。

Fx secondary metrics 为 waveform macro RMSE=0.8789 N、worst-log total
RMSE=1.3294 N、phase-bin waveform RMSE=0.3581 N、peak magnitude
error=0.6475 N、circular peak-phase error=0.0690 rad。相对 validation minimum，
主要代价来自 mean RMSE 增加 0.0659 N，而 waveform 差异只有 0.0138 N。按照冻结
complexity ordering，这不足以保留额外 AoA coefficients。

最终 Fx spec：

```text
no_prior_mean_wb
a_mean=0, a_waveform=0
mean_condition=frequency
waveform_condition=frequency
K=4
ridge_mean=1
ridge_waveform=0.01
mean_weighting=equal_cycle
waveform_weighting=equal_cycle
```

## 11. Fz selection 与 trim 解释

Validation minimum 1.846483 N 的 per-log SE 为 0.118523 N，one-SE threshold
为 1.965006 N。`complete_61ead6598f5c` 的 1.866459 N 位于该范围内。

Selected Fz cycle-mean macro RMSE=0.8372 N，较 absolute best 的 0.7895 N 高
0.0477 N，但仍低于 threshold 内另一 no-prior candidate 的复杂结构需求。它不是
忽略 mean bias：mean branch 明确保留一个不惩罚 intercept，train-fit coefficient
为 -8.4168 N；validation mean bias=0.0586 N。该 intercept-only branch 是对当前
trim-relevant mean discrepancy 的最简稳定描述。

Selected Fz waveform macro RMSE=1.6285 N、downstroke integral error
absolute mean=0.0556 N rad、worst-log total RMSE=2.1829 N。最终 spec：

```text
no_prior_mean_wb
a_mean=0, a_waveform=0
mean_condition=none
waveform_condition=frequency
K=4
ridge_mean=100 (intercept unpenalized)
ridge_waveform=0.01
mean_weighting=equal_log
waveform_weighting=equal_cycle
```

## 12. Selection stability

Fx 与 Fz 在 5 次 leave-one-validation-log-out 中都 5/5 选择正式模型，且每次只
重新聚合剩余 4 logs，不重新训练。

| Component | All-5 selected | LOO selected frequency | Selected primary range (N) | Single-log dominated |
|---|---|---:|---|---|
| Fx | complete_7809de4d75fb | 5/5 | 0.9645–1.0741 | no |
| Fz | complete_61ead6598f5c | 5/5 | 1.7874–1.9616 | no |

因此没有 selection uncertainty，不需要也不允许扩大 validation search。

## 13. Matched-capacity prior incremental value

小型证据 CSV 见
`docs/analysis/results/2026-07-27_static_correction_prior_value.csv`。

- Fx shaped retention `(0, 0.25)` 相对 matched no-prior 的 macro gain=-1.35%，
  0/5 logs 改善；fixed retention `(1,1)` gain=-21.98%，0/5 改善。
- Fz shaped retention `(0.5,0.5)` gain=-27.02%，0/5 改善；fixed retention
  `(1,1)` gain=-94.18%，0/5 改善。

因此 Fx/Fz verdict 均为：

```text
No stable incremental predictive value demonstrated
```

DeLaurier prior 仍保留 physics baseline、error-attribution structure 与
out-of-envelope reference 的角色；本结果不支持声称它提高了当前 flight envelope
内的预测精度。

## 14. Correction magnitude 与 numerical diagnostics

Selected Fx train OOF correction RMS=3.6069 N、peak=15.6165 N、
correction/prior RMS ratio=0.5848。Selected Fz 分别为 8.5290 N、35.5932 N、
0.4379。

最终 train-fitted bundles 均为 full-rank、finite：

| Component | Mean condition number | WB condition number | Mean/WB coefficients | Rank deficient |
|---|---:|---:|---:|---|
| Fx | 1.0000 | 1.1003 | 2 / 16 | no |
| Fz | 1.0000 | 1.1003 | 1 / 16 | no |

Bundle save/load 后 validation replay 最大差异为 Fx `1.78e-15` N、Fz
`3.55e-15` N。Total=`mean+waveform` 最大误差 `1.78e-15` N；逐 cycle waveform
mean 最大值分别为 `6.95e-16` N 与 `1.14e-15` N。

## 15. Validation envelope 与 per-log consistency

Selected features 只包含 AoA/frequency contract 中的 frequency。3192 个
validation cycles 中，AoA/frequency 超出 train envelope 的 cycle 数为 0，因此
all-validation 与 in-envelope-only macro RMSE 完全相同。未删除、clip 或重训任何
validation cycle。

Selected Fx 的 5-log RMSE 为 0.8911、0.9270、1.0408、0.9991、1.3294 N。
Selected Fz 为 1.9958、1.8406、1.4859、2.1829、1.8272 N。日期方向未显示由
单一日期或单一 log 决定选择；LOO 结果提供了更严格的证据。

## 16. Bundles、residuals 与 figures

正式 train-fitted bundles：

```text
artifacts/models/static_correction_selected_train/fx
artifacts/models/static_correction_selected_train/fz
```

状态均为 `selected_static_train_only`，training provenance 只含 train；
`test_labels_loaded=false`、`dynamic_audit_pending=true`。Bundle hashes：

- Fx：`24cf5324f2575c583571e4011dca3e4e78801e395bb55c7d04940bb223ac2f23`；
- Fz：`4b581a1f952a74bf727903e934a36335f4eaa728b7478f1bb13502705807eadc`。

C4 residual artifacts 位于最终 output root：

```text
validation_predictions_fx.parquet
validation_predictions_fz.parquet
validation_residuals_fx.parquet
validation_residuals_fz.parquet
```

Residual identity `residual=label-prediction` 的独立 audit 最大误差为 0。所有
residual rows 的 partition 都是 validation。

`figures/` 下生成了合同要求的 14 张 headless figures，覆盖 train-CV/validation
leaderboard、per-log paired RMSE、mean-vs-waveform、K/condition/retention
sensitivity、prior-family phase curves、Fz downstroke、correction amplitude、
rank consistency 与 LOO stability。

## 17. Quality checks 与测试

`quality_checks.json` 的 27 项 strict checks 全部通过，`strict_failures=[]`。
正式运行后的独立 audit 重新验证：

- shortlist seal/config/artifact hash；
- train 与 validation partition 集合；
- selected bundle train-only provenance；
- bundle replay、total decomposition、waveform zero mean；
- residual identity；
- C1 input hashes；
- LOO metric range；
- test 未加载。

实现阶段指定六个 C3 聚焦模块为 50 passed；随后三个 fail-closed 修复分别把最终
full suite 提升到 502/503/503 passed。Commit B 前最终验证将再次运行聚焦、相关
回归与 full suite。Ruff 不在 `flap-train-gpu` 环境中，未安装或改变依赖。

## 18. Limitations 与 C4 readiness

当前 target 尚未扣除 tail/body；selected structure 只在当前 ratio=8、20/5
whole-log train/validation envelope 内选择；test 仍锁定；selected bundle 不是
production/final/approved-for-simulation。Physical component-scale 因 authoritative
row-level component 缺失而 unavailable。

本阶段没有训练 dynamic residual 或 TCN，没有分析 history/future features，也没有
修改 tail、moment、controller、label、split、canonical dataset、authoritative prior、
DeLaurier physics 或 IsaacLab production physics。

Stage A、sealed shortlist、Stage B、one-SE、LOO stability、prior verdict、
train-only bundle 与 validation residual gate 均已通过，因此：

```text
READY FOR C4
```
