# C0/C1 — 纵向力修正 contract 与 correction-ready dataset

日期：2026-07-17

状态：C0/C1 data preparation complete；未训练 correction model

C2 readiness：`READY WITH LIMITATIONS`（由 `quality_checks.json` 程序化生成）

## 1. 执行摘要

本次通过 prior registry 解析 active authoritative prior `delaurier_attitude_aware_3b5d4ec_trainval_v1`，与 dataset `canonical_v0.2_training_ready_split_measured_massprops_ratio8_sg0p03_v1` 的 train/validation reconstructed effective force label 做 `log_id + timestamp_us` keyed alignment。Test identity 只从 split manifest chain 汇总，test label Parquet 未打开；artifact 明确记录 `test_labels_loaded: false`。

总计检查 15,700 个 canonical cycles，接受 6,490 个、拒绝 9,210 个；accepted waveform table 含 141,436 rows。Mean/WB decomposition、每 cycle zero-mean、K1–K4 centered Fourier basis、train-only normalization、cycle/log/date weights、deterministic rebuild 和输入 hash before/after checks 全部通过。正式 artifact 为：

```text
/home/zn/flap-system-identification/artifacts/correction_ready/longitudinal_mean_wb_20260717T072834Z_153cc9f
```

正式 CLI wall time 为 32.64 s；manifest 内 package build duration 为 31.10 s。Data gate 无 strict failure，可以进入 C2 数据消费阶段。Readiness 保留两项程序化 limitation：upstream prior manifest 未记录 physics source dirty status；少量 validation cycles 超出 train condition min/max。

## 2. Correction contract

完整 active contract 见 `docs/contracts/longitudinal_force_correction_contract.md`。C0/C1 target 是

\[
\mathbf y_t^{(0)}=\mathbf F_{\mathrm{rec},t}^{\mathrm{effective}},
\]

prior 是 wing-only DeLaurier force。当前 residual 可包含 wing discrepancy、tail/body force、interaction、disturbance、asymmetry 和 reconstruction error，不能称为纯 wing aerodynamic error。

后续 C6/C7 只有在 task-oriented tail 参数冻结并完成 keyed tail replay 后，才允许构建

\[
\mathbf y_t^{\mathrm{final}}
=\mathbf F_{\mathrm{rec},t}^{\mathrm{effective}}
-\mathbf F_{\mathrm{tail},t}^{\mathrm{frozen}}.
\]

本阶段没有生成 tail-subtracted target，没有实现或调整 tail。

## 3. Prior 与 label provenance

Prior provenance：

- prior ID：`delaurier_attitude_aware_3b5d4ec_trainval_v1`；lifecycle：`active`；
- physics repository：`https://github.com/dAnte-9029/IsaacLab`；
- physics commit：`3b5d4ec1d28f1384cf042402992ad7ea59995f49`；
- physics dirty：`not_recorded_in_prior_manifest`，本报告不推断 clean；
- frame：`body_frd_force_at_imu_origin_moment_about_cg`；
- airflow：`attitude_ground_wind_3d`；phase：`canonical_mechanical_phase_to_delaurier_v1`；
- attached flow，separation disabled；prescribed dynamic twist disabled，zero tip amplitude；
- train/validation prediction SHA-256 与 prior manifest 一致；production prediction input hash 前后不变。

Label provenance：

- dataset ID：`canonical_v0.2_training_ready_split_measured_massprops_ratio8_sg0p03_v1`；
- label policy：effective wrench recomputed from pre-smoothed kinematic derivatives；force derivative 为 Savitzky–Golay，window 0.03 s，polyorder 3；
- mass metadata：`aircraft_metadata_v0.1`，mass status 为 `measured_combined_neutral_wings`；
- body/local frame：FRD/NED；force units 为 N；\(+F_x\) forward、\(+F_z\) down；
- attitude quaternion：PX4 `wxyz`，body FRD 到 NED；
- split identity 与 dataset ID 相同，whole-log reused assignment lineage 由 source manifest chain 保留；
- partitions：train 与 validation；test labels 未加载。

## 4. Alignment

Alignment key 固定为 `log_id + timestamp_us`。实现复用 Phase-0 package keyed join，未按 row order、DataFrame index 或 equal length 配对。

| Partition | Label rows | Prior rows | Matched | Missing prior | Orphan prior | Duplicate keys |
|---|---:|---:|---:|---:|---:|---:|
| train | 308,702 | 308,702 | 308,702 | 0 | 0 | 0 |
| validation | 79,587 | 79,587 | 79,587 | 0 | 0 | 0 |
| total | 388,289 | 388,289 | 388,289 | 0 | 0 | 0 |

`alignment_mismatches.csv` 只有 header；没有 rejected logs。Timestamp integer-microsecond 与 magnitude checks、partition identity、prior/dataset hash 和 frame/phase/airflow contract 均通过。

## 5. Cycle quality

Train 检查 12,470 cycles，接受 5,047、拒绝 7,423；validation 检查 3,230 cycles，接受 1,443、拒绝 1,787。25 个 logs 均有 accepted cycles；每 log accepted count 范围为 199–336，中位数 243。

Accepted coverage：

| Partition | Cycles | Sample count | Phase coverage (rad) | Maximum gap max (rad) | Frequency from duration (Hz) |
|---|---:|---:|---:|---:|---:|
| train | 5,047 | 17–31，median 21 | 5.514–6.283 | 0.769 | 3.125–5.556 |
| validation | 1,443 | 18–32，median 21 | 5.532–6.283 | 0.751 | 2.941–5.556 |

Date accepted coverage 为 2026-04-12: 550、04-14: 1,284、04-15: 2,492、04-16: 2,164 cycles。

Rejection reason counts 如下；一个 cycle 可以有多个 reason，因此各 reason count 之和可大于 rejected cycle count：

- `phase_unwrap_not_monotonic`: 9,026；
- `maximum_phase_gap_exceeded`: 169；
- `non_finite_label_prior_or_state`: 168；
- `incomplete_phase_coverage`: 162；
- `too_few_samples`: 102；
- `non_monotonic_timestamp`: 15。

主要 rejection 仍来自 canonical corrected-phase 在 source cycle 内不满足 monotonic unwrap。C1 没有修改 phase convention 或修复输入；所有 rejection 与 endpoint removal count 均保存在 cycle quality tables。

## 6. Mean/WB decomposition

最大 reconstruction error 为：label 0、prior 0、residual \(7.105\times10^{-15}\) N。每 cycle waveform 最大绝对均值为：label \(1.881\times10^{-15}\) N、prior \(3.553\times10^{-15}\) N、residual \(4.314\times10^{-15}\) N。K1–K4 centered sine/cosine basis 最大绝对 cycle mean 为 \(5.815\times10^{-17}\)。均低于 contract tolerance \(10^{-10}\)。

数据范围：

| Quantity | Fx range (N) | Fz range (N) |
|---|---:|---:|
| label cycle mean | 1.044 to 6.569 | -14.600 to -5.151 |
| prior cycle mean | 2.056 to 8.410 | -25.613 to -4.810 |
| mean residual | -4.372 to 1.478 | -4.800 to 15.725 |
| label waveform | -8.647 to 15.163 | -20.030 to 23.004 |
| prior waveform | -7.546 to 14.706 | -37.539 to 37.081 |
| WB residual | -16.283 to 14.662 | -24.357 to 23.704 |

这些是 C1 描述性范围，没有重新执行 EDA0 的因果归因，也没有据此选择 correction structure。

## 7. Condition coverage

| Condition | Train min–max | Validation min–max | Validation rows outside train range |
|---|---:|---:|---:|
| AoA (rad) | 0.066–0.742 | 0.158–0.556 | 0 |
| flapping frequency (Hz) | 3.255–5.421 | 3.019–5.181 | 3 |
| airspeed (m/s) | -1.389–11.947 | -3.223–12.406 | 2 |
| density (kg/m³) | 1.110–1.153 | 1.123–1.151 | 0 |
| dynamic pressure (Pa) | 3.536–81.841 | 8.014–88.780 | 1 |

AoA 使用 attitude、ground velocity 与 horizontal wind 重建 body-relative airflow；airspeed condition 保留 canonical validated true-airspeed column 的实际值。少量负 airspeed cycle mean 被原样保留并报告，没有在 C1 trim 或改 label。Validation 的 frequency、airspeed 和 dynamic-pressure extrapolation 已进入 programmatic readiness limitation；normalization 没有因此重拟合。

## 8. Weight 与 normalization

Normalization 的五个变量均只由 5,047 个 train cycle rows 拟合，`source_partition` 全为 `train`；validation 的任何值都没有进入 mean/std。所有 raw std 大于 \(10^{-8}\)，本次真实 artifact 未触发 zero-variance fallback；该 policy 已由 synthetic test 覆盖并可序列化重载。

Weights 只生成、不选择 strategy。数值验证结果：

- 每 cycle 的 `weight_equal_cycle_sample` 总和最大偏差：0；
- 每 log 的 `weight_equal_log_sample` 总和最大偏差：0；
- 每 date 的 `weight_equal_date_sample` 总和最大偏差：0；
- cycle table 每 log weight 总和最大偏差：\(1.11\times10^{-16}\)；
- cycle table 每 date weight 总和最大偏差：0。

没有可靠 keyed sample-level label uncertainty artifact，因此未伪造 uncertainty columns，manifest 标记 `not_available_no_reliable_keyed_sample_uncertainty_artifact`。

## 9. Artifact schema

`cycle_table.parquet` 为 6,490 rows × 33 columns，每行一个 accepted cycle，用于后续 mean branch、condition coverage 与 cycle-level weighting。关键字段包括 deterministic identity、partition/log/date、phase quality、condition means、label/prior means、mean residuals、train-derived standardized conditions 和 cycle/log/date weights。

`waveform_table.parquet` 为 141,436 rows × 65 columns，每行一个 accepted sample，用于后续 WB branch。它保留 phase/stroke identity、original label/prior、copied cycle means、zero-mean label/prior waveform、total/mean/WB residual、K1–K4 raw/centered Fourier basis、conditions、train-derived standardized conditions 和 sample weights。

两个 table 的 semantic SHA-256 分别为：

- cycle：`0e06e74317aa69e12e4155c4de46dc4f280efd8299987092818ab89985398881`；
- waveform：`f84592bb43a691e775b1bc9c07aea68d655b19f8d9e09e3b9f5a25230064401e`。

## 10. Limitations

1. Target 是 provisional effective force，尚未扣除 tail。
2. Body/fuselage force 未单独建模，未来 final target 中仍可能存在。
3. Test label 未使用；本报告不提供 test performance。
4. Moment 不在 C0/C1 范围。
5. 只保留完整且通过质量 gate 的 cycle；所有 rejected cycles 有显式 reason。
6. Cycle condition 使用慢变量 arithmetic-mean approximation。
7. Online virtual-cycle prior mean 尚未在 IsaacLab 实现。
8. Upstream prior manifest 提供 physics repository/commit，但没有提供 physics source dirty status。
9. 少量 validation conditions 超出 train min/max，后续 C2/C3 应保留 extrapolation 标记。
10. Canonical validated airspeed 出现少量负 cycle mean；C1 保留 provenance，不做 trim 或数据修复。
11. 尚未训练任何 correction，也未选择 K、condition feature 或 weighting strategy。

## 11. C2 readiness

```text
READY WITH LIMITATIONS
```

该状态来自 `quality_checks.json`：24 项 strict data checks 全部通过，strict failure count 为 0；limitations 是 `physics_source_dirty_status_not_recorded_upstream` 和 `some_validation_cycles_outside_training_condition_range`。Focused tests、相关回归与 full tests 同样通过，因此 C2 可以消费该 artifact；进入 C2 后仍不得把 provisional residual 当作纯 wing target。

## 12. 复现信息

- Branch：`feat/correction-contract-and-dataset`；
- artifact generation Git base：`153cc9f26badce2689e575f8010c48ed8d03f44d`，manifest 如实记录 build 时 worktree dirty；最终 C0/C1 implementation commit 见本文件所在 branch 的 Git history；
- artifact ID：`longitudinal_mean_wb_20260717T072834Z_153cc9f`；
- output root：`artifacts/correction_ready/longitudinal_mean_wb_20260717T072834Z_153cc9f`；
- key outputs：`manifest.json`、`quality_checks.json`、`cycle_table.parquet`、`waveform_table.parquet`、`normalization.json`；
- focused contract tests：`7 passed in 0.94s`；focused dataset tests：`10 passed in 4.54s`；
- related regression：`63 passed in 6.61s`；
- full tests：`368 passed in 65.17s`；
- Ruff：当前环境未安装，未改变环境；
- git diff check：通过。

复现命令：

```bash
python scripts/build_longitudinal_correction_ready_dataset.py \
  --dataset-root dataset/canonical_v0.2_training_ready_split_measured_massprops_ratio8_sg0p03_v1 \
  --split-manifest dataset/canonical_v0.2_training_ready_split_measured_massprops_ratio8_sg0p03_v1/dataset_manifest.json \
  --prior-registry configs/physics/delaurier_prior_registry.yaml \
  --prior-id delaurier_attitude_aware_3b5d4ec_trainval_v1 \
  --partitions train validation \
  --output-root artifacts/correction_ready
```

本阶段未开始 gain-bias、ridge、harmonic correction、shaped-prior、no-prior、dynamic residual 或 TCN training；production DeLaurier physics、tail、label 与 split 均未修改。
