# Ratio 7.5 与 Ratio 8 phase/frequency 影响对比

日期：2026-07-21

本报告比较历史 ratio=7.5 downstream contract 与硬件确认 ratio=8.0 contract。新结果使用 `mechanical_phase_rad`；旧 EDA0 实际使用 `phase_corrected_rad`。比较以 `log_id + timestamp_us` keyed join 完成，不按行序拼接。

## 1. 总表

| 指标 | Ratio 7.5 | Ratio 8 | 变化 | 解释 |
|---|---:|---:|---:|---|
| dataset rows | 448,960 | 448,960 | 0 | 全部 key 匹配，Fx/Fz label 最大差异 0 N |
| C1 accepted cycles | 6,490 | 15,360 | +8,870 | accepted fraction 从 41.3% 增至 97.8% |
| C1 rejected cycles | 9,210 | 351 | -8,859 | gate 未放宽 |
| phase monotonic rejection | 9,026 | 1 | -9,025 | ratio/Hall phase 修复直接消除了主要 rejection |
| mean flapping frequency (train+val) | 4.2853 Hz | 4.0175 Hz | -0.2678 Hz (-6.25%) | 各 ULog `FLAP_RATIO/8.0`；本批日志 median ratio=0.9375 |
| Fx phase offset | -0.6859 rad | -0.5539 rad | +0.1320 rad | delay -24.420 ms → -21.675 ms |
| Fz phase offset | -0.2428 rad | -0.1590 rad | +0.0838 rad | delay -8.726 ms → -6.382 ms |
| Fx mean/WB energy | 10.7% / 89.3% | 14.0% / 86.0% | mean +3.4 pp | phase branch 仍占主导 |
| Fz mean/WB energy | 22.0% / 78.0% | 33.5% / 66.5% | mean +11.5 pp | mean 与 phase branch 均保留 |
| Fz downstroke residual integral | 19.091 N rad | 15.148 N rad | -3.943 N rad | 仍为 downstroke，跨 log 同号率仍 100% |
| Fx main component correlation | dD_f, -0.701 | dD_f, -0.679 | abs -0.022 | 候选 component identity 不变 |
| Fz main component correlation | dN_c, -0.759 | dN_c, -0.799 | abs +0.040 | 候选 component identity 不变 |
| phase-conditioned gain Fx/Fz | 16.2% / 9.7% | 23.3% / 9.5% | +7.1 / -0.2 pp | AoA+frequency 仍为建议 condition |
| history probe gain Fx/Fz | 9.6% / 9.3% | 9.7% / 15.8% | +0.1 / +6.5 pp | 没有因 phase 修复而消失；TCN verdict 仍为 no |
| matched-capacity prior gain Fx/Fz | -16.7% / -81.9% | -15.5% / -78.3% | +1.2 / +3.6 pp | prior 略改善但仍无稳定 incremental value |
| recommended C2 structure | Fx phase；Fz mean+phase；AoA+f | 同左 | 主结构不变 | 新数据覆盖显著改善；不得沿用旧数值 |

## 2. Phase 与 frequency 的实际变化

448,960 rows 全部 keyed 匹配，row inclusion/exclusion 为零。新旧消费相位的 circular absolute difference 中位数为 0.1673 rad（9.59 deg），p95=0.3988 rad（22.85 deg），最大 1.9331 rad；因此相位影响不是简单常数 6.67%。旧/new cycle boundary 数为 18,071/18,084，只有 2,460 个 timestamp 完全相同；旧 boundary 到最近新 boundary 的距离中位数 10 ms、p95 20 ms。它反映 Hall re-anchoring、错误 ratio 累积与旧 per-cycle phase correction 的共同作用。

Frequency 的变化更接近解析比例：train、validation、test 的 median `f_new/f_old` 均为 0.9375，因为实际 ULog 参数为 7.5、硬件 ratio 为 8.0。Train+validation mean 从 4.2853 降为 4.0175 Hz。这里没有假设固定比例，而是逐日志读取 ULog `FLAP_RATIO` 后由实际数据验证。

## 3. 哪些旧结论保留，哪些改变

保留：Fx 以 phase/WB 为主；Fz 同时需要 mean 与 phase branch；AoA 与 flapping frequency 仍是首轮 condition；downstroke Fz residual 与 dD_f/dN_c component association 保持；全局 phase offset/fixed-delay repair 均未达到先改输入的 verdict；TCN 仍不必要；matched-capacity prior 仍未显示正 incremental value。

显著改变：完整周期覆盖从 41.3% 提升到 97.8%，旧 EDA0 的 6,553 accepted cycles 增至 15,526；Fx/Fz offset 与 delay 均缩小；Fz mean energy 提高 11.5 pp；Fz history gain 从 9.3% 提高到 15.8%，而非下降。旧的具体 offset、energy、half-stroke integral、gain 与 component correlation 均被 supersede，不能用于 C2 数值依据。

## 4. Prior、condition 与 history 解释

正确 phase 后 prior matched-capacity gain只从 Fx/Fz -16.7%/-81.9% 改到 -15.5%/-78.3%，方向仍为负，所以不能宣称 prior 因修复而获得更高结构价值。Phase-conditioned correction 仍必要：Fx joint gain提高到 23.3%，Fz 保持 9.5%。History gain 没有按“旧模型只是在补偿错误 phase”的假设下降；尤其 Fz 升至 15.8%。这只支持后续独立 dynamic audit，不授权本阶段训练 dynamic model 或 TCN。

## 5. Phase gate 与 C1 readiness

`phase_unwrap_not_monotonic` 从 9,026 降为 1，且未改变 `-1e-6` monotonicity threshold。因此保持现有 gate，不另开 tolerance 调整。新 C1 strict checks 全部通过，test labels 未加载，normalization 仅来自 train，partition-aware weights 全部归一为 1。状态为：

```text
READY FOR C2 WITH NON-BLOCKING LIMITATIONS
```

限制是负 Pitot 与少量 validation envelope exceedance；C2 第一轮 condition 仅考虑 AoA 与 flapping frequency。

## 6. Provenance 与复现

- Commit A：`09b4bb6e497a677b5da871e3b600f4ebd6e1dd39`
- Old EDA0：`outputs/force_discrepancy_attribution/20260717T054923Z_5b72620`
- New EDA0：`outputs/force_discrepancy_attribution/20260721T135633Z_09b4bb6`
- New C1：`artifacts/correction_ready/longitudinal_mean_wb_ratio8_20260721T140238Z_09b4bb6`
- Keyed dataset comparison：`outputs/ratio_phase_impact/20260721T140609Z_09b4bb6`
- Dataset comparison 包括完整 split 的 identity/phase/frequency/label immutability；EDA0 与 C1 只打开 train/validation，test rows loaded=0。
- 两次独立 ratio8 materialization 在排除刻意不同的 immutable `dataset_id` 后，三 partition 的 schema/values 全等；两次 prior 的 train/validation keys 与 Fx/Fz predictions bitwise equal。

本任务没有训练 correction model，没有选择最终 K，没有修改 tail、moment、controller、label、split 或 IsaacLab production physics。
