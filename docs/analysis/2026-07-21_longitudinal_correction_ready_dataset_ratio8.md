# C0/C1 — RATIO=8 longitudinal correction-ready dataset

日期：2026-07-21

状态：`READY WITH LIMITATIONS`（来自 `quality_checks.json`，strict failures=0）

## 1. 执行摘要

本次使用 active authoritative prior `delaurier_attitude_aware_3b5d4ec_ratio8_phasefix_trainval_v4` 和 canonical dataset `canonical_v0.4_training_ready_split_measured_massprops_ratio8_phasefix_v3`，仅加载 train/validation。Test label 未加载。`log_id + timestamp_us` 共匹配 388,289 rows，无 duplicate、missing 或 orphan key。

C1 共检查 15,711 cycles，接受 15,360、拒绝 351；waveform table 含 382,297 rows。Mean/WB decomposition、逐 cycle zero mean、K=1..4 centered basis、train-only normalization、partition-aware weights、deterministic rebuild 与输入 immutability 全部通过。Artifact：

```text
artifacts/correction_ready/longitudinal_mean_wb_ratio8_20260721T140238Z_09b4bb6
```

进入 C2 的 gate 为 `READY WITH LIMITATIONS`：validation 有 2 个 airspeed cycle 和 1 个 dynamic-pressure cycle 超出 training range；此外负 Pitot 值要求第一轮 C2 不把 airspeed/q 作为 condition。AoA 与 flapping frequency 均未超出 training envelope。

## 2. Correction contract

完整 contract 见 [`docs/contracts/longitudinal_force_correction_contract.md`](../contracts/longitudinal_force_correction_contract.md)。当前 target 是 provisional whole-aircraft effective longitudinal force，不是纯 wing aerodynamic target；tail replay 尚未实施，body force 也未扣除。最终 tail-subtracted target 留到 C6/C7。

## 3. Prior 与 label provenance

- Dataset：`canonical_v0.4_training_ready_split_measured_massprops_ratio8_phasefix_v3`；manifest SHA-256=`aa12aa66f762390ab1a356b94916694f5ed9689af670f313544aeb57a250cc07`。
- Ratio contract：8.0，`ratio8_v1`，source=`confirmed_physical_hardware`。
- Phase/frequency：`hall_indexed_mechanical_phase_ratio8_v1` / `flap_frequency_ratio8_v1`；唯一相位列为 `mechanical_phase_rad`。
- Label：effective non-gravity external force，body FRD，N，`+Fx` forward、`+Fz` down；SG 0.03 s，polyorder 3；mass=0.90415 kg measured contract。
- Split：原 whole-log train 20 logs、validation 5 logs、test 4 logs 未改变。
- Prior：`delaurier_attitude_aware_3b5d4ec_ratio8_phasefix_trainval_v4`；manifest SHA-256=`c86b34ea10328207b1867b117d44656eacf54b751e883a1ee99de0656695200c`。
- Physics：IsaacLab commit `3b5d4ec1d28f1384cf042402992ad7ea59995f49`；body FRD，attitude-aware 3D airflow，attached flow，dynamic twist disabled，theta tip 0 deg。Local IsaacLab checkout dirty status 已如实记录，但未参与修改 frozen physics 数值。

## 4. Alignment

Train 308,702 rows、validation 79,587 rows均以 `log_id + timestamp_us` one-to-one 对齐；missing label=0、missing prior=0、duplicates=0、row-order dependency=false。Test rows loaded=0。证据见 `alignment_report.json` 与 `alignment_mismatches.csv`。

## 5. Cycle quality

| Partition | Candidate | Accepted | Rejected |
|---|---:|---:|---:|
| train | 12,479 | 12,168 | 311 |
| validation | 3,232 | 3,192 | 40 |
| 合计 | 15,711 | 15,360 | 351 |

拒绝原因按展开计数为 incomplete coverage 185、maximum gap 184、non-finite state 168、too few samples 113、non-monotonic timestamp 18、`phase_unwrap_not_monotonic` 1。一个 cycle 可有多个原因，因此原因数之和大于 rejected cycle 数。Accepted sample count 范围为 19–43，phase coverage 5.501–6.279 rad，duration-derived frequency 2.381–5.556 Hz。没有静默丢弃 cycle。

## 6. Mean/WB decomposition

最大 reconstruction error：label=0、prior=0、residual=7.106e-15 N。逐 cycle 最大 absolute zero mean：label=2.008e-15 N、prior=3.438e-15 N、residual=4.382e-15 N。Centered sine/cosine basis 的最大逐 cycle mean 为 5.353e-17。

| Quantity | Fx range (N) | Fz range (N) |
|---|---:|---:|
| label cycle mean | [-0.087, 26.670] | [-14.003, -3.748] |
| prior cycle mean | [0.826, 7.649] | [-24.509, -0.494] |
| label waveform | [-19.626, 29.641] | [-23.039, 32.825] |
| prior waveform | [-6.743, 13.351] | [-37.334, 35.398] |
| residual mean | [-4.188, 23.696] | [-7.969, 15.623] |
| residual waveform | [-20.769, 32.790] | [-21.040, 20.874] |

Residual mean 的 pooled mean/std 为 Fx -1.137/0.789 N、Fz 4.697/2.736 N；waveform pooled std 为 Fx 3.485 N、Fz 6.671 N。本节只描述数据，不重新扩展 EDA0 的因果结论。

## 7. Condition coverage

| Condition | Train min / max / mean | Validation min / max / mean | Validation outside train |
|---|---|---|---:|
| AoA (rad) | 0.017 / 0.738 / 0.357 | 0.150 / 0.568 / 0.353 | 0 |
| frequency (Hz) | 2.268 / 5.078 / 4.052 | 2.576 / 4.866 / 4.077 | 0 |
| airspeed (m/s) | -2.221 / 12.047 / 7.915 | -2.808 / 12.247 / 7.912 | 2 |
| density (kg/m3) | 1.110 / 1.154 / 1.138 | 1.123 / 1.151 / 1.137 | 0 |
| dynamic pressure (Pa) | 5.376 / 82.990 / 37.421 | 5.806 / 86.529 / 37.815 | 1 |

Sample-level negative airspeed fraction=0.6207%，minimum=-10.158 m/s。原值未做 abs、clip 或替换；15,063/15,360 cycles 的 airspeed/dynamic-pressure validity 为 true。C2 第一轮 condition 只考虑 AoA 与 flapping frequency。

## 8. Weight 与 normalization

Normalization 仅由 12,168 个 train cycles 拟合；validation 全部使用冻结 train statistics。五个 condition 的 `source_partition` 均为 `train`，无 zero-variance feature。Equal-cycle、equal-log、equal-date sample totals 分别在每 cycle、每 `(partition, log)`、每 `(partition, date)` 内精确为 1.0；validation cycle 数变化不会改变 train weights。Label uncertainty keyed artifact 不可用，因此未伪造 uncertainty weight。

## 9. Artifact schema

`cycle_table.parquet` 每行一个 accepted cycle，用于后续 mean branch、condition coverage 与 cycle weighting；`waveform_table.parquet` 每行一个 accepted sample，用于 zero-mean WB branch、K1–K4 centered Fourier basis 和 sample weighting。完整 schema、normalization、weight contract、alignment、cycle quality 与 checks 均位于 artifact 目录。

## 10. Limitations

当前 target 未扣除 tail，body 未单独建模；test 未使用；moment 不在范围；只保留完整周期；cycle condition 使用慢变量近似；online virtual-cycle prior mean 尚未实现；negative Pitot validity 尚未形成独立研究；本阶段没有训练任何 correction model。

## 11. C2 readiness

程序化状态：`READY WITH LIMITATIONS`。所有 strict checks 通过；非阻塞限制仅为少量 validation condition 超出 train range与负 Pitot validity。结合本次完整测试通过后，项目级状态记为 `READY FOR C2 WITH NON-BLOCKING LIMITATIONS`，但这不代表已选择或训练任何 C2 candidate。

## 12. 复现信息

- Branch：`fix/ratio8-phase-regeneration`
- Commit A：`09b4bb6e497a677b5da871e3b600f4ebd6e1dd39`
- Artifact ID：`longitudinal_mean_wb_ratio8_20260721T140238Z_09b4bb6`
- Build time：38.30 s
- Run command：见 artifact `run_command.txt`
- Tests：相关聚焦回归 170 passed；最终 full suite 380 passed（67.55 s）；`git diff --check` 与 Python compile 通过；Ruff 未安装，未改变环境。
