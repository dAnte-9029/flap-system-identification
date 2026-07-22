# C2 — 静态纵向力 Mean/WB 修正候选模型族实现

日期：2026-07-22

状态：`READY FOR C3 WITH NON-BLOCKING LIMITATIONS`

Branch：`feat/static-mean-wb-correction`

起始基线：`bd658a6684bfa362614e4717ab0244a0156d734f`（包含 ratio=8 最终基线）

## 1. 执行摘要

C2 已实现统一 static correction specification、deterministic mean/WB features、固定 prior retention、stable weighted ridge、sample/cycle/log/date weighting、keyed prediction、immutable JSON bundle、candidate-space config、train-only C1 loader 与薄 smoke CLI。实现覆盖 raw prior、gain-bias、physical component scale、fixed-prior mean/WB、shaped-prior mean/WB 和 matched-capacity no-prior mean/WB。

Authoritative C1 artifact 的真实 train-only smoke 已生成 10 个代表性 Fx/Fz bundle。该 smoke 只证明 API、数值、serialization、weighting 和 provenance 可运行；没有遍历候选空间，没有使用 train metric 排名或选择模型。

## 2. 输入与隔离

- Correction-ready artifact：`artifacts/correction_ready/longitudinal_mean_wb_ratio8_20260721T140238Z_09b4bb6`；
- C1 manifest SHA-256：`5298c1c3ef8aabaa96fc439335883d333b4d9c60608e60c6a2194f779bfa0ad2`；
- Dataset：`canonical_v0.4_training_ready_split_measured_massprops_ratio8_phasefix_v3`；
- Dataset manifest SHA-256：`aa12aa66f762390ab1a356b94916694f5ed9689af670f313544aeb57a250cc07`；
- Prior：`delaurier_attitude_aware_3b5d4ec_ratio8_phasefix_trainval_v4`；
- Prior manifest SHA-256：`c86b34ea10328207b1867b117d44656eacf54b751e883a1ee99de0656695200c`；
- Ratio/phase/frequency：`ratio8_v1` / `hall_indexed_mechanical_phase_ratio8_v1` / `flap_frequency_ratio8_v1`；
- C2 fitting rows：12,168 train cycles、303,135 train waveform rows；
- Validation labels loaded：`false`；
- Test labels loaded：`false`。

Loader 在读取 fitting rows 前校验 C1 manifest hash、dataset/prior identity、active lifecycle、ratio=8、phase/frequency contract、C1 clean provenance、strict quality checks 与 test exclusion。返回的 bundle provenance 始终是 `included_partitions: ["train"]`。

## 3. 统一数学形式与候选族

Mean/WB candidate 使用

\[
\hat y_{j,t}=a_{j,\mu}\bar y^P_{j,c}+g_{\mathrm{mean},j}(\mathbf z_c)
+a_{j,w}y_{j,t}^{P\prime}+g_{\mathrm{WB},j}(\phi_t,\mathbf z_c),
\]

其中

\[
\mathbf z_c=[\widetilde\alpha_c,\widetilde f_c].
\]

实现的 candidate type 为 `raw_prior`、`gain_bias`、`physical_component_scale`、`fixed_prior_mean_wb`、`shaped_prior_mean_wb` 和 `no_prior_mean_wb`。Raw 与 gain-bias 保持独立 total-force 语义；fixed/shaped/no-prior 共用完全相同的 branch API。

## 4. Feature schema 与 retention

Mean feature 顺序为 `intercept`，随后按 condition set 加 `alpha_mean_std` 和/或 `flapping_frequency_mean_std`。WB feature 按 harmonic order 由低到高排列；每阶先 centered sine/cosine，再加 alpha interaction，最后加 frequency interaction。支持 (K=1,2,3,4) 和 `none`、`alpha`、`frequency`、`alpha_frequency`。

Retention 作为 specification 中的固定值，通过 target offset 实现：

\[
\bar y-a_\mu\bar y^P,\qquad y'-a_wy^{P\prime}.
\]

它不进入 ridge design，不会作为自由 coefficient 被拟合。Bundle 原样保存 retention，prediction 也不调整该值。

## 5. Ridge、weighting 与数值实现

Weighted ridge 使用 `numpy.linalg.lstsq` 求解 (W^{1/2}X) 与 penalty rows 组成的 augmented system，没有显式矩阵逆。Mean/gain-bias intercept 的 penalty mask 为 false；harmonic 和 condition coefficients 的 penalty mask 为 true。

Mean 支持 `equal_cycle`、`equal_log`、`equal_date`；WB 支持 `equal_sample`、`equal_cycle`、`equal_log`、`equal_date`。除单位 sample weight 外，所有值直接来自 C1 已生成的 partition-aware columns。本次固定 smoke 使用 `equal_log`，不比较或选择 weighting。

真实 smoke 的 Mean/WB design condition numbers 均为 1.1569/1.2198；gain-bias Fx/Fz 为 8.7473/26.6491。Equal-log mean/WB effective weight sums 分别为 20.0/20.0，与 20 个 train logs 一致，证明 smoke 实际消费了 C1 weight columns。所有 prediction 和 coefficient finite；最大逐-cycle waveform mean 为 \(3.84\times10^{-15}\) N。没有发现阻塞 C3 的数值病态。

## 6. Bundle schema

每个 bundle 包含 manifest、model specification、mean/waveform coefficients、feature schema、冻结 normalization、training provenance 和 fit diagnostics。Manifest 记录 model identity、Git state、C1/dataset/prior hashes、ratio/phase contracts、train-only partition、retention、ridge、weights、feature names 和 row/coefficient counts。Hash 覆盖行为相关 payload但排除 creation timestamp；save/load 与 deterministic hash 均由 synthetic tests 验证。

Bundle status 仅允许 `candidate` 或 `smoke_test`。本阶段没有 `selected`、`approved`、`production` 或 `final` bundle。

## 7. Synthetic tests

聚焦与 CLI synthetic suite 共 68 tests，通过内容包括：

- mean-only 已知 coefficient recovery；
- fixed 0.75 retention 的 single harmonic recovery 与 zero mean；
- alpha/frequency conditioned second harmonic recovery；
- fixed-prior 与 shaped (a_\mu=a_w=1) 退化一致；
- no-prior 与 shaped (a_\mu=a_w=0) 退化一致；
- raw prior identity；
- gain-bias recovery；
- bounded physical normal-component scale recovery 与 component-sum gate；
- K1–K4、四种 condition、feature order、missing/NaN/schema gates；
- weighted ridge、intercept penalty、rank-deficiency diagnostics、determinism；
- validation/test fitting refusal；
- keyed row-order prediction、missing/duplicate cycle refusal、bundle round trip/hash/status；
- ratio=7.5、test-loaded、dirty C1 artifact refusal和 headless train-only CLI output completeness。

## 8. Train-only smoke run

输出目录：

```text
artifacts/models/static_correction_smoke
```

固定 smoke 只运行 raw、gain-bias、fixed K=2 alpha+frequency、shaped (a_\mu=a_w=0.5) K=2 alpha+frequency 和 no-prior K=2 alpha+frequency 的 Fx/Fz，共 10 个 bundle。固定 ridge 为 (10^{-4})，不是搜索结果。

| Candidate | Component | Train total RMSE (N) | Train mean RMSE (N) | Train WB RMSE (N) | Coefficients |
|---|---|---:|---:|---:|---:|
| raw_prior | Fx | 3.7187 | 1.3943 | 3.4462 | 0 |
| gain_bias | Fx | 3.2276 | 0.7022 | 3.1514 | 2 |
| fixed_prior_mean_wb | Fx | 1.7018 | 0.7181 | 1.5464 | 15 |
| shaped_prior_mean_wb | Fx | 1.5249 | 0.5882 | 1.4092 | 15 |
| no_prior_mean_wb | Fx | 1.3968 | 0.4972 | 1.3067 | 15 |
| raw_prior | Fz | 8.6121 | 5.5129 | 6.6499 | 0 |
| gain_bias | Fz | 3.1045 | 1.5206 | 2.7059 | 2 |
| fixed_prior_mean_wb | Fz | 3.4977 | 2.4713 | 2.4959 | 15 |
| shaped_prior_mean_wb | Fz | 2.4495 | 1.2376 | 2.1192 | 15 |
| no_prior_mean_wb | Fz | 2.2014 | 0.7679 | 2.0650 | 15 |

> 以上全部是 train-only smoke metrics，不能用于选择模型、比较泛化性能或推荐 candidate。表格只用于证明各接口输出 finite 且不同配置未退化成同一实现。

验证结果：C2 focused/CLI suite 为 71 passed（1.23 s）；C1、registry、ratio-8 lineage、phase、evaluation、normalization 与 training 相关回归为 178 passed（54.34 s）；最终 full suite 为 451 passed（67.22 s）。17 个新增/修改 Python 文件通过 `py_compile`，`git diff --check` 通过。Ruff 与 pyflakes 在既有环境中未安装，因此未运行，也未改变环境。

## 9. Physical component baseline availability

`physical_component_scale` 的 specification、bounded fitting、component-sum check、prediction 和 synthetic recovery 已实现。当前 active prior 的 `train_predictions.parquet` 只有 total `fx_b/fz_b`，C1 waveform table 也没有 row-level normal-force component。现有 EDA0 component CSV 是 phase/half-stroke summary，不是可按 `log_id + timestamp_us` 对齐的 authoritative component artifact。

因此真实 smoke 将 Fz physical component baseline 标记为 `unavailable`，理由和 required columns 写入 `physical_component_availability.json`。该状态不阻塞其他 C2 candidate 进入 C3；后续若要启用，需新增同一 active prior、stable-keyed、component-sum-consistent 的 immutable component artifact。

## 10. 限制与 C3 readiness

1. 当前 target 仍是 provisional whole-aircraft effective longitudinal force，未扣除 tail/body；
2. Physical component real-data candidate 因 authoritative row-level component 缺失而 unavailable；
3. 本实现是 offline complete-cycle model，online virtual-cycle deployment 尚未实现；
4. C2 没有读取 validation/test label，也没有模型选择、validation grid、one-standard-error rule、best alias、final coefficient 或 validation plot；
5. C2 没有训练 dynamic residual 或 TCN；
6. C2 没有修改 tail、moment、controller、label、split 或 IsaacLab production physics。

结论：

```text
READY FOR C3 WITH NON-BLOCKING LIMITATIONS
```

模型族已经实现并通过 train-only smoke test，可以进入 C3。最终候选的 K、condition、ridge、weighting 与 retention 只能在 C3 按冻结 protocol 使用 validation 选择。
