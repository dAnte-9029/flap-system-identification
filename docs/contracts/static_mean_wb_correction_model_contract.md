# C2 静态纵向力 Mean/WB 修正模型 contract

状态：active candidate-family contract

版本：`static_mean_wb_correction_model_contract_v1`

适用阶段：C2 实现与 C3 候选拟合/选择接口

## 1. 范围与统一形式

本 contract 仅覆盖 body FRD 下的纵向力分量 (j\in\{x,z\})。每个 complete cycle (c) 的 label 与 authoritative DeLaurier prior 分解为

\[
y_{j,t}=\bar y_{j,c}+y'_{j,t},\qquad
y^P_{j,t}=\bar y^P_{j,c}+y^{P\prime}_{j,t}.
\]

所有 Mean/WB candidate 的统一预测为

\[
\boxed{\hat y_{j,t}=\hat\mu_{j,c}+\hat w_{j,t}},
\]

\[
\hat\mu_{j,c}=a_{j,\mu}\bar y^P_{j,c}+g_{\mathrm{mean},j}(\mathbf z_c),
\qquad
\hat w_{j,t}=a_{j,w}y^{P\prime}_{j,t}+g_{\mathrm{WB},j}(\phi_t,\mathbf z_c),
\]

其中

\[
\mathbf z_c=[\widetilde\alpha_c,\widetilde f_c].
\]

Mean branch 只使用 `cycle_table.parquet` 的每-cycle 行；WB branch 只使用 `waveform_table.parquet` 的 zero-mean waveform target。禁止把同一 cycle mean 复制到 waveform rows 后拟合 mean branch。

## 2. Condition 与 feature contract

允许的 condition set 只有：

- `none`；
- `alpha`；
- `frequency`；
- `alpha_frequency`。

禁止 airspeed、dynamic pressure、history、future state、timestamp delay、自由 phase offset 与任何并行重建的 phase。Prediction 直接消费 C1 已冻结的 `alpha_mean_std`、`flapping_frequency_mean_std` 和 centered Fourier columns，不重估 normalization，不重算 phase、cycle、mean/WB decomposition 或 harmonic basis。

Mean correction 为

\[
g_{\mathrm{mean},j}=d_{j,0}+d_{j,\alpha}\widetilde\alpha_c+d_{j,f}\widetilde f_c,
\]

并按 condition set 保留对应列。实现不包含二次项、交叉项、spline 或神经网络。

WB correction 为

\[
g_{\mathrm{WB},j}=
\sum_{k=1}^{K}
\left[A_{j,k}(\mathbf z_c)\widetilde{\sin(k\phi_t)}+
B_{j,k}(\mathbf z_c)\widetilde{\cos(k\phi_t)}\right],
\]

其中 (K\in\{1,2,3,4\})，condition interaction 只允许 normalized alpha/frequency 与同阶 centered sine/cosine 的乘积。Feature name 和次序由 model specification 唯一确定；缺列、非有限值或 schema mismatch 均立即失败，额外列不改变 feature order。

## 3. Prior retention

`mean_prior_retention` 与 `waveform_prior_retention` 是固定 model specification hyperparameter，范围为 ([0,1])。实现分别拟合 offset target：

\[
r_{\mathrm{mean},c}=\bar y_c-a_\mu\bar y_c^P,
\qquad
r'_{\mathrm{WB},t}=y'_t-a_wy_t^{P\prime}.
\]

Retention 不作为 design column，也不作为普通 ridge coefficient 联合拟合。非有限值、越界值和 model-specific 不兼容值均失败。Bundle 必须原样记录 retention；prediction 不裁剪、不搜索也不隐式修改该值。

## 4. Candidate family

实现下列统一 specification：

- `raw_prior`：严格回放 (y^P)，无拟合 coefficient；
- `gain_bias`：Fx/Fz 独立拟合 (ay^P+b)，是独立 total-force baseline；
- `physical_component_scale`：当同一 authoritative prior 的 row-level component 可可靠 keyed 对齐时拟合 (F_z^P+(s_N-1)F_{z,N}^P)，并检查 component sum；缺少可靠 component artifact 时显式 unavailable；
- `fixed_prior_mean_wb`：固定 (a_\mu=a_w=1)；
- `shaped_prior_mean_wb`：使用 specification 中显式固定的 arbitrary retention；
- `no_prior_mean_wb`：固定 (a_\mu=a_w=0)，与 fixed/shaped 使用相同 K、condition、ridge、weights、rows 和 metrics API。

Physical component correlation 只表示候选形状关联，不构成物理因果证明。Component 数据不得从 legacy airflow 重算，也不得按行序拼接或伪造。

## 5. Fitting 与 weighting

Mean 和 WB 分别求解 weighted ridge：

\[
\min_\beta\|W^{1/2}(r-X\beta)\|_2^2+\lambda\|P\beta\|_2^2.
\]

实现通过 augmented least squares 求解，不显式计算病态 Gram matrix 的逆。Mean intercept 与 gain-bias intercept 默认不惩罚；condition 与 harmonic coefficients 惩罚。每次拟合记录 row count、coefficient count、matrix rank、condition number、weighted residual norm、effective weight sum、rank-deficient 状态与 finite checks。

Mean weighting 支持 `equal_cycle`、`equal_log`、`equal_date`；WB weighting 支持 `equal_sample`、`equal_cycle`、`equal_log`、`equal_date`。除 `equal_sample` 的单位权重外，其余策略直接消费 C1 partition-aware weight columns，禁止根据 validation coverage 重算 train weights。

## 6. Prediction 与 keyed join

Fx/Fz bundle 独立。Prediction 保留 waveform input row order，通过 `cycle_id` 将唯一 cycle mean 映射到 waveform rows，不依赖 DataFrame index。Duplicate cycle ID、missing mean、feature/schema mismatch 或 non-finite output 均失败。

输出至少包含 `prediction_n`、`predicted_mean_n`、`predicted_waveform_n`、`prior_mean_retained_n`、`prior_waveform_retained_n`、`mean_correction_n` 和 `waveform_correction_n`。总预测满足 `prediction_n = predicted_mean_n + predicted_waveform_n`；WB branch 按 cycle 保持数值零均值。`no_prior_mean_wb` 在 prediction base 中不读取 prior 列，`raw_prior` 严格回放 total prior。

## 7. Bundle contract

每个 immutable bundle 包含：

- `bundle_manifest.json`；
- `model_spec.json`；
- `mean_coefficients.json`；
- `waveform_coefficients.json`；
- `feature_schema.json`；
- `normalization.json`；
- `training_provenance.json`；
- `fit_diagnostics.json`。

Bundle hash 排除 creation timestamp，并覆盖 specification、coefficient、feature order、normalization、train-only provenance、diagnostics summary 与 status；相同输入产生相同 hash。Bundle 只允许 `candidate` 或 `smoke_test`，不得标记 `selected`、`approved`、`production` 或 `final`。

## 8. C2 partition policy 与阶段边界

C2 fitting 只允许 `train`。CLI 对 `validation`、`test` 或组合 partition 在数据读取前 fail closed。C1 source artifact 可以记录 train/validation coverage，但 C2 loader 只返回逻辑 train rows，输出 provenance 固定为 `included_partitions: ["train"]`、`validation_labels_loaded: false`、`test_labels_loaded: false`。

C2 只实现候选族和 train-only interface smoke。C3 才允许使用 validation 做 K、condition、ridge、weighting、retention 与 prior incremental value 的模型选择；C4 才进行 dynamic residual audit。Online deployment、dynamic residual、TCN、IsaacLab integration、tail、trim、moment 与 controller 均未在本 contract 中实现。
