# 纵向力修正 contract

> RATIO=8 phase contract 补充：所有新实验必须使用 aircraft metadata 中唯一的
> `flapping_drive.encoder_to_drive_ratio.value=8.0`。数据、prior、EDA0 与 C1 manifest
> 必须共同记录 `ratio_contract_version=ratio8_v1`、`ratio_source=confirmed_physical_hardware`、
> `phase_contract_version=hall_indexed_mechanical_phase_ratio8_v1` 与
> `frequency_contract_version=flap_frequency_ratio8_v1`；任一字段缺失或不一致均 fail closed。

状态：active contract

版本：`longitudinal_force_correction_contract_v1`

运行配置：`configs/correction/longitudinal_force_correction_v1.yaml`

适用阶段：C0/C1；后续 C2/C3 只能消费本 contract 生成的 artifact，不得在读取时改写其语义。

## 1. 研究对象与范围

Correction 只预测 body frame 下的纵向力

\[
\mathbf F_{xz}=\begin{bmatrix}F_x & F_z\end{bmatrix}^{\mathsf T}.
\]

本 contract 不预测或拟合 \(F_y\)、roll/yaw moment、pitch moment、tail moment 或 body moment。Moment 在后续仿真中继续使用冻结的共同 moment contract；C0/C1 不修改该 contract，也不生成 moment target。

## 2. Authoritative DeLaurier prior

每次运行必须通过 `configs/physics/delaurier_prior_registry.yaml` 解析 prior。省略 `--prior-id` 时使用 registry 的 `default_prior_id`。当前默认值为 `delaurier_attitude_aware_3b5d4ec_trainval_v1`，其 active artifact 是 `artifacts/20260717_delaurier_attitude_aware_3b5d4ec_trainval_v1`。

解析规则为 fail closed：

- lifecycle 必须为 `active`；`legacy`、`superseded`、`historical` 或未分类 prior 均拒绝；
- artifact 缺失、manifest 缺失、请求 partition 不完整或 physics commit 不匹配时立即失败，不回退旧 prior；
- 不按目录日期推断 authority；
- 不接受 July-14 test-window diagnostic 或任何加载过 test partition 的 prior；
- 不接受无 manifest 的 prediction table。

运行 manifest 必须记录 resolved prior ID、lifecycle、absolute artifact path、manifest 与 prediction hash、physics repository/commit/dirty status、frame/phase/airflow contract、separation、dynamic twist、partition coverage 和 key schema。若 upstream manifest 没有记录 physics dirty status，必须显式写成 `not_recorded_in_prior_manifest`，不得推断为 clean。

当前 active prior 的 manifest contract 是：

- physics repository：`https://github.com/dAnte-9029/IsaacLab`；
- physics commit：`3b5d4ec1d28f1384cf042402992ad7ea59995f49`；
- force semantics：two-wing, wing-only DeLaurier force；
- frame：`body_frd_force_at_imu_origin_moment_about_cg`；
- airflow：`attitude_ground_wind_3d`；
- phase：`canonical_mechanical_phase_to_delaurier_v1`；
- separation：disabled, attached flow；
- prescribed dynamic twist：disabled, zero tip amplitude；
- coverage：train 与 validation；test 未加载。

这些字段引用 prior manifest 和 registry；运行时必须重新校验，本文中的当前值不是路径硬编码替代品。

## 3. Reconstructed label contract

Primary label 是 canonical reconstructed effective longitudinal force：

\[
\mathbf y_t=
\begin{bmatrix}
F_{x,t}^{\mathrm{rec}}\\
F_{z,t}^{\mathrm{rec}}
\end{bmatrix}.
\]

Label 来源必须由 `--dataset-root/dataset_manifest.json`、`--split-manifest` 及其 source manifest chain 解析。运行 manifest 记录 dataset ID、每个 partition 的 label artifact、dataset/split hash、label schema、mass-property metadata 与 hash、preprocessing/filter/derivative version、split identity、frame、unit、sign 和 key columns。

当前 canonical ratio-8 SG label contract 由 dataset manifest 与 aircraft metadata 共同声明：

- label 是 `effective_non_gravity_external_force`，即 whole-aircraft reconstructed effective external force；
- body frame 为 FRD，local/inertial frame 为 NED；
- units 为 N；\(+F_x\) 向前，\(+F_z\) 向下；
- attitude quaternion 为 PX4 `wxyz`，表示 body FRD 到 NED；
- mass、CG 与 inertia 的具体版本来自 metadata，不能由文件名推断；
- timestamp 使用整数 `timestamp_us`，单位为 µs；
- alignment key 为 `log_id + timestamp_us`；`sample_index` 只允许作为诊断字段。

## 4. Provisional target scope

C0/C1 的 correction-ready target 为

\[
\boxed{\mathbf y_t^{(0)}=\mathbf F_{\mathrm{rec},t}^{\mathrm{effective}}}.
\]

对应 prior 为

\[
\mathbf F_{D,t}^{P,\mathrm{wing\text{-}only}}.
\]

因此当前 discrepancy

\[
\mathbf r_t^{(0)}=\mathbf y_t^{(0)}-\mathbf F_{D,t}^{P,\mathrm{wing\text{-}only}}
\]

可能同时包含 wing-model discrepancy、tail force、body/fuselage force、wing-body interaction、asymmetry、external disturbance 和 label reconstruction error。

> C0/C1 构建的是 provisional effective longitudinal force correction target，不是最终纯 wing aerodynamic target。

所有表、manifest 和报告均不得把当前 residual 直接称为纯 wing error。

## 5. 后续 final target contract

在后续 C6/C7 中，只有当 task-oriented tail 参数冻结并完成严格 keyed tail replay 后，才允许构建

\[
\boxed{
\mathbf y_t^{\mathrm{final}}
=
\mathbf F_{\mathrm{rec},t}^{\mathrm{effective}}
-
\mathbf F_{\mathrm{tail},t}^{\mathrm{frozen}}
}.
\]

即使到该阶段，body/fuselage force 仍可能留在 final target 中。本阶段不生成 final target、不实现 tail replay、不从 IsaacLab 复制 tail model、不调整 tail 参数，也不修改 label/split。

## 6. Frame、airflow、phase 与 timestamp

运行时从 prior manifest 与 label metadata 解析并交叉检查，不采用手写默认假设：

- label 与 prior 的力必须均为 body FRD；\(+x\) forward、\(+y\) right、\(+z\) down；
- state velocity 的 world/local frame 为 NED；
- quaternion 顺序为 `wxyz`，方向为 body FRD 到 NED；
- air-relative velocity 是 NED ground velocity 减 NED wind 后，通过实测 quaternion 旋转到 body FRD；当前 wind contract 记录 horizontal wind，未记录的 vertical wind 按 upstream authoritative prior contract 处理；
- angle of attack 使用 body-relative airflow 的 \(\operatorname{atan2}(w,u)\) 定义；
- wing phase 使用 canonical mechanical phase；stroke direction 由 `compute_wing_stroke_direction` 的 authoritative direction helper 生成，不凭 \([0,\pi)\) 直觉命名；
- timestamps 是整数 microseconds；`time_s` 只能用于一致性检查，不能替代 stable key。

任一 frame、phase、airflow、timestamp unit、dataset identity 或 partition contract 不一致时，artifact build 失败。

## 7. Partition policy

C0/C1 只允许 `train` 和 `validation`：

- train 用于后续拟合，也是在 C1 中拟合 normalization statistics 的唯一 partition；
- validation 在后续阶段只用于结构选择，本阶段不读取其指标来选择 K、feature 或 model；
- test label 在 C7 结构冻结后的首次 final evaluation 才允许读取；
- C1 cycle/waveform tables 不含 test rows；
- split manifest 可以记录 test log identity 摘要，但不得打开 test label Parquet；
- CLI 请求 `test` 必须在任何输入数据读取前失败；
- manifest 必须写 `test_labels_loaded: false`。

## 8. Keyed alignment

Label、prior、state、phase 与 metadata 以 `log_id + timestamp_us` 做 one-to-one join。禁止依赖 DataFrame row order、index、相同长度或静默 nearest-timestamp matching。

每个 partition 检查 duplicate keys、missing label/prior keys、orphan prior、partition/log identity、timestamp units 及 contract mismatch。所有 mismatch 写入 `alignment_mismatches.csv`，汇总写入 `alignment_report.json`；超过配置阈值时 fail closed。

## 9. 完整 cycle contract

Cycle segmentation 复用 canonical `cycle_id`、唯一相位列 `mechanical_phase_rad`、`cycle_valid` 与 Phase-0 complete-cycle selector。每个 cycle 必须：

- 位于一个 log、一个 partition 与一个 segment；
- timestamp 严格单调；
- phase unwrap 单调且 coverage 达到阈值；
- 不跨 discontinuity 或 invalid phase；
- label、prior、phase 与 condition state 均 finite；
- sample count 达标；
- circular maximum phase gap 不超阈值；
- repeated wrapped-zero endpoint 被显式计数并按 policy 去重。

Artifact `cycle_id` 由 partition、log ID、cycle sequence index 和 start timestamp 确定性 hash 得到。任何 rejection 必须进入 `cycle_quality.csv` 和 `cycle_rejection_reasons.csv`，不得静默丢弃。

## 10. Mean/WB decomposition

对每个完整 cycle \(c\) 与 force component \(j\in\{x,z\}\)：

\[
\bar y_{j,c}=\frac{1}{N_c}\sum_{t\in c}y_{j,t},
\qquad
y'_{j,t}=y_{j,t}-\bar y_{j,c},
\]

\[
\bar y^P_{j,c}=\frac{1}{N_c}\sum_{t\in c}y^P_{j,t},
\qquad
y_{j,t}^{P\prime}=y^P_{j,t}-\bar y^P_{j,c}.
\]

必须逐 cycle 验证

\[
y_{j,t}=\bar y_{j,c}+y'_{j,t},
\qquad
y^P_{j,t}=\bar y^P_{j,c}+y_{j,t}^{P\prime},
\]

\[
\frac{1}{N_c}\sum_{t\in c}y'_{j,t}=0,
\qquad
\frac{1}{N_c}\sum_{t\in c}y_{j,t}^{P\prime}=0.
\]

Residual 定义为

\[
r_{j,t}=y_{j,t}-y^P_{j,t},
\quad
\bar r_{j,c}=\bar y_{j,c}-\bar y^P_{j,c},
\quad
r'_{j,t}=y'_{j,t}-y_{j,t}^{P\prime},
\]

并验证 \(r_{j,t}=\bar r_{j,c}+r'_{j,t}\) 及 \(r'\) 每 cycle 零均值。

## 11. Centered Fourier basis

C1 固定生成 \(k=1,2,3,4\) 的 raw 和 centered sine/cosine basis，不选择最终 K：

\[
\widetilde{\sin(k\phi_t)}
=\sin(k\phi_t)-\frac{1}{N_c}\sum_{s\in c}\sin(k\phi_s),
\]

\[
\widetilde{\cos(k\phi_t)}
=\cos(k\phi_t)-\frac{1}{N_c}\sum_{s\in c}\cos(k\phi_s).
\]

中心化针对每个 cycle 的实际采样点执行，因此非均匀 phase sampling 下仍必须为零均值。

## 12. Condition、normalization 与 weights

Cycle conditions 使用完整 accepted cycle 的 arithmetic mean：AoA、flapping frequency、airspeed、density 与 dynamic pressure。cycle 内 standard deviation 可作为慢变量近似诊断，但不是 C1 的 feature 选择结果。

Normalization 只由 train cycle rows 拟合；validation 直接应用冻结的 train mean/std。每个变量记录 mean、std、minimum、maximum、training row/cycle count、imputation median 与 zero-variance policy。NaN 输入失败；std 小于或等于 \(10^{-8}\) 时保存 raw std 并使用 scale 1.0。Validation 超出 train range 只做 coverage 警告，不参与重拟合。

C1 同时生成但不选择三类 weighting：

\[
w_{t,\mathrm{cycle}}=\frac{1}{N_c},
\qquad
w_{t,\mathrm{log}}=\frac{1}{N_{\mathrm{cycles},\ell}N_c},
\qquad
w_{t,\mathrm{date}}=\frac{1}{N_{\mathrm{cycles},d}N_c}.
\]

因此每个 cycle、log 或 date 的 sample-weight 总和分别相等，长日志不会仅因 sample 数更多而自然主导。若不存在可靠 keyed label uncertainty artifact，相关列不伪造，manifest 标记 `not_available`。

## 13. Artifact schema 与 provenance

Cycle table 每行一个 accepted cycle，包含 identity、quality、condition means、label/prior means、mean residuals 和 cycle/log/date weights。Waveform table 每行一个 accepted sample，包含 stable identity、phase/stroke direction、原始 label/prior、copied cycle means、zero-mean waveforms、total/mean/WB residual、conditions、K1–K4 centered basis 和 sample weights。

输出目录不可覆盖，至少包含 manifest、schema、command、alignment、cycle quality/rejection、cycle/waveform Parquet、normalization、weight contract、dataset summary 和 programmatic quality checks。Manifest 必须记录输入文件 before/after hashes、semantic table hashes、Git state、package versions、dataset/split/prior/metadata provenance 与明确 scope：

```json
{
  "target_scope": "provisional_effective_longitudinal_force",
  "tail_subtracted": false,
  "body_subtracted": false,
  "moment_in_scope": false,
  "test_labels_loaded": false
}
```

## 14. Online deployment note

Offline train/validation artifact 可从 observed complete cycle 直接计算 prior mean；未来在线仿真不能等待真实未来周期。部署时应根据当前慢变量状态 \(\mathbf z_t\)，在完整 virtual phase grid 上评估 frozen DeLaurier prior：

\[
\bar y^P(\mathbf z_t)
=\frac{1}{N_\phi}\sum_i y^P(\phi_i,\mathbf z_t),
\]

\[
y^{P\prime}(\phi_t,\mathbf z_t)
=y^P(\phi_t,\mathbf z_t)-\bar y^P(\mathbf z_t).
\]

C0/C1 不实现 IsaacLab inference。Waveform table 同时保存 current phase、cycle condition、prior mean 与 prior zero-mean waveform，使 C2/C3 可在不改变 target 语义的前提下设计与该 online contract 一致的 inference schema。

## 15. Stage gate

C1 只声明数据是否满足进入 C2 的条件。它不拟合 coefficients、不比较 correction model 性能、不选择 harmonic order 或 condition features，也不生成 gain-bias、ridge/harmonic/shaped-prior/no-prior/dynamic/TCN model。任何这些动作都必须在单独授权的后续阶段执行。
