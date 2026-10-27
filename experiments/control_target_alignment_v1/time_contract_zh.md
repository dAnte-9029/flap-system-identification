# 五种角速度量及时间合同

仅审计明确指定的历史 `velocity_rate_multistep_v1/main_50hz`，不采用当前 canonical 数据。全部角速度为 FRD p/q/r，单位 rad/s；没有坐标翻转或相位平移。

| 量 | 数值及源码路径 | 时间与可用性 | 本轮身份记录 |
|---|---|---|---|
| initial_rate | `velocity_rate_dataset.py::hold(vehicle_angular_velocity, state_publication)`；`prepare_velocity_rate_learning.py::pack_windows` 从历史末端 STATE 取 initial | rate.timestamp 为发布前软件时刻；rate.timestamp_sample 为输入陀螺采样戳；只取已发布值。相对于位置状态名义时间可略早或略晚，但不晚于决策的可用性限制 | `identity_clock_audit.csv` 的 initial_source/publication、origin/decision；全量表的 initial_source_age |
| truth_initial_rate | `paired_target` 对位置 measurement 时钟取角速度 measurement 最近邻；等距时选左侧 | 离线监督，不要求在该决策时刻发布；与 initial 可来自不同原生记录 | 同表 truth_initial_source/publication；原 timing 的 origin_clock_audit 可逐项追溯 |
| target_A_rate(t) | 同一 `paired_target` 规则生成 TARGET；pack_windows 直接复制未来行，缓存为 float32；没有再次原生插值 | 输出节点为位置 measurement 时间，相邻差形成原生 dt；最近邻角速度源时刻带非零 offset。输出点之间评价再作分段线性插值 | full audit 对每个有效缓存 target、initial、dt 与源样本逐值核验；原 timing 的 future_clock_audit 保存逐输出源戳 |
| native_B_rate(t) | 原生 `native_vehicle_angular_velocity` 的 timestamp_sample 轴上分段线性双侧插值，最大间隔20 ms | 离线角速度参考；可能使用决策后发布观测。未宣称滤波延迟为零或等于绝对物理真值 | `origin_support_availability.csv` 保存左右括号源/发布戳；全量支持计数保留拒绝结果 |
| D_rate(t) | B 曲线固定270 ms中心矩形平均，按分段线性曲线精确积分 | 中心时刻需要未来135 ms；不是在线估计。响应仍减瞬时 B(t0)，不减平均后的起点 | oracle curves/points；0–约500ms窗口只保留完整滤波中心，100/400/500 ms不足处保持NaN |

源码依据是基准提交中的 `src/system_identification/data/velocity_rate_dataset.py`：`past_indices`、`nearest_indices`、`hold`、`paired_target`、`extract_flight`；`scripts/prepare_velocity_rate_learning.py::pack_windows`；`velocity_rate_gru.py::forward`、`errors`；原 `train_velocity_rate_learning.py::window_loss`。未改变这些文件。其 SHA256 登记在 source_receipt。

物理链条直接复用 `local_control_response_time_contract_ablation_v1/time_contract_zh.md` 及已登记固件源码证据：VehicleAngularVelocity 经滤波、校准后发布，仍携带陀螺采样戳；EKF local-position 时间是输出预测器所用的 IMU 状态时钟。软件时间关系已有源码依据，实际硬件/滤波群延迟仍未知。此处的“物理时间一致”是统一原生时钟参考，不是未经滤波的真实转动测量认证。

监督允许未来真实观测；这与历史输入/预测递推使用未来值是不同问题。本轮没有运行预测函数，没有把未来值送入模型。全量 train 约33%、validation约32%的 truth_initial 在决策之后发布，这是其离线标签身份，不能误报为输入泄漏。

## Oracle 的精确定义与边界

- O0：在原位置状态网格的每个未来节点取 A 标签，t0 使用因果 initial；节点间沿用原评价的线性连接。
- O1：未来取 A 标签，t0 取 truth_initial；A口径自洽，但不符合模型当前初值接口。
- O2：原生 B 连续曲线减因果 initial。它是刻意隔离锚点的离线差值，不是自主递推；在 t0 可非零。不人为强制为零或用新状态估计修复。
- O3：B连续曲线减 B(t0)，D也对同一连续曲线平均，作为评价自检。若只把B采样到较稀的注册网格再插值，会引入另一层采样误差，因此不把这种变体混入O3。

O0/O1只涉及角速度标签，没有重新积分姿态，也不证明原13维标签能够被同一积分器同时零误差重现。四者均使用未来观测，不能部署，也不构成模型性能理论上界。

对于有效未来节点，以及300ms原始/慢变比较，`E_increment=E_future−E_origin`。但在 t0 与首个输出节点之间，O0要连接因果初值，而原A曲线从 truth_initial 开始；精确分解还需 `bridge=O0绝对曲线−A绝对曲线`。代码保留这个局部边界项，不隐式改变起点。300ms和对应270ms窗的 bridge 仅为舍入误差。O2不采用该连接，仅执行锚点差值诊断。

113对使用既有控制交换/事实评价的共同支持掩码，部分末端即使单条A标签尚可用，也不扩张旧队列的共同时域。原生插值不跨20ms缺口，注册网格不跨40ms缺口；未支持点保留NaN，未作外推。全量H1000审计另受每个注册窗口的 support_start/end 限制，不能把完整航次中存在的更晚原生记录擅自纳入该窗口。
