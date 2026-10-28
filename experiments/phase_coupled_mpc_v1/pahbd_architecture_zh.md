# PA-HBD：架构、训练接口与工程验证

2026-10-28。研究分支 `research/phase-coupled-mpc-v1`；实施前 HEAD `8e6848c2ab0a59fa12cf11f53818c85c6e05346d`，工作区干净。全部新增代码位于本工作区，没有修改旧 GRU、相位预测报告、数据、原始日志或轨迹重建产物。

**交付状态：完整网络、因果历史估计、统一自主积分、原生数据适配、训练/checkpoint/验证接口已连接并执行。正式 GPU 训练、性能排名、超参数搜索均未进行。** 本文的数值是工程验证证据，不是实飞预测能力或周期控制有效性证据。

## 1. 代码入口与网络规模

| 模块 | 代码 | 默认输入、输出及参数量 |
|---|---|---|
| 总模型 | `src/system_identification/models/pahbd/__init__.py` | `PAHBD(Config)`；`encode(window)`、`forward(window,times)`；总计 **50,683** 个可训练参数 |
| 类型合同 | 同目录 `types.py` | `State`、`Observation`、`ControlTape`、`Window` |
| SO(3) | 同目录 `geometry.py` | Hamilton 乘法、指数/对数、旋转残差、右侧切空间修正 |
| Belief Encoder | 同目录 `belief.py` | 70 → 128 → 27；**12,571** 参数 |
| Slow Backbone | 同目录 `dynamics.py` | q 为 22 维；22 → 128 → 128；memory 为 131 → 12；含 retention 共 **21,052** 参数 |
| Phase Propagator | 同目录 `dynamics.py` | 解析推进 f、unwrapped φ；0 可训练参数 |
| Control Mixer | 同目录 `dynamics.py` | 128 → mean/B0:24、forcing:36、low-rank left:72；另有 3×2×3 right 因子；**17,046** 参数 |
| Unified Flow | 同目录 `flow.py` | 唯一物理流；另有 14 个过程不确定性参数 |
| 数据 | `src/system_identification/data/pahbd.py` | 冻结窗口 + 原生异步事件；`ConfiguredWindows` 惰性逐航次缓存 |
| 合成受控系统 | `src/system_identification/data/pahbd_synthetic.py` | 独立 NumPy ODE / SciPy DOP853 真值；不是模型自身 rollout 当作真值 |
| 训练 | `src/system_identification/training/pahbd.py` | trajectory/window loss、train-only normalization、optimizer、checkpoint、validate |
| 评价 | `src/system_identification/evaluation/pahbd.py` | native horizon、flight-equal、翼拍截面、谐波、B/D 响应、内部控制 Jacobian |
| 命令行/配置 | `scripts/run_pahbd.py`、`configs/pahbd/default.json` | `smoke`、显式 `train`、`validate`；A/B 共用接口 |
| 针对性测试 | `tests/test_pahbd.py` | 15 项 CPU 测试 |

单个状态允许前导 batch 维；flow 和 mixer 已测试 batch=2。真实数据是不同长度的异步事件，训练入口逐窗口计算并反向传播，没有通过补齐或共同时间栅格改写观测合同。

## 2. 状态、坐标与信息流

物理状态 `x=[p_N(3),v_N(3),q_NB(4),omega_B(3)]` 共13维。p 单位 m、v 为 m/s、ω 为 rad/s；NED、FRD、Hamilton wxyz，q 是 FRD→NED 主动旋转。扩展状态为 `h(12), f(1), phi(1), actuator(3)`。默认 h 可配置为8–16等维度；不包含另外一条位置或姿态轨迹。

`variance(14)` 对应 `[p,v,右侧旋转切空间,omega,f,phi]` 的有界对角代理。`phase_quality(1)` 标记因果相位参考是否可用于周期项，不是“物理翼位已认证”的概率。h 不另建自由窗口参数或假装有完整贝叶斯协方差。

信息流：

```text
已发布历史观测 + 测量/availability 时间 + mask/quality + 历史控制
  → [统一 flow prediction → SO(3) observation update] 按事件重复
  → 起点 belief (x,h,f,phi,a_s,Sigma)
  → 同一 flow + 候选/记录控制 → 唯一自主物理轨迹
       ↳ 当前 v_B,omega,g_B,f,h → slow context
       ↳ context → 平均/周期强迫/周期控制系数
       ↳ phi、a_s → 单个六维等效作用 → Lie 流反馈到当前状态
```

预测起点之后不调用观测更新。未来状态、相位、频率仅能出现在监督/离线诊断对象中，不进入初始状态修正或动力学输入。`forward` 从 target 获取的仅是查询时间，数值和掩码只由 loss 读取。

`a_eff^B`、`alpha_eff^B` 是等效运动变化率，分别 m/s²、rad/s²。它们不是实际加速度标签，不具有 N/Nm 或孤立翼载荷解释。无需假定单位质量或虚构9–10月惯量；状态参考沿日志合同，尚不能宣布质心精确运动或安装点补偿已完成。

## 3. 因果信念估计

历史按 **availability** 排序；`measurement > availability` 被拒绝。只接收 `history_start ≤ availability ≤ decision`。角速度/位置/姿态分别保留原生时间；编码器和频率沿冻结表使用保守 `availability_us`，不把 backdated `timestamp` 当发布时间。

每一事件先调用统一 flow 从上一事件推进，然后输入：14维有掩码残差、14维掩码、14维质量、14维 log variance、测量年龄、距上次事件 dt、当前 h。姿态残差为 `Log(q_pred^{-1} q_obs)`，观测修正为 `q_pred Exp(delta_theta)`，随后归一化。

观测增益是受 sigmoid 限制的神经门控乘对角信噪比；variance 被限制在 `[1e-5,25]`。h 更新是当前 h 与有界候选的凸组合。缺测事件不修改 h 或物理状态。频率残差是标量差，相位使用圆周残差但状态维持 unwrapped 连续计数。

**迟到观测的当前实现是年龄感知的零阶观测近似**：将旧值与当前预测比较，并按年龄增加噪声；不是精确 delayed-state smoother。已保留测量时间，未来可替换观测模型而不改变动力学结构。窗口最早的可用各源状态作因果 held bootstrap，位置只减去该历史起点的位置；没有利用未来轨迹优化初态。初始相位使用历史编码器和已可用频率做年龄外推，这也属于有明确限制的初始化近似。

## 4. 慢变记忆、相位与控制混合

`q=[R^T v,omega,R^T g,f,h]`，两层128维 SiLU 网络生成 context。默认固定单位尺度，不拟合任何数据统计；提供 `fit_normalization`，只允许 train 的因果初始上下文，统计进入 checkpoint。h 在自主预测中持续推进：

\[
\dot h_i=\frac{\rho_i}{2H_b}(H_b\tanh m_i(c,a_s)-h_i),\quad
0<\rho_i<0.8,\quad H_b=2.
\]

此式给出有界吸引目标和最大演化速率；积分还限制最大步长以保持该约束。它可以随状态/执行器持续受迫，不宣称所有真实气动记忆一定自然指数衰减。限维、限速、有界吸引目标是有限数据下的**归纳偏置**，不是已辨识的物理定律。该约束限制自主 h 直接承担高频波形，但不能证明潜变量完全没有周期信息；历史观测门控也能更新 h。

频率模型逐 ZOH 区间采用原 `phase_forecast.py` 的解析解：

\[
c=b_0+b_u u_m,\ f_*=-c/b_f,\quad
f(t+\Delta)=f_*+(f(t)-f_*)e^{b_f\Delta},
\]
\[
\phi(t+\Delta)=\phi(t)+2\pi[f_*\Delta+(f(t)-f_*)\operatorname{expm1}(b_f\Delta)/b_f].
\]

含 `b_f=0` 极限分支。参数原样来自 `frequency_fit.json` 的 train-only 拟合 `(3.6354511984,23.3935700690,-5.1267253952)`，存为冻结 buffer。不是重训频率模型，不将候选时间常数当真实机械标定。`PhasePropagator.covariance` 提供 (f,φ) 2×2 协方差传播入口；主 belief 当前仍使用有界对角代理，不声称校准覆盖率。

控制映射直接来自冻结 `build_config.yaml`，并逐航次检查 required mapping parameters：motor=`actuator_motors.control[0]`；左/右升降副翼、方向舵=`actuator_servos.control[0/1/2]`。不自行混成舵角或改变方向符号。三路代理满足 `dot a_s=(u_servo-a_s)/0.06s`，解析更新；0.06s 是配置中的建模先验，不是舵机测量。电机进入 f，再通过 f/context 和相位影响运动；不另加无来源的直接推力映射。

Mixer 精确实现任务给定形式。B0 为 6×3；forcing 为 Hp×2×6；周期控制为 Hc×2×6×3，通过每阶6×rank左因子和rank×3右因子生成，默认 rank=2。均值分支不接收 φ，forcing 不直接接收 a_s；周期控制只通过显式矩阵乘 `delta a_s` 进入。默认参考控制为零归一化命令，不宣称是平衡舵角。

A/B 分配相同参数和使用相同积分/训练合同，唯一开关是 B 将直接周期控制项置零。B仍有飞行状态、f、h、a_s 的间接反馈。A的直接增益可从 `control_sensitivity` 查询，但结构分离和模型 Jacobian 均不是因果可辨识证明。

固定 context、a_s 时，非零阶 sin/cos 在相位测度上严格零均值；不是非恒定频率运动的时间平均。令 φ′=φ+δ：每阶 `(c,s)` 变换为 `(c cos kδ−s sin kδ, c sin kδ+s cos kδ)`。`gauge_offset` 对 forcing 和低秩左因子执行此变换，单步作用和完整状态流等价性均有测试。

默认 `phase_mode=relative`，起点有效才开启周期项；`disabled` 或 `phase_quality=0` 时只保留平均项。每窗口明确记录 whole-log encoder reference。未使用 wing_phase 的软件零点作为跨航次真实机械零位，也不拟合每日志相位偏移。**结构可以运行，但混合未知相位零点的正式共享周期模型如何训练仍需后续协议明确；本轮不替用户选择该协议。**

## 5. 唯一积分器与数值合同

\[
\dot p=v,\quad \dot v=g+R a_{eff},\quad
\dot R=R[\omega]_\times,\quad \dot\omega=\alpha_{eff}.
\]

使用显式 Lie midpoint：半步状态计算新的 context/作用，完整 p/v/ω/h 由中点导数推进，姿态右乘 `Exp(omega_mid*dt)`。f/φ/a_s 在半步及完整步均解析推进。网络、观测更新、控制和固定查询时间的完整 flow 可由 PyTorch 反向传播。

每段先切控制事件，再细分到 `dt ≤ min(5ms, 0.15/(2π max(Hp,Hc) max|f|), H_b/rho_max)`。保留真实非均匀事件时间。同值控制事件先精确合并；可微控制变量不合并，避免吞掉独立控制参数的梯度。查询时间改变会改变数值网格，结果只要求在积分误差内一致。整数翼拍边界不会重置状态。

`wingbeat_transitions` 用同一解析频率模型定位下一个相位+2π截面，并调用同一 flow。它是评价接口，支持正频率、有效参考；截面定位在 no-grad 下进行，不声称已经实现事件时刻对参数的可微导数。固定时间轨迹可微能力不受此限制。

## 6. 原生数据和训练目标

从来源工作区原样继承注册表到 `configs/data/velocity_rate_dataset_registry.yaml`。冻结 manifest 身份：`8afb8623bf7174e4da5b2044f644331b8b251433fed3576f6dcdab470935bcb5`。数据根为 `/home/zn/flap-system-identification/dataset/velocity_rate_multistep_v1`。适配器构造时只核对这一 manifest，不遍历或哈希 test 文件。

只允许 `train/validation`，默认 main_50hz、500ms历史、200ms未来；历史/未来时长可配置到冻结窗口已有规格。使用原完整窗口、segment_id、固件/来源有效性过滤，不跨航次、reset、exclude、gap。默认查询到 **109,682** 个合格 train 窗口；本轮只加载极少量窗口验证。`ConfiguredWindows` 不在正式入口预加载所有航次张量。

历史保留原生观测值、测量/availability 时间、掩码、质量和来源。未来 p/v、q、ω 各自在其原生测量时刻查询预测，不把状态标签插值到统一50Hz。控制从发布时间作 ZOH。记录 flight/window/segment/split、相位来源、日志参考及条件重放性质。未来实际控制仅表示 factual conditional replay，不是已经证明的反事实响应。

主要 loss 是经积分的状态残差，姿态使用3维旋转残差；不加载或监督加速度/角加速度。位置速度噪声来自原 eph/epv/evh/evv 且有上下界；姿态0.03rad、角速度0.05rad/s为明示的可修订噪声假设；review/dead-reckoning 降质量。EKF p/v/R 合并为一个归一化来源组，gyro 为另一组；不把它们的和解释成独立测量联合似然，也不额外把重建轨迹当独立真值。

\[
L=L_{state}+0.01L_{variance}+0.1L_{cycle}.
\]

Lstate 使用固定观测噪声，**无法通过扩大预测方差降低**。附加不确定性评分用有界预测variance与观测variance之和，包含 log 项；这只是校准代理。支持多时间尺度分段权重、train-only normalization 和 Adam；不为每窗口创建可自由拟合的 h0/x0。checkpoint 保存模型/优化器、配置、归一化buffer、种子、数据/窗口身份、代码基准revision及dirty状态。

Lcycle 对同一来源两时刻的 p/v/ω增量及相对旋转增量监督。真实配对使用**仅目标侧**编码器相位：30ms以内相邻编码器插值用于选择近似同截面，而状态值始终原生；一周相差2π的容差0.15rad。配对及目标相位保存在 metadata，encode/flow 不读取。它们是近似传感器相位截面，不宣称实际机械零点/精确翼位；未来相位仅用于离线监督，不用于自主初始化。200ms短窗口可能没有完整配对，cycle项即零；已用500ms真实窗口验证非空配对。

## 7. 评价与边界

- `native_horizon_errors`：200ms、500ms、1s、2s 的 p/v/R/ω；在30ms容差内找最近原生观测，保留实际时间、航次和窗口；缺支持则不造指标。`aggregate_flights` 先航次内均方，再航次等权开方。
- `wingbeat_transitions`：自主模型精确相位截面的净状态转移；`section_transitions` 也支持已有离散输出上的近似截面。
- `periodic_amplitude_phase`：仅诊断的含截距傅里叶最小二乘；要求足够相位覆盖，不用于每航次初始相位对齐。
- `matched_control_response`：同一起点 belief 下 B/D 控制带交换，接收可选真实匹配观测作对照；匹配设计、支持域、共享航次依赖和混杂校正由后续协议提供，不自动造匹配对。
- `control_sensitivity`：固定当前 belief/context 的内部6×3控制作用 Jacobian。A/B配置具有相同接口；本轮只做结构差异测试，没有生成A/B性能排名。

## 8. 实际执行与复现

在研究 worktree 下，使用既有环境，没有安装依赖或操作其他进程：

```bash
P=/home/zn/anaconda3/envs/flap-train-gpu/bin/python
MPLCONFIGDIR=/tmp/pahbd-mpl PYTHONPATH=src "$P" -m pytest -q -s tests/test_pahbd.py
MPLCONFIGDIR=/tmp/pahbd-mpl "$P" scripts/run_pahbd.py --source synthetic --steps 2 --output /tmp/pahbd-synthetic-smoke-v1
MPLCONFIGDIR=/tmp/pahbd-mpl "$P" scripts/run_pahbd.py --source native --windows 2 --steps 1 --output /tmp/pahbd-native-smoke-v2
```

输出目录必须不存在；复跑时指定新目录，拒绝覆盖既有 checkpoint。最后一条因惰性载入接口调整重跑过；所有更新均属 CPU 工程 smoke。pytest 内另有一次极短的合成优化及 checkpoint 测试，测试修正后重跑不属于训练实验。

实际结果：

| 验证 | 结果 |
|---|---|
| 完整模块梯度、SO(3)、ZOH、非均匀时间、future poison、mask、相位规范、A/B、batch、评价、checkpoint、原生因果性 | **15 passed in 14.63s** |
| 独立受控周期 DOP853，dt=10/5/2.5ms | 混合单位状态向量差范数 `1.0285443e-4 / 2.6900841e-5 / 6.8683181e-6`；仅数值收敛量，不是物理评分 |
| 三阶谐波解析测试，最高谐波相位步长0.15/0.075rad | p/v组合误差范数 `8.08555e-6 / 2.04650e-6` |
| 合成默认50,683参数模型 | 44历史事件、88目标事件；2次CPU更新，loss/梯度有限，19个参数张量改变；含非零跨翼拍损失 |
| 真实train，2个不同航次，200ms | 各208历史事件，42/40目标事件，37/37控制事件；1次CPU更新，两窗口分别前后向有限，19个参数张量改变 |
| 真实train，500ms | 100原生目标，35对近似同相位跨翼拍监督；loss和梯度有限；0次优化更新 |
| checkpoint | CLI保存/读回逐参数相等；pytest另验证读回后完整预测逐项一致 |
| formal GPU training / test读取 / 性能比较 | **0 / 0 / 0** |

两真实航次为 `4c77a3e7…:1`、`37ff9f8e…:1`；完整身份、smoke结果及代码基准信息集中在 `pahbd_engineering_verification.json`。CLI smoke 的checkpoint留在对应 `/tmp` 目录，未作为训练成果发布。

以下是**后续入口说明，本轮未执行**：

```bash
# 需要下一轮明确授权和确定训练预算；0表示全部已批准train窗口。
"$P" scripts/run_pahbd.py --mode train --source native --variant A --windows 0 --steps 1000 --output /path/to/new-run
# 只对显式checkpoint和validation运行；不是封存test入口。
"$P" scripts/run_pahbd.py --mode validate --source native --checkpoint /path/to/new-run/checkpoint.pt --windows 2 --output /path/to/new-validation
```

1000仅演示CLI参数，不是推荐预算。可修改训练配置的horizon以载入相应冻结窗口。验证入口已用合成数据执行函数级测试，没有把未训练网络的实飞误差作为研究指标输出。

## 9. 设计依据与尚未验证内容

本轮按任务给定的截断仿真、因果事件更新、潜在自主动态、条件系数和显式周期旋转结构实施；没有声称逐篇复现所列论文，也没有扩展文献路线选择。已有相位报告、候选频率公式、原native-rate SO(3)实现和冻结数据合同已查阅；另只读核对轨迹重建 Stage02.2–02.6 关键结论：周期观测可重复不等于全机运动归属，慢变/周期补偿存在耦合，GPTR加速度未达到真实标签条件，删除GNSS速度的固定条件实验不能推广成物理真轨迹。

**已验证的工程能力**：模型连接、可微固定时间自主预测、规范等价、几何约束、时间/控制合同、目标隔离、极少量真实train接口和数值收敛。

**结构设计假设**：低维限速记忆、低秩控制效能、固定噪声/年龄膨胀、执行器一阶代理、冻结候选频率模型。它们均可配置，但本轮没有搜索优劣或物理辨识。

**仍无实飞证据**：跨航次统一真实翼位、9–10月实际舵角/安装/质量惯量、显式周期控制增益的因果真实性、反事实控制响应、MPC闭环能力和不确定性校准。

**需正式训练才能判断**：泛化误差、A相对B的额外预测价值、h是否有效、频率模型与周期阶数的实际适用性。本轮到可复现架构与必要验证为止，不自动启动下一阶段。
