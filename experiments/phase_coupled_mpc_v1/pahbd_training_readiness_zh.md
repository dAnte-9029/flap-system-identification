# PA-HBD 正式训练准备：实现、协议与必要验证

2026-10-29；原 worktree `phase-coupled-mpc-v1`，分支 `research/phase-coupled-mpc-v1`，实施基线 `ab1896ed181a60c1d8f781c3db8e9b62ba238792`。本轮保留五模块动力学和 A/B 定义，没有重新选择 Backbone。使用 `/home/zn/anaconda3/envs/flap-train-gpu/bin/python`，CPU、float64、单线程；没有改依赖、混合精度或 Lie 积分数值方法。没有正式 A/B/C 训练、性能排名、超参数搜索、test 数据访问、push 或 PR；既有历史报告和数据未改。

## 1. 结论与可启动范围

- **VERIFIED**：航次均衡/多时域抽样、checkpoint 精确恢复、测量时刻重放、联合频率–相位协方差、谐波不确定性作用、场景预测、冻结 B/D 评价适配均已实现。27 个不同的针对性测试案例通过；四个真实 train 窗口完成默认模型自主前向/反向，500ms/1s/2s 的跨翼拍监督非空。另一个明确单日志相对相位窗口验证了周期 forcing/control 两个头的非零梯度。
- **SUPPORTED**：当前可开展**多航次平均动力学学习**，相位无法获准共享时周期项严格关闭；也可开展**明确单日志参考的探索性周期学习**。后者不能外推为跨日志统一控制增益。
- **UNRESOLVED / 当前共享周期 A/B 的实际阻断**：没有已认证的跨日志共享传感器索引区间；候选 Hall 发布上界仍依赖发布源/执行次序核实，实际机翼机械零点更未建立。默认正式入口对此报错，不能把周期项全关闭后的 A/B 称为有效相位对照。
- **HYPOTHESIS**：相位相关控制增益可改善 MPC 候选控制预测，仍须在合法相位合同下正式训练与评价。本轮没有此类性能结论。

因此，工程入口已具备执行条件；**共享周期 A/B 研究尚不能直接启动**。剩余工作是补充独立传感器参考及可用性证据，而不是再补训练循环。平均动力学的执行范围已经明确，不受真实翼位零点缺失阻断。

## 2. 区间级相位合同

代码：[pahbd_phase.py](../../src/system_identification/data/pahbd_phase.py)、[数据适配器](../../src/system_identification/data/pahbd.py)、[参考合同](../../configs/pahbd/phase_reference.json)。沿用 `velocity_rate_multistep_v1` 注册表、冻结 split/窗口与逐日志 `FLAP_RATIO`；没有读取原始 ULog 或重新做全量原始日志审计。

本轮复用 `phase_per_flight.csv`、`phase_by_date.csv`、`phase_resets.csv.gz`、`raw_verification.json` 及现有报告。为了区分日志级 Hall 存在与窗口内事件存在，只读取了 train/validation 已冻结原生 Hall 表的 `flight_id/timestamp_us` 两列。质量判断不使用模型误差，不拟合航次偏移。

| 项目 | 可执行判定 / 解释 |
|---|---|
| 编码器连续性 | 继承完整编码器/扑频可用性及支持区间；不得跨冻结 `reset_times_us`。该集合同时含 EKF 复位与编码器倒退，不把它误称为纯 Hall 复位。仍不等于已证明传动机构绝无滑移。 |
| 软件相位参考更新 | 按窗口整个支持区间统计既有更新次数及最大圆周修正；`software_reference_unchanged` 仅表示没有记录到更新。正常每拍更新也计数，不能把“有更新”统称物理故障；不自行设 10°/30° 等新准入阈值。 |
| Hall 可追溯性 | `hall_log_recorded` 与 `hall_event_traceable` 分开；后者要求窗口支持区间内确有独立事件。存在事件不证明其发布时刻已知。 |
| 共享传感器索引 | 必须有独立证据支持的区间证书，包含逐日志 `reference_count`、Hall 测量时刻、已知发布时间上界、连续性及同一 `reference_id`。整个窗口支持须位于证书内，锚点必须在支持起点前已发布。当前证书列表为空。 |
| 实际机械零点 | 独立字段 `mechanical_zero_verified`；共享 Hall 索引不自动升级为真实翼位标定。目前全部未证实。 |

三种用途：

1. `shared_sensor_index`：合格证书下使用逐日志计数锚点转换相位，才允许共享周期 forcing 和控制增益。不同未知参考组禁止直接混用；train/validation 必须具有同一认证参考。数学上的全局 `gauge_offset` 仅保留为坐标等价性检查，不承担数据对齐。
2. `log_relative`：冻结连续编码器给出的日志相对坐标。软件 Hall 参考波动不通过全日志固定偏移“修复”。默认用于平均动力学和离线相邻翼拍目标匹配；要激活周期项，须显式 `phase_policy=relative_single` 并指定一个日志参考，不能多日志混训。
3. `average_only`：相对相位本身不可用时，刚体状态监督仍可保留，周期项关闭，跨翼拍匹配不生成；扑频初始化使用已声明的因果先验。

`phase_quality` 现在确实门控 forcing/interaction。默认 `shared_or_average` 在本数据上全部关闭周期项；这不是只增加 metadata。原始 Hall/软件相位信息没有成为未来传播输入。

## 3. Train / validation 覆盖

最终计数见 [coverage.json](pahbd_readiness_checks/coverage.json)，逐航次/时域见 [train](pahbd_readiness_checks/train_coverage_per_flight.csv) 与 [validation](pahbd_readiness_checks/validation_coverage_per_flight.csv)。

继承冻结 `main_50hz`、500ms 历史及四个已存在的预测时域；只保留固件/来源已验证、完整状态序列、非 REVIEW 的窗口。不增加数据、不跨 segment、异常/控制有效性边界，也不生成插值真实状态。主采样组与质量筛选后为 train 93 个、validation 27 个物理飞行片段，不把先前诊断的全部 122/39 航次都算为本训练覆盖。

| split | 名义时域 | 平均动力学窗口 | 日志相对相位合格窗口 | 窗口内有 Hall 事件 | 共享周期窗口 |
|---|---:|---:|---:|---:|---:|
| train | 200ms | 107816 | 107816 | 52540 | 0 |
| train | 500ms | 107054 | 107054 | 52166 | 0 |
| train | 1s | 104690 | 104690 | 50948 | 0 |
| train | 2s | 103312 | 103312 | 50362 | 0 |
| validation | 200ms | 35998 | 35998 | 18667 | 0 |
| validation | 500ms | 35850 | 35850 | 18573 | 0 |
| validation | 1s | 35310 | 35310 | 18274 | 0 |
| validation | 2s | 35147 | 35147 | 18151 | 0 |

这些是冻结合同下的**合格索引计数**，不是对每个窗口都跑过完整神经网络的数量。每个实际装载窗口仍检查因果 bootstrap、segment 与控制年龄，失败即报错，不隐式跳过后改变 A/B 抽样。

原生实际末端相对决策时刻的范围：train 分别 0.188774–0.209183、0.488013–0.508644、0.987174–1.007703、1.989382–2.010574s；validation 分别 0.188782–0.209159、0.488206–0.508576、0.987172–1.007598、1.989279–2.010308s。名义 2s 不伪称每个目标恰好位于 2.000000s。

## 4. 正式训练与验证协议

实现：[training/pahbd.py](../../src/system_identification/training/pahbd.py)、[入口](../../scripts/run_pahbd.py)、[默认配置](../../configs/pahbd/default.json)。

- 先打乱并完整遍历一轮物理航次；每个航次内再轮换打乱其可用时域；最后在该航次/时域合格窗口中均匀抽取。窗口多的航次不因此增加更新权重。相邻更新通常来自不同航次；不宣称抽出的重叠窗口是独立统计样本。
- CPU 元数据索引 + 每次一个窗口的惰性原生装载；无需把全量数据搬到 GPU。A/B 用相同索引、seed=17 与独立于模型 RNG 的采样器。采样器保存随机数状态、未完成轮次、每组抽样次数、已访问窗口及索引身份。
- 实测仅抽索引 372 次，覆盖 93 航次 × 4 时域，每组一次、372 个不同窗口。没有为此执行训练。见 [sampler_coverage.json](pahbd_readiness_checks/sampler_coverage.json)。
- 统一状态流预测原生 p/v/R/ω；主状态损失先按 EKF 与 gyro 来源族归约，翼拍净变化也按来源族归约。窗口内损失归一化，再通过航次/时域采样等权；不按总目标点数累计所有梯度。
- 离线连续编码器仅确定同来源相隔一拍的目标配对：保留既有 30ms 编码器 bracket 限制，把原固定 0.15rad 匹配容差改为目标附近原生同来源相位间隔的一半。不插值 p/v/R/ω；φ 插值只用于离线配对，不进入 encode/flow。净姿态变化比较相对旋转，而非欧拉角相减。
- 固定单位尺度归一化继续使用；若调用统计拟合接口，只接受 train。验证固定子集按 seed、航次、时域、相位模式选取，每组默认一个窗口；当前为 108 窗，不是全 validation 扫描。选模目标先窗口→航次/时域→航次等权；同时记录原生状态误差及逐航次指标。可改变事先声明的验证覆盖预算，但不得按误差重新选择验证样本。
- `validation_interval=1000`、`patience=10` 是明确可配置的初始调度，不是经过搜索的最优参数。训练总步数必须命令行明确给出。每次验证保存新 checkpoint 和 best 指针；恢复保留优化器、采样轮次、Torch RNG、累计步数、最佳分数及早停计数，检查配置、索引与验证成员未变。
- `training_draws.jsonl` 逐更新记录窗口、航次、名义/实际预测时域；`progress.json` 记录验证与覆盖。新 checkpoint 为 `pahbd_v2`，旧 v1 的延迟/不确定性语义不同，拒绝静默恢复成新模型。

未来正式调用形式（**本轮未执行**）：

```bash
/home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/run_pahbd.py \
  --source native --mode train --variant A --steps <明确总步数> --output <新目录>
# B 仅改 --variant B；恢复另外指定 --checkpoint <step_*.pt>，steps 表示累计总步数。
```

默认 `objective=shared_periodic_AB` 会因当前零共享覆盖而停止；若下一轮决定先做平均动力学，显式将配置中的 objective 改为 `average_dynamics`，并将 `data.phase_policy` 改为 `average`。不能将此结果用来评价周期控制增益 A/B 假设。

## 5. 延迟观测：有界决策时刻重放

实现：[belief.py](../../src/system_identification/models/pahbd/belief.py)。

窗口保留历史观测、初始状态与原生 ZOH 控制；缓存容量为 0.6s，覆盖冻结 500ms 历史及原生端点余量。决策时刻先按发布时间筛选，尚未发布的观测连值都不读取。拒绝测量在缓存起点之前、延迟超过容量的观测；测量晚于发布、历史超容量则直接报错。

对最终已到达观测集合按 `(measurement, availability, source)` 排序，使用同一个 Lie 流传播到测量时刻，计算创新并修正，再沿**此前实际发生的控制**重放到下一个测量时刻及最终决策时刻。这是把逐次插入迟到观测后重放后缀合并为一次决策时刻重放，避免训练中反复执行相同后缀。没有第二套生产动力学；也不是完整贝叶斯平滑器或常驻在线消息服务。

修正为 `x⁺=Retract(x⁻,K r(t_measurement))`。年龄仍是 Encoder 特征，但不再用年龄噪声掩盖测量时刻错位；f/φ 协方差用带学习对角增益的 Joseph 形式更新，保留交叉项。它是学习滤波近似，不宣称精确贝叶斯后验。

**VERIFIED**：恒速已知轨迹的历史创新/当前状态误差 <1e-12；迟到顺序与测量顺序相反仍正确；未来发布的 NaN 不影响当前估计；缓存外读数明确拒绝；含控制切换与一阶执行器响应的解析轨迹在细化至 0.5ms 的同一 Lie 积分下误差 <2e-6。初次较粗中点步长的速度误差 2.253e-6 超出该测试容差，因此细化测试积分，而没有放宽容差。

真实四窗最大已接受延迟约 10.05ms，各有两条测量早于缓存起点的到达记录被拒绝。bootstrap 仍由缓存前各传感器最近因果观测构成，保留已有新鲜度约束及初始化方差；这是近似初始先验，不宣称这些异步种子天然同时。

## 6. f–φ 联合不确定性与周期状态流

实现：[dynamics.py](../../src/system_identification/models/pahbd/dynamics.py)、[flow.py](../../src/system_identification/models/pahbd/flow.py)、[场景入口](../../src/system_identification/evaluation/pahbd.py)。

保留 train 拟合并冻结的一阶扑频模型与解析均值传播。连续误差系统为

\[
A=\begin{bmatrix}b_f&0\\2\pi&0\end{bmatrix},\quad
Q=\operatorname{diag}(q_f,q_\phi),\quad
\Sigma(t+\Delta)=F\Sigma(t)F^T+\int_0^\Delta e^{As}Qe^{A^Ts}\,ds.
\]

用 Van Loan 矩阵指数计算精确离散协方差；Σ 是独立完整 2×2 状态，14维接口最后两个对角仅同步其边缘值，不再独立累加丢掉相关性。默认 `q_f=0.04 Hz²/s`、`q_φ=0.01 rad²/s` 是标注为**未校准的建模假设**，不是用实飞误差估计的物理过程噪声。

默认 `uncertainty_mode=gaussian`：周期第 k 阶 cos/sin 乘 `exp(-k² σ_φ²/2)`，中点也使用中点协方差，因而不确定性直接改变周期 forcing 与控制作用。它是在局部状态下的 wrapped-normal 谐波期望闭合，不是非线性全轨迹分布的精确期望。`deterministic` 明确忽略这项衰减，不应默认为长时域可靠预测。

`phase_scenarios()` 使用四个等正权初始联合 sigma 点，保留 f/φ 相关性，沿相同候选控制和同一 Lie 流演化。场景内初始协方差置零以免重复计入，后续过程扩散继续传播；这不是完整随机过程路径 Monte Carlo。输出场景、权重、初始扰动与解析均值参考，区分于单条预测。φ 均值保持 unwrapped，只有周期基与相位创新使用圆周表示；不对 unwrapped 协方差做模 2π 操作。

**VERIFIED**：带过程噪声的协方差半群、PSD、非零交叉项；谐波衰减解析规律及 2π 周期等价；四场景重现初始联合协方差；2s 四场景相位均值相对解析均值误差 `2.13e-13 rad`。A/B 使用完全相同机制，只在显式相位相关控制 interaction 是否启用上不同。

**UNRESOLVED**：实飞概率覆盖率未校准。已有 2s 三阶相干量约 0.025 是长时域相位可靠性警示，未被用来假装标定 Q，也未把确定性三阶波形当作可信事实。

## 7. 冻结 B/D 控制响应入口

`FrozenRollResponse` 只读引用已有 `local-control-response-rollout-v1/experiments/local_control_response_rollout_v1`。保留其 `pairs.csv/cohorts.csv/windows.csv`、原匹配质量字段、原航次与 pair_uid；复用原 `analyze.py` 的依赖标签、等权算法、方向/幅度度量，不重选匹配、不重估阈值。

**VERIFIED**：主 train 队列仍是 48 对、11 个共享依赖分量；validation 主队列仍只有原来一对。数值 label-oracle 只用于验证评价适配，得到零响应误差，不是模型预测结果。

`evaluate(predict, split)` 接受通用预测回调；`factual_predictor(model, windows)` 接通已训练 PA-HBD A/B，也允许将来 C 的同接口实现。两侧分别以自身历史 encode、事实控制和原生时刻预测，然后比较两侧状态增量之差。保留原离线标签原点、因果预测原点差异和原始测量时钟，并校正测量原点到决策原点的时间偏移。评价插值仅复用既有标签评价规则；超出共同支持不外推，短端点不改名精确 500ms。

输出方向、幅度、事实轨迹误差、增量误差、每依赖分量及 pair/flight/component 三种归约。每个时刻只对有支持的对重归一化权重，依赖身份始终保留完整冻结队列的11分量，不因缺失支持重新拆分。方向近零阈值沿用原协议 0.1（相应速度/角速度单位），预测弃判阈值 1e-8；不据未训练模型改变。`matched_control_response()` 的同历史交换控制接口继续明确标记 model-internal，不能充当真实反事实标签。

入口：`run_pahbd.py --mode response --source native --checkpoint ... --response-source ... --response-split validation --output ...`。本轮没有对未训练模型运行性能评价或报告控制有效性提升。

## 8. 必要验证与运行成本

完整复现入口：[verify_pahbd_readiness.py](../../scripts/verify_pahbd_readiness.py)。输出：[verification.json](pahbd_readiness_checks/verification.json)、[真实 train 检查](pahbd_readiness_checks/real_train_checks.json)。正式实飞训练优化器步数为零。另执行一次命令行 synthetic smoke（1次合成优化更新），验证入口与 v2 checkpoint 产出；没有将其当作研究性能结果。

| 名义时域 | 实际时域/s | 前向/s | 反向/s | 原生目标数 | 跨翼拍配对数 | 进程峰值 RSS/MiB |
|---|---:|---:|---:|---:|---:|---:|
| 200ms | 0.206448 | 1.005 | 2.194 | 42 | 0 | 1919 |
| 500ms | 0.498521 | 1.307 | 2.022 | 100 | 38 | 1944 |
| 1s | 1.004862 | 1.816 | 2.896 | 201 | 139 | 2101 |
| 2s | 1.995453 | 3.190 | 4.878 | 398 | 336 | 2452 |

四窗来自四个独立 train 片段；均采用默认宽度128/latent12，原生数据装载约0.11–0.27s/窗。RSS 是含解释器、CPU 索引和缓存的累计进程峰值，不能称为纯激活或 GPU 显存。这里默认共享准入未通过，周期项关闭；另一个单日志、width16 的真实窗口单独验证周期头非零梯度。合成有已知相位的检查覆盖完整激活结构。

所有真实窗 loss/梯度有限；四元数范数最大误差 2.22e-16；联合协方差最小特征值 ≥4.82e-4，末端 fφ 交叉项非零。完整翼拍配对是相关监督，不是38/139/336个独立实验。

**SUPPORTED 成本判断**：默认 CPU 入口可执行；按已测单窗计算，1000 次更新约0.9–2.2小时量级，另加数据初始化、验证和保存时间。这只是少量样本算术外推，未实测完整预算、未测 GPU 吞吐，也没有为速度改变精度/控制事件/相位积分。

测试命令：

```bash
MPLCONFIGDIR=/tmp/pahbd-matplotlib PYTHONPATH=src \
  /home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest \
  tests/test_pahbd.py tests/test_pahbd_readiness.py -q
```

**VERIFIED**：完整原有/新增组合运行 `26 passed（31.17s）`；最终增加正式科学准入测试并调整有缺失支持的分量权重后，对新增/受影响文件复测 `12 passed（12.59s）`。合计27个不同测试案例通过。覆盖：完整梯度与 SO(3)、控制等价事件表示与时间分段、解析相位/联合协方差、A/B 仅 interaction 差异、离线相位不泄漏、迟到观测解析参考、抽样均衡/可复现/跨checkpoint连续性、合成两次更新的正式调度恢复、未知参考隔离、合成共享证书的逐日志锚点/因果门槛、冻结48/11评价及真实单日志周期反向。调度测试只用合成微小窗口，没有真实正式训练。

## 9. 真正剩余的边界

1. 共享周期 A/B 需要认证的共享传感器索引与可用时间证据；真实翼位标定是进一步的物理解释要求。不能用模型验证误差或未来 IMU/轨迹优化航次偏移来替代它们。
2. 原生 p/v/R 多来自同一 EKF，gyro 也非独立无误差真值；本轮指标是观测一致性。全机有效加速度/角加速度的学习不需要伪造质量/惯量；若以后解释成隔离翼面载荷或真实力矩，则还需要对应硬件与参考点合同。
3. 未校准过程噪声、单日志相对相位研究和观察性匹配均有上述解释限制；48 对不得按48个独立干预计算置信度。
4. 正式预算、选定科学范围和最终 C 实现尚未执行；本轮完成后停止，不自动进入后续训练。
