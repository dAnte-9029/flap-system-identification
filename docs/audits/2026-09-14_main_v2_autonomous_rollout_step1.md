# Main V2 autonomous rollout 审计（计划 Step 1）

日期：2026-09-14。审计代码 HEAD：`1482b87`，分支 `step5-actuator-aware-trajectory-main-v2`。
范围：只完成 `docs/plans/2026-09-14plan` 的 Step 1。保留原工作区改动、历史权重与结果；未训练、未打开 sealed test，未执行后续步骤。

## 明确判断

**按问题 5 的严格输入契约，NO：现有 Main V2 没有“只接收初始飞机状态、初始执行器状态和未来指令”的完整启动/续跑接口。** 它需要历史状态、历史指令和掩码初始化内部状态，并显式接收积分时间步。不能把实测初始舵面位置直接交给现有接口。

**但历史初始化后的单次 forward 已经是自主、无未来状态反馈的递推，答案是 YES。** 核心循环没有 2 秒长度限制，可以运行 250 × 0.02 s。因此，NO 的原因不是“每步偷偷读取真实频率/姿态”，也不是必须重写模型才能 free-run。若把完整 GRU/执行器内部状态纳入 initial state，核心转移已经具备所需因果结构，只是当前接口未暴露这些状态。

可以构造只有 t0 的单点历史让代码运行（已有短历史掩码机制）；这是一种初始化假设，不等价于保存的 26 点历史评估，也不能恢复任意过去形成的隐状态。没有证据证明这样的冷启动保持精度。

## 审计对象与演进

当前 checkout 没有 README，`git ls-files '*README*' '*readme*'` 也无返回；改用已有契约、阶段报告、脚本、模型与结果交叉核对。未联网确认远端最新 HEAD。

| 阶段 | 已有工作及与 Main V2 的关系 |
| --- | --- |
| Step 0 | 八月 ULog 固件/结构审计，冻结 F5+C4 cohort；相对编码器相位，无可信跨日志机械零点 |
| Step 1 | `trajectory_dataset_v1`；原生约 50 Hz、过去值 ZOH 对齐、2 秒窗口、整日期划分 |
| Step 2 | persistence/constant twist/ridge/MLP；未来指令为外部输入，状态自主积分；直接加控制未稳定改善 |
| Step 3 / Main V1 | 增加 26 点因果历史、64 维 GRU、1 秒多步损失；history-only 是强参考 |
| Step 4 | 控制可观测性与滞后诊断；发现闭环相关、分布偏移及缺少舵面实测反馈 |
| Step 5 / Main V2 | 冻结 history-only backbone，加 motor drive 和 gated tail 残差；非全新动力学积分器 |
| 后续九月路线 | 有独立数据版本及 5 秒 GRU 诊断，但不是八月 Main V2 checkpoint 的 5 秒验证证据 |

本次历史复核明确使用 Main V2 manifest 指定的 `dataset/trajectory_v1_august_f5_c4`，不是新实验选择旧默认数据。当前 trajectory registry 默认九月版本；不混用两套相位或结果契约，也不涉及 canonical/DeLaurier prior。

## 每一步的实际依赖

入口：`src/system_identification/models/trajectory_main_v2.py::ActuatorAwareTrajectoryModel.forward`。

| 量 | 初始化 | t > t0 的来源 |
| --- | --- | --- |
| position / velocity | t0 日志状态 | 上一步预测，积分预测 NED 加速度 |
| quaternion / body rates | t0 日志状态 | 预测角加速度；中点角速度更新四元数并归一化 |
| flap frequency | t0 测量 | 预测 frequency rate 积分，裁剪到 0.5–20 Hz |
| relative wing phase | t0 编码器相对相位 | 预测频率的梯形积分并取模；phase anchor 固定为 t0 |
| GRU hidden，64 维 | 至多 26 点、截至 t0 的掩码历史编码 | 预测下一步状态特征输入同一个 GRUCell |
| drive state | 历史归一化 motor command 的一阶滤波 | 当前外部 motor command 驱动的一阶递推 |
| tail states，3 维 | 历史 sym/diff/rudder command 的一阶滤波 | 当前外部 tail commands 驱动的一阶递推 |
| controls，4 通道 | 历史用于初始化 | 每步从预先给定的 future_controls 数组取一行 |
| dt | 历史滤波固定 0.02 s | 当前评估从日志 timestamp 差生成；forward 可接受外部固定步长 |
| 参数/归一化 | 已保存权重和 train-only buffers | 固定，不在线拟合 |

网络状态特征为 body velocity、body rates、body-frame gravity、相对起点的相位 sin/cos、frequency；绝对 position 不进入气动网络。base 不使用 controls；motor residual 仅直接改变 frequency rate，再经预测频率间接影响 backbone。tail residual 按对称/差动/方向舵掩码改变加速度或角加速度。

每步先用当前 drive/tail state 计算导数和积分，再以当前 command 更新执行器滤波状态。因此指令经这些路径影响后续步，不是同一步瞬时改变导数。drive state 是归一化指令滤波代理，不是 Hz；tail state 是归一化指令代理，不是已验证的真实舵角。

## Future ground-truth leakage 检查

`training/trajectory_main_v1.py::_model_call` 虽然装载整个 truth tensor 供 loss/metrics 使用，但传给模型的六类状态均明确切片 `[:, 0]`。未来真值仅作为监督/评分目标，不用于 teacher forcing。历史组装选取同一 `(log_id, segment_id)` 的 `first:start+1`，不会取 t0 之后的状态。

| 待查信号 | 是否作为未来状态进入 Main V2 递推 |
| --- | --- |
| measured motor/flapping frequency | 否；未来频率只用于训练辅助 loss / 评估 |
| measured wing phase | 否；未来相位只用于 loss，预测相位自行积分 |
| tail / servo position | 否；当前契约根本没有实测舵角输入；PWM/servo command 不能冒充实测位置 |
| position / velocity / attitude / body rates | 否；仅 t0 状态和过去历史进入 forward |
| wind / airspeed | 否；模型既不读取未来风，也没有显式风状态；地速代替空速的适用性是另一项模型限制 |
| derived acceleration | 否；训练差分目标用于训练统计/监督，不逐步送入 rollout |
| 日志未来其他信息 | future command tape 和实际 timestamp 差是评估输入；未来质量信息用于离线窗口筛选，不进入动力学网络 |

所以“没有任何未来日志信息”若逐字理解不成立：指令是用户明确允许的未来输入，当前 dt 也来自日志。dt 不是未来飞机状态反馈，但若要求完全脱离未来日志时间轴，应由 simulator 时钟提供。未来区间有效性筛选属于离线 cohort 定义，会限制结果适用范围，不是状态 teacher forcing。

八月提取器使用 sample-time/原生主时间轴及过去值对齐；本次结论针对已处理数据到 forward 的依赖链，不将其扩展为所有 PX4 估计器信号的 publication-time 延迟证明。

## 严格 simulator 接口仍缺少什么

1. 显式接收/输出完整 GRU hidden state 的接口。仅 p/v/q/rates/frequency/phase 不能唯一确定当前预测器内部状态。
2. 显式接收/输出 drive 和三路 tail filter states 的接口，以及物理测量到归一化代理状态的初始化约定。目前每次 forward 从历史重新构造。
3. 保存 phase anchor 的续跑状态；简单分段调用并重新设置 t0 会改变相位参考。当前输出仅六类物理预测，不包含 hidden、drive、tail 或 anchor，不能直接无损接续。
4. 由调用方提供明确 dt/时钟；现有 runner 从日志提供且限定为 100 步窗口。
5. 无历史时的初始化策略与验证。历史是合法过去信息，不是 future leakage，但超出了问题 5 所列的现有物理初始状态。

以上是启动、持久化和运行器依赖。没有发现必须额外提供未来风、实测频率、舵角、速度或姿态才能运行的依赖。

另外，当前学习的是 body acceleration / angular acceleration / frequency rate，并不是显式质量、惯量、力矩的完整 Newton–Euler 模型。四元数归一化、导数裁剪和 0.5–20 Hz 频率裁剪不足以证明符合实机包线；20 Hz 不能解释为已验证安全上限。缺少物理异常 fail gate、动作反事实验证及 Main V2 的正式 5 秒多日志评估。这些阻止 RL-ready 声明，但不阻止核心循环运行。

## 当前结果与本次验证

历史结果由 `docs/analysis/results/trajectory_main_v2/summary.json` 与 CSV 核对：4214 train / 3920 validation 窗口，6/5 条日志，8 月 26 日 test 保持封存。模型训练目标 50 步，正式评估最多 100 步。0.10/0.04 s 是根据 Step 4 滞后诊断选定的固定参数，不能称作已独立辨识的真实执行器时间常数。

2 秒日志等权 frequency **RMSE**：history-only 0.37735 Hz → Main V2 0.19567 Hz。不是 MAE。2 秒 Main V2 p/v/attitude/body-rate RMSE 为 1.979 m / 2.124 m/s / 18.285° / 0.766 rad/s；改善并不跨 horizon/指标一致，原 `recommend_enter_h1_h2=false` 不变。

本次用冻结 Main V2 checkpoint、每条 validation 日志的第六个窗口，共五个起点，完成两项有限检查：

- 保持历史、t0、commands、dt 不变，把六类未来 truth 全部替换为 NaN；100 步预测逐位相同。这验证现有调用链没有消费这些未来标签。
- 保持 t0 command，外部给定 0.02 s × 250 步；输出全部有限，检查四元数范数。只证明历史初始化后的 5 秒运算可执行，不是记录的 5 秒 command tape 验证，更不是精度、包线或动作响应通过。

机器结果与 checkpoint SHA256：`docs/audits/results/2026-09-14_main_v2_step1.json`。复核脚本只读 validation 和本地冻结权重，不重训；历史结果未重新全量评估。

复现小规模审计：

```bash
MPLCONFIGDIR=/tmp/matplotlib-main-v2-audit /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/audit_main_v2_step1.py
```

下一步应先把现有隐状态/执行器状态的初始化和续跑契约明确化，随后按计划建立真正的 5 秒 held-out command-tape benchmark；无需以“当前有未来状态泄漏”为理由重写整个模型。本轮停在 Step 1，不宣称可用于 RL。

相关回归验证：`tests/test_trajectory_main_v1.py` 与 `tests/test_trajectory_main_v2.py` 共 **13 passed in 6.20s**；`git diff --check` 通过。未修改生产模型、训练或数据处理代码。
