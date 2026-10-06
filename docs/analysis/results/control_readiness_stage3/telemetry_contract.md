# 后续固件同步日志契约（待实现）

目的：消除当前“不同 topic 记录周期不同、无法确认同一控制周期、缺少滤波初值”的回放歧义。此文件是字段/时序规格，不是已经编译验证的固件补丁。

建议在同一个角速度控制更新内形成一条原子记录，包含：

|类别|最少字段及含义|
|---|---|
|身份|boot/session identity、单调 controller_cycle_id、发布 timestamp、实际 gyro timestamp_sample、实际 dt|
|模式|rates/attitude/manual enable、armed、landed、nav_state、reset 原因及状态版本|
|实际使用输入|三轴 rates/accel、实际消费的 rate_sp 及其源时间戳、有效空速及其源时间戳、分配器饱和标记及其源周期|
|更新前状态|I before、空速滤波状态、compression gain 与 cached gain、高通值、能量滤波值、previous effort|
|更新后状态|I after、上述滤波/增益状态 after，或足以在明确 update 顺序下重构的最小集合|
|输出|P/I/D/FF、空速缩放、trim、限幅前后 torque、输出样本身份|
|外生激励|experiment_id、event_id、injection point、目标与实际注入值、开始/结束及提前终止原因|

- 在源码实际执行点取值，不在另一个低频诊断线程重新读各 topic 后拼成“同步”记录。
- 至少在开始和每次 reset 存完整状态；如不能无损记录所有周期，明确跳过的周期号，不用重采样补齐伪装完整记录。
- 当前 gain_compression 主动 10 Hz 发布、allocator_status 约 5 Hz 记录，单改 logger 不足。控制器真正消费的反馈应随该周期一起记录。
- 分配/驱动端保留源 controller_cycle_id 或明确的源样本身份，以便区分计算延迟、发布延迟与真实执行器延迟。
- 先在台架验证记录吞吐和 dropout，再由实际消息大小×更新频率计算带宽。尚未确定字段布局，不能在这里给出未经验证的 SD 卡负载数值。
- 实测舵角若可获取，保留其独立源时间戳；PWM 命令不能重命名为舵角。

外生激励优先为原控制算法的俯仰参考变化。若改为直接舵面扰动，必须另立 injection-point 契约；pitch torque / symmetric elevon / PWM 不可共用幅值单位。当前离线配置 amplitude_deg 未定，因此无法生成真实执行指令。
