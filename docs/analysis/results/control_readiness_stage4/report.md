# 同步日志实现准备结果

已完成可审阅的 e624 俯仰同步日志补丁和逐周期记录检查工具；**没有完成新的实机试验，也未证明模型控制可用**。

产物：

- `firmware/px4_pitch_replay/pitch_replay.patch`：五文件补丁；基准、修改后源码和哈希一并保存。
- `scripts/prepare_px4_pitch_telemetry.py`：从指定 Git 提交生成补丁，逐处核对源码锚点，不依赖当前检出的 TCN 内容。
- `scripts/verify_px4_pitch_telemetry.py`：独立临时 Git 树验证应用与完整文件 roundtrip，使用 e624 的消息生成器生成头文件。
- `scripts/check_px4_pitch_cycle_records.py`：对单个连续 rates-enabled 段的 NPZ 记录检查版本、连续周期号、时间戳、dt、PID 求和、缩放和限幅关系。不是新 ULog 的自动准入器，不能绕过原数据隔离；输入 NPZ 应由明确准入的台架/开发日志导出。
- `bench_mapping.csv`：现场舵面映射记录表。物理字段留空，软件 left/right 命名不是物理接线已确认的证据。

同步消息记录实际参与计算的 pitch rate/accel/setpoint、源时间戳、控制 dt、积分更新前后、空速滤波状态前后、gain/HPF/energy/previous-effort 状态、trim 和限幅前后输出。饱和标记在现有 setter 的同一位置镜像，未增加反馈订阅读取。reset_integral 的请求在原有 reset 使用点捕获，避免读取后来刷新的设定值而错配。

存在边界：记录是 pitch 专项，不含完整 Roll B2b 状态机；空速 valid 判断时刻未逐字段记录；未实现完整状态连续回放验证；分配器/驱动端周期关联未扩展。外生激励未实现，external_event_known=false，不能把记录默认零值解释为已知无激励。新增消息开销也需台架测量。

验证结果：

1. exact-base 补丁应用及文件 roundtrip 通过。
2. e624 uORB 消息生成通过。
3. 11 项针对性测试通过，包括丢周期、错缩放、旧 schema、非有限值拒绝，以及原俯仰组件状态测试。
4. 固件语法编译尝试未通过：本地现有 SITL 生成配置与源码不匹配，缺 CONFIG_WQ_VTE_STACKSIZE；错误位于外部 WorkQueueManager.hpp，见补丁目录 syntax_compile.log。没有伪造配置值绕过此问题，尚无可刷写构建。

下一步实际依赖：准备 e624 + 对应子模块的板级构建环境，完成整固件编译和台架吞吐检查；现场填写物理舵面对应/反向/范围及已验证的俯仰参考幅度；然后才能冻结激励与事件生产者并采集新飞行。允许幅度未收到前，既有试验配置 amplitude_deg 仍为空。真实现场动作无法由当前离线工具执行。

本次没有读取任何测试/9.19 数据，没有修改外部 PX4 工作区、原始日志、数据注册表或模型，没有安装依赖。
