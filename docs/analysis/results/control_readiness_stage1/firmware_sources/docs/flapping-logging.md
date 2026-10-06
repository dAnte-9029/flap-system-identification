# 扑翼控制日志采集

本机使用 Pixhawk 6C Mini，对应构建目标 `px4_fmu-v6c_default`，固件路径为 `build/px4_fmu-v6c_default/px4_fmu-v6c_default.px4`。该板配置已包含 AS5600、RPM capture 和 wing_phase；板级传感器脚本已启动 AS5600 和 wing_phase。

此目标带有 PX4IO，RPM capture 使用 `PWM_AUX_FUNCn` 配置 FMU 侧的捕获通道；应按实际接线确认 n，不能用 MAIN 输出参数代替。

在飞控 NSH 控制台执行 `param set SDLOG_PROFILE 4097`，保存参数并重启，启用默认日志与扑翼数据集组合。已有机型、板级默认参数不作修改。若需保留其他 profile，应在现有值上按位 OR 4096；扑翼配置最后应用并覆盖同名 topic 的记录间隔。

先检查 SD 卡 `/fs/microsd/etc/logging/logger_topics.txt`：有效的自定义列表会替代 profile，需同步更新或在备份后停用。下表以默认 logger 倍率为准，间隔是记录上限，不保证发布端、调度和 SD 卡均能达到对应频率。

| 数据 | 记录间隔 | 目标频率 |
| --- | --- | --- |
| rate_ctrl_status、rate_ctrl_terms | 20 ms | 50 Hz |
| flap_frequency、wing_phase、encoder_count | 不限速 | 编码器正常时约 100 Hz |
| vehicle_angular_velocity | 10 ms | 100 Hz |
| vehicle_attitude、vehicle_attitude_setpoint、vehicle_rates_setpoint | 20 ms | 50 Hz |
| actuator_servos、vehicle_torque_setpoint、vehicle_thrust_setpoint | 20 ms | 50 Hz |
| actuator_outputs，各输出实例 | 10 ms | 最高 100 Hz，发布端需至少 50 Hz |
| airspeed、airspeed_validated | 不限速 | 原始最高 50 Hz，selector 调度 20 Hz |
| tecs_status | 50 ms | 20 Hz |
| vehicle_local_position、vehicle_local_position_setpoint、fixed_wing_longitudinal_setpoint | 20 ms | 最高 50 Hz，设定值随控制模式发布 |
| battery_status | 100 ms | 10 Hz |
| wind | 50 ms | 最高 20 Hz |

`rate_ctrl_terms` 只在固定翼 rate controller 执行更新时发布。三轴顺序为控制器坐标系的 roll/pitch/yaw，tailsitter 使用固定翼控制坐标系。`d_term` 已含负号；`i_term` 是本次输出实际使用的积分值，早于本周期积分更新。`output` 为四项之和，位于输出空速缩放、增益压缩、trim、yaw 覆盖/前馈和限幅之前，不等同于最终舵机命令。`timestamp_sample` 来自本次角速度输入。VTOL 的分配前固定翼命令还可查看 `vehicle_torque_setpoint_virtual_fw` 和 `vehicle_thrust_setpoint_virtual_fw`。

翼相位需要 AS5600 编码器、有效 `FLAP_RATIO` 和 Hall 零位。确认实际接线后，将对应 PWM 通道功能设为 RPM Input（2070），启用 `RPM_CAP_ENABLE` 并重启。这里不指定未知的通道号。用 `rpm_capture status`、`wing_phase status` 和 `listener hall_event` 检查捕获链路；首次 Hall 校准前相位为 NaN 且 `phase_valid=false` 是预期行为。仅启动 wing_phase 或提高日志频率不能替代 Hall 校准。

台架验收需覆盖扑翼运行和固定翼角速度控制工作区间：检查 `logger status`、`uorb top`，采集 ULog 后按各 topic 的 timestamp 统计采样间隔和最大间隙，检查 dropout。空速同时核对源采样时间（airspeed.timestamp_sample、airspeed_selector_quality_status.decision_timestamp_sample），排除重复发布旧测量。确认 Hall 校准后 phase_valid 为 true、相位字段有限，PID 四项之和与 output 一致。PWM 日志记录的是输出命令，不是引脚波形测量。
