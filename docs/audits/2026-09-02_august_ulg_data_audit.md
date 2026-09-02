# 2026 年 8 月 ULG 数据审计（Step 0）

生成时间：`2026-09-02T16:58:25.054778+08:00`
审计版本：`august_ulg_audit_v1`
QgcLogs HEAD：`9db63ebac045ea93dd7144d47bb235e9a02df13f`

## 结论

共识别 44 条 2026 年 8 月 ULG；42 条可解析，2 条损坏或不可解析，26 条达到自动准入线。达到准入线的日志提供 3818.4 s 核心 trajectory-ready 数据，其中 3561.6 s 同时具有新鲜空速和风估计。有效 logged mechanical phase 仅 0.0 s；另有 3817.3 s 只具备 encoder 相对 phase 重建条件。

这些数据足以进入下一阶段的数据契约与基线设计，但不能直接合并训练：月份内存在明确的飞控硬件、固件和结构参数分组，必须先固定单一同构 cohort，并保持整日志、整日期的 split 边界。

## 范围与方法

- 扫描根目录：`/home/zn/QgcLogs`。按文件名时间选择 2026-08；另有 237 条非 8 月 ULG 被显式排除。8 月命名目录内误放的非 8 月文件为 `2026.8.10-8.20/log_6_2026-4-15-10-48-46.ulg`。
- 事件时间在 `timestamp_sample` 有效时优先使用；全零等无效值显式回退到 `timestamp`。可用时长在 `vehicle_local_position` 时间轴上积分，仅计入 armed、airborne、状态有效且姿态/角速度/电机/三个舵面同时新鲜的相邻区间；gap 上限为 50 ms 与 2.5 倍原生状态周期中的较大者。
- logged mechanical phase、encoder 相对 phase、空速、风和 RTK 均单独报告覆盖率；它们不作为核心 trajectory 准入硬条件。机动标签是审计用运动学代理，不是监督真值。
- 自动准入线为状态采样率至少 40 Hz、核心可用时长至少 60 s，且不少于 airborne 时长的 75%。

## 固件、硬件与 schema 变化

可解析日志形成 6 个固件/硬件组、5 个结构参数组和 4 个关键 topic schema 组。

| 固件组 | ver_sw | 硬件 | subtype | 飞控 UUID | 日志数 | 状态 Hz | actuator Hz |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| F1 | `54d8e5a69bdf1472e3c445c034f3853ff0db083f` | PX4_FMU_V6C | V6C002002 | `000600000000393039303533510b00430031` | 1 | 10.0 | 99.8 |
| F2 | `275ee3ce44432ed1cc9b7c3266d3b696ddeccaf8` | PX4_FMU_V6C | V6C002002 | `000600000000393039303533510b00430031` | 4 | 10.0 | 99.8 |
| F3 | `e44285af8dd9d066e27fddd4ec8975b619bb530f` | PX4_FMU_V6C | V6C002001 | `000600000000353438323233510d00310036` | 7 | 50.1 | 50.1 |
| F4 | `d5d881421891f7a706a48b3b8ce07303f406d055` | PX4_FMU_V6C | V6C002001 | `000600000000353438323233510d00310036` | 8 | 50.1 | 50.1 |
| F5 | `73b0886239d256deb70df3b56fa4e4714438d52e` | PX4_FMU_V6C | V6C002001 | `000600000000353438323233510d00310036` | 21 | 50.1 | 50.1 |
| F6 | `bd6c81c457c4c93ece40a192991c6c97a5445432` | PX4_FMU_V6C | V6C002001 | `000600000000353438323233510d00310036` | 1 | 50.1 | 50.1 |

| 参数组 | 日志数 | FLAP_RATIO | FW_USE_AIRSPD | AIRSPD trim | pitch offset | allocation pitch gain | PWM reverse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | 5 | 8.0 | 1 | 15.0 | 0.0 | 0.5 | 0 |
| C2 | 7 | 8.0 | 1 | 6.0 | 15.0 | 0.5 | 17 |
| C3 | 2 | 8.0 | 1 | 8.0 | 15.0 | 1.0 | 17 |
| C4 | 24 | 7.909090995788574 | 1 | 8.0 | 15.0 | 1.0 | 17 |
| C5 | 4 | 7.909090995788574 | 0 | 8.0 | 15.0 | 1.0 | 17 |

| schema 组 | 日志数 | 关键 topic 数 | 缺失的审计 topic |
| --- | ---: | ---: | --- |
| S1 | 4 | 18 | `hall_event`, `sensor_gnss_relative`, `sensor_gps`, `vehicle_global_position` |
| S2 | 2 | 17 | `hall_event`, `position_setpoint_triplet`, `sensor_gnss_relative`, `sensor_gps`, `vehicle_global_position` |
| S3 | 34 | 21 | `hall_event` |
| S4 | 2 | 20 | `hall_event`, `position_setpoint_triplet` |

结构参数变化不能视为普通日志噪声：`FLAP_RATIO`、固定翼空速目标、pitch offset、control allocation 增益以及 PWM 范围/反向均出现在审计参数集中；逐日志旧值/新值见机器汇总的 `selected_parameter_changes_from_previous`。`ASPD_SCALE_1` 属于逐次标定量，保留报告但不用于结构参数分组。

## 信号覆盖

| 信号 | 可解析日志 topic 覆盖 | eligible 核心时段有效覆盖 | 角色 |
| --- | ---: | ---: | --- |
| `vehicle_local_position` | 42/42 (100.0%) | 100.0% | 位置/速度与主时间轴 |
| `vehicle_attitude` | 42/42 (100.0%) | 100.0% | 姿态 |
| `vehicle_angular_velocity` | 42/42 (100.0%) | 100.0% | 角速度 |
| `actuator_motors` | 42/42 (100.0%) | 100.0% | 扑翼主驱动控制 |
| `actuator_servos` | 42/42 (100.0%) | 100.0% | 三个舵面控制 |
| `wing_phase` | 42/42 (100.0%) | 0.0% | logged mechanical phase |
| `encoder_count` | 42/42 (100.0%) | 100.0% | 仅相对 encoder phase 重建 |
| `flap_frequency` | 42/42 (100.0%) | 100.0% | 扑翼频率 |
| `airspeed_validated` | 42/42 (100.0%) | 93.5% | 空速（附加） |
| `wind` | 42/42 (100.0%) | 100.0% | 风估计（附加） |
| `sensor_gps` | 36/42 (85.7%) | 90.4% | GPS/RTK fix（附加） |
| `sensor_gnss_relative` | 36/42 (85.7%) | 0.0% | 相对 GNSS（附加） |

## 飞行与控制覆盖

达到准入线的日志按互斥运动学代理累计：

| 状态 | 时长 s |
| --- | ---: |
| `stable_level` | 468.8 |
| `climb` | 434.7 |
| `descent` | 313.6 |
| `turn` | 1908.5 |
| `transition` | 692.8 |

下表是 eligible 日志的逐日志 p01/p99 外包络，避免少量单点极值主导范围判断：

| 变量 | 最低 per-log p01 | 最高 per-log p99 | 单位 |
| --- | ---: | ---: | --- |
| `ground_speed_m_s` | 1.137 | 14.535 | m/s |
| `vertical_velocity_ned_m_s` | -2.576 | 3.185 | m/s |
| `roll_deg` | -40.883 | 39.562 | deg |
| `pitch_deg` | -3.149 | 36.173 | deg |
| `yaw_rate_deg_s` | -58.296 | 57.802 | deg/s |
| `motor_control_0` | 0.097 | 0.980 | normalized |
| `servo_control_0` | -0.764 | 1.000 | normalized |
| `servo_control_1` | -0.861 | 1.000 | normalized |
| `servo_control_2` | -0.700 | 0.946 | normalized |
| `flap_frequency_hz` | 0.857 | 5.251 | Hz |
| `true_airspeed_m_s` | -15.668 | 16.294 | m/s |
| `wind_north_m_s` | -3.045 | 3.400 | m/s |
| `wind_east_m_s` | -5.529 | 2.478 | m/s |

每条日志的 min/p01/p50/p99/max 位于机器汇总的 `flight_analysis.ranges`。该范围可用于下一阶段定义状态/控制归一化和 OOD 边界，但统计量不得从 sealed test 拟合。

## 逐日志准入

| 日期时间 | 日志 | 记录 s | airborne s | 核心可用 s | logged phase s | encoder 相对 phase s | 状态 | split |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 2026-08-10T18:29:08 | `log_0_2026-8-10-18-29-08.ulg` | 150.9 | 150.7 | 139.3 | 0.0 | 139.1 | `review_low_state_rate` | `exclude_or_review` |
| 2026-08-10T19:44:38 | `log_1_2026-8-10-19-44-38.ulg` | 193.7 | 192.4 | 109.0 | 0.0 | 106.6 | `review_short_or_incomplete` | `exclude_or_review` |
| 2026-08-10T20:02:16 | `log_3_2026-8-10-20-02-16.ulg` | 70.1 | 69.0 | 1.3 | 0.0 | 0.0 | `exclude_short_or_incomplete` | `exclude_or_review` |
| 2026-08-10T20:06:14 | `log_4_2026-8-10-20-06-14.ulg` | 141.0 | 79.2 | 17.1 | 0.0 | 17.1 | `exclude_short_or_incomplete` | `exclude_or_review` |
| 2026-08-10T20:09:48 | `log_5_2026-8-10-20-09-48.ulg` | 140.6 | 139.3 | 40.5 | 0.0 | 40.5 | `review_short_or_incomplete` | `exclude_or_review` |
| 2026-08-15T19:33:10 | `log_209_2026-8-15-19-33-10.ulg` | 163.7 | 146.4 | 140.9 | 0.0 | 140.4 | `eligible` | `ood_holdout` |
| 2026-08-15T20:17:48 | `log_0_2026-8-15-20-17-48.ulg` | 221.9 | 71.1 | 34.1 | 0.0 | 34.1 | `review_short_or_incomplete` | `exclude_or_review` |
| 2026-08-16T14:23:36 | `log_11_2026-8-16-14-23-36.ulg` | 77.3 | 49.3 | 37.7 | 0.0 | 37.3 | `review_short_or_incomplete` | `exclude_or_review` |
| 2026-08-16T15:12:08 | `log_12_2026-8-16-15-12-08.ulg` | 207.2 | 62.3 | 48.4 | 0.0 | 48.2 | `review_short_or_incomplete` | `exclude_or_review` |
| 2026-08-16T15:39:32 | `log_13_2026-8-16-15-39-32.ulg` | 276.1 | 203.2 | 193.8 | 0.0 | 193.4 | `eligible` | `ood_holdout` |
| 2026-08-16T16:47:52 | `log_9_2026-8-16-16-47-52.ulg` | 302.3 | 110.4 | 99.1 | 0.0 | 99.0 | `eligible` | `ood_holdout` |
| 2026-08-16T23:02:42 | `log_15_2026-8-16-23-02-42.ulg` | 199.2 | 193.9 | 138.7 | 0.0 | 138.7 | `review_short_or_incomplete` | `exclude_or_review` |
| 2026-08-16T23:41:22 | `log_0_2026-8-16-23-41-22.ulg` | - | - | - | - | - | `exclude_corrupt_or_unparseable` | `exclude_or_review` |
| 2026-08-17T15:46:12 | `log_12_2026-8-17-15-46-12.ulg` | 110.0 | 37.9 | 31.3 | 0.0 | 31.1 | `review_short_or_incomplete` | `exclude_or_review` |
| 2026-08-17T15:51:16 | `log_13_2026-8-17-15-51-16.ulg` | 296.9 | 77.1 | 33.3 | 0.0 | 33.3 | `review_short_or_incomplete` | `exclude_or_review` |
| 2026-08-17T18:31:02 | `log_21_2026-8-17-18-31-02.ulg` | 143.6 | 77.2 | 68.7 | 0.0 | 68.7 | `eligible` | `ood_holdout` |
| 2026-08-17T18:46:38 | `log_22_2026-8-17-18-46-38.ulg` | 233.7 | 138.2 | 130.8 | 0.0 | 130.8 | `eligible` | `ood_holdout` |
| 2026-08-17T19:06:20 | `log_23_2026-8-17-19-06-20.ulg` | - | - | - | - | - | `exclude_corrupt_or_unparseable` | `exclude_or_review` |
| 2026-08-17T21:31:42 | `log_24_2026-8-17-21-31-42.ulg` | 99.2 | 81.4 | 77.5 | 0.0 | 77.5 | `eligible` | `ood_holdout` |
| 2026-08-18T18:47:12 | `log_30_2026-8-18-18-47-12.ulg` | 293.0 | 177.1 | 161.4 | 0.0 | 161.4 | `eligible` | `ood_holdout` |
| 2026-08-18T19:31:38 | `log_32_2026-8-18-19-31-38.ulg` | 234.6 | 164.8 | 152.0 | 0.0 | 152.0 | `eligible` | `ood_holdout` |
| 2026-08-18T21:04:28 | `log_36_2026-8-18-21-04-28.ulg` | 374.9 | 177.7 | 144.4 | 0.0 | 144.4 | `eligible` | `ood_holdout` |
| 2026-08-19T06:27:52 | `log_14_2026-8-19-06-27-52.ulg` | 182.4 | 174.3 | 170.8 | 0.0 | 170.8 | `eligible` | `train` |
| 2026-08-19T06:52:42 | `log_15_2026-8-19-06-52-42.ulg` | 817.8 | 180.5 | 174.2 | 0.0 | 174.2 | `eligible` | `train` |
| 2026-08-19T07:03:30 | `log_18_2026-8-19-07-03-30.ulg` | 324.1 | 198.3 | 171.9 | 0.0 | 171.9 | `eligible` | `train` |
| 2026-08-19T07:11:20 | `log_19_2026-8-19-07-11-20.ulg` | 264.8 | 199.4 | 193.2 | 0.0 | 193.2 | `eligible` | `ood_holdout` |
| 2026-08-19T17:20:36 | `log_1_2026-8-19-17-20-36.ulg` | 206.8 | 160.1 | 136.7 | 0.0 | 136.7 | `eligible` | `train` |
| 2026-08-19T17:28:38 | `log_2_2026-8-19-17-28-38.ulg` | 171.5 | 68.1 | 48.9 | 0.0 | 47.4 | `review_short_or_incomplete` | `exclude_or_review` |
| 2026-08-19T17:31:40 | `log_4_2026-8-19-17-31-40.ulg` | 119.2 | 12.0 | 5.5 | 0.0 | 5.5 | `exclude_short_or_incomplete` | `exclude_or_review` |
| 2026-08-19T18:31:42 | `log_5_2026-8-19-18-31-42.ulg` | 170.0 | 126.2 | 115.5 | 0.0 | 115.5 | `eligible` | `train` |
| 2026-08-19T18:40:42 | `log_6_2026-8-19-18-40-42.ulg` | 174.4 | 134.9 | 125.5 | 0.0 | 125.5 | `eligible` | `train` |
| 2026-08-19T18:53:36 | `log_7_2026-8-19-18-53-36.ulg` | 316.4 | 175.3 | 123.0 | 0.0 | 123.0 | `review_short_or_incomplete` | `exclude_or_review` |
| 2026-08-19T19:01:44 | `log_8_2026-8-19-19-01-44.ulg` | 200.3 | 131.9 | 41.7 | 0.0 | 41.7 | `review_short_or_incomplete` | `exclude_or_review` |
| 2026-08-20T06:08:16 | `log_15_2026-8-20-06-08-16.ulg` | 213.5 | 168.9 | 159.9 | 0.0 | 159.9 | `eligible` | `validation` |
| 2026-08-20T06:20:52 | `log_16_2026-8-20-06-20-52.ulg` | 198.2 | 160.0 | 150.9 | 0.0 | 150.9 | `eligible` | `validation` |
| 2026-08-20T06:29:34 | `log_17_2026-8-20-06-29-34.ulg` | 306.1 | 208.7 | 194.9 | 0.0 | 194.9 | `eligible` | `validation` |
| 2026-08-20T06:42:46 | `log_18_2026-8-20-06-42-46.ulg` | 197.9 | 143.5 | 132.1 | 0.0 | 132.1 | `eligible` | `ood_holdout` |
| 2026-08-20T06:51:18 | `log_19_2026-8-20-06-51-18.ulg` | 260.0 | 163.3 | 157.8 | 0.0 | 157.8 | `eligible` | `validation` |
| 2026-08-20T06:59:54 | `log_20_2026-8-20-06-59-54.ulg` | 205.6 | 165.4 | 159.7 | 0.0 | 159.7 | `eligible` | `ood_holdout` |
| 2026-08-20T07:09:54 | `log_22_2026-8-20-07-09-54.ulg` | 211.4 | 171.0 | 160.1 | 0.0 | 160.0 | `eligible` | `validation` |
| 2026-08-26T05:48:12 | `2026-08-26_054812_log12.ulg` | 189.9 | 146.1 | 121.0 | 0.0 | 121.0 | `eligible` | `sealed_test` |
| 2026-08-26T06:06:14 | `2026-08-26_060614_log09.ulg` | 202.5 | 170.7 | 161.7 | 0.0 | 161.7 | `eligible` | `sealed_test` |
| 2026-08-26T06:30:46 | `2026-08-26_063046_log13.ulg` | 339.5 | 180.2 | 164.2 | 0.0 | 164.2 | `eligible` | `sealed_test` |
| 2026-08-27T18:56:46 | `log_11_2026-8-27-18-56-46.ulg` | 356.8 | 154.2 | 113.0 | 0.0 | 113.0 | `review_short_or_incomplete` | `exclude_or_review` |

## 主要质量问题与风险

- 两条文件不可用于任何建模：`2026.8.10-8.20/log_0_2026-8-16-23-41-22.ulg` (invalid ULog magic: 00000000000000); `2026.8.10-8.20/log_23_2026-8-17-19-06-20.ulg` (error: unpack requires a buffer of 3 bytes)。
- 5 条 8/10 日志的主状态约 10 Hz，而后续批次约 50 Hz；它们可保留作低速率/OOD 研究，但不能直接混入主 cohort。
- 全部可解析日志虽含 `wing_phase` topic，但 eligible 核心时段的有效 logged phase 为 0%；`encoder_count` 只支持相对相位重建，且所有日志均无 `hall_event`。如果 Step 1 需要跨日志一致 mechanical phase，当前数据链尚未闭合。
- eligible 中空速有效覆盖低于 80% 的日志有 3 条；出现负 `true_airspeed_m_s` 的 eligible 日志有 2 条；舵面命令接近归一化边界超过 5% 的日志有 1 条。名单在机器汇总 `quality_summary`。
- 最大 ULog dropout 为 `2026.8.10-8.20/2026-08-26_060614_log09.ulg` 的 0.282 s；所有窗口生成仍需按原始 gap 切断。
- 关键 topic 共检测到 4914 个重复时间戳、165 个倒序时间戳；重复全部来自 `manual_control_setpoint`，倒序主要来自无效的 `sensor_gnss_relative`。20 条日志在全记录范围出现 estimator reset counter 变化。
- 损坏文件、缺 topic、短航段和低同步覆盖均按显式状态保留，不静默回退。每个 topic 的采样率、finite 比例、重复/倒序时间戳、p99/max gap、发布延迟以及 ULog dropout 见机器汇总。
- `time_ref_utc` 在抽取的 ULog 元数据中不能提供可靠绝对 UTC；本审计以文件名时间组织 session，不把文件名当传感器级同步证据。`sensor_gps.timestamp_sample` 在本批次为全零，审计显式回退到其 `timestamp`；下游通用 event-time 逻辑必须覆盖这一边界。
- `wind` 与 `airspeed_validated` 是估计器/融合输出，未来模型若把它们同时作为输入和评估真值，可能引入闭环或 estimator leakage；应在 Step 1 明确因果可用性和部署时可得性。
- 禁止随机拆 sample/window。同一次飞行、相邻 session 和相同标定必须整体进入一个 split；任何 causal window 不得跨日志、模式变化、reset 或大 gap。
- 本次允许查看 sealed-test 候选仅用于输入完整性和覆盖审计。一旦接受以下划分，就冻结日志名单，不得再用 sealed test 的分布、轨迹或指标选择特征、阈值或模型。

## 推荐划分

自动策略选择数据量最大的完全一致固件+结构参数 cohort：`{'firmware_group': 'F5', 'configuration_group': 'C4'}`。其中 `2026-08-26` 整日作为 sealed test，`2026-08-20` 整日作为 validation，更早日期作为 train；其余达到准入线但配置不同的日志只作为 OOD holdout，不混入主模型。

- train：6 条，894.5 s
- validation：5 条，823.5 s
- sealed test：3 条，446.9 s
- OOD holdout：12 条，1653.5 s

机器可读日志名单以 JSON 中的 `recommended_splits` 为准。若 Step 1 决定显式建模硬件/固件/参数条件，应另立数据契约并重新冻结 split，不能直接把 OOD holdout 并入训练。

## Step 1 前建议

1. 冻结主 cohort、整日志 split 和 exclusion reasons，sealed test 只保存清单，不生成训练统计。
2. 明确 trajectory state、control、预测 horizon、frame 和 causal availability；特别决定是否使用估计风/空速。
3. 在 canonical 生成前增加 reset/gap guard，并验证 actuator 命令语义在各 schema/配置组内一致。
4. 先做无训练的数据窗口计数与覆盖检查；本 Step 0 不训练、不拟合归一化，也不构造未来标签。
