# 扑翼飞行器系统辨识数据集 Contract Draft

_面向 `/home/zn/system-identification` 的第一版 supervised dataset 定义草案，版本 `v0.1-draft`，2026-03-23_

---

## 📋 目标

这份 contract 用来固定第一版数据集的定义，目标是让后续：

- 日志提取
- 标签计算
- 样本过滤
- 训练集/验证集划分
- 不同 backbone 的特征适配

都围绕同一份 **canonical dataset** 展开，而不是每换一个模型就重新发明一套数据口径。

第一版数据集的监督目标明确为：

- **机体系有效外力** `F_eff^B = [Fx, Fy, Fz]`
- **机体系有效外力矩** `M_eff^B = [Mx, My, Mz]`

也就是常说的 `wrench = force + moment`，这里统一翻成：

- **有效外力/外力矩**

## 🔗 配套文档

这份 contract 现在和下面两份配套草案一起看会更完整：

- [扑翼飞行器 Aircraft Metadata Contract Draft](./2026-03-23-flapping-aircraft-metadata-contract-draft.md)
- [ULog 到 Canonical Parquet 预处理规范 Draft](./2026-03-23-ulg-to-canonical-parquet-preprocessing-spec.md)

## 🧠 设计原则

### 1. 数据集先独立于 backbone

第一版 contract 明确分成两层：

- **Canonical dataset**
  - 保存物理量、标签、mask、质量标记
  - 不带任何某个模型专属特征工程
- **Model adapter**
  - 针对 `MLP`、`TCN`、`Transformer`、`RNN` 生成输入张量
  - 允许加 `sin/cos phase`、时间窗、标准化、delta feature

这样做的目的很直接：

- 数据定义先稳定
- 模型实验后移
- 避免“换个 backbone 就换数据口径”

### 2. 第一版只做“能从日志稳健反推”的标签

第一版不追求 pure aerodynamic force/moment，而是定义为：

- **解释当前刚体运动状态所需的净非重力外力/外力矩**

它包含的实际来源可能包括：

- 非定常气动力
- 尾翼气动力
- 扑翼推进/升力综合效应
- 扑翼机构未建模惯性反作用

这正是第一版最适合做 supervised learning 的目标。

### 3. 坐标系必须与 PX4 保持一致

本项目第一版统一使用：

- **机体系 `B`**: `FRD`
  - `x`: forward
  - `y`: right
  - `z`: down
- **局部导航系 `N`**: `NED`
  - `x`: north
  - `y`: east
  - `z`: down

这和 PX4 当前消息定义一致，因此可以避免额外坐标系歧义。

## 🏗️ Canonical Sample 定义

### 一个 sample 是什么

第一版定义一个 sample 为：

- 一个统一时间戳 `t_k`
- 在该时刻对齐后的输入状态 `x_k`
- 对应的标签 `y_k`
- 对应的质量标记 `mask_k`

第一版推荐主时间轴：

- **100 Hz**
- 采样间隔 `Δt = 0.01 s`

选择 `100 Hz` 的原因是：

- `flap_frequency` / `rpm` / `encoder_count` / `debug_vect` 原生就是 `100 Hz`
- `vehicle_local_position` / `vehicle_odometry` 也接近 `100 Hz`
- 这是当前日志中最自然的公共高价值频率
- 不会被 `actuator_*` 的 `405 Hz` 牵着走，也不会被 `airspeed/GPS` 的低频拖太慢

### 时间轴规则

统一时间轴定义为：

```text
t_k = t_start + k * 0.01 s
```

其中：

- `t_start` 取当前 flight segment 内第一帧可用的公共起点
- `t_end` 取当前 segment 内最后一帧可用的公共终点

segment 的切分规则见后文。

## 🧭 坐标与物理量约定

### 力和力矩标签

输出标签统一为机体系 `FRD`：

| 字段 | 单位 | 含义 |
| --- | --- | --- |
| `fx_b` | `N` | 机体系 forward 方向有效外力 |
| `fy_b` | `N` | 机体系 right 方向有效外力 |
| `fz_b` | `N` | 机体系 down 方向有效外力 |
| `mx_b` | `N*m` | 绕机体系 `x` 轴有效外力矩 |
| `my_b` | `N*m` | 绕机体系 `y` 轴有效外力矩 |
| `mz_b` | `N*m` | 绕机体系 `z` 轴有效外力矩 |

注意：

- 因为 `z` 是 down，所以“升力向上”在这个约定里通常表现为 **负的 `fz_b`**
- 这和日常空气动力学里 `Lift > 0` 的符号习惯不同，但和 PX4/FRD 是一致的

### 姿态与速度

第一版 canonical dataset 中同时保留：

- 机体系角速度 `ω_B`
- 局部系位置/速度 `p_N, v_N`
- 姿态四元数 `q_NB`

建议的姿态约定：

- 直接沿用 `vehicle_attitude.q`
- 其含义为 **body FRD 到 world NED 的四元数**

### 位置量的使用原则

第一版建模时：

- **平动速度、姿态、角速度、空速、扑翼相位** 是核心输入
- **绝对位置** 不是必须核心输入

因此 canonical dataset 里可以保留位置，但建议同时提供：

- `position_ned`
- `position_ned_rel_segment`

其中 `position_ned_rel_segment` 定义为当前 segment 首帧归零后的相对位置，便于模型保持平移不变性。

## 📤 输出标签 Contract

### 输出字段

第一版标签字段固定为：

| 字段组 | 字段 |
| --- | --- |
| 有效外力 | `fx_b`, `fy_b`, `fz_b` |
| 有效外力矩 | `mx_b`, `my_b`, `mz_b` |

### 标签计算所需常量

标签计算不从日志里猜，而是从单独 metadata 提供：

| 常量 | 说明 |
| --- | --- |
| `mass_kg` | 机体总质量 |
| `inertia_b` | 机体系惯量矩阵 `3x3` |
| `cg_offset_b` | 如需要，机体系重心偏置 |
| `gravity_m_s2` | 默认 `9.81` |

这些量应该放在单独的 aircraft metadata 文件里，而不是散落在 notebook 里。

### 标签计算口径

第一版推荐口径：

```text
F_eff_B = m * R_BN * (a_N - g_N)
M_eff_B = I_B * alpha_B + omega_B x (I_B * omega_B)
```

其中：

- `a_N` 来自 `vehicle_local_position.ax/ay/az`
- `g_N = [0, 0, 9.81]`，因为使用 `NED`
- `R_BN` 由姿态四元数得到
- `omega_B` 和 `alpha_B` 来自 `vehicle_angular_velocity`

这一定义的物理含义是：

- **非重力净外力**
- **绕重心的净外力矩**

第一版先不再细分为“纯气动”与“机构反作用”。

### 标签质量要求

以下任一条件不满足，则该 sample 不产生 label：

- `vehicle_local_position.xy_valid == true`
- `vehicle_local_position.z_valid == true`
- `vehicle_local_position.v_xy_valid == true`
- `vehicle_local_position.v_z_valid == true`
- `vehicle_attitude.q` 有效
- `vehicle_angular_velocity.xyz` 有效
- `vehicle_angular_velocity.xyz_derivative` 有效
- 不处于 estimator reset 附近窗口

## 📥 输入字段 Contract

第一版把输入分成五组。

### A. 扑翼运动学

当前用户已经补充了一条非常关键的机构关系：

```text
如果 φ_drive 是驱动齿轮相位
则机翼扑动角 θ_wing ≈ deg2rad(30) * sin(φ_drive)
```

同时用户已经明确：

```text
φ_drive = 0
  对应正弦横轴为 0 的位置

θ_wing 增大
  = 上扑

θ_wing 减小
  = 下扑

因此：
0°   到  90°   -> 上扑
90°  到 270°   -> 下扑
270° 到 360°   -> 上扑
```

这意味着第一版最好显式区分两个量：

- **旋转相位**
  - 这是周期变量，适合用 `sin/cos` 进入模型
- **机翼扑动角**
  - 这是由机构映射出来的几何角度，不足以单独区分上扑/下扑

也就是说，对 phase-aware 模型来说，优先保留的核心周期输入应该是：

- `drive/encoder phase`

而不是只保留：

- `wing stroke angle`

因为同一个扑动角在一个周期里通常会出现两次，但对应的速度方向不同。

| 字段 | 来源 | 说明 |
| --- | --- | --- |
| `encoder_phase_rad` | `debug_vect.x` 或 `encoder_count.position_raw` 派生 | 编码器测得的旋转相位，范围 `[0, 2π)` |
| `encoder_phase_unwrapped_rad` | `encoder_count` 派生 | 编码器展开相位，用于长期连续性 |
| `drive_phase_rad` | `encoder_phase + transmission metadata` 派生 | 驱动齿轮相位，范围 `[0, 2π)` |
| `drive_phase_sin` | 派生 | `sin(drive_phase_rad)` |
| `drive_phase_cos` | 派生 | `cos(drive_phase_rad)` |
| `wing_stroke_angle_rad` | `drive_phase` 派生 | 机翼扑动角，当前建议口径 `deg2rad(30) * sin(drive_phase_rad)` |
| `wing_stroke_angle_deg` | 派生 | 便于检查和画图 |
| `wing_stroke_direction` | `drive_phase` 或 `wing_stroke_rate` 派生 | `upstroke/downstroke`，便于分析滞回 |
| `flap_frequency_hz` | `flap_frequency` | 扑频 |
| `encoder_rpm_raw` | `rpm` | 编码器测得的原始转速 |
| `encoder_rpm_est` | `rpm` | 编码器测得的滤波转速 |
| `encoder_total_count` | `encoder_count` | 长时相位展开辅助量 |
| `encoder_position_raw` | `encoder_count` | 原始编码器位置 |

第一版模型输入建议优先使用：

- `drive_phase_sin`
- `drive_phase_cos`
- `wing_stroke_angle_rad`
- `flap_frequency_hz`
- `encoder_rpm_est`

而不是直接只用 `phase_rad`，因为 `sin/cos` 更适合处理相位环绕。

这里要特别强调一件事：

- `debug_vect.x` 当前记录的是 **编码器轴角**
- `rpm.rpm_raw / rpm.rpm_estimate` 当前记录的是 **编码器轴转速**
- 它们都不是“已经定义好的机翼 phase”

因此第一版推荐的相位/机构重建口径应写成两步：

```text
encoder_phase_unwrapped_rad
  = 2π * shaft_count_cont / encoder_counts_per_rev

drive_phase_unwrapped_rad
  = encoder_to_drive_sign * encoder_phase_unwrapped_rad / encoder_to_drive_ratio
    + drive_phase_zero_offset_rad

drive_phase_rad
  = wrap_to_2pi(drive_phase_unwrapped_rad)

wing_stroke_angle_rad
  = wing_stroke_amplitude_rad * sin(drive_phase_rad + wing_stroke_phase_offset_rad)
```

其中：

- `encoder_counts_per_rev` 对 AS5600 默认为 `4096`
- 当前用户给出的第一版机构口径是 `wing_stroke_amplitude_rad = deg2rad(30)`
- 如果编码器直接装在驱动齿轮上，则 `encoder_to_drive_ratio = 1`
- 如果编码器装在上游轴上，则历史上的 `FLAP_RATIO = 7.5` 更像是这里的 `encoder_to_drive_ratio`
- `encoder_to_drive_ratio`、`encoder_to_drive_sign`、`drive_phase_zero_offset_rad` 不能靠日志猜，必须来自 aircraft metadata
- `debug_vect.x` 更适合作为 `encoder_phase_rad` 的交叉校验量

基于用户这次确认，第一版还可以直接采用：

```text
wing_stroke_direction = sign(d/dt wing_stroke_angle_rad)
```

并在 `drive_phase_zero_offset_rad = 0` 时解释为：

- `0 < drive_phase_rad < π/2` -> `upstroke`
- `π/2 < drive_phase_rad < 3π/2` -> `downstroke`
- `3π/2 < drive_phase_rad < 2π` -> `upstroke`

### B. 控制输入

| 字段 | 来源 | 说明 |
| --- | --- | --- |
| `motor_cmd_0` | `actuator_motors.control[0]` | 主扑翼驱动归一化命令 |
| `servo_left_elevon` | `actuator_servos.control[0]` | 左平尾 |
| `servo_right_elevon` | `actuator_servos.control[1]` | 右平尾 |
| `servo_rudder` | `actuator_servos.control[2]` | 垂尾 |

由于 `actuator_*` 原生约 `405 Hz`，而主时间轴是 `100 Hz`，第一版 canonical 聚合方式建议为：

- 对每个 `10 ms` bin 取 **均值**

可选附加字段：

- `motor_cmd_0_last`
- `servo_*_last`
- `motor_cmd_0_std`
- `servo_*_std`

但这些先不进入 `v0.1` 必选字段。

### C. 刚体状态

| 字段 | 来源 | 说明 |
| --- | --- | --- |
| `q_nb_wxyz` | `vehicle_attitude` | body->NED 姿态四元数 |
| `roll`, `pitch`, `yaw` | 派生 | 便于可视化和统计 |
| `p_b`, `q_b`, `r_b` | `vehicle_angular_velocity` | 机体系角速度 |
| `pdot_b`, `qdot_b`, `rdot_b` | `vehicle_angular_velocity` | 机体系角加速度 |
| `x_n`, `y_n`, `z_n` | `vehicle_local_position` / `vehicle_odometry` | 局部位置 |
| `vx_n`, `vy_n`, `vz_n` | `vehicle_local_position` / `vehicle_odometry` | 局部速度 |
| `ax_n`, `ay_n`, `az_n` | `vehicle_local_position` | 局部加速度 |
| `vx_b`, `vy_b`, `vz_b` | 派生 | 由姿态把 NED 速度转到 body FRD |

第一版建模建议优先使用：

- `q_nb_wxyz`
- `p_b`, `q_b`, `r_b`
- `vx_b`, `vy_b`, `vz_b`

这样输入更贴近气动力的机体系依赖。

### D. 气动环境

| 字段 | 来源 | 说明 |
| --- | --- | --- |
| `ias_m_s` | `airspeed_validated` | indicated airspeed |
| `cas_m_s` | `airspeed_validated` | calibrated airspeed |
| `tas_m_s` | `airspeed_validated` | true airspeed |
| `airspeed_source` | `airspeed_validated` | 当前日志里全程为 sensor 1 |
| `rho_kg_m3` | `vehicle_air_data` | 空气密度 |
| `wind_n`, `wind_e`, `wind_d` | `wind` | NED 风估计 |
| `airspeed_q` | `ekf2_airspeed_quality` | 空速质量 |
| `fuse_enabled` | `ekf2_airspeed_quality` | 空速是否允许融合 |
| `flap_active_flag` | `ekf2_airspeed_quality` | 当前日志中基本无效，但保留 |

低频 topic 的对齐规则：

- `airspeed_validated`、`wind`、`vehicle_air_data` 采用 **ZOH**
- 同时保留一个 `age_s` 字段
- 超过 freshness 阈值则置 `valid = false`

推荐 freshness 阈值：

- airspeed: `0.2 s`
- wind: `0.3 s`

### E. 模式与质量标签

| 字段 | 来源 | 说明 |
| --- | --- | --- |
| `nav_state` | `vehicle_status` | 模式标签 |
| `armed` | `vehicle_status` | 是否解锁 |
| `landed` | `vehicle_land_detected` | 是否已着陆 |
| `allocator_torque_ok` | `control_allocator_status` | torque 是否分配成功 |
| `allocator_thrust_ok` | `control_allocator_status` | thrust 是否分配成功 |
| `actuator_saturation_mask` | `control_allocator_status` | 执行器饱和信息 |
| `gps_fix_type` | `sensor_gps` | 绝对定位质量 |
| `relative_gnss_valid` | `sensor_gnss_relative` | 当前日志中通常为假 |
| `segment_id` | 派生 | 当前连续飞行段编号 |

这些字段原则上不一定全部进模型，但必须进 canonical dataset。

## ⏱️ 时间对齐与重采样规则

### 统一规则

不同信号按以下方式对齐到 `100 Hz` 主时间轴：

| 信号类型 | 对齐方式 |
| --- | --- |
| 100 Hz 编码器/扑频 | nearest 或直接匹配 |
| 高频 actuator | bin mean |
| 连续状态量 | 线性插值 |
| 四元数 | `slerp` 后归一化 |
| 低频 airspeed/wind/GPS | ZOH + age |
| 离散 mode / flag | ZOH |

### 不允许的简化

第一版不建议做下面这些偷懒操作：

- 直接按最近邻对齐所有连续量
- 对四元数逐元素线性插值后不归一化
- 对低频 airspeed/GPS 不记录 age 就直接硬插值

因为这些会把时间对齐误差偷偷塞进标签噪声里。

## ✂️ Segment 与样本过滤

### Segment 切分规则

一条 log 不直接视为一个连续样本流，而是先切成多个 `segment`。

出现以下任一事件时切段：

- `nav_state` 变化
- `armed/disarmed` 变化
- `landed` 状态变化
- local position / heading / attitude reset counter 变化
- logger dropout 超过阈值
- 关键 topic freshness 超时

### Hard filter

以下条件不满足则样本直接剔除：

- `armed == true`
- `landed == false`
- `nav_state` 属于 `{MANUAL, STAB, AUTO_MISSION}`
- 扑翼运动学字段有效
- 姿态、角速度、局部速度、局部加速度有效
- 必要的 airspeed 与 density 字段有效
- 标签可计算

### Soft flag

以下条件不一定删样本，但必须打标：

- `allocator_torque_ok == false`
- 任一 actuator saturation 非零
- `gps_fix_type < 6`
- `relative_gnss_valid == false`
- `airspeed_q < threshold`
- `fuse_enabled == false`

建议第一版先生成两个 dataset 视图：

- `clean_v1`
  - 只保留 hard filter 通过且 soft flag 较好的样本
- `full_v1`
  - 保留全部 hard filter 通过样本，并把 soft flag 留给训练时使用

## 🗂️ 文件结构建议

第一版建议每条 log 处理后输出两类文件：

```text
dataset/
  logs/
    2026-03-22-log_6_good/
      samples.parquet
      segments.parquet
      metadata.json
      quality_report.json
```

其中：

- `samples.parquet`
  - 每一行一个 `100 Hz` sample
- `segments.parquet`
  - 每个 segment 的起止、模式、长度、质量摘要
- `metadata.json`
  - 日志路径、飞机参数、contract version、预处理配置
- `quality_report.json`
  - 过滤掉多少样本、每类 mask 的统计

## 🔌 Model Adapter 约定

为了保持 dataset 和 model 解耦，后续模型只能从 canonical dataset 派生，不允许反过来改 canonical 字段定义。

### MLP adapter

- 单时刻输入
- 输入为 `x_k`
- 输出为 `y_k`

### TCN / RNN / Transformer adapter

- 时间窗输入
- 例如 `x_{k-T+1:k}`
- 输出 `y_k` 或 `y_{k:k+H}`

### 允许的派生特征

- 标准化
- 相位 `sin/cos`
- control delta
- velocity magnitude
- angle of attack surrogate
- sideslip surrogate
- low-pass / high-pass 分量

这些都属于 **adapter 层**，不改 canonical contract。

## ⚠️ 当前仍未定死的内容

当前还需要后续确认的只有少数几项：

1. `mass_kg` 与 `inertia_b`
   - 没有它们就没法稳定产出标签
2. airspeed 是否作为硬依赖
   - 如果某些日志空速差，也许要允许生成 “no-airspeed variant”
3. 是否把绝对位置放进第一版模型输入
   - 我倾向于 canonical 保留，baseline 默认不用
4. 是否保留 `control_allocator_status` 不达标样本
   - 我倾向于先保留为 soft flag，同时提供 clean 版本
5. 机翼 phase 零点与正方向定义
   - 这是 phase-aware 模型跨日志泛化的前提

## 📝 当前建议

如果按这个 draft 往下走，下一步最合理的是：

1. 按 [Aircraft Metadata Contract Draft](./2026-03-23-flapping-aircraft-metadata-contract-draft.md) 先填出第一版 `aircraft_metadata.yaml`
   - `mass`
   - `inertia`
   - `cg`
   - `phase zero/sign`
2. 按 [ULog 到 Canonical Parquet 预处理规范 Draft](./2026-03-23-ulg-to-canonical-parquet-preprocessing-spec.md) 实现 `ulog -> canonical parquet` 原型脚本
3. 用当前 good log 先跑通一条端到端样例，检查
   - 时间对齐
   - phase 重建
   - `effective wrench` 标签量级
4. 最后再讨论 baseline backbone

这样顺序是对的，因为 backbone 不该反过来决定数据集口径。
