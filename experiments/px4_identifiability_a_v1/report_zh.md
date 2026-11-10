# 路线 A：外生激励与工具变量，train-only

结论：**NO-GO：本轮没有找到可用于差动尾舵 plant 控制增益监督的可信 IV。** 已完成实际日志筛选，不进行 2SLS 或探索性因果响应估计。此结论是当前候选与现有来源证据未通过门槛，并非证明这些实飞数据永远不可能辨识。

## 数据和估计对象

从 CRM-v1 `9c15fffd24bfcacc7c57cd980cc149098cb83f82` 建立独立工作树。使用主 Agent 合同中的 velocity_rate_multistep_v1 train：114 flights/114 ULogs、17 日期、11221 个既有 A2/CRM 窗口；未访问 validation/test，未训练神经网络。读取原始日志仅在已核定 train 列表内。15 个日志固件的精确 git 对象均存在于只读 `/home/zn/PX4-Autopilot`，本轮 10 条源码路径逐 SHA 检查，完整证据见 `source_evidence.json`。

本轮第一阶段因变量 U 为 `[t,t+0.1s]` 的记录差动命令 ZOH 平均减去 t 前基线；差动 `d=(servo[1]-servo[0])/2`，单位为 PX4 normalized command，不是舵面角。其他控制对应 motor、symmetric、rudder 的相同增量。参考候选 Z 是 `[t-0.1s,t]` 的参考变化，按发布 timestamp 左侧保持获取，最大 age 0.5s；航点为有效坐标记录保持、变化指示。参考与控制窗口不重叠，Z 在 t 已可知，但这不代表外生。

结果只说明 **O：闭环观察性控制可预测性**。没有识别 **R：外部参考干预总响应** 或 **P：固定差动控制干预 plant 响应**；因此不报告可与 B/C 或 PACLD 控制导数比较的响应增益。共同主目标仍为 FRD p 的原生约 0.2s 增量，本路线没有重新插值该目标，也没有进入响应估计阶段。

## 来源追踪与候选判定

以下代表行号来自实飞 SHA `c9af571575af33d5403fbb479256848797c801e2`；其他 SHA 的对应行号、内容和缺失情况由机器可读文件逐一记录，而不是引用当前 HEAD。

- **任务航点坐标/任务序列**：坐标可以预先规划，但 `src/modules/navigator/mission_base.cpp:335–345` 先调用 `is_mission_item_reached_or_completed()` 再 `advance_mission()`；`mission_block.cpp:176–180` 使用实时全球位置计算到航点距离。因而切换时刻是状态触发，不能把既定坐标顺序等同于独立激励时刻。这里未建立某个时间触发任务条目的独立来源证明。
- **自动飞行姿态参考**：`src/modules/fw_mode_manager/FixedWingModeManager.cpp:833–843` 把当前位置、地速、风送入航点导航；`fw_lateral_longitudinal_control/FwLateralLongitudinalControl.cpp:278–299` 由当前空速方向误差生成横向加速度和 roll setpoint。参考本身已经包含运动反馈和风估计，并可能联动 TECS/速度/推力。
- **roll-rate reference**：`fw_att_control/fw_roll_controller.cpp:67–74` 有 `roll_error=roll_setpoint-roll`、除以时间常数、pitch/yaw 耦合及限幅；`FixedwingAttitudeControl.cpp:304–312,345–351` 调用控制器并发布 rates setpoint。它不是上游独立命令。
- **手动 roll 指令**：`FixedwingAttitudeControl.cpp:98–111` 由 manual roll 乘角度限制生成姿态。日志证明信号来自操纵通道，无法证明飞手没有根据运动或风作反馈；没有随机时序、预先固定激励脚本或独立操作记录。故独立性 **UNRESOLVED**，不能提升为 IV。
- **vehicle_command**：日志中有速度改变及云台/相机命令。`msg/versioned/VehicleCommand.msg` 定义 178 为 change speed，521/1000/1001/2003 为相机/云台相关；`FixedWingModeManager.cpp:140–151` 把 178 写入空速参考。速度指令改变推进/空速状态，不能在排除限制未建立时用作纯差动尾舵 IV；相机/云台命令没有已验证差动第一阶段。本轮 reviewed 区间134条速度命令全部 `from_external=false` 且 `param2=8 m/s`，未见可用于横向激励的速度设定水平变化。`from_external` 仅代表消息来源，不证明统计外生性。
- **最终尾舵**：rates reference→含当前角速度输入的 `fw_rate_control/FixedwingRateControl.cpp`→torque setpoint→`control_allocator/ControlAllocator.cpp` 的 allocation→`ActuatorEffectivenessControlSurfaces.cpp:118–121,205` 参数化控制面矩阵→actuator_servos。B 路线负责内部重建。这里仅把 actuator_servos 当记录命令，未假定它是执行器实际位置。

## 定向数据实验

固定 alpha=10 的标准化线性 ridge，仅 train 内按日期留一，X 使用既有 A2 的 24 维当前状态及控制历史，不使用参考候选或未来状态；比较 X 与 [X,Z] 对未来记录命令 U 的预测误差。未调参，未假设窗口独立，报告 17 日期的描述性稳定性而不将窗口量当有效试验数。使用 numpy 求解（指定 Python 没有 sklearn；未修改依赖）。partial corr 是外折 residual 的相关，不是弱 IV 检验 F，也不是因果增益。

|候选 Z|有效窗口 / 非零变化|差动 partial corr|LODO MSE 相对降低|日期正相关|日期 MSE 改善|
|---|---:|---:|---:|---:|---:|
|roll-rate reference 变化|11221 / 10723|0.3634|9.06%|12/17|6/17|
|roll attitude reference 变化|11221 / 9629|0.4280|18.08%|17/17|16/17|
|manual roll 变化|11221 / 1433|0.0547|0.184%|12/17|10/17|
|航点坐标变化指示|6928 / 35|0.0479|−0.088%|9/17|8/17|

角速度参考逐日期相关范围 −0.0732 至 0.7172，姿态参考 0.1966 至 0.6815，manual −0.0884 至 0.1797，航点 −0.3239 至 0.3047。姿态参考第一阶段最稳定，但源码恰恰证明其反馈路径；不能以第一阶段好倒推有效 IV。

完整 reviewed 区间里检测到 917 次有效航点坐标变化，稀疏共同窗口只捕获 35 次。此计数是来源/覆盖统计，不把917次视为独立干预，也不因35次少便偷偷扩展新队列做响应拟合。

## 提前运动与其他通道

辅助提前运动量 W=`p_asof(t-0.1s)-p_asof(t-0.3s)`，角速度读取发布时刻可用记录、age≤50ms；这是严格早于参考变化区间的描述性运动量，不替换共同合同原生 0.2s 目标。roll-rate / attitude / manual / waypoint 对 W 的 residual corr 分别 0.0901 / 0.1014 / 0.0426 / −0.00155；对应 OOF MSE 改善 0.501% / 0.931% / 0.115% / −0.0219%。

**限制**：条件 X 在 t 取得、含 earlier motion 的后代；因此该 residual 检查不能证明独立性，甚至不能单独确证混杂方向。raw correlation 分别为 −0.1076 / −0.03749 / 0.00275 / −0.00765，同时保存在 CSV；条件化导致符号改变正体现不能过度解释这项检验。对于源代码已明确反馈的候选，它提供相容的描述性线索；航点提前相关接近零也不能推翻状态触发的源码反证。无合格独立事件来源时没有进一步包装“通过预趋势”的因果估计。

多通道同一筛选：roll-rate reference 对 symmetric/rudder 的 partial corr 为 −0.0990/−0.1000，OOF MSE 改善 0.609%/0.456%；attitude reference 对 rudder 为 0.0868，改善0.578%；manual 对 rudder 为0.0496，改善0.232%。motor 增量改善很小。它们仍是关联，不能认定确定因果驱动，但足以阻止把参考只影响差动通道当已验证条件；物理上的姿态/yaw耦合与纵横向导航路径也没有被排除。

## 可行性和下一步边界

**VERIFIED**：精确固件源码中存在反馈生成参考、状态触发航点切换；当前候选及数据覆盖、固定 LODO 描述指标可复现。

**SUPPORTED**：姿态参考含稳定的控制预测信息，适合描述反馈策略或完善控制器条件信息；当前数据不支持把该信息当纯差动 plant 反事实监督。

**UNRESOLVED**：飞手独立性、任务时序独立性、风/估计误差与候选的独立性、对其余控制/状态的排除限制、真实执行器状态。

A 路线目前不建议改 PACLD 的控制增益训练目标。可将来源验证留作 B 的观测控制器输入整理；如现有历史资料另有与这些日志严格对应的预置时间表、脚本执行记录和随机源元数据，可再核验，不需要本轮新增实飞。即使找到外生参考，也应先估 R，再明确完整多输入系统和闭环结构下识别 P 的条件，不能直接把两者相等。

## 复现

```bash
cd /home/zn/flap-system-identification/.worktrees/px4-identifiability-a-v1
OPENBLAS_NUM_THREADS=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/px4_iv_sources_v1.py
```

输出 `summary.json`、`source_evidence.json`、`screening_by_date.csv`、`topic_by_date.csv`、`events_by_date.csv`、`commands_by_date.csv`。`private_windows.npz` 是忽略的本地派生中间件，不公开推送。复现所需原始日志和既有共同数据位于合同指定绝对路径，不另复制历史数据。
