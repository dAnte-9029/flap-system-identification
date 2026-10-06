# Mission＋Stabilized：筛选与训练准备结果

状态：**数据准备完成，尚未训练模型，也未验证控制可用性。** 新分支`reference-dynamics-data-v1`，起点`a810913`。没有改论文基线、原整flight划分、固件或依赖。

## 数据访问与筛选

从registry默认`trajectory_v3_september_expanded`解析manifest，核验固定哈希，逐个核验原41 train＋17 validation的ULog SHA256。只解析这58个已授权日志；不读取sealed test或reserved evaluation。未重新扫描当前契约之外的其他旧日志。

本轮不只寻找正负脉冲：原始整日志中的Mission(nav3)和Stabilized(nav15)都进入候选。要求armed、airborne、无failsafe、姿态/角速度环开启、相应auto/manual标志一致，状态/参考/命令/相位记录有效。20ms网格采用过去消息保持，发布时间和sample时间均不得晚于查询点；不以插值冒充更高记录率。模式切换、数据缺口、估计器姿态/速度重置处切段，并在各段边界裁掉0.5s。

参考变化固定标签：未来1s滚转或俯仰范围≥1°，或偏航速率参考范围≥0.05rad/s，或推力范围≥0.03；补齐手动yaw通道后其范围≥0.03也标为活动。保留保持参考的窗口。没有使用未来响应方向、误差大小或“是否听话”筛选。命令接近±0.98仅是诊断标签，不冒充完整的分配饱和检测。

初筛39,304个窗口，57个日志有通过的窗口：

| 分区 | Mission | Stabilized | 合计 |
|---|---:|---:|---:|
| train | 16,822 | 9,984 | 26,806 |
| validation | 5,516 | 6,982 | 12,498 |

窗口是H26历史＋K50预测，间隔20ms，起点步长10步（0.2s）。重叠窗口不能按独立试验计数。`quality_counts.csv`提供每日志各质量条件失败计数，条件可能重叠，不相加当互斥拒绝数。

## 配置分组与首轮子集

按日志固件标识及全部已记录FW_/CA_/PWM_/DSHOT_/FLAP_/IMU_/SENS_BOARD_参数精确分组。记录全初始参数快照，缺项不默认为零；有参数更新的日志不直接跨更新混用。18个有窗口的配置组中，训练最多的一组有11个train flight、7,440个窗口，却没有同配置validation。`selection.json`保留该训练量优先选择的不足，不将其当作可独立验证的首轮模型。

随后在原train和validation各至少2个flight的配置组中，仅按训练窗口量选首轮组（不使用验证响应或误差）。首选组前缀`3e1b1922309f`，记录固件`c9af571575af33d5403fbb479256848797c801e2`；进一步核对TRIM_、MAN_YR_MAX和SYS_AUTOSTART，在该组14个flight中一致。该检查仍不是所有硬件/传感器/控制实现完全等价的证明。

归档控制链源码提示：手动yaw杆量可能在Stabilized中直接叠加到角速度参考，姿态消息的yaw_sp_move_rate不足以表示它。因此最终契约补入第五个未来参考字段manual_yaw，并按0.3s有效年龄检查；Mission未开启手动路径时将其置零。98个validation窗口因该补充通道缺失、无效或过期而拒绝。最终训练准备子集：

| 分区 | flights | Mission | Stabilized | 总窗口 |
|---|---:|---:|---:|---:|
| train | 5 | 1,076 | 2,459 | **3,535** |
| validation | 9 | 1,521 | 5,550 | **7,071** |

Mission目标类型按最后发布的有效position_setpoint_triplet事件保持，并保存其年龄；不能因该topic长时间不重新发布就推断飞行数据无效。最终首选组Mission窗口的记录目标类型均为loiter，**不是已经确认的直线Mission**。`current.type=position`本身也不足以证明几何直线或平飞；本轮没有强行作这类标注。

## 最终训练接口

**以`training_contract/manifest.json`为准。** `training_ready/`是补manual yaw前的中间版本，不作为正式训练输入。

- H26过去/当前状态：体轴速度3、pqr3、体轴重力方向3、相对起点相位sin/cos2、扑频、有效空速及valid标志，共14维。
- H26参考历史：roll、pitch、logged yaw_sp_move_rate、thrust_x、manual_yaw，共5维。
- H26执行命令历史：电机和前三路舵面命令，共4维，最后一行严格在起点之前。
- 当前mode上下文：Mission/Stabilized两维；当前四元数、导航系速度和体轴角速度作为初值。
- K50未来输入：仅上述5维参考＋dt。禁止未来实测执行命令、状态、空速、相位、扑频或反馈状态进入预测器。
- 目标：后续四元数、导航系速度、体轴角速度；相位只采用September原始logged契约，物理零位仍未确认。

manual记录约200ms间隔，0.3s有效年龄容许最后值保持，不代表已恢复真实高频杆量；未来自动激励或参考优化的带宽不能由这份数据直接无限外推。首选组logged yaw_sp_move_rate恒为0，不能据此识别该字段非零时的响应。归档e624源码与首选c9af固件不同，训练前控制路径核对和控制晋升时必须保留这个限制。

最终归一化只用首选train窗口涉及的唯一行拟合；phase sin/cos不缩放，loader按起点相位重算。内部train leave-one-flight-out必须在每折重新拟合，不能复用全train的归一化。`training_ready/train_leave_one_flight_out_folds.json`提供原5个train flight的固定fold编号，最终子集没有改变这些flight。

## 可复现产物与验证

- [最终清单](training_contract/windows.csv)、[最终manifest](training_contract/manifest.json)、[模式/活动参考统计](training_contract/summary.csv)、[训练参考范围](training_contract/train_reference_ranges.csv)。范围是数据覆盖描述，不是安全飞行包线。
- 缓存及归一化位于`dataset/reference_dynamics_data_v1/contract/`，按仓库规则留在本地；manifest记录哈希及来源链。
- 读取器：`src/system_identification/data/reference_windows.py`；配置：`configs/training/reference_dynamics_data_v1.yaml`；训练方案见`docs/plans/reference-dynamics-data-v1.md`。
- 初筛39,304个窗口通过读取预检；最终10,606个窗口全部检查有限值/维度、时间边界和模式边界。14个最终缓存各作一次未来测量污染测试：改变未来真实状态及当前/未来执行命令，不改变预测输入。train-only归一化独立重算一致。12项针对性单元测试通过。

```bash
# 以下命令中的 python 均替换为 /home/zn/anaconda3/envs/flap-train-gpu/bin/python
python scripts/prepare_reference_dynamics_data.py
python scripts/check_reference_dynamics_data.py
python scripts/prepare_reference_training_selection.py
python scripts/finalize_reference_training_contract.py
python scripts/verify_reference_training_contract.py
python -m pytest -q tests/test_reference_windows.py tests/test_control_reference_semantics.py
```

提取入口拒绝覆盖已有结果；复现需在独立工作副本中保留旧产物并准备空的新输出目录。只读核验可直接运行最后两条命令。原始日志和上游准备缓存需满足已固定哈希，不得换成测试数据或其他版本。

本轮没有启动训练、GPU任务或控制器优化。下一轮可在该有限同配置数据上做保持基线、局部参考辨识与因果滚动模型对照；先回答未见参考变化是否预测正确，再评估闭环收益和模型误差利用。不能把未来日志Mission参考驱动的预测成功等同于导航外环重新规划成功。
