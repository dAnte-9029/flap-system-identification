# Stabilized滚转联合控制链核对

## 结论与对先前建议的修正
用户描述与现有源码和日志一致：主要人为输入是roll，elevon形成滚转操纵，rudder多在yaw摇杆中位时自动参与。**不应再以rudder孤立变化少作为这种正常操纵工况的数据不足证据；优先工作应改为Stabilized联合命令下的模型响应核验，暂缓专门rudder激励。**

不过，rudder不是直接测量roll误差然后替代elevon稳定滚转：源码链路包含协调转弯yaw设定值、姿态到体轴角速度的耦合转换、yaw角速度闭环及分配输出。需要保留完整链路含义。

## 源码与实际参数
核对58架开发日志所记录固件commit的准确源码，不以当前外部工作区替代历史实现。STAB枚举=15，同时要求manual/attitude/rates enabled、climb_rate disabled。Stabilized的roll姿态目标为manual.roll×FW_MAN_R_MAX（日志均35度）。yaw姿态参考采用当时姿态，不等同yaw摇杆直接给固定航向。

YawController根据受约束roll、pitch与airspeed构造协调转弯Euler yaw rate，转换为body yaw rate，另可叠加manual.yaw×FW_MAN_YR_MAX。rate controller使用角速度误差及其控制状态形成yaw输出。代码支持roll输出到yaw的前馈，但所有58架日志FW_RLL_TO_YAW_FF=0，因此本批没有启用这项直接前馈；不能将rudder变化归因于该禁用项。参数初值及运行中变更清单见parameters.csv。

实际elevon、rudder使用冻结数据中的最终软件命令；common=(left+right)/2，differential=(left-right)/2。归一化命令不等于实测舵角；尚未逐周期完全复现yaw控制器，因此本轮确认结构和联合活动，不量化每个内部项的贡献。

## 数据覆盖
只读41train+17validation的准入原始日志和缓存。没有访问Sep8/Sep19、没有训练或推理。只在valid_core样本内统计Stabilized。遥控和设定值按publication时间过去值保持，manual最大龄期200ms、姿态/角速度设定值50ms、控制模式2s。该额外freshness是诊断可对齐规则，不改原训练数据。

| split | flights | stabilized_s | fresh_chain_rows | yaw_neutral_pct | roll_runs | positive_runs | negative_runs | full_neutral_cycles | all51_stabilized | all51_fresh | median_rudder_range | median_differential_range | yaw_neutral_runs | yaw_neutral_rudder_changes_gt01 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 41 | 2304.1413 | 112294 | 99.8878 | 1594 | 1102 | 492 | 19 | 10764 | 2105 | 0.0633 | 0.2453 | 1590 | 1587 |
| validation | 17 | 1480.3279 | 71826 | 99.9443 | 900 | 401 | 499 | 20 | 1445 | 293 | 0.0519 | 0.1895 | 898 | 897 |

训练约2304秒、验证约1480秒有效Stabilized数据；yaw摇杆中位阈值abs≤0.05，接近100%的可对齐行符合。Roll活动阈值abs≥0.10，回中abs≤0.05，区间内滞回，持续≥0.2s。原始1s origins中训练10764、验证1445个完整51状态都在Stabilized；进一步要求每行诊断链新鲜后为2105和293。**不能把后者较少解释为原模型样本不足**：只要短暂缺一个新鲜诊断字段，整窗就不满足这个额外条件。

识别到1594个训练、900个验证roll活动run，正负分别1102/492和401/499。它们可能被日志间隔/诊断新鲜度切碎，不是独立操纵次数。全run yaw摇杆中位的1590/898个中，1587/897个rudder变化范围大于0.01。该0.01只是描述数值变化，不能自动称物理显著动作。

差模变化范围中位数训练0.2453、验证0.1895；rudder分别0.0633/0.0519（归一化命令）。这直接说明应保留elevon与rudder的联合变化，而不是要求其他通道安静。

## 建立、保持与回中
活动run前后均有至少0.2s中位段的完整命令周期，在当前严格连续/新鲜规则下有训练19、验证20个。这里“完整”是遥控指令的进入/保持/退出，不保证实际姿态已经回零或稳定，更不等于真正阶跃。缺少完整周期可能来自分段和字段新鲜度，而非实飞没有回正。

固定按event_id排序取各分区完整周期的中位identity画图，选择不依赖实测响应或模型误差；原始stick、姿态目标/实测、common/diff/rudder和p/r设定/实测全部保留。所有run清单与轨迹见episodes.csv、traces.csv。

Roll目标的记录网格公式检查有残差：逐flight RMSE等权平均训练约2.31度、验证约1.72度（最大约4.03/2.29度）。这是异步as-of字段的公式核对，不是飞控逐周期复现，不把差异直接归因成模型或控制器bug。需要继续对照精确参与计算的时刻才能分解。

## 下一步明确调整
不启动专门rudder输入激励。先复用已冻结H26/K50三个seed的validation完整预测，在已有1445个全窗Stabilized origins上做联合命令响应评价；其中293个全链路新鲜窗口可作额外可视化子集，主结果保留全部有效Stabilized窗口。按roll进入、保持、回中相关段分别观察p、r、roll姿态和速度，不能将rudder独立出来当成唯一自变量。

若模型在正常联合操纵下能复现实测响应，就没有依据优先补rudder实验；若仍持续偏离，再针对具体阶段排查状态递推、观测时序和训练覆盖。只有确实缺少重复roll操纵或某类保持/释放工况时，才按正常操纵方式补充正负roll参考试验。实际动作幅值和执行范围不在本轮确定。

## 工程记录与限制
首次实现误用了nav_state=10（Acro），得到零覆盖；核对exact firmware后改为15，并保留initial_wrong_mode全部初版结果和说明。最终表不是该错误版本。manual yaw参数名从不存在的MAN_YR_MAX改为源码中的FW_MAN_YR_MAX，重新核对允许日志，不改变模型或数据。不能把初版零覆盖当成数据缺失证据。

四项测试检查摇杆滞回/换向、过去值新鲜度、四元数滚转角及逐固件STAB枚举；58份原始日志哈希及缓存哈希通过。只新增审计脚本、表格与图，未修改PX4、模型、旧实验报告或split，未commit/push。
