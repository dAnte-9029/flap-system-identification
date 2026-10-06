# Reference dynamics：Mission＋Stabilized 训练准备

分支`reference-dynamics-data-v1`从`a810913`创建，既有未提交改动保留。本轮范围为筛选、数据契约、缓存、读取器及训练前检查；未启动正式训练。

最终权威输入为`docs/analysis/results/reference_dynamics_data_v1/training_contract/manifest.json`，不是上游初筛的四参考字段版本。配置为`configs/training/reference_dynamics_data_v1.yaml`，结果见同目录[报告](../analysis/results/reference_dynamics_data_v1/report_zh.md)。

模型边界是姿态参考端以下的固定内环、分配、执行器和飞机。未来只能输入候选滚转/俯仰参考、日志yaw_sp_move_rate、推力参考和手动偏航参考；手动偏航在Mission中置零，在Stabilized中保留最后有效记录及年龄。当前mode为上下文。执行命令历史严格止于起点之前，未来执行命令不进入输入。未来Mission参考序列用于条件预测评价，不能直接当作重新规划的闭环有效性证据。

先采用一个匹配固件/控制参数组，分别报告Mission、Stabilized及mode条件混合模型。其他配置组保留为后续域外审计候选，不默认等价。初次训练对照应包含状态/角速度保持、正则化局部参考响应模型和因果历史编码的滚动模型；模型结构和训练预算尚未在本轮执行或搜索。

开发选择优先使用5个train flight的leave-one-flight-out，各fold重新拟合归一化；最终全train拟合后才使用原9个validation flight。最终归一化仅供全train拟合，不能在内部交叉验证中复用。旧sealed test和reserved evaluation继续关闭。数据窗口重叠不构成独立重复试验，以完整flight聚合误差。

评价100/200/500/1000ms的姿态、速度、各轴角速度，分模式、参考变化/保持、初始工况和接近命令限幅标签报告。不能按未来响应方向、预测误差或模型表现筛选数据。活动参考标签只表示输入变化超过固定描述阈值，不表示独立激励或可辨识通过。

进一步做控制器之前还需核验首选固件对应实现、内环其他旁路/调度与硬件一致性；评估参考变化的幅值、时序、重复性及模型优化对误差的利用。数据准备通过不等于控制通过。归档e624源码帮助发现手动yaw旁路，但不能替代首选c9af固件的实际源码核验。
