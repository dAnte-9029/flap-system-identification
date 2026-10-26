# 方法、假设与控制链证据

本实验对象是显式指定的历史 `velocity_rate_multistep_v1/main_50hz`。没有使用 canonical 新相位，也没有估计气动力或使用 DeLaurier 先验。程序时间与物理时间限制沿用上游 `local_control_response_time_contract_ablation_v1/time_contract_zh.md`。

## 文献适用性

| 方法与全文来源 | 关键假设 | 本数据可检查／未满足的部分 |
|---|---|---|
| [Colin et al., 2020](https://doi.org/10.1016/j.automatica.2020.109171)，Drive `26_2020_Colin_ClosedLoopMIMO_DataInformativity_Published.pdf`，§3–5 | 指定包含真系统的 LTI 预测误差模型类及参数化；稳定闭环有理控制器；规定的稳定、逆稳定、monic 对角噪声模型；相关结果要求外部激励与创新独立；结构矩阵由控制器、模型阶次和激励构成 | 可核查控制器软件和候选变量协方差；没有冻结真实 LTI 对象／噪声模型，也无可确认独立注入。本文的 Gram 矩阵**不是**该文的结构矩阵，满秩不认证 GRU。未获证书不等于一切闭环参数不可辨识。 |
| [Dinkla et al., 2026](https://doi.org/10.1016/j.automatica.2025.112556)，本次复用 Drive `03_2026_Dinkla_ClosedLoopDeePC.pdf` | 最小离散 LTI 创新模型、适当稳定／well-posed 闭环；过去数据足够丰富；IV 与待消除的未来创新正交、与回归量有足够秩；一致性还涉及过去窗口与数据长度的渐近条件 | 可描述有限阶输入矩阵；尚不能认证创新正交或已选有限历史的渐近条件。**版本校核发现该 Drive 文件内容为 arXiv:2402.14374v1，2024-02-23，§3.2 假设1–3，而非2026出版社终稿。**正式 DOI 与 TU Delft 元数据已核对，终稿下载超时已登记，不把预印本页码冒充终稿。本文未实现 CL-DeePC/IV 动力学。 |
| [Matt et al., 2025](https://doi.org/10.2514/1.C038147)，Drive `M08_Matt_2025_飞翼开闭环系统辨识_出版社在线版.pdf`，§II–IV | 围绕配平的低阶模型，明确控制律与闭环通路；定制开／闭环频率扫掠，以及不同 doublet/step 飞行作验证；关注相关输入与频域质量 | 本数据是常规任务反馈记录，未发现受登记的独立扫频注入与独立执行器响应标签。可借鉴输入相关性、通路与多轴检查，不能把一般航点转弯等同其系统辨识机动。 |

三份已有全文均实际读取；Drive ID、版本及本地文本 SHA256 见 `sources/literature_receipt.json` 和 `literature_text_hashes.json`。版权全文仅本地缓存，不提交或重复发布。Stage4 关于 Forssell–Ljung 直接法／间接法的总结复用为背景，没有声称本轮重新精读其全文。

## 实际调用链与可用信号

14 个 AUTO_stable 固件标识各恢复 7 个精确提交源文件（98 条），另恢复 MissionBase/MissionBlock（28 条）。内容按 SHA256 去重保存在 `sources/`；来源 URL、哈希及行号在 `firmware_sources.csv`、`navigator_sources.csv`、`source_evidence.csv`。这些源码与日志标识对应，但仍不认证实飞 binary 或全部 dirty build 选项。

| 层 | 实际调用路径／源码证据 | 观测与限制 |
|---|---|---|
| 任务参考 | `MissionBase::on_active` 检查 `is_mission_item_reached_or_completed()`，成立且 autocontinue 后调用 `advance_mission → setNextMissionItem`；`Mission::set_mission_items → mission_item_to_position_setpoint` | `position_setpoint_triplet` 有原始地理航点及 type/valid。MissionBase 快照 `a8bb…` L335–347、L475–483；MissionBlock `a58e…` 的到点检查依赖实际位置等状态。航点坐标可能预先设定，但**活动航点切换时刻受状态反馈影响**。 |
| 外环 | `FixedWingModeManager` 消费任务 triplet 与实际状态，生成横／纵向设定；固定翼横纵向控制再生成姿态／推力参考（完整机制沿用 Stage4） | `fixed_wing_lateral_setpoint`、`fixed_wing_longitudinal_setpoint` 和 `vehicle_local_position_setpoint`。后者 vx/vy/vz 的合法 NaN 不补零。新鲜消息不自动证明独立外部激励。 |
| 姿态到角速度 | `FixedwingAttitudeControl` 消费姿态参考与实际姿态，发布 `vehicle_rates_setpoint` | q_d、roll/pitch/yaw rate setpoint 可得。下游设定值不是独立随机参考。 |
| 角速度到力矩 | `FixedwingRateControl::_rate_control.update(rates,body_rates_setpoint,angular_accel,dt,…)`；`rate_control.cpp` 的 P/I/D/FF；另有空速缩放、SLOW 积分转移和 gain compression | `rate_ctrl_terms/status` 与 torque/thrust 可得。terms 的 I 与 status 的 I 属于不同更新阶段，不能直接相减归因；不同项高度相关，不能用相关系数分解因果贡献。 |
| 控制分配 | `ControlAllocator` → `ControlAllocationPseudoInverse` → clip／输出；Stage4 已验证的参数下，理想未限幅关系 d≈τ_roll/1.1、common≈τ_pitch/2 | motor/servo 命令沿用原 Parquet 原生 ZOH。`d=(right-left)/2`；与旧 `(left-right)/2` 相反。归一化命令不是舵角／推力，分配矩阵不是气动效能矩阵。 |

本轮只补取所需 13 个 topic，121 条允许日志均先核对 SHA256，121 条日志的 changed_parameters 均为空。全部 train/validation main_50hz 记录覆盖表在 `topic_coverage.csv` 与 `field_coverage.csv`；这是对 Stage4 8 条证据范围的补充，不假定旧8条代表其余数据。

## 时间与因果合同

- 决策 t 为冻结窗口 `decision_us`（位置状态发布时刻），各历史 lag 的状态行按 publication≤t−lag 取最近过去值；每个字段沿用上游已核可用性。原生控制与参考按其消息 timestamp 向后保持，无未来插值。
- 下游参考超过100ms置缺失，25/50/100/200ms覆盖率另表；任务航点按事件保持不设TTL，同时记录年龄／valid。相对任务几何使用当时可得 global position，后者200ms新鲜度门限。局部位置参考的非法／合法NaN均不假装成已知速度。
- 回归目标是未来完整四路命令的精确ZOH平均偏离，`mean(u[t,t+H))−u(t)`。历史命令、状态、相对任务几何、下游参考及已发布内部项分别递增加入。只预测命令，不预测或拟合运动。
- 运动诊断仅离线使用 `vehicle_angular_velocity.timestamp_sample` 作线性插值，原生括号宽度≤50ms，无外推。保守规则也屏蔽大间隔右端孤立样本；不据运动结果选事件。p/q/r 是有滤波的日志观测，不是独立、无延迟的物理真值。
- 参考事件附加分析使用20ms backward-ZOH展示网格，改变时刻的显示量化不超过20ms；不能据此估计执行器机械延迟。未来参考／运动数组只用于离线诊断，`feature_columns` 不读取它们。
- 全部控制坐标和统计为归一化命令。控制变化幅度与角速度测量误差不同量纲，不能直接拿二者数值作信噪比。

## 固定辅助分析

主回归为标准化后的岭回归，最小化航次等权平均平方误差加0.01倍系数平方；0.001与0.1为预定敏感性。每次缺失中位数、标准化和配置类别字典仅由拟合航次确定，缺失指示单列。按原ULog SHA排序轮转5折；同日志的所有片段同折。只用完整支持不重叠训练窗口拟合，原重叠候选和固定匹配窗口全部得到对应航次留出预测。validation只应用完整train拟合结果，不定门限。

200/500ms历史，U／U+状态／再加任务几何／再加下游参考／再加内部项共7种固定变量组。控制变化方向描述门限0.02；运动门限仍为0.1rad/s。这两个方向口径不同。剩余误差不是最优非线性条件方差，也不是白噪声／外生创新，不能把可解释方差比例解读为因果贡献百分比。

信息矩阵用四维未来100ms平均命令偏离的1／3／6阶连续注册窗口滞后，分别报告原命令单位和标准化的航次等权中心化Gram。初次实现错误地期待精确100ms起点并得到空6阶矩阵；实际注册步长常为108–118ms，修正为连续窗口序号滞后，间隔>150ms断开，实际年龄全部保存。该修正因采样支持而作，未搜索最优阶次或参考运动误差。矩阵“相对秩”门限为最大奇异值的1%；熵有效秩、条件数、最弱向量及逐航次删除敏感性均报告。它不是 LTI 信息量定理或非线性全局辨识认证。

历史状态描述包含NED速度、FRD角速度、四元数、频率及命令的滞后。未把log-local相位作为跨日志共享绝对机械相位回归项；也未穷尽非线性状态交互或全部控制器隐状态。因此剩余量是该有限变量组／线性类的留出误差，不是全部已知信息条件下的不可预测创新。固定匹配两端可能属于不同辅助交叉拟合折；配对剩余对比也含两套辅助系数的估计误差，不是同一确定反馈律下的物理效应分解。
