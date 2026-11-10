# PX4闭环实飞控制动力学可辨识性：三路线综合结论

本轮完成 A/B/C 并行研究。**现有已核验信息支持闭环控制预测与部分控制链重建，尚不足以把差动尾舵到约0.2秒滚转的关联转成可信的反事实 plant 增益监督。** 推荐后续以 **B：版本化控制器与创新/测量模型** 为主、**C：条件响应集合与敏感性分析** 为辅；这是基于本轮结果的后续优先级，不是已证实能改善MPC的方案。本轮停止，不训练PACLD。

## 共同合同与实际核验

三个实际仓库是 `/home/zn/QgcLogs`、`/home/zn/PX4-Autopilot` 和 `/home/zn/flap-system-identification`。前后两者 origin 为 dAnte-9029；本地 PX4 origin 实为 jiashunjiang2022/PX4-Autopilot。本轮不更换仓库或修改PX4，而以训练日志内 `ver_sw` 的精确SHA取历史源码。

基线均为指定 CRM-v1 `9c15fffd24bfcacc7c57cd980cc149098cb83f82`。主Agent先形成 shared_contract.md，再启动三Agent。逐一读取共同队列114个训练ULog固件元数据，15固件SHA均与既有provenance一致，全部本地git对象可用。使用 velocity_rate_multistep_v1 既有划分/quality/原生时序，共11221窗口、114 flights、17日期。未打开validation/test数据或计算其哈希。

主Agent核对 features/context 的既有A2身份校验值，CRM/A2的 `(flight,t)` 键集合、Y、U[:,5] 和目标时刻逐键一致。差动为 `(right-left)/2`、normalized command；p为FRD rad/s。实际目标时距0.189061–0.209147s，保留原生目标不重新插值。命令不是舵角，normalized torque不是N·m。仅观察整体飞行器响应，没有声称隔离尾翼载荷。

## 三路线结果

| 路线 | 真正回答的问题 | 关键实测结果 | 判断 |
|---|---|---|---|
| A | 是否有来源可信、满足排除条件的独立激励 | 姿态参考变化提高差动预测LODO MSE 18.08%，17/17日期正相关；角速度参考提高9.06%，仅6/17日期MSE改善。源码确认反馈生成；航点切换状态触发 | 当前候选IV NO-GO；未估计IV响应 |
| B | 控制链能重建多少，已知控制器是否足以辨识plant | 分配层92766记录/114日志，RMSE .0014144、R² .999975；P项88025记录/90日志，RMSE .0025011、R² .991196 | 部分链重建SUPPORTED；plant监督NO-GO |
| C | 未独立限定误差时能否约束局部响应核及动作方向 | 四分箱Gram rank4、条件数131；自由总误差下可行核R^4，正负核观测等价至<5e-16 | 数据证明的有限区间和响应排序NO-GO |

A 的17日期是train内留日期诊断，不是独立validation。姿态参考第一阶段强并不能修复工具变量独立性；917次航点变化只有35次落入共同窗口，而且时序由到达条件生成。reviewed区间134条速度命令均内部来源、设定值恒8m/s。提前运动残差和其他通道存在描述关联；由于条件X(t)含早期运动后代，预趋势检查本身也不能作为正式外生性检验。A源码依据覆盖15固件150路径。

B发现角速度控制模块14种内容、RateControl库6种内容；积分转移、压缩及FF增强随版本变化。rate_ctrl_terms缺24日志。记录PID/GC/即时空速到torque的近似oracle检查RMSE .067725、R² .950107，允许+2ms离线匹配，**不属于因果递推输入**。P/I/D/FF求和误差约1.8e-8只是日志代数一致性。分配重建成功说明命令生成链可信，不证明真实舵机位置或plant。控制器结构图和逐日期结果见B报告。

B的局部端点八列回归rank8、标准化Gram条件数2.6175，OLS RMSE .50791 rad/s；自由创新条件下profile信息为零，改变差动系数并补偿创新后预测差异≤1.42e-14。这不是“闭环永远不可辨识”的普遍定理，而是指出尚缺可独立支撑的创新结构约束。没有把静态设计满秩称为动态持续激励充分。

C在四个原生时间段内积分真实差动波形；控制parity<1e-12。观察投影核为[-2.646,-.700,8.425,-5.180]，常值波形投影和约-.100，但不是因果核。跨日期该观察投影和约[-2.726,4.410]，不能解释成plant变号。10月两日期Gram条件数约5310/4989，时间方向分离较弱。没有独立真p/舵角/扰动界，模型预测残差不能充当传感器噪声硬界。

C仅展示一个明确额外假设截面：若总误差RMS被限定为零增益残差 .500736 rad/s，常值波形区间[-9.170339,8.970017]；这是敏感性示例，含零是该选择的构造性质，不是物理置信区间。未限定误差时任何非零波形方向都可正可负，不能保证基于滚转响应的MPC动作排序；这不排除因控制惩罚等其他代价项而获得某些排序。

## 交叉评议：不能把哪些量互相比较

1. A估计的是参考变化对记录命令的第一阶段**闭环观察性预测**，没有估计外部参考干预总响应R，更没有plant干预P。
2. B重建的是控制器/分配器，不是plant；端点回归只是观察诊断，前0.1秒命令不能代替固定全时域干预波形。
3. C定义了固定其他通道和初态的局部四分箱响应模型类，但其有限集合需要额外误差界。C的投影系数、B的回归系数和A第一阶段系数不得直接比较增益。
4. 共同的不可辨识反例依赖“未受约束的创新/总误差”模型类。它严格说明现有合同缺什么，不证明一切有物理依据的创新模型都不可能辨识。
5. 当前最可信的控制相关信息是B的命令分配与可观测P项重建，以及A的参考反馈来源；最可信的plant结论是C所揭示的**证据边界**，而非某个有限增益数值。

## 推荐主路线与后续具体变化

**B为主、C为辅，条件推进。** B有精确源码和可观测内部量，能把“混杂”具体化为可检查的控制器/测量/执行器状态缺口；C用于检验这些缺口在不同假设下是否仍允许相反动作响应。A暂不继续2SLS，仅在既有历史资料发现可对齐的预置时间表、脚本执行/随机源记录时重新核验，不新增实飞。

先从现有日志补查：每次采样与uORB消费的对应关系、全速滤波输入、GC/积分转移与FF增强内部状态、PWM到实际舵角反馈、独立测量误差或扰动约束。若历史记录不存在，应明确缺失，不能从控制源码补造。控制器完整递推残差是否能由已知状态解释，以及创新的条件结构，是小型闭环PEM之前需解决的问题；残差白化通过也不能独自证明外生性。

若上述条件得到支持，后续网络可显式区分物理隐状态、控制器状态、测量创新和执行器动态，并保留固件/模式/采样时序信息；控制器状态可作为联合观测模型的辅助状态，不能把反馈参考当作独立物理作用。训练目标先考虑条件预测似然及记录命令一致性；只有取得有依据的响应集合后，才考虑集合约束或稳健动作排序目标。**目前不推荐把A观察系数、B控制器R²、C假设椭球或CRM moment直接作为因果增益监督。** 后续独立确认再按批准协议使用保留validation；本轮未执行，MPC收益仍UNRESOLVED。

## 交付、提交与复现

主合同与综合报告 worktree：`/home/zn/flap-system-identification/.worktrees/px4-identifiability-contract-v1`，分支 `research/px4-identifiability-contract-v1`。

| 路线 | worktree（前缀 /home/zn/flap-system-identification/.worktrees/） | 本地提交SHA | 报告 |
|---|---|---|---|
| A | px4-identifiability-a-v1 | 70e50245de58a31c0cd7a803f651f66088407c41 | experiments/px4_identifiability_a_v1/report_zh.md |
| B | px4-identifiability-b-v1 | c7683865f053ac7aa165a13e7e68641a7198a814 | experiments/px4_b_v1/report_zh.md |
| C | px4-identifiability-c-v1 | 220f5ceae16d6e514ddb906b82bb1aa622bdfa1c | experiments/px4_partial_identifiability_c_v1/report_zh.md |

分支分别 research/px4-identifiability-{a,b,c}-v1。均从9c15fff出发，仅提交本轮新增脚本/聚合/报告，不推送。逐窗A中间件与主合同逐日志列表保留本地且忽略。

```bash
# 先在主合同worktree生成只含train的共享来源文件并核验
cd /home/zn/flap-system-identification/.worktrees/px4-identifiability-contract-v1
/home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/verify_px4_shared_contract.py
/home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/check_px4_cohort.py

cd /home/zn/flap-system-identification/.worktrees/px4-identifiability-a-v1
OPENBLAS_NUM_THREADS=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/px4_iv_sources_v1.py
cd /home/zn/flap-system-identification/.worktrees/px4-identifiability-b-v1
OPENBLAS_NUM_THREADS=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/px4_b_v1.py
OPENBLAS_NUM_THREADS=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/px4_b_information.py
cd /home/zn/flap-system-identification/.worktrees/px4-identifiability-c-v1
OPENBLAS_NUM_THREADS=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/px4_partial_identifiability_c.py
cd /home/zn/flap-system-identification/.worktrees/px4-identifiability-contract-v1
/home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/crosscheck_px4_routes.py
```

上述分析均已实际执行。主Agent独立重算B全部汇总SSE/R²，核对A的3366条源码摘录与精确SHA行号，检查C控制parity和集合口径；结果见 crosscheck.json。三Agent完成针对性断言、语法/差异检查；无仿真plant、无NN训练、无正式模型/checkpoint修改、无holdout评估、无远程发布。本轮到此结束。
