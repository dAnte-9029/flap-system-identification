# C：部分可辨识性与控制响应集合

结论：**NO-GO：当前证据不足以产生数据证明的有限 plant 增益区间、确定符号或普适 MPC 动作排序。** 可以开展带明确假设的响应集合敏感性分析。本结论限定于下述局部模型与当前已核验信息，不等于真实飞机具有无限响应，也不等于任何增加假设的方法均无效。

## 数据和实际执行

从 CRM-v1 `9c15fffd24bfcacc7c57cd980cc149098cb83f82` 独立建分支。共享合同位于相邻 `px4-identifiability-contract-v1/experiments/px4_identifiability_v1/shared_contract.md`。使用 velocity_rate_multistep_v1 原生 train；11221 窗口、114 flights、17 日期。主合同逐 ULog 核对的 15 固件版本全匹配；C 读取其已验证映射及 train provenance，无需改变控制器源码。

通过 `(flight,t)` 一对一连接 A2 features 与 CRM windows。源为 `pacld-control-identifiability-v1/.../control_identifiability/features.pkl`、`pacld-control-residual-moment-v1/.../control_residual_moment_v1/windows.csv`（中间路径均为 `experiments/phase_anchored_coupled_latent_v1`），以及主数据目录 `native_actuator_servos_train.parquet`。不打开 validation/test，不训练网络、不修改 PACLD、checkpoint、原始日志或 PX4。

差动 `d=(right-left)/2`，单位 PX4 normalized command；p 为 FRD rad/s。每个窗口保留原生目标 `tau=0.2+offset[0]*1e-6`，范围 0.189061–0.209147 s。在 [0,.05)、[.05,.1)、[.1,.15)、[.15,tau] 四段逐事件 ZOH 积分，减去 t 前差动 baseline，得到 z。检查 reviewed flight 和控制最大 age 160 ms。前两段均值与原 U[:,5] 最大误差 <1e-12。控制计算先转 float64，避免 float32 两通道相减产生的约 4.7e-8 舍入差异。不是将未来记录控制作为决策时已知输入。

## 明确的局部模型类与估计对象

对每个窗口：

`Y_i = p(t_i+tau_i)-p_initial_i = m(H_i) + z_i^T b + e_i`。

`m(H)` 固定为既有仅 train 留日期交叉拟合的 M0_prediction，H 是 t 前可用历史。`b∈R^4` 是四个时间段平均差动的局部端点线性系数，单位 rad/s per normalized command。由于 tau 微变，这是共享原生短时域核近似；其线性、局部常系数与分箱近似均属 **HYPOTHESIS**。四分箱不恢复段内形状敏感性，也不声称识别连续冲激响应。

反事实解释需要指定：相同初始状态、其余控制通道固定，改变差动四段波形；执行器与气动总作用映射到观测 p。这里不是力矩或孤立尾翼增益。真实闭环数据中其他通道也变化、状态未完全观测，因此 e 包含遗漏通道、未知气动/扰动、观测误差、M0误差、局部近似误差及这些项与控制的相关性。

实际拟合 `b_obs=argmin_b ||Y-m-Zb||_W²` 仅是**固定历史预测残差对未来闭环控制波形的观察性投影**，没有同时正交化 Z，也不是因果增益估计。W 为日期等权、日期内 flight 等权、flight 内窗口等权；未把窗口当独立试验，没有输出虚假的参数置信区间。

## 数据支持了什么

**VERIFIED** 的描述结果：

| 项目 | 结果 |
|---|---:|
| 四分箱 b_obs | [-2.645735, -0.699504, 8.425193, -5.180116] |
| 常值波形投影增益 sum(b_obs) | -0.100161 |
| 设计 Gram rank / condition number | 4 / 131.209 |
| Gram 特征值 | [0.000116664, 0.000288603, 0.001637310, 0.015307271] |
| 零增益残差 RMS | 0.500736 rad/s |
| 最小投影残差 RMS | 0.451181 rad/s |
| 相反核 -b_obs 残差 RMS | 0.626301 rad/s |
| PACLD 初始 belief 与观测差 RMS | 0.491713 rad/s |
| PACLD 增量误差 RMS | 0.701469 rad/s |
| PACLD 端点误差 RMS | 0.495457 rad/s |

后面三项不是独立标定的传感器误差，更不是其硬上界；分位数、最大值也不能自动变成未来扰动界。这里没有可验证的真 p 参照，不能把 belief 与观测的差归因给哪一方。归一化命令范围 [-1,1] 与因果时间顺序已知，但不能推出实际舵角、执行器动态、气动导数或角加速度上界。翼展约 1.6 m 也不足以给出这些界；本分析不引入未经核验的惯量或力矩先验。

Gram 只描述记录控制波形的代数信息，不是已知创新噪声模型下的 Fisher 信息。满秩不能解除误差与控制的混杂。

## 可行集合与不可辨识性证据

若额外假设总误差满足已知 RMS 界 sigma，则

`B(sigma)={b: ||Y-m-Zb||_W <= sigma}`。

G=`Z^T W Z` 满秩时，`B(sigma)` 为以 b_obs 为中心的椭球：

`(b-b_obs)^T G (b-b_obs) <= sigma² - s_min²`。

任意线性响应 c^T b 的范围为 `c^T b_obs ± sqrt((sigma²-s_min²)c^T G^-1 c)`，当 sigma<s_min 集合为空。脚本在各日期和各工况都计算同样的剖面。但**现有证据没有提供可识别的有限 sigma**。经验残差不能排除真实 plant 与相关扰动之间的补偿。

在未界定 e 的模型类内，任意 b 都可取 `e=Y-m-Zb`，于是观测完全相同：

`B_data = union_{sigma>=0} B(sigma) = R^4`。

等价地，增广设计 `[Z I]` 对每个 v 有零方向 `[v;-Zv]`。脚本对零核、b_obs、-b_obs 逐窗数值构造，端点重建最大误差 <5e-16；零方向误差为 0。这是 **VERIFIED 的结构性不可辨识结果（在该误差模型类内）**。若要求误差外生、已知随机创新结构或已知物理幅界，模型类会缩小，但这些是需要独立支撑的额外约束。

只做一个透明敏感性截面：**假设** sigma 等于零增益模型的观测残差 RMS 0.500736，常值波形 c=(1,1,1,1) 的区间是 [-9.170339,8.970017]。这不是置信区间、不是数据证明的物理边界、不是推荐噪声界。它含零是这种 sigma 选择的构造性质，不能把“所有组都含零”包装成独立实证发现。

## 跨日期与工况

完整结果在 `group_profiles.json` 和 `group_summary.csv`。各组数据-only 集合都是 R^4；不存在可比较的有限物理宽度。观察投影常值增益跨日期范围约 [-2.726,4.410]，有正有负；这是观察分布及固定 nuisance 下的变化，不能直接断言真实增益变号。10月10日/12日的 Gram condition number 分别约 5310/4989，较多数9月日期约100–340明显更差，说明这些波形的时间方向分离更弱。

| 分组 | 观察投影常值增益 | 上述假设截面区间 |
|---|---:|---:|
| mode 3 | -0.624 | [-10.541,9.294] |
| mode 15 | 0.787 | [-7.262,8.836] |
| 低速组 | 0.963 | [-6.846,8.772] |
| 中速组 | -0.426 | [-9.471,8.619] |
| 高速组 | -0.449 | [-10.176,9.277] |

速度分组与 phase 四分组沿用 CRM；不新增阈值。四相位组投影和约 [-2.064,0.157,0.387,1.014]，其假设截面亦见机器结果。这些条件集合只能支持敏感性分析，不能当作经过验证的调度控制增益。

## MPC 动作区分能力与后续条件

对任何非零局部波形差 v，因 B_data=R^4，`inf_b v^T b=-inf`、`sup_b v^T b=+inf`。脚本对四个单段方向和一个常值方向分别构造 ±1 的响应核及与原始观测完全一致的误差，五个方向均无法保证响应符号。因此不能仅靠本轮证据稳定排序以滚转响应为依据的候选动作。方向指可执行命令内点处的无穷小差分，非要求超出 actuator bounds。完整 MPC 代价的特定动作可能因控制惩罚等项获得排序；本结论不排除那种与辨识无关的排序，也不声称已运行 MPC。

后续有价值的输入是 A 路线能独立支撑的外部激励与排除约束，或 B 路线能够验证的创新/扰动模型、控制器状态及执行命令时间链；还需实际执行器与 p 测量误差的独立约束。只有这些依据成立才把它们显式加入集合。不建议直接把 b_obs 或上述椭球作为 PACLD 控制增益监督。

若未来联合辨识能支撑误差结构，可考虑将输出从单点 Jacobian 扩展为带明确假设的响应集合，并用可验证的集合约束或动作排序目标；当前不启动任何网络或训练目标修改。C 的价值在于限制结论和检查稳健性，不能凭空制造物理监督。

## 复现与验证

```bash
cd /home/zn/flap-system-identification/.worktrees/px4-identifiability-c-v1
OPENBLAS_NUM_THREADS=1 /home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/px4_partial_identifiability_c.py
```

脚本执行 train 身份、键唯一性、native 控制时刻与 quality、U parity、目标恒等式、观测等价构造与零方向断言；结果为 summary.json/group_profiles.json/group_summary.csv。没有逐窗口输出。另执行 Python 编译和 git diff --check。本地提交不推送。
