"""Render frozen audit statistics and reviewed Chinese interpretation."""
import json
import numpy as np
import pandas as pd
from audit import HERE,CHANNELS,sha,write_json
from summarize import table

p=HERE/'results'
def read(name):return pd.read_csv(p/(name+'.csv'))
s=read('match_sensitivity');c=read('coverage');ec=read('event_counts_per_flight');sc=read('strengthened_contrast_sensitivity')
primary=s.query('caliper=="primary" and contrast_scale==1 and group=="all"')
assert primary.query('channel=="any"').nonoverlap_pairs.tolist()==[323,339], 'interpretation requires review if statistics change'
events=ec.groupby(['channel','scale']).events.sum().unstack().reindex(CHANNELS)
sync=read('event_independence').groupby('channel')[['events','isolated_50ms']].sum().reindex(CHANNELS)
events.columns=['阈值×0.5','主阈值','阈值×2'];events['±50ms无其他通道事件']=sync.isolated_50ms;events=events.reset_index()
f=read('control_coverage_per_flight');ranges=[]
for ch in CHANNELS:
 g=f[f.channel==ch]
 ranges.append(dict(通道=ch,最小指令=g['min'].min(),最大指令=g['max'].max(),贴近限幅时间占比百分数=100*np.average(g.near_limit_fraction,weights=g.duration_s),
                    最长无有效变化秒=ec.query('channel==@ch and scale==1').max_dwell_s.max()))
quality=read('selected_pair_quality');q=quality[(quality.channel=='any')&quality.feature.isin(['v_rmse','R_rmse_deg','w_rmse','f_rmse','initial_v_gap','initial_R_gap_deg'])]
q=q.pivot(index='horizon_ms',columns='feature',values='p50').reset_index()
train=read('train_increment_statistics');train=train.query('sampling_group=="main_50hz" and measure=="delta"')[['channel','p50','p90','p99']]
strata=read('match_strata');es=read('response_evaluation_support')
strength=sc.query('ratio==2 and split_pair=="all"')[['horizon_ms','channel','raw_pairs','nonoverlap_pairs','flights','flight_disjoint_pairs']]
examples=json.loads((p/'examples.json').read_text())
figures='\n\n'.join(f"### {x['horizon_ms']} ms · {x['channel']} · pair {x['pair_index']}\n\n![历史状态及原生控制序列]({x['figure']})" for x in examples if x['channel'] in ['any','motor_strengthened'])
report=f'''# 控制可辨识性与控制响应数据覆盖审计 v1

日期：2026-10-21。基础提交：`8964b44933c7364c066d3cb09440a6d2a7f558f1`。分支：`exp/control-identifiability-audit-v1`。本报告只审计数据，不加载或评价神经网络、不训练、不开展 MPC。

## 研究结论

现有数据有大量控制变化，但**相近历史及起点下、跨航次的独立通道对照明显不足**。它支持有限的探索性控制敏感性与混合控制序列差异验证，不足以支撑“四路控制的反事实响应已验证”。

- 161 段获准飞行全部进入覆盖审计：train 122、validation 39；33 段测试飞行仅核对共享划分元数据，从未打开测试 Parquet 或测试原始日志。共 947,397 条原生状态行，其中 940,700 条 valid_core；有效控制时间 23,489.302 s（6.525 h），941,203 个原生控制联合时间点。
- 500 ms / 1 s 主设置分别得到 **907 / 1,356 个唯一无序对**；按整个历史和未来支持区间去重后仅 **323 / 339 对**，涉及 **123 / 131 段航次**。每段航次至多使用一次的贪心配对只有 **50 / 53 对**；这仍不等于独立随机干预样本。
- 单通道主设置无重叠对数：电机 **27 / 7**，左舵面 **0 / 0**，右舵面 **0 / 0**，方向舵 **2 / 1**。要求未来差异至少为历史及起点差异的两倍后，电机仅 **12 / 3**，其余三个通道均为 **0**。
- 将历史和起点阈值收紧后，两个时域均只剩 **5 个无重叠综合对照**。这说明主设置对照并不是近乎相同的初始条件；结论对匹配容差敏感。
- 对已有 `main_50hz` 模型最重要的是：仅在 validation 内配对时，综合对照为 **21 / 24 对**，单电机对照 **0 / 0**；加强控制差异条件后综合对照为 **18 / 21 对**。低频组不能被混入这些数字。

建议先做 **main_50hz、500 ms、稳定巡航区域的混合控制探索性检查**。若专门研究单通道，电机最值得优先补充激励和执行器反馈；现有训练侧 500 ms 电机匹配只能用于设计诊断，不能充当未见航次响应验证。1 s 可作为补充，不能因综合匹配数稍多就认定独立控制识别更可靠。

## 1. 数据边界、来源与实际覆盖

按 `configs/data/velocity_rate_dataset_registry.yaml` 解析显式指定的数据集，清单 SHA-256 为 `8afb8623bf7174e4da5b2044f644331b8b251433fed3576f6dcdab470935bcb5`。复用既有原生状态、质量掩码、EXCLUDE、连续 segment、控制通道映射、划分与 100 ms 窗口起点。没有修改任何观测、排除区间或划分。

{table(c)}

控制时间按 valid_core 连续区间内的原生控制保持时长求和；区间首个有效控制事件之前的零碎时间不外推，因此略保守。覆盖统计不是把全部 reviewed_duration 都视为有效。采样组分别分析，绝不把 sparse_10hz 重新插值成 50 Hz 独立观测。

读取了授权分区的 samples 历史字段、windows 元数据、native_actuator_motors 和 native_actuator_servos。代码不读取 `target_*` 列；每个匹配特征只索引该窗口起点及之前的状态。整航次的历史字段会为不同起点重复提供上下文，但任何窗口的未来运动均未用于选对、打分或判断控制差异。

八个实际使用的 Parquet、共享清单/划分/溯源文件和 QgcLogs 来源表均校验哈希。按 train/validation × 两种采样组抽查了四个原始 ULog：原生指令值与时间戳逐项相等，原始日志 SHA-256 与登记值一致，控制分配及 PWM 功能映射符合数据合同。详见 `results/run_manifest.json`。

161 段日志的传动比均为约 7.90909；156 段属于一个硬件 UUID，4 段属于另一 UUID，1 段未知。审计覆盖仍包含全部 161 段；匹配要求硬件 UUID 一致且历史频率可用，未知固件恢复日志未进入匹配。全部 PWM_MAIN / CA_SV_CS 记录参数一致，日志内无记录到的参数变更；共有 17 个已知固件版本，另有 1 段固件未知，另做同固件敏感性筛选。

## 2. 控制激励：变化很多，不等于独立实验

### 2.1 量纲和事件定义

扑翼电机是 `actuator_motors.control[0]` 的归一化推力命令，本数据合同范围 [0,1]；左右舵面与方向舵依次为 `actuator_servos.control[0:3]`，范围 [-1,1]。固件消息定义及原始参数证明它们是控制输出命令，**不是舵角实测、转速实测或实际气动力**。本审计没有把 PWM 映射当作执行器响应标定。

主事件阈值（motor / left / right / rudder）为 **0.02 / 0.05 / 0.05 / 0.02**。在每个有效连续区间内，以第一条命令为基准；命令相对上一次接受的基准累计偏移达到阈值、且距上次事件至少 100 ms 时，接受事件并更新基准。100 ms 防重复时间与较稀疏日志分辨率相当；它不是对执行器响应延迟的估计。慢斜坡可以产生事件；事件间隔表示未达到下一次有效偏移，不表示中间命令严格恒定。

训练集 main_50hz 原生相邻命令绝对变化的分布如下（阈值不由验证运动结果选择）：

{table(train)}

阈值高于典型微小更新，覆盖归一化全量程的有意义幅值，并用 ×0.5、×2 检查敏感性。闭环变化中的真实调节与控制抖动无法单靠这些日志完全分离；这里没有估计物理测量噪声，也不声称阈值对应最小可辨识气动力。变化率直接按原生相邻 dt 计算，各采样组和划分见 `increment_rate_statistics.csv`。

{table(events)}

“±50 ms 无其他通道事件”只描述阈值化事件的同步性；其他通道仍可能在阈值以下变化，不能把它称为单通道干预。四通道这些占比分别约为 20.9%、21.2%、23.4%、15.9%。阈值改变时事件数量大幅变化，不能引用某个事件数作为固定的独立样本数。

### 2.2 幅值、持续时间、耦合与限幅

{table(pd.DataFrame(ranges))}

归一化命令距允许边界不超过 0.02 定义为“贴近限幅”，并不证明实际执行器已饱和。总体限幅时间较少；右舵面约 0.449%，但个别航次达 6.79%，不能只看总体均值。电机最长约 119.7 s 没有达到 0.02 的有效变化；方向舵约 18.6 s。没有任何航次的整段 p99−p01 四通道范围小于 0.02，但局部长期低激励仍存在。

主阈值事件间隔中位数：电机约 0.200 s、左舵面 0.140 s、右舵面 0.140 s、方向舵 0.219 s。事件明细及间隔保存在 `effective_events.csv.gz`；其数量与独立按连续区间累计的事件计数已复核一致。

![控制幅值时间加权分布](results/figures/control_distributions.png)

左右舵面指令的时间加权相关系数为 **−0.730**；main_50hz / sparse_10hz 分别约 −0.711 / −0.762。相邻指令增量的左右相关分别约 −0.591 / −0.757。全通道标准化增量相关矩阵的条件数约 3.90 / 7.59，均非奇异；将其他通道线性分量去掉后，左右舵面仍有约 65% / 42% 的增量方差。因此不是“四路完全一起变化”，但这种全局代数独立性不能证明在相同状态条件下有可辨识响应。

![控制相关性](results/figures/control_correlation.png)

逐航次幅值、近限幅比例、模式/速度段/高动态/REVIEW 条件的完整覆盖表见 `control_coverage_per_flight.csv`、`coverage_by_condition.csv`。这些是覆盖统计，不是控制效果估计。

## 3. 匹配方法：没有使用未来运动筛选

### 3.1 历史、初值和跨航次条件

- 使用既有 500 ms 历史窗口，在 −500:20:0 ms 的 26 个时刻取已经可用的历史状态保持值。稀疏日志仍只有原生约 100 ms 的信息，重复保持值不是新增观测；窗口可以包含左侧包围样本，实际支持长度略大于 500 ms。
- 比较 NED 速度三维向量、FRD 机体系角速度三维向量，以及 FRD→NED 姿态的 SO(3) 测地角。四元数先归一化、使用内积绝对值消除正负号二义性。保留完整航向；不通过旋转 NED 或对齐绝对位置人为创造近邻，绝对位置不进入特征。
- 同时比较历史扑翼频率、四路历史控制。完整原生历史必须频率/模式有效、无 failsafe、模式稳定；两端 nav_state 相同，硬件 UUID 相同、传动比差不超过 0.001。所有源时间不晚于对应决策时刻。
- 跨日志机械相位零点没有统一，**不比较绝对相位的 sin/cos，也不对未来真实相位或频率进行匹配**。频率只能约束周期速度，不能消除相位混杂。未来控制从 decision/publication 起算，采用原生事件半开区间 [t,t+H) 精确零阶保持回放；不以未来控制重建其他匹配输入。
- 主候选仅跨日志，排除同日志重复片段；允许 train–train、train–validation、validation–validation 描述性配对并分别报告。同日志长时间分离片段未计入，因此不是所有潜在对照的数学上限，不能从某通道 0 对推断其物理不可辨识。

历史各项按时间网格计算 RMS，**起点瞬时误差也必须分别通过相同物理阈值**：

| 设置 | 速度向量 m/s | 姿态 ° | 角速度向量 rad/s | 频率 Hz | 每路控制 |
| --- | --- | --- | --- | --- | --- |
| 严格 | 0.5 | 10 | 0.5 | 0.3 | 0.03 |
| 主设置 | 0.75 | 15 | 0.75 | 0.5 | 0.05 |
| 宽松 | 1.5 | 30 | 1.5 | 1.0 | 0.10 |

这些是可解释的审计容差，不是已证明可忽略的物理状态差。没有利用未来响应优化阈值；不估计跨 train/validation 的归一化尺度。完整枚举使用均值特征的必要条件做精确半径剪枝，然后检查全历史及起点；没有 KNN 数量上限，也没有按结果挑选匹配。

### 3.2 控制序列差异和去重

未来 H 上每路控制差异为 \(D_c=\sqrt{{H^{{-1}}\int_0^H (u_{{A,c}}-u_{{B,c}})^2dt}}\)。主“明显差异”阈值为 **0.05 / 0.10 / 0.10 / 0.05**；还要求绝对差异超过该阈值的持续时间占比至少 20%，且不少于 100 ms。事件时间并集积分完整保留区间内变化，无人工 20 ms 控制采样。数值使用 float32 原始命令、float64 时间加权累计。

“单通道对照”要求目标通道满足上述条件，其他三路未来 RMS 差异分别不超过 **0.02 / 0.04 / 0.04 / 0.02** 中的相应值。其他通道并非严格相等，不能称为严格干预。补充强化筛选要求目标未来 RMS 至少为 max(历史控制 RMS 差、起点控制差) 的两倍；这专门排除主要由已有配平差异解释的对照，不使用未来运动。

去重先去掉无序重复对，再按 16 项历史/初值归一化误差分数从小到大贪心选取。同一日志中任何已选窗口的完整源支持区间 [历史源起点,未来支持终点] 不得重叠，端点接触也拒绝；另外给出每航次只用一次的贪心配对数。它们都是保守支持数量，不是最大图匹配、统计有效样本量或独立随机试验数。子集变化时贪心数不保证单调；不同通道/分组各自选取，不能简单相加。

### 3.3 匹配数量与质量

500 ms / 1 s 原注册窗口共 179,879 / 176,984 个，满足历史频率、状态模式和精确未来控制时间覆盖的匹配候选为 177,205 / 174,354 个，来自 160 段飞行。`window_eligibility.csv` 逐航次记录差异；频率/状态/failsafe 不满足的窗口分别有 1,166 / 1,140 个，其余主要为模式变化或精确时间支持边界，不能把所有剔除归因于同一原因。

{table(primary[['horizon_ms','channel','raw_pairs','matched_windows','flights','nonoverlap_pairs','nonoverlap_flights','flight_disjoint_pairs']])}

500 ms / 1 s 只有约 **0.85% / 1.22%** 的合格候选窗口能找到主设置控制差异对。这里的 flights 为原始对涉及的航次数，nonoverlap_flights 为贪心去重后涉及航次。单通道行已经要求其他通道接近，不仅是某一通道有变化。

{table(s.query('caliper=="primary" and contrast_scale==1 and channel=="any" and group!="all"')[['horizon_ms','group','raw_pairs','nonoverlap_pairs','nonoverlap_flights']])}

主设置无重叠综合对的误差中位数如下；例如历史速度仍相差约 0.55 m/s，角速度相差约 0.52 rad/s，因此不能当作相同初始状态：

{table(q)}

完整差异范围及 p01/p10/p50/p90/p99/max 见 `selected_pair_quality.csv`，各通道未来差异也在其中。500 ms 综合对的未来控制 RMS 差中位数约为 0.044 / 0.090 / 0.097 / 0.023；由于“综合对”只需其中一路显著不同，其余通道不一定达到各自阈值。

### 3.4 容差敏感性与强化对照

{table(s.query('group=="all" and channel=="any"')[['horizon_ms','caliper','contrast_scale','raw_pairs','nonoverlap_pairs']])}

![阈值与数量取舍](results/figures/match_threshold_sensitivity.png)

宽松设置能得到数千个无重叠对，但容许 1.5 m/s、30°、1.5 rad/s 的历史及初始差异，不能用它替代严格控制响应证据。严格设置两个时域都只有 5 对。控制差异阈值加倍后主设置分别只有 51 / 113 对综合样本，表明“有效覆盖充分”并不稳健。

强化未来/历史差异比 ≥2 的结果：

{table(strength)}

同固件限定后，主设置综合无重叠对降至 260 / 269，单电机为 24 / 7；它并没有弥补独立舵面响应缺口。所有阈值和原始/去重计数见 `match_sensitivity.csv`、`same_firmware_sensitivity.csv`、`strengthened_contrast_sensitivity.csv`。

## 4. 航次、状态与验证集证据

### 4.1 条件覆盖集中于稳定巡航

主设置综合无重叠样本的历史平均速度范围约 **4.15–10.62 m/s**，10%–90% 主要在 **5.52–8.77 m/s**；平均角速度模长主要在 **0.44–0.74 rad/s**，频率主要约 **3.34–4.05 Hz**。没有覆盖历史平均速度 <4.15 m/s 的合格主设置对。低速、起降、强机动、低频/停扑翼、接近控制限幅区域缺少可比对照；不能从整体飞行数据包含这些状态就认定控制响应验证也覆盖它们。

| 分组（独立去重，不相加） | 500 ms 无重叠对 | 1 s 无重叠对 |
| --- | --- | --- |
| AUTO_MISSION（nav_state=3） | 289 | 299 |
| STAB（nav_state=15） | 34 | 40 |
| AUTO_LOITER（nav_state=4） | 0 | 0 |
| 历史内任一原生角速度模长 >2 rad/s | 3 | 5 |
| 整个窗口含 REVIEW | 2 | 2 |

模式整数为数据中的原值；名称参照本地 PX4 VehicleStatus 定义。高动态采用完整原生历史任一步超过 2 rad/s，不把平均角速度阈值和瞬时高动态混为一谈。REVIEW 保留且单独标记，未修改已有排除规则。这些子集太小，不支持强机动或 REVIEW 区间的泛化结论。

![500 ms 航次覆盖](results/figures/flight_support_500ms.png)

![1 s 航次覆盖](results/figures/flight_support_1000ms.png)

### 4.2 未见航次必须单独看 validation–validation

{table(strata[strata.grouping=='split_pair'][['horizon_ms','label','raw_pairs','nonoverlap_pairs','flights','flight_disjoint_pairs']])}

对于以训练航次拟合的模型，train–train 和 train–validation 不能等同于两端均未见的响应差异验证；同一天的不同航次也可能共享风况和控制策略，航次数不能直接当作 IID 样本数。

- `main_50hz`：validation–validation 综合主对 21 / 24；强化后 18 / 21。单电机 0 / 0，独立左右舵面 0 / 0；方向舵主设置仅各 1 对，强化后也为 0。
- `sparse_10hz`：validation–validation 综合主对 33 / 36；强化后 26 / 31。强化单电机只有 2 / 2 对，分别涉及 2 / 4 段航次。这些不能充当 50 Hz 短时控制响应的数据覆盖。
- 强化电机对在全部分区中，500 ms 有 12 对：main_50hz 8 对（7 个训练内、1 个训练–验证），sparse_10hz 4 对；1 s 仅 3 对（1 个主组训练内、2 个低频组验证内）。支持区域以速度约 5–9.3 m/s、频率约 3.4–4.5 Hz、motor 约 0.57–0.83 为主。

因此，最有希望的单通道是电机，但目前不足以对既有主组模型做稳定的未见航次独立电机响应验证。详细交叉表在 `response_evaluation_support.csv`；未列出的通道/组/划分组合为零。

## 5. 典型匹配可视化

以下样例按历史/起点匹配分数选择；没有看未来运动结果。左侧展示原生历史速度和角速度模长、有效频率、相对同一参考姿态的测地角；真正筛选仍使用三维向量和完整 SO(3) 误差，而不是仅匹配模长。右侧展示原生四路命令，虚线为各自决策时刻。强化电机样例满足未来差异至少为历史/起点差异两倍。

{figures}

其他方向舵主设置样例也保存在 figures 中；它们不通过强化筛选，不能当作可靠单方向舵证据。图中的未来只画控制命令，不画或比较未来真实运动效果。

## 6. 能支持什么，不能支持什么

1. **存在控制变化：成立。** 四路都有变化和一定非共线增量，但存在长时间低激励、左右舵面耦合和状态覆盖集中。
2. **控制效应有一定可辨识性：只具备探索条件，尚未证明效应已被识别。** 可以用冻结的少量匹配集做模型控制敏感性、混合控制序列差异诊断；必须同时报告初始误差、飞行簇和不同筛选标准。单舵面和方向舵证据不足；500 ms 电机相对最值得先补数据。1 s 综合对稍多，但独立电机对更少、未来多通道反馈更难保持接近。
3. **模型反事实预测已被证明：不成立。** 本轮不读目标状态来计算控制效应，没有评价网络，更没有独立干预真值。

闭环控制会随参考轨迹、状态误差和未测扰动改变，因此未来记录命令与后续运动共同受到反馈影响。匹配历史不能消除风、空速/迎角、载荷、电池状态、执行器滞后、残余相位、固件差异和初始误差混杂；相同 NED 地速不意味着相同气流。500 ms 历史控制接近也不能证明实际舵角或电机状态接近。即便后续模型准确预测这些真实片段之间的运动差异，也只是受这些条件约束的观测性证据。

### 下一步优先顺序

- 先冻结 main_50hz 验证内的 500 ms 综合对照，做有限探索性检查；同步报告 1 s、严格阈值、同固件和强化差异子集，不能只选效果最好的筛选条件。不得据此宣称控制可用。
- 设计额外安全飞行激励时优先电机：在稳定巡航、相近速度/姿态/频率下获得重复的不同指令序列，尽量控制其他通道并记录真实执行器/电机反馈；按独立航次预留验证，不能靠更多重叠窗口扩充样本数。当前数据不支持直接给出可安全执行的激励幅值或 MPC 方案，本轮也不实施飞行试验。
- 左右舵面应分别检查同向/反向组合及单通道变化的独立覆盖，再考虑方向舵；不能从负相关的舵面指令单独拆出可靠气动响应。需要统一或可追溯的机械相位零点，补充空气相对状态及延迟证据。
- 补充低速、高动态、模式转换和控制极限附近的数据后，再扩大响应验证域。数据补充和真实干预结果到位前，不将匹配片段称为严格反事实真值。

## 7. 验证、复算与归档

- 11 项定向测试全部通过：封存分区与目标列访问保护、累计事件与防重复间隔、半开控制区间及短事件、向量化积分对独立标量实现、四元数符号和 SO(3)、完整支持区间去重、半径剪枝必要条件、历史/初值隔离，以及全量结果的来源/划分/排除区间/计数一致性。
- 每个时域固定随机抽取 100 个已选对，独立从原始历史字段及原生事件重新计算历史误差、控制 RMS 和持续时间占比；校验通过。命令 RMS 数值容差 1e-7，持续时间占比容差 1e-12；未利用未来状态。
- 四个原始日志抽查均通过；读过的数据哈希在复算时再次检查。完整主计算约 166 s（不含后处理和独立复算），Python 使用指定 flap-train-gpu 环境，分析仅 CPU，无依赖安装。
- 代码、配置、中文报告、CSV/GZip 统计和 PNG 图表提交到独立本地实验分支；完整宽松候选对及窗口 Parquet 保留在本实验目录的本地 results 中，约 0.65 GB，不加入 Git。它们可由同版本数据重建，哈希登记在 `archive_manifest.json`。未推送远端。

复现步骤见 [README.md](README.md)。统计入口为 [match_sensitivity.csv](results/match_sensitivity.csv)、[response_evaluation_support.csv](results/response_evaluation_support.csv)、[selected_pairs.csv.gz](results/selected_pairs.csv.gz)、[run_manifest.json](results/run_manifest.json)、[scalar_verification.json](results/scalar_verification.json)。审计基础版本及运行时代码哈希记录于清单；最终 Git 提交以交付说明为准。
'''
(HERE/'report_zh.md').write_text(report)
print('report_zh.md written',len(report),'characters')
