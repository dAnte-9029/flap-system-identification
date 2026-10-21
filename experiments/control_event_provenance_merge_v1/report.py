import json
import numpy as np,pandas as pd
from common import *

def table(d):
 def fmt(v):return f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v)
 return '\n'.join(['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |']+['| '+' | '.join(map(fmt,r))+' |' for r in d.itertuples(index=False,name=None)])

def main():
 s=pd.read_csv(OUT/'paired_comparisons.csv');u=pd.read_csv(OUT/'training_intervals_per_flight.csv');r=pd.read_csv(OUT/'raw_control_per_flight.csv');cache=json.loads((OUT/'cache_statistics.json').read_text())
 fw=pd.read_csv(OUT/'firmware_logging.csv');sources=json.loads((OUT/'source_manifest.json').read_text())
 assert len(sources)==153 and all(x['status']=='verified_exact_commit' for x in sources)
 unique=u.groupby('split')[['unique_intervals','tokens','redundant_tokens','count1','count2','count3plus']].sum().reset_index();unique['两段比例 %']=100*unique.count2/unique.unique_intervals
 raw=r.groupby(['split','group']).agg(flights=('flight_id','nunique'),motor_period_median_ms=('motor_period_median_ms','median'),servo_period_median_ms=('servo_period_median_ms','median'),motor_sample_age_median_ms=('motor_sample_age_median_ms','median')).reset_index()
 global_table=s.query('scope=="all" and seed=="all"')[['horizon_ms','metric','native_rmse','merged_rmse','delta','change_pct']]
 affected=s.query('scope=="affected" and seed=="all"')[['horizon_ms','metric','windows','flights','native_rmse','merged_rmse','change_pct']]
 seeds=s.query('scope=="all" and seed!="all" and metric=="velocity_m_s"')[['horizon_ms','seed','native_rmse','merged_rmse','change_pct']]
 text=f'''# 控制事件来源审计与精确合并后的轨迹误差

**两项结论：现有事件条目主要按日志记录时刻及状态区间边界构造，不代表一次次独立物理动作；只在推理时合并完全同值的冗余事件，没有改善现有模型的整体轨迹预测精度。** 这不能证明模型完全依赖日志捷径，但足以说明训练控制编码需要重新审视，不能把一个推理端补丁等同于问题已解决。

基于 `1ea494d`，独立分支 `exp/control-event-provenance-merge-v1`。所有新代码和结果位于本目录；未修改数据集、旧模型、训练/正式推理代码或 PX4。测试集保持封存。

## 1. 从 PX4 到 GRU 的实际生成链

| 环节 | 实际行为 | 物理含义与限制 |
| --- | --- | --- |
| 控制分配器 | 发布 `actuator_motors.control[0]` 及 `actuator_servos.control[0:3]`；不以四路数值是否改变作为发布条件 | 归一化电机/舵面设定值，不是实测推力或舵角 |
| 消息时间戳 | `timestamp` 为发布时系统时钟；`timestamp_sample` 为该控制响应依据的数据采样时刻 | 不是执行器实际达到位置/推力的时间；也不是写入 ULog 磁盘的时刻 |
| logger | 对主题订阅限频，只保存实际被 logger 接收的记录；相同主题后加入的 profile 可覆盖记录间隔 | 记录次数受发布频率、logger 配置和调度影响；不能保证捕获每次控制更新 |
| 数据集提取 | `topic_frames()` 读取实例 0，保留每条日志中的 `timestamp` 和控制值；飞行内裁剪并带边界上下文 | 不做按值变化检测，不合并相同控制消息，不重采样原生命令 |
| 训练打包 | `interval_events()` 取 motor/servo 发布时间并集，在每个原生状态区间起点额外插入当前保持值 | 同时包含实际记录边界和人为状态区间边界；同一物理保持段可被拆开 |
| 持续时间 | 各分段起点至下一起点或状态区间终点的差；四路命令各自向前零阶保持 | 是日志重建函数的持续时间，不保证是真实硬件命令保持时长 |
| GRU 输入 | `[四路命令, duration/0.02]` 逐事件进入 event GRU，按有效条目数取最后嵌入 | 条目数、分段时长和发布相位都会影响编码，即使连续重建函数相同 |

对应代码：`src/system_identification/data/velocity_rate_dataset.py:116`（日志字段读取）、`:392` 附近（边界裁剪）；`scripts/prepare_velocity_rate_learning.py:41`（时间并集、起点插入及 ZOH）；`src/system_identification/models/velocity_rate_gru.py:97`（事件递归）和 `:99`（依事件长度取嵌入）。完整代码哈希在两个 manifest 中。

这里的“事件”并非 `hall_event`、脉冲触发或执行器动作检测事件，只是控制命令序列的分段表示。motor/servo 常由同一次 allocator 调用发布、使用相同时间戳；并集会把相同时刻合为一个四路条目，不等于两个独立控制动作。

## 2. 固件与实测记录节奏的证据

对 train/validation 161 个飞行片段中的 17 个已知固件 SHA 核对了 153 个精确版本源码文件（本地 Git 对象 63 个，按完整提交 SHA 从项目远端取得 90 个）。覆盖 160 个飞行；另一个重建日志没有固件身份，保持未知，不替它推断固件。原始 QgcLogs 文件未重新读取，使用注册数据集保存的来源元数据及原生控制 Parquet，并重新核对哈希。

已知 main_50hz 的 14 个固件版本具有 FLAPPING_DATASET profile（bit 12），其中 motor/servo 的记录间隔配置为 **20 ms**。三个 sparse_10hz 旧版本没有该 profile，实现中的默认 actuator 记录间隔为 **100 ms**。允许分区的来源参数均记录 SDLOG_PROFILE=4097；旧版本不识别 bit 12 时不能套用新版本含义。logger 还支持 SD 卡 `logger_topics.txt` 覆盖，原始 SD 卡配置未独立取得，因此这里区分源码配置、来源参数和下表实际记录节奏，不声称参数本身能排除所有覆盖。

可复核的精确源码链接：[main 固件日志配置](https://github.com/jiashunjiang2022/PX4-Autopilot/blob/926693950a4b2fe448d4b18d859c54bf9aa8a260/src/modules/logger/logged_topics.cpp)、[旧 sparse 固件日志配置](https://github.com/jiashunjiang2022/PX4-Autopilot/blob/229adb4c62d9883a7a0d7e9cbc65ac319f5de97d/src/modules/logger/logged_topics.cpp)。全部本地快照、路径和 SHA256 在 `results/source_manifest.json` 及 `source_objects/`，没有修改 PX4 仓库。

以下对每航次先计算中位数，再列组内中位数，单位 ms：

{table(raw)}

注意：约 0.1–0.3 ms 的 `timestamp - timestamp_sample` 不是机械响应延迟；它只是软件发布相对其输入采样的间隔。记录间隔约 20/100 ms 更不等于执行器机械响应时间。

**不能从相邻已记录值推断中间一定没有更快、未记录的控制更新。** 日志 ZOH 是目前模型采用的可复现观测契约，不是已认证的真实连续执行器输入。不能把合并后名义保持时间解释为硬件真正一直保持了这么久。

输出驱动侧的 `FunctionServos` 主要更新最新设定值，`FunctionMotors` 对最新设定值进行静态映射；未发现“日志中每一条相同记录都代表一次物理重置”的依据。但记录消息不是完整硬件协议，本轮没有认证 ESC/舵机全部复位、超时或实际输出时序，不能把记录次数当作这类语义的替代物。

## 3. 现有训练编码里，日志节奏有多强

检查了实际 1 s 模型训练缓存的 **107,920 个训练窗口 / 94 航次**及 **36,332 个验证窗口 / 27 航次**。先按 `(flight_id, 原生状态步)` 去重，避免重叠窗口把同一事件重复当成证据：

{table(unique)}

训练约 **99.31%**、验证约 **99.74%** 的唯一状态区间含两条事件：一条是区间起点处的保持值，另一条通常来自区间内记录的发布。两段并不代表两个独立执行器动作。按重叠训练窗口加权计算时两段比例略有差别，详见 `cache_statistics.json`，不混用两种口径。

每个区间都至少有一条人为起点条目；在当前编码中，它们约占全部有效 token 的一半。训练各航次首段时长的中位数再取中位数约 5.127 ms，验证约 5.132 ms，约为状态步长的四分之一。这些分段值同时受到状态时间轴与控制记录发布时间的相位关系影响，不能全部解释为新的物理激励。

四路同时完全不变的记录其实很少：main 训练原生并集记录 668,984 条，仅 181 次相邻四路全同；验证 220,294 条，仅 89 次。单通道重复更常见，例如电机在训练/验证并集相邻记录中分别有 183,909 / 88,845 次保持相同，但其它舵面可以同时变化。**只合并整条四路向量完全相同的事件，不能消除“每次日志更新都进入 GRU”这一整体编码特征。** 不能为了减少条数而擅自把近似相等的舵面数值也合并，那会改变记录控制函数。

事实与推论要分开：上一轮等价分段实验已证明模型输出依赖这种表示方式；本轮证明这种表示方式由日志及打包规则决定。仍未定量分解已有预测精度中有多少来自真实控制值、多少来自发布节奏或其它闭环信息，不能宣称所有精度都是日志捷径。

## 4. 精确合并后，真实轨迹预测误差如何变化

使用冻结 1 s GRU-Control 的三个种子 17/29/43，完整评价同一批 36,332 个 main_50hz 验证窗口。只在各原生状态区间内合并**四路浮点控制值完全相同**的相邻条目，在整数微秒轴上相加持续时间；保留每次真实变化时刻。历史、初态、dt、状态步数、批次顺序、模型参数、相位外推及张量 padding 形状完全一致，没有改变正式推理函数。

对所有实际合并区间，在原/合并边界并集逐段验证 ZOH 函数严格相同；不只比较几个采样点。绝不跨状态更新步删除必要的控制信息。

自然冗余仅涉及 **88 个唯一验证状态区间**，因窗口重叠出现在 728 个区间实例中，共影响 1 s 的 708 个窗口（1.949%）、24 航次；500 ms 内已发生合并的有 362 个窗口。其余窗口的预测逐元素完全不变。三个种子全部实际在 RTX 4090 推理，原生与合并各跑一次；完全相同输入重复推理结果相同。

评价的是预测状态对记录真实状态的误差，不是上一轮两条模型预测之间的差异。速度和角速度为向量范数，姿态为 SO(3) 测地角，位置为积分轨迹位置误差；单位分别 m、m/s、度、rad/s。主表先按每航次平均误差平方，再对航次和种子等权，最后开方；另存平均范数和窗口等权结果。

{table(global_table)}

正的 change_pct 表示合并后更差。**总体几乎不变，没有实质精度收益：位置略改善，速度、姿态、角速度总体略恶化。** 速度 RMSE 的恶化约为 0.004%，远小于通常的航次差异，不能将其宣传成显著退化；同样没有依据宣称精确合并能提高普通轨迹精度。

仅看发生合并的窗口，变化稍大但仍混合：

{table(affected)}

受影响子集上，500 ms 速度约恶化 0.414%，1 s 约恶化 0.206%；姿态略恶化，角速度略改善。不同子集使用各自航次等权，因此其方向不必与全体权重下的微小结果完全一致。并未根据误差挑选受影响样本，只根据是否有精确重复事件定义子集。

三个种子各自的全体验证速度结果：

{table(seeds)}

![逐航次速度误差变化](results/merge_effect_per_flight.png)

模型种子和重叠窗口不是独立飞行样本。本次给出配对描述结果，不把这些很小的差异宣称为统计显著的性能变化。

## 5. 对训练控制编码的判断

**仅修补推理代码不足以解决已发现的表示依赖，训练数据及编码契约确实需要重新考虑。** 精确合并可以作为消除显式冗余的第一步，但本轮只改动约 2% 的验证窗口，其余大多数原生发布值不同；日志节奏相关的事件结构仍然存在。而对从未在训练中见过的大量额外分段，上一轮已显示明显不稳定。

下一步应先明确目标：是预测“当前日志记录机制下的离散控制条件轨迹”，还是预测“与记录机制无关的物理控制响应”。后者至少需要：

1. 区分发布时刻、实际执行时刻、日志采样时刻及数据可用时刻；记录和验证更完整的控制更新/执行器输出，不能把更密重采样当作补回缺失真值。
2. 在保留真实控制变化及持续时间信息的前提下，让编码对同值拆分/合并保持不变；不要单纯把事件 GRU 的条目数当作物理时间推进。固定网格、时长积分特征或时间一致的编码各有信息损失/训练代价，需要另行对照，不能直接选一种上线。
3. 训练、验证和部署采用一致的控制表示；增加等价分段不变性验收，并同时检查真实轨迹误差与控制响应差异。若改变输入语义或编码结构，通常需要重新训练/至少重新适配并验证，不能靠推理端规范化自动获得物理正确性。

本轮不修改网络、不重训、不开展新的架构搜索或 MPC；上述是基于审计的后续建议，而非已验证修复。

## 6. 验证与复现

8 项定向测试覆盖精确/近似值区别、全部 728 个真实合并区间的全函数等价、数据划分/冻结哈希、未变化窗口预测不变、全部 36,332 窗口 × 3 种子 × 2 表示 × 2 时域 × 4 指标的独立 NumPy/SciPy 复算、逐航次聚合及源码函数边界解析。结果见 `tests.log`。

源码中换行形式不同的函数声明曾使自动摘取范围跨过函数边界，现已改为花括号配对并增加回归测试，来源统计已重算；模型评价不受此文本解析修正影响。所有最终归档结果来自修正后的审计。

`results/endpoint_errors.parquet`、`predictions.npz` 保存完整验证预测与误差，`per_flight.csv`、`paired_comparisons.csv` 保存配对结果；`source_manifest.json` 与 `source_objects/` 保存精确固件来源，`raw_control_per_flight.csv`、`training_intervals_per_flight.csv` 保存原生及去重统计。数据/缓存/权重 SHA256 位于 provenance/evaluation manifest；所有新文件由 `archive.py` 归档。复现见 [README.md](README.md)。
'''
 (HERE/'report_zh.md').write_text(text);print('Report written')
if __name__=='__main__':main()
