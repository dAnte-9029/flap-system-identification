"""Render the Chinese report from archived statistics; never selects evaluation samples."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from evaluate import HERE,OUT

def table(df):
    def cell(v):
        if isinstance(v,(float,np.floating)):return '—' if not np.isfinite(v) else f'{v:.3f}'
        return str(v)
    return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+['| '+' | '.join(cell(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)])

def main():
    s=pd.read_csv(OUT/'response_summary.csv');c=pd.read_csv(OUT/'paired_comparisons.csv');b=pd.read_csv(OUT/'swap_summary.csv')
    p=pd.read_parquet(OUT/'cohorts.parquet');inv=pd.read_csv(OUT/'source_inventory.csv')
    cov=json.loads((OUT/'coverage_diagnostics.json').read_text());verify=json.loads((OUT/'independent_verification.json').read_text())
    main=s.query('cohort=="signal" and split=="validation" and seed=="all" and point_ms==horizon_ms')
    core=c.query('cohort=="signal" and split=="validation" and point_ms==horizon_ms')
    primary=core[core.metric.isin(['velocity','rate'])][['horizon_ms','metric','control_rmse','nocontrol_rmse','improvement_pct','better_pairs','pairs']].copy()
    primary.columns=['时域 ms','指标','Control RMSE','NoControl RMSE','改善 %','改善样本对','总对数']
    metric=s.query('cohort=="signal" and split=="validation" and seed=="all" and point_ms==horizon_ms and metric in ["velocity","rate"]')
    details=metric[['horizon_ms','metric','model','response_mae','amplitude_mae','true_response_rms','direction_accuracy','effective_pairs']].copy();details['direction_accuracy']*=100
    details.columns=['时域 ms','指标','模型','响应 MAE','幅值 MAE','观测差异 RMS','方向正确 %','有效方向对数']
    seeds=s.query('cohort=="signal" and split=="validation" and seed!="all" and point_ms==horizon_ms and metric in ["velocity","rate"]')
    seeds=seeds.pivot(index=['horizon_ms','metric','seed'],columns='model',values='response_rmse').reset_index()
    seeds['改善 %']=100*(1-seeds['GRU-Control']/seeds['GRU-NoControl'])
    seedtab=table(seeds)
    deps=[]
    for key,g in p.groupby(['horizon_ms','cohort','split']):
        deps.append(dict(zip(['时域 ms','队列','划分'],key))|{'对数':len(g),'航次':len(set(g.flight_a)|set(g.flight_b)),'依赖组':g.component.nunique(),'航次互斥对数':int(g.flight_disjoint.sum())})
    dep=table(pd.DataFrame(deps))
    sup=main[main.metric.isin(['position','attitude'])][['horizon_ms','metric','model','response_rmse','response_mae','amplitude_mae']]
    train=c.query('cohort=="signal" and split=="train" and point_ms==horizon_ms and metric in ["velocity","rate"]')[['horizon_ms','metric','control_rmse','nocontrol_rmse','improvement_pct','pairs']]
    strong=c.query('cohort=="strengthened" and split=="validation" and point_ms==horizon_ms and metric in ["velocity","rate"]')[['horizon_ms','metric','control_rmse','nocontrol_rmse','improvement_pct','pairs']]
    noise=s.query('cohort=="small_control" and seed=="all" and model=="GRU-Control" and point_ms==horizon_ms and metric in ["velocity","rate"]')[['horizon_ms','split','metric','pairs','flights','true_response_rms']]
    bmain=b.query('cohort=="signal" and split=="validation" and model=="GRU-Control" and seed=="all" and point_ms==horizon_ms and metric in ["velocity","rate"]')[['horizon_ms','metric','swap_rms','common_grid_response_rms','timing_refinement_rms','history_direction_agreement','native_common_direction_agreement']].copy()
    bmain.iloc[:,-2:]*=100
    bmain.columns=['时域 ms','指标','原生交换 RMS','共同事件网格交换 RMS','相同命令细分 RMS','两历史响应同向 %','原生/共同网格同向 %']
    timings=pd.read_csv(OUT/'swap_timing.csv')
    alignment=pd.read_csv(OUT/'initial_alignment_sensitivity.csv');alignment['mse']=alignment.response_rmse**2
    alignment=alignment.groupby(['horizon_ms','metric','model','truth_anchor']).mse.mean().pow(.5).reset_index(name='RMSE')
    text=f'''# 第一轮控制响应预测验证：已观测混合控制变化的探索性证据

日期：2026-10-21。实验分支：`exp/control-response-eval-v1`，基于审计提交 `1745345`；冻结模型来自 `8964b44`。完整代码版本以包含本报告的本地 Git 提交为准。

## 结论与适用范围

**在有限的 main_50hz 验证航次匹配样本中，GRU-Control 对实际运动增量差异的预测优于 GRU-NoControl，三个训练种子的主指标改善方向一致。证据仍属于探索性观察结果，不能证明反事实控制响应准确。**

500 ms 有 **21 对、16 个验证航次**，1 s 有 **24 对、17 个验证航次**。按共享航次构图，两者都只有 **3 个依赖组**；若要求每个航次只出现一次，均只剩 **7 对**。不能把 21/24 对或三个模型种子当作同等数量的独立试验。

速度差异预测的改善较明确；500 ms 角速度的改善在航次互斥子集上反转。更严格状态匹配下验证样本为零。实验 B 显示模型会随控制值改变而响应，但还发现：**保持连续控制信号相同，仅改变事件分段，也会改变 Control 输出**。共同事件网格检查确认控制值敏感性仍存在，但此编码依赖需要后续专门处理。本轮未改网络或权重。

## 1. 来源、数据边界与实际执行

- 显式使用 `configs/data/velocity_rate_dataset_registry.yaml` 注册的 `velocity_rate_multistep_v1`。清单 SHA256：`8afb8623bf7174e4da5b2044f644331b8b251433fed3576f6dcdab470935bcb5`。读取 train/validation 的状态及原生四路命令，重新核对 8 个 Parquet、共享元数据及来源/排除/REVIEW 表哈希。33 段测试飞行的数据文件和原始日志均未读取。
- 源审计覆盖 train 122、validation 39 段；本次复用其归档匹配候选及过滤规则，不重新搜索或放宽条件。先冻结队列及哈希，再读取未来运动标签。审计文件路径与 SHA256 在 `results/selection_manifest.json`。
- main_50hz 内 train–train 与 validation–validation 分开；train–validation 和 sparse_10hz 只统计可用性，不做模型推理。验证航次未参与梯度训练，但已有 checkpoint 曾按验证集选择，因此这是既有验证集上的后续诊断，并非新封存盲测。
- 六个冻结 checkpoint：Control/NoControl × seeds 17、29、43；均为 1 s 多步训练 GRU64、37,590 参数。未训练、未调参、未重新选择 checkpoint。完整权重 SHA256 在 `results/run_manifest.json`。
- 实际使用 `/home/zn/anaconda3/envs/flap-train-gpu/bin/python`、PyTorch 2.7.0+cu128、RTX 4090。完整数据校验、打包和 GPU 推理约 20.92 s；最大 PyTorch 已分配显存约 97.2 MiB（非设备总占用）。分析、独立复算与出图耗时另计。
- 原审计、模型、训练程序、数据集及 QgcLogs 未修改；新实验独立 worktree，所有新文件位于本实验目录。

## 2. 匹配、去重与有效样本量

历史上下文为 500 ms。沿用审计的历史 RMS 与末态双重门限：速度 0.75 m/s、SO(3) 姿态 15°、机体系角速度 0.75 rad/s、频率 0.5 Hz、各路历史及初始命令差 0.05。严格门限对应 0.5 m/s、10°、0.5 rad/s、0.3 Hz、0.03。跨日志、相同硬件 UUID、相同采样组、历史稳定且相同 nav_state，传动比差不超过 0.001；固件相同与否另作诊断。跨航次机械相位零点未统一，不将绝对相位当作匹配坐标。

命令量纲为归一化指令：电机 [0,1]、舵面 [-1,1]。显著差异要求至少一路未来命令 RMS 差达到 `[0.05,0.10,0.10,0.05]`，且超过该幅度的时间占比不少于 20% 并不少于 100 ms。它是**命令差异**，不是实测舵偏角/推力响应。strengthened 还要求同一路未来 RMS 差至少为历史 RMS 差、初始命令差较大者的两倍；strict 使用原严格状态门限。small_control 在主状态门限内要求四路未来 RMS 差均不超过 `[0.02,0.04,0.04,0.02]`，仅作为观察性参照。

各划分/采样组内沿用原 `disjoint_pairs`，按历史匹配 score、端点索引稳定排序，贪心去除完整历史—未来支持区间重叠；不使用未来运动结果。各队列分别去重，队列之间可能共享样本，不能相加当作新证据。原审计全体 323/339 对与本报告分层后的数量不是同一个统计池。

主 signal 队列的组别分布：

{table(inv.query('cohort=="signal"')[['horizon_ms','group','split_pair','raw_pairs','nonoverlap_pairs','flights','inference_requested']])}

实际推理队列及依赖结构（严格验证队列两时域均为 **0 对**，故表中无行）：

{dep}

主验证匹配仍不很接近：500 ms 历史速度差中位数 0.609 m/s、角速度差 0.571 rad/s、姿态差 5.944°；1 s 对应 0.605、0.558、6.570°。全部主验证 score >0.5，没有 strict 验证样本，不能声称消除了初态混杂。

这些样本的历史速度约 5.45–9.34 m/s，历史角速度均值范数约 0.43–1.13 rad/s，扑翼频率约 3.24–4.43 Hz。两时域主验证样本均无历史 REVIEW、无审计定义的高动态样本（历史角速度均值范数 >2 rad/s）。nav_state 仅覆盖数值 3 和 15；详细计数见 `coverage_diagnostics.json`，不扩展解读为覆盖所有飞行模式。独立单通道对照：两时域均只有 1 对方向舵，其余通道为 0；不足以估计任何单通道响应规律。

![匹配质量与误差](results/figures/matching_quality.png)

## 3. 实验 A 的定义与自主递推

对速度、角速度及位移，真实差异为 `r(t) = [x_A(t)-x_A(0)] - [x_B(t)-x_B(0)]`；预测同样从各自输入初态计算增量。没有用终点状态直接相减替代增量差。速度及位置用 NED，角速度用各机体 FRD 分量；这是同型机体轴分量比较，不是把两航次的角速度当成共同空间向量。姿态补充指标采用 NED 中的空间增量 `D_i=R_i(t) R_i(0)^T`，对比 `D_A D_B^T`，以预测与观测该相对旋转的 SO(3) 测地角计误差，四元数正负号不影响结果。

向量响应 RMSE 定义为 `sqrt(mean(||r_hat-r||²))`，不是逐轴 RMSE；MAE 为 `mean(||r_hat-r||)`；幅值 MAE 为 `mean(abs(||r_hat||-||r||))`。主汇总先在每个航次内平均与其相连的样本对，再对航次等权；三个种子平均误差平方后开根，**不是三模型轨迹集成**。另存逐对等权指标。

模型读取真实历史状态、允许的历史信息及 Control 的历史/已记录未来命令。未来步骤只有预测状态、递推隐状态和允许命令；没有未来速度、角速度、姿态或扑翼测量回灌。NoControl 不读取历史或未来命令。相位/频率处理完全沿用冻结推理：历史掩码、最后有效历史频率保持和相位因果外推；这不消除跨航次相位零点误差。

未来已记录命令由闭环控制器根据随后观测状态产生，本身携带反馈信息。本对照同时比较控制历史和未来命令提供的总体预测信息，没有把两者贡献单独分离；更低误差不能自动解释为独立控制干预的因果效果。

使用原生实际 dt，时间点取最近原生预测步，不插值制造观测真值。主验证所有报告点中，500 ms 队列两片段时刻最大差为 16.102 ms，1 s 队列为 13.684 ms；各自距名义点最大偏差为 10.836/11.990 ms。该误差会影响快速角速度响应比较，归档有每对实际时刻。

真实初态用原生对齐的 target 初态，预测用实际因果输入初态。二者角速度采样对齐可能有差异。额外把真实增量改锚至因果输入初态只作敏感性检查：角速度 RMSE 在 500 ms 为 Control 0.687 / NoControl 0.780，在 1 s 为 0.753 / 0.974；主结论方向未变。正式结果仍使用各自真实对齐初态，不修正观测以强迫闭合。

## 4. 实验 A：主验证结果

velocity 单位 m/s；rate 单位 rad/s。以下都是**运动增量差异预测误差**，不是普通单条轨迹 RMSE。

{table(primary)}

误差相对观测增量差异幅值较小，但这不等于控制因果效应被预测准确。额外指标：

{table(details)}

方向正确定义为预测与观测响应向量点积 >0，零预测算错；仅在观测响应幅值超过训练 small_control 参照的航次等权 P90（并应用固定物理下限）时统计。所有样本仍参与 RMSE/MAE。500 ms 速度/角速度阈值为 0.557 m/s、1.147 rad/s；1 s 为 0.592 m/s、0.965 rad/s。1 s 训练参照仅 4 对、3 依赖组，方向阈值可靠性不足；100% 不是全体样本方向正确，也不是统计保证。`direction_sensitivity.csv` 给出阈值 0.5/1/2 倍的全部结果，不据此挑选最优阈值。

三个种子分别计算的主验证 RMSE：

{seedtab}

主指标三个种子的改善方向一致，但种子只反映训练随机性，不能增加独立飞行证据。位置和姿态是补充：

{table(sup)}

position 单位 m，attitude 单位度；它们继承速度/角速度误差及离散积分、EKF 观测不一致，不能单独证明控制响应。

![响应误差增长](results/figures/response_error_growth.png)

两时域是分别匹配的队列，不能把跨队列端点差直接归因于预测时间；应在图中同一队列内看时间演化。速度误差随时域增加，角速度误差有明显波动。1 s 的相对改善更大，绝对误差也更大，不意味着更适合闭环控制。

![逐对比较](results/figures/paired_error_distribution.png)

![逐航次改善](results/figures/flight_error_gains.png)

## 5. 可靠性、状态混杂与观察性参照

样本依赖用共享航次连通分量处理。配置要求至少 5 个依赖组才给出按组 bootstrap 区间；主验证只有 3 组，**不输出置信区间或显著性声明**。完整单组/留一组结果在 `component_sensitivity.csv`。这些仍可能共享日期、控制器及风况，连通分量只是可观测依赖的一部分。

每航次只出现一次的保守子集均为 7 对，Control−NoControl RMSE 为：500 ms 速度 −0.109 m/s、角速度 **+0.041 rad/s**；1 s 速度 −0.362 m/s、角速度 −0.222 rad/s。因此不能称 500 ms 角速度改善对样本结构稳健。该子集按原历史匹配优先规则选取，不按未来响应选取，也不代表最大匹配数或精确有效样本量。

提高未来/历史控制差异比的 strengthened 验证结果仍有改善：

{table(strong)}

条件分组全量见 `condition_comparisons.csv`；这些描述性子组保留主队列航次权重后在组内重新归一化，不重新定义主队列。最大控制 RMS 差 ≥0.2 的主验证仅 500 ms 1 对、1 s 5 对；1 s 角速度在这 5 对上 Control 1.129、NoControl 1.151 rad/s，改善很小。两个时域各有 1 对跨固件匹配，个别指标 Control 更差；小样本不能支持固件效应推断。高动态和 REVIEW 没有主验证覆盖，不能给出这些工况的可靠优劣结论。

小控制差异参照中的真实运动增量差异 RMS：

{table(noise)}

这些差异同时含状态匹配误差、闭环反馈、风扰、执行器滞后、相位/频率不确定性及观测误差，**不是纯传感器噪声、不是严格预测误差下界，也不与 signal 队列完全同分布**。验证参照每个时域仅 1 对；1 s 训练参照仅 4 对。主 Control 误差虽然低于 signal 观测差异 RMS，却普遍高于训练 small_control 参照 RMS，无法证明误差显著小于观察性噪声水平。

训练集诊断单列如下，不能当作未见飞行证据：

{table(train)}

## 6. 实验 B：固定历史后的控制序列交换

分别固定 A 和 B 的完整历史及由此得到的模型隐状态，比较命令 A−B 的预测差异。重跑相同历史编码时隐状态一致，已测试。交换时按 `recipient_decision - donor_decision` 平移原生**发布时间**，不拉伸时间；保留接收方测量初态到决策时刻之前的自身命令。按接收方原生 dt 分割区间，保留区间内全部控制变化。命令值范围、半开区间、时长和自交换均核验。

只交换命令，不交换未来状态、真实相位或真实频率。极少数原生终点越过名义命令时域，按预先记录规则保持供体末条命令：全部推理队列中最大外延 500 ms 为 8.561 ms、1 s 为 7.703 ms；它是已声明的末值保持，不使用更远未来真值。

下表响应范数对固定历史 A/B 的平方值先平均，再按航次及种子汇总开根；原生交换、共同网格交换、事件细分检查使用同一汇总定义：

{table(bmain)}

原生交换让控制值和事件序列分段同时变化。为分开检查，增加两项无需重训的诊断：

1. **相同命令细分**：连续命令函数完全不变，仅加入供体事件时刻。独立复算核对 32,909 个区间的命令值完全相同，但模型仍有显著输出差异：1 s 速度约 0.304 m/s、角速度约 0.176 rad/s。这说明事件 GRU 对分段/重复事件 token 敏感；不能把所有原生交换输出都解释为纯控制值响应。
2. **共同事件网格交换**：两个命令选择使用相同事件时刻与区间时长，只改变四路命令值。响应仍明显，且总体与原生交换同向。该结果支持模型存在控制值敏感性；共同网格本身也改变了模型输入表示，不是反事实准确性修正器。

NoControl 在共同事件网格下交换输出完全相同，原生批次推理差异仅为浮点量级；完整状态最大约 2.4e-6。两历史的 Control 响应并非总同向：1 s 角速度一致率约 85.8%，与历史条件和非线性有关，不能直接判定为物理错误或合理因果效应。端点没有范数 <0.001 的主验证交换响应，但这个数值阈值不足以定义物理可用性。

![控制值与事件编码敏感性](results/figures/swap_and_event_encoding.png)

实验 B **没有**同一历史下未执行命令的真实运动标签，不能计算或宣称其反事实准确率。A 的运动差异标签也含初态混杂，不能作为 B 的严格真值。模型内对历史/控制贡献的对称分解仅为数值诊断，相关列见 `swap_metrics.parquet`，不是因果贡献比例。

## 7. 典型成功与失败案例

以下仅作可视化：从已冻结的主验证队列按端点速度/角速度归一化 MSE 改善最大/最小各取一例；归一化使用已有训练尺度。**事后案例排名不改变队列、指标、权重或模型选择**。阴影表示三个种子的范围，不是统计置信区间。完整航次 ID、样本对索引及选择规则在 `results/cases.json`。

'''
    for case in json.loads((OUT/'cases.json').read_text()):
        text+=f"### {case['horizon_ms']} ms / {case['label']} / {case['pair_uid']}\n\n"
        text+=f"![实验 A：实际与预测增量差异]({case['experiment_A']})\n\n![实验 B：固定历史敏感性，无反事实真值]({case['experiment_B']})\n\n![实际命令及各片段相对轨迹]({case['commands_and_paths']})\n\n"
    text+=f'''失败案例中，模型可以对控制产生清晰响应，同时真实运动差异预测仍不准确；两者是不同问题。例如 1 s 最差相对案例的两个历史下角速度响应不同，不能仅凭速度响应平滑或幅度明显就判定控制规律正确。

## 8. 能支持什么、下一步做什么

**观测事实**：主验证队列中 Control 的运动增量差异 RMSE 更低，三种子一致；共同事件时间网格下仍有明显控制值响应。这比单条轨迹 RMSE 优势多提供了一层支持。

**统计结论**：可以表述为“在已观测、混合控制变化与有限匹配条件下，模型显示一定响应差异预测能力”。不能表述为“各通道控制响应已准确辨识”或“反事实预测已验证”。独立依赖组少、严格验证匹配为空、噪声参照稀少及 500 ms 角速度子集反转，使稳健总体结论仍不足。

**研究推测及优先事项**：先复核事件编码对等价命令表示的敏感性，再在相近巡航状态（约 5.5–9.3 m/s、3.2–4.4 Hz）扩充独立航次和主动激励；以 500 ms 速度响应为主、1 s 为延伸，角速度需要更严谨的时间对齐和风况/相位控制。当前每个独立通道都缺证据，不能凭唯一方向舵样本决定某通道已可验证。后续采集宜设计单通道或正交混合激励，记录实际执行器响应与滞后，并重复覆盖相近状态，减少反馈和风扰混杂。本轮没有设计或执行飞行控制，也没有 MPC/控制可用性结论。

## 9. 测试、独立复算与归档

- 定向测试：11 passed。涵盖自身初态增量、SO(3)/四元数符号、发布时间交换/自交换、半开区间及命令合法性、航次权重与依赖分组、源划分和无重叠去重、飞行边界与排除区间、NoControl 隔离、未来标签污染不影响输出、相同历史隐状态、相位掩码、自主预测反馈、共同事件网格及置信区间禁用规则。
- 独立复算：{verify['raw_target_windows']} 个窗口逐项核对原始授权 Parquet 标签和初态；{verify['response_scalar_values']} 项响应值复算，向量最大差 {verify['max_vector_difference']:.1g}，四元数独立乘法与 SO(3) 指标最大差 {verify['max_quaternion_difference_deg']:.3g}°；逐航次独立聚合最大差 {verify['max_flight_aggregate_difference']:.3g}；{verify['equivalent_control_intervals']} 个共同网格控制区间校验。重新核对数据与 checkpoint 哈希。
- 模型预测全部有限，四元数范数最大误差约 1.2e-7。源码检查与运行时 hook 均确认每一步输入来自上一预测，未回灌未来真值。
- `results/predictions_*.npz` 保留事实/交换/细分/共同网格预测、实际 dt、原生事件、真值及索引；`response_metrics.parquet`、`swap_metrics.parquet` 保存逐对逐种子逐时刻结果；CSV 保存全部汇总、敏感性及来源分布。完整文件 SHA256 由 `archive.py` 生成，包含代码、配置、报告、图表和日志。
- 复现命令与文件说明见 [README.md](README.md)。依赖原审计大体量候选档案及注册数据；缺失或哈希不符直接失败，不回退数据集、不读取测试集。当前实验全部预测与指标也随本地提交保存，可无需再次推理复算。
'''
    (HERE/'report_zh.md').write_text(text)
    print('Chinese report rendered from archived metrics')

if __name__=='__main__':main()
