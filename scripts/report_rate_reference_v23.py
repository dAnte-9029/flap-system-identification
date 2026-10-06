"""Report the fixed conditional-prediction comparison without control promotion."""
import json
import numpy as np
import pandas as pd
from rate_reference_v23 import OUT


def main():
    e=json.loads((OUT/'evaluation.json').read_text())
    c=pd.read_csv(OUT/'comparison.csv');o=c[c.cohort=='oof']
    main=o[(o.kind=='wave')&(o.horizon==50)]
    table=[]
    for key in ['p','q','r','velocity','attitude']:
        z=main[main.metric==key];a=z.baseline.mean();b=z.rate.mean()
        table.append(f'| {key} | {a:.6f} | {b:.6f} | {100*(1-b/a):+.2f}% |')
    byseed=[]
    for r in main[main.metric=='q'].itertuples():byseed.append(f'| {r.seed} | {r.baseline:.6f} | {r.rate:.6f} | {r.improvement_percent:+.2f}% |')
    protect=o[(((o.kind=='wave')&(o.horizon==50)&o.metric.isin(['p','r']))|
               ((o.kind=='endpoint')&o.horizon.isin([25,50])&o.metric.isin(['velocity','attitude'])))]
    failures=protect[protect.improvement_percent < -5]
    lines=[f'- seed{r.seed}：{r.metric}，{r.horizon*.02:.2f}s {r.kind}，{r.baseline:.6f}→{r.rate:.6f}，退化{-r.improvement_percent:.2f}%。' for r in failures.itertuples()]
    hold=pd.read_csv(OUT/'reference_hold_diagnostic.csv')
    hold=hold.groupby(['seed','log_id','metric'])[['factual_mse','held_mse','effect_mse']].mean().pow(.5).groupby('metric').mean()
    htable=[f'| {k} | {r.factual_mse:.6f} | {r.held_mse:.6f} | {r.effect_mse:.6f} |' for k,r in hold.iterrows()]
    points=pd.read_csv(OUT/'source_points.csv')
    ages=points.reference_age_ms.quantile([.5,.95,1]).to_numpy()
    text=f'''# v23：角速度参考合同核验与同预算训练对照

已完成新输入合同、407个原train窗口重建和15组训练（5折×3种子，每组1600步）。**本次衡量的是给定日志角速度/推力参考的条件预测，不能将收益解释为原姿态参考模型的自主预测收益或新控制器效果。**保留全部窗口、原训练预算和既有模型基线；未访问开发验证、sealed或reserved。

冻结的条件预测比较门槛：**{'通过' if e['descriptive_gate'] else '未通过'}**。q完整1秒波形平均RMSE改善{e['primary_q_wave_improvement_percent']:.2f}%，{e['positive_flights']}/5飞行改善，所有种子均改善={e['all_seeds_q_positive']}。状态保护通过{e['protection_passed']}/{e['protection_total']}；原0.5秒活动窗口绝对门槛通过{e['absolute_active_passed']}/{e['absolute_active_total']}。**控制用途未晋升。**

## 输入合同实际改变了什么

- 原模型输入为姿态参考2维、yaw_sp_move_rate、推力、manual_yaw。新五个槽位为机体系p/q/r角速度参考（rad/s）、同消息推力参考（归一化需求，不是N）、reset_integral布尔指令。
- reset槽位保留控制接口含义，但本批全部为0，归一化固定mean=0/scale=1；未以无穷小标准差放大它。以后有重置事件的输入不在本次训练覆盖内。
- 新模型包含固定PX4角速度环、控制分配、执行器和飞机；不包含姿态控制器。非VTOL的Mission/Stabilized记录均核验通过，不能直接推广至Manual、Acro、VTOL或变更控制参数的固件。
- 精确日志固件仍为c9af5715。姿态模块把三轴机体角速度参考与姿态参考中的推力发布至同一条vehicle_rates_setpoint；角速度模块读取其中rate/thrust，推力可受电池缩放影响。本批该开关全为0，且未记录参数变更。
- 逐原支撑核验无落地、failsafe、角速度/姿态控制关闭或reset_integral事件。所有横向推力字段为0，纵向推力在[0,1]内。
- 从同一条rate消息提取全部五个字段，避免使用更新时刻不同的姿态推力拼接。两条话题因异步记录有小差异，最大约0.01801；这不是测得的推力误差。

源码依据沿用v21精确快照：[姿态参考到rate参考](../control_windows_v21/source/FixedwingAttitudeControl.cpp#L123)、[rate/thrust消费链](../control_windows_v21/source/FixedwingRateControl.cpp#L359)。源blob与日志固件逐飞行复核，详细参数快照见[source_audit.json](source_audit.json)。

## 时间、数据与公平比较

仍是原407个互不重叠父窗口、H26/K50、50Hz：历史状态和参考为origin−25..origin，历史执行命令为origin−26..origin−1，未来计划参考为origin..origin+49，标签为origin+1..origin+50。源消息用因果保持，无未来插值或拟合时间偏移。

31339个原支撑点的rate参考发布年龄中位/P95/最大为{ages[0]:.3f}/{ages[1]:.3f}/{ages[2]:.3f}ms；话题没有独立sample timestamp，不能把发布年龄称为物理采样年龄，也不能说50Hz记录完全还原了更快的控制器输入。

逐项验证全部非参考输入和标签与v8原train数组完全相同。新数据仅保存407个train窗口。非参考归一化沿用对应v8折；参考归一化仅使用同折四个fit飞行的唯一时间行，查询飞行不参与。单位重力GRU64、40070参数、导数限制、完整1秒损失和优化器均未改变，不叠加v18首步q辅助损失。

15组均从与对应v8 hold相同的可训练参数初始化，直接复用其1600×32个抽样索引；最终固定取1600步，未选最佳检查点。旧v8输出和新输出比较同一批真值，但**信息边界不同，因此不是同一预测任务中的纯架构优劣比较**。当前训练飞行反复用于开发，OOF不是未开封独立测试。

## 真实记录下的条件预测结果

以下为完整1秒波形RMSE，先在每飞行内部汇总平方误差并开方，再等权平均飞行和种子；p/q/r单位rad/s，velocity为三分量均方的平方根（m/s），attitude为四元数角差（rad）。正百分比表示改善。

| 量 | v8姿态参考基线 | v23角速度参考 | 改善 |
|---|---:|---:|---:|
{chr(10).join(table)}

q的逐种子结果：

| seed | 基线 | 新模型 | 改善 |
|---|---:|---:|---:|
{chr(10).join(byseed)}

冻结保护项包括p/r完整1秒波形及速度/姿态0.5/1秒端点，每种子容许退化不超过5%。失败项如下：

{chr(10).join(lines) if lines else '无保护项失败。'}

全部时域、fit/OOF及逐飞行结果见[comparison.csv](comparison.csv)和[metrics_per_flight.csv](metrics_per_flight.csv)。原活动窗口标记未改变，0.5秒p/q≤0.15rad/s、姿态≤3度、速度≤0.5m/s的逐种子检查见[absolute_active_gate.csv](absolute_active_gate.csv)；相对改善与绝对精度应分开判断。

## 参考保持诊断

把未来p/q/r/thrust固定为起点值，保留全部真实历史及reset指令。下表仍按飞行/种子等权。固定输入是人工诊断输入，不是实际执行过的反事实；保持误差也不能当成部署时真实零操纵结果。

| 轴 | 原rate参考下误差 | 保持rate参考下误差 | 两组预测之差RMS |
|---|---:|---:|---:|
{chr(10).join(htable)}

日志rate参考来自姿态反馈，可能携带未来状态/扰动的信息。它对预测有帮助不等于网络已经学会干预效应。此诊断不能消除闭环输入相关性，更不能把这些未来日志作为原姿态参考模型可用的外生计划。

## 验证和后续边界

9项针对性测试通过；5份源日志/cache/固件语义校验、31339点独立原始消息重建、原非参考输入/标签一致性、5折归一化独立复算、15组初始化/抽样一致、检查点重载与计划参考前缀因果性通过。最终完整OOF407个窗口各每种子一次；交付哈希覆盖数据、检查点、指标、代码及报告。

本轮没有将该模型接入现有控制器，也没有用条件预测替代真实控制验收。更密的输入合同未通过q主门槛，说明仅更换输入边界在这组固定实验中没有解决主要误差；不能据此将全部误差归因为采样、模型容量或某个物理机制。

下一步先核验日志中的历史PID积分、空速缩放和有效状态能否作为内环记忆的可用观测，并冻结“只增加过去控制器状态”的单因素对照。这些状态目前由26点历史隐式表达；它们是否造成当前误差尚未证实。未来积分项、真实执行命令或状态不得作为额外已知输入。该对照尚未执行，不继续扫描训练权重或删除困难窗口。

以后若进行姿态参考闭环重放，必须恢复控制器历史，由模型状态驱动同一姿态控制器生成rate参考，同时比较模拟状态和模拟控制量；不能继续输入未来日志rate参考并称为自主闭环。当前不直接部署，不以模型内优化成功替代实证。

交付入口：[冻结协议](protocol.json)、[数据manifest](dataset_manifest.json)、[407窗清单](windows.csv)、[训练状态](launch/status.json)、[评价结论](evaluation.json)。新模型位于fold0–4/seed17、23、42/final.pt，输入合同不同，不可直接替换原姿态参考检查点。
'''
    (OUT/'report_zh.md').write_text(text)


if __name__=='__main__':main()
