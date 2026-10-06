"""Evidence and replay-entry decision for the frozen observed-response audit."""
from pathlib import Path
import numpy as np
import pandas as pd
import train_control_windows_v2 as base
from audit_controller_support_v26 import verify_parent
from audit_response_v36 import OUT, DATA, CONTROL, SUPPORT, LOCAL, V35, read


def main():
    protocol=read(OUT/'protocol.json')
    for name,digest in protocol['code'].items():assert base.sha(base.ROOT/name)==digest
    assert read(OUT/'verification.json')['status']==read(OUT/'independent_verification.json')['status']=='passed'
    e=pd.read_csv(OUT/'events.csv');s=pd.read_csv(OUT/'event_summary.csv').set_index(['arm','channel','axis'])
    counts=e.groupby('channel').agg(events=('array_index','size'),parents=('array_index','nunique'),flights=('log_id','nunique'),positive=('positive_command','sum'),
        coupled=('simultaneous_other_channels',lambda v:float((v>0).mean())))
    counts['negative']=counts.events-counts.positive;counts.to_csv(OUT/'event_counts.csv')
    table=[]
    for channel in ['motor','common','diff','yaw']:
        for axis in 'pqr':
            b=s.loc[('baseline',channel,axis)];j=s.loc[('joint',channel,axis)]
            table.append(f'| {channel} | {axis} | {b.response_mse:.4f}→{j.response_mse:.4f} | {b.direction_match*100:.1f}%→{j.direction_match*100:.1f}% | {j.predicted_power/j.truth_power:.3f} | {j.shape_cosine:.3f} |')
    count_table='\n'.join(f'| {c} | {r.events} | {r.positive}/{r.negative} | {r.flights} | {r.coupled*100:.1f}% |' for c,r in counts.iterrows())
    rates=pd.read_csv(OUT/'rate_errors.csv');cols=['total_mse','bias_mse','amplitude_mse','shape_mse','early_mse','late_mse']
    parts=rates.groupby(['arm','axis','seed','log_id'])[cols].mean().groupby(['arm','axis']).mean()
    parts.to_csv(OUT/'equal_flight_error_decomposition.csv')
    part_table='\n'.join(f'| {axis} | {parts.loc[("joint",axis),"bias_mse"]/parts.loc[("joint",axis),"total_mse"]*100:.1f}% | '
        f'{parts.loc[("joint",axis),"amplitude_mse"]/parts.loc[("joint",axis),"total_mse"]*100:.1f}% | '
        f'{parts.loc[("joint",axis),"shape_mse"]/parts.loc[("joint",axis),"total_mse"]*100:.1f}% |' for axis in 'pqr')
    strata=pd.read_csv(OUT/'rate_by_strata.csv');strata['rmse']=np.sqrt(strata.total_mse)
    strat_summary=strata.groupby(['condition','stratum','axis','arm','seed']).rmse.mean().groupby(['condition','stratum','axis','arm']).mean()
    strat_summary.to_csv(OUT/'stratified_rmse.csv')
    covered=float(strat_summary.loc[('support_past_joint','covered','p','joint')])
    weak=float(strat_summary.loc[('support_past_joint','weak','p','joint')])
    p=pd.read_csv(OUT/'p_accumulation_contributions.csv')
    pf=p.groupby(['fold','seed'])[['baseline','joint']].mean().pow(.5).groupby('fold').mean()
    pf['improvement_percent']=100*(1-pf.joint/pf.baseline);pf.to_csv(OUT/'p_accumulation_by_flight.csv')
    p_table='\n'.join(f'| {fold} | {r.baseline:.6f} | {r.joint:.6f} | {r.improvement_percent:+.2f}% |' for fold,r in pf.iterrows())
    attitude=pd.read_csv(OUT/'attitude_oracles.csv').groupby(['arm','replacement','seed','log_id']).end_mse.mean().pow(.5).groupby(['arm','replacement']).mean()
    attitude.to_csv(OUT/'attitude_oracle_summary.csv')
    att_table='\n'.join(f'| {axis} | {attitude.loc[("baseline",axis)]:.6f} | {attitude.loc[("joint",axis)]:.6f} |' for axis in ['none','p','q','r','all'])
    lag=pd.read_csv(OUT/'event_best_lag.csv');q=lag[(lag.arm=='joint')&(lag.axis=='q')]
    assert (q.lag==0).all()
    covariance=[read(path) for path in sorted(OUT.glob('*oof_input_covariance.json'))]
    conditions=[d['condition_number'] for d in covariance]
    next_step='Prepare a bounded same-PX4-inner-loop offline diagnostic replay: first audit full controller/allocator state initialization, timestamps, mode/parameter fidelity and observed command support. Use model states to generate future commands and compare both states and commands. No new-controller design, promotion, online deployment or future logged-command substitution.'
    base.write(OUT/'decision.json',dict(status='frozen_response_audit_complete',trained_models=0,windows_retained=407,
        input_channel_anchors=len(e),parents_with_anchors=int(e.array_index.nunique()),simultaneous_channel_fraction=float((e.simultaneous_other_channels>0).mean()),
        full_prediction_relative_gate_retained=True,absolute_gate_passed=False,model_promoted=False,control_ready=False,
        same_controller_diagnostic_replay_preparation_supported=True,replay_execution_ready=False,
        missing_replay_evidence='Complete controller/allocator initial state and faithful closed-loop time ordering have not been audited in this stage.',
        p_accumulation_worse_flights=int((pf.joint>pf.baseline).sum()),q_all_group_best_oracle_lags_zero=bool((q.lag==0).all()),
        baseline_retained='A1600',next_step=next_step,validation_accessed=False,sealed_test_accessed=False,reserved_accessed=False))
    text=f'''# v36：冻结模型的操纵响应审计

**完成v30/v35全部407父窗、15对模型的审计；未重训或删窗。v35的相对预测收益仍成立，但响应幅值、方向一致性和持续偏差尚不足以支持新控制器设计。下一步可以进入同一PX4内环的受限离线诊断重放准备；本轮尚未核验控制器状态恢复，不能宣布重放入口已经就绪。**

## 范围与事件定义

保留原五飞行的OOF划分、全部种子和v31模式/扑频/输入与状态支持/分配标记。每个父窗和命令通道，在前25个命令位置找“后5点均值−前5点均值”绝对值最大的位置；只有非零且达到对应fit飞行等权75分位数才记为命令事件。阈值和事件选择完全不读取状态响应、预测误差或query分布；同一个父窗可在多个通道出现，不能算多个独立飞行试验。

共{len(e)}个通道事件，覆盖{e.array_index.nunique()}/407父窗，全部5飞行有事件。其余父窗仍参与完整误差分解和支持分层；未删除弱响应、反向响应或困难窗口。事件是事后命令变化的锚点，不是额外施加的外生激励，也不声称是实际运动起始时刻。

| 通道 | 事件数 | 正/负命令 | 飞行数 | 伴随其他通道变化 |
|---|---:|---:|---:|---:|
{count_table}

共模=(servo0+servo1)/2，差模=(servo1−servo0)/2；仍是归一化命令，不是实测舵角/气动力。伴随变化判据：按fit事件阈值归一化后，其他通道变化≥目标通道的50%；总体{100*(e.simultaneous_other_channels>0).mean():.2f}%。这只是描述并发变化，没有剔除并发事件。

## 方向、幅值、形状和耦合

比较事件之后25点（20–500ms）的角速度，分别减去各自事件前5点均值；预测前史在原起点后使用模型自己的预测，原起点前使用真实已知历史。这样突出变化过程，但会消去部分常数偏差，不能替代完整预测误差。方向定义为最后20点平均变化的同号率，数值零阈值仅1e−12，弱/反向/中性响应均保留；没有经独立噪声标定，因此该比率不是因果舵效正确率。

下表先飞行内汇总，再等权平均飞行和种子；RMSE为rad/s，幅值比是预测/真实响应RMS之比，形状为各自中心化波形余弦。各通道列出的全部三轴都是同期观测，不能解释成单通道传递函数。

| 输入事件 | 响应轴 | RMSE v30→v35 | 方向一致率 v30→v35 | v35幅值比 | v35形状余弦 |
|---|---|---:|---:|---:|---:|
{chr(10).join(table)}

例如共模事件q的误差改善，幅值仍明显低估；差模事件p的形状相似性和幅值仍不足。yaw事件r虽RMSE下降，方向一致率反而下降，不能用一个总体误差指标代替响应验收。全部12组合及分层明细已保存，没有从表中挑出“效果好”的样本重新评价。

四路模态命令差分在各OOF飞行的fit尺度标准化协方差条件数为{min(conditions):.3f}–{max(conditions):.3f}，本项没有显示严重的全局线性秩退化。**这不消除闭环反馈混杂，也不等于在相同状态下存在充分的独立输入比较。**事件并发比例和线性相关系数是不同问题。

## 时间形状

对固定15点真值支持比较±100ms位移，并分别中心化，每个飞行/种子/通道/响应轴汇总后选最小误差。v35全部q组（{len(q)}组）最优位移均为0；p/r多数也在0附近。因此本检查不支持给q统一添加固定时移。

这不是实际执行器时延测量。最优位移看过未来真值，是离线形状诊断；命令、状态和控制器反馈同时变化，不能从这个结果认定物理延迟为零。未改输入对齐或模型推理。

## 完整波形与持续偏差

独立复现v35的完整1秒p/q/r指标和18/18保护通过结果；原活动窗口绝对门槛仍为3/12，不能把新事件子集的表现拿来替换它。完整误差分解为常值偏差、幅值差和形状差，按飞行/种子等权MSE统计，三项重组逐值核验：

| v35轴 | 偏差项占比 | 幅值项占比 | 形状项占比 |
|---|---:|---:|---:|
{part_table}

p完整1秒RMSE在已有past_joint支持覆盖层仍为{covered:.6f}rad/s，弱支持层为{weak:.6f}；弱层样本更少、参与飞行数不同，不能直接当作总体分布比较。Stabilized和Mission、既有低中高扑频层均保留；问题并非只存在于支持之外。

v35滚动H26局部p误差下降，但其预测增量离线累计在{int((pf.joint>pf.baseline).sum())}/5飞行变差：

| fold | v30累计p RMSE | v35累计p RMSE | 改善 |
|---|---:|---:|---:|
{p_table}

最大退化在fold4/log_28。按全部父窗的增量MSE排序可见窗口404、396、381等；404与381在既有过去状态/命令和联合支持中仍被覆盖，396为弱支持，不能只删范围外样本来消除问题。这些窗口仅作事后定位，未进入新的筛选规则。该累计每步依赖真实观测历史，是持续偏差诊断，**不是自主1秒预测误差**。

## 姿态误差由哪些角速度贡献

原中点积分逐值重建全部模型预测姿态，再仅在运动学积分中替换一轴为真实未来角速度；不重新运行GRU、各轴效果不可相加。下表为1秒姿态端点RMSE，单位rad：

| 替换轴 | v30 | v35 |
|---|---:|---:|
{att_table}

替换r带来的下降仍较大，不能只盯着q或p；全轴替换的残差是采样/姿态估计/积分合同综合差异，不是不可约噪声下界。这是未来真值oracle，不是可部署修正。

## 判断与下一步

这次结果支持保留v35作为后续研究候选：相对收益可重复，在真实命令变化事件上也有改善。它尚未提供独立反事实舵效、完整控制器状态恢复或闭环保真证据，绝对精度也未达标；A1600仍保留，不晋升、不全407再训练、不设计或部署新控制器。

**下一步准备同一PX4内环的离线诊断重放。**先核查固件/参数/模式、控制器积分与滤波历史、分配状态、时间顺序、模型输入单位及支持边界能否恢复；随后才让模型状态驱动控制链产生未来动作，同时比较状态和动作。不能继续喂未来日志执行命令并称为闭环，也不能靠额外动作限幅掩盖偏差。当前只支持进入这项准备工作，重放执行是否具备条件需由初始化审计决定；即使诊断重放完成也不直接授予控制可用结论。

本轮3项测试通过；独立实现重算412个事件、fit-only阈值和2472条模型事件轨迹的误差/持续方向，全部一致。既有支持距离重算、分层重组407父窗、完整v35指标与姿态积分核验通过。父产物/检查点/源码哈希保持，未读取开发验证、sealed或reserved；这些五飞行已多轮开发，不能称未开封最终测试。

入口：[冻结方案](protocol.json)、[事件清单](events.csv)、[全部响应](event_summary.csv)、[支持分层](stratified_rmse.csv)、[位移诊断](event_best_lag.csv)、[p持续偏差](p_accumulation_by_flight.csv)、[验证](verification.json)、[独立重算](independent_verification.json)、[决定](decision.json)。CSV中event_summary的response_mse/predicted_power/truth_power列已开方为RMSE/RMS，event_response逐事件列仍为平方量；聚合顺序见协议。
'''
    (OUT/'report_zh.md').write_text(text)
    for root in [DATA,CONTROL,SUPPORT,LOCAL,V35]:verify_parent(root)
    code=dict(protocol['code'])
    for name in ['scripts/report_response_v36.py','scripts/verify_response_v36.py']:code[name]=base.sha(base.ROOT/name)
    base.write(OUT/'delivery_manifest.json',dict(artifacts={str(p.relative_to(OUT)):base.sha(p)
        for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'},code=code,
        training_started=False,model_promoted=False,validation_accessed=False,sealed_test_accessed=False,reserved_accessed=False))
    print(read(OUT/'decision.json'),flush=True)


if __name__=='__main__':main()
