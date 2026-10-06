"""Summarize frozen v31 evidence without data removal or control promotion."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_control_windows_v2 as base
from audit_controller_support_v26 import verify_parent
from actuator_diagnostics_v31 import angle_squared,modal_commands
from audit_actuator_response_v31 import OUT,DATA,V12,V23,V28,V30


def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    for name,digest in protocol['code'].items(): assert base.sha(base.ROOT/name)==digest
    for root in [V12,V23,V28,DATA,V30]: verify_parent(root)
    rate=pd.read_csv(OUT/'rate_errors.csv');att=pd.read_csv(OUT/'attitude_oracles.csv')
    dist=pd.read_csv(OUT/'support_per_window.csv');strata=pd.read_csv(OUT/'q_strata.csv')
    windows=pd.read_csv(DATA/'windows.csv');q=rate[rate.axis=='q'];target=windows.iloc[97].log_id
    f=q.groupby(['cohort','seed','log_id'])[['baseline_total_mse','actuator_total_mse']].mean().pow(.5).groupby(['cohort','log_id']).mean().reset_index()
    f['improvement_percent']=100*(1-f.actuator_total_mse/f.baseline_total_mse);f.to_csv(OUT/'q_by_flight.csv',index=False)
    support_table=dist[dist.cohort=='oof'].groupby(['log_id','block']).weak.agg(['sum','count']).reset_index()
    support_table.to_csv(OUT/'support_summary.csv',index=False)
    chosen=q[(q.cohort=='oof')&(q.log_id==target)]
    whole=chosen.delta_total_mse.mean();event=chosen[chosen.array_index==97];rest=chosen[chosen.array_index!=97]
    contribution=event.delta_total_mse.mean()/78
    share=contribution/whole
    bias_delta=chosen.delta_bias_mse.mean();late_delta=chosen.delta_late_mse.mean();early_delta=chosen.delta_early_mse.mean()
    rest_rmse=rest.groupby('seed')[['baseline_total_mse','actuator_total_mse']].mean().pow(.5).mean()
    a_cols=[k for k in att if k.endswith('_mse')]
    a=att.groupby(['cohort','arm','seed','log_id'])[a_cols].mean().pow(.5).groupby(['cohort','arm']).mean()
    a.to_csv(OUT/'attitude_summary.csv')
    lag=pd.read_csv(OUT/'lag_curves.csv').groupby(['cohort','log_id','arm','lag_steps']).centered_mse.mean().reset_index()
    best=lag.loc[lag.groupby(['cohort','log_id','arm']).centered_mse.idxmin()].copy()
    zero=lag[lag.lag_steps==0].drop(columns='lag_steps').rename(columns={'centered_mse':'zero_mse'})
    best=best.merge(zero,on=['cohort','log_id','arm']);best['oracle_improvement_percent']=100*(1-np.sqrt(best.centered_mse/best.zero_mse))
    best.to_csv(OUT/'lag_summary.csv',index=False)
    # Mean-squared contributions are additive; RMSE percentages are not.
    contr=strata[(strata.cohort=='oof')&(strata.log_id==target)].copy()
    contr['weighted_delta_mse']=contr.windows*contr.delta_total_mse/78
    contr=contr.groupby(['condition','stratum'])[['windows','weighted_delta_mse']].mean().reset_index()
    for _,g in contr.groupby('condition'):np.testing.assert_allclose(g.weighted_delta_mse.sum(),whole,atol=1e-12)
    contr['signed_share_percent']=100*contr.weighted_delta_mse/whole
    contr.to_csv(OUT/'log18_stratum_contributions.csv',index=False)
    # The event is chosen for review after observing errors, never for filtering.
    with np.load(DATA/'windows.npz') as z:
        truth=z['label_rates_b'][97];quat_true=z['label_quaternion'][97]
        command=z['input_future_command'][97]
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    t=np.arange(1,51)*.02;predictions={}
    colors={17:'tab:blue',23:'tab:orange',42:'tab:green'}
    for k,seed in enumerate([17,23,42]):
        for arm,root in [('baseline',V23),('actuator',V30)]:
            with np.load(root/f'fold1/seed{seed}/oof_predictions.npz') as z:
                i=np.flatnonzero(z['array_index']==97).item()
                predictions[seed,arm]=(z['rates_b'][i],z['quaternion'][i])
        ax=axes[0,k];ax.plot(t,truth[:,1],color='black',label='logged q')
        ax.plot(t,predictions[seed,'baseline'][0][:,1],'--',color='tab:gray',label='rate-reference model')
        ax.plot(t,predictions[seed,'actuator'][0][:,1],color=colors[seed],label='actuator model')
        ax.set_title(f'q, seed {seed}');ax.set_ylabel('rad/s');ax.legend(fontsize=8)
    m=modal_commands(command)
    for j,label in enumerate(['motor','common','diff','yaw']):axes[1,0].plot(t-.02,m[:,j],label=label)
    axes[1,0].set_title('Post-allocation command combinations');axes[1,0].legend(fontsize=8)
    axes[1,1].plot(t,truth[:,2],color='black',label='logged r')
    for seed in [17,23,42]:
        axes[1,1].plot(t,predictions[seed,'actuator'][0][:,2],color=colors[seed],label=f'actuator {seed}')
        for arm,style in [('baseline','--'),('actuator','-')]:
            degrees=np.rad2deg(np.sqrt(angle_squared(predictions[seed,arm][1][None],quat_true[None])[0]))
            axes[1,2].plot(t,degrees,style,color=colors[seed],label=f'{arm} {seed}')
    axes[1,1].set_title('r: actuator predictions');axes[1,1].set_ylabel('rad/s');axes[1,1].legend(fontsize=8)
    axes[1,2].set_title('Quaternion angle error');axes[1,2].set_ylabel('deg');axes[1,2].legend(fontsize=7)
    for ax in axes.flat:ax.grid(alpha=.25);ax.set_xlabel('seconds after forecast origin')
    fig.suptitle('log_18 / window 97: offline frozen-model diagnostic; no data exclusion')
    fig.savefig(OUT/'window97_diagnostic.png',dpi=160);fig.savefig(OUT/'window97_diagnostic.svg');plt.close(fig)
    findings=dict(status='diagnostic_complete',windows_retained=407,focus_window=97,focus_log=target,
        log18_delta_q_mse=float(whole),window97_weighted_delta_mse=float(contribution),window97_signed_mse_share=float(share),
        log18_bias_delta_share=float(bias_delta/whole),log18_first_half_delta_mse=float(early_delta),log18_last_half_delta_mse=float(late_delta),
        remaining77_q_baseline=float(rest_rmse.baseline_total_mse),remaining77_q_actuator=float(rest_rmse.actuator_total_mse),
        model_promoted=False,baseline_retained='A1600',root_cause_proven=False,training_started=False,
        interpretation='A seed-sensitive low-support/allocator-flagged event contributes most log18 q excess MSE, but covered windows also regress. Mean bias and late rollout accumulation dominate. Fixed lag shift not supported on log18. Attitude accumulation involves multiple rate axes; oracle axis replacements are nonadditive, not causal efficacy.',
        next_step='Freeze data and checkpoints; trace per-step predicted q/r derivatives, hidden states and output clipping for all407 OOF windows, with event97 highlighted. Separate local state-conditioned prediction error from autoregressive error propagation using explicitly offline true-state diagnostic inputs; do not train, retime inputs or delete windows.')
    base.write(OUT/'decision.json',findings)
    ft='\n'.join(f'| {r.log_id.split("/")[-1].split("_2026")[0]} | {r.baseline_total_mse:.6f} | {r.actuator_total_mse:.6f} | {r.improvement_percent:+.2f}% |' for r in f[f.cohort=='oof'].itertuples())
    st=support_table.pivot(index='log_id',columns='block',values='sum')
    stlines='\n'.join(f'| {i.split("/")[-1].split("_2026")[0]} | {int(r.past_command)} | {int(r.future_command)} | {int(r.past_state)} | {int(r.past_joint)} |' for i,r in st.iterrows())
    focused=strata[(strata.cohort=='oof')&(strata.log_id==target)].groupby(['condition','stratum'])[['windows','baseline_rmse','actuator_rmse']].mean()
    lines='\n'.join(f'| {cond}/{level} | {int(r.windows)} | {r.baseline_rmse:.6f} | {r.actuator_rmse:.6f} |' for (cond,level),r in focused.iterrows() if cond in ['mode','frequency_mean','support_past_joint','allocator_flag'])
    ar=a.loc[('oof','actuator')];br=a.loc[('oof','baseline')]
    oracle='\n'.join(f'| {label} | {br[k+"_end_mse"]:.6f} | {ar[k+"_end_mse"]:.6f} |' for k,label in [('none','原预测'),('p','替换p'),('q','替换q'),('r','替换r'),('all','替换全部三轴')])
    text=f'''# v31：执行命令模型退化诊断

**诊断完成；未重训、改输入时间、删窗或晋升模型。保留A1600。**原407窗、v23/v30各15模型和所有评价标签冻结。当前证据把问题定位到跨飞行条件下的均值偏差与后段误差积累，尚不能确定唯一根因。

## 主要发现

1. log_18的q退化不是整架次普遍严重超出输入覆盖。78窗中，过去命令弱支持5窗、未来命令4窗、过去状态和联合特征各1窗。该共同状态异常窗是97，也带有v28的分配受限记录。阈值只由各折4个fit飞行定义，不使用误差。
2. 窗口97对log_18的候选减基线q MSE增量贡献{share*100:.2f}%。这是有符号MSE贡献，不是“解释了多少RMSE”的因果比例。其seed23 q MSE从1.066626升至6.283879，后半秒从0.611317升至8.615458；seed17/42该窗q反而改善。seed17/42的r在该窗明显恶化，提示种子相关的不同轴偏差增长，不能只归因于俯仰。
3. 其余77窗q仍从{rest_rmse.baseline_total_mse:.6f}升至{rest_rmse.actuator_total_mse:.6f}，退化{100*(rest_rmse.actuator_total_mse/rest_rmse.baseline_total_mse-1):.2f}%。Mission和Stabilized两层都退化。因此，排除这个困难片段既不能消除问题，也会缩小有效工况；本轮保留全部窗口。
4. log_18新增q MSE为{whole:.6f}，其中均值偏差项贡献{bias_delta/whole*100:.2f}%；前/后半秒新增MSE为{early_delta:.6f}/{late_delta:.6f}，退化主要集中后半秒。训练内log_18 q略改善而OOF退化，支持进一步查泛化/递推误差，不证明过拟合是唯一原因。
5. 固定40点真值支持、±100ms的q中心化波形移位诊断中，log_18两模型在fit和OOF的最优移位均为0。此有限诊断不支持直接补一个统一时延；它既不能排除时变延迟，也不能测量真实执行器延迟。

## 跨飞行与支持范围

q完整1秒RMSE，先飞行内开方再等权平均种子，单位rad/s：

| 飞行 | v23参考模型 | v30执行命令模型 | 改善 |
|---|---:|---:|---:|
{ft}

各飞行弱支持窗口数（总数分别97/78/87/64/81）：

| 飞行 | 过去命令 | 未来命令 | 过去状态 | 过去联合 |
|---|---:|---:|---:|---:|
{stlines}

支持特征采用均值、标准差、末值、步差RMS。命令为motor、servo共模/差模、yaw，组合不等于实测力矩；状态为过去速度、角速度、重力方向、扑频、空速及有效标记。各fit飞行等权标准化，最近邻校准排除整个同飞行，95分位阈值只用fit。未来命令支持是离线条件输入覆盖，不能用于在线信任判定。汇总特征不覆盖全部时序关系，标为覆盖内不代表物理可辨识。

log_28弱状态支持较多但q整体略改善，因此弱支持本身不是退化的充分解释。log_18舵面共/差模大部分位于fit逐通道范围内；联合/状态分布比单通道上下限更有信息，但仍是描述性关联。

## log_18分层

沿用既有fit-only扑频分位点与模式；窗口数量和误差均按种子平均，未按结果重定义分箱：

| 条件/层 | 窗口 | v23 q RMSE | v30 q RMSE |
|---|---:|---:|---:|
{lines}

各分层贡献见[log18_stratum_contributions.csv](log18_stratum_contributions.csv)。同一条件下贡献能重组成完整飞行的MSE差；不同条件交叉重叠，不能相加。低扑频层包含窗口97，不能据此把扑频定为根因。分配受限消息较低速，也不能等同于该窗全程舵面饱和。

窗口97为原log时间364.685188–366.205188秒，预测起点365.205188秒；Stabilized，仍保留在数据中。此窗历史扑频约2.976–3.468Hz，未来servo0达到归一化1，servo1最低约−0.982。没有实测舵角/气动力，不能从命令直接认定物理舵效。

![窗口97逐种子q、r、命令和姿态误差](window97_diagnostic.png)

矢量版本：[window97_diagnostic.svg](window97_diagnostic.svg)。绘图用已有记录与冻结预测，不是模拟的新干预轨迹。

## 姿态退化：不能只看q

用保存的预测角速度重新执行同一中点四元数积分，全部fit/OOF轨迹与保存姿态逐值相同；未发现这条回放路径的实现不一致。随后只替换积分使用的某一轴为未来真值，其他轴保持原预测，得到下表。此为使用未来标签的离线运动学oracle，不重跑GRU、不形成可部署模型，轴间效应不可加。

1秒姿态端点RMSE（rad，飞行/种子等权）：

| 离线替换 | v23 | v30 |
|---|---:|---:|
{oracle}

替换r带来的端点下降比单独替换p/q更大，说明偏航角速度的持续误差也值得优先追查。仅替换q不能消除姿态差异。全三轴真值积分仍约{ar.all_end_mse:.6f}rad误差，表明50Hz角速度积分与日志姿态之间也有离散化/估计器/时间等未分离差异；这不是不可约噪声下界，不能把它全算作模型误差或测量延迟。

## 验证与下一步

6项测试通过；全部5折支持距离用独立逐样本最近邻复算，原扑频分层核验，误差分解恒等式与分层加权重组通过；v30的p/q/r和姿态主指标复现，15组模型全部fit/OOF四元数从保存角速度精确重建。父产物与代码哈希通过；未访问开发验证、sealed或reserved。

下一步建议冻结模型，记录全部407个OOF窗口每一步q/r预测导数、隐藏状态与输出限幅，重点展示97而不只筛它；使用明确标为离线oracle的真实状态输入对照，区分局部状态条件误差和自由递推误差传播，再决定是否需要新的训练对照。本轮尚未执行该进一步诊断，不据此调整时延、删除困难窗或部署控制器。

入口：[冻结协议](protocol.json)、[结论](decision.json)、[验证](verification.json)。完整明细包括[rate_errors.csv](rate_errors.csv)、[error_timecourse.csv](error_timecourse.csv)、[attitude_oracles.csv](attitude_oracles.csv)、[lag_curves.csv](lag_curves.csv)、[support_per_window.csv](support_per_window.csv)。五飞行已经多轮开发使用，结论不是独立测试认证。
'''
    (OUT/'report_zh.md').write_text(text)
    code=dict(protocol['code']);code[str(Path(__file__).resolve().relative_to(base.ROOT))]=base.sha(__file__)
    base.write(OUT/'delivery_manifest.json',dict(artifacts={str(p.relative_to(OUT)):base.sha(p)
        for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'},code=code,
        training_started=False,model_promoted=False,validation_accessed=False,sealed_test_accessed=False))
    print(json.dumps(findings,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
