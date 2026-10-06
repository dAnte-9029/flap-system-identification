"""Publish state-path diagnostics with explicit local/oracle limitations."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_control_windows_v2 as base
from audit_controller_support_v26 import verify_parent
from audit_rollout_trace_v32 import OUT,DATA,V23,V30,V31
from rollout_trace_v32 import accumulated_rates,VARIANTS


def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    for name,digest in protocol['code'].items():assert base.sha(base.ROOT/name)==digest
    for root in [DATA,V23,V30,V31]:verify_parent(root)
    m=pd.read_csv(OUT/'per_window.csv');c=pd.read_csv(OUT/'clipping.csv')
    columns=['output_mse','accumulated_mse','accumulated_end_mse','persistence_mse','derivative_mse','derivative_change_mse']
    summary=m.groupby(['arm','variant','axis','seed','log_id'])[columns].mean().pow(.5).groupby(['arm','variant','axis']).mean()
    summary.to_csv(OUT/'summary_rmse.csv')
    byflight=m.groupby(['arm','variant','axis','seed','log_id'])[columns].mean().pow(.5).groupby(['arm','variant','axis','log_id']).mean()
    byflight.to_csv(OUT/'by_flight_rmse.csv')
    event=m[m.array_index==97];event.to_csv(OUT/'window97_metrics.csv',index=False)
    rest=m[(m.fold==1)&(m.array_index!=97)]
    rest_summary=rest.groupby(['arm','variant','axis','seed'])[columns].mean().pow(.5).groupby(['arm','variant','axis']).mean()
    rest_summary.to_csv(OUT/'log18_remaining77_rmse.csv')
    def row(arm,variant,axis):return summary.loc[(arm,variant,axis)]
    free=row('actuator','free','q');teacher=row('actuator','true_state_recursive_hidden','q')
    control=row('baseline','true_state_recursive_hidden','q')
    maxraw=float(c.max_raw_abs.max());clipped=int(c.clipped_components.sum())
    # Select previously designated diagnostic example, preserving every window in metrics.
    with np.load(DATA/'windows.npz') as z:
        truth=z['label_rates_b'][97].astype(float);initial=z['input_initial_rates_b'][97].astype(float);dt=z['input_dt_s'][97].astype(float)
    cache={}
    for seed in [17,23,42]:
        with np.load(OUT/f'fold1/actuator/seed{seed}/trace.npz') as z:
            pos=np.flatnonzero(z['array_index']==97).item()
            cache[seed]={k:z[k][pos] for k in z.files if k!='array_index'}
    t=np.arange(1,51)*.02;true_derivative=(truth-np.vstack([initial,truth[:-1]]))/dt[:,None]
    fig,axes=plt.subplots(3,2,figsize=(13,10),constrained_layout=True)
    color=['tab:blue','tab:orange','tab:green'];labels=['free rollout','true state / free hidden','true state / recursive hidden']
    for column,(seed,axis,j) in enumerate([(23,'q',1),(17,'r',2)]):
        data=cache[seed];axes[0,column].plot(t,truth[:,j],color='black',label='logged rate')
        axes[1,column].plot(t,true_derivative[:,j],color='gray',alpha=.55,label='logged 20ms finite difference')
        for variant,col,label in zip(VARIANTS,color,labels):
            accumulated=accumulated_rates(initial[None],data[variant+'_derivative'][None],dt[None])[0,:,j]
            axes[0,column].plot(t,accumulated,color=col,label=label)
            axes[1,column].plot(t,data[variant+'_derivative'][:,3+j],color=col,label=label)
        axes[0,column].set_title(f'{axis}, seed{seed}: free vs offline derivative sums')
        axes[0,column].set_ylabel('rad/s');axes[1,column].set_title(f'{axis} derivative, seed{seed}');axes[1,column].set_ylabel('rad/s2')
        axes[0,column].legend(fontsize=7);axes[1,column].legend(fontsize=7)
    for seed,col in zip([17,23,42],color):
        data=cache[seed]
        delta=data['true_state_recursive_hidden_hidden']-data['free_hidden']
        axes[2,0].plot(t,np.sqrt(np.mean(delta**2,1)),color=col,label=f'seed{seed}')
        axes[2,1].plot(t,abs(data['free_raw_head']).max(1),color=col,label=f'seed{seed} free')
    axes[2,0].set_title('Hidden-state difference (oracle versus free)');axes[2,0].set_ylabel('RMS over64 units')
    axes[2,1].axhline(6,color='black',linestyle='--',label='clamp threshold');axes[2,1].set_title('Raw output max absolute value')
    axes[2,0].legend(fontsize=8);axes[2,1].legend(fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.25);ax.set_xlabel('seconds from original forecast origin')
    fig.suptitle('Window97 / v30: oracle sums are diagnostic paths, not deployable rollouts')
    fig.savefig(OUT/'window97_trace.png',dpi=160);fig.savefig(OUT/'window97_trace.svg');plt.close(fig)
    next_step=('Prepare a frozen rolling-H26-history re-encoding diagnostic on original supports before any new training. '
               'Use complete observed histories with explicit phase/frequency/airspeed/time contracts; compare with current oracle path to test hidden/context distribution mismatch. '
               'If local increment bias persists, design a separately frozen local-increment plus full-rollout training comparison, retaining all original gates. '
               'Do not interpret this diagnostic as permission to retime/delete data or deploy.')
    decision=dict(status='diagnostic_complete',trained_models=0,windows_retained=407,models_traced=30,
        all_variants_clipped_components=clipped,maximum_raw_head_abs=maxraw,
        actuator_q_free_rmse=float(free.output_mse),actuator_q_true_state_local_rmse=float(teacher.output_mse),
        actuator_q_previous_true_rate_persistence_rmse=float(teacher.persistence_mse),
        actuator_q_true_state_derivative_sum_rmse=float(teacher.accumulated_mse),
        baseline_q_true_state_derivative_sum_rmse=float(control.accumulated_mse),
        conclusion='No head-clipping bottleneck observed. State-reset local errors improve against persistence, but signed derivative accumulation on observed-state paths grows strongly, including baseline. Pure free-state drift alone is insufficient as an explanation. Oracle inputs change state/hidden distribution and retain imperfect held context, so this does not establish unique physical or training root cause.',
        next_step=next_step,model_promoted=False,baseline_retained='A1600',validation_accessed=False,sealed_test_accessed=False)
    base.write(OUT/'decision.json',decision)
    names={'free':'原自由递推','true_state_free_hidden':'真实状态＋自由路径隐藏状态','true_state_recursive_hidden':'真实状态＋独立递推隐藏状态'}
    local_rows=[];sum_rows=[]
    for arm,label in [('baseline','v23'),('actuator','v30')]:
        for variant in VARIANTS[1:]:
            qr=row(arm,variant,'q');rr=row(arm,variant,'r')
            local_rows.append(f'| {label} | {names[variant]} | {qr.output_mse:.6f} | {rr.output_mse:.6f} |')
        for variant in VARIANTS:
            qr=row(arm,variant,'q');rr=row(arm,variant,'r')
            sum_rows.append(f'| {label} | {names[variant]} | {qr.accumulated_mse:.6f} | {rr.accumulated_mse:.6f} |')
    event_rows=[]
    for seed,axis in [(23,'q'),(17,'r'),(42,'r')]:
        for variant in VARIANTS:
            er=event[(event.arm=='actuator')&(event.seed==seed)&(event.axis==axis)&(event.variant==variant)].iloc[0]
            event_rows.append(f'| {seed}/{axis} | {names[variant]} | {er.derivative_bias:+.6f} | {np.sqrt(er.accumulated_end_mse):.6f} |')
    text=f'''# v32：逐步导数、隐藏状态及递推路径诊断

**全部407窗、v23/v30共30个OOF模型诊断完成。没有重训、改检查点、改输入时间或删窗；保留A1600，不晋升控制模型。**本轮未发现输出限幅瓶颈，但也不支持“只有自由状态漂移，纠正状态就能恢复正确持续响应”的简单解释。

## 对照边界

三条路径共享原26点历史、未来参考/命令、模式及原来的固定扑频、相位推进和空速代理：

- 原自由递推：状态和隐藏状态均由模型推进，与原检查点预测逐值一致。
- 真实状态＋自由隐藏：每一步用上一时刻记录的速度/姿态/角速度，保留该步自由路径的更新前隐藏状态，分支只预测一步，随后丢弃分支隐藏更新。
- 真实状态＋独立隐藏：使用相同上一时刻真实状态，但沿这条路径连续更新自己的隐藏状态。

所有路径首步完全相同。第k步的oracle只使用第k−1步标签作为状态输入，不提前读取第k步目标。尽管如此，这两条路径仍在原预测起点之后持续接收真实观测，**属于离线诊断，不是原起点可实现的1秒预测**。它们也没有使用未来真实相位、扑频或空速。

## 没有触发输出限幅

全部模型、窗口、50步和三条路径中，raw head超过±6的分量共 **{clipped}** 个；最大绝对值为 **{maxraw:.6f}**。因此这批退化不能归因于输出被该clamp截断，不据此放宽限幅。

部分隐藏单元接近±1，但这只是隐藏数值接近边界，不是直接测得GRU门饱和，也不是根因证明。原始隐藏64维、归一化递推输入21维、限幅前6维输出和限幅后导数均已保存，避免只凭隐藏范数下结论。

## 局部误差和持续偏差必须分开

每步重置真实状态后的20ms角速度误差（全50次局部预测聚合，飞行/种子等权，rad/s）：

| 模型 | 离线路径 | q局部RMSE | r局部RMSE |
|---|---|---:|---:|
{chr(10).join(local_rows)}

同一批点的“保持上一真实角速度”基线q/r为{teacher.persistence_mse:.6f}/{row('actuator','true_state_recursive_hidden','r').persistence_mse:.6f}。例如v30独立隐藏路径q局部误差{teacher.output_mse:.6f}，确实低于保持基线；但不能与自由递推1秒的{free.output_mse:.6f}直接比较并宣称长时性能改善，因为可用观测不同。

为检查局部增量是否存在持续同向偏差，另外从原初始角速度累加各路径预测角加速度×dt，**不在求和时重置角速度**。oracle导数仍是在真实状态路径上算出的，此累积不是重新运行的自洽动力学轨迹，更不是可部署预测。

| 模型 | 导数来源路径 | q累计曲线RMSE | r累计曲线RMSE |
|---|---|---:|---:|
{chr(10).join(sum_rows)}

v30真实状态＋独立隐藏的q累计误差升至{teacher.accumulated_mse:.6f}，v23也有同类现象（{control.accumulated_mse:.6f}）。即使局部误差较小，符号相关的增量误差仍可持续积累；当前结果不支持把退化全部归结为积分器数值误差或单纯的自由状态漂移。

这里的“真值导数”是相邻20ms日志角速度的有限差分，即区间平均变化率，含测量/滤波/采样影响，不是独立测得的瞬时物理角加速度。真实状态、原训练隐藏状态与固定相位/频率代理的组合也改变了输入分布，**不能据此证明真实状态路径就是训练一致的裸机向量场，或将偏差唯一归因于损失、隐藏状态或某个物理机制**。

## 窗口97的异常怎样形成

v30 seed23的自由路径首步预测dq/dt约8.022rad/s²，0.1秒约12.337rad/s²；q随后维持明显正偏差。沿真实状态路径计算时，该偏差没有自动消失。1秒平均导数误差与累计端点绝对误差如下（前者rad/s²，后者rad/s）：

| seed/轴 | 导数路径 | 平均导数误差 | 累计端点绝对误差 |
|---|---|---:|---:|
{chr(10).join(event_rows)}

同一步比较自由路径与“真实状态＋自由隐藏”可看到物理状态输入对输出的影响；后两条oracle的差异则发生在相同物理输入、不同隐藏历史下。但这是模型内部、路径相关的比较，不是独立因果分解。隐藏差异增长可以是状态偏差的后果，不能仅凭同时增长认定隐藏状态失稳。

![窗口97导数、累积曲线与隐藏状态](window97_trace.png)

图中的oracle累计曲线只用于显示偏差累积；虚线限幅阈值没有被触及。[SVG版本](window97_trace.svg)。完整407窗及log_18其余77窗均保留，分别见[汇总](summary_rmse.csv)、[逐飞行](by_flight_rmse.csv)、[其余77窗](log18_remaining77_rmse.csv)，没有按图示结果筛除数据。

## 检查及下一步

3项针对性测试通过，覆盖完整自由路径复现、以模型轨迹作真值时三路径逐值一致、未来标签扰动不能改变自由路径或此前oracle结果，以及真实clamp/积分记录。对30个实际检查点，另用未修改模型的forward hooks逐值核验hidden/rawhead/recurrent input，自由输出与原文件逐值一致；每种子每组407窗唯一，首步三路径一致，积分恒等式及原v30角速度主指标复现。原检查点与父产物/源码哈希未变。无开发验证、sealed或reserved访问。

下一步建议先准备一个**完整H26观测历史重新编码的滚动单步对照**，明确相位/扑频/空速与时间合同，检验当前真实状态路径是否主要受隐藏状态和上下文分布不一致影响。若局部增量偏差仍存在，再冻结“局部增量监督＋原完整递推监督”的训练对照；不能因本轮oracle变好就直接调权重。旧v18只强化原起点首步q，且未通过保护门槛，不等同于这个尚未准备的对照，也不能作为其成功依据。

本轮没有执行该下一阶段或启动任何新训练。当前数据仍不足以认证因果控制作用，五飞行OOF也已反复用于开发。

入口：[协议](protocol.json)、[结论](decision.json)、[验证](verification.json)、[逐窗口指标](per_window.csv)、[逐步q/r明细](per_step.csv.gz)。全部原始内部轨迹在fold*/{{baseline,actuator}}/seed*/trace.npz。
'''
    (OUT/'report_zh.md').write_text(text)
    code=dict(protocol['code']);code[str(Path(__file__).resolve().relative_to(base.ROOT))]=base.sha(__file__)
    # Explicitly pin the prior-stage context cited only to distinguish proposed follow-up.
    context=base.ROOT/'docs/analysis/results/control_windows_v18/report_zh.md'
    base.write(OUT/'report_context.json',dict(prior_report=str(context.relative_to(base.ROOT)),sha256=base.sha(context),
        use='Distinguish earlier origin-only q auxiliary loss; no prior stage data loaded.'))
    base.write(OUT/'delivery_manifest.json',dict(artifacts={str(p.relative_to(OUT)):base.sha(p)
        for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'},code=code,
        models_traced=30,trained_models=0,model_promoted=False,validation_accessed=False,sealed_test_accessed=False))
    print(json.dumps(decision,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
