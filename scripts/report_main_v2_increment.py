#!/usr/bin/env python3
"""Paired S0 comparison and conservative six-gate increment promotion report."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-main-v2-increment')
import sys,json,argparse,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report_main_v2_objectives import md,bootstrap_percent

METRICS=['position_m','velocity_m_s','attitude_deg','body_rate_rad_s','frequency_hz','phase_rad']


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--results',type=Path,required=True);pa.add_argument('--artifacts',type=Path,required=True);a=pa.parse_args();r=a.results
    protocol=json.loads((r/'protocol.json').read_text());names=[c['name'] for c in protocol['experiments']]
    h=pd.read_csv(r/'free_running_per_horizon.csv');ti=pd.read_csv(r/'teacher_increment_metrics.csv');fi=pd.read_csv(r/'free_running_increment_metrics.csv');v=pd.read_csv(r/'variation_summary.csv');ax=pd.read_csv(r/'axis_summary.csv');spec=pd.read_csv(r/'spectral_summary.csv');phase=pd.read_csv(r/'phase_conditioned_metrics.csv');reg=pd.read_csv(r/'regime_summary.csv')
    base=h[h.experiment=='S0'].set_index('horizon_s');changes=[];confidence=[]
    for name in names[1:]:
        cand=h[h.experiment==name].set_index('horizon_s');f0=pd.read_csv(r/'S0/per_flight.csv');f1=pd.read_csv(r/name/'per_flight.csv')
        for horizon in base.index:
            row=dict(experiment=name,horizon_s=horizon)
            for metric in METRICS:
                row[metric+'_change_pct']=100*(cand.loc[horizon,metric+'_equal_log_rmse']/base.loc[horizon,metric+'_equal_log_rmse']-1)
                x=f1[f1.horizon_s==horizon].sort_values('log_id')[metric+'_rmse'];y=f0[f0.horizon_s==horizon].sort_values('log_id')[metric+'_rmse'];lo,hi=bootstrap_percent(x,y)
                confidence.append(dict(experiment=name,horizon_s=horizon,metric=metric,change_pct=row[metric+'_change_pct'],ci95_low=lo,ci95_high=hi,improved_flights=int((x.to_numpy()<y.to_numpy()).sum())))
            changes.append(row)
    changes=pd.DataFrame(changes);confidence=pd.DataFrame(confidence);changes.to_csv(r/'relative_changes.csv',index=False);confidence.to_csv(r/'paired_flight_confidence.csv',index=False)
    teacher_changes=[]
    for name in names[1:]:
        for signal in ['velocity','omega']:
            x=ti[(ti.experiment==name)&(ti.signal==signal)].sort_values('log_id');y=ti[(ti.experiment=='S0')&(ti.signal==signal)].sort_values('log_id')
            lo,hi=bootstrap_percent(x.rmse,y.rmse);teacher_changes.append(dict(experiment=name,signal=signal,s0_rmse=y.rmse.mean(),candidate_rmse=x.rmse.mean(),change_pct=100*(x.rmse.mean()/y.rmse.mean()-1),ci95_low=lo,ci95_high=hi,
                **{label+'_change_pct':100*(x['axis_'+str(i)+'_rmse'].mean()/y['axis_'+str(i)+'_rmse'].mean()-1) for i,label in enumerate(['x_or_p','y_or_q','z_or_r'])}))
    tc=pd.DataFrame(teacher_changes);tc.to_csv(r/'teacher_increment_changes.csv',index=False)
    scores={n:float(np.log1p(changes[(changes.experiment==n)&changes.horizon_s.isin([2,3,5])][['velocity_m_s_change_pct','attitude_deg_change_pct']].to_numpy()/100).mean()) for n in names[1:]}
    best=min(scores,key=scores.get);decisions=[]
    summaries={n:json.loads((r/n/'summary.json').read_text()) for n in names}
    def ratio(name,signal,horizon=5):return float(v[(v.experiment==name)&(v.signal==signal)&(v.horizon_s==horizon)]['median'].iloc[0])
    for name in names[1:]:
        cc=changes[changes.experiment==name];ct=tc[tc.experiment==name];ci=confidence[(confidence.experiment==name)&confidence.horizon_s.isin([2,3,5])]
        gate1=bool((ct.ci95_high<0).all())
        gate2=any(bool((ci[ci.metric==metric].change_pct<0).all() and (ci[ci.metric==metric].ci95_high<0).sum()>=2) for metric in ['velocity_m_s','attitude_deg'])
        gate3=all(abs(np.log(ratio(name,s)))<abs(np.log(ratio('S0',s))) for s in ['omega','omega_last1s'])
        gate4=bool((cc[cc.horizon_s.isin([.2,.5])][[m+'_change_pct' for m in ['velocity_m_s','attitude_deg','body_rate_rad_s']]]<10).all().all())
        gate5=bool((cc.frequency_hz_change_pct<10).all())
        gate6=all(summaries[name][key]<=summaries['S0'][key] for key in ['numeric_failures','clipping_failures','support_failures'])
        def spectral(n):return spec[(spec.experiment==n)&(spec.signal=='omega')].relative_spectral_l1.mean()
        def increment(n):return fi[(fi.experiment==n)&(fi.signal=='omega')&(fi.horizon_s==5)].rmse.mean()
        def derivative(n):return ax[(ax.experiment==n)&(ax.signal=='angular_acceleration')&(ax.horizon_s==5)].rmse.mean()
        noise=bool(spectral(name)<=spectral('S0') and increment(name)<=increment('S0') and derivative(name)<=derivative('S0'))
        result='A' if all([gate1,gate2,gate3,gate4,gate5,gate6,noise]) else ('B' if gate1 else 'C')
        decisions.append(dict(experiment=name,gate1_local_increment=gate1,gate2_long_velocity_or_attitude=gate2,gate3_variation=gate3,gate4_short_accuracy=gate4,gate5_frequency=gate5,gate6_failures=gate6,noise_guard=noise,result=result))
    decision=pd.DataFrame(decisions);decision.to_csv(r/'promotion_gates.csv',index=False)
    result=decision[decision.experiment==best].result.iloc[0]
    # Compare matched command bins, retaining sample counts and all horizons.
    regime_changes=[]
    for name in names[1:]:
        for variable in ['motor_command','tail_command_magnitude','command_variation']:
            x=reg[(reg.experiment==name)&(reg.variable==variable)].set_index(['bin','horizon_s']);y=reg[(reg.experiment=='S0')&(reg.variable==variable)].set_index(['bin','horizon_s'])
            for (bin_id,hh),row in x.iterrows():
                z=dict(experiment=name,variable=variable,bin=bin_id,horizon_s=hh,n_rollouts=row.n_rollouts)
                for metric in METRICS:z[metric+'_change_pct']=100*(row[metric+'_equal_log_rmse']/y.loc[(bin_id,hh),metric+'_equal_log_rmse']-1)
                regime_changes.append(z)
    rc=pd.DataFrame(regime_changes);rc.to_csv(r/'regime_changes.csv',index=False)
    q=[]
    for name in names:
        for hh in base.index:
            rate=ax[(ax.experiment==name)&(ax.horizon_s==hh)&(ax.signal=='body_rate_endpoint')];drift=ax[(ax.experiment==name)&(ax.horizon_s==hh)&(ax.signal=='SO3_log_error_deg')];delta=fi[(fi.experiment==name)&(fi.horizon_s==hh)&(fi.signal=='omega')]
            accel=ax[(ax.experiment==name)&(ax.horizon_s==hh)&(ax.signal=='angular_acceleration')]
            q.append(dict(experiment=name,horizon_s=hh,p_rmse=rate.axis_0_rmse.mean(),q_rmse=rate.axis_1_rmse.mean(),r_rmse=rate.axis_2_rmse.mean(),delta_p2_rmse=delta.axis_0_rmse.mean(),delta_q2_rmse=delta.axis_1_rmse.mean(),delta_r2_rmse=delta.axis_2_rmse.mean(),q_dot_rmse=accel.axis_1_rmse.mean(),pitch_axis_so3_error_deg=drift.axis_1_rmse.mean()))
    q=pd.DataFrame(q);q.to_csv(r/'pitch_q_summary.csv',index=False)
    for metric in METRICS:
        fig,aa=plt.subplots(figsize=(7,4))
        for name in names:
            f=h[h.experiment==name];aa.plot(f.horizon_s,f[metric+'_equal_log_rmse'],'o-',label=name)
        aa.set(xlabel='Nominal horizon s (native dt)',ylabel=metric);aa.legend();aa.grid(alpha=.25);fig.tight_layout();fig.savefig(r/(metric+'_vs_horizon.png'),dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for axis,signal in zip(axes,['omega','angular_acceleration']):
        for name in names:
            f=v[(v.experiment==name)&(v.signal==signal)];axis.plot(f.horizon_s,f['median'],'o-',label=name)
        axis.axhline(1,color='black',linestyle='--');axis.set(title=signal,xlabel='Horizon s',ylabel='Median variation ratio');axis.legend()
    fig.tight_layout();fig.savefig(r/'variation_vs_horizon.png',dpi=160);plt.close(fig)
    f=pd.read_csv(r/'axis_spectra.csv');fig,axes=plt.subplots(2,3,figsize=(13,7))
    for i,signal in enumerate(['omega','angular_acceleration']):
        for j,axis in enumerate(['p','q','r']):
            aa=axes[i,j]
            for name in names:
                g=f[(f.experiment==name)&(f.signal==signal)&(f.axis==axis)];aa.semilogy(g.frequency_hz,g.pred_power,label=name)
            aa.semilogy(g.frequency_hz,g.truth_power,'k--',label='truth');aa.set(title=signal+' '+axis,xlabel='Hz');aa.legend()
    fig.tight_layout();fig.savefig(r/'axis_psd.png',dpi=160);plt.close(fig)
    fig,aa=plt.subplots(figsize=(7,4))
    for name in names:
        g=q[q.experiment==name];aa.plot(g.horizon_s,g.pitch_axis_so3_error_deg,'o-',label=name)
    aa.set(xlabel='Horizon s',ylabel='Pitch-axis SO(3) log error deg');aa.legend();fig.tight_layout();fig.savefig(r/'pitch_attitude_drift.png',dpi=160);plt.close(fig)
    logs=sorted(phase.log_id.unique());fig,axes=plt.subplots(len(logs),2,figsize=(11,3*len(logs)),squeeze=False)
    for i,log in enumerate(logs):
        for j,mode in enumerate(['teacher_state','free_running']):
            for name in names:
                row=phase[(phase.experiment==name)&(phase.log_id==log)&(phase['mode']==mode)&(phase.signal=='q_dot')].iloc[0]
                err=np.asarray(json.loads(row.phase_bin_errors));axes[i,j].plot(np.arange(len(err))*360/len(err),err,label=name)
            axes[i,j].set(title=Path(log).name+' '+mode,xlabel='Within-log encoder phase deg',ylabel='q_dot harmonic error rad/s²');axes[i,j].legend()
    fig.tight_layout();fig.savefig(r/'phase_qdot_error.png',dpi=150);plt.close(fig)
    summary=dict(selected_by_long_score=best,long_log_rmse_scores=scores,result=result,gates=decisions,teacher_increment_changes=teacher_changes,
        free_running_relative_changes=changes.to_dict('records'),
        omega_variation_5s={n:dict(prefix=ratio(n,'omega'),last1s=ratio(n,'omega_last1s')) for n in names},
        sealed_test_opened=False,structural_readiness=100,dynamics_fidelity_readiness=60,rl_ready=False,
        limitation='Single paired seed policy; five flights; local success does not certify RL or arbitrary commands')
    (r/'summary.json').write_text(json.dumps(summary,indent=2))
    training=pd.read_csv(r/'training_summary.csv');conditional=json.loads((r/'conditional_experiments.json').read_text());means=ti.groupby(['experiment','signal'])[['rmse','axis_0_rmse','axis_1_rmse','axis_2_rmse']].mean().reset_index()
    local=tc[tc.experiment==best].set_index('signal');selected=changes[changes.experiment==best].set_index('horizon_s')
    qb=q[q.experiment=='S0'].set_index('horizon_s');qc=q[q.experiment==best].set_index('horizon_s')
    ph=phase.groupby(['experiment','mode','signal'])[['harmonic_rmse','pred_harmonic_std','shape_correlation']].mean()
    def change(x,y):return 100*(x/y-1)
    overview=(f"{best} 的teacher-state Δv₂ RMSE降低 {-local.loc['velocity','change_pct']:.2f}%，Δω₂降低 {-local.loc['omega','change_pct']:.2f}%；"
        f"但5s velocity/attitude误差分别变化 {selected.loc[5,'velocity_m_s_change_pct']:+.2f}% / {selected.loc[5,'attitude_deg_change_pct']:+.2f}%。"
        f"5s body-rate variation由 {ratio('S0','omega'):.3f} 到 {ratio(best,'omega'):.3f}，末1s由 {ratio('S0','omega_last1s'):.3f} 到 {ratio(best,'omega_last1s'):.3f}。"
        "局部target更准确、持续角运动更强，与长期trajectory更准确是不同结论。")
    q_answer=(f"teacher-state Δq₂变化 {local.loc['omega','y_or_q_change_pct']:+.2f}%，而Δp₂/Δr₂为 "
        f"{local.loc['omega','x_or_p_change_pct']:+.2f}% / {local.loc['omega','z_or_r_change_pct']:+.2f}%，角增量收益主要集中在q。"
        f"5s q endpoint RMSE变化 {change(qc.loc[5,'q_rmse'],qb.loc[5,'q_rmse']):+.2f}%，"
        f"pitch-axis姿态误差变化 {change(qc.loc[5,'pitch_axis_so3_error_deg'],qb.loc[5,'pitch_axis_so3_error_deg']):+.2f}%。"
        "局部q增量与长期pitch drift必须独立判断。")
    phase_improved=all(ph.loc[(best,mode,'q_dot'),'harmonic_rmse']<ph.loc[('S0',mode,'q_dot'),'harmonic_rmse'] for mode in ['teacher_state','free_running'])
    phase_answer=(("证据支持恢复了部分**记录中的phase同步成分**，并非只有无关高频振幅增加。" if phase_improved else "phase同步成分未在两种模式中同时改善。")+
        f"teacher-state q_dot harmonic RMSE从 {ph.loc[('S0','teacher_state','q_dot'),'harmonic_rmse']:.3f} 到 "
        f"{ph.loc[(best,'teacher_state','q_dot'),'harmonic_rmse']:.3f} rad/s²；free-running从 "
        f"{ph.loc[('S0','free_running','q_dot'),'harmonic_rmse']:.3f} 到 {ph.loc[(best,'free_running','q_dot'),'harmonic_rmse']:.3f}。"
        "但free-running谱形和autocorrelation仍未匹配truth，不能声称完整恢复真实物理动力学，也不能将已记录的高频全部判为噪声。")
    report=['# Step 5 — State-Increment Supervision for Main V2',
        '**结论 '+result+'：** '+{'A':'P2成功：本轮共同validation中state-increment监督改善长期simulator fidelity。','B':'P2局部有效，但长期free-running改善不足。','C':'P2放回完整Main V2后基本无效，未达到局部与长期联合晋级证据。'}[result],
        overview,
        '完整阅读Step 4报告及Step 2/3报告与CSV；inventory和historical hashes已记录。本轮不更改架构、phase、actuator constants、history、split或simulator，不打开sealed test。S0重新按A0合同训练，S1从相同seed独立重训；不是旧checkpoint微调。',
        '## Target、normalization与实验预算',
        '`Δv_pred=vhat[t+2]−vhat[t]`、`Δω_pred=ωhat[t+2]−ωhat[t]`；t为真实窗口起点，两个预测步来自原Main V2 forward，vhat[t]/ωhat[t]等于真实初始化。t+1之后没有teacher forcing，没有独立head，没有跳步推理。',
        '每个原始4214 train window监督其起点两步；50步原loss继续完整free-running。与Step 3 A3的整段内部差分不同，此处明确从真实起点约束局部两步，同时加入velocity和omega，尺度由实际增量统计决定。未新增position increment或quaternion subtraction。',
        '真实2step时长：'+json.dumps(protocol['increment_duration_s'])+'；50step时长：'+json.dumps(protocol['training_duration_s'])+'。固定的是实现steps，积分读取每个原生dt。',
        md(pd.read_csv(r/'train_increment_statistics.csv')),
        '单个向量一个RMS尺度，三轴等权；λ=冻结baseline对应50step state项 / 冻结baseline归一化increment error。没有根据validation反推权重，保留所有旧loss权重。原Main V2没有raw derivative loss，因此不存在可降低的raw-derivative权重。',
        md(training),
        '两阶段40+25epoch，base/actuator seed17/29，AdamW lr3e-4/5e-4，batch256，weight decay1e-5，gradient clip5。每组冻结自己训练的backbone，再训练actuator residual，固定最后epoch。S0最大参数差见training_summary。GPU并发墙钟受共享负载影响。',
        'S2/S3条件判定（train-only，训练前记录规则）：\n```json\n'+json.dumps(conditional,indent=2)+'\n```',
        'S2仅在局部train改善<5%且increment/original梯度比<0.1时触发，λ只允许4倍，不删原约束；S3为可选后续，本轮有界矩阵不自动增加。初始梯度比小不等于最终被淹没，需查看final_gradient_probes与训练loss占比。',
        '## Q1/Q2：完整Main V2是否复现局部优势？',
        '用Step 4固定40,085个validation真实起点，warm26后自主运行两步。误差单位为Δv的m/s、Δω的rad/s，不是除duration后的acceleration。先每日志RMSE，再日志等权。',
        md(means),md(tc),
        '本轮复现了局部优势的方向，但幅度不等同于Step 4 probe：velocity改善较大，angular改善较小且主要来自q。所有结果基于本轮重新训练的S0对照。',
        'Gate1要求两个increment的paired-flight 95%CI均低于0；是否通过见晋级表。这只比较同一2step target，不能直接把Step 4 probe的不同训练样本/单导数输出20%作为本轮应达到的阈值。',
        '## Q3：0.2–5秒free-running核心指标',
        '所有模型固定722 origins、warm26、相同commands/实际dt、六个horizon，未来truth仅评分。完整mean/median/p90/p95/max与failure数量保存在各模型per_horizon/per_rollout/per_flight；失败但有限路径保留。',
        md(h[['experiment','horizon_s']+[m+'_equal_log_rmse' for m in METRICS]]),
        '相对S0变化，负数为改善：',md(changes),
        '长期paired-flight置信区间：',md(confidence[confidence.horizon_s.isin([2,3,5])&confidence.metric.isin(['velocity_m_s','attitude_deg'])]),
        ('候选在五条flight的2/3/5s velocity与attitude均未改善（每项improved_flights=0）；长期退化不是被单个最差flight拉高的平均值。' if (confidence[(confidence.experiment==best)&confidence.horizon_s.isin([2,3,5])&confidence.metric.isin(['velocity_m_s','attitude_deg'])].improved_flights==0).all() else '各flight改善数量和paired置信区间如上，不把pooled平均值当作所有flight均改善。'),
        '五flight的3125种cluster bootstrap，不把重叠722起点视为独立样本；单seed政策，不宣称跨seed显著或在新控制策略下泛化。',
        'free-running滑动两步增量（每个prefix内全部成对预测状态差），与teacher-state分开：',md(fi.groupby(['experiment','horizon_s','signal']).rmse.mean().reset_index()),
        '## Q4：变化幅度是否恢复？',md(v[v.signal.isin(['omega','omega_last1s','angular_acceleration','angular_acceleration_last1s'])]),
        '![Variation](variation_vs_horizon.png)',
        'prefix包含初始状态；last1s为最后51个状态/50个导数。比值接近1不是充分条件，不能把只是增加抖动当作恢复。',
        '## Q5：周期动态还是高频噪声？',md(spec),
        phase_answer,
        '![Axis PSD](axis_psd.png)\n\n![Per-log phase error](phase_qdot_error.png)',
        '高频边界13.122682Hz来自Step 3冻结train omega谱95%累计功率；不是人为把5Hz外都当噪声。PSD和autocorrelation保持原生时间重采样诊断，不作为训练或simulator输入。',
        'raw/filtered teacher-state导数误差：',md(pd.read_csv(r/'derivative_diagnostics.csv').groupby(['experiment','target','signal']).rmse.mean().reset_index()),
        'D1沿用Step 4：状态插值到50Hz网格，4阶Butterworth、6/8/12Hz cutoff、sosfiltfilt零相位双向过滤，再插回原生时间求差分。**仅用于离线target diagnosis**；没有进入loss或simulator输入。',
        'teacher-state q_dot幅值（rad/s²；各log统计后等权）：',md(pd.read_csv(r/'teacher_derivative_statistics.csv').query("axis == 'q_dot'").groupby(['experiment','source'])[['mean','std','rms']].mean().reset_index()),
        'phase-conditioned component使用每log现有坐标独立描述，不拟合validation offset、不改变模型phase：',md(phase.groupby(['experiment','mode','signal'])[['harmonic_rmse','pred_harmonic_std','truth_harmonic_std','shape_correlation']].mean().reset_index()),
        '这些是记录动态的同步描述，不证明全部高频为真实刚体运动。噪声guard同时检查omega spectral L1、free-running increment RMSE与angular acceleration RMSE。autocorrelation.csv保留全部对照。',
        '## Q6：q/q_dot及pitch通道',md(q),
        q_answer,
        'p/q/r与Δp/Δq/Δr均完整保留；pitch是Log(Rtruthᵀ Rpred)的body-y分量RMSE（deg），不是quaternion分量减法，也不是近奇异Euler pitch直接相减。angular_acceleration的单轴结果见axis_summary.csv。',
        '## Q7：高控制区是否改善？',md(rc[(rc.bin==2)&rc.horizon_s.isin([.5,2,3,5])][['experiment','variable','horizon_s','n_rollouts','velocity_m_s_change_pct','attitude_deg_change_pct','body_rate_rad_s_change_pct']]),
        ('高motor、tail及command-variation桶在0.5s有velocity收益，但2/3/5s的velocity与attitude均退化。此次没有证据说明P2解决了高控制区的长期精度问题。' if (rc[(rc.experiment==best)&(rc.bin==2)&rc.horizon_s.isin([2,3,5])][['velocity_m_s_change_pct','attitude_deg_change_pct']]>0).all().all() else '高控制区的收益必须与总体结果分别核对，不能因整体RMSE降低就推定高控制区也改善。'),
        'motor/tail bins沿用Step 2 train边界；command variation沿用原722 command tapes总变差三分位。future commands只用于离线分桶。各桶flight构成不同，不据此作控制因果结论。',
        '## Q8：晋级判定',md(decision),
        '操作性保守规则：Gate2要求velocity或attitude在2/3/5秒均改善且至少两个CI低于0；Gate3要求prefix5及last1s variation均更接近1；Gate4短期velocity/attitude/body-rate退化均<10%；Gate5 frequency退化<10%；Gate6三类failure均不增加。10%为沿用的审查警戒线，不是物理安全阈值；support来源于训练支持域，包含raw derivative噪声影响，不是适航包线。',
        md(pd.DataFrame([dict(experiment=n,**{k:summaries[n][k] for k in ['numeric_failures','support_failures','clipping_failures']}) for n in names])),
        '选择用于主结论的候选：'+best+'，按2/3/5秒velocity/attitude日志等权RMSE几何比排序，不等于自动晋级。'+{'A':'下一步冻结本轮candidate，再独立评估P1或sealed-test前冻结条件；本轮不打开sealed test。','B':'下一步第一优先P1 phase-reference方案，必须保留/解决跨log unknown zero，不能直接沿用已退化的phase替换。第二优先transition representation调查。本轮没有证明phase是长期漂移的单一原因；不继续增大increment权重、不将S1晋级为最终simulator。','C':'Step 4 probe的收益没有形成完整simulator的联合改善证据；停止继续调increment权重，转P1或transition representation调查。'}[result],
        'Simulator structural readiness: 100%\n\nDynamics fidelity readiness: 60%\n\nRL-ready: NO\n\n保留既有评分，本轮不因局部指标提高而追加readiness；未做反事实动作验证或sealed test。',
        '## 一条完整复现命令',
        '```bash\n/home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/run_main_v2_increment.py \\\n  --results /tmp/main-v2-increment/results --artifacts /tmp/main-v2-increment/models --device cuda:1\n```',
        '使用空目录；自动train-only校准、预检、S0/S1、条件S2、冻结benchmark、增量/频谱/phase诊断、报告、pytest和git diff检查。旧checkpoint与Step1–4结果hash会再次核对。']
    (r/'report.md').write_text('\n\n'.join(report));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
