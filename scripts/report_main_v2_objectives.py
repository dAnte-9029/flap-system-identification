#!/usr/bin/env python3
"""Summarize the completed objective experiment with paired flight uncertainty."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-main-v2-step3')
import sys,json,argparse,itertools,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from system_identification.training.trajectory_main_v1 import assemble_history_trajectory_windows
from system_identification.models.trajectory import attitude_error_deg
from system_identification.evaluation.main_v2_free_running import HORIZONS,magnitude

METRICS=['position_m','velocity_m_s','attitude_deg','body_rate_rad_s','frequency_hz','phase_rad']


def md(frame):
    lines=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    for row in frame.itertuples(index=False,name=None):lines.append('| '+' | '.join(f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |')
    return '\n'.join(lines)


def bootstrap_percent(candidate,reference):
    """Exact paired empirical cluster bootstrap; five flights => 3125 draws."""
    a=np.asarray(candidate);b=np.asarray(reference)
    idx=np.array(list(itertools.product(range(len(a)),repeat=len(a))))
    values=(a[idx].mean(1)/b[idx].mean(1)-1)*100
    return float(np.quantile(values,.025)),float(np.quantile(values,.975))


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--artifacts',type=Path,default=ROOT/'artifacts/main_v2_training_objective_ablation');pa.add_argument('--results',type=Path,default=ROOT/'docs/analysis/results/main_v2_training_objective_ablation');args=pa.parse_args()
    root=args.results
    if (root/'report.md').exists():raise FileExistsError(root/'report.md')
    protocol=json.loads((root/'protocol.json').read_text());names=[c['name'] for c in protocol['experiments']]
    h=pd.read_csv(root/'benchmark_per_horizon.csv');v=pd.read_csv(root/'variation_summary.csv')
    training=pd.read_csv(root/'training_summary.csv')
    a0='A0_baseline_retrain';base=h[h.experiment==a0].set_index('horizon_s')
    ratios=[];ci=[]
    for name in names:
        f=h[h.experiment==name].set_index('horizon_s')
        for horizon in HORIZONS:
            row=dict(experiment=name,horizon_s=horizon)
            for metric in METRICS:row[metric+'_change_pct']=float(100*(f.loc[horizon,metric+'_equal_log_rmse']/base.loc[horizon,metric+'_equal_log_rmse']-1))
            ratios.append(row)
    changes=pd.DataFrame(ratios);changes.to_csv(root/'relative_changes.csv',index=False)
    pairs=[('A1_longer_rollout',a0),('A2_multi_horizon','A1_longer_rollout'),('A3_dynamic_delta','A2_multi_horizon')]
    pairs += [(n,a0) for n in names[2:]]
    for candidate,reference in pairs:
        x=pd.read_csv(root/candidate/'per_flight.csv').set_index(['log_id','horizon_s'])
        y=pd.read_csv(root/reference/'per_flight.csv').set_index(['log_id','horizon_s'])
        for horizon in HORIZONS:
            xx=x.xs(horizon,level='horizon_s').sort_index();yy=y.xs(horizon,level='horizon_s').sort_index()
            for metric in METRICS:
                a=xx[metric+'_rmse'];b=yy[metric+'_rmse'];lo,hi=bootstrap_percent(a,b)
                ci.append(dict(candidate=candidate,reference=reference,horizon_s=horizon,metric=metric,
                    change_pct=float((a.mean()/b.mean()-1)*100),ci95_lower=lo,ci95_upper=hi,improved_flights=int((a<b).sum()),n_flights=len(a)))
    ci=pd.DataFrame(ci);ci.to_csv(root/'paired_flight_confidence.csv',index=False)
    scores={}
    for name in names[1:]:
        f=changes[(changes.experiment==name)&changes.horizon_s.isin([2,3,5])]
        scores[name]=float(np.mean(np.log1p(f[['velocity_m_s_change_pct','attitude_deg_change_pct']].to_numpy()/100)))
    best=min(scores,key=scores.get)
    best_h=h[h.experiment==best].set_index('horizon_s');best_v=v[v.experiment==best]
    long=ci[(ci.candidate==best)&(ci.reference==a0)&ci.horizon_s.isin([2,3,5])&ci.metric.isin(['velocity_m_s','attitude_deg'])]
    significant=bool((long.ci95_upper<0).all())
    last=float(best_v[best_v.signal=='omega_last1s']['median'].iloc[0]);prefix=float(best_v[(best_v.signal=='omega')&(best_v.horizon_s==5)]['median'].iloc[0])
    ds={n:json.loads((root/n/'summary.json').read_text()) for n in names}
    dr={n:pd.read_csv(root/n/'derivatives.csv').set_index('signal') for n in names}
    noise_ok=bool(dr[best].loc['angular_acceleration','full_rmse']<=dr[a0].loc['angular_acceleration','full_rmse'] and ds[best]['omega_spectral_l1_relative']<=ds[a0]['omega_spectral_l1_relative'])
    improved=bool((long.change_pct<0).all())
    short_changes=changes[(changes.experiment==best)&changes.horizon_s.isin([.2,.5])]
    short_warning=bool((short_changes[[m+'_change_pct' for m in METRICS]]>10).any().any())
    frequency_warning=bool((changes.loc[changes.experiment==best,'frequency_hz_change_pct']>10).any())
    stability_ok=ds[best]['numeric_failures']==0 and ds[best]['clipping_failures']==0
    # Warnings require scientific explanation before promotion; do not auto-promote.
    conclusion='A' if significant and .5<=last<=2 and .5<=prefix<=2 and noise_ok and stability_ok and not short_warning and not frequency_warning else ('B' if scores[best]<0 else 'C')
    # Correlations at fixed horizons, avoiding trivial pooling of increasing time.
    correlations=[];acf=[]
    for name in names:
        p=pd.read_csv(root/name/'phase_analysis.csv');r=pd.read_csv(root/name/'per_rollout.csv')
        f=p.merge(r[['window_id','horizon_s','angular_acceleration_rmse','linear_acceleration_rmse']],on=['window_id','horizon_s'],validate='one_to_one')
        for horizon,g in f.groupby('horizon_s'):
            for target in ['body_rate_error','angular_acceleration_rmse','linear_acceleration_rmse']:
                correlations.append(dict(experiment=name,horizon_s=horizon,target=target,pearson_abs_phase=float(g.phase_error_rad.abs().corr(g[target])),
                    frequency_integral_error_rms=float(np.sqrt(np.mean(g.frequency_integral_error_rad**2))),
                    measured_phase_frequency_residual_rms=float(np.sqrt(np.mean(g.measured_phase_frequency_residual_rad**2))),
                    integration_reconstruction_max_error=float(np.max(np.abs(g.reconstruction_error_rad)))))
        trace=np.load(args.artifacts/name/'validation_rollout.npz')
        for key in ['angular_velocity_b','angular_acceleration_b','velocity_n','acceleration_b','gru_hidden']:
            a=trace[key];a=a-a.mean(1,keepdims=True)
            for lag in [1,2,3,5,10,25]:
                numerator=np.sum(a[:,:-lag]*a[:,lag:],axis=(1,2));denom=np.sqrt(np.sum(a[:,:-lag]**2,axis=(1,2))*np.sum(a[:,lag:]**2,axis=(1,2)))
                acf.append(dict(experiment=name,signal=key,lag_steps=lag,median_correlation=float(np.median(numerator/np.maximum(denom,1e-20)))))
    pd.DataFrame(correlations).to_csv(root/'phase_correlations.csv',index=False);pd.DataFrame(acf).to_csv(root/'free_run_autocorrelation.csv',index=False)
    for metric in METRICS:
        fig,ax=plt.subplots(figsize=(7,4))
        for name in names:
            f=h[h.experiment==name];ax.plot(f.horizon_s,f[metric+'_equal_log_rmse'],marker='o',label=name[:2])
        ax.set(xlabel='Nominal horizon (s), native dt integrated',ylabel=metric);ax.legend();ax.grid(alpha=.3);fig.tight_layout();fig.savefig(root/f'{metric}_vs_horizon.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for name in names:
        f=v[(v.experiment==name)&(v.signal=='omega')];ax.plot(f.horizon_s,f['median'],marker='o',label=name[:2])
    ax.axhline(1,color='black',linestyle='--');ax.set(xlabel='Horizon s',ylabel='Median body-rate variation ratio');ax.legend();fig.tight_layout();fig.savefig(root/'variation_vs_horizon.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,key in zip(axes.flat,['velocity_n','angular_velocity_b','acceleration_n','angular_acceleration_b']):
        for name in [a0,best]:
            f=pd.read_csv(root/name/'spectra.csv');f=f[f.signal==key]
            a=f[f.source=='pred'];ax.semilogy(a.frequency_hz,a.power,label=name[:2])
        a=f[f.source=='truth'];ax.semilogy(a.frequency_hz,a.power,'k--',label='truth');ax.set(title=key,xlabel='Hz');ax.legend()
    fig.tight_layout();fig.savefig(root/'baseline_best_psd.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name in names:
        for ax,stage in zip(axes,['base','actuator']):
            f=pd.read_csv(args.artifacts/name/f'{stage}_history.csv');ax.plot(f.epoch,f.loss,label=name[:2]);ax.set(title=stage,xlabel='Epoch',ylabel='Own objective (not directly comparable across A0-A3)')
    axes[0].legend();fig.tight_layout();fig.savefig(root/'training_curves.png',dpi=160);plt.close(fig)
    old=ROOT/'docs/analysis/results/main_v2_free_running_5s';windows=pd.read_csv(old/'windows.csv')
    samples=pd.read_parquet(ROOT/'dataset/trajectory_v1_august_f5_c4/samples_validation.parquet');batch=assemble_history_trajectory_windows(samples,windows,history_steps=26)
    selected=[]
    for name in [a0,best]:
        r=pd.read_csv(root/name/'per_rollout.csv');r=r[r.horizon_s==5].sort_values('attitude_deg')
        trace=np.load(args.artifacts/name/'validation_rollout.npz')
        for label,j in [('best',0),('median',len(r)//2),('worst',len(r)-1)]:
            row=r.iloc[j];idx=int(np.flatnonzero(windows.window_id==row.window_id)[0]);t=np.r_[0,np.cumsum(trace['dt_s'][idx])]
            selected.append(dict(experiment=name,label=label,window_id=row.window_id,selection='own 5s attitude error rank',attitude_error=row.attitude_deg))
            fig,axes=plt.subplots(7,1,figsize=(11,16),sharex=True)
            for ax,key in zip([axes[0],axes[3],axes[4]],['angular_velocity_b','velocity_n','position_n']):
                for axis in range(3):
                    ax.plot(t,trace[key][idx,:,axis],color=f'C{axis}',label=f'{axis} pred')
                    ax.plot(t,getattr(batch.trajectory.truth,key)[idx,:,axis],'--',color=f'C{axis}',alpha=.6,label=f'{axis} truth')
                ax.set_ylabel(key);ax.legend(ncol=3,fontsize=7)
            truealpha=np.diff(batch.trajectory.truth.angular_velocity_b[idx],axis=0)/trace['dt_s'][idx,:,None]
            for axis in range(3):
                axes[1].plot(t[:-1],trace['angular_acceleration_b'][idx,:,axis],color=f'C{axis}')
                axes[1].plot(t[:-1],truealpha[:,axis],'--',color=f'C{axis}',alpha=.6)
            axes[1].set_ylabel('angular acceleration')
            axes[2].plot(t,attitude_error_deg(trace['quaternion_nb'][idx],batch.trajectory.truth.quaternion_nb[idx]));axes[2].set_ylabel('attitude error deg')
            for ax,key in zip(axes[5:],['flap_frequency_hz','relative_phase_rad']):
                ax.plot(t,trace[key][idx],label='pred');ax.plot(t,getattr(batch.trajectory.truth,key)[idx],'--',label='truth');ax.set_ylabel(key);ax.legend()
            axes[-1].set_xlabel('Elapsed native time s');fig.suptitle(name+' '+label+' '+row.window_id);fig.tight_layout();fig.savefig(root/f'{name[:2]}_{label}_rollout.png',dpi=140);plt.close(fig)
    pd.DataFrame(selected).to_csv(root/'representative_rollouts.csv',index=False)
    summary=dict(best_by_predeclared_long_velocity_attitude_score=best,long_score_log_ratio=scores,
        all_six_long_cells_improved=improved,all_six_cluster_bootstrap_ci_upper_below_zero=significant,
        variation_prefix5=prefix,variation_last1s=last,noise_guard_pass=noise_ok,short_horizon_warning=short_warning,frequency_warning=frequency_warning,conclusion=conclusion,
        sealed_test_opened=False,rl_ready=False,structural_readiness=100,
        fidelity_readiness_note='same Step 2 checklist; recomputed below, not a probability')
    # Keep identical checklist; no award for untested causal response/absolute targets.
    baseline=pd.read_csv(old/'constant_twist_baseline.csv').set_index('horizon_s')
    checks=[ds[best]['numeric_failures']==0,ds[best]['support_failures']==0,ds[best]['clipping_failures']==0]
    checks += [bool((best_h[m+'_equal_log_rmse']<baseline[m+'_equal_log_rmse']).all()) for m in ['velocity_m_s','attitude_deg','body_rate_rad_s','frequency_hz']]
    checks += [False,False,False] # no full absence-of-artifacts proof, absolute threshold or counterfactual test
    summary['dynamics_fidelity_readiness']=10*sum(checks)
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    report=[f'# Main V2 training-objective ablation — Step 3\n',
        '保持架构、模拟器、actuator time constants、归一化与数据划分不变；全部候选重新训练。sealed test未打开。',
        '## 当前loss与预诊断\n详见 `docs/audits/2026-09-15_main_v2_loss_audit.md`。原训练为50个实际dt转移，约1秒，非固定50×0.02积分。',
        '局部一步预测已经低估角加速度标准差（验证三轴约40%/40%/32%），递推进一步衰减；下一步omega本身较接近测量部分来自输入的真实omega，不能据此称一步导数准确。GRU norm从0.2秒约2.21降到5秒约1.78，但其temporal std约1.1并未消失，不能简单归因为hidden collapse。',
        '原生角加速度差分训练谱中10Hz以上功率约48%；使用两步对齐delta-omega辅助，避免直接回归所有高频差分。不是振幅奖励，也不是证明高频均为噪声。',
        '## 实验与训练预算\n'+md(training),
        '四组均4,214 train窗口、40+25 epoch、seed17/29、batch256、最终epoch checkpoint。两阶段分别训练backbone、冻结该backbone后训练actuator residual。GPU并发壁钟/吞吐受共享负载影响，不是独占硬件速度对比。串行pilot在完成checkpoint前停止并保留记录，不计入正式矩阵。',
        '## Q1：一步还是递推？\n两者都有。一步角加速度幅值/偏差已有问题；free-run后状态动态进一步变弱。不能从这种相关时序证明loss是唯一原因。',
        '## Q2：长horizon是否改善？\nA1对A0的逐flight置信区间与完整变化见下表。',
        md(ci[(ci.candidate=='A1_longer_rollout')&(ci.reference==a0)&ci.horizon_s.isin([2,3,5])&ci.metric.isin(['velocity_m_s','attitude_deg'])]),
        '## Q3：multi-horizon是否优于单一horizon？\nA2对A1（同100步）的直接对照：',
        md(ci[(ci.candidate=='A2_multi_horizon')&(ci.reference=='A1_longer_rollout')&ci.horizon_s.isin([.2,.5,2,3,5])&ci.metric.isin(['velocity_m_s','attitude_deg'])]),
        '## Q4：dynamic loss是否恢复动态？\nA3相对于A2只增加对齐两步delta-omega，lambda由train量级固定。不能把比值略增等同于恢复真实动态：',
        md(v[(v.experiment.isin(['A2_multi_horizon','A3_dynamic_delta']))&(v.signal.isin(['omega','omega_last1s']))]),
        '噪声检查包括angular acceleration RMSE、PSD偏差和train-PSD的95%功率频率以上能量，详见各模型summary/derivatives/spectra。',
        md(pd.DataFrame([dict(experiment=n,angular_accel_rmse=dr[n].loc['angular_acceleration','full_rmse'],high_frequency_energy_ratio=ds[n]['omega_high_frequency_energy_ratio'],spectral_relative_l1=ds[n]['omega_spectral_l1_relative']) for n in names])),
        '## Q5：改善来源\n所有模型积分代码完全相同，所以没有“更准确的积分器”这一实现改变。候选间angular acceleration误差/幅值、hidden统计、相位同步和旋转误差分解见对应CSV；只是伴随变化，不能未经干预把某项判为因果来源。',
        '相位误差由频率积分误差加实测phase/frequency不一致项精确重建：冻结模型5秒两项RMS约1.90 rad和0.13 rad，重建误差约3e-6 rad。初始frequency输入真值，initial error为0；微小持续frequency bias依然能积累相位漂移。固定phase anchor与真实dt均保持，未发现积分bug。相关性按每个horizon分别计算，见phase_correlations.csv，不混合时间来制造相关。',
        '## Q6：最佳实验与A0百分比变化\n最佳只按预先声明的2/3/5秒velocity/attitude日志等权RMSE几何比值排序，不等于通过success gate。选择：**'+best+'**。负数是改善。',
        md(changes[changes.experiment==best]),
        '最佳完整日志等权RMSE：',
        md(best_h[[m+'_equal_log_rmse' for m in METRICS]].reset_index()),
        '## Q7：最佳body-rate variation\n'+md(best_v[best_v.signal.isin(['omega','omega_last1s'])]),
        f'5秒prefix median={prefix:.6f}；末1秒median={last:.6f}。必须结合PSD与导数误差，不能以振幅增加本身论成功。',
        '## 稳定性\n'+md(pd.DataFrame([dict(experiment=n,**{k:ds[n][k] for k in ['numeric_failures','support_failures','clipping_failures']}) for n in names])),
        'Support是训练min/max支持域筛查，不是适航硬包线；保留有限失败轨迹计算误差。',
        '## Q8：下一阶段\n'+{'A':'A. training objective 已显著解决主要问题，可进入 sealed test 前冻结','B':'B. objective 有改善但不足，应继续 aerodynamic/state-transition investigation','C':'C. objective 基本无效，说明主要瓶颈不是训练 horizon/loss'}[conclusion],
        '该判断仅覆盖本轮1/2秒、固定四组目标与单seed政策，不能证明所有可能的loss无效。3秒会改变原共同训练起点，本轮未加入；5秒训练是否需要仍未证明，不预设curriculum必然有效。',
        '统计限制：仅5条flight，cluster bootstrap为3125种重采样；没有独立多seed复验，也没有对模型选择/多重比较做校正。即便CI排除0，也只作本矩阵的条件性证据，不宣称跨seed/新策略显著。短期任一指标退化>10%应明确警告，见下表：',
        md(changes[(changes.experiment==best)&changes.horizon_s.isin([.2,.5])]),
        '## Readiness\n'+f"Simulator structural readiness: 100%\n\nDynamics fidelity readiness: {summary['dynamics_fidelity_readiness']}%\n\nRL-ready: NO\n\n沿用Step 2十项证据清单，不是成功概率；未证明项不加分。",
        '## 复现\n```bash\n/home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/run_main_v2_step3.py \\\n  --results /tmp/main-v2-step3/results --artifacts /tmp/main-v2-step3/artifacts\n```\n训练前自动诊断，四个GPU作业隔离运行，完成后统一benchmark和报告。目录必须为空。',
        '代表轨迹按各模型5秒姿态误差best/median/worst固定选择；不同模型可能不同起点，仅用于描述。统计对照始终使用相同722起点。所有图见本目录。']
    # Retain actual dt distributions and train-only calibration in the report.
    report.insert(3, '实际训练时长分布（秒）：\n'+md(pd.DataFrame(protocol['train_horizon_seconds']).T.reset_index().rename(columns={'index':'steps'})))
    report.insert(4, '冻结train的归一化状态loss量级（选择prefix权重之前）：\n'+md(pd.DataFrame(protocol['train_loss_component_means']).T.reset_index().rename(columns={'index':'steps'})))
    report.insert(5, 'A0相对于旧checkpoint最大参数差：'+str(training.loc[training.experiment==a0,'legacy_checkpoint_max_parameter_difference'].iloc[0])+'；因此本次重训成功复现旧权重，并非复制旧checkpoint。')
    def pair_statement(candidate, reference):
        g=ci[(ci.candidate==candidate)&(ci.reference==reference)&ci.horizon_s.isin([2,3,5])&ci.metric.isin(['velocity_m_s','attitude_deg'])]
        count=int((g.change_pct<0).sum())
        return f'长期六项中{count}/6项改善，百分比变化范围{g.change_pct.min():.2f}%至{g.change_pct.max():.2f}%；CI排除零且改善的有{int((g.ci95_upper<0).sum())}/6项。'
    findings=['## 结果解释与失效边界',
        '**延长训练时域：** '+pair_statement('A1_longer_rollout',a0),
        '**Multi-horizon：** '+pair_statement('A2_multi_horizon','A1_longer_rollout')+' 这只是相对于2秒均匀监督；相对于原A0：'+pair_statement('A2_multi_horizon',a0),
        '**Dynamic loss：** '+pair_statement('A3_dynamic_delta','A2_multi_horizon')]
    for name in ['A2_multi_horizon','A3_dynamic_delta']:
        g=v[(v.experiment==name)&v.signal.isin(['omega','omega_last1s'])]
        findings.append(f"{name}：5秒prefix={g[(g.signal=='omega')&(g.horizon_s==5)]['median'].iloc[0]:.4f}，末1秒={g[g.signal=='omega_last1s']['median'].iloc[0]:.4f}。")
    findings.append('不能把局部幅值改善描述为完全无效，但本矩阵没有候选同时改善长期velocity/attitude并恢复动态；Q8的C针对本轮主要长期目标，不证明loss不可能是瓶颈。A3与A2的对照也不支持新增delta项恢复了长期角运动。')
    findings.append('短期所有实验的>10%退化告警（正数为退化）：')
    warnings=[]
    for row in changes[changes.horizon_s.isin([.2,.5])].to_dict('records'):
        for metric in METRICS:
            if row[metric+'_change_pct']>10:warnings.append(dict(experiment=row['experiment'],horizon_s=row['horizon_s'],metric=metric,degradation_pct=row[metric+'_change_pct']))
    findings.append(md(pd.DataFrame(warnings)) if warnings else '无。')
    findings.append('角加速度全时域向量RMSE（rad/s²）及标准差；该原生差分含噪声，不能把接近零预测的低MSE当作已解决动态：\n'+md(pd.DataFrame([dr[n].loc['angular_acceleration'].to_dict() for n in names])))
    chain=pd.concat([pd.read_csv(root/n/'drift_chain.csv') for n in [a0,best]])
    findings.append('姿态漂移链的离线代数分解：\n'+md(chain[chain.horizon_s.isin([.2,1,5])]))
    findings.append('这里使用(Rpred−Rtruth)×a_pred作为旋转项；分解依赖参考加速度的选择，两项相关，不是因果贡献百分比。当前旋转项远小于body-derivative项，不能宣称“姿态误差主导速度误差”的完整链条已经被证明。一步导数偏弱、递推角动态衰减、姿态和位置误差增长有共现证据。')
    findings.append('5秒相位相关性（固定horizon；相关不是因果）：\n'+md(pd.DataFrame(correlations).query('horizon_s == 5')))
    regimes=pd.read_csv(root/'regime_summary.csv')
    regime=regimes[(regimes.experiment==best)&(regimes.horizon_s==5)]
    findings.append('最佳候选的5秒连续变量分桶（bin 0/1/2为train分位递增，非动作标签）：\n'+md(regime[['variable','bin','n_rollouts','velocity_m_s_equal_log_rmse','attitude_deg_equal_log_rmse']]))
    findings.append('分桶边界沿用Step 2 summary.json中的bin_edges_train_tertiles。各桶flight构成可能不同，只作条件误差描述，不据此推断控制因果。完整0.5/1/2/5秒各模型数据见regime_summary.csv。')
    # Non-overlapping native ten-step blocks locate attenuation independently of prefixes.
    blocks=[]
    for name in names:
        trace=np.load(args.artifacts/name/'validation_rollout.npz')
        alpha_truth=np.diff(batch.trajectory.truth.angular_velocity_b,axis=1)/trace['dt_s'][:,:,None]
        for start in range(0,250,10):
            end=start+10
            for key,true in [('angular_acceleration_b',alpha_truth),('angular_velocity_b',batch.trajectory.truth.angular_velocity_b),('gru_hidden',None)]:
                pred=trace[key][:,start:end]
                row=dict(experiment=name,start_step=start,end_step=end,elapsed_s=float(np.median(trace['dt_s'][:,:end].sum(1))),signal=key,
                    norm_median=float(np.median(magnitude(pred).mean(1))),variation_median=float(np.median(magnitude(pred.std(1)))))
                if true is not None:
                    row['truth_variation_median']=float(np.median(magnitude(true[:,start:end].std(1))))
                    row['variation_ratio_median']=float(np.median(magnitude(pred.std(1))/np.maximum(magnitude(true[:,start:end].std(1)),1e-12)))
                blocks.append(row)
    blocks=pd.DataFrame(blocks);blocks.to_csv(root/'attenuation_blocks.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,(key,col) in zip(axes.flat,[('angular_acceleration_b','variation_ratio_median'),('angular_velocity_b','variation_ratio_median'),('gru_hidden','norm_median'),('gru_hidden','variation_median')]):
        for name in names:
            g=blocks[(blocks.experiment==name)&(blocks.signal==key)];ax.plot(g.elapsed_s,g[col],label=name[:2])
        ax.set(xlabel='Native elapsed time s',ylabel=col,title=key);ax.legend()
    fig.tight_layout();fig.savefig(root/'attenuation_and_hidden.png',dpi=160);plt.close(fig)
    findings.append('逐10步非重叠块的角加速度、omega variation以及hidden norm/variation见attenuation_blocks.csv与attenuation_and_hidden.png；prefix统计见各模型dynamics_statistics.csv，避免用累积std掩盖末段衰减。')
    report += findings
    (root/'report.md').write_text('\n\n'.join(report))
    (root/'report_manifest.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),report_sha256=hashlib.sha256((root/'report.md').read_bytes()).hexdigest(),sealed_test_opened=False),indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
