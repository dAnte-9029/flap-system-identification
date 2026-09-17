#!/usr/bin/env python3
"""Evidence-limited Step 6 report: distinguish local oracle from simulator evidence."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-phase-reference')
import sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report_main_v2_objectives import md,bootstrap_percent
from system_identification.models.phase_reference import harmonic_design


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);a=pa.parse_args();out=a.output
    gate=json.loads((out/'probe_gate.json').read_text());template=json.loads((out/'canonical_template.json').read_text());choice=json.loads((out/'selected_estimator.json').read_text())
    if gate['full_training_authorized_by_gate']:raise RuntimeError('Probe gate passed: full simulator experiments are required before a final conclusion')
    m=pd.read_csv(out/'teacher_metrics.csv');lags=pd.read_csv(out/'phase_lag_vs_horizon.csv');freq=pd.read_csv(out/'frequency_bias_analysis.csv');consistency=pd.read_csv(out/'phase_conditioned_consistency.csv')
    means=m[m.scope=='validation'].groupby(['contract','trained_target','evaluation_target','signal']).rmse.mean().reset_index()
    rows=[];seeds=[]
    for (target,evaltarget,signal),g in m[m.scope=='validation'].groupby(['trained_target','evaluation_target','signal']):
        base=g[g.contract=='P0'].groupby('log_id').rmse.mean().sort_index()
        for name in ['P1','P2','P_oracle']:
            cand=g[g.contract==name].groupby('log_id').rmse.mean().sort_index();lo,hi=bootstrap_percent(cand,base)
            rows.append(dict(contract=name,trained_target=target,evaluation_target=evaltarget,signal=signal,baseline_rmse=base.mean(),candidate_rmse=cand.mean(),change_pct=100*(cand.mean()/base.mean()-1),ci95_low=lo,ci95_high=hi,improved_flights=int((cand<base).sum()),oracle_not_deployable=name=='P_oracle'))
            for seed in g.seed.unique():
                b=g[(g.contract=='P0')&(g.seed==seed)].rmse.mean();c=g[(g.contract==name)&(g.seed==seed)].rmse.mean()
                seeds.append(dict(contract=name,seed=seed,trained_target=target,evaluation_target=evaltarget,signal=signal,change_pct=100*(c/b-1)))
    changes=pd.DataFrame(rows);changes.to_csv(out/'probe_changes.csv',index=False);pd.DataFrame(seeds).to_csv(out/'probe_seed_changes.csv',index=False)
    changes[changes.contract=='P_oracle'].assign(scope='LOCAL_PROBE_ONLY_NOT_FULL_SIMULATOR').to_csv(out/'phase_oracle_upper_bound.csv',index=False)
    lag_summary=[];components=[]
    for keys,g in lags.groupby(['model','horizon_s','signal']):
        z=g[g.identifiable];supported=z[z.lag_supported.eq(True)]
        lag_summary.append(dict(model=keys[0],horizon_s=keys[1],signal=keys[2],n_total=len(g),n_full_cycle_fit=len(z),n_lag_supported=len(supported),
            all_full_cycle_median_abs_lag_deg=float(np.rad2deg(z.best_circular_lag_rad.abs().median())),median_abs_lag_deg=float(np.rad2deg(supported.best_circular_lag_rad.abs().median())),p90_abs_lag_deg=float(np.rad2deg(supported.best_circular_lag_rad.abs().quantile(.9))),
            median_shape_correlation=z.shape_correlation.median()))
        if len(z):
            parts=z[['amplitude_mse','phase_mse','dc_mse']].mean();total=parts.sum()
            components.append(dict(model=keys[0],horizon_s=keys[1],signal=keys[2],**parts.to_dict(),
                **{k+'_fraction':float(v/total) for k,v in parts.items()},native_residual_mse=z.native_residual_mse.mean()))
    lag_summary=pd.DataFrame(lag_summary);components=pd.DataFrame(components);lag_summary.to_csv(out/'phase_lag_summary.csv',index=False);components.to_csv(out/'harmonic_error_decomposition.csv',index=False)
    # All plots are fixed summaries, not a hyperparameter-selection loop.
    phi=np.arange(360)*2*np.pi/360;y=harmonic_design(phi)@np.array(template['coefficients'])
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for j,label in enumerate(['q_dot rad/s²','az_n m/s²']):ax[j].plot(np.rad2deg(phi),y[:,j]);ax[j].set(xlabel='Train canonical phase deg',ylabel=label)
    fig.tight_layout();fig.savefig(out/'canonical_phase_templates.png',dpi=160);plt.close(fig)
    e=pd.read_csv(out/'phase_offset_estimation.csv');fig,ax=plt.subplots(figsize=(8,5))
    for (method,k),g in e[e.train_temporal_holdout&e.complete_history].groupby(['method','requested_history_steps']):
        x=np.sort(np.rad2deg(g.reference_offset_error_rad));ax.plot(x,np.arange(1,len(x)+1)/len(x),label=f'{method} {k} intervals')
    ax.set(xlabel='Train-reference absolute circular error deg (not mechanical truth)',ylabel='Empirical CDF',xlim=(0,90));ax.legend();fig.tight_layout();fig.savefig(out/'offset_error_distribution.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,signal in zip(axes,['q_dot','az_n']):
        for model,g in lag_summary[lag_summary.signal==signal].groupby('model'):
            values=g.median_abs_lag_deg.where(g.n_lag_supported>=.5*g.n_total)
            ax.plot(g.horizon_s,values,'o-',label=model)
        ax.set(title=signal,xlabel='Horizon s',ylabel='Median |circular lag| deg');ax.legend()
    fig.suptitle('Lag shown only where at least half the origins have supported fits');fig.tight_layout();fig.savefig(out/'phase_lag_vs_horizon.png',dpi=160);plt.close(fig)
    for signal,file in [('q_dot','qdot_phase_alignment.png'),('az_n','az_phase_alignment.png')]:
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for model in ['S0','S1']:
            g=components[(components.model==model)&(components.signal==signal)]
            for component in ['amplitude_mse','phase_mse','dc_mse']:axes[0].semilogy(g.horizon_s,g[component],label=model+' '+component)
            z=lag_summary[(lag_summary.model==model)&(lag_summary.signal==signal)];axes[1].plot(z.horizon_s,z.median_shape_correlation,'o-',label=model)
        axes[0].set(xlabel='Horizon s',ylabel='Uniform-phase harmonic MSE',title=signal);axes[0].legend(fontsize=8)
        axes[1].set(xlabel='Horizon s',ylabel='Shape correlation',title='No fitted lag is fed back');axes[1].legend();fig.tight_layout();fig.savefig(out/file,dpi=160);plt.close(fig)
    f=pd.read_csv(out/'free_running_per_horizon.csv');fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,metric in zip(axes,['velocity_m_s','attitude_deg']):
        for name,g in f.groupby('experiment'):ax.plot(g.horizon_s,g[metric+'_equal_log_rmse'],'o-',label='Frozen Step5 '+name)
        ax.set(xlabel='Horizon s',ylabel=metric);ax.legend()
    fig.suptitle('Existing references only: no full phase model passed the training gate');fig.tight_layout();fig.savefig(out/'free_running_comparison.png',dpi=160);plt.close(fig)
    main=changes[(changes.trained_target=='increment2')&(changes.evaluation_target=='increment2_state')&changes.signal.isin(['angular','linear'])]
    matched=pd.read_csv(out/'matched_history_ablation.csv');st=pd.read_csv(out/'offset_stability.csv');hist=pd.read_csv(out/'history_length_ablation.csv')
    summary=dict(result='C' if not gate['oracle_pass'] else 'B',result_scope='Current constant-offset template reference and matched local probes; NOT a measured negative full-simulator oracle result',
        full_models_trained=False,reason='User-required probe gate failed; do not train full Main V2 without oracle benefit',
        selected_estimator=choice,probe_gate=gate,unknown_cross_log_zero_deployability='PARTIAL',structural_readiness=100,dynamics_fidelity_readiness=60,rl_ready=False,sealed_test_opened=False,
        validation_oracle_exception='Full-log truth used only for labelled oracle offsets/probes; never for P2 templates, weights, estimator selection or reset',
        local_increment_results=main.to_dict('records'))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    report=['# Step 6 — Deployable Phase Reference under Unknown Cross-Log Mechanical Zero',
        '**结论 C（范围受限）：当前constant cross-log offset路线没有显示足够新增预测收益，按预先定义的probe gate停止完整模型训练。** 这不是“已经测得完整Main V2 oracle的5s结果无效”，也不证明所有phase问题都无关；本轮没有训练F1/F2/F_oracle。',
        'past-only E2确实能显著改善训练模板在validation的相位条件一致性，但合法26点history probe已间接利用了大部分相关信息。额外给出oracle reference后，2step Δω只有约0.32%改善、Δv约2.34%；不足以支持再训练完整Main V2。架构、actuator constants、旧checkpoint、Step1–5结果及train/validation split均保持不变。',
        '## 边界、命名与可复现合同',
        'P0=每window φ−φt0；P1=raw within-log φ；P2=φ+past-only估计offset；P_oracle=φ+full-log truth alignment offset，**ORACLE / NOT DEPLOYABLE**。P2在本轮是phase contract，勿与Step5的P2 state-increment优先级混淆。',
        'canonical gauge固定为第一个排序training log的统计相位参考，使用train前70%减2s purge拟合3阶harmonic模板。没有可信机械trigger，所以“oracle”也是针对该统计模板的整日志最佳常数对齐，不是实测机械真值，更不是数学上保证优于所有因果模型的性能上界。validation oracle文件与deployable选择明确隔离。',
        '先冻结train模板、E1/E2和12/25/50步history选择，再生成单独oracle文件，再训练probe。E1仅用past body-rate q backward差分；E2加past navigation velocity z backward差分，az_n不是IMU specific force。v_NED是已有可观测state；仍需真实系统导航链的延迟/带宽验证，不能把数据对齐后的50Hz直接当独立高频加速度传感器。',
        'backward difference只使用[t−1,t]，无中心差分、无zero-phase filter。模板使用区间harmonic平均，即sin/cos中点乘sinc(kΔφ/2)，在线性区间phase假设下显式处理差分区间时移与幅频响应。匹配DC和非负gain只用当前history，权重为train variance倒数，角度网格1°，没有validation调参。',
        '## Q1：oracle上限改善多少？',
        md(main),
        '单位：increment2_state为真实native两步duration乘probe平均导数后的Δv(m/s)/Δω(rad/s)。所有contract训练相同420→64→64→6 MLP、30epoch、AdamW 0.001、batch512、seeds415/416/417，fit第一70%各train log、purge2s、stride2；固定原始target normalization。分别训练raw与2step-average target，共24个小诊断模型，不是24个Main V2模型。',
        '95%CI对五条flight做cluster bootstrap，先平均配对seeds；每seed变化另存probe_seed_changes.csv，不能把重叠起点当独立样本。raw target下oracle三维角导数反而变差，完整数字如下：',
        md(changes[(changes.trained_target=='D0_raw')&(changes.evaluation_target=='D0_raw')]),
        '完整simulator oracle上限：**NOT RUN**。您的条件“P_oracle≈P0则停止”已触发，不为填表而继续完整训练。phase_oracle_upper_bound.csv明确标记LOCAL_PROBE_ONLY。',
        '## Q2：仅past history能否恢复canonical phase？',
        '对于训练定义的统计参考：有较强证据；对于绝对mechanical zero：本数据不能验证。固定模板用φ(t0)、f(t0)及native dt预测下一间隔，不使用future encoder phase；下表validation没有拟合gain、DC、权重或offset（oracle列除外且不可部署）。',
        md(consistency[consistency.split=='validation'].groupby(['contract','signal'])[['rmse','r2']].mean().reset_index()),
        'P2的az/q_dot模板R²约0.936/0.647，但这不能直接推导完整recurrent simulator也会改善。局部窗口估计可适应部分非平稳记录相位，因此P2有时优于常数oracle；不能将它解释为发现了真正机械零点。',
        '![Templates](canonical_phase_templates.png)',
        '## Q3：需要多少history / cycles？',md(matched),
        '匹配同12,168个train temporal holdout起点后，E2的0.5s与1s在1°网格下几乎并列：median≈6°、p90≈17°、p95≈20°；没有证据称1s必需。脚本按预声明median+p90选中了1s，微小浮点差及原始eligible cohort差异不是1s显著更好的证据。E1尾部更差。validation没有absolute offset error。',
        '低于一个周期的训练子集（样本很少的桶不能泛化）：',md(hist[(hist.split=='train')&(hist.cycle_bin=='less_than_one_cycle')]),
        '![Offset CDF](offset_error_distribution.png)',
        '选择的E2/50在validation逐点重估的45°以上jump率为0；这只是日志上的稳定性诊断，不采用逐点更新驱动simulator。实际合同是episode reset估一次并固定，history不足时使用可用past并显式记录duration/cycles/fallback。',
        md(st[st.split=='validation'].groupby(['method','history_steps'])[['median_circular_jump_rad','p95_circular_jump_rad','jump_over_45deg_fraction']].mean().reset_index()),
        '## Q4：S1长期退化是否伴随phase lag？',md(lag_summary),
        'YES，S1 q_dot/az在0.5→5s的lag约15.5→38°、7.5→39.5°。0.2s未覆盖完整周期，不能稳定估3阶circular lag，保留NaN及有效数量，不伪造0.2s点。S0后期振幅很弱，lag角度缺少可辨识性；lag_supported额外要求预测harmonic RMS≥truth的10%，是透明的弱信号标记，不是物理安全门。图中若某horizon不足半数起点通过弱信号检查，则不连接该点；表保留全部数量，避免把S0末期仅2个可用q_dot样本误读为lag下降。',
        '![Lag](phase_lag_vs_horizon.png)',
        '## Q5：amplitude / phase / DC谁主导？',md(components[components.horizon_s==5]),
        '这是uniform phase grid上的精确harmonic误差分解：每阶0.5(Apred−Atruth)² + Apred·Atruth·(1−cosΔθ)，另加DC²。native未解释residual单独保留。S1的az以phase误差为主，q_dot仍以幅值不足为主；不支持“幅值都已恢复，剩下全部是phase”。小DC误差能长期一致积累，不能用它在高频误差能量中占比小来排除轨迹bias。该分解不是对position/attitude误差的因果百分比分摊。',
        '![qdot](qdot_phase_alignment.png)\n\n![az](az_phase_alignment.png)',
        '频率误差与phase-state drift：',md(freq[freq.horizon_s==5]),
        '每log mean bias、约1s慢分量、zero-mean high-pass residual分开保存；残差不被假定为白噪声。2π∫Δfdt与phase drift相关约0.995–0.998，encoder phase与自身frequency trapezoid积分仍有约0.12–0.14rad closure residual。完整代数identity及constant/slow/residual积分在frequency_integral_per_rollout.csv；这些是离线分解，不能当部署phase校正。',
        '常数cross-log offset只校正reset reference，不能消除后续frequency error积分造成的phase drift；本轮probe的负结果不能否定后一种同步问题。S1 phase lag增长与drift相伴，但未证明它造成S1全部轨迹退化。autonomous benchmark不读t0后encoder。在线predictor可另研究真实encoder观测更新；RL simulator phase是内部真状态，两者不能混成一个精度结论。',
        '## Q6：causal phase是否改善teacher-state及长预测？',
        md(main[main.contract=='P2']),
        'teacher Δω/Δv只有约1.04%/1.82%改善；q_dot/az部分单通道改善而三维raw angular aggregate退化。0.5–1s及2–5s新phase候选：**NOT EVALUATED**，因为oracle probe收益不足。下面图及free_running_per_horizon/variation_summary/regime_summary仅引用已冻结S0/S1的722共同起点，不冒称P1/P2/F1的新结果。',
        '![Frozen comparison](free_running_comparison.png)',
        '## Q7：unknown cross-log zero是否成为部署时可解决问题？ PARTIAL',
        '已实现对已记录信号有效且past-only的统计相位对齐；尚无跨log mechanical truth、真实传感器延迟验证，低于一个周期有明显歧义，也没有完整simulator fidelity收益。不能称问题已完全解决。',
        'state合同：保留raw phase；P0 anchor=φt0，P1 anchor=0，P2 anchor=−offset，canonical phase=wrap(raw−anchor)。phase_anchor本已在SimulatorState snapshot中，故不重复保存可推导offset。reset_phase仅在reset估计/接收offset；step与restore不重估、不重编码history。新API的250步/100+150步续跑已用严格相等回归验证；它是接口能力，不代表旧checkpoint适合新phase输入。',
        '## Q8：选择C，并停止当前constant-offset phase路线',
        'oracle在相同合法history probe中没有显示足够新增收益，因而不进入完整Main V2 phase训练。下一步应调查transition / recurrent representation如何利用已有history信息，而不是继续搜索offset权重或增加网络大小。本轮证据无法做“完整simulator oracle已证明无效”的更强结论；论文方法候选A未成立，oracle明显有效但估计不足的B也不符合数据。',
        'Simulator structural readiness: **100%**\n\nDynamics fidelity readiness: **60%**\n\nRL-ready: **NO**',
        '## 复现与验收',
        '```bash\n/home/zn/anaconda3/envs/flap-train-gpu/bin/python /home/zn/flap-system-identification/scripts/run_main_v2_phase_reference.py --output /tmp/main-v2-phase-reference/results --artifacts /tmp/main-v2-phase-reference/probes --device cuda:1\n```',
        '使用空目录；依次构建train模板、past-only估计、独立oracle、24个固定预算probe、gate、冻结trace诊断、报告和回归。若复现时gate意外通过，脚本会明确停止要求完整模型实验，不会静默输出C。pytest、git diff、旧文件hash及source hashes见verification.json。']
    (out/'report.md').write_text('\n\n'.join(report));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
