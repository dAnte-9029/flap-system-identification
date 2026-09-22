"""Paper tables, native-time curves and bilingual interpretation, no inference."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report_paper_standard_gru_multiseed import table
from system_identification.evaluation.future_control_diagnostic import METRICS,squared_errors
from system_identification.models.trajectory import TrajectoryPrediction

LABELS={'position_rmse_m':'Position vector RMSE [m]','velocity_rmse_m_s':'Velocity vector RMSE [m/s]',
        'attitude_error_deg':'Attitude geodesic RMS [deg]','body_rate_rmse_rad_s':'Body-rate vector RMSE [rad/s]'}
DYNAMICS=METRICS[1:]
COLORS={'Actual':'#2166ac','Hold':'#b2182b'}


def formatted(multi):
    frame=multi.copy()
    frame['mean ± seed SD']=frame.apply(lambda r:f'{r["mean"]:.5f} ± {r["std"]:.5f}',axis=1)
    return frame.pivot(index=['kind','cohort','group','step','condition'],columns='metric',values='mean ± seed SD').reset_index()


def euler(q):
    q=q/np.linalg.norm(q,axis=-1,keepdims=True);w,x,y,z=q.T
    values=np.stack([np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)),np.arcsin(np.clip(2*(w*y-z*x),-1,1)),np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))],axis=-1)
    return np.degrees(np.unwrap(values,axis=0))


def findings(paired):
    p=paired.query('level=="macro" and kind=="endpoint" and step==25 and cohort=="ALL"')
    overall=p[p.group=='ALL'].set_index('metric')
    high=p[p.group=='high'].set_index('metric')
    all_positive=all(overall.loc[m,'absolute_gain']>0 for m in DYNAMICS)
    high_positive=len(high)==4 and all(high.loc[m,'absolute_gain']>0 for m in DYNAMICS)
    if all_positive and high_positive:
        core='实际未来命令在总体及高控制变化组均降低三项主要动力学预测误差，支持其具有增量预测信息。'
    elif high_positive:
        core='高控制变化组的实际命令回放改善三项主要动力学指标，但总体结果不一致，输入信息价值具有工况依赖。'
    else:
        core='实际命令的信息价值随指标和控制变化组而变，结果不支持无条件的统一改善结论。'
    english='At the nominal 500 ms horizon, '
    english+='; '.join(f'{m.replace("_", " ")}: Actual {overall.loc[m,"actual"]:.4f}, Hold {overall.loc[m,"hold"]:.4f} ({overall.loc[m,"relative_gain_pct"]:.2f}% error reduction)' for m in DYNAMICS)+'. '
    if len(high):english+='In the train-threshold high-control-change subset, the corresponding reductions were '+', '.join(f'{high.loc[m,"relative_gain_pct"]:.2f}%' for m in DYNAMICS)+'. '
    english+='Metrics were computed within each flight, averaged equally across flights, and summarized over three frozen seeds. These comparisons quantify the incremental predictive information of logged future commands under the observed flight distribution.'
    return dict(core_conclusion=core,results_paragraph=english,overall_dynamics_positive=all_positive,high_dynamics_positive=high_positive,
        engineering_pass_is_not_scientific_significance=True,no_success_threshold=True)


def generate(out,art,protocol,batch):
    out=Path(out);art=Path(art);root=out.parents[3]
    multi=pd.read_csv(out/'multiseed_summary.csv');paired=pd.read_csv(out/'paired_differences.csv');curve=pd.read_csv(out/'error_evolution.csv')
    coverage=pd.read_csv(out/'control_group_coverage.csv');threshold=pd.read_csv(out/'control_group_thresholds.csv')
    times=pd.read_csv(out/'native_elapsed_time.csv')
    tables=formatted(multi);primary=tables.query('kind=="endpoint" and step==25')
    primary.to_csv(out/'paper_table_500ms.csv',index=False)
    endpoint=tables.query('kind=="endpoint"');endpoint.to_csv(out/'paper_table_all_horizons.csv',index=False)
    macro=paired.query('level=="macro"').copy()
    primarygain=macro.query('kind=="endpoint" and step==25')
    primarygain.to_csv(out/'paper_table_500ms_gains.csv',index=False)
    conclusion=findings(paired)
    (out/'interpretation.json').write_text(json.dumps(conclusion,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    def save(fig,name):
        fig.tight_layout()
        for suffix in ('png','pdf'):fig.savefig(out/f'{name}.{suffix}',dpi=180)
        plt.close(fig)
    for metric in DYNAMICS:
        fig,ax=plt.subplots(figsize=(6.5,4))
        for condition in ('Actual','Hold'):
            g=curve[(curve.metric==metric)&(curve.condition==condition)&(curve.cohort=='ALL')&(curve.group=='ALL')].sort_values('step')
            x=g.elapsed_median_s.to_numpy();y=g['mean'].to_numpy();sd=g['std'].to_numpy()
            ax.plot(x,y,label=condition,color=COLORS[condition],lw=2)
            ax.fill_between(x,y-sd,y+sd,alpha=.15,color=COLORS[condition])
        ax.set(xlabel='Median elapsed native time across origins [s]',ylabel=LABELS[metric],title='Equal-flight error; bands = ±1 seed SD (n=3)')
        ax.legend();ax.grid(alpha=.2);save(fig,f'error_evolution_ALL_{metric}')
        fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
        for ax,group in zip(axes,['low','middle','high']):
            for condition in ('Actual','Hold'):
                g=curve[(curve.metric==metric)&(curve.condition==condition)&(curve.cohort=='ALL')&(curve.group==group)].sort_values('step')
                if len(g):
                    x=g.elapsed_median_s.to_numpy();y=g['mean'].to_numpy();sd=g['std'].to_numpy()
                    ax.plot(x,y,label=condition,color=COLORS[condition]);ax.fill_between(x,y-sd,y+sd,alpha=.15,color=COLORS[condition])
            ax.set(title=f'{group}: fixed 500 ms membership',xlabel='Median native elapsed time [s]');ax.grid(alpha=.2)
        axes[0].set_ylabel(LABELS[metric]);axes[0].legend();save(fig,f'error_evolution_fixed_groups_{metric}')
        fig,ax=plt.subplots(figsize=(6.5,4))
        for offset,condition in [(-.12,'Actual'),(.12,'Hold')]:
            g=multi[(multi.kind=='endpoint')&(multi.step==25)&(multi.cohort=='ALL')&(multi.metric==metric)&(multi.condition==condition)]
            g=g.set_index('group').reindex(['low','middle','high'])
            ax.errorbar(np.arange(3)+offset,g['mean'],yerr=g['std'],fmt='o-',capsize=4,label=condition,color=COLORS[condition])
        ax.set(xticks=range(3),xticklabels=['Low','Middle','High'],ylabel=LABELS[metric],title='500 ms; train-derived control-change groups')
        ax.legend();ax.grid(alpha=.2);save(fig,f'control_group_500ms_{metric}')
    # Cases fixed from identities before inspecting Hold outputs; never choose a seed by error.
    with np.load(root/protocol['actual_predictions']['17'],allow_pickle=False) as f:
        actual=TrajectoryPrediction(**{k:f[k] for k in vars(batch.trajectory.truth)})
    with np.load(art/'Hold_seed17_predictions.npz',allow_pickle=False) as f:
        hold=TrajectoryPrediction(**{k:f[k] for k in vars(batch.trajectory.truth)})
    errors={name:squared_errors(pred,batch.trajectory.truth) for name,pred in [('Actual',actual),('Hold',hold)]}
    cases=[]
    for case in protocol['case_selection']['cases']:
        if case.get('missing'):continue
        i=case['source_index'];k=25;t=np.r_[0,batch.trajectory.dt_s[i,:k].cumsum()]
        u=batch.trajectory.controls[i,:k]
        fig,axes=plt.subplots(5,3,figsize=(13,12),sharex=True)
        for j,(row,col) in enumerate([(0,0),(0,1),(0,2),(1,0)]):
            ax=axes[row,col];ax.step(t,np.r_[u[:,j],u[-1,j]],where='post',label='Actual command',color=COLORS['Actual'])
            ax.step(t,np.full(len(t),u[0,j]),where='post',label='Hold command',color=COLORS['Hold'],ls='--')
            ax.set_ylabel(['motor','left elevon','right elevon','rudder'][j]+' command');ax.grid(alpha=.2)
        for ax in axes[1,1:]:ax.axis('off')
        axes[1,1].text(0,.6,'Commands are normalized allocation signals.\nNot physical angles or measured frequency.\nBoth predictions compared with original Actual truth.',fontsize=9)
        for row,(field,names,unit) in enumerate([('velocity_n',['vN','vE','vD'],'m/s'),('quaternion_nb',['roll','pitch','yaw'],'deg'),('angular_velocity_b',['p','q','r'],'rad/s')],start=2):
            for name,pred in [('Truth',batch.trajectory.truth),('Actual',actual),('Hold',hold)]:
                values=getattr(pred,field)[i,:k+1];values=euler(values) if field=='quaternion_nb' else values
                for j in range(3):
                    axes[row,j].plot(t,values[:,j],label=name,color='black' if name=='Truth' else COLORS[name],ls='--' if name=='Hold' else '-')
                    axes[row,j].set_ylabel(f'{names[j]} [{unit}]');axes[row,j].grid(alpha=.2)
        axes[0,0].legend(fontsize=8);axes[2,0].legend(fontsize=8)
        for ax in axes[-1]:ax.set_xlabel('Native time from origin [s]')
        fig.suptitle(f'{case["name"]}, seed17\n{case["window_id"]}',fontsize=10)
        save(fig,f'case_{case["name"]}')
        for j,metric in enumerate(METRICS):
            a=float(np.sqrt(errors['Actual'][i,k-1,j]));h=float(np.sqrt(errors['Hold'][i,k-1,j]))
            cases.append(dict(case=case['name'],window_id=case['window_id'],seed=17,step=k,metric=metric,actual=a,hold=h,absolute_gain=h-a))
    pd.DataFrame(cases).to_csv(out/'representative_case_metrics.csv',index=False)
    discussion=('Both conditions were evaluated against the same observed trajectory generated under the logged commands. '
        'Hold is therefore an information-limited forecast control, not a counterfactual flight with matched ground truth. '
        'Closed-loop feedback and correlated state estimates can contribute to the predictive association. '
        'These results do not establish causal accuracy for arbitrary control actions, closed-loop improvement, or suitability for long-horizon simulation/RL. '
        'Control groups use realized future commands offline; three-seed SD describes initialization sensitivity, not predictive uncertainty or a confidence interval.')
    (out/'paper_results_paragraph.txt').write_text(conclusion['results_paragraph']+'\n')
    (out/'paper_discussion_paragraph.txt').write_text(discussion+'\n')
    gaincols=['cohort','group','metric','actual','hold','absolute_gain','relative_gain_pct','seeds_improved','flights_improved','n_flights','n_origins']
    overall=primary.query('cohort=="ALL" and group=="ALL"')
    grouped=primary.query('cohort=="ALL" and group!="ALL"')
    direction=primarygain[primarygain.metric.isin(DYNAMICS)]
    evolution_ends=curve[(curve.cohort=='ALL')&(curve.group=='ALL')&curve.step.isin([1,5,10,25,50])]
    short=[
        '# Frozen Future-Control Information Diagnostic',
        '## Research question and protocol',
        'Under identical initial states, H26 context and frozen GRU64 dynamics, does replaying realized future commands predict observed motion better than holding the origin command? No training or checkpoint selection. All3seeds17/23/42 retained. Full protocol is frozen before Hold inference, including train-only thresholds and case identities.',
        'Actual reuses the original4-channel future tape; Hold repeats its first command50times. Controls are normalized motor/left/right/rudder allocation commands before PWM; not physical angles/frequency. History, true origin, zero hidden initialization before history encoding, origin phase reference, future labels and native dt are unchanged.',
        '## Coverage and control grouping',table(threshold),table(coverage.query('step==25')),
        'E_K is RMS across time and channels of (u[t+k]-u[t])/frozen_control_std, k=0..K-1. Train25th/75th linear quantiles define low<=q25, middle(q25,q75], high>q75. Degenerate thresholds disable grouping. Horizon-specific groups may differ; all evolution group plots keep the500ms membership fixed. Full per-flight zero-inclusive coverage: control_group_per_flight.csv. These are offline groups using realized future commands, not an online foreknowledge classifier.',
        '## Primary500ms results',table(overall[['condition']+list(METRICS)]),table(grouped[['group','condition']+list(METRICS)]),
        '## Paired information gains',table(direction[gaincols]),
        'Positive gain=Hold−Actual. Relative gain=100*(Hold−Actual)/Hold at the reported macro error level; zero denominators are undefined (blank CSV/NaN), never epsilon-adjusted. Seed rows pair identical seeds; flight rows first average the3seed flight errors. No window-independence tests.',
        '## All endpoint horizons',table(endpoint[['cohort','group','step','condition']+list(METRICS)]),
        '## Error evolution and interval metric',
        'Error_evolution.csv retains steps1..50, nominal step*.02 labels, and native cumulative elapsed-time distributions. Figure x is median elapsed time over the participating origins at each step, not a common timestamp or resampled trajectory. Mean and shaded±1SD use equal-flight error followed by3seeds. The median time distribution is origin-weighted; errors are flight-weighted.',
        table(evolution_ends[['condition','step','metric','mean','std','elapsed_median_s','elapsed_min_s','elapsed_max_s']]),
        'Auxiliary interval metric: I_i,K² = sum_{k=1..K} dt_i,k-1 * e_i,k² / sum_{k=1..K}dt_i,k-1. This uses right-endpoint errors, excludes t0, then averages I_i,K² equally over origins in each flight before square root. Flights are averaged equally, then seeds summarized. No duration pooling across origins and no replacement of frozen endpoint metrics. See trajectory_interval_errors.csv.',
        table(formatted(multi.query('kind=="interval" and cohort=="ALL" and group=="ALL"'))[['step','condition']+list(METRICS)]),
        '## Dynamic response cases',
        'All three predetermined cases are retained: original Step1 representative plus identity-median high-E25 origins from Sep7 and Sep17. Seed17 fixed; no best-seed selection. Each figure shows all4commands, vN/vE/vD, illustrative Euler angles and p/q/r over25native steps. Quantitative attitude remains quaternion geodesic error. A case may favor Hold; it is never omitted.',table(pd.DataFrame(cases)),
        '## Scientific interpretation',conclusion['results_paragraph'],
        '## Limitations',discussion,
        'History horizon-dependence remains the frozen Step4 Mixed result; no H1/H5/H13 inference or new history conclusions were generated. Dedicated flight input-excitation data would be needed to validate action-response direction, timing and amplitude under interventions.',
        '## Engineering verification and sealed status',
        'Tests, actual replay parity on first128origins per seed, constant-command equality, shared first-step prediction, label poisoning/prefix checks, finite values/unit quaternions/native timing, identical coverage and bitwise unchanged model buffers passed. Required explicit artifacts are hash-pinned; unrelated registry/workspace differences are recorded rather than repaired. Only train/open-validation are used. Sealed Sep8/reserved Sep19 remain unopened. No automatic next experiment or commit/push.',
    ]
    (out/'report.md').write_text('\n\n'.join(short)+'\n')
    zh=[
        '# Paper Step5：未来控制输入预测价值与误差演化',
        '## 1. 本实验回答的论文问题',conclusion['core_conclusion'],
        '问题严格限定为：同一历史、起点和冻结动力学模型下，知道后续实际命令，是否比假设命令维持当前值更有助于预测原始真实飞行？这是已知输入回放的信息价值诊断，不是训练优化或干预辨识。',
        '## 2. 实验设置与覆盖情况',
        'Standard GRU64 / H26，seed17/23/42，last epoch65，原归一化不变。41个train flights/28,293origins仅提供固定控制变化分位阈值；17个validation flights/2,582origins全部保留，Sep7为9flights、Sep17为8flights。Actual完整预测复用；Hold仅从第2条未来命令起改变4通道，history及其末尾u_t、初始状态、phase、dt不变。控制量为归一化分配命令，不是实际舵角或扑频。',
        table(threshold),table(coverage.query('step==25')[['cohort','group','n_origins','n_flights','E_median','E_p95']]),
        '所有误差均为每flight内RMSE→flight等权平均→三seed mean/sampleSD(ddof=1)。某组没有样本的flight不参与该组误差，零计数保存在control_group_per_flight.csv。±seedSD不是预测置信区间或模型不确定性。',
        '## 3. 500 ms主结果',table(overall[['condition']+list(METRICS)]),
        table(primarygain.query('cohort=="ALL" and group=="ALL"')[gaincols]),
        'gain=Hold−Actual；正值有利于Actual。相对增益按聚合误差计算，以Hold为分母；分母0时不可定义。每个seed及先三seed平均后的每flight配对差值保存在paired_differences.csv，没有对大量windows做独立t检验。',
        '## 4. 控制变化分组结果',table(grouped[['group','condition']+list(METRICS)]),
        table(direction.query('group=="high"')[gaincols]),
        'E_K是相对于起点命令的标准化变化RMS，不是total variation。各horizon分别用train的25%/75%分位数分组，组别随horizon可能变化。阈值、identity案例在Hold结果产生前冻结，未看误差调阈值。若阈值相等，则该horizon只保留ALL并记录分布。组别使用已实现的未来命令，不能称为提前可知的在线分类器。',
        '## 5. 不同预测时域及完整误差过程',
        table(endpoint.query('cohort=="ALL" and group=="ALL"')[['step','condition']+list(METRICS)]),
        '1–50步曲线保留原生step，横轴为参与origin累计dt的中位数；原始时间分布另存，既不重采样也不假装各origin共享同一timestamp。分组曲线始终使用固定500ms组别。首步使用相同命令，因此预测一致；后续差异才可能体现future tape的作用。误差端点不必随时间单调增加。',
        '辅助区间轨迹误差公式：每origin的I²=Σ(dt×该步误差²)/Σdt，从第1步到K步，排除t0；flight内对origin的I²等权平均后开方，再flight等权、seed汇总。使用右端点矩形权重。该指标补充整段误差，不替代冻结的四个端点主指标。',
        table(formatted(multi.query('kind=="interval" and cohort=="ALL" and group=="ALL"'))[['step','condition']+list(METRICS)]),
        '已有history报告显示history相对收益集中于较短horizon；本轮仅引用该冻结结论，不重新运行其他H或修改Mixed分类。',
        '## 6. 动态响应案例',
        '采用Step1固定origin，以及Sep7/Sep17各自500ms high组按(log_id,segment_id,start_sample_in_segment)排序的中位origin；均预先固定seed17。缺失cohort案例时记录缺失，不用误差选替代样本。所有存在的案例同时展示Actual/Hold命令、真实运动及两种预测；Euler仅辅助可视化。',table(pd.DataFrame(cases)),
        '## 7. 能够支持的论文结论',conclusion['core_conclusion'],
        '具体效应大小、seed方向和flight方向见主表及分组表；不设“必须改善X%”的科学通过阈值。工程核验通过与科学结果是否支持正向信息增益分别报告。',
        '## 8. 不能支持的结论及限制',
        'Actual和Hold都与同一条原始Actual飞行轨迹比较。Hold是信息受限的预测对照，没有对应真实Hold飞行反事实ground truth；因此不能由其误差断言Hold模拟轨迹本身物理上错误。日志处于闭环，控制命令可能携带状态反馈和策略相关信息。不能直接证明任意候选动作的因果响应、实机控制成功、闭环改善或长期RL仿真能力。有限flight/两session及3seeds也限制外推。',
        '## 9. 对应论文章节与可直接采用的表述',
        'Results: Contribution of Future Control Inputs — 使用paper_table_500ms.csv、paper_table_500ms_gains.csv及control_group_500ms图。Results: Prediction Horizon and Dynamic Response — 使用error_evolution_ALL、fixed_groups及全部三个case图。Discussion — 明确known-input replay与因果响应验证的区别。',
        '> '+conclusion['results_paragraph'],
        '> '+discussion,
        '## 10. 下一步建议，但不自动执行',
        '若论文只主张当前日志分布中的已知输入预测价值，本轮证据可直接使用，无需为该限定结论立即新采数据。若要扩展到任意动作响应或控制用途，则需要专门的真实输入激励实验：在可安全执行的匹配工况下验证命令改变后的响应方向、时延和幅值，因为本轮Hold没有反事实实测。是否开展由用户决定，本轮未启动训练、输入激励采集或其他实验。',
        '封存状态：sealed Sep8/reserved Sep19仍未打开。所有输出位于本实验新目录，旧分类与阈值保持原样。本轮不自动commit/push。建议commit：`feat: add frozen future-control information diagnostic`。',
    ]
    (out/'review_report.md').write_text('\n\n'.join(zh)+'\n')
