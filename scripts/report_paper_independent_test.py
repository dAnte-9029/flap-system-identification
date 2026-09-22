"""Held-out quantitative report; every date/model/group retained, no model selection."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paper_independent_test_core import METRICS
LABELS=dict(zip(METRICS,['Position RMSE [m]','Velocity RMSE [m/s]','Attitude geodesic RMS [deg]','Body-rate RMSE [rad/s]']))
NAMES={'B0':'B0 kinematic','MLP':'MLP','H1':'GRU/H1','H5':'GRU/H5','H13':'GRU/H13','H26':'Standard GRU/H26'}


def markdown(df):
    def f(x):
        if isinstance(x,(float,np.floating)):return '—' if pd.isna(x) else f'{x:.5g}'
        return str(x).replace('|','/')
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(f(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))


def table(m):
    rows=[]
    for (date,model,condition,group,horizon),g in m.groupby(['date','model','condition','group','horizon_s'],sort=False):
        r=dict(date=date,model=NAMES[model],condition=condition,group=group,horizon_s=horizon)
        for metric in METRICS:
            x=g[g.metric==metric].iloc[0]
            r[LABELS[metric]]='unavailable' if pd.isna(x['mean']) else (f'{x["mean"]:.4f}' if model=='B0' else f'{x["mean"]:.4f} ± {x["std"]:.4f}')
        rows.append(r)
    return pd.DataFrame(rows)


def savefig(fig,out,name):
    fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(out/f'{name}.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def plot(out,m,curve,paired,duration):
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    colors={'B0':'#888888','MLP':'#d77d2b','H26':'#1774b4'}
    for metric,label in LABELS.items():
        fig,axes=plt.subplots(1,2,figsize=(10,3.6))
        for ax,date in zip(axes,['Sep8','Sep19']):
            for model in ['B0','MLP','H26']:
                d=m[(m.date==date)&(m.model==model)&(m.condition=='Actual')&(m.group=='ALL')&(m.kind=='endpoint')&(m.metric==metric)].sort_values('horizon_s')
                ax.plot(d.horizon_s,d['mean'],'o-',color=colors[model],label=NAMES[model])
                if model!='B0':ax.fill_between(d.horizon_s,d['mean']-d['std'],d['mean']+d['std'],alpha=.18,color=colors[model])
            ax.set(title=date,xlabel='Nominal horizon [s]',xticks=[.1,.2,.5,1.]);ax.grid(alpha=.2)
        axes[0].set_ylabel(label);axes[1].legend(fontsize=8);fig.suptitle('Held-out dates: equal-flight mean ± 1 seed sample SD')
        savefig(fig,out,'main_horizon_'+metric)
        fig,axes=plt.subplots(1,2,figsize=(10,3.6))
        for ax,date in zip(axes,['Sep8','Sep19']):
            d=m[(m.date==date)&m.model.isin(['H1','H5','H13','H26'])&(m.condition=='Actual')&(m.group=='ALL')&(m.kind=='endpoint')&(m.metric==metric)&(m.step==25)].sort_values('history_steps')
            ax.errorbar(np.arange(len(d)),d['mean'],yerr=d['std'],fmt='o-',capsize=4)
            labels=[]
            for h in d.history_steps:
                t=duration[(duration.date==date)&(duration.session=='ALL')&(duration.history_steps==h)]
                labels.append(f'H{h}\n{t["median"].iloc[0]*1000:.0f} ms' if len(t) else f'H{h}')
            ax.set_xticks(np.arange(len(d)),labels);ax.set_title(date);ax.set_xlabel('Context (median actual span)');ax.grid(alpha=.2)
        axes[0].set_ylabel(label);fig.suptitle('500 ms history comparison; H26 remains main model')
        savefig(fig,out,'history_500ms_'+metric)
    for date in ['Sep8','Sep19']:
        for grouped in [False,True]:
            groups=['low','middle','high'] if grouped else ['ALL']
            fig,axes=plt.subplots(3,len(groups),figsize=(5*len(groups),9),squeeze=False)
            for i,metric in enumerate(METRICS[1:]):
                for j,group in enumerate(groups):
                    ax=axes[i,j]
                    for condition,color in [('Actual','#1774b4'),('Hold','#d77d2b')]:
                        d=curve[(curve.date==date)&(curve.group==group)&(curve.metric==metric)&(curve.condition==condition)].sort_values('step')
                        ax.plot(d.elapsed_median_s,d['mean'],label=condition,color=color)
                        ax.fill_between(d.elapsed_median_s,d['mean']-d['std'],d['mean']+d['std'],alpha=.18,color=color)
                    ax.set(title=group,ylabel=LABELS[metric],xlabel='Median cumulative native dt [s]');ax.grid(alpha=.2)
            axes[0,-1].legend();fig.suptitle(date+': Actual/Hold; '+('fixed 500 ms control groups' if grouped else 'all eligible origins')+'; ±1 seed SD')
            savefig(fig,out,f'future_control_evolution_{date}_{"groups" if grouped else "ALL"}')
        for comparison in ['GRU_vs_MLP','H1_to_H26','H13_to_H26','Actual_vs_Hold']:
            fig,axes=plt.subplots(2,2,figsize=(12,6))
            d=paired[(paired.date==date)&(paired.comparison==comparison)&(paired.step==25)&(paired.group=='ALL')]
            ids=sorted(d.flight_id.unique());pd.DataFrame(dict(index=np.arange(1,len(ids)+1),flight_id=ids)).to_csv(out/f'flight_key_{date}.csv',index=False)
            for ax,(metric,label) in zip(axes.flat,LABELS.items()):
                z=d[d.metric==metric].set_index('flight_id').reindex(ids)
                ax.bar(np.arange(1,len(ids)+1),z.absolute_gain);ax.axhline(0,color='black',lw=.8)
                ax.set(xlabel='Flight index (identity key CSV)',ylabel='Reference − comparison: '+label);ax.grid(axis='y',alpha=.2)
            fig.suptitle(f'{date}: {comparison}, 500 ms; per-flight means across3seeds; positive favors comparison')
            savefig(fig,out,f'paired_{comparison}_{date}')


def generate(out,p):
    out=Path(out);root=out.parents[3];m=pd.read_csv(out/'multiseed_summary.csv');g=pd.read_csv(out/'relative_improvement.csv')
    pair=pd.read_csv(out/'paired_flight_differences.csv');ps=pd.read_csv(out/'paired_seed_differences.csv')
    coverage=pd.read_csv(out/'test_flight_coverage.csv');groups=pd.read_csv(out/'control_group_coverage.csv');dur=pd.read_csv(out/'history_duration_summary.csv')
    curve=pd.read_csv(out/'error_evolution.csv')
    # Explicit names for two distinct dispersions: seed SD vs native-time distribution.
    rename={**{k+'_x':k for k in ['mean','std','min','max']},**{k+'_y':'elapsed_'+k+'_s' for k in ['mean','std','min','max']},
            **{k:'elapsed_'+k+'_s' for k in ['median','p05','p25','p75','p95']}}
    curve=curve.rename(columns=rename);curve.to_csv(out/'error_evolution.csv',index=False)
    endpoint=m[m.kind=='endpoint'];main=endpoint[endpoint.model.isin(['B0','MLP','H26'])&(endpoint.condition=='Actual')&(endpoint.group=='ALL')]
    history=endpoint[endpoint.model.isin(['H1','H5','H13','H26'])&(endpoint.condition=='Actual')&(endpoint.group=='ALL')]
    control=endpoint[endpoint.model=='H26']
    main_table=table(main[main.step==25]);hist_table=table(history[history.step==25]);control_table=table(control[control.step==25])
    for name,t in [('paper_table_main_500ms',main_table),('paper_table_history_500ms',hist_table),('paper_table_future_control_500ms',control_table),('paper_table_all_endpoints',table(endpoint))]:t.to_csv(out/f'{name}.csv',index=False)
    # Historical validation values remain historical, not relabeled test rows.
    path=root/'docs/analysis/results/paper_mlp_multiseed_comparison_v1/multiseed_summary.csv'
    c=json.loads((path.parent/'completion.json').read_text());assert hashlib.sha256(path.read_bytes()).hexdigest()==c['artifact_sha256'][str(path.relative_to(root))]
    val=pd.read_csv(path);vrows=[]
    mapping={'B0':'B0_ConstantVelocity','MLP':'B1_MLP','H26':'B2_StandardGRU'}
    for r in main.itertuples():
        vv=val[(val.model==mapping[r.model])&(val.horizon_s==r.horizon_s)&(val.metric==r.metric)]
        for v in vv.itertuples():vrows.append(dict(test_date=r.date,validation_cohort=v.cohort,model=r.model,horizon_s=r.horizon_s,metric=r.metric,
            test_mean=r.mean,test_seed_sd=r.std,validation_mean=v.mean,validation_seed_sd=v.std,
            test_over_validation_ratio=r.mean/v.mean if v.mean else np.nan))
    vc=pd.DataFrame(vrows);vc.to_csv(out/'validation_test_comparison.csv',index=False)
    plot(out,m,curve,pair,dur)
    bad=pair[pair.absolute_gain<=0];bad.to_csv(out/'paired_flight_exceptions.csv',index=False)
    badseed=ps[ps.absolute_gain<=0];badseed.to_csv(out/'paired_seed_exceptions.csv',index=False)
    gaincols=['comparison','date','group','horizon_s','metric','reference_error','comparison_error','relative_gain_pct','seeds_improved','seeds_worse','flights_improved','flights_worse','n_flights','n_origins']
    main_gain=g[(g.comparison=='GRU_vs_MLP')&(g.step==25)]
    hist_gain=g[g.comparison.isin(['H1_to_H26','H13_to_H26'])&(g.step==25)]
    control_gain=g[(g.comparison=='Actual_vs_Hold')&(g.step==25)]
    summaries=[]
    for date in ['Sep8','Sep19']:
        admitted=coverage[(coverage.date==date)&(coverage.status=='admitted')]
        summaries.append(f'{date}: {len(admitted)}/{int((coverage.date==date).sum())} flights admitted; {int(admitted.final_origins.sum())} fixed origins.')
    results=[]
    for date in ['Sep8','Sep19']:
        d=main_gain[(main_gain.date==date)&main_gain.metric.isin(METRICS[1:])].set_index('metric')
        if len(d)==3:
            values=[d.loc[x,'relative_gain_pct'] for x in METRICS[1:]]
            results.append(f'On {date}, the signed mean reductions in velocity, attitude and body-rate error of Standard GRU/H26 relative to the MLP were {values[0]:.2f}%, {values[1]:.2f}% and {values[2]:.2f}%, respectively, at the nominal500ms horizon (positive denotes lower GRU error).')
    results=' '.join(results)+' Errors were computed within each flight, averaged equally across flights within each date, and summarized across three seeds. These are descriptive held-out-date results; no window-independence significance test was used.'
    discussion=('The test dates were selected and the model checkpoints, preprocessing gates and analysis rules were frozen before this evaluation. '
        'Sep8 had prior descriptive quality-audit exposure and Sep19 had prior file-inventory exposure; we do not claim first-ever access to the raw logs. '
        'The inherited v2 and v3 flight-admission gates differ, and results apply to eligible windows of these dates, not all recordings or the flight envelope. '
        'The MLP/GRU comparison changes capacity and architecture; controlled history comparisons supply separate evidence. '
        'Both Actual and Hold forecasts use the same logged-trajectory truth, without matched counterfactual Hold flights. '
        'Neither relative accuracy nor small seed SD establishes arbitrary-action causal fidelity, generalization to arbitrary winds or airframes, long-horizon stability, or closed-loop benefits. '
        'Sep8 and Sep19 are now used test data for this model version and cannot subsequently be treated as unopened independent tests after development on their results.')
    (out/'paper_results.txt').write_text(results+'\n');(out/'paper_discussion.txt').write_text(discussion+'\n')
    access=[json.loads(line) for line in (out/'test_access_log.jsonl').read_text().splitlines()]
    first={d:next(x['at_utc'] for x in access if x['date']==d) for d in ['Sep8','Sep19']}
    allmainpositive=bool((main_gain[main_gain.metric.isin(METRICS[1:])].relative_gain_pct>0).all())
    mainclaim='两日期500 ms的三项动力学指标均保留GRU相对MLP的均值优势。' if allmainpositive else '两日期500 ms未全面保留GRU相对MLP的优势，需要按日期和指标收窄论断。'
    findings=[]
    for date in ['Sep8','Sep19']:
        for comp in ['GRU_vs_MLP','H1_to_H26','H13_to_H26','Actual_vs_Hold']:
            d=g[(g.date==date)&(g.comparison==comp)&(g.step==25)&(g.group=='ALL')].set_index('metric')
            if len(d):
                txt=[]
                for metric in METRICS[1:]:
                    if metric not in d.index:continue
                    x=d.loc[metric]
                    txt.append(f'{metric}: {x.relative_gain_pct:.2f}%，seed {int(x.seeds_improved)}/3、flight {int(x.flights_improved)}/{int(x.n_flights)}支持比较条件')
                findings.append(date+' '+comp+'：'+'；'.join(txt)+'。')
    unsupported=g[(g.relative_gain_pct<=0)&g.metric.isin(METRICS[1:])]
    unsupported.to_csv(out/'nonpositive_dynamics_comparisons.csv',index=False)
    failure=json.loads((root/'artifacts/paper_independent_test_v1/inference_status.json').read_text())
    fails=[x for x in failure['outcomes'] if x['status']!='valid']
    report=['# Frozen held-out flight evaluation','## Registration and access',
        f'Registered {p["frozen_at_utc"]}; first read this run: {first}. HEAD `{p["git_commit"]}`. Prior access disclosures are retained; neither date is now unopened.',
        '\n'.join(summaries),'See evaluation_registration.md, protocol.json and append-only test_access_log.jsonl. No training, parameter tuning, checkpoint selection or ensemble.',
        '## Data quality',markdown(coverage[['date','session','flight_id','status','raw_samples','valid_samples','final_origins','reason']]),
        'Exact gates and exclusion counters are in dataset_quality_report.md. Sep8 inherits the original v2 flight gate; Sep19 inherits the expanded v3 gate. All admitted models share longest-history valid origins and native dt. Raw possible counts ignore gaps; quality counts apply original valid_core. No error-based filtering.',
        '## 500 ms main performance',markdown(main_table),markdown(main_gain[gaincols]),
        '## History',markdown(hist_table),markdown(hist_gain[gaincols]),
        'H1 is current-sample context with recurrent future transitions. H26 remains main even if a shorter context is better. No physical memory time constant or replacement of validation Mixed classification is inferred.',
        '## Future command information',markdown(control_table),markdown(control_gain[gaincols]),
        'Train quantiles are reused exactly. Group proportions need not be25/50/25. Empty groups are explicit unavailable rows/zero coverage. Horizon-specific groups change membership; full curves fixK25 membership.',
        markdown(groups[(groups.flight_id=='ALL')&(groups.step==25)]),
        '## All horizons and directions',markdown(g[gaincols]),
        'Positive gain favors comparison. Flight directions compare per-flight three-seed error means, not all windows. Relative gains use aggregated errors. No pooled-date mean; no window independent tests; sampleSD is not confidence/uncertainty.',
        '## Validation versus test',markdown(vc[(vc.validation_cohort=='ALL')&(vc.horizon_s==.5)]),
        '## Exceptions and engineering failures',f'{len(fails)} invalid model/date/condition runs. {len(bad)} nonpositive flight comparisons and {len(badseed)} nonpositive seed comparisons retained in CSVs.',
        markdown(unsupported[gaincols]),
        '## Results text',results,'## Discussion',discussion,
        '## Next step','Prediction experiments can be organized for writing with all admission/negative-result caveats and independent date tables. The next research stage would separately validate input-response direction, delay/amplitude, timing/actuator constraints and bounded-horizon controller robustness, then plan real-flight validation. None is executed here. No commit/push.']
    (out/'report.md').write_text('\n\n'.join(report)+'\n')
    zh=['# Paper Step7：冻结方案与留出日期评价','## 1. 冻结时间与数据访问',
        f'方案于{p["frozen_at_utc"]}冻结；本轮Sep8首次读取{first["Sep8"]}，Sep19首次读取{first["Sep19"]}。先通过validation/合成接口检查再开封。',
        'Sep8是主独立测试，Sep19是补充留出日期，均按预定方案执行且分开报告。旧manifest明确Sep8曾接受描述性质量审计；Sep19原inventory已有文件身份记录。因此不称为原始日志首次被任何人读取。没有在已知冻结实验manifest/对应输出目录中发现这15模型此前的测试性能评价记录，但不对未记录的外部访问作绝对保证。两日期本轮均已使用，不能再称尚未打开。',
        '## 2. 数据质量与实际覆盖','\n\n'.join(summaries),markdown(coverage[['date','session','flight_id','status','final_origins','reason']]),
        'Sep8沿用v2、Sep19沿用v3既有准入条件，具体差异在注册方案中预先写定，不是看测试误差后放宽或收紧。Sep19各文件夹是同一日期的session来源，不是多个独立日期。'
        '所有模型共享H26合法origin及50步future，短history仅取精确后缀；control、labels、dt一致。未因MLP无需历史或较短horizon增加窗口。',
        '实际历史跨度：',markdown(dur[dur.session=='ALL']),
        '## 3. 主模型独立测试',mainclaim,markdown(main_table),markdown(main_gain[gaincols]),
        '## 4. History结论是否保留',markdown(hist_table),markdown(hist_gain[gaincols]),
        '\n\n'.join(x for x in findings if 'H1_to_' in x or 'H13_to_' in x),
        'H1→H26的正值支持当前数据上历史上下文的预测价值，负值则在该条件下未得到支持。H13→H26须逐日期和指标判断，不能预设长history一定更好。H26不切换；旧validation Mixed分类不回写；有效上下文跨度不等于物理记忆常数。',
        '## 5. Future-control结论是否保留',markdown(control_table),markdown(control_gain[gaincols]),
        '正值支持实际未来命令的增量预测信息；负值需要明确限制。ALL不等于全部flight/window改善；high组也不自动代表收益最大。low/短horizon反例完整保留。控制组使用冻结训练阈值，仅是离线分组，不是在线可提前知道的工况分类器。',
        '## 6. 与validation的相同点和差异',markdown(vc[(vc.validation_cohort=='ALL')&(vc.horizon_s==.5)]),
        '比值>1表示测试绝对误差高于开发validation均值；相对优于MLP与绝对误差跨日期升高可以同时成立，不能互相替代。小seed SD不能证明跨日期泛化稳定。',
        '## 7. 不利结果、失败与边界',f'模型/日期/条件工程失败数：{len(fails)}。完整失败记录见artifact状态；不丢弃失败origins后报告成功子集。',
        '主要动力学的非正向聚合比较如下，均保留；负值表示对应验证结论在该测试条件下未得到支持：',markdown(unsupported[gaincols]),
        '全部flight和seed反例分别保存在paired_flight_exceptions.csv、paired_seed_exceptions.csv。数据准入被排除的flight不属于已评价样本，不能称全部原始航次都得到验证。',
        '## 8. 可用于论文Results的表述',results,
        '## 9. Discussion限制',discussion,
        '## 10. 是否具备预测实验整理写作条件',
        '已形成按日期的主性能、history、Actual/Hold主表及全时域补充结果，可整理预测实验章节；科学结论应逐日期、指标和工况限定，独立测试表优先，validation表保留为开发证据。若存在工程失败则对应单元保留缺项，不伪装完整成功。三seed SD仅描述初始化变化。',
        '## 11. 下一步问题，但不执行',
        '进入闭环仿真或实机验证前，需要独立验证输入响应的方向、时延与幅值，明确状态估计/坐标/控制分配和时间接口，考虑执行器约束及闭环误差累积，再定义有界时域和安全退出的控制验证方案。'
        '本轮没有开展MPC/RL、闭环控制、实机激励或任何训练。之后若利用这两日期改模型，必须如实记为已用于开发的数据，新的独立结论需要其他未用于开发的证据。'
        '无自动commit/push；建议提交：feat: add frozen held-out flight evaluation。']
    interpretation = (
        '主性能结论在两个留出日期得到支持：500 ms三项动力学指标均有3/3配对seed支持GRU；'
        '按每flight先平均三seed误差后，Sep8的6/6和Sep19的22/22架flight均支持GRU。'
        '这不是每个窗口都改善，也不是基于独立窗口假设的显著性结论。\n\n'
        'History结论需要收窄到日期、指标与时域：500 ms下H1→H26在Sep8速度/姿态/角速度改善6.97%/0.66%/2.90%，'
        'Sep19改善13.05%/11.03%/8.06%。Sep8姿态仅2/3seed、4/6flight同方向，不能称稳定全面改善。'
        'H13→H26在Sep8速度和角速度分别为−1.10%和−0.57%，即H13略好；Sep19三项边际改善为2.91%/0.07%/0.92%，'
        '姿态flight方向11胜11负。约240 ms上下文已获取大部分500 ms收益的叙述仍可保留，但更长历史持续稳定更好的说法不受支持。'
        '到1 s，Sep8 H1→H26速度和角速度分别−0.94%和−3.05%，对应结论在该条件下未得到支持；Sep19的速度/姿态均值收益也缩小至不足1%。'
        '不切换H26，不修改旧Mixed分类，不把历史跨度解释为物理时间常数。\n\n'
        '500 ms未来命令信息价值在两日期ALL和high组均得到支持；三项动力学均为3/3seed且每flight三seed均值方向全部正向。'
        'ALL速度/姿态/角速度改善为Sep8的19.36%/16.34%/9.15%及Sep19的18.24%/15.43%/17.30%；'
        'high组分别15.34%/19.90%/13.42%及22.03%/18.88%/18.44%。'
        '不能扩大为所有时域、组别和flight皆改善：Sep8 100 ms ALL角速度由Hold略好0.28%，high组也有0.57%的反向均值；'
        '500 ms low组两日期各有姿态flight反例，Sep8 middle速度也有局部反例。Hold无对应反事实实测轨迹，结论仍是增量预测信息。\n\n'
        '相对优势与绝对日期变化必须分开：H26在Sep19的500 ms速度、姿态误差较validation ALL分别高29.02%和21.65%，'
        '但角速度误差反而低11.40%。因此不能声称所有状态都同等跨日期稳定或全部退化。'
        '独立测试主表可作为论文性能叙述的主要依据，适用范围限定于原质量规则准入的两日期日志。')
    (out/'scientific_interpretation.md').write_text(interpretation+'\n')
    zh.insert(2,interpretation)
    (out/'review_report.md').write_text('\n\n'.join(zh)+'\n')
    with (out/'report.md').open('a') as stream:
        stream.write('\n## Focused interpretation\n\n'+interpretation+'\n')
    session=coverage.groupby(['date','session','status'],dropna=False).agg(flights=('flight_id','size'),raw_samples=('raw_samples','sum'),valid_samples=('valid_samples','sum'),origins=('final_origins','sum')).reset_index()
    session.to_csv(out/'test_session_coverage.csv',index=False)
