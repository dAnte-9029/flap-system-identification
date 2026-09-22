"""Descriptive, equal-flight comparison of the frozen MLP and GRU families."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METRICS = {'position_rmse_m': 'Position RMSE [m]', 'velocity_rmse_m_s': 'Velocity RMSE [m/s]',
           'attitude_error_deg': 'Attitude geodesic RMS [deg]', 'body_rate_rmse_rad_s': 'Body-rate RMSE [rad/s]'}
B0, MLP, GRU = 'B0_ConstantVelocity', 'B1_MLP', 'B2_StandardGRU'
LABELS = {B0: 'B0 Kinematic', MLP: 'B1 MLP', GRU: 'B2 Standard GRU / H26'}
SEEDS = {17, 23, 42}
HORIZONS = {.1, .2, .5, 1.}
COHORTS = {'ALL', 'Sep7', 'Sep17'}


def relative_gain(mlp, gru):
    return 100 * (mlp - gru) / mlp if mlp != 0 else np.nan


def aggregate_seeds(summary):
    """Input rows already contain equal-flight macro errors, never raw windows."""
    rows = []
    assert set(summary.model) == {B0, MLP, GRU}
    for model, data in summary.groupby('model'):
        assert set(zip(data.cohort, data.horizon_s)) == {(c,h) for c in COHORTS for h in HORIZONS}
        for (cohort, horizon), group in data.groupby(['cohort', 'horizon_s']):
            deterministic = model == B0
            assert (len(group) == 1 and group.seed.isna().all()) if deterministic else (len(group) == 3 and set(group.seed) == SEEDS)
            assert group.n_flights.nunique() == group.n_windows.nunique() == 1
            for metric in METRICS:
                values = group[metric].to_numpy()
                assert np.isfinite(values).all()
                rows.append(dict(model=model, cohort=cohort, horizon_s=horizon, metric=metric,
                    mean=values.mean(), std=np.nan if deterministic else values.std(ddof=1),
                    min=values.min(), max=values.max(), n_seeds=0 if deterministic else 3,
                    deterministic=deterministic, n_flights=int(group.n_flights.iloc[0]), n_origins=int(group.n_windows.iloc[0])))
    return pd.DataFrame(rows)


def paired_seeds(summary):
    keys = ['seed', 'cohort', 'horizon_s']
    a = summary[summary.model == MLP].set_index(keys).sort_index()
    b = summary[summary.model == GRU].set_index(keys).sort_index()
    assert a.index.equals(b.index) and a.index.is_unique
    np.testing.assert_array_equal(a[['n_flights','n_windows']], b[['n_flights','n_windows']])
    rows=[]
    for idx in a.index:
        for metric in METRICS:
            av,bv=a.loc[idx,metric],b.loc[idx,metric]
            rows.append(dict(zip(keys,idx),metric=metric,mlp_error=av,gru_error=bv,absolute_gain=av-bv,
                             relative_gain_pct=relative_gain(av,bv),gru_better=bool(av>bv)))
    return pd.DataFrame(rows)


def paired_flights(per):
    keys=['flight_id','cohort','horizon_s']
    learned=per[per.model.isin([MLP,GRU])]
    for _,g in learned.groupby(['model']+keys):
        assert len(g)==3 and set(g.seed)==SEEDS and g.n_windows.nunique()==1
    a=learned[learned.model==MLP].groupby(keys)[list(METRICS)+['n_windows']].mean().sort_index()
    b=learned[learned.model==GRU].groupby(keys)[list(METRICS)+['n_windows']].mean().sort_index()
    assert a.index.equals(b.index)
    np.testing.assert_array_equal(a.n_windows,b.n_windows)
    rows=[]
    for idx in a.index:
        for metric in METRICS:
            av,bv=a.loc[idx,metric],b.loc[idx,metric]
            rows.append(dict(zip(keys,idx),metric=metric,n_origins=int(a.loc[idx,'n_windows']),
                mlp_error=av,gru_error=bv,absolute_gain=av-bv,relative_gain_pct=relative_gain(av,bv),gru_better=bool(av>bv)))
    return pd.DataFrame(rows)


def gains_table(aggregate, seeds, flights):
    rows=[]
    for (cohort,horizon,metric),g in aggregate.groupby(['cohort','horizon_s','metric']):
        av=float(g[g.model==MLP]['mean'].iloc[0]);bv=float(g[g.model==GRU]['mean'].iloc[0])
        s=seeds[(seeds.cohort==cohort)&(seeds.horizon_s==horizon)&(seeds.metric==metric)]
        f=flights[(flights.horizon_s==horizon)&(flights.metric==metric)]
        if cohort!='ALL':f=f[f.cohort==cohort]
        rows.append(dict(cohort=cohort,horizon_s=horizon,metric=metric,mlp_mean=av,gru_mean=bv,
            absolute_gain=av-bv,relative_gain_pct=relative_gain(av,bv),seeds_gru_better=int((s.absolute_gain>0).sum()),
            seeds_mlp_better=int((s.absolute_gain<0).sum()),seeds_tied=int((s.absolute_gain==0).sum()),
            flights_gru_better=int((f.absolute_gain>0).sum()),flights_mlp_better=int((f.absolute_gain<0).sum()),
            flights_tied=int((f.absolute_gain==0).sum()),n_flights=len(f)))
    return pd.DataFrame(rows)


def markdown(df):
    def fmt(x):
        if isinstance(x,(float,np.floating)):return '' if pd.isna(x) else f'{x:.5g}'
        return str(x).replace('|','/')
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))


def formatted_table(aggregate, cohort=None, horizon=None):
    d=aggregate
    if cohort is not None:d=d[d.cohort==cohort]
    if horizon is not None:d=d[d.horizon_s==horizon]
    rows=[]
    for (c,h,m),g in d.groupby(['cohort','horizon_s','model']):
        row={'Cohort':c,'Horizon [s]':h,'Model':LABELS[m]}
        for metric,label in METRICS.items():
            r=g[g.metric==metric].iloc[0]
            sd=f'{r["std"]:.2e}' if 0<r['std']<.00005 else f'{r["std"]:.4f}'
            row[label]=f'{r["mean"]:.4f}' if m==B0 else f'{r["mean"]:.4f} ± {sd}'
        rows.append(row)
    return pd.DataFrame(rows)


def figures(out,aggregate,flights):
    plt.rcParams.update({'font.size':10,'savefig.dpi':180,'pdf.fonttype':42})
    colors={MLP:'#d97922',GRU:'#1671ae'}
    for metric,label in METRICS.items():
        fig,axes=plt.subplots(1,3,figsize=(13,3.5),sharex=True)
        for ax,cohort in zip(axes,['ALL','Sep7','Sep17']):
            for model in [MLP,GRU]:
                g=aggregate[(aggregate.model==model)&(aggregate.cohort==cohort)&(aggregate.metric==metric)].sort_values('horizon_s')
                x=g.horizon_s.to_numpy();y=g['mean'].to_numpy();sd=g['std'].to_numpy()
                ax.plot(x,y,'o-',label=LABELS[model],color=colors[model]);ax.fill_between(x,y-sd,y+sd,alpha=.18,color=colors[model])
            ax.set(title=cohort,xlabel='Nominal horizon [s]',xticks=[.1,.2,.5,1.]);ax.grid(alpha=.2)
        axes[0].set_ylabel(label);axes[-1].legend(fontsize=8)
        fig.suptitle('Equal-flight macro error; mean ± 1 sample SD across 3 seeds',fontsize=11)
        fig.tight_layout()
        for suffix in ['png','pdf']:fig.savefig(out/f'error_vs_horizon_{metric}.{suffix}')
        plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,7))
    ids=flights[['flight_id','cohort']].drop_duplicates().sort_values(['cohort','flight_id'])
    ids.to_csv(out/'figure_flight_key.csv',index=False)
    for ax,(metric,label) in zip(axes.flat,METRICS.items()):
        d=flights[(flights.horizon_s==.5)&(flights.metric==metric)].set_index('flight_id').loc[ids.flight_id]
        ax.bar(np.arange(len(d)),d.absolute_gain,color=['#1671ae' if c=='Sep7' else '#a457a4' for c in d.cohort])
        ax.axhline(0,color='black',lw=1);ax.set_xticks(np.arange(len(d)),[f'{c}-{i+1}' for i,c in enumerate(d.cohort)],rotation=60,fontsize=8)
        ax.set_ylabel('MLP − GRU: '+label);ax.grid(axis='y',alpha=.2)
    fig.suptitle('500 ms paired flights; each flight error averaged across 3 seeds\nPositive = GRU lower error; all 17 flights shown')
    fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(out/f'paired_flights_500ms.{suffix}')
    plt.close(fig)


def generate(out,protocol):
    out=Path(out);summary=pd.read_csv(out/'per_seed_summary.csv');per=pd.read_csv(out/'per_flight_wide.csv')
    # Verify macro errors from flight rows; do not trust a pooled summary.
    for row in summary.itertuples():
        d=per[per.model==row.model]
        if row.model!=B0:d=d[d.seed==row.seed]
        d=d[d.horizon_s==row.horizon_s]
        if row.cohort!='ALL':d=d[d.cohort==row.cohort]
        assert len(d)==row.n_flights and d.n_windows.sum()==row.n_windows
        for metric in METRICS:np.testing.assert_allclose(d[metric].mean(),getattr(row,metric),rtol=1e-12,atol=1e-12)
    agg=aggregate_seeds(summary);paired=paired_seeds(summary);flight=paired_flights(per);gains=gains_table(agg,paired,flight)
    for name,df in [('multiseed_summary',agg),('paired_seed_differences',paired),('paired_flight_differences',flight),('relative_improvement',gains)]:df.to_csv(out/f'{name}.csv',index=False)
    main=formatted_table(agg,'ALL',.5);main.to_csv(out/'paper_table_500ms.csv',index=False)
    full=formatted_table(agg);full.to_csv(out/'paper_table_all_horizons.csv',index=False)
    (out/'paper_tables.md').write_text(markdown(full)+'\n')
    figures(out,agg,flight)
    primary=gains[(gains.cohort=='ALL')&(gains.horizon_s==.5)].set_index('metric')
    dyn=list(METRICS)[1:];percent=[primary.loc[m,'relative_gain_pct'] for m in dyn]
    results=('Under the frozen matched-budget protocol, the Standard GRU/H26 and memoryless MLP were evaluated using three training seeds '
        '(17, 23, 42) on the same 2,582 validation origins from 17 flights. At the nominal 500 ms horizon, the GRU reduced '
        f'the mean velocity, attitude, and body-rate errors by {percent[0]:.2f}%, {percent[1]:.2f}%, and {percent[2]:.2f}%, respectively, '
        'relative to the MLP. Errors were first computed within each flight, averaged equally across flights, and then summarized across seeds. '
        'Reported dispersions are sample standard deviations across three seeds. These are descriptive validation results for the frozen model families.')
    limits=('The MLP and GRU differ in architecture and parameter count (5,703 versus 21,383); this comparison therefore cannot isolate history '
        'as the cause of all performance differences. The separate H1–H26 experiment provides the controlled history-length evidence. '
        'Matching seed identifiers pairs training repetitions, not initial weights or identical random trajectories. Three seeds offer limited '
        'coverage of optimization variability, and overlapping windows are not independent trials. All models replay logged future commands; '
        'these validation errors do not establish causal responses to arbitrary actions, independent test generalization, long-horizon simulation '
        'validity, or closed-loop control performance. No sealed or reserved test was accessed.')
    (out/'paper_results.txt').write_text(results+'\n');(out/'paper_discussion.txt').write_text(limits+'\n')
    direction_cols=['cohort','horizon_s','metric','relative_gain_pct','seeds_gru_better','seeds_mlp_better','flights_gru_better','flights_mlp_better','n_flights']
    bad=flight[flight.absolute_gain<=0].sort_values(['horizon_s','metric','cohort','flight_id'])
    bad.to_csv(out/'flight_exceptions.csv',index=False)
    seed_bad=paired[paired.absolute_gain<=0];seed_bad.to_csv(out/'seed_exceptions.csv',index=False)
    training=pd.read_csv(out/'training_runs.csv')
    traincols=['model','seed','status','final_epoch','final_train_loss','training_time_s','checkpoint_sha256']
    methods=('B1 MemorylessTrajectoryModel: current state/control input16 → ReLU64 → ReLU64 →7; 5,703 trainable parameters; '
        'original output zero initialization, derivative scales/clipping and physical integrator. It ignores history but uses H26-eligible origins. '
        'B2 Standard GRU64/H26:21,383 parameters, unchanged. B0:constant inertial velocity and body rate with the original integrator. '
        'All learned models use40 epochs AdamW lr3e-4 plus25 epochs fresh AdamW lr5e-4; batch256, weight decay1e-5, clip5, '
        '111 updates/epoch,7,215 total; full50-step rollout, frozen lag2 increment supervision, continuation frequency MSE0.2. '
        'Last epoch65 only. Base seeds17/23/42; continuation29/35/54. No validation selection. '
        'Native dt;5/10/25/50 endpoints; vector RMSE and quaternion geodesic RMS. ' +
        'The protocol aggregation field defines this three-seed analysis; metrics.uncertainty is inherited Step1 single-pilot metadata, clarified before evaluation in metric_metadata_clarification.json. ' +
        'The old factory hardcodes seed17; the new helper calls the same class with the requested seed and preserves constructor draw order. '
        'Same-seed repeatability, different-seed random layers and seed17 factory parity are tested. Frozen six normalization buffers are loaded once, never fitted.')
    report=['# Frozen three-seed MLP comparison', '\n## Goal and scope',
        'Overall Prediction Performance: assess repeatability of the frozen model-family gap, not a one-factor history ablation.',
        '\n## Frozen methods and reproducibility',methods,
        f'Source HEAD: `{protocol["git_commit"]}`. Branch: `{protocol["branch"]}`. Exact paths, versions, seeds, hashes and settings: protocol.json and per-run runtime/config.json.',
        '\n## Data and reuse',
        '41 training flights /28,293 origins;17 validation flights /2,582 origins (Sep7:9 flights/1,481; Sep17:8 flights/1,101). '
        'Actual commands only. Identical cached labels, control tapes, dt, origins and normalization for all runs. '
        'Only B1 seeds23/42 trained from fresh initialization. B1 seed17, all H26 seeds and deterministic B0 reused; hashes rechecked. '
        'Original B1 seed17 full prediction metrics reproduce stored results; a fixed128-origin GPU replay passes predeclared tolerance. '
        'Original logs for B1 seed17 did not record final validation loss; it is left blank rather than inventing a value. '
        'New final validation losses are post-training reporting only. Source metrics include endpoint increments but they are algebraically redundant and not separate evidence.',
        '\n## 500 ms ALL',markdown(main),
        '\n## Complete cohort/horizon results',markdown(full),
        '\n## Paired directions and gains',
        'Absolute gain = MLP − GRU; relative gain =100×(MLP − GRU)/MLP on reported macro errors; zero denominator undefined. '
        'Each flight direction compares its3-seed mean errors. Positive favors GRU. No window-independence tests.',markdown(gains[direction_cols]),
        '\n## Exceptions',f'{len(seed_bad)} nonpositive seed/cohort/horizon/metric comparisons and {len(bad)} nonpositive flight/horizon/metric comparisons are retained without filtering.',
        markdown(bad[bad.horizon_s==.5][['flight_id','cohort','metric','absolute_gain','relative_gain_pct']]) if len(bad[bad.horizon_s==.5]) else 'No 500 ms flight exceptions.',
        'All exceptions, including other horizons: flight_exceptions.csv and seed_exceptions.csv. Full seed comparisons: paired_seed_differences.csv.',
        markdown(bad[['flight_id','cohort','horizon_s','metric','absolute_gain','relative_gain_pct']]),
        'Endpoint error need not grow monotonically: the MLP velocity error decreases from100 to200 ms, then rises at500 ms and1 s. This pattern is retained; initial alignment, prefix and native-timing checks pass. The current experiment does not isolate its physical or architectural cause.',
        '\n## Training completion and engineering checks',markdown(training[traincols]),
        'Both new runs must complete65 epochs without retries. Tests cover initialization, original baseline/increment behavior and descriptive statistics. '
        'Sanity checks cover finite losses/gradients/weights/predictions, normalization immutability, exact origin pairing, native timing, unit quaternions, '
        'future-label poisoning, prediction prefix consistency, fixed updates and final-epoch selection. Full details: tests.json, preflight.json, sanity_checks.json and completion.json.',
        '\n## Paper Results',results,'\n## Limitations / Discussion',limits,
        '\n## Scientific interpretation and next step',
        'Use the directions and magnitudes above rather than imposing a post-hoc success threshold. The validation Overall Prediction Performance table is now complete for B0/MLP/GRU. '
        'This does not complete independent-test evidence. Next candidate:freeze the final evaluation protocol and prepare independent testing, without opening test data in this task. '
        'H26 remains main model; prior history classification remains Mixed; future-control conclusions remain unchanged. '
        'Unrelated working-tree/registry differences are recorded, not repaired. No automatic commit/push.',
        '\n## Figures','Error-vs-horizon figures show all four metrics and ALL/Sep7/Sep17. Paired-flight500ms figure includes all17 flights; flight key is saved. '
        'Shading denotes ±1 sample SD across3seeds, not confidence intervals or model uncertainty. B0 has no artificial seedSD.',
        '\n## Exact split', 'Training flights:\n\n'+'\n'.join('- '+x for x in protocol['train_flights']),
        '\nValidation flights:\n\n'+'\n'.join('- '+x for x in protocol['validation_flights'])]
    (out/'report.md').write_text('\n\n'.join(report)+'\n')
    positive=all(primary.loc[m,'relative_gain_pct']>0 for m in dyn)
    directions=[]
    trends=[]
    for m in dyn:
        r=primary.loc[m]
        directions.append(f'{METRICS[m]}：500 ms ALL有{int(r.seeds_gru_better)}/3个seed、{int(r.flights_gru_better)}/{int(r.n_flights)}架flight支持GRU，'
                          f'{int(r.flights_mlp_better)}架支持MLP、{int(r.flights_tied)}架持平。')
        c=gains[(gains.horizon_s==.5)&(gains.metric==m)].set_index('cohort')
        directions.append(f'Sep7改善{c.loc["Sep7","relative_gain_pct"]:.2f}%，Sep17改善{c.loc["Sep17","relative_gain_pct"]:.2f}%；'
                          '正值为GRU更低，负值为MLP更低。')
        h=gains[(gains.cohort=='ALL')&(gains.metric==m)].sort_values('horizon_s')
        monotone=np.all(np.diff(h.relative_gain_pct)>=0)
        trends.append(f'{METRICS[m]}相对改善从100 ms的{h.relative_gain_pct.iloc[0]:.2f}%变化至1 s的{h.relative_gain_pct.iloc[-1]:.2f}%；'
                      +('四个时域单调不减。' if monotone else '四个时域并非单调增加，不将其概括为时域越长优势必然越大。'))
    zh=['# Paper Step6：冻结MLP三seed比较',
        '## 1. 论文问题与核心结论',
        ('补齐三seed后，500 ms三项主要动力学指标的均值仍支持Standard GRU/H26优于当前MLP。' if positive else '补齐三seed后，500 ms主要指标呈混合方向，不能概括为GRU全面优于MLP。')+
        '这是冻结模型家族与matched-budget协议下的验证集比较；不预设所有seed、flight和时域都胜出。',
        '## 2. 实验设置、复用与覆盖',
        '只新增MLP seed23与42，各自随机初始化；seed17和H26三个seed直接复用。B0是单个确定性参照，没有复制seed或人为标准差。'
        '41架训练flight/28,293 origins；17架验证flight/2,582 origins；Sep7为9架/1,481，Sep17为8架/1,101。所有模型origins、标签、Actual控制和native dt一致。'
        'MLP为5,703参数、GRU为21,383参数。40+25 epochs、两阶段重建AdamW、7215次更新、最后epoch65，不按验证误差选模。'
        '归一化沿用原B1六个冻结buffer，未重新fit。内部硬编码seed17的旧factory未修改，新入口保留原类及构造顺序，明确设置23/35、42/54。',
        '## 3. 500 ms主结果',markdown(main),
        f'GRU相对MLP的velocity、attitude、body-rate均值改善分别为{percent[0]:.2f}%、{percent[1]:.2f}%、{percent[2]:.2f}%。'
        '计算先flight内RMSE，再flight等权，再3seed均值及sample SD(ddof=1)；百分比由聚合误差计算，未平均窗口百分比。',
        '## 4. seed与flight方向','\n\n'.join(directions),markdown(gains[gains.horizon_s==.5][direction_cols]),
        'flight方向先对每架flight的三个seed误差取均值，再计算MLP−GRU。正值支持GRU。匹配seed编号只表示配对训练重复，不表示不同架构有相同初始权重或随机轨迹。',
        '500 ms反向或持平flight：',markdown(bad[bad.horizon_s==.5][['flight_id','cohort','metric','absolute_gain','relative_gain_pct']]) if len(bad[bad.horizon_s==.5]) else '无。',
        '## 5. Sep7/Sep17及100 ms至1 s','\n\n'.join(trends),markdown(gains[direction_cols]),
        '完整绝对误差与跨seed离散程度见下表；不能将总体改善替代每个cohort、时域和flight的具体方向。',markdown(full),
        '全部时域的flight例外如下；未删除。配对seed例外数量为'+str(len(seed_bad))+'。',
        markdown(bad[['flight_id','cohort','horizon_s','metric','absolute_gain','relative_gain_pct']]),
        'MLP速度端点误差从100 ms到200 ms下降，随后升高，不能把其误差描述为随时域单调增长。初始状态、prefix与native时间对齐检查通过；本轮不将这种形状归因为已被识别的物理机制。',
        '## 6. 可采用的Results表述',results,
        '## 7. Discussion限制',limits,
        '容量和架构同时不同，因此不能用本实验单独证明“历史导致全部改善”，也不能推广为所有GRU普遍优于所有MLP。'
        '历史的单因素证据由既有H1→H26同结构消融承担，正式Mixed分类不变。这里不做window独立性显著性检验，不把3seed SD称为置信区间、模型不确定性或独立测试证据。',
        '## 8. 训练与工程核验',markdown(training[traincols]),
        '仅有两次新增正式训练，无失败seed静默重跑。旧checkpoint、结果与源文件hash复核；有限性、future-label poisoning、prefix、四元数、步数和聚合测试见tests.json/sanity_checks.json。'
        '旧seed17没有历史final validation loss记录，保留空值；新seed最终validation loss仅供报告。',
        '## 9. 论文表图',
        '主表：paper_table_500ms.csv；补表：paper_table_all_horizons.csv、paired_seed_differences.csv、paired_flight_differences.csv。'
        '主图建议velocity/attitude/body-rate随horizon曲线，以及500 ms全部flight配对差值图。position同样保留，不筛选“好看”指标或seed。'
        '本轮已补齐B0/MLP/GRU的Overall Prediction Performance验证集主表，独立test尚未执行。',
        '## 10. 下一步建议与停止边界',
        '可进入最终评价方案冻结与独立测试准备：明确主表、统计单位、checkpoint及报告范围，再决定开启独立测试。这里只提出建议，不执行。'
        'H26主模型不变、history Mixed不变、future-control结论不变；sealed Sep8/reserved Sep19的数据、预测、结果均未打开。'
        '没有启动其他模型、消融、控制或RL实验，没有自动commit/push。']
    (out/'review_report.md').write_text('\n\n'.join(zh)+'\n')
