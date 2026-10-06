"""Deliver condition audit; mode/feature-overlap follow-up is explicitly post hoc."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_control_windows_v2 as base
from run_reference_condition_diagnostics_v12 import OUT,V8,V10,V11
from reference_condition_diagnostics_v12 import aggregate,covariates,weighted_terciles,bins,FEATURES
from reference_groupcv_v8 import load_train
from run_reference_generalization_v7 import verify_manifest
from report_reference_generalization_v7 import table


def main():
    verify_manifest(OUT/'protocol.json',['source_pins'])
    for source in [V8,V10,V11]:verify_manifest(source/'delivery_manifest.json',['artifacts','code'])
    x,y,w=load_train();rows=pd.read_csv(OUT/'waveform_per_window.csv');strata=pd.read_csv(OUT/'strata.csv');cv=pd.read_csv(OUT/'history_covariates.csv');threshold=pd.read_csv(OUT/'thresholds.csv')
    # Independent history, threshold, and every saved membership reconstruction.
    for axis,j in [('p',0),('q',1)]:
        f=covariates(x,j);old=cv[cv.axis==axis].sort_values('array_index');np.testing.assert_allclose(old[FEATURES],f,rtol=1e-12,atol=1e-12)
        for fold in range(5):
            info=json.loads((OUT/f'fold{fold}/manifest.json').read_text());fit=np.array(info['fit_indices'])
            for feature in FEATURES:
                cut=weighted_terciles(f.iloc[fit][feature].to_numpy(),w.iloc[fit].log_id.to_numpy())
                a=threshold[(threshold.fold==fold)&(threshold.axis==axis)&(threshold.condition==feature)].iloc[0]
                np.testing.assert_allclose(cut,[a.low,a.high],rtol=1e-12,atol=1e-12)
                g=strata[(strata.fold==fold)&(strata.axis==axis)&(strata.condition==feature)]
                np.testing.assert_array_equal(bins(f.iloc[g.array_index][feature],cut),g.stratum)
    assert len(rows[rows.cohort=='oof'])==407*3*2*2
    assert not rows[rows.cohort=='oof'].duplicated(['array_index','seed','axis','horizon_s']).any()
    # Truth origin modes are already constant over accepted support; no future condition used.
    modes=w[['array_index','mode']]
    oof=rows[rows.cohort=='oof'].merge(modes,on='array_index',validate='many_to_one')
    ass=strata[strata.cohort=='oof'].drop(columns='cohort');expanded=oof.merge(ass,on=['fold','axis','array_index','log_id'],validate='many_to_many')
    mode=expanded.groupby(['seed','log_id','axis','horizon_s','mode','condition','stratum']).apply(aggregate,include_groups=False).reset_index()
    mode.to_csv(OUT/'posthoc_within_mode_per_flight.csv',index=False)
    within=[]
    for (axis,h,md,condition),g in mode[(mode.condition!='mode')&mode.stratum.isin(['low','high'])].groupby(['axis','horizon_s','mode','condition']):
        v=g.pivot(index=['seed','log_id'],columns='stratum',values=['n','normalized_centered','mean_correlation'])
        if not all((k,s) in v for k in ['n','normalized_centered','mean_correlation'] for s in ['low','high']):continue
        v=v[(v['n']['low']>=8)&(v['n']['high']>=8)].dropna()
        if v.empty:continue
        ratios=np.log(v['normalized_centered']['high']/v['normalized_centered']['low']).groupby('log_id').mean()
        delta=(v['mean_correlation']['high']-v['mean_correlation']['low']).groupby('log_id').mean()
        within.append(dict(axis=axis,horizon_s=h,mode=int(md),condition=condition,paired_flights=len(ratios),ratio=float(np.exp(ratios.mean())),corr_difference=float(delta.mean()),
            ratio_high_worse_flights=int((ratios>0).sum()),ratio_low_worse_flights=int((ratios<0).sum())))
    within=pd.DataFrame(within);within.to_csv(OUT/'posthoc_within_mode_summary.csv',index=False)
    correlations=[]
    for log,g in cv[cv.axis=='q'].groupby('log_id'):
        corr=g[FEATURES].corr(method='spearman')
        for a in FEATURES:
            for b in FEATURES:
                if a<b:correlations.append(dict(log_id=log,feature_a=a,feature_b=b,spearman=corr.loc[a,b]))
    corr=pd.DataFrame(correlations);corr.to_csv(OUT/'posthoc_covariate_correlations.csv',index=False)
    corr.groupby(['feature_a','feature_b']).spearman.agg(['mean','min','max']).to_csv(OUT/'posthoc_covariate_correlation_summary.csv')
    summary=pd.read_csv(OUT/'conditions_summary.csv');screen=pd.read_csv(OUT/'repeatability_screen.csv');overall=pd.read_csv(OUT/'overall_summary.csv');support=pd.read_csv(OUT/'condition_support.csv')
    q=summary[(summary.cohort=='oof')&(summary.axis=='q')&(summary.horizon_s==1)&summary.condition.isin(['frequency_mean','frequency_std','actuator_step_rms','mode'])]
    wp=within[(within.axis=='q')&(within.horizon_s==1)&within.condition.isin(['frequency_mean','frequency_std','actuator_step_rms'])]
    print(wp.to_string(index=False),flush=True)
    print(corr.groupby(['feature_a','feature_b']).spearman.mean().sort_values().to_string(),flush=True)
    # Reproduce old p/q reference-effect summaries to check held-branch semantics.
    old=pd.read_csv(V11/'reference_sensitivity_summary.csv');old=old[(old.arm=='baseline')&old.axis.isin(['p','q'])]
    joined=overall[overall.cohort=='oof'].merge(old,on=['axis','horizon_s'],validate='one_to_one')
    err=float(abs(joined.reference_effect_rms-joined.effect_rms).max());assert err<1e-7
    log=Path('/tmp/reference-v12-tests.log').read_text();assert '9 passed' in log and 'failed' not in log;(OUT/'tests.txt').write_text(log)
    verification=json.loads((OUT/'verification.json').read_text());verification.update(focused_tests_passed=9,thresholds_and_assignments_recomputed=True,held_branch_summary_max_abs=err,
        posthoc_mode_overlap_only=True,no_new_training=True,development_evaluated=False)
    base.write(OUT/'verification.json',verification)
    # Report associations without promoting them to independent causal evidence.
    lines=['# 参考模型 v12：p/q 波形失配的历史条件分层','',
        '**q的失配存在跨飞行、跨种子重复的工况关联；p没有条件达到预先声明的幅度与一致性标准。本轮定位了研究方向，尚未产生更好的模型。**','',
        '## 范围和口径','',
        '保持407个训练窗口、5个整飞行留一折、3个冻结hold GRU种子；评价当前模型在同折fit与OOF上的完整0.5/1秒波形。没有新训练、样本删除或739个开发验证评价；sealed/reserved未访问。当前OOF飞行已反复用于开发，不是盲测。',
        '所有条件仅使用历史26点：体轴速度模长、扑频均值/标准差、本轴角速度标准差/绝对值均值、本轴姿态参考标准差/逐步变化RMS、4通道执行命令逐步变化RMS，以及起点模式。参考条件没有读取未来参考变化或飞机是否响应。',
        '每折只用4个拟合飞行，以飞行等权经验三分位确定low/mid/high阈值，原样应用于第5飞行。阈值重复时允许空层，报告样本支持。模式3=Mission，15=Stabilized。飞行轨迹和模式的差异没有通过这些单变量分层全部消除。',
        '以下波形RMSE覆盖整个前缀，不能与旧报告的单一时刻端点RMSE混用。归一化去均值误差为sqrt(mean(centered MSE)/mean(truth variance))；幅值比为sqrt(mean(predicted variance)/mean(truth variance))；相关系数先逐窗口计算，再按飞行/种子平均。真值波动用于评价归一化，不用于样本分层。','',
        '## 总体误差','',table(overall[['cohort','axis','horizon_s','rmse','normalized_centered','amplitude_ratio','mean_correlation','reference_utility']]),'',
        '1秒p在训练内已经只有约0.48的波形相关，OOF约0.39；因此不能只把p问题归于飞行间变化。q训练内相关约0.87、OOF约0.74，既有共同失配，也有泛化差距。幅值比与旧报告逐窗口标准差比口径不同，不应直接当成待乘的校准增益。','',
        '## 预声明的重复性筛查','',
        '对1秒high/low极端层比较，要求每个飞行两层各至少8窗口、至少4个可比较飞行；归一化误差几何平均比至少1.2倍或其倒数；同方向至少4飞行及全部3种子；相关系数在至少4飞行同步变差。以下列出全部16项，不只展示通过项。该筛查是多条件探索，无多重检验校正，不证明原因。','',
        table(screen[['axis','condition','paired_flights','high_low_normalized_ratio','high_low_rmse_ratio','mean_correlation_difference','consistent_flights','repeatable_screen']]),'',
        'q的低平均扑频、高扑频波动、高执行命令变化关联最明显；高/低归一化误差比分别约0.677、1.315、1.353，三个条件各有5/5飞行同向、3/3种子同向且相关性同步变化。另有历史|q|与俯仰参考逐步变化达到筛查标准，应视为并存线索，不能事后忽略。',
        'p的高历史角速度波动与较好相对波形匹配有关（归一化误差比约0.845、相关性增加约0.22，5飞行同向），但没有达到固定1.2倍标准；不能为了选出一个p候选而放宽门槛。高历史运动强度也会提高真值波动分母，因此绝对误差和相对误差必须一起看。','',
        'q关键条件的1秒数值：','',table(q[['condition','stratum','rmse','normalized_centered','amplitude_ratio','mean_correlation','reference_utility']]),'',
        '低扑频边界在各折约3.65–3.69Hz，高扑频边界约3.97–4.00Hz；这些是当前训练分布的分层阈值，不是物理临界频率或安全边界。0.5秒具有相同的低频、高波动、高执行命令变化较差趋势，完整数值见conditions_summary.csv。','',
        '## 参考信息是否仍有预测价值','',
        '用同一个模型分别输入日志参考和保持起点参考，定义参考预测收益=1−原参考RMSE/保持参考RMSE，两者都与同一实际记录比较。1秒OOF平均收益p约2.51%、q约11.39%。Stabilized中的q收益约5.33%，Mission约26.49%；模式的参考变化幅度、航段与状态不同，不能把差距当作模式本身的因果作用。',
        '此项只能说明计划参考在当前模型上的预测信息是否有用；保持参考的模拟没有对应真实反事实轨迹。它不能直接证明参考到飞机的控制增益、方向或时延正确，也不能把收益低解释成飞机没有舵效。','',
        '## 结果之后追加的模式与条件重合检查','',
        '以下是观察主结果后追加的描述性审计，明确不参与原筛查门槛。仍使用原分层边界；每模式、每飞行高低层各至少8窗口，不满足时不算独立重复。','',table(wp),'',
        '在Stabilized内部，平均扑频high/low归一化误差比约0.634、扑频波动high/low约1.410，仍分别有5个可比较飞行同向；因此不能仅由两种模式混合解释。执行命令变化在Stabilized仅3个飞行有足够支持；Mission下各条件只有1–2个，且执行命令变化方向反转，不能宣称各模式均成立。',
        '历史扑频均值与标准差的逐飞行Spearman平均约−0.853，二者高度重合；不能双重加权后把效果归给两个独立机制。扑频波动是日志频率估计值的统计量，未在本轮分离真实机械变化与估计/时间量化成分。',
        '条件之间的逐飞行Spearman相关完整保存在posthoc_covariate_correlations.csv。当前频率、执行命令、模式和先前运动同时变化，单变量关联不能区分气动变化、控制器/分配行为、输入表示或未测扰动；不能把三个关联当成三个独立原因。','',
        '## 下一步与交付','',
        '本轮没有训练新模型或更改基线。优先研究q的工况泛化，p保留为保护指标。扑频、模式、历史执行命令已经进入现有GRU，不能将结果解释成“模型忘了输入频率”，也不应立刻加入新的经验舵效或套用oracle增益。',
        '建议下一轮只检验一个因素：保持原GRU、全部有效窗口、1600更新、同初始化/抽样序列和5折×3种子，在q损失中对每个飞行的历史扑频三分位工况作受限均衡加权，比较训练内低频层和OOF是否同时改善。加权只依赖历史频率和训练分布，保留高频及弱响应窗口；所有状态总体误差、逐工况误差、参考预测收益都在共同样本上验收。先另行冻结权重上限与门槛，不在此报告结果后反复试权重。',
        '若训练内与OOF都没有改善，则不能再把该关联简单解释为采样不足，应检查q动态表示、时序与未建模条件；不能仅靠增加网络、放宽门槛或删除困难工况宣布成功。',
        '9项针对性测试通过；30组原模型重放、407个OOF窗口唯一性、所有历史特征/阈值/分层复算、误差分解恒等式和旧参考保持对照通过。源v8/v10/v11及新交付哈希校验通过。候选模型仍为原A1600；本轮没有证明必须补飞。','']
    (OUT/'report_zh.md').write_text('\n'.join(lines))
    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    for ax,feature in zip(axes,['frequency_mean','frequency_std','actuator_step_rms']):
        for h,marker in [(.5,'o'),(1.,'s')]:
            a=summary[(summary.cohort=='oof')&(summary.axis=='q')&(summary.horizon_s==h)&(summary.condition==feature)].set_index('stratum').loc[['low','mid','high']]
            ax.plot(range(3),a.normalized_centered,marker+'-',label=f'{h:g} s')
        ax.set_xticks(range(3),['Low','Middle','High']);ax.set(title=feature,ylabel='Normalized centered q error');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Training-flight OOF conditions; associations, not causal effects');fig.tight_layout();fig.savefig(OUT/'condition_comparison.png',dpi=150);plt.close(fig)
    paths=list((base.ROOT/'scripts').glob('*condition_diagnostics_v12.py'))+[base.ROOT/'tests/test_reference_condition_diagnostics_v12.py']
    base.write(OUT/'delivery_manifest.json',dict(status='complete',code={str(p.relative_to(base.ROOT)):base.sha(p) for p in paths},artifacts={str(p.relative_to(base.ROOT)):base.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'},
        focused_tests_passed=9,model_promoted=False,development_evaluated=False,sealed_test_accessed=False))
    verify_manifest(OUT/'delivery_manifest.json',['code','artifacts']);print('Delivery verified',flush=True)

if __name__=='__main__':main()
