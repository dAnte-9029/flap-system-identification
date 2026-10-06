"""Verify derived coverage and publish the bounded v26 diagnostic."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import train_control_windows_v2 as base
from audit_controller_support_v26 import OUT,V12,V23,V24,V25,verify_parent


def table(frame):
    def fmt(v):return f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v)
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])


def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    for p,h in protocol['code'].items():assert base.sha(base.ROOT/p)==h
    for p,h in protocol['parents'].items():assert base.sha(base.ROOT/p/'delivery_manifest.json')==h
    d=pd.read_csv(OUT/'support_per_window.csv');o=pd.read_csv(OUT/'q_per_window.csv')
    marginal=pd.read_csv(OUT/'marginal_distribution.csv');summary=pd.read_csv(OUT/'history_summaries.csv')
    q=pd.read_csv(OUT/'q_strata_summary.csv');pf=pd.read_csv(OUT/'q_strata_per_flight.csv')
    w=pd.read_csv(V23/'windows.csv');logs=w.log_id.to_numpy()
    with np.load(V24/'history_controller.npz') as z:h=z['history_controller'].astype(float)
    query_checks=[]
    # Recompute every query nearest-neighbor distance independently from saved fit statistics.
    for fold in range(5):
        params=json.loads((OUT/f'fold{fold}/support_parameters.json').read_text());fit=np.array(params['fit_indices']);ids=np.array(params['query_indices'])
        for block,channels in [('integral',[0,1,2]),('compression',[3,4,5]),('age',[6,7]),('all',list(range(8)))]:
            x=h[:,:,channels];x=np.concatenate([x.mean(1),x.std(1),x[:,-1]],axis=1)
            p=params['blocks'][block];weights=np.array([1/sum(logs[fit]==log) for log in logs[fit]]);weights/=weights.sum()
            mean=np.average(x[fit],axis=0,weights=weights)
            scale=np.maximum(np.sqrt(np.average((x[fit]-mean)**2,axis=0,weights=weights)),1e-6)
            np.testing.assert_allclose(mean,p['mean']);np.testing.assert_allclose(scale,p['scale'])
            xf=(x[fit]-mean)/scale;xq=(x[ids]-mean)/scale
            expect=np.array([np.linalg.norm(xf-value,axis=1).min()/np.sqrt(x.shape[1]) for value in xq])
            actual=d[(d.fold==fold)&(d.block==block)&(d.cohort=='oof')].set_index('array_index').loc[ids]
            np.testing.assert_allclose(expect,actual.distance,rtol=1e-12,atol=1e-12)
            cal=d[(d.fold==fold)&(d.block==block)&(d.cohort=='fit')].set_index('array_index').loc[fit].distance.to_numpy()
            order=np.argsort(cal,kind='stable');cumulative=np.cumsum(weights[order]);cumulative[-1]=1.
            threshold=cal[order[np.flatnonzero(cumulative>=.95)[0]]]
            np.testing.assert_allclose(threshold,p['threshold'])
            np.testing.assert_array_equal(expect>threshold,actual.weak_support)
            query_checks.append(dict(fold=fold,block=block,query_distance_and_fit_statistics=True))
    strata=pd.read_csv(OUT/'strata.csv');modes=strata[strata.condition=='mode']
    np.testing.assert_array_equal(modes.stratum.astype(int),w.set_index('array_index').loc[modes.array_index]['mode'])
    oo=d[d.cohort=='oof'];coverage=oo.groupby(['block','log_id']).weak_support.agg(['sum','size']).reset_index()
    coverage.to_csv(OUT/'support_coverage_per_flight.csv',index=False)
    paired=[]
    for block,g in coverage.groupby('block'):
        paired.append(dict(block=block,weak_windows=int(g['sum'].sum()),total_windows=int(g['size'].sum()),
                           weak_flights=int((g['sum']>0).sum()),both_strata_at_least8_flights=int(((g['sum']>=8)&(g['size']-g['sum']>=8)).sum())))
    paired=pd.DataFrame(paired);paired.to_csv(OUT/'support_comparability.csv',index=False)
    # Signed contributions use original full-population weights, not subset renormalization.
    qo=o[o.cohort=='oof'].copy();qo['weight']=1/(15*qo.groupby(['seed','log_id']).array_index.transform('size'))
    contribution=[]
    for block in oo.block.unique():
        a=qo.merge(oo[oo.block==block][['array_index','weak_support']],on='array_index',validate='many_to_one')
        for weak,g in a.groupby('weak_support'):
            contribution.append(dict(block=block,weak_support=weak,weighted_delta_mse=float((g.weight*g.delta_mse).sum())))
    contribution=pd.DataFrame(contribution);contribution.to_csv(OUT/'signed_error_contributions.csv',index=False)
    total=float(q[(q.cohort=='oof')&(q.condition=='all')].delta_mse.iloc[0])
    np.testing.assert_allclose(contribution.groupby('block').weighted_delta_mse.sum(),total,rtol=1e-12)
    assoc=pd.read_csv(OUT/'support_error_association.csv');assoc=assoc[assoc.cohort=='oof']
    assoc=assoc.groupby(['block','log_id']).spearman.mean().reset_index();assoc.to_csv(OUT/'association_per_flight.csv',index=False)
    eta=pd.read_csv(OUT/'fit_feature_structure.csv').groupby(['feature','stat']).fit_flight_eta2.mean().unstack().reset_index()
    age=summary.groupby('log_id')[['integral_publication_age_s_mean','integral_publication_age_s_std']].median().mul(1000).reset_index()
    age.columns=['log_id','median_window_age_mean_ms','median_window_age_std_ms'];age.to_csv(OUT/'age_per_flight.csv',index=False)
    frequency=q[(q.cohort=='oof')&q.condition.isin(['frequency_mean','frequency_std','mode'])][['condition','stratum','zero_rmse','observed_rmse','improvement_percent']]
    frequency=frequency.merge(pd.read_csv(OUT/'strata_coverage.csv'),on=['condition','stratum'],validate='one_to_one')
    flight=pf[(pf.cohort=='oof')&(pf.condition=='all')].groupby('log_id')[['zero_rmse','observed_rmse','delta_mse','delta_bias_mse','delta_centered_mse']].mean().reset_index()
    flight['regression_percent']=100*(flight.observed_rmse/flight.zero_rmse-1);flight.to_csv(OUT/'q_per_flight.csv',index=False)
    decomp=q[q.condition=='all'][['cohort','delta_mse','delta_bias_mse','delta_centered_mse','delta_early_mse','delta_late_mse']]
    near=q[(q.cohort=='oof')&q.condition.str.startswith('support_')][['condition','stratum','zero_rmse','observed_rmse','improvement_percent']]
    checks=json.loads((OUT/'verification.json').read_text());checks.update(independent_query_checks=query_checks,
        original_mode_labels=True,signed_contributions_reconcile=True,focused_tests_passed=3)
    base.write(OUT/'verification.json',checks)
    design=dict(status='prepared_not_started',question='Does removing publication-age channels prevent loss of generalization while retaining integral/compression history?',
        evidence='Age summary support differs strongly forlog17, but coveredlog19 also regresses. Association is insufficient; controlled ablation required before causal attribution.',
        candidate='Samev25architecture33input/41606parameters; use first6historicalcontroller channels, set normalized last2publication-age channels tozero. No futurecontroller features.',
        controls='Frozenv25 zero_history_control andobserved_history_control. No retraining controls if exact shared weights/draws/environment reproduce. Candidate15runs5foldx3seed1600steps, sameoriginal407trainwindows,normalization,init,stored32batchdraws,AdamW/full1sloss.',
        primary='Againstzero arm: qfull1sOOF>=5% improvement, all3seedpositive,>=4/5flightpositive; same18protections<=5%regression andoriginal12absoluteactivegates. Full8arm comparison secondary.',
        limit='No tuned thresholds, sample removal, developmentvalidation/test access or automaticcontrolpromotion; a failed candidate does not justify deletinglog17.',training_started=False)
    base.write(OUT/'next_ablation_protocol.json',design)
    base.write(OUT/'decision.json',dict(status='diagnostic_complete',windows_retained=407,models_changed=0,model_promoted=False,
        finding='Age history carries flight-dependent support shift, most clearlylog17; neither support distance nor mode/frequency uniquely explains q regression. Centered waveform error increases while mean bias slightly improves.',
        next='Prepared same-budget six-physical-history-channel ablation with age normalized tozero; not started.',
        baseline_retained='A1600',validation_accessed=False,sealed_test_accessed=False))
    report=f'''# 历史控制器特征覆盖与q退化诊断 v26

诊断完成，原407个窗口全部保留；30个v25模型及其预测冻结，本轮没有重训或模型干预。开发验证、sealed/reserved未访问，保留A1600，不晋升控制能力。

**发布年龄的跨飞行差异值得单独消融，但尚不能认定它是退化根因。**log_17的97个窗口在年龄组合上全部低支持，积分发布年龄在每个26点窗口内的标准差中位约7.24ms，其余飞行约1.22ms。另一方面，整体组合覆盖正常的log_19仍退化4.82%，覆盖距离与q退化的飞行内相关也没有一致方向。下一步准备了去掉年龄输入、保留积分和增益的受控消融，尚未启动。

## 固定方法与边界

仅使用September自定义407train父窗口（274Stabilized、133Mission），不使用April canonical。H26/K50与v25完全一致，相位/扑频合同及全部父产物哈希见protocol.json。机械相位物理零位未确认；September比值7.909090995788574，不能替换成April ratio8。

单通道覆盖以各折4个fit飞行的过去26点、飞行等权1%/99%分位及全范围定义。联合覆盖分别对积分3列、压缩增益3列、发布年龄2列、完整8列，取每窗均值/标准差/末值。只在fit飞行做等权标准化（尺度下限1e-6），计算标准化摘要的RMS欧氏距离。校准最近邻排除整个同飞行，以fit飞行等权95分位为阈值；query与全部fit比较，超阈值标为低支持。规则在查看本轮诊断结果前冻结，未按误差选阈值或删窗。

摘要距离忽略完整时序排列，也未恢复特征协方差密度；它是描述性支持代理，不是概率、故障或不可控判定。发布年龄是消息可用性观测，不是舵机或气动力时延。

## 覆盖差异

{table(paired)}

逐飞行低支持数（sum）和总数（size）：

{table(coverage)}

完整块99个低支持窗口中，96个来自log_17、3个来自log_28；年龄块97个全来自log_17。**完整块和年龄块都没有任何飞行同时具备至少8个低支持与8个覆盖窗口**，不能把两组平均误差之差当作同条件比较。

{table(age)}

log_17积分年龄历史点中64.31%超出fit中央98%范围，15.07%超出fit全范围；但其原模型归一化均值偏移只有约−0.235个标准差。差别主要体现在窗口内波动，不能只检查总体均值。其他通道也存在分布差异，详见marginal_distribution.csv。

fit飞行内的飞行间方差占比η²如下（每折仅4fit飞行，表为五折均值；描述性统计，非飞行身份分类准确率）：

{table(eta)}

积分年龄std约0.666、mean约0.382；偏航积分mean约0.383。飞行分组解释了部分特征变化，但可能同时反映真实飞行工况，不能据此证明网络利用了飞行身份或年龄导致错误。

## q退化位置

以下为完整1s波形，每飞行先平均MSE后开方，再飞行/种子等权。正regression_percent表示退化。

{table(flight)}

原有扑频均值、扑频波动三分位和模式标签均逐项复核，没有重新分箱。改善百分比为负表示退化：

{table(frequency)}

各扑频层与两种模式的平均结果均退化，不能把问题局限为某个模式或低扑频。Mission支持飞行数见表，不能按133个窗口当作133次独立飞行。

## 平均偏差与波形分解

ΔMSE = 新增历史组MSE − 零组MSE；单位(rad/s)²，正值为退化。每窗严格满足总MSE=均值偏差平方+去均值残差MSE；前/后0.5s的均值等于完整1sMSE。

{table(decomp)}

OOF增加0.005908，其中平均偏差项反而减少0.000294，去均值波形项增加0.006202；前半秒增加0.007891、后半秒增加0.003925。退化不只在长递推末端，不能用“平均配平偏差增加”或“长时域积分漂移”概括。训练内两项均略改善，仍不能唯一归因于过拟合。

## 覆盖与误差关联的限制

{table(near)}

上表各层使用本层实际支持的飞行等权，组成和数量不同；尤其完整低支持层仅2飞行，其10.38%退化不能直接与覆盖层0.38%相减作因果解释。按原全体窗口权重计算的有符号ΔMSE贡献（不重新归一化子集）：

{table(contribution)}

完整低支持组贡献约0.003805，覆盖组仍贡献约0.002103，合计原0.005908。这是定位贡献，不是删除低支持窗口后的收益估计。

每飞行分别计算支持距离与ΔMSE的Spearman，再平均3种子：

{table(assoc.pivot(index='log_id',columns='block',values='spearman').reset_index())}

方向不一致、数值普遍较小，不支持“越超出支持就一定越退化”的统一解释。5飞行已用于多轮开发，这些相关无因果或独立最终测试含义。

## 验证与下一步

3项测试通过：同飞行最近邻被排除、query不能改变fit统计/阈值、误差分解恒等式。20组fold/block训练校准与query距离独立复算；原扑频阈值/分层、mode标签、v25 fit/OOF总体q指标、每seed407唯一OOF和全部有符号贡献复核通过。核验v12/v23/v24/v25父清单和冻结代码；30检查点仅核验哈希，未重新运行模型。首次预检修正了v12使用仓库相对路径的清单解析，在协议生成前完成，未修改父数据或门槛。

next_ablation_protocol.json准备了一个候选：33维历史编码器保持不变，仅将归一化后两列发布年龄置零，保留6列积分/增益历史。沿用v25初值、抽样、归一化和预算，5折×3seed共15次；与冻结零组作主比较，与完整8维组作次比较，主/保护/绝对门槛不变。这才能检验去掉年龄是否改善泛化；本轮尚未训练，不把当前关联作为已成立的原因。

主要产物：history_summaries.csv、support_per_window.csv、marginal_distribution.csv、q_per_window.csv、q_strata_per_flight.csv、association_per_flight.csv、verification.json；完整哈希见delivery_manifest.json。模型和数据保留在v23/v24/v25原位置。
'''
    (OUT/'report_zh.md').write_text(report)
    code=dict(protocol['code']);code[str(Path(__file__).relative_to(base.ROOT))]=base.sha(Path(__file__))
    base.write(OUT/'delivery_manifest.json',dict(status='complete',artifacts={str(p.relative_to(OUT)):base.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'},code=code,
        training_started=False,model_promoted=False,validation_accessed=False,sealed_test_accessed=False))


if __name__=='__main__':main()
