"""Reviewable v7 report and provenance checks; no fitting or candidate selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import train_control_windows_v2 as base
from reference_generalization_v7 import phase_features, phase_prediction
from run_reference_generalization_v7 import OUT, V6, verify_manifest


def table(frame):
    def fmt(v):
        return f'{v:.6g}' if isinstance(v,(float,np.floating)) else str(v)
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |',
        '| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(map(fmt,row))+' |' for row in frame.itertuples(index=False,name=None)])


def main():
    assert json.loads((OUT/'status.json').read_text())['status']=='complete'
    verify_manifest(OUT/'protocol.json',['source_pins'])
    verify_manifest(V6/'delivery_manifest.json',['artifacts','code','source_pins'])
    x,_,_=base.load();w=pd.read_csv(base.DATA/'windows_v2.csv');h=x['history_state'].numpy()
    selection=json.loads((OUT/'phase_selection.json').read_text())
    with np.load(OUT/'phase_candidate.npz') as z:
        np.testing.assert_array_equal(z['array_index'],w.array_index)
        if selection['winner'].startswith('ridge'):
            expected=phase_prediction('hold',h)[0]+(phase_features(h)-z['mean'])/z['scale']@z['coefficient']+z['target_mean']
        else: expected=phase_prediction(selection['winner'],h)[0]
        maximum=float(np.max(abs(expected-z['prediction'])))
        assert maximum<1e-10
        implied_freq=np.diff(np.column_stack([np.zeros(len(h)),z['prediction']]),axis=1)/(.02*2*np.pi)
        frequency_audit=dict(negative_fraction=float((implied_freq<0).mean()),
            minimum=float(implied_freq.min()),maximum=float(implied_freq.max()),
            meaning='implied phase trajectory increments only; frequency decoder context remains original held frequency')
    fold_checks=[]
    for path in sorted(OUT.glob('phase_outer_fold*.npz')):
        with np.load(path) as z:
            fit=z['fit_indices'];query=z['query_indices'];a=w.iloc[fit];b=w.iloc[query]
            assert (a.partition=='train').all() and (b.partition=='train').all()
            assert not set(a.log_id)&set(b.log_id) and len(set(b.log_id))==1
            fold_checks.append(dict(file=path.name,fit_flights=a.log_id.nunique(),query_flights=b.log_id.nunique(),disjoint=True))
    verify=json.loads((OUT/'verification.json').read_text())
    verify.update(phase_candidate_reload_max_abs=maximum,outer_fold_audit=fold_checks,
        focused_tests_passed=24,future_phase_oracle_not_deployable=True,frequency_increment_audit=frequency_audit)
    base.write(OUT/'verification.json',verify)
    pf=pd.read_csv(OUT/'frozen_per_flight.csv');cv=pd.read_csv(OUT/'window_covariates.csv')
    cond=pd.read_csv(OUT/'condition_summary.csv');support=pd.read_csv(OUT/'support_summary.csv')
    phase=pd.read_csv(OUT/'phase_per_flight.csv').groupby(['model','partition','horizon_s'])[['circular_rmse','full_prefix_harmonic_rmse']].mean().reset_index()
    phase.to_csv(OUT/'phase_summary.csv',index=False)
    ps=phase[(phase.partition=='validation')&(phase.horizon_s==1.)].set_index('model')
    phase_gain=float(1-ps.loc['selected','full_prefix_harmonic_rmse']/ps.loc['hold','full_prefix_harmonic_rmse'])
    replay=pd.read_csv(OUT/'replay_summary.csv');decision=json.loads((OUT/'replay_decision.json').read_text())
    comparisons=decision['comparisons'];candidate,oracle=comparisons
    a=pf[(pf.arm=='integrated1600')&(pf.horizon_s==.5)].copy();a['flight']=a.log_id.str.replace('2026.9.','9.',regex=False)
    counts=cv.groupby(['partition','mode']).size().rename('windows').reset_index()
    quality=cv.groupby(['partition','quality']).size().rename('windows').reset_index()
    condition_counts=[]
    for col in ['mode','quality','frequency_bin','speed_bin','history_rate_rms_bin']:
        g=cv.groupby(['partition',col]).agg(windows=('array_index','size'),flights=('log_id','nunique')).reset_index().rename(columns={col:'stratum'})
        g['condition']=col;condition_counts.append(g)
    counts_all=pd.concat(condition_counts);counts_all.to_csv(OUT/'condition_counts.csv',index=False)
    assoc=pd.read_csv(OUT/'phase_error_association.csv').groupby(['partition','axis']).spearman.median().reset_index()
    budget=pf[(pf.arm=='integrated6400')&(pf.partition=='validation')&(pf.horizon_s==.5)].set_index('log_id')
    baseline=a[a.partition=='validation'].set_index('log_id')
    worsened=int((budget.p>baseline.p).sum())
    gap=cond[(cond.arm=='integrated1600')&(cond.horizon_s==.5)&cond.condition.isin(['mode','frequency_bin','quality'])]
    lines=['# 参考模型 v7：跨飞行泛化与相位外推诊断','',
        '**结论：历史数据足以改善相位预测，但直接替换冻结GRU的相位输入没有改善状态预测；保留A1600研究基线，不晋升控制用途。**','',
        '## 范围与可复现性','',
        '原407训练/739开发验证窗口、H26/K50、50Hz不变，保留所有弱响应和困难样本。训练5架次均来自9月6日，验证9架次均来自9月7日；日期和分区完全混杂，不能把差距唯一归为跨日期变化。验证已反复用于开发，不称盲测；sealed/reserved未打开。',
        '相位来自9月日志的相对wing phase，物理零点尚未确认；沿用ratio7.909090995788574，未使用April canonical数据或将其ratio8契约混入。所有角速度误差单位rad/s、速度m/s、姿态rad。状态误差先在各飞行内求RMSE，再等权平均飞行和种子。','',
        '## 各飞行与工况','',
        'A1600在0.5秒的全窗口结果（3seed均值）：','',table(a[['partition','flight','p','q','r','attitude','velocity']]),'',
        f'从1600延长到6400步后，9个验证飞行中有{worsened}个的p误差上升，退化不集中在单一飞行。训练内误差不是分组留出误差，不能把它当作泛化准确率。','',
        '模式与质量层样本数：','',table(counts),'',table(quality),'',
        '训练Mission占32.7%，验证占25.0%。两个模式均保留；仅凭模式不能解释全部p差距。同模式、同扑频分层中仍能看到差距；这些是描述性分层，未控制所有混杂因素。Q2训练只有7个窗口、Q3两组各5个，分层均值不稳定，不能据此删除某类数据。','',
        table(gap[['partition','condition','stratum','p','q','r','attitude','velocity']]),'',
        'frequency_bin使用训练数据的三分位边界，边界存于training_bins.json。每层样本和飞行数见condition_counts.csv；速度与历史角速度分层见condition_summary.csv。','',
        '训练1%–99%边缘范围覆盖检查（不等于联合支持或安全包线）：','',table(support[support.partition=='validation'][['covariate','low','high','outside_fraction','median']]),'',
        '多数验证窗口的扑频、速度、历史角速度落在这些训练边缘范围内；这排除了“绝大多数验证样本明显超出单变量训练范围”的简单解释，但不证明联合工况、机体配置或未测扰动相同。','',
        '## 训练飞行内选择相位预测器','',
        '固定8个候选：当前扑频保持、最近5点平均、相位历史6/11/26点线性斜率、阻尼扑频趋势，以及alpha10/100岭回归。后者仅用15维历史特征预测未来相对相位残差；未来相位作为训练标签，推理不读取未来测量。',
        '外层整飞行留一：在其余4个训练飞行内再次留一选择候选，再评价第5个飞行。最终候选通过5个训练飞行的留一结果选定，并在读取验证评价结果前保存选择。岭回归标准化与均值仅使用相应训练折，五个外层均选择ridge10。',
        '本轮交叉验证仅针对相位预测器，未重新交叉训练GRU。相位主指标为1秒内一、二阶谐波弦长误差的均方根，先各飞行后等权平均；它不是以rad计的相位RMSE。','',
        table(pd.DataFrame([{k:d[k] for k in ['log_id','winner','hold','candidate','gain']} for d in selection['outer_folds']])), '',
        f'嵌套留一主误差改善{selection["nested_gain"]:.2%}，5/5飞行改善，达到预定相位门槛。固定最终候选在开发验证上改善{phase_gain:.2%}。下面另报各端点圆周相位RMSE（rad），避免混用指标：','',table(phase),'',
        '相位预测器参数保存在phase_candidate.npz，重载复算已核对；全部输入来自历史，50个未来相位输出不依赖未来参考或测量。','',
        '## 冻结状态模型是否获益？','',
        '使用原A1600三个种子的相同权重，仅替换每步开始时的相位；第一步仍为0，之后用前一时刻的相位预测。频率上下文、历史编码器和状态递推保持原样。原始相位重放与旧保存预测一致。训练与推理的相位分布可能不同，因此这是输入敏感性诊断，不是训练一致的新模型。','',
        table(replay[(replay.partition=='validation')&(replay.subset=='all')&replay.horizon_s.isin([.5,1.])][['arm','horizon_s','p','q','r','attitude','velocity']]),'',
        f'- 历史相位候选：0.5/1秒主动p/q主误差改善{candidate["primary_gain"]:.2%}，即退化；按飞行配对bootstrap95%改善区间[{candidate["bootstrap95"][0]:.2%}, {candidate["bootstrap95"][1]:.2%}]。',
        f'- 真实未来相位oracle：同指标改善{oracle["primary_gain"]:.2%}，同样没有收益。oracle使用未来测量，仅是离线诊断。',
        '- 候选未通过相对及绝对状态误差门槛；不替换基线，不启动控制优化。','',
        '这些结果只说明对已用恒定扑频训练的冻结网络，替换相位不能立即改善预测。不能据此认定相位没有作用，也不能推断以一致相位契约重新训练后仍无效。oracle也不是可实现模型的严格精度上界。','',
        '按飞行计算的1秒p/q状态平方误差与相位误差Spearman相关中位数：','',table(assoc),'',
        '相关性不构成因果归因；p的中位相关接近零，也不排除其他时间尺度、非线性或共同状态因素。','',
        '## 交付与下一步','',
        '24项针对性测试通过；6组原模型重放一致；5个外层训练/查询飞行互斥；相位预测器重载一致；源数据、旧权重及v6交付哈希复核通过。模型文件与完整误差表均保留，未删数据。',
        '下一步优先建立动力学模型本身的训练飞行留一基线：每折重新拟合归一化并重训，判断同日跨飞行是否也出现相似差距；然后才在同一折划分中比较训练/推理一致的相位方案。不能重复使用已经看过该折的GRU权重来冒充交叉验证，也不再依据开发验证结果扩展相位候选网格。',
        '本轮没有证明必须新增飞行试验。控制响应和真实闭环有效性仍需独立证据，不能用辅助相位预测的改善替代。','']
    (OUT/'report_zh.md').write_text('\n'.join(lines))
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for part,color in [('train','tab:blue'),('validation','tab:orange')]:
        z=a[a.partition==part];axes[0].scatter(z['p'],z['q'],label=part,color=color)
        for row in z.itertuples():axes[0].annotate(row.log_id.split('/')[-1].split('_')[1],(row.p,row.q),xytext=(3,3),textcoords='offset points',fontsize=8)
    axes[0].set(xlabel='p RMSE at 0.5 s (rad/s)',ylabel='q RMSE at 0.5 s (rad/s)',title='A1600 per flight; labels are log numbers')
    for name,color in [('hold','tab:gray'),('selected','tab:green')]:
        z=phase[(phase.model==name)&(phase.partition=='validation')];axes[1].plot(z.horizon_s,z.circular_rmse,'o-',label=name,color=color)
    axes[1].set(xlabel='Forecast horizon (s)',ylabel='Circular phase RMSE (rad)',title='Fixed phase predictor, development validation')
    for ax in axes:ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(OUT/'generalization_phase.png',dpi=160);plt.close(fig)
    code=[Path(__file__),base.ROOT/'scripts/run_reference_generalization_v7.py',base.ROOT/'scripts/reference_generalization_v7.py',base.ROOT/'tests/test_reference_generalization_v7.py']
    base.write(OUT/'delivery_manifest.json',dict(status='complete',source_protocol_sha256=base.sha(OUT/'protocol.json'),
        artifacts={str(p.relative_to(base.ROOT)):base.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'},
        code={str(p.relative_to(base.ROOT)):base.sha(p) for p in code},focused_tests=24,sealed_test_accessed=False,
        model_promoted=False,phase_forecaster_selected_on_training_only=True))
    print(json.dumps(dict(phase_nested_gain=selection['nested_gain'],phase_validation_gain=phase_gain,
        state_primary_gain=candidate['primary_gain'],oracle_primary_gain=oracle['primary_gain']),indent=2))

if __name__=='__main__':main()
