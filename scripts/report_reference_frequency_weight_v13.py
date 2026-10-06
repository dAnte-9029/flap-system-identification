"""Verify fixed-budget training provenance and report weighting evidence."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_control_windows_v2 as base
from prepare_reference_frequency_weight_v13 import OUT,V8,V12
from reference_groupcv_v8 import load_train,predict
from reference_frequency_weight_v13 import frequency_weights
from evaluate_reference_frequency_weight_v13 import model_from
from run_reference_generalization_v7 import verify_manifest
from report_reference_generalization_v7 import table
from evaluate_control_windows_v2 import METRICS


def main():
    torch.set_num_threads(1);verify_manifest(OUT/'protocol.json',['source_pins'])
    for source in [V8,V12]:verify_manifest(source/'delivery_manifest.json',['artifacts','code'])
    decision=json.loads((OUT/'decision.json').read_text());x,y,w=load_train();checks=[]
    for fold in range(5):
        info=json.loads((OUT/f'fold{fold}/manifest.json').read_text());fit=np.array(info['fit_indices']);query=np.array(info['query_indices'])
        assert not set(w.iloc[fit].log_id)&set(w.iloc[query].log_id)
        weight,cuts,labels,_=frequency_weights(x['history_state'][fit,:,11].numpy().mean(1,dtype=float),w.iloc[fit].log_id.to_numpy())
        with np.load(OUT/f'fold{fold}/weights.npz') as z:
            np.testing.assert_array_equal(z['fit_indices'],fit);np.testing.assert_array_equal(z['query_indices'],query)
            np.testing.assert_allclose(z['weight'],weight,rtol=0,atol=0);np.testing.assert_array_equal(z['cuts'],cuts);np.testing.assert_array_equal(z['stratum'],labels)
        for seed in [17,23,42]:
            run=OUT/f'fold{fold}/seed{seed}';old=V8/f'fold{fold}/hold_{seed}';status=json.loads((run/'status.json').read_text());assert status['status']=='complete' and status['step']==1600
            a=json.loads((run/'run_protocol.json').read_text());b=json.loads((old/'run_protocol.json').read_text());assert a['initial_parameter_sha256']==b['initial_parameter_sha256']
            assert a['source_draws_sha256']==base.sha(old/'sample_draws.npz')
            assert a['protocol_sha256']==base.sha(OUT/'protocol.json')
            with np.load(run/'sample_draws.npz') as z:draws=z['array_index']
            with np.load(old/'sample_draws.npz') as z:np.testing.assert_array_equal(draws,z['array_index'])
            model=model_from(run/'final.pt');ids=query[:4];prediction=predict(model,x,ids)
            with np.load(run/'oof_predictions.npz') as z:
                np.testing.assert_array_equal(z['array_index'],query);error=max(float(np.max(abs(prediction[k]-z[k][:4]))) for k in prediction)
            assert error<1e-5
            batch={k:v[ids].clone() for k,v in x.items()}
            with torch.no_grad():p=model(**batch);batch['future_reference'][:,5:]+=10;q=model(**batch)
            assert all(torch.equal(p[k][:,:5],q[k][:,:5]) for k in p)
            checks.append(dict(fold=fold,seed=seed,identical_initialization=True,identical_all1600_sample_draws=True,reload_max_abs=error,prefix_causal=True,training_elapsed_s=status['elapsed_s']))
    sentinel=json.loads((OUT/'fold0/identity/status.json').read_text());assert sentinel['identity_parameter_max_abs']<1e-6
    test=Path('/tmp/reference-v13-tests.log').read_text();assert '7 passed' in test and 'failed' not in test;(OUT/'tests.txt').write_text(test)
    base.write(OUT/'verification.json',dict(runs=checks,identity_parameter_max_abs=sentinel['identity_parameter_max_abs'],weights_recomputed=True,focused_tests_passed=7,development_evaluated=False,sealed_test_accessed=False))
    wave=pd.read_csv(OUT/'waveform_summary.csv');strata=pd.read_csv(OUT/'strata_summary.csv');support=pd.read_csv(OUT/'weight_support.csv')
    state=pd.read_csv(OUT/'state_per_flight.csv');state=state[(state.subset=='all')&state.horizon_s.isin([.5,1])].groupby(['arm','cohort','horizon_s'])[METRICS].mean().reset_index();state.to_csv(OUT/'state_summary.csv',index=False)
    q=wave[wave.axis=='q'];qs=strata[(strata.axis=='q')&strata.horizon_s.eq(1.)]
    sf=pd.read_csv(OUT/'strata_per_flight.csv');condition=sf[(sf.axis=='q')].groupby(['arm','cohort','stratum']).rmse.mean().unstack('arm');condition['relative_gain']=1-condition.weighted/condition.baseline;condition.to_csv(OUT/'stratum_gains.csv')
    show=support.groupby('stratum').agg(min_weight=('weight','min'),max_weight=('weight','max'),mean_weight=('weight','mean'),mean_original_mass=('original_mass','mean'),mean_weighted_mass=('weighted_mass','mean')).reset_index()
    summary='通过相对研究门槛，但绝对状态误差与独立控制证据仍未通过。' if decision['relative_gate'] else '未通过冻结的相对研究门槛，保留原A1600研究基线。'
    lines=['# 参考模型 v13：历史扑频分层的 q 损失均衡','',f'**15个固定预算训练完成。{summary}本轮没有得到已验证可用于控制的模型。**','',
        '## 唯一改动及可比性','',
        '原407训练窗口、原5个整飞行留一折、seed17/23/42、GRU64/H26/K50、每轮batch32与1600更新保持不变；没有重新筛选样本，也没有叠加v10/v11的r修正。所有任务从原种子新初始化，与对应v8 hold训练的全部1600批次逐项一致，归一化仍来自同折4个训练飞行。',
        '只有q归一化平方误差项按窗口乘权重；p/r、速度、姿态的直接损失不变。网络共享参数，所以实际预测可能在所有轴变化，不能因此省略保护指标。AdamW lr3e-4、wd1e-4、梯度裁剪1保持原值。',
        '先以4个训练飞行等权确定历史平均扑频的共同三分位，再在每个飞行内部，对非空频率层按样本数倒数均衡。把缩放后的权重截在[0.5,2]，通过确定性一维求解使每个飞行的平均权重为1。不按每个飞行单独切等数三分位，否则均衡几乎不产生改动；没有凭结果给低频固定加倍，也没有对小批次再次归一化。',
        '权重层统计（每行先按飞行/折记录，以下为描述性汇总）：','',table(show),'',
        '实际权重约0.83–1.26，未触及0.5/2边界；跨折/飞行的平均三层占比原本已约各三分之一。因此这是幅度较小的单因素对照，失败不能排除所有数据加权方案；也不能以此为理由事后扩大本轮权重。',
        '这里的均衡不必对所有飞行的低频都加权；当某飞行低频样本已经占多数时，其低频权重可能下降。这轮检验的是工况占比均衡，而非“强行优先低频难样本”，两种假设不能混为一谈。','',
        '## 共同留出样本结果','',
        '主指标为q完整0.5/1秒波形RMSE，先每飞行求RMSE，再等权平均时域、飞行和种子；评价不乘训练权重。当前OOF已经多轮用于开发，不能称独立盲测。',
        f'主指标改善{decision["primary_gain"]:.2%}；逐种子改善'+', '.join(f'{s}: {g:.2%}' for s,g in zip([17,23,42],decision['seed_gains']))+'。',
        f'配对整飞行bootstrap95%探索区间[{decision["bootstrap95"][0]:.2%}, {decision["bootstrap95"][1]:.2%}]；仅5个开发飞行，没有校正研究选择过程。','',
        table(q[['arm','cohort','horizon_s','rmse','normalized_centered','amplitude_ratio','mean_correlation','reference_utility']]),'',
        table(pd.DataFrame(dict(log_id=decision['flight_order'],gain=decision['flight_gains']))),'',
        '1秒分层结果与两时域平均相对收益：','',table(qs[['arm','cohort','stratum','rmse','mean_correlation']]),'',table(condition.reset_index()),'',
        '按同一未加权指标，训练内q两时域平均误差约增加0.31%；低频训练内也没有改善（约退化0.02%），而低频OOF仅改善0.16%，中/高频OOF分别退化约0.22%/0.26%。因此这不是一个已有明确训练收益、只在跨飞行时丢失收益的候选；当前均衡本身没有解决q波形问题。',
        '## 保护和参考信息','',
        f'OOF每种子/时域的状态端点最大退化{decision["max_state_regression"]:.2%}，p/r完整波形最大退化{decision["max_pr_waveform_regression"]:.2%}，q各频率层最大退化{decision["max_q_stratum_regression"]:.2%}。各保护阈值均为5%；完整数值见state_regressions.csv、strata_per_flight.csv。',
        '0.5/1秒状态端点（姿态rad，速度m/s，角速度rad/s；与上面的整段波形指标不同）：','',table(state[state.cohort=='oof'][['arm','horizon_s','p','q','r','attitude','velocity']]),'',
        '1秒q保持起点参考对照：原参考与保持参考都对同一观测轨迹评分，收益为1−原参考RMSE/保持参考RMSE。该收益不是飞机反事实控制作用。按种子保存如下：','',
        table(pd.DataFrame([dict(seed=s,effect_ratio=decision['reference_effect_ratios'][str(s)],utility_delta=decision['reference_utility_deltas'][str(s)]) for s in [17,23,42]])),'',
        '相对门槛要求q主指标至少改善5%、全部种子正收益、至少4个飞行正收益，上述状态/分层保护通过；参考收益下降不超过0.02、响应RMS为基线90%–110%。绝对门槛另要求0.5秒p/q/r≤0.15rad/s、姿态≤3度、速度≤0.5m/s。','',table(pd.DataFrame(decision['absolute'])),'',
        '## 结论、复核与后续','',summary,
        '本轮权重不依赖未来响应、当前模型误差或窗口是否“听话”；所有有效样本均保留。工况关联如果没有转化为均衡训练收益，不能据此证明该工况不可建模或必须补飞，也不能事后加大权重、删困难片段或改门槛。',
        ('下一阶段才考虑全407窗口重训及固定开发验证；先保留该OOF候选为研究结果，不在同一阶段自动扩大数据评价。' if decision['relative_gate'] else '下一步把v10固定支撑的时移/波形诊断按v12扑频层拆开，检查低频条件是否有稳定偏移或随预测时间增加的相位失配，并核对输入时间链，再决定是否修改动态表示。事后最佳偏移仍是oracle，不能直接当作真实执行器时延或用来修改时间戳。暂不继续扫描权重或扩大开发验证。'),
        f'7项针对性测试通过；权重全1的额外1600步复现任务与原v8最终参数最大差{sentinel["identity_parameter_max_abs"]:.3g}；15个候选的初始化、全部抽样、权重/阈值复算、checkpoint重载和参考前缀因果性通过。',
        '训练协议protocol.json、权重fold*/weights.npz、模型fold*/seed*/final.pt、判定decision.json、逐样本结果及delivery_manifest.json均保存。v8/v12源产物哈希校验通过，本轮未评价开发验证，sealed/reserved未访问。','']
    (OUT/'report_zh.md').write_text('\n'.join(lines))
    fig,axes=plt.subplots(1,2,figsize=(11,4.5));xx=np.arange(3)
    for ax,cohort in zip(axes,['fit','oof']):
        for arm,offset in [('baseline',-.18),('weighted',.18)]:
            a=qs[(qs.cohort==cohort)&(qs.arm==arm)].set_index('stratum').loc[['low','mid','high']]
            ax.bar(xx+offset,a.rmse,.36,label=arm)
        ax.set_xticks(xx,['Low freq','Middle','High freq']);ax.set(title=f'1 s q waveform: {cohort}',ylabel='Unweighted RMSE (rad/s)');ax.legend();ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'frequency_weight_comparison.png',dpi=150);plt.close(fig)
    paths=list((base.ROOT/'scripts').glob('*frequency_weight_v13.py'))+[base.ROOT/'tests/test_reference_frequency_weight_v13.py']
    base.write(OUT/'delivery_manifest.json',dict(status='complete',code={str(p.relative_to(base.ROOT)):base.sha(p) for p in paths},artifacts={str(p.relative_to(base.ROOT)):base.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'},focused_tests_passed=7,relative_gate=decision['relative_gate'],control_validated=False,development_evaluated=False,sealed_test_accessed=False))
    verify_manifest(OUT/'delivery_manifest.json',['artifacts','code']);print('Delivery verified',flush=True)

if __name__=='__main__':main()
