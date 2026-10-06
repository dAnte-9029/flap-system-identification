"""Audit v16 against frozen data and v15, then write diagnostic report."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_control_windows_v2 as base
from reference_groupcv_v8 import load_train, model_from, SEEDS
from reference_increment_v16 import history_spectrum
from reference_condition_diagnostics_v12 import weighted_terciles, bins
from run_reference_increment_v16 import OUT,V8,V12,V15
from run_reference_generalization_v7 import verify_manifest
from report_reference_generalization_v7 import table


def main():
    torch.set_num_threads(1)
    verify_manifest(OUT/'protocol.json',['source_pins'])
    for rel,digest in json.loads((OUT/'model_pins.json').read_text()).items():
        assert base.sha(base.ROOT/rel)==digest
    wdf=pd.read_csv(OUT/'per_window.csv');pf=pd.read_csv(OUT/'per_flight.csv')
    summary=pd.read_csv(OUT/'summary.csv');source=pd.read_csv(OUT/'source_flight_summary.csv')
    x,y,w=load_train();hf,share=history_spectrum(x['history_state'].numpy()[:,:,3:6])
    old=pd.read_csv(V15/'global_age_per_flight.csv')
    old=old[(old.arm=='free50')&(old.stratum=='all')&(old.global_age_s==.02)]
    checks=[];full_peak=np.zeros(3)
    cuts=json.loads((OUT/'history_cuts.json').read_text())
    for cut in cuts:
        j='pqr'.index(cut['axis']);ids=np.array(cut['fit_indices'])
        expected=weighted_terciles(hf[ids,j],w.iloc[ids].log_id.to_numpy())
        np.testing.assert_array_equal(expected,[cut['low_cut'],cut['high_cut']])
        subset=wdf[(wdf.fold==cut['fold'])&(wdf.axis==cut['axis'])]
        np.testing.assert_array_equal(bins(hf[subset.array_index,j],expected),subset.hf_stratum)
    for fold in range(5):
        for seed in SEEDS:
            model=model_from(V8/f'fold{fold}/hold_{seed}/final.pt')
            for cohort in ['fit','oof']:
                with np.load(OUT/f'fold{fold}/{cohort}_seed{seed}.npz') as z:
                    ids=z['array_index'];origin=x['initial_rates_b'][ids].numpy()
                    np.testing.assert_array_equal(z['truth_delta'],y['rates_b'][ids,0].numpy()-origin)
                    full_peak=np.maximum(full_peak,abs(z['raw_head'][:,:,3:]).max(axis=(0,1)))
                    # Independent short forward: request only first planned reference.
                    inp={k:v[ids[:3]].clone() for k,v in x.items()}
                    inp['future_reference']=inp['future_reference'][:,:1];inp['dt_s']=inp['dt_s'][:,:1]
                    with torch.no_grad():pred=model(**inp)['rates_b'][:,0].numpy()
                    error=float(np.max(abs((pred-origin[:3])-z['pred_delta'][:3])))
                    assert error<1e-6
                    checks.append(dict(fold=fold,seed=seed,cohort=cohort,independent_short_forward_max_abs=error))
                for j,axis in enumerate('pqr'):
                    rows=wdf[(wdf.fold==fold)&(wdf.seed==seed)&(wdf.cohort==cohort)&(wdf.axis==axis)]
                    np.testing.assert_array_equal(rows.array_index,ids)
                    np.testing.assert_allclose(rows.hf_rms,hf[ids,j],atol=1e-12)
                    np.testing.assert_allclose(rows.hf_share,share[ids,j],atol=1e-12)
    current=pf[pf.condition=='all']
    previous=old.melt(id_vars=['fold','seed','cohort','log_id'],value_vars=list('pqr'),var_name='axis',value_name='previous_rmse')
    match=current.merge(previous,on=['fold','seed','cohort','log_id','axis'],validate='one_to_one')
    old_difference=float(abs(match.rmse-match.previous_rmse).max());assert old_difference<1e-6
    coverage=pf.groupby(['cohort','axis','condition','stratum']).agg(min_windows_per_flight_model=('n','min'),
        max_windows_per_flight_model=('n','max'),undefined_correlations=('correlation',lambda s:s.isna().sum()),
        flight_model_groups=('n','size')).reset_index()
    coverage.to_csv(OUT/'coverage.csv',index=False)
    tests=Path('/tmp/reference-v16-tests.log').read_text()
    assert '4 passed' in tests and 'failed' not in tests
    (OUT/'tests.txt').write_text(tests)
    v=json.loads((OUT/'verification.json').read_text())
    v.update(independent_short_forwards=checks,previous_v15_max_abs=old_difference,
        full_rollout_peak_abs_raw_rate_head=full_peak.tolist(),history_features_and_fit_cuts_verified=True,focused_tests_passed=4)
    base.write(OUT/'verification.json',v)
    main=summary[summary.condition=='all']
    oof=main[main.cohort=='oof']
    hfrows=summary[(summary.cohort=='oof')&(summary.condition=='history_hf_rms')]
    fq=summary[(summary.cohort=='oof')&(summary.condition=='history_frequency')&(summary.axis=='q')]
    association=source[(source.cohort=='oof')&(source.condition=='all')].groupby(['axis','log_id'])[['hf_error_spearman','correlation','hold_gain']].mean().reset_index()
    association.to_csv(OUT/'oof_association_per_flight.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(12,3.7))
    for ax,axis in zip(axes,'pqr'):
        # Average only predictions across seeds; truth and windows appear once.
        g=wdf[(wdf.cohort=='oof')&(wdf.axis==axis)].groupby('array_index')[['truth_delta','pred_delta']].mean()
        ax.scatter(g.truth_delta,g.pred_delta,s=9,alpha=.5)
        limit=max(abs(g.to_numpy()).max(),.1);ax.plot([-limit,limit],[-limit,limit],color='gray',lw=1)
        ax.set(xlabel='Observed increment (rad/s)',ylabel='Predicted increment (rad/s)',title=f'{axis}: OOF, seed mean',xlim=(-limit,limit),ylim=(-limit,limit))
        ax.set_aspect('equal')
    fig.tight_layout();fig.savefig(OUT/'increment_scatter.png',dpi=160);plt.close(fig)
    decision=dict(stage_complete=True,diagnostic_only=True,derivative_clamp_binding=False,
        model_promoted=False,control_validated=False,development_evaluated=False,sealed_test_accessed=False,
        next_stage='Freeze same-input same-fold single-step objective diagnostic against 1s-trained baseline, evaluate first20ms and full1s tradeoff; do not simply rescale increments or remove higherror windows.')
    base.write(OUT/'decision.json',decision)
    lines=['# 参考模型 v16：原始20ms角速度增量诊断','',
        '**导数限幅没有触发。p增量幅值偏小且相关性有限；q比p更可预测，但也有明显一步误差。训练内同样存在这些现象，暂不能归因为限幅、长递推或不可预测噪声。保留A1600研究基线。**','',
        '## 数据与口径','',
        '原407个非重叠父窗口、5个原训练飞行、5折×3个冻结v8 hold模型。每折仅在其4个训练飞行上计算fit，在排除于模型及归一化的第5个飞行上计算OOF。未重训、重新筛选或改变数据划分；未评价739个开发验证窗口，sealed/reserved未访问。OOF已经多次研究，不能称盲测。',
        '本轮是原始窗口起点到20ms后一个点的增量：Δω=ω(t+20ms)−ω(t)。预测和真值共用同一个已观测初值，所以增量误差恰等于首次角速度预测误差。未来实测量只用于评价，不是模型输入。这里的相关性是跨窗口的一步增量相关性，不是1秒波形相关性或控制因果效应。',
        '先对每模型/飞行计算指标；fit中重复出现的源飞行先平均其四个模型，再等权平均飞行和种子。比值/收益也是逐飞行计算后平均，不能由表中两个总体均值相除精确复原。标准差幅值比、相关性、分位数和逐飞行统计均保存在CSV中。','',
        '## 首个20ms结果','',table(main[['cohort','axis','rmse','truth_rms','pred_rms','std_ratio','correlation','sign_agreement','hold_gain']]),'',
        'RMSE、RMS单位rad/s；std_ratio为预测增量标准差/真值标准差，hold_gain为相对保持当前角速度（零增量）的RMSE改善比例。方向一致率只在|真实增量|>0.01rad/s时计算，|预测增量|≤0.01也计为未预测正确；其他指标保留全部窗口，未删除弱响应。这个0.01是报告死区，并非传感器噪声阈值。',
        'OOF p/q增量标准差约为真值39.3%/72.9%，相关系数0.555/0.739，方向一致率68.0%/81.2%。fit也只有39.7%/73.7%、相关0.593/0.794；不能只解释为留出分布变化。p/q仍较保持值改善14.4%/32.6%，说明已有局部可预测信息。r改善仅2.2%，且一个飞行退化，作为对照保留。',
        '幅值偏小不自动支持放大输出：带不确定性的平方误差预测可以出现向条件均值收缩，相关性不足时放大也会放大错误。此次未拟合输出校准，未把事后真值尺度当成可部署增益。','',
        '![OOF增量散点](increment_scatter.png)','',
        '图中每个窗口只出现一次，预测取3种子均值；表中按单个模型先算指标，因此图与表不采用同一集成口径。','',
        '## 限幅是否造成瓶颈','',table(oof[['axis','first_saturation','rollout_saturation','truth_exceeds_limit','clamp_oracle_rmse','exceed_error_share']]),'',
        '模型头先限幅±6，再乘20rad/s²，dt=0.02s，故每步角速度增量界为±2.4rad/s。所有15模型、fit/OOF、全部50步均无角速度头限幅。首步raw head最大绝对值p/q/r约1.634/1.964/0.399，远未达到6。完整递推各轴峰值见verification.json。',
        '只有1/407个独立窗口的真实Δp超过2.4rad/s，q/r没有。表中超限比例按飞行等权，约0.3125%，不是简单的1/407。允许逐点使用真值、仅受这个增量界约束时的理论最小RMSE，p仅0.00215rad/s，远小于实际0.523；这是不可部署的约束诊断下界，不是剩余误差的噪声下界。没有放宽限幅的证据。','',
        '## 高频历史与误差','',
        '只对起点前26点角速度去均值、加Hann窗，计算单边谱>8Hz的能量RMS与占比。频率栅格约1.923Hz，Nyquist25Hz；该指标受短窗泄漏和已有缓存重采样影响，不能视为原始IMU谱或噪声测量。每轴按当前折4个fit飞行等权三分位定界，再应用OOF，所有窗口保留。','',
        table(hfrows[['axis','stratum','rmse','truth_rms','std_ratio','correlation','hold_gain']]),'',
        'p高频RMS高层一步误差0.575、低层0.457rad/s；但绝对误差与历史高频强度的逐飞行Spearman均值仅0.162，5飞行均正，幅度不强。q相应均值仅0.026且跨飞行正负不一致，未支持“高频越大就是主要误差来源”。这里是描述性关联，没有分离工况、信号幅值、采样和真实运动，不能按高频强度删样本或宣称不可预测噪声。',
        '部分p分层在单飞行/模型内仅1个样本，相关性无法定义，CSV保留空值；汇总相关性仅平均可定义组，不对小层作稳定性结论。coverage.csv记录各层样本数和未定义相关数。全体指标每飞行至少64窗口，不受该小层问题影响。','',
        table(association[['axis','log_id','hf_error_spearman','correlation','hold_gain']]),'',
        '## 原有扑频层','',table(fq[['stratum','rmse','truth_rms','std_ratio','correlation','hold_gain']]),'',
        '沿用v12 q历史扑频层：低/高层20ms绝对误差相近（0.275/0.272），但真值增量RMS为0.334/0.473，相关性0.592/0.837，相对保持值改善14.5%/42.3%。说明低频层相对预测困难仍在；不能简单说低频层每个时刻绝对误差都更大，也不能由关联推出频率因果机制。','',
        '## 验证与下一步','',
        '4项针对性测试通过；30组完整冻结预测重放、首步及50步head→增量重构、30组独立单步推理、原始标签/历史频谱/fit分层阈值复算、407个OOF唯一性、v15首次20ms误差逐飞行逐种子复现和源哈希核验通过。原始数据、权重、模型限幅均未修改。',
        '下一步应冻结“相同输入和同折模型，专门优化20ms单步目标”的对照，保留原1秒训练模型与保持值基线。先看训练内和OOF一步误差能否同时改善，再评价相同1秒完整递推与其他状态是否退化，用来区分长时域训练目标的取舍与当前历史/表示的局限。预算、初始化、样本及归一化固定，不根据查询结果调参。',
        '这是一项新的诊断对照，尚未执行。即便一步改善，也不等于1秒模型或控制作用验收通过；若训练内改善而OOF无收益，应报告泛化差距，不能继续把训练收益当控制能力。本轮不修改限幅、不按响应强弱删除窗口，不能据此认定必须补飞。','']
    (OUT/'report_zh.md').write_text('\n'.join(lines))
    code=[base.ROOT/'scripts/reference_increment_v16.py',base.ROOT/'scripts/run_reference_increment_v16.py',Path(__file__),base.ROOT/'tests/test_reference_increment_v16.py']
    base.write(OUT/'delivery_manifest.json',dict(artifacts={str(p.relative_to(base.ROOT)):base.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'},code={str(p.relative_to(base.ROOT)):base.sha(p) for p in code}))
    verify_manifest(OUT/'delivery_manifest.json',['artifacts','code'])
    print('v16 report, independent audit and delivery hashes complete',flush=True)


if __name__=='__main__':main()
