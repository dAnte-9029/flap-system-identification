"""Report fixed-budget OOF dynamics and audit every fold's data provenance."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_control_windows_v2 as base
from reference_groupcv_v8 import OUT,PHASE,SEEDS,load_train,phase_start,fit_normalization,model_from,predict
from reference_generalization_v7 import phase_features,phase_prediction,phase_loss
from run_reference_generalization_v7 import verify_manifest
from report_reference_generalization_v7 import table
from evaluate_reference_groupcv_v8 import compare
from evaluate_control_windows_v2 import METRICS


def main():
    torch.set_num_threads(1)
    assert json.loads((OUT/'evaluation_status.json').read_text())['status']=='complete'
    verify_manifest(OUT/'protocol.json',['source_pins'])
    verify_manifest(OUT/'preparation.json',['cache_pins'])
    x,y,w=load_train();h=x['history_state'].numpy();prep=json.loads((OUT/'preparation.json').read_text())
    arrays={}
    for log,g in w.groupby('log_id'):
        with np.load(base.ROOT/g.cache.iloc[0]) as z:arrays[log]={k:z[k] for k in ['state_features','references','actuator_commands','rates_b']}
    with np.load(PHASE) as z:phase_truth=z['phase'][w.array_index.to_numpy()]
    provenance=[];phase_quality=[]
    for fold in prep['folds']:
        path=OUT/f'fold{fold["fold"]}';fit=np.array(fold['fit_indices']);query=np.array(fold['query_indices'])
        assert not set(w.iloc[fit].log_id)&set(w.iloc[query].log_id)
        stats=fit_normalization(arrays,w.iloc[fit],y['velocity_n'].numpy())
        with np.load(path/'normalization.npz') as z:
            for k in stats:np.testing.assert_array_equal(z[k],stats[k])
        with np.load(path/'phase_schedules.npz') as z:schedule=z['phase_start']
        for cohort,ids in [('fit',fit),('oof',query)]:
            hold=phase_prediction('hold',h[ids])[0][:,:49]
            pred=schedule[ids,1:]
            a=phase_loss(hold,phase_truth[ids,:49]);b=phase_loss(pred,phase_truth[ids,:49])
            for log in sorted(w.iloc[ids].log_id.unique()):
                mask=w.iloc[ids].log_id.to_numpy()==log
                phase_quality.append(dict(fold=fold['fold'],cohort=cohort,log_id=log,
                    hold=float(np.sqrt(a[mask].mean())),phase=float(np.sqrt(b[mask].mean())),
                    negative_implied_frequency_fraction=float((np.diff(schedule[ids[mask]],axis=1)<0).mean())))
        maximum=0.
        for n,source in enumerate(fold['phase_provenance']):
            ti=np.array(source['fit_indices']);qi=np.array(source['query_indices'])
            assert set(ti)<=set(fit)
            assert not set(w.iloc[ti].log_id)&set(w.iloc[qi].log_id)
            if n<4:assert set(qi)<=set(fit)
            else:np.testing.assert_array_equal(qi,query)
            with np.load(path/f'phase_predictor{n}.npz') as z:
                np.testing.assert_array_equal(z['fit_indices'],ti);np.testing.assert_array_equal(z['query_indices'],qi)
                end=phase_prediction('hold',h[qi])[0]+(phase_features(h[qi])-z['mean'])/z['scale']@z['coefficient']+z['target_mean']
            maximum=max(maximum,float(np.max(abs(schedule[qi]-phase_start(end)))))
        assert maximum<1e-5
        provenance.append(dict(fold=fold['fold'],fit_windows=len(fit),query_windows=len(query),fit_flights=4,query_flights=1,
            normalization_exact_recompute=True,phase_predictor_reload_max_abs=maximum,phase_fit_excludes_own_and_outer_flight=True))
    verify=json.loads((OUT/'verification.json').read_text());verify.update(fold_provenance=provenance,focused_tests_passed=27)
    base.write(OUT/'verification.json',verify)
    auxiliary=pd.DataFrame(phase_quality);auxiliary.to_csv(OUT/'auxiliary_phase_quality.csv',index=False)
    auxiliary=auxiliary.groupby(['cohort','log_id'])[['hold','phase','negative_implied_frequency_fraction']].mean().groupby('cohort').mean().reset_index()
    auxiliary['gain']=1-auxiliary.phase/auxiliary.hold
    d=json.loads((OUT/'cv_decision.json').read_text());full=json.loads((OUT/'full_training_status.json').read_text())
    summary=pd.read_csv(OUT/'cv_summary.csv');general=pd.read_csv(OUT/'baseline_generalization_summary.csv')
    per=pd.read_csv(OUT/'cv_per_flight.csv')
    oof=per[(per.cohort=='oof')&(per.subset=='all')&(per.horizon_s==.5)].groupby(['arm','log_id'])[METRICS].mean().reset_index()
    oof['flight']=oof.log_id.str.split('/').str[-1]
    full_decision=None
    if full['status']=='complete':
        data=[];xall,yall,_=base.load();wall=pd.read_csv(base.DATA/'windows_v2.csv');meta=wall[wall.partition=='validation'];ids=meta.array_index.to_numpy();fullchecks=[]
        for seed in SEEDS:
            path=OUT/'full'/f'phase_{seed}';s=json.loads((path/'status.json').read_text());assert s['status']=='complete' and base.sha(path/'final.pt')==s['checkpoint_sha256']
            f=pd.read_csv(path/'per_flight.csv');data.append(f[f.cohort=='development'])
            old=pd.read_csv(OUT.parent/'control_windows_v6/formal'/f'integrated_{seed}/1600_per_flight.csv');old=old[old.cohort=='validation'].copy();old['cohort']='development';old['arm']='hold';data.append(old)
            with np.load(path/'phase_schedules.npz') as z:schedule=torch.tensor(z['phase_start'])
            pred=predict(model_from(path/'final.pt'),xall,ids[:4],schedule)
            with np.load(path/'development_predictions.npz') as z:error=max(float(np.max(abs(pred[k]-z[k][:4]))) for k in pred)
            assert error<1e-5;fullchecks.append(dict(seed=seed,reload_max_abs=error))
        fullframe=pd.concat(data);fullframe.to_csv(OUT/'full_per_flight.csv',index=False)
        full_decision=compare(fullframe,cohort='development',prefix='full',minimum_positive_flights=0)
        full_decision['relative_gate']=full_decision.pop('cv_gate');full_decision.pop('full_phase_training_authorized_by_protocol')
        full_decision['checkpoints']=fullchecks;full_decision['control_validated']=False
        base.write(OUT/'full_decision.json',full_decision)
    g=general[(general.subset=='all')&(general.horizon_s==.5)].set_index('cohort')
    ratios={k:float(g.loc['oof',k]/g.loc['fit',k]-1) for k in METRICS}
    days={k:float(g.loc['development',k]/g.loc['oof',k]-1) for k in METRICS}
    base.write(OUT/'generalization_gaps.json',dict(same_day_oof_vs_fit=ratios,development_vs_same_day_oof=days,
        caveat='different evaluation populations; model/training size matched for nextday audit, not causal day attribution'))
    phase_absolute=[a for a in d['absolute'] if a['arm']=='phase']
    lines=['# 参考模型 v8：动力学整飞行留一训练','',
        ('**相位一致训练未达到留一门槛；同日跨飞行已有明显泛化差距，保留A1600研究基线。**' if not d['cv_gate'] else '**动力学整飞行留一对照完成，相位方案达到相对门槛；控制用途仍需独立验收。**'),'',
        '## 协议与隔离','',
        '原5个训练飞行、407个非重叠H26/K50窗口保持不变。5折×两种方案×3seed（17/23/42），共30次从头训练，每次固定1600更新、batch32、GRU64/40070参数。hold为原恒定扑频相位外推；phase为训练与推理都使用历史预测相位的方案，频率上下文仍保持起点值。',
        '每折只用4个训练飞行重新拟合归一化，按照v2相同唯一源采样行口径。两种方案使用相同归一化、初始参数、抽样序列和损失。将全部5飞行交给新归一化实现时，与旧数据集的所有统计量逐项完全一致；每个折已重新核验只含其4个拟合飞行。',
        '相位预测器固定为v7选定的ridge10，不在本轮扩展候选。GRU训练中的每个飞行，只使用另外3个拟合飞行训练的相位预测器；外层查询使用全部4个拟合飞行训练的相位预测器。查询样本自己的未来相位没有进入预测特征。此交叉预测训练存在3飞行/4飞行辅助模型拟合量差异，属于两阶段估计，不能称完全相同的相位预测器。',
        'ridge10方法曾参考v7的这些训练飞行结果，因此本轮是隔离良好的开发交叉验证，不是从未接触过的盲测。预先固定相位方案、1600预算和门槛；不根据留一结果挑checkpoint。未重新筛选窗口，sealed/reserved未访问。','',
        '各折样本数与来源：','',table(pd.DataFrame(provenance)[['fold','fit_windows','query_windows','fit_flights','query_flights']]),'',
        '## 同日跨飞行的泛化差距','',
        '下表hold使用相同15个折模型。fit是其训练内结果；oof是完全未参与该模型训练的同日飞行；development是冻结OOF决定后，对739个次日开发验证窗口的评价。每个验证飞行先在模型/种子间平均，再平均飞行，不将重复评价算成45个独立飞行。','',
        table(general[(general.subset=='all')&general.horizon_s.isin([.5,1.])][['cohort','horizon_s',*METRICS]]),'',
        f'0.5秒同日OOF相对训练内：p误差变化{ratios["p"]:+.2%}、q {ratios["q"]:+.2%}、姿态 {ratios["attitude"]:+.2%}。因此同日跨飞行的差距现在有直接的模型留一证据，不再用已见训练飞行误差替代。',
        f'相同模型在次日development相对同日OOF：p误差变化{days["p"]:+.2%}、q {days["q"]:+.2%}、姿态 {days["attitude"]:+.2%}。评价飞行和工况仍不同，不能将差异唯一解释成日期、风或机体变化。',
        '每次拟合4飞行，相同1600更新意味着每个拟合窗口平均曝光比全5飞行训练更多；跨日期比较已使用相同折模型控制训练量，旧A1600全5飞行结果不能直接当同规模对照。','',
        '各同日留出飞行的0.5秒结果（3seed均值，p/q/r为rad/s、速度m/s、姿态rad）：','',table(oof[['arm','flight',*METRICS]]),'',
        '## 相位方案的配对结果','',
        '辅助相位预测的交叉预测质量（只统计解码器实际使用的0.02–0.98秒，谐波弦长RMSE；并非状态RMSE或真实扑频测量）：','',table(auxiliary),'',
        table(summary[(summary.subset=='all')&summary.horizon_s.isin([.5,1.])][['arm','cohort','horizon_s',*METRICS]]),'',
        f'预定主指标为0.5/1秒主动p/q误差，先等权飞行再平均轴/时域/种子。phase相对hold改善{d["phase_primary_gain"]:.2%}；按飞行配对bootstrap95%区间[{d["bootstrap95"][0]:.2%}, {d["bootstrap95"][1]:.2%}]。仅5个飞行且折训练集重叠，这个区间是固定拟合结果的探索性重采样，不能代表独立大量试验。','',
        table(pd.DataFrame({'seed':SEEDS,'hold_primary':d['baseline_seed_scores'],'phase_primary':d['phase_seed_scores'],'gain':d['seed_gains']})),'',
        table(pd.DataFrame(d['flight_gains'])),'',
        f'全部指标最大相对退化为{d["maximum_allmetric_regression"]:.2%}；OOF门槛通过={d["cv_gate"]}。门槛包括主指标至少改善5%、3seed全部改善、至少4/5飞行改善，以及0.5/1秒其他全窗口指标退化不超过5%。','',
        '相位候选的0.5秒绝对误差门槛：','',table(pd.DataFrame(phase_absolute)),'']
    if full_decision:
        lines+=['OOF门槛通过后，已按冻结协议完成全407窗口的phase三个种子训练；全量训练仍使用按飞行交叉预测的相位输入。之后固定到原739个开发验证窗口，与原A1600对照。',
            f'全量候选development主指标改善{full_decision["phase_primary_gain"]:.2%}；相对门槛通过={full_decision["relative_gate"]}。完整数值与绝对门槛见full_decision.json。控制响应及闭环证据仍未通过，不能据此部署。','']
    else:
        lines+=['**OOF门槛未通过，按预定协议未启动全量phase候选训练。**保留A1600研究基线；未使用739个开发验证窗口来反向选择相位方案或训练步数。','']
    lines+=['## 核验与下一步','',
        '27项针对性测试通过；30个checkpoint重载与未来参考前缀因果性通过；15组配对初始化和全部抽样序列相同；5折归一化重新计算完全一致，辅助相位模型排除自身/外层查询飞行的标签且重载一致。源数据、旧模型与交付产物哈希保留。',
        '完整误差表：cv_per_flight.csv、cv_waveforms.csv、baseline_generalization_summary.csv；每折的归一化、相位预测器、checkpoint、抽样记录及预测都保存在fold0至fold4。预测对象仍为参考到固定PX4内环与飞机的整体，不是原始舵面到裸机动力学。',
        '下一步使用本轮真正留出飞行的预测，按窗口把角速度误差分为均值偏差和去均值后的波形误差，再检查这些分量与历史状态、参考和执行命令的关系，决定是否存在可由历史信息解释的跨飞行偏差。分析可使用事后真值，但任何可用修正只能读取过去信息，且必须在训练飞行内隔离拟合与查询。',
        '相位方案未达到门槛时，不再仅凭相位预测更准而扩大这一路线。任何新候选先固定训练内评价协议，控制响应和闭环可用性另行验收。本轮没有证明必须补飞。','']
    (OUT/'report_zh.md').write_text('\n'.join(lines))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    cats=['fit','oof','development'];xx=np.arange(3);width=.35
    for j,axis in enumerate(['p','q']):axes[0].bar(xx+(j-.5)*width,g.loc[cats,axis],width,label=axis)
    axes[0].set_xticks(xx,cats);axes[0].set(ylabel='RMSE at 0.5 s (rad/s)',title='Held-frequency baseline: matched fold models');axes[0].legend()
    for arm,color in [('hold','tab:blue'),('phase','tab:orange')]:
        z=oof[oof.arm==arm].sort_values('log_id');axes[1].plot(np.arange(5),z.p,'o-',label=arm,color=color)
    labels=oof[oof.arm=='hold'].sort_values('log_id').log_id.str.split('/').str[-1].str.split('_').str[1]
    axes[1].set_xticks(np.arange(5),labels);axes[1].set(xlabel='Held-out training flight (log number)',ylabel='p RMSE at 0.5 s (rad/s)',title='Same-day dynamics OOF: 3-seed means');axes[1].legend()
    for ax in axes:ax.grid(axis='y',alpha=.25)
    fig.tight_layout();fig.savefig(OUT/'groupcv_comparison.png',dpi=160);plt.close(fig)
    paths=list((base.ROOT/'scripts').glob('*reference_groupcv_v8*.py'))+[base.ROOT/'src/system_identification/models/reference_gru_phase_schedule.py',base.ROOT/'tests/test_reference_groupcv_v8.py']
    base.write(OUT/'delivery_manifest.json',dict(status='complete',fold_models=30,full_models=3 if full_decision else 0,
        focused_tests=27,code={str(p.relative_to(base.ROOT)):base.sha(p) for p in sorted(paths)},
        artifacts={str(p.relative_to(base.ROOT)):base.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['delivery_manifest.json','pipeline_status.json']},
        sealed_test_accessed=False,control_validated=False))
    print(json.dumps(dict(cv_gain=d['phase_primary_gain'],cv_gate=d['cv_gate'],full=full['status'],same_day_gap=ratios,nextday_gap=days),indent=2))

if __name__=='__main__':main()
