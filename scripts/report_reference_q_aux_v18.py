"""Audit and report the fixed q-auxiliary experiment without changing its gates."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_control_windows_v2 as base
from reference_q_aux_v18 import OUT,V8,AUX_WEIGHT,METRICS,loss_terms
from reference_groupcv_v8 import load_train,model_from,predict,SEEDS
from run_reference_generalization_v7 import verify_manifest
from report_reference_generalization_v7 import table


def main():
    torch.set_num_threads(1);verify_manifest(OUT/'protocol.json',['source_pins'])
    amendment=json.loads((OUT/'evaluation_amendment.json').read_text())
    assert base.sha(base.ROOT/amendment['corrected_path'])==amendment['corrected_sha256']
    decision=json.loads((OUT/'decision.json').read_text());s=pd.read_csv(OUT/'summary.csv')
    source=pd.read_csv(OUT/'source_flight.csv');inc=pd.read_csv(OUT/'increment_summary.csv')
    ages=pd.read_csv(OUT/'age_summary.csv');sat=pd.read_csv(OUT/'saturation.csv')
    x,y,w=load_train();checks=[]
    for fold in range(5):
        info=json.loads((V8/f'fold{fold}/manifest.json').read_text());ids=np.array(info['fit_indices'])[:3]
        query=np.array(info['query_indices'])[:64]
        for seed in SEEDS:
            folder=OUT/f'fold{fold}/seed{seed}';status=json.loads((folder/'status.json').read_text())
            assert status['status']=='complete' and status['step']==1600
            assert base.sha(folder/'final.pt')==status['checkpoint_sha256']
            checkpoint=torch.load(folder/'final.pt',map_location='cpu',weights_only=False)
            assert checkpoint['train_unroll_steps']==50 and checkpoint['auxiliary_weight']==AUX_WEIGHT
            model=model_from(folder/'final.pt');pred=model(**{k:v[ids] for k,v in x.items()});target={k:v[ids] for k,v in y.items()}
            full,aux=loss_terms(pred,target,model)
            oracle=base.loss_fn(pred,target,model)+((pred['rates_b'][:,0,1]-target['rates_b'][:,0,1])/model.rate_scale[1]).square().mean()/3
            np.testing.assert_allclose(float((full+aux).detach()),float(oracle.detach()),atol=1e-7)
            # Reproduce held-reference saved predictions after independent reload.
            held={k:v.clone() for k,v in x.items()};held['future_reference']=held['future_reference'][:,:1].expand_as(held['future_reference']).clone()
            h=predict(model,held,query)
            with np.load(folder/'oof_held_reference_predictions.npz') as z:
                np.testing.assert_array_equal(z['array_index'][:len(query)],query)
                discrepancy=max(float(np.max(abs(h[k]-z[k][:len(query)]))) for k in h)
            assert discrepancy<1e-6
            curve=pd.read_csv(folder/'training_curve.csv');assert int(curve.step.iloc[-1])==1600
            np.testing.assert_allclose(curve.loss,curve.full_loss+curve.aux_loss,atol=1e-6)
            checks.append(dict(fold=fold,seed=seed,held_reference_reload_max_abs=discrepancy,checkpoint_hash_verified=True))
    old=pd.read_csv(OUT.parent/'control_windows_v17/summary.csv')
    a=s[s.arm=='baseline'].sort_values(['cohort','kind','horizon_s']);b=old[old.arm=='baseline'].sort_values(['cohort','kind','horizon_s'])
    np.testing.assert_allclose(a[METRICS],b[METRICS],atol=1e-12)
    first=source[(source.cohort=='oof')&(source.kind=='prefix')&(source.horizon_s==.02)&source.arm.isin(['aux','baseline'])]
    records=[]
    for key in ['log_id','seed']:
        frame=first.groupby([key,'arm'])[METRICS].mean().unstack('arm')
        for item,row in frame.iterrows():records.append(dict(group=key,identity=str(item),**{k:1-row[(k,'aux')]/row[(k,'baseline')] for k in METRICS}))
    repeated=pd.DataFrame(records);repeated.to_csv(OUT/'first_step_repeatability.csv',index=False)
    overall=[]
    for horizon in [.02,.1,.2,.5,1.]:
        g=s[(s.cohort=='oof')&(s.kind=='prefix')&(s.horizon_s==horizon)].set_index('arm')
        for metric in METRICS:overall.append(dict(horizon_s=horizon,metric=metric,baseline=g.loc['baseline',metric],aux=g.loc['aux',metric],gain=1-g.loc['aux',metric]/g.loc['baseline',metric]))
    overall=pd.DataFrame(overall);overall.to_csv(OUT/'overall_gains.csv',index=False)
    tests=Path('/tmp/reference-v18-tests.log').read_text();assert '6 passed' in tests and 'failed' not in tests
    (OUT/'tests.txt').write_text(tests)
    v=json.loads((OUT/'verification.json').read_text());v.update(focused_tests_passed=6,additional_checks=checks,
        v17_all_baseline_metrics_reproduced=True,total_optimizer_updates_including_identity=25600)
    base.write(OUT/'verification.json',v)
    fig,axes=plt.subplots(1,3,figsize=(12,3.7))
    for ax,axis in zip(axes,'pqr'):
        for arm,label in [('baseline','Full1s baseline'),('aux','Full1s + firstq auxiliary')]:
            g=ages[(ages.cohort=='oof')&(ages.arm==arm)]
            ax.plot(g.age_s,g[axis],label=label)
        ax.set(xlabel='Autonomous forecast age (s)',ylabel=f'{axis} RMSE (rad/s)',title=f'{axis}: equal-flight OOF');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'rollout_age.png',dpi=160);plt.close(fig)
    shown=s[(s.kind=='prefix')&s.horizon_s.isin([.02,1.])&s.arm.isin(['baseline','aux'])].copy();shown['attitude_deg']=np.rad2deg(shown.attitude)
    d=decision['diagnostic'];failed=pd.DataFrame(decision['failed_protection_checks'])
    ref=pd.DataFrame(json.loads((OUT/'reference_checks.json').read_text()))
    relative=bool(d['oof']['passes'] and decision['rollout_protection_passes'] and decision['reference_protection_passes'])
    verdict=dict(diagnostic_pass=d['oof']['passes'],state_protection_pass=decision['rollout_protection_passes'],reference_protection_pass=decision['reference_protection_passes'],all_relative_pass=relative,absolute_pass=decision['absolute_passes'],control_validated=False)
    base.write(OUT/'gate_summary.json',verdict)
    lines=['# 参考模型 v18：完整递推损失＋首步q辅助损失','',
        f"**15组候选与1组零权重复现全部完成。首20ms q的OOF改善为{100*d['oof']['primary_gain']:.2f}%；状态保护{'通过' if decision['rollout_protection_passes'] else '未通过'}，参考敏感性保护{'通过' if decision['reference_protection_passes'] else '未通过'}，绝对预测阈值{'通过' if decision['absolute_passes'] else '未通过'}。本轮未晋升控制模型。**",'',
        '## 冻结实验','',
        '仍使用407个原训练父窗口、5外层飞行折×种子17/23/42。保持H26、50Hz、K50、40070参数、同折归一化、初始参数、1600×32抽样索引、AdamW3e-4/weight_decay1e-4/梯度裁剪1和最终checkpoint规则。模型与推理路径、导数限幅均未变，没有新窗口、筛样本或未来测量输入。',
        '损失固定为 L = L原完整50步 + (1/3)·mean_batch[((q预测首步−q真实首步)/q_scale)²]。没有缩短完整递推或归一化总损失。额外项系数等于原损失中整段平均q项系数；逐点q平方误差系数从首步1/150变成51/150，其余49步仍为1/150，总名义q权重加倍。并非只给首步乘一个轻微权重，也不是已知最优系数；本轮不搜索其他系数。',
        '从v17的q线索预先选定单轴主指标，不再使用v17的p/q联合指标；这次切换写入训练前协议。其他轴仍受保护，共享网络可能改变全部输出。0权重复现fold0/seed17的最终参数必须与v8基线一致；实际最大差见verification.json。',
        '数据/模型查询边界不变：fit为训练内，OOF飞行排除于其模型与归一化，但已被多轮研究查看，不是盲测。未评价739个开发验证窗口，未访问sealed/reserved；未启动控制优化或实飞。','',
        '## 首步与完整1秒误差','',
        '先按飞行/模型计算RMSE，再等权平均源飞行与种子，fit重复模型先按源飞行平均。prefix=1s为完整50点误差而非终点；角速度rad/s，速度分量m/s，表中姿态deg。','',
        table(shown[['cohort','arm','horizon_s','p','q','r','velocity','attitude_deg']]),'',
        f"首20ms q：fit改善{100*d['fit']['primary_gain']:.2f}%，OOF改善{100*d['oof']['primary_gain']:.2f}%；OOF有{d['oof']['positive_flights']}/5飞行、{d['oof']['positive_seeds']}/3种子改善。预设主门槛为至少5%、全部种子同正、至少4飞行同正；结果为{'通过' if d['oof']['passes'] else '未通过'}。",'',
        table(repeated[['group','identity','p','q','r','velocity','attitude']]),'',
        '数值为相对原基线改善比例，负值表示退化；按种子/飞行分别报告，不能只挑改善的窗口。','',
        table(inc[inc.arm.isin(['baseline','aux'])][['cohort','arm','axis','std_ratio','correlation','sign_agreement']]),'',
        '增量幅值比与相关性为跨窗口20ms统计，方向统计沿用真实增量绝对值>0.01rad/s，近零预测计为未预测正确。它们不是执行器舵效或真实参考反事实的验证。','',
        '## 长期与其他状态保护','',
        '冻结保护共42项：3种子×5状态×0.5/1秒完整前缀共30项，另有首20ms的p/r/速度/姿态共12项，每项允许最多5%退化。全部输出有限；有限性只是数值检查，不是控制稳定性。',
        f'保护失败{len(failed)}/42项，详情如下：','',table(failed) if len(failed) else '无失败项。','',
        '![OOF预测年龄](rollout_age.png)','',
        '0.5s端点绝对阈值仍为每种子p/q/r≤0.15rad/s、速度≤0.5m/s、姿态≤3deg；结果保存在absolute_checks.csv。所有原预测和训练曲线保留，没有通过选择中间checkpoint或修改门槛获得结果。','',
        '## 参考输入敏感性保护','',
        '比较相同OOF起点下的日志计划参考与将5个参考通道保持首值的预测。q整段1秒参考收益定义为1−实际参考预测RMSE/保持参考预测RMSE，effect为两组预测差的RMS。按飞行等权，要求每种子收益下降不超过0.02、effect相对原模型比例在[0.9,1.1]。','',
        table(ref),'',
        '这只约束模型对已记录参考的敏感性和预测收益，不是真实飞行反事实证据；即使比例接近旧模型，也不能证明控制正确。','',
        '## 验证与交付','',
        '6项针对性测试通过，覆盖零辅助项恢复旧损失/梯度、首q直接梯度系数51及其余输出直接梯度不变、后续目标仍进入完整损失、稳定四元数角距离。16组完成检查、零权重全程训练复现、15组初始化/抽样/归一化配对、30组完整checkpoint重放、15组保持参考重载、参考前缀因果性、v17基线全部指标与源哈希核验完成。',
        '训练前冻结的评价脚本在静态复核时发现候选CSV列名沿用single而检查读取aux；保留原脚本，evaluation_amendment.json记录只改列名的更正脚本及哈希。修复发生于读取本轮结果前，未改指标、权重、门槛、数据或训练。',
        '候选在fold*/seed*/final.pt；零权重复现在fold0/identity；原始抽样、完整预测、参考保持预测、逐飞行/种子指标、协议与判定均随交付清单保存。','']
    note=OUT/'interpretation_zh.md'
    if note.exists():lines+=['## 结论与下一步','',note.read_text(),'']
    (OUT/'report_zh.md').write_text('\n'.join(lines))
    code=[base.ROOT/p for p in ['scripts/reference_q_aux_v18.py','scripts/prepare_reference_q_aux_v18.py','scripts/train_reference_q_aux_v18.py','scripts/run_reference_q_aux_v18.py','scripts/evaluate_reference_q_aux_v18.py','scripts/evaluate_reference_q_aux_v18_corrected.py','scripts/report_reference_q_aux_v18.py','tests/test_reference_q_aux_v18.py']]
    base.write(OUT/'delivery_manifest.json',dict(artifacts={str(p.relative_to(base.ROOT)):base.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'},code={str(p.relative_to(base.ROOT)):base.sha(p) for p in code}))
    verify_manifest(OUT/'delivery_manifest.json',['artifacts','code'])
    print('v18 report and hashes complete')


if __name__=='__main__':main()
