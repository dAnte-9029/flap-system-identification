"""Report paired target-time refresh diagnostics and inferential limitations."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_control_windows_v2 as base
from reference_groupcv_v8 import load_train,model_from,predict
from system_identification.data.reference_windows import extract_window
from run_reference_refresh_v15 import OUT,V8,V12,V14,METRICS
from run_reference_generalization_v7 import verify_manifest
from report_reference_generalization_v7 import table


def main():
    torch.set_num_threads(1);verify_manifest(OUT/'protocol.json',['source_pins'])
    for src in [V8,V12,V14]:verify_manifest(src/'delivery_manifest.json',['artifacts','code'])
    df=pd.read_csv(OUT/'prefix_per_window.csv');mse=[k+'_mse' for k in METRICS]
    # Keep fit models separate until after root-MSE; the same fit flight occurs in4folds.
    f=df.groupby(['arm','cohort','fold','seed','log_id','horizon_s'])[mse].mean().pow(.5).reset_index().rename(columns={k+'_mse':k for k in METRICS});f.to_csv(OUT/'prefix_per_flight.csv',index=False)
    summary=f.groupby(['arm','cohort','horizon_s'])[METRICS].mean().reset_index();summary.to_csv(OUT/'prefix_summary.csv',index=False)
    fs=df.groupby(['arm','cohort','fold','seed','log_id','stratum','horizon_s'])[mse].mean().pow(.5).reset_index().rename(columns={k+'_mse':k for k in METRICS});fs.to_csv(OUT/'strata_per_flight.csv',index=False)
    strata=fs.groupby(['arm','cohort','stratum','horizon_s'])[METRICS].mean().reset_index();strata.to_csv(OUT/'strata_summary.csv',index=False)
    ga=pd.read_csv(OUT/'global_age_per_flight.csv');gs=ga.groupby(['arm','cohort','stratum','global_age_s'])[METRICS].mean().reset_index();gs.to_csv(OUT/'global_age_summary.csv',index=False)
    la=pd.read_csv(OUT/'local_age_per_flight.csv');ls=la.groupby(['arm','cohort','stratum','period_s','local_age_s'])[METRICS].mean().reset_index();ls.to_csv(OUT/'local_age_summary.csv',index=False)
    flights=sorted(f.log_id.unique());oof=f[(f.cohort=='oof')&(f.horizon_s==1.)];comparisons=[];rng=np.random.default_rng(915)
    for period in [5,10]:
        cube=np.stack([oof[oof.arm==arm].pivot(index='seed',columns='log_id',values=METRICS).reindex(index=[17,23,42],columns=pd.MultiIndex.from_product([METRICS,flights])).to_numpy().reshape(3,5,5).transpose(0,2,1) for arm in ['free50',f'refresh{period}',f'persistence{period}']])
        mean=cube.mean(axis=(1,2));gain=1-mean[1]/mean[0];pers_gain=1-mean[1]/mean[2];fgain=1-cube[1].mean(0)/cube[0].mean(0);seedgain=1-cube[1].mean(1)/cube[0].mean(1)
        boot=[]
        for _ in range(2000):
            ix=rng.integers(5,size=5);m=cube[:,:,ix].mean(axis=(1,2));boot.append(1-m[1]/m[0])
        ci=np.quantile(boot,[.025,.975],axis=0)
        for j,key in enumerate(METRICS):comparisons.append(dict(period_s=period*.02,metric=key,free=float(mean[0,j]),refresh=float(mean[1,j]),persistence=float(mean[2,j]),gain_vs_free=float(gain[j]),gain_vs_persistence=float(pers_gain[j]),positive_flights=int((fgain[:,j]>0).sum()),positive_seeds=int((seedgain[:,j]>0).sum()),bootstrap_low=float(ci[0,j]),bootstrap_high=float(ci[1,j])))
    comparison=pd.DataFrame(comparisons);comparison.to_csv(OUT/'paired_diagnostic_gains.csv',index=False)
    x,y,w=load_train();caches={}
    for cache in w.cache.unique():
        with np.load(base.ROOT/cache) as z:caches[cache]={k:z[k] for k in z.files}
    checks=[]
    for fold in range(5):
        info=json.loads((V8/f'fold{fold}/manifest.json').read_text());ids=np.array(info['query_indices'])[:2]
        for seed in [17,23,42]:
            model=model_from(V8/f'fold{fold}/hold_{seed}/final.pt')
            for period,offset in [(5,15),(10,20)]:
                local=[]
                for idx in ids:
                    row=w.iloc[idx];inp,lab=extract_window(caches[row.cache],int(row.origin_index)+offset,horizon=period);local.append(inp)
                    for k,v in lab.items():np.testing.assert_array_equal(v,y[k][idx,offset:offset+period].numpy())
                xx={k:torch.tensor(np.stack([v[k] for v in local]),dtype=torch.float32) for k in local[0]};pred=predict(model,xx,np.arange(2))
                with np.load(OUT/f'fold{fold}/oof_refresh{period}_seed{seed}.npz') as z:
                    np.testing.assert_array_equal(z['array_index'][:2],ids);error=max(float(np.max(abs(pred[k]-z[k][:2,offset:offset+period]))) for k in pred)
                assert error<1e-5;checks.append(dict(fold=fold,seed=seed,period=period,offset=offset,reload_max_abs=error,labels_match_original_target_times=True))
    assert not df[df.cohort=='oof'].duplicated(['array_index','seed','arm','horizon_s']).any()
    test=Path('/tmp/reference-v15-tests.log').read_text();assert '12 passed' in test and 'failed' not in test;(OUT/'tests.txt').write_text(test)
    v=json.loads((OUT/'verification.json').read_text());v.update(focused_tests_passed=12,independent_block_reloads=checks,oof_unique_parent_windows=int(df[df.cohort=='oof'].array_index.nunique()),no_zero_error_observations_inserted=True);base.write(OUT/'verification.json',v)
    base.write(OUT/'decision.json',dict(stage_complete=True,diagnostic_only=True,comparisons=comparisons,autonomous_model_improved=False,control_validated=False,development_evaluated=False,sealed_test_accessed=False))
    shown=summary[(summary.cohort=='oof')&(summary.horizon_s==1.)].copy();shown['attitude_deg']=np.rad2deg(shown.attitude)
    rates=ls[(ls.cohort=='oof')&(ls.stratum=='all')&ls.arm.isin(['refresh5','refresh10'])]
    initial=gs[(gs.stratum=='all')&(gs.arm=='free50')&gs.global_age_s.isin([.02,.04,.1,.2])]
    qstrata=strata[(strata.cohort=='oof')&(strata.horizon_s==1)&strata.arm.isin(['free50','refresh5','refresh10'])]
    lines=['# 参考模型 v15：观测历史刷新与预测年龄诊断','',
        '**刷新观测明显降低姿态、速度和r误差，但没有解决p误差，q也只取得有限改善。原始首次20ms预测已有较大p/q误差，不能把主要问题只归于长时间递推。此次没有训练或晋升新模型。**','',
        '## 比较的是什么','',
        '保持407个非重叠父窗口、5外层飞行折、3个v8 hold种子及原归一化。free50为原1秒自主预测；refresh5/10分别每0.1/0.2秒用真实观测重新提取26点历史、重编码GRU、初始化物理状态并重置相对相位/扑频/空速，再向前预测5/10步。拼接后评价同一原始50个目标时刻。',
        '每个局部预测只看到自身起点及以前的测量，执行命令历史仍结束于起点前20ms，未来只给该段计划参考。第一项输出在新起点之后20ms，未把刚观测值当作零误差预测。所有新历史与标签均在原父窗口支撑内部，原始窗口及划分不变；新切出的重叠局部历史不是新增独立训练样本。',
        'persistence5/10接收同频率观测更新，但只保持最后的角速度、速度和四元数；persistence50只用最初观测。保持四元数不是物理积分模型，此基线只检查简单保持值是否已足够。未训练模型、未评价开发验证，sealed/reserved未访问。','',
        '## 相同目标时刻的整段误差','',
        '先在每飞行/种子求完整1秒目标上的RMSE，再等权平均。这里refresh对应多次短预测的拼接，不是具有初始一次观测的1秒自主预测。角速度rad/s、速度为分量均方根m/s、姿态为角距离deg。','',table(shown[['arm','p','q','r','velocity','attitude_deg']]),'',
        '相对原自主预测以及同刷新频率保持值的收益：','',table(comparison[['period_s','metric','gain_vs_free','gain_vs_persistence','positive_flights','positive_seeds','bootstrap_low','bootstrap_high']]),'',
        'Bootstrap仅按5个重复开发飞行配对重采样，不能当成新模型通过独立验收。刷新带来了额外观测，因此这些改善也不能归功于修改后的动力学模型。','',
        '## p/q局部预测仍有明显误差','',
        '按局部预测年龄分组（每段第一点仍需预测20ms，未计入刷新时刻的观测点）：','',table(rates[['arm','period_s','local_age_s','p','q','r']]),'',
        '0.1秒刷新后的首个20ms点，OOF p约0.542、q约0.276rad/s；随后q约0.34rad/s，p仍在0.52–0.57rad/s附近。将刷新间隔从0.2秒减到0.1秒没有使p/q误差持续下降。不同刷新周期的同一局部年龄对应不同全局目标集合，因此local_age_per_flight.csv同时保存自由预测在相同目标上的配对分数，不能随意横向比较两个周期的局部年龄均值。',
        '原始窗口起点的预测误差如下，可排除“只有新增刷新起点才出现短时误差”的解释：','',table(initial[['cohort','global_age_s','p','q','r']]),'',
        '原始首次20ms在OOF上p约0.523、q约0.280rad/s，训练内也约0.510/0.253rad/s；p的主要误差已存在于一步预测。该事实不证明噪声不可预测，也不能单独区分导数表达、历史编码、测量频带或初始化误差。',
        '刷新仍明显优于同频率保持当前角速度的基线，说明模型学到了一部分局部动态；但优于保持值不等于误差已满足控制需求。','',
        '## 扑频与训练内/留出差异','',table(qstrata[['arm','stratum','p','q','r','velocity','attitude']]),'',
        '低扑频q从约0.432下降至0.375rad/s（0.1秒刷新），仍高于高扑频约0.308。姿态、速度和r对刷新更敏感；该差异有助于定位，但不能把它解释成只有积分器出错。',
        '训练内按相同指标的结果：','',table(summary[(summary.cohort=='fit')&(summary.horizon_s==1)&summary.arm.isin(['free50','refresh5','refresh10'])][['arm','p','q','r','velocity','attitude']]),'',
        '训练内refresh的p/q不比原始自由预测更好。刷新起点没有作为原模型独立训练样本，同时刷新会改变编码历史、物理状态、相位锚点、扑频、观测新鲜程度和预测年龄；即使源飞行属于fit，也不能把所有新起点都叫做训练过的历史输入。这里既不能把改善归给积分漂移，也不能把变差完全归给跨飞行泛化。','',
        '## 结论与下一步','',
        '当前证据支持分轴处理：r、速度和姿态的长期误差可被观测刷新大幅限制；p/q存在显著局部预测误差，仅缩短自主滚动或更频繁刷新历史不能解决。此结论不是新控制器验证，也不是将刷新状态直接拼接成连续物理模拟的许可。',
        '下一步优先检查原始20ms角速度增量：比较真实与预测Δp/Δq的幅值、相关性、模型导数限幅占比，以及错误与已记录高频波动的关系；训练内和OOF同做，不删除大误差窗口。先确定局部输出表达或历史信息是否存在可修复问题，再考虑改变递推或观测更新结构，不再把1秒误差统一称为累积漂移。',
        '12项测试、60组原始前缀一致性、30次独立刷新块重载/目标时刻核对、全部15个模型参考前缀因果性与源产物哈希检查通过。完整预测、按全局/局部预测年龄的逐飞行结果、频率分层和delivery_manifest.json均保留；主基线不变。','']
    (OUT/'report_zh.md').write_text('\n'.join(lines))
    fig,axes=plt.subplots(1,3,figsize=(15,4.4))
    for ax,key in zip(axes,['p','q','r']):
        for arm in ['free50','refresh5','refresh10','persistence5']:
            a=gs[(gs.cohort=='oof')&(gs.stratum=='all')&(gs.arm==arm)];ax.plot(a.global_age_s,a[key],label=arm)
        ax.set(xlabel='Global target time (s)',ylabel=f'{key} RMSE (rad/s)',title='Same target samples; refreshed observations');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'refresh_age_comparison.png',dpi=150);plt.close(fig)
    paths=list((base.ROOT/'scripts').glob('*refresh_v15.py'))+[base.ROOT/'tests/test_reference_refresh_v15.py']
    base.write(OUT/'delivery_manifest.json',dict(status='complete',code={str(p.relative_to(base.ROOT)):base.sha(p) for p in paths},artifacts={str(p.relative_to(base.ROOT)):base.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'},focused_tests_passed=12,autonomous_model_improved=False,control_validated=False,development_evaluated=False,sealed_test_accessed=False))
    verify_manifest(OUT/'delivery_manifest.json',['artifacts','code']);print(comparison.to_string(index=False),flush=True)

if __name__=='__main__':main()
