"""Validation-only evaluation and descriptive K25/K50 comparison."""
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run_paper_rollout_horizon_ablation as r
from system_identification.training.trajectory_main_v1 import TorchTrajectoryPrediction
m=r.m;OUT=r.OUT;ART=r.ART
METRICS=m.METRICS

def common_objectives(pred,truth,source):
    # Saved float32 predictions, identical original loss implementation, no inference.
    result={}
    for k in (25,50):
        total=0.
        for start in range(0,len(pred.position_n),128):
            def convert(obj):
                return TorchTrajectoryPrediction(*(torch.as_tensor(getattr(obj,key)[start:start+128],dtype=torch.float32) for key in vars(truth)))
            p,t=convert(pred),convert(truth)
            total+=float(r.objective(p,t,k,source))*len(p.position_n)
        result[k]=total/len(pred.position_n)
    return result

def summarize(per,summary):
    long=summary.melt(id_vars=['model','seed','cohort','horizon_s'],value_vars=METRICS,var_name='metric',value_name='value')
    multi=long.groupby(['model','cohort','horizon_s','metric']).value.agg(['mean','std','min','max']).reset_index()
    keys=['seed','cohort','horizon_s','metric']
    pair=long.pivot(index=keys,columns='model',values='value').reset_index()
    pair['paired_error_delta']=pair.K25-pair.K50
    pair['absolute_gain']=-pair.paired_error_delta
    pair['relative_gain_pct']=100*pair.absolute_gain/pair.K50.replace(0,np.nan)
    f=per.melt(id_vars=['model','seed','cohort','log_id','horizon_s','n_windows'],value_vars=METRICS,var_name='metric',value_name='value')
    fp=f.groupby(['model','cohort','log_id','horizon_s','metric']).value.mean().unstack('model').reset_index()
    fp['paired_error_delta']=fp.K25-fp.K50;fp['absolute_gain']=-fp.paired_error_delta
    means=multi.pivot(index=['cohort','horizon_s','metric'],columns='model',values='mean').reset_index()
    means['absolute_gain']=means.K50-means.K25
    means['relative_gain_pct']=100*means.absolute_gain/means.K50.replace(0,np.nan)
    return multi,pair,fp,means

def markdown(frame):
    def fmt(v): return f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v)
    return '| '+' | '.join(map(str,frame.columns))+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))

def figures(multi,fp):
    for metric in METRICS:
        fig,axes=plt.subplots(1,3,figsize=(12,3.5))
        for ax,cohort in zip(axes,['ALL','Sep7','Sep17']):
            for model in ('K25','K50'):
                g=multi.query('metric == @metric and cohort == @cohort and model == @model').sort_values('horizon_s')
                ax.errorbar(g.horizon_s,g['mean'],yerr=g['std'],marker='o',capsize=3,label=model)
            ax.set(title=cohort,xlabel='Nominal horizon [s]',ylabel=metric);ax.grid(alpha=.2);ax.legend()
        fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(OUT/f'error_vs_horizon_{metric}.{ext}',dpi=180)
        plt.close(fig)
    for h in (.5,1.):
        fig,axes=plt.subplots(3,1,figsize=(11,8))
        for ax,metric in zip(axes,METRICS[1:]):
            g=fp.query('horizon_s == @h and metric == @metric').sort_values(['cohort','log_id'])
            ax.bar(np.arange(len(g)),g.paired_error_delta,color=['C0' if c=='Sep7' else 'C1' for c in g.cohort])
            ax.axhline(0,color='black',lw=.8);ax.set(ylabel=metric,title=f'{h:g} s: K25 − K50; flight means over three seeds')
            ax.set_xticks(range(len(g)),[f'{c}-{i+1}' for i,c in enumerate(g.cohort)],rotation=60)
        fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(OUT/f'paired_flights_{h:g}s.{ext}',dpi=180)
        plt.close(fig)

def reports(multi,pair,fp,gains):
    tables={}
    for h,name in ((.5,'500ms'),(1.,'1s')):
        rows=[]
        for model in ('K25','K50'):
            row={'model':model}
            for metric in METRICS:
                g=multi.query('model == @model and cohort == "ALL" and horizon_s == @h and metric == @metric').iloc[0]
                row[metric]=f'{g["mean"]:.4f} ± {g["std"]:.4f}'
            rows.append(row)
        tables[h]=pd.DataFrame(rows);tables[h].to_csv(OUT/f'paper_table_{name}.csv',index=False)
    directions=[]
    for h in (.1,.2,.5,1.):
        for cohort in ('ALL','Sep7','Sep17'):
            for metric in METRICS:
                s=pair.query('horizon_s == @h and cohort == @cohort and metric == @metric')
                f=fp.query('horizon_s == @h and metric == @metric')
                if cohort!='ALL': f=f[f.cohort==cohort]
                directions.append(dict(horizon_s=h,cohort=cohort,metric=metric,seed_wins=int((s.paired_error_delta<0).sum()),seed_n=len(s),flight_wins=int((f.paired_error_delta<0).sum()),flight_n=len(f)))
    d=pd.DataFrame(directions);d.to_csv(OUT/'direction_summary.csv',index=False)
    primary=gains.query('cohort == "ALL" and horizon_s == .5 and metric != "position_rmse_m"')
    extended=gains.query('cohort == "ALL" and horizon_s == 1. and metric != "position_rmse_m"')
    results='; '.join(f'{x.metric}: {x.relative_gain_pct:+.2f}%' for x in primary.itertuples())
    costs='; '.join(f'{x.metric}: {x.relative_gain_pct:+.2f}%' for x in extended.itertuples())
    text=f'''# Training Rollout Horizon Ablation — 开发阶段验证结果

## 1. 研究定位与数据访问
本轮在已知旧独立测试结果后开展候选模型开发。prior_heldout_results_known=true；heldout_data_accessed_this_run=false。本轮仅显式加载原train/validation数据与K50 validation预测；未调用独立测试入口，也未读取Sep8/Sep19数据、预测或指标。不能称为新独立测试。H26/K50保持正式冻结主模型，旧history Mixed及future-control结论不修改。

## 2. 冻结比较
CausalHistoryTrajectoryModel / GRU64 / H26 / 21,383参数；同一41 flights、28,293 train origins及17 flights、2,582 validation origins（Sep7/Sep17）。使用原50步数据对象、历史与冻结normalization，仅新模型训练前向和监督取25步。K50三个seed只读复用；K25三个seed17/23/42均随机初始化，未从K50微调。

40 epochs AdamW lr3e-4 +25 epochs重建AdamW lr5e-4；batch256、weight decay1e-5、clip5，每epoch111更新，共7,215。base/continuation seed为s/s+12；last epoch65，无validation选模。相同optimizer预算不等于相同转移数、监督状态数、FLOPs或耗时，不据历史耗时计算严格加速率。

## 3. 损失范围及共同目标
完整公式见loss_definition.md。原位置/速度/姿态/角速度/相位/频率损失均取1..K平均，continuation额外0.2 frequency MSE同样取1..K；两步增量仍为t0到t2，原scales/weights不变。无teacher forcing、额外detach或第25步重置。各K原训练loss不可直接排名；common_validation_objectives.csv在相同validation上分别列L25与L50（含最终频率项），仅作辅助。

## 4. 500 ms ALL主结果
误差单位依次为m、m/s、deg、rad/s。先flight内vector RMSE/姿态geodesic RMS，再flight等权，再三seed mean±sample SD（ddof=1）。

{markdown(tables[.5])}

K25相对K50动力学改善率：{results}。正值表示K25更好。

## 5. 1 s扩展结果
所有K25均连续自主递推50步、返回含起点51状态，未读取第25步真值。

{markdown(tables[1.])}

动力学改善率：{costs}。完整绝对差值与相对差值见relative_improvement.csv；负改善率为退化，不省略。

## 6. Seed、flight、cohort与短时域
下表为三个seed均值下的物理误差改善率，逐seed差值见paired_seed_differences.csv；flight方向先取三seed误差均值，不代表全部窗口改善。

{markdown(gains)}

{markdown(d)}

不根据某个seed或cohort优势更换模型。不把三个seed SD视为置信区间，也不宣称统计显著；小差异需结合方向不一致和跨seed波动解释。

## 7. 可用于Results的表述
With architecture, history, normalization, training origins and optimizer-update budget fixed, we compared training unrolls of 25 and 50 native steps using three seeds. At the nominal 500 ms validation horizon, the relative error reductions of K25 were {results}. At 1 s, the corresponding reductions were {costs}. Both candidates were evaluated using uninterrupted 50-step autonomous prediction. These results describe training-horizon sensitivity under the current development protocol rather than a universally optimal unroll length.

## 8. Discussion与限制
不同K的原始训练目标覆盖范围不同，最终训练loss不能作为同口径优劣证据。固定更新数不意味着固定计算量；历史K50训练环境与本轮耗时不能严格比较。任何500 ms收益均需同时衡量1 s代价、cohort和flight反例；三seed为描述性重复。此次实验发生在已知Sep8/Sep19结果之后，不能据validation优势宣称新独立测试验证、长时稳定、闭环控制或RL改进；后续确认候选需优先使用预先保留的新飞行数据。

## 9. 工程验证与下一步
测试记录见tests.json与sanity_checks.json，训练记录及checkpoint hashes见training_runs.csv，输入范围和注册见protocol.json。无自动重跑、删seed或筛窗口。建议先审阅本表中的时域折中；结构解耦实验如获后续授权，应另行冻结单独研究问题、预算和新数据确认边界，不把本轮validation结果用于声称结构改进已获独立验证。本轮不执行下一实验，不切换主模型。
'''
    text += '\n## 10. 完成后结果解读（2026-09-26核验）\n本轮不支持将K25作为500 ms综合性能改进：速度与姿态分别退化3.70%和5.05%，角速度改善3.67%。500 ms三个配对seed均支持上述各指标方向；按flight三seed均值，K25速度仅3/17更好、姿态0/17更好、角速度16/17更好。角速度唯一反例是Sep17 log_13_06-52-30，差值仅+0.000125 rad/s，应保留但不夸大。\n\nSep7的500 ms速度差异仅-0.21%改善率，且2/3 seed方向反而支持K25，不应声称该cohort存在稳定速度劣势；Sep17速度退化7.30%，三个seed及8/8 flights均支持K50。姿态在Sep7/Sep17分别退化3.53%/6.41%，角速度分别改善5.10%/1.86%。\n\n100 ms三项动力学误差平均改善6.86%（速度）、4.22%（姿态）、4.82%（角速度）；200 ms对应5.03%、1.86%、5.40%。但200 ms姿态在Sep17退化1.33%，不能将ALL均值改善写成两个cohort全面改善。\n\n1 s三项均退化：速度+0.06909 m/s（10.67%），姿态+0.58930 deg（8.81%），角速度+0.01170 rad/s（1.88%）。三个seed的ALL方向一致；flight均值下速度/姿态17/17支持K50，角速度14/17支持K50。Sep7的1 s角速度差异仅0.26%，不能夸大这一小差异。\n\n因此，这不是“500 ms三项一起改善、1 s退化”，而是100–200 ms平均动力学收益及500 ms角速度局部收益，对应500 ms速度/姿态及1 s表现的代价。继续保留H26/K50更符合当前500 ms主目标，不建议仅凭本轮结果优先推进K25或切换主模型。\n\n共同验证目标三seed均值：K25/K50的L25分别为0.227508/0.246912，L50为0.381655/0.379568。这说明更低的区间平均L25并不保证500 ms端点速度和姿态更低；损失混合不同状态、相位与频率，且其窗口等权口径与主指标flight等权不同，不能替代物理单位结果。\n\n工程状态：三个K25各完成65 epochs/7,215 updates；独立复算1,632个per-flight端点指标单元通过，所有显式登记输入与输出hash匹配。复用未变化代码的22项通过测试。训练日志沿用原helper，逐epoch保存total、original_loss聚合项、两项增量、梯度范数及峰值显存；original_loss内部六项及额外频率MSE没有逐epoch单独列出，这是日志粒度限制，不能宣称保存了每个内部子项的独立曲线。未为补日志重训。\n\n建议下一项结构解耦研究若获授权，继续以K50为冻结对照，单独预注册结构变量和预算，不将K25训练范围与结构改变同时混入比较；候选确认优先采用预先保留的新飞行数据。本轮不启动该实验。\n'
    (OUT/'review_report.md').write_text(text);(OUT/'report.md').write_text(text)

def evaluate():
    protocol=m.read_json(OUT/'protocol.json');m.verify_pins(protocol['frozen_input_sha256'])
    source,batches,stats=r.inputs();b=batches['validation'];rows=[];objectives=[];runs=[];checks={}
    for k in (25,50):
        for s in r.SEEDS:
            folder=ART/f'seed{s}';path=folder/'model.pt' if k==25 else m.checkpoint_path(s)
            cp=torch.load(path,map_location='cpu',weights_only=False)
            assert m.normalization_hash({key:cp['state_dict'][key].numpy() for key in m.STATS})==source['normalization_sha256']
            model=m.build_model(s,stats);model.load_state_dict(cp['state_dict'],strict=True)
            if k==25:
                assert cp['final_epoch']==65 and cp['protocol_sha256']==m.file_hash(OUT/'protocol.json')
                checks[f'K{k}_seed{s}']=r.causality_check(model,m.subset(b,3),r.DEVICE)
                pred=r.predict_history_trajectory_model(model,b,use_history=True,batch_size=128,device=r.DEVICE)
                np.savez_compressed(folder/'predictions.npz',**vars(pred),window_ids=b.trajectory.window_ids.astype(str))
                hist=[]
                for stage,n,*_ in m.stages(s):
                    h=pd.read_csv(folder/f'{stage}_history.csv');m.validate_history(h,n);hist.append(h)
                runs.append(dict(train_unroll_steps=k,seed=s,checkpoint_path=m.rel(path),sha256=m.file_hash(path),status='complete',final_epoch=65,optimizer_updates=7215,final_train_loss=hist[-1].loss.iloc[-1],training_time_s=sum(h.wall_time_s.iloc[-1] for h in hist)))
            else:
                with np.load(r.predpath(s),allow_pickle=False) as f:
                    np.testing.assert_array_equal(f['window_ids'],b.trajectory.window_ids)
                    pred=r.TrajectoryPrediction(**{key:f[key] for key in vars(b.trajectory.truth)})
                old=pd.read_csv(m.OUT/'training_runs.csv').set_index('seed').loc[s]
                runs.append(dict(train_unroll_steps=k,seed=s,checkpoint_path=m.rel(path),sha256=m.file_hash(path),status='reused',final_epoch=65,optimizer_updates=7215,final_train_loss=old.final_train_loss,training_time_s=old.training_time_s))
            frame=r.endpoint_metrics(pred,b.trajectory,model=f'K{k}',seed=s);rows.append(frame)
            for steps,value in common_objectives(pred,b.trajectory.truth,source).items():
                objectives.append(dict(train_unroll_steps=k,seed=s,validation_objective_steps=steps,loss=value,aggregation='equal-origin frozen final-stage objective'))
    per,summary,_=r.aggregate_flights(pd.concat(rows,ignore_index=True))
    expected=pd.read_csv(m.OUT/'per_seed_summary.csv').sort_values(['seed','cohort','horizon_s'])
    actual=summary[summary.model=='K50'].sort_values(['seed','cohort','horizon_s'])
    np.testing.assert_allclose(actual[METRICS],expected[METRICS],rtol=1e-11,atol=1e-11)
    timing=[]
    for h,k in r.m.HORIZONS.items():
        t=b.trajectory.dt_s[:,:k].sum(1)
        timing.append(dict(horizon_s=h,native_steps=k,mean_s=float(t.mean()),median_s=float(np.median(t)),p05_s=float(np.quantile(t,.05)),p95_s=float(np.quantile(t,.95)),min_s=float(t.min()),max_s=float(t.max())))
    pd.DataFrame(timing).to_csv(OUT/'validation_horizon_timing.csv',index=False)
    per.to_csv(OUT/'per_flight.csv',index=False);summary.to_csv(OUT/'per_seed_summary.csv',index=False)
    pd.DataFrame(objectives).to_csv(OUT/'common_validation_objectives.csv',index=False)
    pd.DataFrame(runs).to_csv(OUT/'training_runs.csv',index=False)
    multi,pair,fp,gains=summarize(per,summary)
    for name,df in [('multiseed_summary',multi),('paired_seed_differences',pair),('paired_flight_differences',fp),('relative_improvement',gains)]:df.to_csv(OUT/f'{name}.csv',index=False)
    figures(multi,fp);reports(multi,pair,fp,gains)
    sanity=m.read_json(OUT/'sanity_checks.json');sanity.update(final_checks=checks,K50_metric_parity=True,all_prediction_counts=2582,normalization_identical=True,heldout_data_accessed_this_run=False)
    r.write_json(OUT/'sanity_checks.json',sanity)
    m.verify_pins(protocol['frozen_input_sha256'])
    r.write_json(OUT/'completion.json',dict(status='complete',new_training_runs=3,reused_K50=3,prior_heldout_results_known=True,heldout_data_accessed_this_run=False,
        artifact_sha256={m.rel(p):m.file_hash(p) for p in OUT.iterdir() if p.is_file() and p.name!='completion.json'},
        checkpoint_sha256={x['checkpoint_path']:x['sha256'] for x in runs},
        source_sha256={m.rel(p):m.file_hash(p) for p in [Path(__file__),r.ROOT/'scripts/run_paper_rollout_horizon_ablation.py',r.ROOT/'tests/test_paper_rollout_horizon_ablation.py']},
        prediction_sha256={m.rel(ART/f'seed{s}/predictions.npz'):m.file_hash(ART/f'seed{s}/predictions.npz') for s in r.SEEDS}))
    print('COMPLETE',flush=True)

if __name__=='__main__':
    import traceback
    m.configure()
    try:
        evaluate()
    except BaseException:
        r.write_json(OUT/'evaluation_failure.json',dict(status='failed',error=traceback.format_exc(),automatic_retry=False))
        raise
