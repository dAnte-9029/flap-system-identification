"""Summarize fixed v4 experiments without choosing checkpoints or changing gates."""
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_control_windows_v2 as base
from audit_reference_v4 import OUT
from train_reference_unit_gravity import ART,model_from
from system_identification.models.reference_rate_readout import causal_average_rates
from control_windows_v3 import predict
from evaluate_control_windows_v2 import METRICS
DEST=OUT/'slow_objective'


def markdown_table(frame,index=True):
    """Render simple numeric report tables without an optional tabulate install."""
    data=frame.reset_index() if index else frame
    def cell(value):
        if isinstance(value,(float,np.floating)):return f'{value:.6g}'
        return str(value).replace('|','\\|')
    rows=['| '+' | '.join(map(str,data.columns))+' |','| '+' | '.join(['---']*len(data.columns))+' |']
    rows.extend('| '+' | '.join(cell(v) for v in row)+' |' for row in data.itertuples(index=False,name=None))
    return '\n'.join(rows)


def main():
    torch.set_num_threads(1)
    assert json.loads((DEST/'status.json').read_text())['status']=='complete'
    d=pd.read_csv(DEST/'per_flight.csv');dd=d[(d.partition=='validation')&(d.readout=='causal_mean_200ms')&d.horizon_s.isin([.5,1.])]
    agg=d.groupby(['arm','partition','readout','subset','horizon_s'])[METRICS].mean();agg.to_csv(OUT/'model_comparison.csv')
    allrows=dd[dd.subset=='all'];allseed=allrows.groupby(['arm','seed','horizon_s'])[METRICS].mean()
    primary=[];absolute=[]
    for arm in ['A1600','slow']:
        for seed in [17,23,42]:
            vals=[];axisvals={}
            for subset,axis in [('roll_active','p'),('pitch_active','q')]:
                for horizon in [.5,1.]:
                    value=float(dd[(dd.arm==arm)&(dd.seed==seed)&(dd.subset==subset)&(dd.horizon_s==horizon)][axis].mean());vals.append(value)
                    if horizon==.5:axisvals[axis]=value
            primary.append(dict(arm=arm,seed=seed,score=np.mean(vals)))
            aa=allseed.loc[(arm,seed,.5)];absolute.append(dict(arm=arm,seed=seed,active_p=axisvals['p'],active_q=axisvals['q'],attitude_deg=float(np.rad2deg(aa.attitude)),velocity=float(aa.velocity),
                passed=bool(axisvals['p']<=.15 and axisvals['q']<=.15 and aa.attitude<=np.deg2rad(3) and aa.velocity<=.5)))
    pp=pd.DataFrame(primary).pivot(index='seed',columns='arm',values='score');pp['gain_fraction']=1-pp.slow/pp.A1600;pp.to_csv(OUT/'primary_scores.csv')
    degradation=(allseed.loc['slow']/allseed.loc['A1600']-1);degradation.to_csv(OUT/'relative_regressions.csv')
    mean_gain=float(1-pp.slow.mean()/pp.A1600.mean());relative=mean_gain>=.05 and (pp.gain_fraction>0).all() and (degradation<=.05).all().all()
    absolute_pass=all(a['passed'] for a in absolute if a['arm']=='slow')
    # Paired cluster bootstrap: flight is the resampling unit; seeds kept paired.
    flights=sorted(dd.log_id.unique());cube=np.full((2,3,len(flights),4),np.nan)
    for ai,arm in enumerate(['A1600','slow']):
        for si,seed in enumerate([17,23,42]):
            for fi,flight in enumerate(flights):
                for ki,(subset,axis,hor) in enumerate([(s,a,h) for s,a in [('roll_active','p'),('pitch_active','q')] for h in [.5,1.]]):
                    z=dd[(dd.arm==arm)&(dd.seed==seed)&(dd.log_id==flight)&(dd.subset==subset)&(dd.horizon_s==hor)]
                    if len(z):cube[ai,si,fi,ki]=z.iloc[0][axis]
    rng=np.random.default_rng(904);boot=[]
    for _ in range(2000):
        ids=rng.integers(0,len(flights),len(flights));scores=np.nanmean(np.nanmean(cube[:,:,ids,:],axis=2),axis=(1,2));boot.append(1-scores[1]/scores[0])
    ci=np.quantile(boot,[.025,.975]).tolist()
    wa=pd.read_csv(DEST/'nominal_waveforms.csv');chosen=wa[(wa.readout=='causal_mean_200ms')&wa.axis.isin(['p','q'])].copy()
    chosen['active']=np.where(chosen.axis=='p',chosen.roll_range_rad>=np.deg2rad(1),chosen.pitch_range_rad>=np.deg2rad(1))
    # Keep weak-response records. Peak timing is reported only as a descriptive tag,
    # not a reconstructed servo delay or proof of a causal reference response.
    fields=['waveform_rmse','amplitude_error','peak_time_error_s']
    chosen[chosen.active].groupby(['arm','seed','log_id','axis','horizon_s'])[fields].mean().groupby(['arm','axis','horizon_s']).mean().to_csv(OUT/'nominal_response_summary.csv')
    decision=dict(primary_gain_fraction=mean_gain,paired_flight_bootstrap_gain_95percent=ci,relative_gate_passed=bool(relative),absolute_by_seed=absolute,
        absolute_slow_gate_passed=bool(absolute_pass),causal_control_response_validated=False,optimizer_started=False,
        reason='No independent causal response certification; absolute/relative gates must also pass before bounded optimizer.',
        main_horizon_s=[.5,1.],long_free_running_used_as_nominal_gate=False,instantaneous_model_replaced=False,sealed_test_accessed=False)
    base.write(OUT/'decision.json',decision)
    # Reload/finiteness check of every delivered checkpoint, also verify train-only normalization exactly.
    x,y,stats=base.load();w=pd.read_csv(base.DATA/'windows_v2.csv');train=w[w.partition=='train'];items={k:[] for k in ['state','reference','actuator','rates']}
    for cache,g in train.groupby('cache'):
        with np.load(base.ROOT/cache) as z:
            hi=np.unique(np.concatenate([np.arange(i-25,i+1) for i in g.origin_index]));ri=np.unique(np.concatenate([np.arange(i-25,i+50) for i in g.origin_index]))
            items['state'].append(z['state_features'][hi]);items['reference'].append(z['references'][ri]);items['actuator'].append(z['actuator_commands'][hi-1])
            items['rates'].extend([z['rates_b'][i:i+51] for i in g.origin_index])
    recomputed={}
    for k in ['state','reference','actuator']:
        z=np.concatenate(items[k]).astype(float);recomputed[k+'_mean']=z.mean(0);recomputed[k+'_scale']=z.std(0).clip(1e-4)
    recomputed['state_mean'][9:11]=0;recomputed['state_scale'][9:11]=1
    recomputed['rate_scale']=np.concatenate(items['rates']).std(0).clip(.1)
    recomputed['velocity_scale']=y['velocity_n'][train.array_index.to_numpy()].numpy().reshape(-1,3).std(0).clip(.5)
    for key in stats:np.testing.assert_allclose(stats[key],recomputed[key],rtol=1e-6,atol=1e-6)
    reload_diff=[];evaluation=w[w.partition=='validation'];ix=evaluation.array_index.to_numpy()[:3]
    for seed in [17,23,42]:
        baseline_manifest=json.loads((ART/f'A_long_{seed}/manifest.json').read_text())
        assert base.sha(ART/f'A_long_{seed}/final.pt')==baseline_manifest['checkpoint_sha256']
        m=model_from(DEST/f'slow_{seed}.pt');ppp=predict(m,x,ix)
        with np.load(DEST/f'slow_{seed}_validation_predictions.npz') as z:
            delta=max(float(abs(ppp[k]-z[k][:3]).max()) for k in ppp)
        assert delta<1e-5;reload_diff.append(delta)
    base.write(OUT/'verification.json',dict(normalization_reproduced_train_only=True,checkpoint_reload_max_abs=reload_diff,baseline_checkpoint_hashes_verified=True,focused_tests=18,sealed_test_accessed=False))
    # Time-selected example, not best response. Show raw and causal mean separately.
    j=int(evaluation.iloc[len(evaluation)//2].array_index);loc=int(np.flatnonzero(evaluation.array_index.to_numpy()==j)[0]);t=np.arange(1,51)*.02
    fig,axes=plt.subplots(2,2,figsize=(11,6),sharex=True)
    curves={'Observed':y['rates_b'][j:j+1].numpy()}
    for name,file in [('A1600','A1600_17_validation_predictions.npz'),('Slow objective','slow_17_validation_predictions.npz')]:
        with np.load(DEST/file) as z:curves[name]=z['rates_b'][loc:loc+1]
    for row,axid in enumerate([0,1]):
        for col,filtered in enumerate([False,True]):
            ax=axes[row,col]
            for name,z in curves.items():
                v=causal_average_rates(x['history_state'][j:j+1,:,3:6],torch.from_numpy(z)).numpy() if filtered else z
                ax.plot(t,v[0,:,axid],label=name,linewidth=1.2)
            ax.set_ylabel(('p' if axid==0 else 'q')+' (rad/s)');ax.grid(alpha=.3)
    axes[0,0].set_title('Instantaneous rate (diagnostic)');axes[0,1].set_title('Causal 0.2 s average (90 ms lag)');axes[0,0].legend(fontsize=8)
    for ax in axes[-1]:ax.set_xlabel('Forecast horizon (s)')
    fig.suptitle('Time-selected validation window: '+str(j));fig.tight_layout();fig.savefig(OUT/'example_waveforms.png',dpi=150);plt.close(fig)
    base.write(OUT/'plot_provenance.json',dict(array_index=j,window_id=w.iloc[j].window_id,selection='middle row of fixed validation table, not selected by error',seed=17))
    tiny=pd.read_csv(OUT/'tiny_per_flight.csv');tiny=tiny[(tiny.subset=='all')&(tiny.horizon_s==.5)].groupby(['arm','cohort'])[METRICS].mean()
    spec=pd.read_csv(OUT/'spectral_summary.csv');proj=pd.read_csv(OUT/'waveform_projection.csv')
    lines=['# Reference dynamics v4：契约、拟合能力与较慢状态目标','',
        '本轮使用原407训练窗口和739开发验证窗口；没有新增数据筛选、没有读取封存测试。验证集此前已多次查看，不能称为盲测。',
        '', '## 1. 契约审计', '',
        '1146窗口与14份源缓存逐项复核通过；单位重力、机体速度、四元数、20ms采样、H26历史和K50标签一致。14份原始ULog核对哈希后检查发布与采样时间：重采样未使用未来消息。训练归一化由5个训练日志独立重算一致。',
        '角速度发布间隔中位数约10ms、姿态与参考约20ms。发布/采样时差不是舵机物理延迟。日志相位物理零点仍未认证；记录参数相同不能证明硬件完全等价。',
        '', '## 2. 小样本与误差诊断', '',
        '预先按时间索引选8个训练窗口、固定1600次更新，仅作记忆能力检查：', '',markdown_table(tiny),'',
        '8窗口拟合后的滚转误差仍较大；不能仅把问题归因于跨飞行泛化，也不能据此证明网络容量绝对不足。',
        f"训练p误差前两阶相位谐波投影比例中位数为{proj[(proj.partition=='train')&(proj.axis=='p')].harmonic_fraction.median():.1%}，未达到预定50%触发条件，因此未训练谐波候选。投影是事后描述，不是因果解释率。",
        '1秒记录频率分辨率为1Hz；训练p误差中>8Hz能量比例中位数约61%，验证约59%。这不证明高频成分全是噪声。简单相位外推表现更差，即使提供真实未来相位也未解决问题。',
        '追加100Hz日志角速度重采样诊断发现：p在25Hz以上的能量比例中位数训练18.2%、验证15.7%；q约0.6%/0.7%。当前50Hz缓存直接保持最近消息，没有降采样前抗混叠处理，存在高频混叠可能。100Hz诊断本身仍受日志采样和保持影响，不能把这些比例直接解释成50Hz误差的因果贡献。契约逐项一致并不意味着信号带宽设计已充分。',
        '', '## 3. 单因素对照', '',
        '保留GRU64、全部训练样本、归一化、种子17/23/42、1600次更新及其他损失，只把角速度损失改为10点因果平均。滤波器由已观测历史初始化；未来参考仍是唯一未来外部输入。原始潜变量仍参与姿态积分。',
        '基线与新模型都通过相同读出评价。新目标输出是延迟约90ms的平均角速度；不能将其称为瞬时角速度模型，也不能直接接入要求瞬时状态的控制器。',
        '', '验证集同目标、全有效窗口的0.5/1秒结果（跨种子与飞行平均；角速度rad/s，姿态rad）：','',
        markdown_table(agg.loc[(slice(None),'validation','causal_mean_200ms','all',slice(None)),:]),'',
        f'主动滚转/俯仰、0.5/1秒联合主指标改善{mean_gain:.2%}；按飞行成对bootstrap的95%区间[{ci[0]:.2%}, {ci[1]:.2%}]。相对门槛通过：{bool(relative)}。',
        '',markdown_table(pd.DataFrame(absolute),index=False),'',
        '原始瞬时角速度结果另存model_comparison.csv，不与平均状态的误差混为一谈。峰值时刻差只是波形指标，不是执行器延迟。所有弱响应窗口保留。',
        '', '## 4. 能力结论与下一步', '',
        f'较慢状态绝对研究门槛通过：{bool(absolute_pass)}。尚无独立因果控制响应认证，因此没有启动参考优化或真实控制部署。大于1秒自由滚动不再用作本轮短时模型的主验收依据。',
        '下一步先做采样带宽的受控对照：同一批原始片段，比较保留100Hz角速度与经明确因果抗混叠后降到50Hz的目标，记录滤波状态、有效时延和原生发布频率；较慢量不能靠插值当成100Hz测量。先复用8窗口拟合诊断，再决定正式三种子重训。当前慢损失候选若未改善就不晋升，不继续叠加结构，也不删除弱响应样本。',
        '即使某一平均状态目标改善，也仍需限定时域内参考响应的独立支持、时延建模和鲁棒性验证，才能进入控制器设计。旧日志能支持到哪一级，以证据为准；本轮没有证明必须补飞。','']
    (OUT/'report_zh.md').write_text('\n'.join(lines))
    base.write(OUT/'delivery_manifest.json',dict(status='complete',artifacts={str(p.relative_to(base.ROOT)):base.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'},
        code={str(p.relative_to(base.ROOT)):base.sha(p) for p in [base.ROOT/'scripts/audit_reference_v4.py',base.ROOT/'scripts/audit_reference_v4_bandwidth.py',base.ROOT/'scripts/diagnose_reference_v4_spectrum.py',base.ROOT/'scripts/train_reference_v4_slow.py',base.ROOT/'scripts/report_reference_v4.py',base.ROOT/'src/system_identification/models/reference_gru_unit_gravity.py',base.ROOT/'src/system_identification/models/reference_rate_readout.py',base.ROOT/'src/system_identification/data/reference_windows.py',base.ROOT/'tests/test_reference_v4.py']}))
if __name__=='__main__':main()
