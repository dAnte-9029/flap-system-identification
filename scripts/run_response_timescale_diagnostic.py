"""Native-time error decomposition and command-excitation coverage; no fitting."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/flap-paper-mpl')
import json,sys,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import run_model_flight_response_diagnostic as r
m=r.m;ROOT=r.ROOT
OUT=ROOT/'docs/analysis/results/response_timescale_diagnostic_v1'
TAUS=(.05,.10,.20)

def lowpass(x,dt,tau):
    if tau<=0 or x.shape[1]!=dt.shape[1]+1 or np.any(dt<=0):raise ValueError('invalid filter inputs')
    y=np.empty_like(x,dtype=float);y[:,0]=x[:,0]
    for k in range(dt.shape[1]):
        alpha=-np.expm1(-dt[:,k]/tau)
        y[:,k+1]=y[:,k]+alpha[:,None]*(x[:,k+1]-y[:,k])
    return y

def channel_scores(controls,scale):
    c=r.coordinates(controls[:,:25]);d=c-c[:,:1]
    score=np.sqrt(np.mean((d/scale)**2,axis=1))
    rms=np.sqrt(np.mean(d*d,axis=1));mean=d.mean(1)
    persistence=np.full(rms.shape,np.nan);np.divide(np.abs(mean),rms,out=persistence,where=rms!=0)
    return score,mean,persistence

def isolated_masks(score,q25,q75):
    result={}
    for j,ch in enumerate(r.CHANNELS):
        others=np.arange(4)!=j
        high=score[:,j]>q75[j]
        result[ch]={'high':high,'isolated':high&(score[:,others]<=q25[others]).all(1),
                    'predominant':high&(score[:,others]<=q75[others]).all(1)}
    return result

def prepare():
    assert not (OUT/'protocol.json').exists();OUT.mkdir(parents=True)
    sp,batches,stats=r.source.inputs();old=m.read_json(r.OUT/'protocol.json');m.verify_pins(old['frozen_input_sha256'])
    _,scale=r.activity(batches['train'].trajectory.controls,stats['control_std'])
    a,_,_=channel_scores(batches['train'].trajectory.controls,scale);q25,q75=np.quantile(a,[.25,.75],axis=0)
    thresholds=pd.DataFrame(dict(channel=r.CHANNELS,scale=scale,train_q25=q25,train_q75=q75));thresholds.to_csv(OUT/'channel_thresholds.csv',index=False)
    prev=pd.read_csv(r.OUT/'channel_thresholds.csv');np.testing.assert_allclose(q75,prev.train_q75,rtol=1e-12)
    pins=old['frozen_input_sha256'].copy()
    for path in [r.OUT/'origin_selection.csv',Path(__file__),ROOT/'tests/test_response_timescale_diagnostic.py']:
        pins[m.rel(path)]=m.file_hash(path)
    p=dict(experiment='response_timescale_diagnostic_v1',frozen_at_utc=pd.Timestamp.now(tz='UTC').isoformat(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        model='frozen StandardGRU/H26/K50',seeds=[17,23,42],frozen_input_sha256=pins,
        prior_heldout_results_known=True,heldout_data_accessed_this_run=False,
        filters=dict(primary_analysis_tau_s=.1,sensitivity_tau_s=list(TAUS),formula='y[k+1]=y[k]+(1-exp(-dt[k]/tau))*(x[k+1]-y[k]); y0=x0',
            scope='identical posthoc causal native-dt filters on predicted/measured body rates; no resampling; these are analysis time scales, NOT actuator taus or model changes',
            components='slow=L(x); fast=x-L(x); error=e_slow+e_fast; MSEraw=MSEslow+MSEfast+2mean(e_slow*e_fast), weighted by native dt',
            intervals='states1..25 and26..50; filtering continuous from t0, no reset at25; report finite-window initial transient, NOT nonoverlapping Fourier frequency bands',
            normalized='error RMS/true component RMS change; slow/reference=xlow-x0; fast/reference=x-xlow; macro ratio after flight/seed aggregation; zero denominator undefined'),
        coverage=dict(activity='same independent coordinates/scales and25-command RMS departure as preceding diagnostic',
            high='channel>TRAINq75',isolated='high and ALL other channels<=their TRAINq25',predominant='high and ALL others<=their TRAINq75; secondary, not substituted for isolated',
            persistence='abs(mean(command departure))/RMS(command departure); >=0.8 reported as largely one-direction departure; not a controlled step event',
            sign='sign(mean departure); no dynamics outcomes used',correlation='within-flight Pearson correlation of all25 standardized departure samples, equal-flight average; descriptive only, no dynamics fitting'),
        metrics='native-dt weighted per-origin component MSE->flight equal-origin mean->sqrt->equal-flight mean->three-seed mean/sampleSD; cross term reported on squared-error scale',
        exclusions='no origins dropped from errors; undefined correlations/ratios counted, no epsilon; no model predictions generated or weights fitted',
        initial_claims='fast residual is not automatically sensor noise or flapping; isolated command changes are not randomized causal experiments')
    m.write_json(OUT/'protocol.json',p);print('FROZEN before error analysis')

def coverage(batches,thresholds):
    rows=[];identities=[];corr=[];timing=[]
    for partition,b in batches.items():
        if partition not in ('train','validation'):raise ValueError('partition not allowed')
        t=b.trajectory;sc,mean,persist=channel_scores(t.controls,thresholds.scale.to_numpy())
        masks=isolated_masks(sc,thresholds.train_q25.to_numpy(),thresholds.train_q75.to_numpy())
        for j,ch in enumerate(r.CHANNELS):
            for kind,mask in masks[ch].items():
                # Include zeros for every admitted flight.
                for flight in sorted(set(t.log_ids)):
                    take=mask&(t.log_ids==flight)
                    rows.append(dict(partition=partition,channel=ch,subset=kind,flight_id=flight,cohort=r.cohort(flight) if partition=='validation' else 'train',
                        n_origins=int(take.sum()),positive=int((mean[take,j]>0).sum()),negative=int((mean[take,j]<0).sum()),zero=int((mean[take,j]==0).sum()),
                        one_direction=int((persist[take,j]>=.8).sum())))
                for i in np.flatnonzero(mask):identities.append(dict(partition=partition,channel=ch,subset=kind,window_id=t.window_ids[i],flight_id=t.log_ids[i],activity=sc[i,j],mean_departure=mean[i,j],persistence=persist[i,j]))
        c=r.coordinates(t.controls[:,:25]);d=(c-c[:,:1])/thresholds.scale.to_numpy()
        for flight in sorted(set(t.log_ids)):
            x=d[t.log_ids==flight].reshape(-1,4);cv=np.corrcoef(x,rowvar=False)
            for j,ch in enumerate(r.CHANNELS):
                for k,other in enumerate(r.CHANNELS):corr.append(dict(partition=partition,flight_id=flight,channel=ch,other=other,correlation=cv[j,k]))
        dt=t.dt_s;freq=t.truth.flap_frequency_hz[:,1:]
        timing.append(dict(partition=partition,n_origins=len(t.window_ids),n_flights=len(set(t.log_ids)),dt_median_s=float(np.median(dt)),dt_p05_s=float(np.quantile(dt,.05)),dt_p95_s=float(np.quantile(dt,.95)),dt_max_s=float(dt.max()),flap_frequency_median_hz=float(np.median(freq)),flap_frequency_p05_hz=float(np.quantile(freq,.05)),flap_frequency_p95_hz=float(np.quantile(freq,.95))))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'excitation_per_flight.csv',index=False)
    pd.DataFrame(identities).to_csv(OUT/'excitation_identities.csv',index=False)
    pd.DataFrame(corr).to_csv(OUT/'control_correlation_per_flight.csv',index=False)
    pd.DataFrame(timing).to_csv(OUT/'timing_frequency_context.csv',index=False)
    expanded=pd.concat([frame,frame.assign(cohort='ALL')]);agg=expanded.groupby(['partition','cohort','channel','subset']).agg(n_origins=('n_origins','sum'),flights_with_samples=('n_origins',lambda x:int((x>0).sum())),admitted_flights=('flight_id','nunique'),positive=('positive','sum'),negative=('negative','sum'),one_direction=('one_direction','sum')).reset_index()
    agg.to_csv(OUT/'excitation_coverage.csv',index=False)
    return agg

def analyze():
    p=m.read_json(OUT/'protocol.json');m.verify_pins(p['frozen_input_sha256']);sp,batches,stats=r.source.inputs()
    thresholds=pd.read_csv(OUT/'channel_thresholds.csv');cov=coverage(batches,thresholds)
    b=batches['validation'];t=b.trajectory;truth=t.truth.angular_velocity_b;dt=t.dt_s;selection=pd.read_csv(r.OUT/'origin_selection.csv')
    masks={'ALL':np.ones(len(dt),bool),'low':(selection.activity_group=='low').to_numpy(),'high':(selection.activity_group=='high').to_numpy()}
    rows=[];max_identity=0.
    for seed in (17,23,42):
        with np.load(r.source.predpath(seed),allow_pickle=False) as f:
            np.testing.assert_array_equal(f['window_ids'],t.window_ids);pred=f['angular_velocity_b'].astype(float)
        for tau in TAUS:
            tl=lowpass(truth,dt,tau);pl=lowpass(pred,dt,tau)
            raw=pred[:,1:]-truth[:,1:];slow=pl[:,1:]-tl[:,1:];fast=raw-slow
            trueparts={'slow':tl[:,1:]-truth[:,:1],'fast':truth[:,1:]-tl[:,1:],'raw':truth[:,1:]-truth[:,:1]}
            predparts={'slow':pl[:,1:]-pred[:,:1],'fast':pred[:,1:]-pl[:,1:],'raw':pred[:,1:]-pred[:,:1]}
            for label,start,end in [('0_500ms',0,25),('500_1000ms',25,50)]:
                w=dt[:,start:end];ms={k:r.weighted_mean(v[:,start:end]**2,w) for k,v in [('raw',raw),('slow',slow),('fast',fast)]}
                cross=2*r.weighted_mean(slow[:,start:end]*fast[:,start:end],w)
                identity=np.max(np.abs(ms['raw']-ms['slow']-ms['fast']-cross));max_identity=max(max_identity,float(identity));assert identity<1e-10
                tm={k:r.weighted_mean(v[:,start:end]**2,w) for k,v in trueparts.items()};pm={k:r.weighted_mean(v[:,start:end]**2,w) for k,v in predparts.items()}
                for group,mask in masks.items():
                    for flight in sorted(set(t.log_ids[mask])):
                        take=mask&(t.log_ids==flight)
                        for j,axis in enumerate(('p','q','r')):
                            for component in ('raw','slow','fast'):
                                rows.append(dict(seed=seed,tau_s=tau,interval=label,group=group,flight_id=flight,cohort=r.cohort(flight),n_origins=int(take.sum()),axis=axis,component=component,
                                    error_mse=float(ms[component][take,j].mean()),rmse=float(np.sqrt(ms[component][take,j].mean())),
                                    true_amplitude=float(np.sqrt(tm[component][take,j].mean())),pred_amplitude=float(np.sqrt(pm[component][take,j].mean())),cross_term=float(cross[take,j].mean())))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'component_per_flight.csv',index=False)
    ex=pd.concat([frame,frame.assign(cohort='ALL')]);keys=['tau_s','interval','group','cohort','axis','component']
    per=ex.groupby(keys+['seed'])[['rmse','true_amplitude','pred_amplitude','error_mse','cross_term']].mean().reset_index();per.to_csv(OUT/'component_per_seed.csv',index=False)
    agg=per.groupby(keys)[['rmse','true_amplitude','pred_amplitude','error_mse','cross_term']].agg(['mean','std']).reset_index();agg.columns=['_'.join(x).rstrip('_') if isinstance(x,tuple) else x for x in agg.columns]
    agg['relative_error']=agg.rmse_mean/agg.true_amplitude_mean.replace(0,np.nan);agg['amplitude_ratio']=agg.pred_amplitude_mean/agg.true_amplitude_mean.replace(0,np.nan)
    agg.to_csv(OUT/'component_summary.csv',index=False)
    m.write_json(OUT/'sanity_checks.json',dict(status='passed',native_dt=True,all_origins_preserved=True,exact_error_decomposition_max_residual=max_identity,heldout_data_accessed_this_run=False))
    plot_report(agg,cov)
    m.verify_pins(p['frozen_input_sha256'])
    m.write_json(OUT/'completion.json',dict(status='complete',training_runs=0,new_predictions=0,heldout_data_accessed_this_run=False,
        artifact_sha256={m.rel(f):m.file_hash(f) for f in OUT.iterdir() if f.is_file() and f.name!='completion.json'}))
    print('COMPLETE')

def markdown(f):
    return '| '+' | '.join(f.columns)+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+'\n'.join('| '+' | '.join(f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |' for row in f.itertuples(index=False,name=None))

def plot_report(agg,cov):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for group in ('ALL','high','low'):
        fig,axs=plt.subplots(1,3,figsize=(12,3.8))
        for ax,axis in zip(axs,('p','q','r')):
            for comp in ('raw','slow','fast'):
                g=agg.query('group==@group and axis==@axis and component==@comp and interval=="0_500ms" and cohort=="ALL"').sort_values('tau_s')
                ax.errorbar(g.tau_s*1000,g.rmse_mean,yerr=g.rmse_std,marker='o',label=comp)
            ax.set(title=axis,xlabel='Analysis smoothing time [ms]',ylabel='Component error RMS [rad/s]');ax.legend();ax.grid(alpha=.2)
        fig.suptitle(group+' / 0–500 ms; overlapping components, not additive powers');fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(OUT/f'components_{group}.{ext}',dpi=170)
        plt.close(fig)
    primary=agg.query('group=="high" and tau_s==.1 and interval=="0_500ms" and cohort=="ALL"')[['axis','component','rmse_mean','rmse_std','true_amplitude_mean','amplitude_ratio','relative_error']]
    primary.to_csv(OUT/'primary_component_table.csv',index=False)
    text='''# p/r响应误差时间尺度与日志激励覆盖诊断

本轮只读复用Standard GRU/H26/K50三个冻结模型的validation预测及原train/validation数据。没有训练、推理新轨迹、修改模型或控制器，没有读取Sep8/Sep19。阈值和分析规则先于分量误差计算冻结。

## 方法及解释边界
对真实与预测角速度施加相同的native-dt一阶因果低通，分析平滑时间固定50/100/200 ms；100 ms为预先指定主展示，全部敏感性结果保留。它们是后处理分析尺度，不是执行器tau。慢分量为滤波输出，快分量为原信号减慢分量。起点用各自原始t0初始化，500 ms处不重置。

这不是不重叠频带分解。MSE(raw)=MSE(slow)+MSE(fast)+2E[e_slow e_fast]；cross term单独保存并核验。不能把两个RMS或平方百分比直接相加，更不能因为平滑后误差降低就声称模型性能提高。有限1s窗口存在滤波启动效应，尤其200ms尺度，没有据此估计精确控制带宽或执行器延迟。

每origin按native dt加权，再flight内等权MSE开方，再flight等权，再三seed mean/sampleSD。幅值比和相对误差用macro量相除，不平均window百分比。慢分量真实幅值以滤波真实角速度减t0为参考，快分量以真实残差为参考。快分量可能是真实运动、扑翼相关振动、状态估计或测量成分；本轮不能将其直接称为噪声。

## 高控制变化组：0–500 ms主展示
单位rad/s；454origins，17flights。

'''+markdown(primary)+'''

## 激励覆盖
每通道活动度沿用前一轮500ms RMS命令变化和固定尺度。high为超过训练q75；isolated要求同时其他三个通道均<=各自训练q25；predominant仅要求其他通道均<=q75，作为明确较宽的辅助条件，不取代严格条件。所有计数、正负方向、较单向变化(abs(mean departure)/RMS>=0.8)及每flight零计数都保存。

'''+markdown(cov.query('cohort=="ALL"'))+'''

不能把这些片段称为随机独立激励：控制仍由反馈产生，未观测气流/初始状态/其他通道小变化仍可能混杂。反过来，严格筛选样本少也不能数学上证明所有MIMO辨识方法不可用，只说明现有日志难以支持简单的隔离单通道事件比较。

## 数据来源与测量限制
原状态管线从vehicle_angular_velocity对齐角速度，允许最大50ms freshness；缓存数据基于native状态网格。samples_validation没有逐轴原始gyro的完整高频序列或逐条角速度源时间戳，只有状态/相位等来源时间字段。因此本轮不能确认快分量来自哪一级滤波、采样相位、混叠或传感器噪声，也不能从当前50Hz左右网格恢复实飞约400Hz控制环输入。实际dt与扑频分布见timing_frequency_context.csv；这些统计不证明快分量具有扑翼因果来源。

## 下一步边界
依据结果区分低频运动趋势误差与较快波形残差，再决定是否需要专门输入激励和更完整同步记录。不能仅凭平滑后的误差或观察到的命令关联直接调增益、放大舵效、调执行器tau或启动结构改造。本轮不运行后续实验。
'''
    (OUT/'report.md').write_text(text);(OUT/'review_report.md').write_text(text)

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='analyze':analyze()
