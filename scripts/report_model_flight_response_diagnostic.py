"""Logged response waveform fidelity and observational reinitialization diagnostic."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/flap-paper-mpl')
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run_model_flight_response_diagnostic as r
m=r.m;OUT=r.OUT;ART=r.ART
from system_identification.models.trajectory import TrajectoryPrediction
from system_identification.evaluation.future_control_diagnostic import squared_errors
METRICS=m.METRICS

def smooth(x):
    return np.stack([x[:,max(0,k-4):k+1].mean(1) for k in range(x.shape[1])],axis=1)

def alignment(pred,truth,dt):
    """Fixed interior comparison, native elapsed lag; not causal actuator delay."""
    p,t=smooth(pred),smooth(truth);idx=np.arange(5,45);base=t[:,idx]
    base=base-base.mean(1,keepdims=True);tn=np.sqrt(np.sum(base**2,axis=1))
    best=np.full(tn.shape,-np.inf);lag=np.zeros(tn.shape,int);zero=np.full(tn.shape,np.nan)
    valid=tn>1e-12
    for shift in (0,-1,1,-2,2,-3,3,-4,4,-5,5):
        a=p[:,idx+shift];a=a-a.mean(1,keepdims=True);pn=np.sqrt(np.sum(a*a,axis=1));den=pn*tn
        c=np.full(den.shape,np.nan);np.divide(np.sum(a*base,axis=1),den,out=c,where=(den>1e-12))
        if shift==0:zero=c.copy()
        better=np.isfinite(c)&(c>best);best[better]=c[better];lag[better]=shift
    valid&=np.isfinite(best);best[~valid]=np.nan
    times=np.cumsum(dt,axis=1);lag_seconds=np.zeros(tn.shape)
    for j in range(pred.shape[2]):
        ix=idx[None,:]+lag[:,j,None]
        lag_seconds[:,j]=(np.take_along_axis(times,ix,axis=1)-times[:,idx]).mean(1)
    lag_seconds[~valid]=np.nan
    return zero,best,lag,lag_seconds,valid

def macro(df,keys,value_columns):
    expanded=pd.concat([df,df.assign(cohort='ALL')],ignore_index=True)
    per=expanded.groupby(keys+['cohort','seed'])[value_columns].mean().reset_index()
    multi=per.groupby(keys+['cohort'])[value_columns].agg(['mean','std']).reset_index()
    multi.columns=['_'.join(c).rstrip('_') if isinstance(c,tuple) else c for c in multi.columns]
    return per,multi

def md(df):
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))

def run():
    protocol=m.read_json(OUT/'protocol.json');m.verify_pins(protocol['frozen_input_sha256'])
    sp,batches,stats=r.source.inputs();b=batches['validation'];truth=b.trajectory.truth
    selection=pd.read_csv(OUT/'origin_selection.csv');groups=r.masks(selection)
    logs=b.trajectory.log_ids;cohorts=np.array([r.cohort(x) for x in logs]);dt=b.trajectory.dt_s;n=len(logs)
    axes=r.state_axes(truth);change=axes[:,1:]-axes[:,:1]
    floors=np.array([protocol['direction_floor'][a] for a in r.AXES])
    local=torch.load(ART/'local_batch.pt',map_location='cpu',weights_only=False)
    wave=[];lagrows=[];errors=[];cases=m.read_json(OUT/'representative_selection.json')
    predictions={}
    for seed in (17,23,42):
        with np.load(r.source.predpath(seed),allow_pickle=False) as f:
            np.testing.assert_array_equal(f['window_ids'],b.trajectory.window_ids)
            pred=TrajectoryPrediction(**{k:f[k] for k in vars(truth)})
        with np.load(ART/f'local_seed{seed}.npz',allow_pickle=False) as f:
            np.testing.assert_array_equal(f['window_ids'],local.trajectory.window_ids)
            lp=TrajectoryPrediction(**{k:f[k] for k in vars(truth)})
        if seed==17:predictions=dict(actual=pred,local=lp)
        pe=squared_errors(pred,truth);le=squared_errors(lp,local.trajectory.truth).reshape(10,n,5,4)
        dp=r.state_axes(pred)[:,1:]-axes[:,:1]
        for name,mask in groups.items():
            for log in sorted(set(logs[mask])):
                take=mask&(logs==log);c=cohorts[take][0]
                for step in range(1,51):
                    for condition,origin_mse in [('autonomous',pe[:,step-1])]+([('local100ms',le[step//5-1,:,4])] if step%5==0 else []):
                        rmse=np.sqrt(origin_mse[take].mean(0))
                        for j,metric in enumerate(METRICS):
                            errors.append(dict(seed=seed,group=name,flight_id=log,cohort=c,n_origins=int(take.sum()),step=step,condition=condition,metric=metric,value=float(rmse[j])))
        for label,start,end in [('early',0,5),('middle',5,15),('late',15,25),('extended',25,50)]:
            weights=dt[:,start:end];t=change[:,start:end];p=dp[:,start:end]
            tm=r.weighted_mean(t,weights);pm=r.weighted_mean(p,weights)
            mse=r.weighted_mean((p-t)**2,weights);ta=r.weighted_mean(t*t,weights);pa=r.weighted_mean(p*p,weights)
            for name,mask in groups.items():
                for log in sorted(set(logs[mask])):
                    take=mask&(logs==log);c=cohorts[take][0]
                    for j,axis in enumerate(r.AXES):
                        eligible=take&(np.abs(tm[:,j])>floors[j]);count=int(eligible.sum())
                        agree=float(np.mean((tm[eligible,j]*pm[eligible,j])>0)) if count else np.nan
                        wave.append(dict(seed=seed,group=name,flight_id=log,cohort=c,interval=label,axis=axis,n_origins=int(take.sum()),n_direction=count,
                            waveform_rmse=float(np.sqrt(mse[take,j].mean())),true_amplitude=float(np.sqrt(ta[take,j].mean())),pred_amplitude=float(np.sqrt(pa[take,j].mean())),
                            mean_change_bias=float((pm[take,j]-tm[take,j]).mean()),direction_agreement=agree))
        zero,best,lag,seconds,valid=alignment(dp,change,dt)
        for name,mask in groups.items():
            for log in sorted(set(logs[mask])):
                take=mask&(logs==log)
                for j,axis in enumerate(r.AXES):
                    ok=take&valid[:,j]
                    lagrows.append(dict(seed=seed,group=name,flight_id=log,cohort=cohorts[take][0],axis=axis,n_origins=int(take.sum()),n_valid=int(ok.sum()),
                        zero_lag_correlation=float(np.nanmean(zero[ok,j])) if ok.any() else np.nan,
                        best_correlation=float(np.nanmean(best[ok,j])) if ok.any() else np.nan,
                        median_alignment_lag_s=float(np.nanmedian(seconds[ok,j])) if ok.any() else np.nan,
                        bound_fraction=float(np.mean(np.abs(lag[ok,j])==5)) if ok.any() else np.nan))
    wf=pd.DataFrame(wave);ef=pd.DataFrame(errors);lf=pd.DataFrame(lagrows)
    wf.to_csv(OUT/'response_per_flight.csv',index=False);ef.to_csv(OUT/'errors_per_flight.csv',index=False);lf.to_csv(OUT/'alignment_per_flight.csv',index=False)
    ws,wm=macro(wf,['group','interval','axis'],['waveform_rmse','true_amplitude','pred_amplitude','mean_change_bias','direction_agreement'])
    wm['amplitude_ratio']=wm.pred_amplitude_mean/wm.true_amplitude_mean.replace(0,np.nan)
    # Denominators for sign comparison and missing-flight coverage must be visible.
    expanded=pd.concat([wf,wf.assign(cohort='ALL')]);coverage=expanded.groupby(['group','interval','axis','cohort','seed']).agg(n_origins=('n_origins','sum'),n_direction=('n_direction','sum'),n_flights=('flight_id','nunique'),n_direction_flights=('n_direction',lambda x:int((x>0).sum()))).reset_index()
    coverage.to_csv(OUT/'direction_coverage.csv',index=False)
    es,em=macro(ef,['group','step','condition','metric'],['value'])
    ls,lm=macro(lf,['group','axis'],['zero_lag_correlation','best_correlation','median_alignment_lag_s','bound_fraction'])
    for name,frame in [('response_per_seed',ws),('response_summary',wm),('error_per_seed',es),('error_evolution',em),('alignment_per_seed',ls),('alignment_summary',lm)]:frame.to_csv(OUT/f'{name}.csv',index=False)
    pair=es.pivot(index=['group','step','metric','cohort','seed'],columns='condition',values='value').dropna().reset_index()
    pair['local_gain_pct']=100*(pair.autonomous-pair.local100ms)/pair.autonomous.replace(0,np.nan)
    pair.to_csv(OUT/'local_paired_seeds.csv',index=False)
    fp=ef[ef.step%5==0].groupby(['group','step','metric','cohort','flight_id','condition']).value.mean().unstack('condition').reset_index()
    fp['local_gain']=fp.autonomous-fp.local100ms;fp.to_csv(OUT/'local_paired_flights.csv',index=False)
    timing=pd.DataFrame([dict(step=k,median_s=float(np.median(dt[:,:k].sum(1))),p05_s=float(np.quantile(dt[:,:k].sum(1),.05)),p95_s=float(np.quantile(dt[:,:k].sum(1),.95))) for k in range(1,51)])
    timing.to_csv(OUT/'elapsed_time.csv',index=False)
    for group in ('ALL','low','high'):
        fig,axs=plt.subplots(1,3,figsize=(13,3.6))
        for ax,metric in zip(axs,METRICS[1:]):
            for condition in ('autonomous','local100ms'):
                g=em.query('group == @group and metric == @metric and condition == @condition and cohort == "ALL"').sort_values('step')
                x=timing.set_index('step').loc[g.step,'median_s'].to_numpy()
                ax.plot(x,g.value_mean,label=condition);ax.fill_between(x,g.value_mean-g.value_std,g.value_mean+g.value_std,alpha=.15)
            ax.set(xlabel='Median elapsed time [s], native dt',ylabel=metric,title=group);ax.legend();ax.grid(alpha=.2)
        fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(OUT/f'error_process_{group}.{ext}',dpi=170)
        plt.close(fig)
    for case in cases:
        i=int(np.flatnonzero(b.trajectory.window_ids==case['window_id'])[0]);time=np.r_[0,np.cumsum(dt[i])]
        name=case['cohort']+'_'+case['group'];fig,axs=plt.subplots(2,3,figsize=(12,6))
        pa=r.state_axes(predictions['actual']);la=r.state_axes(predictions['local']).reshape(10,n,6,6)
        for j,ax in enumerate(axs.flat):
            ax.plot(time,axes[i,:,j],color='k',label='Flight');ax.plot(time,pa[i,:,j],label='Autonomous')
            for block,offset in enumerate(r.OFFSETS):ax.plot(time[offset:offset+6],la[block,i,:,j],color='C2',alpha=.8,label='Local 100 ms' if block==0 else None)
            ax.set(xlabel='Actual elapsed [s]',ylabel=r.AXES[j]+(' [m/s]' if j<3 else ' [rad/s]'));ax.legend(fontsize=8)
        fig.suptitle(name+'; fixed identity median, seed17');fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(OUT/f'case_{name}.{ext}',dpi=170)
        plt.close(fig)
        fig,axs=plt.subplots(4,1,figsize=(10,6),sharex=True);u=r.coordinates(b.trajectory.controls[i])
        for j,ax in enumerate(axs):ax.step(time[:-1],u[:,j],where='post');ax.set(ylabel=r.CHANNELS[j]);ax.grid(alpha=.2)
        axs[-1].set_xlabel('Actual elapsed [s]; normalized command coordinates, not physical angles/Hz');fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(OUT/f'controls_{name}.{ext}',dpi=170)
        plt.close(fig)
    write_report(wm,em,lm,pair,fp)
    m.verify_pins(protocol['frozen_input_sha256'])
    m.write_json(OUT/'completion.json',dict(status='complete',heldout_data_accessed_this_run=False,prior_heldout_results_known=True,training_runs=0,
        actual_predictions_reused=3,local_forecasts_per_seed=25820,
        artifact_sha256={m.rel(p):m.file_hash(p) for p in OUT.iterdir() if p.is_file() and p.name!='completion.json'},
        prediction_sha256={m.rel(ART/f'local_seed{s}.npz'):m.file_hash(ART/f'local_seed{s}.npz') for s in (17,23,42)},
        report_source_sha256=m.file_hash(Path(__file__))))
    print('REPORT COMPLETE')

from pathlib import Path

def write_report(wm,em,lm,pair,fp):
    cover=pd.read_csv(OUT/'per_flight_coverage.csv');rows=[]
    for group in ('ALL','low','middle','high',*(ch+'_high' for ch in r.CHANNELS)):
        c=cover[cover.group==group];rows.append(dict(group=group,n_origins=int(c.n_origins.sum()),n_flights=c.flight_id.nunique(),Sep7=int(c[c.cohort=='Sep7'].n_origins.sum()),Sep17=int(c[c.cohort=='Sep17'].n_origins.sum())))
    pd.DataFrame(rows).to_csv(OUT/'coverage_summary.csv',index=False)
    endpoint=em[(em.step.isin([5,10,25,50]))&(em.group.isin(['ALL','low','high']))]
    endpoint.to_csv(OUT/'endpoint_summary.csv',index=False)
    comparison=endpoint.pivot(index=['group','step','metric','cohort'],columns='condition',values='value_mean').reset_index()
    comparison['local_gain_pct']=100*(comparison.autonomous-comparison.local100ms)/comparison.autonomous.replace(0,np.nan)
    comparison.to_csv(OUT/'local_comparison.csv',index=False)
    response=wm[(wm.group=='high')&(wm.cohort=='ALL')&(wm.interval.isin(['early','late','extended']))][['interval','axis','waveform_rmse_mean','amplitude_ratio','direction_agreement_mean']]
    text='''# 当前Standard GRU模型—实飞响应一致性诊断

本轮只评价冻结H26/K50三个seed，原41个train flights用于阈值，原17个validation flights/2,582 origins用于报告。没有训练、调参或重新读取Sep8/Sep19；此前测试结果已知，本轮为开发诊断。

## 比较对象与边界
Actual连续预测从原真实起点/H26历史出发，使用同一日志控制、native dt，自主递推50步。直接复用原三seed预测。
Local100ms每5个native steps从当时的真实状态及H26历史重新初始化，独立预测5步；每个原origin有10个局部窗口，全部在原1s区间内。不是可提前得到真值的部署方法，也不是修改原自主模型分数。局部预测拥有更新观测、更短预测年龄，改善不能精确归因于某一个隐藏状态或积分机制。

控制坐标是drive、common=(left+right)/2、differential=(left-right)/2、rudder，来自分配后的归一化命令，非实测舵角/扑频。分组仅由Actual控制及训练阈值决定，规则先于本轮预测误差分析冻结。高通道组重叠，可能存在其他通道同时变化，不视为单通道干预。

## 覆盖
'''+md(pd.DataFrame(rows))+'''

## 端点与误差过程
先flight内vector RMSE/四元数geodesic RMS，再flight等权，再三seed mean/sampleSD。无窗口独立性检验。完整结果见endpoint_summary.csv及error_evolution.csv；时间轴是各origin实际累计dt中位数，范围见elapsed_time.csv。

'''+md(comparison[(comparison.cohort=='ALL')&(comparison.step.isin([25,50]))])+'''

## 响应幅值、方向与持续过程
以下改变均相对原实测t0。early=1..5步、middle=6..15、late=16..25、extended=26..50；用各步实际dt加权。幅值比为同口径macro预测RMS改变/真实RMS改变，不平均逐窗口比例。方向是区间平均状态改变的符号一致率，仅在真实改变超过预先冻结的train q25幅度门槛时计算；全部origins仍参与RMSE及幅值统计，方向分母与flight覆盖见direction_coverage.csv。它不是控制效应符号正确率；真实状态改变还包含初始惯性、其他控制和扰动。

'''+md(response)+'''

## 波形时间对齐
alignment_summary.csv给出两边相同的因果5sample滑动均值后的零延迟相关及±5steps内对齐统计。正lag表示模型波形较晚。固定内部时间索引比较，常量波形不可定义，边界解比例也保存。该数据没有统一、孤立的阶跃事件，相关最优lag不是执行器物理延迟，不能据此调tau。弱相关或搜索边界结果尤其不能解释为精确延迟。

## 固定案例
代表案例在每个cohort的low/high组中按identity排序取中位origin，固定seed17。四个案例的全部速度/角速度轴和四通道命令曲线均保存；不按误差挑图或替换不利案例。绿色Local100ms是独立短段，有重新初始化边界，不是连续长时预测。

## 解释与下一步边界
该分析直接比较同一日志输入下的状态响应匹配程度，但没有实测反事实操纵轨迹，不能证明某一指令的因果舵效，也不能把净状态改变不足直接解释成执行器响应不足。旧actuator-aware分支抵消机制不直接适用于当前Standard GRU。

若局部预测明显更好而自主误差随时间增长，只支持更多实时状态/历史信息有助于预测，不能唯一确定为积分漂移；未观测气流、执行器状态、隐藏状态估计及较长时域误差都可能参与。若方向/波形仍存在系统偏差，应先针对真实响应证据定位，不能只根据仿真输出增大舵效或更改控制器。没有自动启动结构解耦、新训练、实机激励或闭环试验。
'''
    (OUT/'report.md').write_text(text);(OUT/'review_report.md').write_text(text)

if __name__=='__main__':m.configure();run()
