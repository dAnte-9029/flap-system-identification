"""Complete interpretation and independent parity checks; no new inference."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/flap-paper-mpl')
import numpy as np
import pandas as pd
from pathlib import Path
import run_model_flight_response_diagnostic as r
m=r.m;OUT=r.OUT;ART=r.ART

def main():
    protocol=m.read_json(OUT/'protocol.json');m.verify_pins(protocol['frozen_input_sha256'])
    complete=m.read_json(OUT/'completion.json')
    for key in ('artifact_sha256','prediction_sha256'):m.verify_pins(complete[key])
    old=pd.read_csv(m.OUT/'per_seed_summary.csv')
    new=pd.read_csv(OUT/'error_per_seed.csv').query('group == "ALL" and condition == "autonomous"')
    cells=0
    for row in old.itertuples():
        step=m.HORIZONS[row.horizon_s]
        for metric in m.METRICS:
            value=new.query('seed == @row.seed and cohort == @row.cohort and step == @step and metric == @metric').value.iloc[0]
            np.testing.assert_allclose(value,getattr(row,metric),rtol=1e-11,atol=1e-11);cells+=1
    sp,batches,stats=r.source.inputs();b=batches['validation'];truth=r.state_axes(b.trajectory.truth);selection=pd.read_csv(OUT/'origin_selection.csv');masks=r.masks(selection);logs=b.trajectory.log_ids;n=len(logs)
    rows=[]
    for seed in (17,23,42):
        with np.load(r.source.predpath(seed),allow_pickle=False) as f:pred=np.concatenate([f['velocity_n'],f['angular_velocity_b']],axis=-1)
        with np.load(ART/f'local_seed{seed}.npz',allow_pickle=False) as f:local=np.concatenate([f['velocity_n'],f['angular_velocity_b']],axis=-1).reshape(10,n,6,6)
        for step in (25,50):
            for group,mask in masks.items():
                for log in sorted(set(logs[mask])):
                    take=mask&(logs==log)
                    for condition,p in [('autonomous',pred[:,step]),('local100ms',local[step//5-1,:,5])]:
                        rmse=np.sqrt(np.mean((p[take]-truth[take,step])**2,axis=0))
                        for j,axis in enumerate(r.AXES):rows.append(dict(seed=seed,group=group,step=step,condition=condition,flight_id=log,cohort=r.cohort(log),axis=axis,rmse=rmse[j]))
    f=pd.DataFrame(rows);f.to_csv(OUT/'axis_local_per_flight.csv',index=False)
    expanded=pd.concat([f,f.assign(cohort='ALL')]);per=expanded.groupby(['seed','group','step','condition','cohort','axis']).rmse.mean().reset_index()
    agg=per.groupby(['group','step','condition','cohort','axis']).rmse.agg(['mean','std']).reset_index();agg.to_csv(OUT/'axis_local_summary.csv',index=False)
    response=pd.read_csv(OUT/'response_summary.csv');cr=[]
    for ch,axis in [('drive','vz'),('common','q'),('differential','p'),('rudder','r')]:
        row=response.query('group == @ch+"_high" and axis == @axis and cohort == "ALL" and interval == "late"').iloc[0]
        cr.append(dict(channel=ch,axis=axis,amplitude_ratio=row.amplitude_ratio,direction_agreement=row.direction_agreement_mean,waveform_rmse=row.waveform_rmse_mean))
    pd.DataFrame(cr).to_csv(OUT/'channel_response_table.csv',index=False)
    overlap=[]
    for ch in r.CHANNELS:
        take=selection[ch+'_high'].to_numpy(bool)
        others=[x+'_high' for x in r.CHANNELS if x!=ch]
        overlap.append(dict(channel=ch,n_origins=int(take.sum()),fraction_with_other_high_channel=float(selection.loc[take,others].any(axis=1).mean())))
    pd.DataFrame(overlap).to_csv(OUT/'channel_coactivation.csv',index=False)
    text='''
## 完成后解读
**当前模型大体复现记录运动变化的方向，但角速度波形幅值偏低；局部更新观测并未消除大部分角速度误差。尚不能确认其干预舵效与实飞一致，也未证明旧模型的持续响应机制在当前GRU中重现。**

高控制变化组454 origins覆盖17 flights（Sep7 185、Sep17 269）；low888、middle1240，同样均覆盖17 flights。

500 ms对齐时刻，高组连续预测的速度/姿态/角速度误差为0.5081 m/s、5.0629 deg、0.6688 rad/s；从约400 ms真实状态和历史出发的100 ms局部预测为0.1896、1.5517、0.6207，分别降低62.69%、69.35%、7.20%。高组速度/姿态局部收益覆盖17/17 flights，角速度16/17（均为每flight三seed均值方向）。这是不同预测年龄、不同观测信息的诊断对比，不是公平部署性能改进。

高组约300–500 ms区间内，滚转p、俯仰q、偏航r的预测/真实RMS运动改变幅值比分别为0.848、0.918、0.866；符合方向评估门槛的origins分别362、356、378，按flight等权的方向一致率约96.7%、98.2%、91.3%。同一幅值偏低趋势在Sep7/Sep17均存在：p约0.847/0.849，q约0.938/0.898，r约0.903/0.836。不能把这些比值写成“真实舵效的百分比”，因为测量的是相对t0的净运动改变，仍混合惯性、其他控制和扰动。

约500–1000 ms区间的p/q/r幅值比约0.828/0.938/0.817。数据没有显示所有轴随时间一致衰减的简单模式，因此不足以判定统一的持续响应坍塌。尤其q变化方向和波形匹配较好，p的短时波形误差更突出。

按高通道变化片段观察，drive对应vz、common对应q、differential对应p、rudder对应r的300–500 ms幅值比分别0.965、0.924、0.840、0.863。完整六轴结果均保留。这些组并非隔离单轴实验；channel_coactivation.csv记录其他通道共同高变化比例。

平滑波形的零lag相关，高组p/q/r约0.686/0.829/0.590；q相对更好，p/r仍有形状误差。不能把相关最优lag当作执行器延迟；rudder相关的r波形约16.8%的对齐结果位于±5steps搜索边界，缺乏可靠统一延迟估计。没有据此调tau。

固定Sep17-high案例展示：局部预测能跟踪速度变化，但滚转角速度的快速峰谷仍未充分复现。这是图形观察，不等于确认高频部分就是噪声；其他三个固定案例全部保留。

目前更合理的定位是：速度和姿态的误差对更新真实状态/历史很敏感，而角速度存在显著的短时波形匹配残差。未观测气流/执行器状态、扑翼相关运动、状态估计或测量高频成分、模型表达及训练目标平滑倾向都是待区分的解释，本轮没有证明哪一项是根因。局部重置同时改变预测年龄、物理状态和隐藏状态，不能把改善全部归因于隐藏状态漂移。

建议先审阅p/r响应波形和真实记录质量，判断现有控制激励是否足以分离控制作用；若需要确认舵效方向、持续幅值或时延，应另行设计可重复、有明确事件与执行器记录的实飞激励。不要直接增大舵效、调控制器或将结构解耦作为已证实的修复。本轮没有启动后续实验。
'''
    marker='\n## 完成后解读\n'
    for name in ('report.md','review_report.md'):
        path=OUT/name;s=path.read_text().split(marker)[0];path.write_text(s+text)
    tests=dict(status='passed',pytest='7 passed in 1.10s',checks=['coordinate contract','constant-control activity','local identity bounds','native dt weighting','alignment zero and known lag sign','undefined constant waveform','equal-flight and sample SD'],actual_metric_parity_cells=cells,
        source_sha256={m.rel(r.ROOT/'tests'/name):m.file_hash(r.ROOT/'tests'/name) for name in ['test_model_flight_response_diagnostic.py','test_model_flight_response_reporting.py']})
    m.write_json(OUT/'tests.json',tests)
    complete['artifact_sha256']={m.rel(p):m.file_hash(p) for p in OUT.iterdir() if p.is_file() and p.name!='completion.json'}
    complete['source_sha256']={m.rel(r.ROOT/'scripts'/name):m.file_hash(r.ROOT/'scripts'/name) for name in ['run_model_flight_response_diagnostic.py','report_model_flight_response_diagnostic.py','finalize_model_flight_response_diagnostic.py']}
    complete['actual_metric_parity_cells']=cells
    m.write_json(OUT/'completion.json',complete)
    print('FINALIZED; parity cells',cells)

if __name__=='__main__':main()
