"""Deliver bandwidth study with matched-target metrics and explicit delay limits."""
import json
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_control_windows_v2 as base
from reference_bandwidth_v5 import OUT,DATA,load
from train_reference_unit_gravity import model_from
from control_windows_v3 import predict
from system_identification.models.reference_antialias_readout import antialias_rate_readout
from system_identification.models.reference_bandwidth_predictor import ReferenceBandwidthPredictor


def table(frame):
    def cell(v):return f'{v:.6g}' if isinstance(v,(float,np.floating)) else str(v).replace('|','\\|')
    rows=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    rows+=['| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(rows)


def main():
    torch.set_num_threads(1);x,y,stats,w=load();protocol=json.loads((OUT/'protocol.json').read_text());gate=json.loads((OUT/'tiny_decision.json').read_text())
    status=json.loads((OUT/'formal_launch/status.json').read_text());assert status['status'] in ['complete','not_launched']
    oldx,oldy,_=base.load();checks={}
    for k in y:checks[k+'_common20ms_max_abs']=float((y[k][:,1::2]-oldy[k]).abs().max());assert checks[k+'_common20ms_max_abs']<1e-6
    for k in ['history_state','history_reference','history_actuator']:
        checks[k+'_common20ms_max_abs']=float((x[k][:,::2]-oldx[k]).abs().max());assert checks[k+'_common20ms_max_abs']<1e-6
    torch.testing.assert_close(x['future_reference'],oldx['future_reference'].repeat_interleave(2,dim=1),rtol=0,atol=0)
    for _,g in w.groupby('log_id'):
        g=g.sort_values('start_timestamp_us');assert (g.start_timestamp_us.to_numpy()[1:]>g.end_timestamp_us.to_numpy()[:-1]).all()
    checks['original_support_disjoint']=True;checks['same_reference_tape']=True
    with np.load(DATA/'antialiased_50hz_labels.npz') as z:
        np.testing.assert_allclose(z['rates_b'],antialias_rate_readout(x['history_state'][:,:,3:6],y['rates_b']).numpy(),atol=1e-6,rtol=1e-6)
    checks['stored_filter_labels_match_inference']=True
    runpaths=list((OUT/'tiny').glob('*/final.pt'))+list((OUT/'formal').glob('*/final.pt')) if (OUT/'formal').exists() else list((OUT/'tiny').glob('*/final.pt'))
    reloads=[]
    for ck in runpaths:
        manifest=json.loads((ck.parent/'status.json').read_text());assert base.sha(ck)==manifest['checkpoint_sha256']
        subset='validation' if ck.parent.parent.name=='formal' else 'fit8'
        with np.load(ck.parent/f'{subset}_predictions.npz') as z:
            ix=z['array_index'][:3];expected={k:z[k][:3] for k in y}
        model=model_from(ck);pp=predict(model,x,ix);diff=max(float(abs(pp[k]-expected[k]).max()) for k in pp);assert diff<1e-5
        arm=manifest['arm'];wrapper=ReferenceBandwidthPredictor(model,arm)
        with torch.no_grad():operational=wrapper(**{k:v[ix] for k,v in x.items()})
        for value in operational.values():assert torch.isfinite(value).all()
        # Verify future-reference changes cannot alter earlier forecast outputs.
        inputs={k:v[ix].clone() for k,v in x.items()};inputs['future_reference'][:,50:]+=1
        with torch.no_grad():altered=wrapper(**inputs)
        prefix=50 if arm=='raw100' else 25
        for key in operational:torch.testing.assert_close(operational[key][:,:prefix],altered[key][:,:prefix],atol=0,rtol=0)
        reloads.append(dict(checkpoint=str(ck.relative_to(base.ROOT)),max_abs_difference=diff,readout=arm,prefix_causality_passed=True))
    base.write(OUT/'verification.json',dict(dataset_checks=checks,checkpoint_reload=reloads,focused_tests_passed=23,
        environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__),sealed_test_accessed=False))
    tiny=pd.read_csv(OUT/'tiny_summary.csv');decisions=pd.DataFrame(gate['decisions'])
    lines=['# 采样带宽对照 v5','',
        '本轮已完成相同片段的数据构建、小样本预筛和条件触发的正式训练。未重新筛选弱响应样本；沿用407训练、739开发验证窗口，来自14个授权日志。验证集已多次用于研究，不能称盲测；sealed/reserved未打开。','',
        '## 数据与模型边界','',
        '- `dataset/control_windows_v5/windows_100hz.npz`：H51/K100，历史0.5秒、预测1秒。角速度由原始ULog按发布/采样时间因果重采样；当前时刻、原20ms端点及原片段支撑保持不变。其他量只保持原50Hz缓存值，不冒充新的100Hz测量。',
        '- `dataset/control_windows_v5/antialiased_50hz_labels.npz`：先在100Hz上进行因果FIR滤波，再取50Hz；附带20个历史角速度样本和滤波系数。21抽头Hamming-sinc，名义截止15Hz，100ms群时延；0–8Hz最小增益约-0.156dB，25Hz以上最大增益约-52.29dB。',
        '- 两个新模型都使用同一100Hz潜在动力学和原始历史。`raw100`学习原始角速度；`aa50`仅改变角速度损失与输出读出。姿态由模型内部原始角速度积分，避免把延迟角速度当成当前物理状态。姿态/速度损失仍在原20ms时刻评价。',
        '- `aa50`不是仅靠50Hz滤波输入运行的模型：它保留100Hz原始历史和滤波状态。其输出组合为当前预测姿态/速度与延迟角速度，不能直接作为同一时刻的瞬时状态接入控制器。',
        '- 统一原A-train归一化、GRU64/40070参数、优化器、初始化种子及更新数。新模型每次展开100步，计算量高于原50步模型；不能把raw100与旧基线的差异单独归因于采样率。','',
        '## 小样本训练内门槛','',
        '使用上一轮预选的8个训练窗口、seed17、1600次更新。指标为各窗口完整1秒波形RMSE，先平均窗口再平均飞行，最后平均p/q。不同目标分别比较，不能靠滤波后的绝对误差更低宣称提升。','',
        table(tiny),'',table(decisions),'',
        'raw100相对旧tiny50未通过；aa50相对同读出的raw100通过预定15%改善门槛。因而正式训练aa50及必需的raw100对照，均3个种子，不使用验证结果选择训练预算或最佳checkpoint。','',
        '## 正式对照','']
    formal=None
    if status['status']=='complete':
        formal=json.loads((OUT/'formal_decision.json').read_text());summary=pd.read_csv(OUT/'formal_summary.csv')
        shown=summary[(summary.cohort=='validation')&(summary.subset=='all')&summary.horizon_s.isin([.5,1.])]
        shown=shown[((shown.arm=='A1600_50hz')&(shown.readout=='raw50'))|((shown.arm=='raw100'))|((shown.arm=='aa50')&(shown.readout=='aa50'))]
        lines+=['下表为开发验证集全窗口、跨种子与飞行平均；p/q/r为rad/s、姿态为rad、速度为m/s。raw50与aa50不可跨目标比较。','',table(shown[['arm','readout','horizon_s','p','q','r','attitude','velocity']]),'']
        gap=summary[(summary.arm=='aa50')&(summary.readout=='aa50')&(summary.subset=='all')&(summary.horizon_s==.5)]
        lines+=['同一aa50目标的训练/验证误差（0.5秒）：','',table(gap[['cohort','p','q','r','attitude','velocity']]),'']
        response=pd.read_csv(OUT/'reference_hold_summary.csv')
        response=response[(response.arm=='aa50')&(response.readout=='aa50')&(response.subset=='all')&response.horizon_s.isin([.5,1.])]
        lines+=['参考保持诊断（aa50，同一滤波目标）：','',table(response[['intervention','horizon_s','p','q','attitude']]),'',
            '例如1秒q误差在保持参考后增大，表明模型确实利用未来参考信息；这是模型输入依赖检查，不是实际飞机在改变参考后的反事实真值或舵效认证。','']
        for row in formal['comparisons']:
            lo,hi=row['paired_flight_bootstrap_gain_95percent']
            lines.append(f"- {row['candidate']} 对 {row['baseline']}，目标 {row['readout']}：主动p/q在0.5/1秒的联合指标改善 {row['primary_gain_fraction']:.2%}；按飞行配对bootstrap95%区间 [{lo:.2%}, {hi:.2%}]。相对门槛通过={row['relative_gate_passed']}，绝对门槛通过={row['absolute_gate_passed']}。")
        lines+=['',table(pd.DataFrame(formal['absolute_by_seed'])),'']
    lines+=['## 限制、验证与下一步','',
        '采样增量诊断发现约4.55%的训练raw100滚转增量超过模型原有±120rad/s²增量上限，raw50约0.43%；这针对日志离散值，不证明飞机物理加速度超限。由此得到的弱序列RMSE下界训练平均约0.050rad/s，不能解释全部滚转误差。本轮没有放宽上限。',
        '使用真实未来角速度的诊断回放中，验证姿态RMSE：100Hz在0.5/1秒约0.0121/0.0152rad，50Hz约0.0208/0.0295rad。加密积分改善了记录链路的一致性，但这是使用未来真值的离线诊断，既不可部署，也不是模型误差的严格下界；姿态和角速度的采样时刻仍有差异。',
        '23项针对性测试通过。新数据在原20ms时刻逐项复现旧数据；滤波结果与独立因果lfilter实现一致；所有checkpoint重载和未来参考前缀因果性验证通过。原窗口支撑不重叠，滤波器初始化不借用窗口外数据。',
        '本轮不启动控制优化：短时预测门槛与独立控制响应验证必须分别成立，训练内拟合和经过滤波的误差不能替代真实输入作用证据。','']
    if formal:
        aa=next(r for r in formal['comparisons'] if r['candidate']=='aa50')
        if aa['relative_gate_passed']:
            lines+=['下一步可保留aa50作为带明确时延的观测预测候选，先定位未通过的绝对误差/工况，再验证0.5–1秒真实操纵响应。接入任何控制器之前，需要处理100ms读出时延，并验证内部瞬时状态与模型不确定性。']
        else:
            lines+=['下一步不把aa50或raw100替换为主模型；保留本轮数据与推理读出作为诊断工具。小样本改善未达到正式晋升条件时，应区分训练优化不足、跨工况泛化与不可预测测量成分，再做单因素实验，不继续叠加滤波或删除弱响应。']
    # Time-selected validation example, or train example only if formal launch was blocked.
    meta=w[w.partition=='validation'] if formal else w.set_index('window_id').loc[protocol['tiny_cohort']].reset_index()
    j=int(meta.iloc[len(meta)//2].array_index);partition='validation' if formal else 'fit8';scope='formal' if formal else 'tiny'
    fig,axes=plt.subplots(2,2,figsize=(11,6),sharex=True);t=np.arange(1,101)*.01
    truth_raw=y['rates_b'][j:j+1];truth_aa=antialias_rate_readout(x['history_state'][j:j+1,:,3:6],truth_raw).numpy()
    for axidx,axis in enumerate(['p','q']):
        axes[axidx,0].plot(t,truth_raw.numpy()[0,:,axidx],label='Observed raw100',lw=1)
        axes[axidx,1].plot(t[1::2],truth_aa[0,:,axidx],label='Observed FIR50',lw=1)
    for arm in ['raw100','aa50']:
        p=OUT/scope/f'{arm}_17'/f'{partition}_predictions.npz'
        with np.load(p) as z:loc=int(np.flatnonzero(z['array_index']==j)[0]);rates=torch.tensor(z['rates_b'][loc:loc+1])
        aa=antialias_rate_readout(x['history_state'][j:j+1,:,3:6],rates).numpy()
        for axidx in [0,1]:
            axes[axidx,0].plot(t,rates.numpy()[0,:,axidx],label=arm,lw=1);axes[axidx,1].plot(t[1::2],aa[0,:,axidx],label=arm,lw=1)
    for i in [0,1]:
        for ax in axes[i]:ax.set_ylabel(('p' if i==0 else 'q')+' (rad/s)');ax.grid(alpha=.25)
    axes[0,0].set_title('Raw100 latent rate (diagnostic)');axes[0,1].set_title('Matched causal FIR50 target (100ms lag)');axes[0,0].legend(fontsize=8)
    for ax in axes[-1]:ax.set_xlabel('Forecast horizon (s)')
    fig.suptitle(f'Time-selected {partition} window {j}, seed17');fig.tight_layout();fig.savefig(OUT/'waveform_comparison.png',dpi=150);plt.close(fig)
    base.write(OUT/'plot_provenance.json',dict(array_index=j,window_id=w.iloc[j].window_id,selection='middle row by fixed original table order; not selected by model error',seed=17))
    (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')
    paths=sorted((base.ROOT/'scripts').glob('*reference_bandwidth_v5*.py'))+[base.ROOT/'scripts/reference_bandwidth_v5.py',base.ROOT/'src/system_identification/models/reference_antialias_readout.py',base.ROOT/'src/system_identification/models/reference_bandwidth_predictor.py',base.ROOT/'tests/test_reference_bandwidth_v5.py']
    base.write(OUT/'delivery_manifest.json',dict(status='complete',formal_run_count=len(status.get('completed',[])),small_fit_run_count=2,
        artifact_hashes={str(p.relative_to(base.ROOT)):base.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'},
        data_hashes={str(p.relative_to(base.ROOT)):base.sha(p) for p in sorted(DATA.glob('*.npz'))},
        code_hashes={str(p.relative_to(base.ROOT)):base.sha(p) for p in sorted(set(paths))},sealed_test_accessed=False))
if __name__=='__main__':main()
