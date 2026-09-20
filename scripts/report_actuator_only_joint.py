"""Three-model comparison; report prediction, control diagnostics and closed-loop separately."""
from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_main_v2_free_running import sha
ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'docs/analysis/results/actuator_only_joint_v1';sources={
        'original':(ROOT/'docs/analysis/results/september_expanded_main_v2',ROOT/'docs/analysis/results/isaac_straight_flight_chain_v1'),
        'raw_joint':(ROOT/'docs/analysis/results/joint_control_v1',ROOT/'docs/analysis/results/joint_control_v1/isaac_closed_loop'),
        'actuator_only':(out,out/'isaac_closed_loop')}
    errors=[];summaries=[];traces=[];inputs=[]
    for name,(p,c) in sources.items():
        e=pd.read_csv(p/'per_window.csv');e=e[e.model=='expanded'] if name=='original' else e;e=e.copy();e['model']=name;errors.append(e)
        s=pd.read_csv(c/'summary.csv');s['model']=name;summaries.append(s)
        t=pd.read_csv(c/'traces.csv');t=t[t['mode']=='closed'].copy();t['model']=name;traces.append(t)
        inputs.extend([p/'per_window.csv',c/'summary.csv',c/'traces.csv'])
    identity=[set(zip(e.window_id,e.horizon_s)) for e in errors];assert identity[0]==identity[1]==identity[2]
    metrics=['position_m','velocity_m_s','attitude_deg','body_rate_rad_s','frequency_hz'];errors=pd.concat(errors);agg=errors.groupby(['model','horizon_s'])[metrics].agg(lambda x:float(np.sqrt(np.mean(x*x))));agg.to_csv(out/'prediction_comparison.csv')
    summary=pd.concat(summaries);stats=summary.groupby(['model','mode']).agg(n=('case','size'),completed=('termination',lambda x:int((x=='completed').sum())),median_duration_s=('duration_s','median'),surface_limit_fraction=('surface_limit_fraction','mean'));stats.to_csv(out/'closed_loop_comparison.csv')
    traces=pd.concat(traces);common=[]
    for case,g in traces.groupby('case'):
        assert g.log_id.nunique()==1 and g.model.nunique()==3
        horizon=g.groupby('model').time_s.max().min()
        for model,h in g[g.time_s<=horizon+1e-9].groupby('model'):
            common.append(dict(case=case,log_id=h.log_id.iloc[0],model=model,common_duration_s=horizon,**{c:float(np.sqrt(np.mean(h[c]**2))) for c in ['height_error_m','cross_track_m','speed_error_m_s']}))
    common=pd.DataFrame(common);common.to_csv(out/'closed_loop_common_horizon.csv',index=False);means=common.groupby('model')[['height_error_m','cross_track_m','speed_error_m_s']].mean()
    response=pd.read_csv(out/'response_per_window.csv');previous=pd.read_csv(sources['raw_joint'][0]/'response_per_window.csv');previous=previous[previous.model=='joint_control_v1'];response=pd.concat([response,previous]);response['model']=response.model.replace({'expanded_baseline':'original','joint_control_v1':'raw_joint','actuator_only_joint_v1':'actuator_only'});r=response[response.eligible].groupby(['model','channel','horizon_s']).agg(n=('response','size'),sign_correct_fraction=('sign_correct','mean'),median=('response','median'));r.to_csv(out/'three_model_response.csv')
    direct=pd.read_csv(out/'direct_path_summary.csv');direct.to_csv(out/'direct_path_checked.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(12,8));ax=axes.flat
    for name in sources:
        h=summary[(summary.model==name)&(summary['mode']=='closed')].sort_values('case');ax[0].plot(h.case,h.duration_s,'o-',label=name)
    ax[0].set(xlabel='Case',ylabel='Diagnostic duration (s)',ylim=(0,21));ax[0].legend()
    for a,c in zip(list(ax)[1:],['common','differential','rudder']):
        for name in sources:
            h=r.reset_index();h=h[(h.model==name)&(h.channel==c)];a.plot(h.horizon_s,h['median'],'o-',label=name)
        a.axhline(0,color='gray');a.set(xlabel='Horizon (s)',ylabel='Median rate response / unit command',title=c);a.legend()
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160)
    report='''# 执行器独占控制通道的联合模型

本轮为固定协议结构实验：41个训练日志、17个开发验证日志，未打开封存测试；从头训练65轮、seed17、最后一轮检查点。不根据闭环表现调控制器、舵效幅值或时间常数。

## 时序证据与结构

训练日志的100ms增量相关峰中位数：共模-40ms、差模+80ms、rudder 0ms，各日志存在差异。这是闭环相关而非因果执行器延迟，不能将这些峰设为舵机时间常数。源日志各主题发布/采样时间及过去值保持年龄另存于 `../control_response_lags_v1/raw_topic_timing.csv`。部分展开数据没有保留采样时间，原日志检查补全了该诊断；执行器主题的timestamp_sample也不代表实际舵面运动。

历史26步状态与动作进入GRU初态。滚动时，主干的四个控制输入槽固定为训练均值（归一化后为零），新控制只更新驱动/舵面一阶代理；代理先输出、再更新，因此第一步物理量和GRU隐状态不受新指令影响。主干和执行器分支联合训练。直接舵效使用softplus约束共模→+q_dot、差模(left-right)/2→-p_dot、rudder→+r_dot；幅值仍由训练学习，没有强行设置正的最小舵效。该先验用于当前飞行数据覆盖范围，不保证失速、倒飞等范围成立。

驱动0.10s、舵机0.04s继续作为未标定代理参数；历史隐状态仍可能编码闭环偏差，动力学反馈与耦合也可能使较长响应反向。直接分支符号约束不等于全系统响应保证。相比上轮同时改变滚动输入路径和直接舵效参数化，不能分离两者贡献。

## 记录动作预测

所有2582个相同验证窗口，窗口级RMSE：

| 模型 | horizon(s) | 位置(m) | 速度(m/s) | 姿态(deg) | 角速度(rad/s) | 频率(Hz) |
|---|---:|---:|---:|---:|---:|---:|
'''
    for (name,h),row in agg.iterrows():report+='| '+name+f' | {h:.1f} | '+' | '.join(f'{row[c]:.4f}' for c in metrics)+' |\n'
    report+='\n## 控制方向、幅值与持续性\n\n每日志固定16个均匀窗口；相同完整初态，记录动作±0.02，越界整窗排除。模型内部探测，无真实干预真值。\n\n| 模型 | 通道 | horizon(s) | 有效数 | 方向正确率 | 响应中位数 |\n|---|---|---:|---:|---:|---:|\n'
    for (name,c,h),row in r.iterrows():
        if c=='motor' or h not in [.1,.5,1.]:continue
        report+=f'| {name} | {c} | {h:.1f} | {int(row.n)} | {row.sign_correct_fraction:.1%} | {row["median"]:.4f} |\n'
    report+='\n首步角加速度有限差分见direct_path_summary.csv，完整20ms至1s曲线见CSV和下图。直接符号约束是设计性质，不能作为独立真实物理验证。\n\n## IsaacLab 固定实飞控制链路\n\n同一个e624基线A和17个物理初态；各模型用自己的编码器重建隐状态。代理承担动力学，IsaacLab负责场景/状态接口。零风、水平地速作空速、50Hz模型和外环、400Hz内环保持观测等近似沿用。完成20s仅指未触发既定停止阈值。\n\n| 模型 | 模式 | 完成/17 | 时长中位数(s) | 舵面限幅占比均值 |\n|---|---|---:|---:|---:|\n'
    for (name,mode),row in stats.iterrows():report+=f'| {name} | {mode} | {int(row.completed)}/{int(row.n)} | {row.median_duration_s:.2f} | {row.surface_limit_fraction:.2%} |\n'
    report+='\n三模型每例共同存续时长内，先算各例RMSE再对案例等权平均：\n\n| 模型 | 高度(m) | 横向(m) | 速度(m/s) |\n|---|---:|---:|---:|\n'
    for name,row in means.iterrows():report+='| '+name+' | '+' | '.join(f'{v:.3f}' for v in row)+' |\n'
    report+='\n![对照](comparison.png)\n\n## 解读与边界\n\n短时预测、模型内部控制响应、模型内闭环分别评价。单次种子、已用于开发的验证集不能作为最终泛化或实机控制有效性的证明。新候选不自动替换原基线。训练权重、归一化、数据契约和源码哈希见protocol/manifest；仿真完成状态见isaac_closed_loop/verification.json；不以进程退出码代替落盘验证。\n'
    (out/'report.md').write_text(report)
    (out/'report_manifest.json').write_text(json.dumps(dict(dataset=json.loads((out/'manifest.json').read_text())['protocol'],script_sha256=sha(__file__),input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},test_opened=False),indent=2));print(agg.to_string());print(stats.to_string());print(means.to_string())
if __name__=='__main__':main()
