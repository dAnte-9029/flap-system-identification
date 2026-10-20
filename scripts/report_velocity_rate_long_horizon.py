"""Long-horizon validation plots and Chinese report; no fitting or selection."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_velocity_rate_long_horizon import OUT,SOURCE,CACHE,HORIZONS,MODELS,METRICS
from report_velocity_rate_learning import table
COLORS=['#777777','#2878b5','#dd7622']
LABELS=['位置误差 (m)','NED速度误差 (m/s)','姿态测地角误差 (°)','FRD角速度误差 (rad/s)']
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':10})


def phase_diagnostic():
    meta=pd.read_parquet(OUT/'common_windows.parquet')
    columns=['flight_id','state_measurement_us','relative_flap_phase_sin','relative_flap_phase_cos',
        'flap_frequency_hz','valid_encoder','valid_frequency','target_relative_flap_phase_rad',
        'target_flap_frequency_hz','target_phase_valid','target_frequency_valid']
    samples=pd.read_parquet(ROOT/'dataset/velocity_rate_multistep_v1/samples_validation.parquet',columns=columns)
    groups=dict(tuple(samples.groupby('flight_id',sort=False))); rows=[]
    for fid,w in meta.groupby('flight_id',sort=False):
        s=groups[fid].reset_index(drop=True);origin=w.origin_sample.to_numpy(int)
        phi=np.arctan2(s.relative_flap_phase_sin.to_numpy()[origin],s.relative_flap_phase_cos.to_numpy()[origin])
        f=s.flap_frequency_hz.to_numpy()[origin];ts=s.state_measurement_us.to_numpy()
        for h in HORIZONS:
            end=origin+w[f'index_{h}ms'].to_numpy(int)+1
            elapsed=(ts[end]-ts[origin])*1e-6
            valid=s.valid_encoder.to_numpy()[origin]&s.valid_frequency.to_numpy()[origin]&s.target_phase_valid.to_numpy()[end]
            predicted=phi+2*np.pi*f*elapsed
            difference=predicted-s.target_relative_flap_phase_rad.to_numpy()[end]
            angle=abs(np.arctan2(np.sin(difference),np.cos(difference)))*180/np.pi
            fv=s.valid_frequency.to_numpy()[origin]&s.target_frequency_valid.to_numpy()[end]
            fd=abs(f-s.target_flap_frequency_hz.to_numpy()[end])
            rows.append(dict(flight_id=fid,horizon_ms=h,n_phase=int(valid.sum()),n_frequency=int(fv.sum()),
                phase_circular_mae_deg=angle[valid].mean(),frequency_hold_mae_hz=fd[fv].mean()))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'phase_extrapolation_per_flight.csv',index=False)
    return frame


def main():
    s=pd.read_csv(OUT/'summary.csv'); flight=pd.read_csv(OUT/'per_flight.csv'); native=pd.read_csv(OUT/'native_steps_per_flight.csv')
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,metric,label in zip(axes.flat,METRICS,LABELS):
        for model,color in zip(MODELS,COLORS):
            f=native[native.model==model].groupby(['flight_id','native_step'])[[metric,'actual_time_s']].mean().reset_index()
            g=f.groupby('native_step')[metric].agg(['mean','std']);t=f.groupby('native_step').actual_time_s.mean()
            ax.plot(t,g['mean'],label=model,color=color)
            ax.fill_between(t,np.maximum(0,g['mean']-g['std']),g['mean']+g['std'],alpha=.1,color=color)
        ax.axvline(.2,color='black',lw=1,ls='--'); ax.set(xlabel='实际递推时间 (s)',ylabel=label);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('200ms训练的冻结模型 → 2s自主递推；航次等权均值 ± 航次间标准差')
    fig.tight_layout();fig.savefig(figdir/'error_growth_2s.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(3,4,figsize=(15,10))
    for row,h in enumerate([500,1000,2000]):
        for col,(metric,label) in enumerate(zip(METRICS,LABELS)):
            ax=axes[row,col]; vals=[]
            for model in MODELS:
                vals.append(flight[(flight.model==model)&(flight.horizon_ms==h)].groupby('flight_id')[metric].mean().to_numpy())
            ax.boxplot(vals,tick_labels=['Baseline','NoCtrl','Control'])
            for i,v in enumerate(vals):ax.scatter(np.full(len(v),i+1)+np.linspace(-.1,.1,len(v)),v,s=9,color=COLORS[i],alpha=.6)
            ax.set(title=f'{h/1000:g}s · {label}');ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(figdir/'flight_distributions.png',dpi=170);plt.close(fig)
    ex=np.load(OUT/'trajectory_examples.npz');meta=pd.read_csv(OUT/'example_windows.csv')
    fig=plt.figure(figsize=(14,9))
    for i in range(6):
        ax=fig.add_subplot(2,3,i+1,projection='3d');mask=ex['future_mask'][i];start=ex['initial'][i,:3]
        truth=np.vstack((start,ex['target'][i,mask,:3]));ax.plot(truth[:,1],truth[:,0],-truth[:,2],color='black',lw=2,label='观测')
        for model,color in zip(MODELS,COLORS):
            seed=-1 if model==MODELS[0] else 17;p=np.vstack((start,ex[model.replace(' ','_')+f'_seed_{seed}'][i,mask,:3]))
            ax.plot(p[:,1],p[:,0],-p[:,2],color=color,label=model)
        ax.set(xlabel='东 (m)',ylabel='北 (m)',zlabel='上 (m)',title=f'航次 {meta.flight_id.iloc[i][:10]}')
    fig.axes[0].legend(fontsize=7);fig.suptitle('固定六航次中间窗口，连续2s自主递推，无中途真值重置；展示种子17')
    fig.tight_layout();fig.savefig(figdir/'trajectory_examples_2s.png',dpi=160);plt.close(fig)
    seeds=pd.read_csv(OUT/'per_seed.csv')
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for ax,metric,label in zip(axes.flat,METRICS,LABELS):
        for seed in [17,29,43]:
            a=seeds[(seeds.model==MODELS[1])&(seeds.seed==seed)].set_index('horizon_ms')[metric]
            b=seeds[(seeds.model==MODELS[2])&(seeds.seed==seed)].set_index('horizon_ms')[metric]
            ax.plot(a.index/1000,100*(a-b)/a,marker='o',label=f'seed {seed}')
        ax.axhline(0,color='k',lw=.7);ax.set(xlabel='时域 (s)',ylabel='加入控制的相对改善 (%)',title=label);ax.grid(alpha=.2)
    axes[0,0].legend();fig.tight_layout();fig.savefig(figdir/'control_gain_by_seed.png',dpi=170);plt.close(fig)
    phase=phase_diagnostic();pg=phase.groupby('horizon_ms')[['phase_circular_mae_deg','frequency_hold_mae_hz']].agg(['mean','std'])
    pg.columns=['_'.join(c) for c in pg.columns];pg.to_csv(OUT/'phase_extrapolation_summary.csv')
    audit=json.loads((OUT/'data_audit.json').read_text());obs=pd.read_csv(OUT/'observation_per_flight.csv')
    pairs=pd.read_csv(OUT/'paired_comparisons.csv')
    interpretation=(OUT/'interpretation_zh.md').read_text() if (OUT/'interpretation_zh.md').exists() else ''
    lines=['# GRU64冻结模型：500ms、1s、2s验证', '',
        '**本轮没有重新训练。** 使用第一轮200ms训练并选好的6个checkpoint，保持500ms历史，连续递推至2s。测试集封存。', '',
        interpretation, '', '## 验证时到底给了什么', '',
        '| 输入 | GRU-NoControl | GRU-Control |','|---|---|---|',
        '| 过去500ms NED速度、FRD角速度、姿态 | 有 | 有 |',
        '| 有效历史相位/频率、mask、历史与未来实际dt | 有 | 有 |',
        '| 初始位置 | 仅积分初值，网络不使用绝对位置 | 同左 |',
        '| 历史四路执行器命令 | 无 | 有 |',
        '| 已记录未来控制事件及持续时间 | 无 | 有 |',
        '| 未来真实速度、角速度、姿态或位置 | 仅评价目标，不进模型 | 同左 |',
        '| 未来真实相位/频率 | 无；保持起点频率、线性推进相位 | 同左 |','',
        'Control并非只用“当前运动状态＋未来控制”；还用历史控制与历史扑翼观测。两模型都只在起点初始化一次，之后状态和隐藏状态完全自主递推。未来实际dt是共同的离线时间条件。未来命令来自闭环日志，可能包含控制器对未来观测的反馈信息；更长时域下这种条件回放的预测优势仍不是反事实控制能力。','',
        '## 数据与共同起点','',
        f'数据为注册的velocity_rate_multistep_v1/main_50hz。所有时域和模型使用共同的 **{audit["common_windows"]:,}窗口、{audit["common_flights"]}航次**；先求100/200/500/1000/2000ms注册窗口起点的交集，再从2s序列中取各注册端点。没有按模型误差筛选窗口。', '',
        table(pd.DataFrame([dict(horizon_ms=h,**rec) for h,rec in audit['counts'].items()])), '',
        '因此本报告100/200ms数值是共同长窗口起点的重算，不应与第一轮36,970窗口数值直接混同。原生dt、99/100步有效mask和全部控制事件保留；端点距名义时域不超过12ms。原生逐步曲线按实际时间绘制，每步支持量保存在native_steps_per_flight.csv。','',
        '## 端点结果','',
        '数值为向量模长/姿态测地角的 **均值 ± 航次间样本标准差**：先在航次内平均窗口、再平均三个种子、最后27航次等权。不是逐轴RMSE；重叠窗口不是独立飞行。窗口等权、P95/P99、种子间标准差另见summary.csv。','']
    rows=[]
    for h in HORIZONS:
        for model in MODELS:
            f=s[(s.model==model)&(s.horizon_ms==h)].set_index('metric')
            rows.append([h/1000,model]+[f'{f.loc[m,"flight_mean"]:.4f} ± {f.loc[m,"flight_std"]:.4f}' for m in METRICS])
    lines += [table(pd.DataFrame(rows,columns=['时域s','模型',*LABELS])),'','## Control相对NoControl','',
        '区间为航次配对bootstrap 95%区间（5000次，种子先平均）；同日期航次可能相关，不能将窗口数量当作独立样本量。','']
    c=pairs[(pairs.reference==MODELS[1])&(pairs.horizon_ms>=500)].copy()
    c['更好航次']=c.better_flights.astype(str)+'/27';c['改善%']=c.improvement_pct.map(lambda v:f'{v:.2f}')
    c['误差差值区间']=c.apply(lambda r:f'[{r.bootstrap_low:.5f}, {r.bootstrap_high:.5f}]',axis=1)
    lines += [table(c[['horizon_ms','metric','改善%','更好航次','误差差值区间']]),'',
        '## 积分与相位外推参照','',
        '真实未来v/ω仅用于以下独立诊断：从相同初态进行同一积分，观察它与EKF目标的不一致。它不是严格误差下界，不能将误差模长相加作方差归因。','']
    rows=[]
    for h,g in obs.groupby('horizon_ms'):
        rows.append([h,f'{g.position_m.mean():.4f} ± {g.position_m.std():.4f}',f'{g.attitude_deg.mean():.4f} ± {g.attitude_deg.std():.4f}'])
    lines += [table(pd.DataFrame(rows,columns=['时域ms','观测积分位置残差m','观测积分姿态残差°'])),'',
        '另用真实未来相位/频率作事后诊断，不送入网络。因果外推采用起点频率保持、相位线性推进，其相位圆周MAE及频率保持MAE如下。此诊断不能单独证明模型误差的根因；相位为日志相对坐标，并非跨日志绝对机械零位。','',
        table(pg.round(4).reset_index()),'','## 图表','',
        '![误差增长](figures/error_growth_2s.png)','','![航次分布](figures/flight_distributions.png)','',
        '![种子控制收益](figures/control_gain_by_seed.png)','','![2s轨迹](figures/trajectory_examples_2s.png)','',
        '轨迹窗口在评价前按航次顺序等间隔选取六航次的中间起点，不挑选最好结果；所有种子同窗预测已保存。','',
        '## 验证与边界','',
        '- checkpoint、原训练归一化及第一轮全部归档文件哈希已核对，未修改；无模型选择、调参或重新训练。',
        '- 100步前向的前缀不变性、未来真值隔离、NoControl控制隔离通过定向测试。GPU全体窗口输出有限；独立SciPy重算与数据loader对照见numerical_verification.json及verification.json。',
        '- 归一化来自第一轮训练，未用长时域验证统计重新拟合。固定评估batch=2048、TF32设置沿用第一轮。',
        '- 训练只覆盖200ms，500ms/1s/2s分别是2.5/5/10倍训练时域外推；这不等价于用这些时域的损失正式训练后的结果。',
        '- 27航次未参加训练，但曾用于第一轮200ms checkpoint选择；本轮是验证诊断，不是新的独立测试结论。',
        '- 高动态、REVIEW和模式变化分组采用整个2s窗口标记，属于事后分组，不进入模型。',
        '- 逐窗口五时域端点指标归档在metrics/；完整逐原生步误差NPZ及打包缓存保存在artifacts/velocity_rate_gru64_long_horizon_v1，并由归档清单记录哈希。',
        '- 不开展MPC/PX4修改，不解封测试集，不宣称控制可用。','']
    (OUT/'report_zh.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
