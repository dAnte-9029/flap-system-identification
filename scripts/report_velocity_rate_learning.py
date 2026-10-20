"""Deterministic plots and Chinese report from frozen validation artifacts."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/velocity_rate_gru64_round1'
METRICS=['position_m','velocity_m_s','attitude_deg','rate_rad_s']
LABELS=['位置误差 (m)','NED速度误差 (m/s)','姿态测地角误差 (°)','FRD角速度误差 (rad/s)']
MODELS=['Kinematic Baseline','GRU-NoControl','GRU-Control']
COLORS=['#777777','#2878b5','#dd7622']
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':10})


def table(frame):
    columns=list(frame.columns)
    lines=['| '+' | '.join(map(str,columns))+' |','|'+'|'.join(['---']*len(columns))+'|']
    for row in frame.itertuples(index=False,name=None): lines.append('| '+' | '.join(map(str,row))+' |')
    return '\n'.join(lines)


def main():
    config=json.loads((EXP/'config.json').read_text())
    summary=pd.read_csv(EXP/'summary.csv'); flight=pd.read_csv(EXP/'per_flight.csv')
    comparisons=pd.read_csv(EXP/'paired_comparisons.csv'); groups=pd.read_csv(EXP/'group_metrics.csv')
    oracle=pd.read_csv(EXP/'observation_per_flight.csv')
    dynamics=pd.read_csv(EXP/'dynamics_vs_oracle_per_flight.csv')
    figs=EXP/'figures'; figs.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for ax,metric,label in zip(axes.flat,METRICS,LABELS):
        for model,color in zip(MODELS,COLORS):
            r=summary[(summary.model==model)&(summary.metric==metric)].sort_values('horizon_ms')
            ax.plot(r.horizon_ms,r.flight_mean,label=model,color=color)
            ax.fill_between(r.horizon_ms,np.maximum(0,r.flight_mean-r.flight_std),r.flight_mean+r.flight_std,alpha=.12,color=color)
        ax.set(xlabel='名义预测时域 (ms)',ylabel=label); ax.axvline(100,color='k',lw=.6,ls=':'); ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8); fig.suptitle('验证集：按航次等权均值；阴影为航次间样本标准差（种子先平均）')
    fig.tight_layout(); fig.savefig(figs/'error_growth.png',dpi=170); plt.close(fig)
    native_rows=[]
    for model in MODELS:
        for seed in ([-1] if model==MODELS[0] else config['seeds']):
            path=EXP/f'metrics/{model.replace(" ","_")}_seed_{seed}_native_steps.parquet'
            raw=pd.read_parquet(path)
            raw[METRICS]=raw[METRICS].astype(float)
            g=raw.groupby(['flight_id','native_step'])[[*METRICS,'actual_time_s']].mean().reset_index()
            g['n_windows']=raw.groupby(['flight_id','native_step']).size().to_numpy()
            g['model']=model; g['seed']=seed; native_rows.append(g)
    native=pd.concat(native_rows,ignore_index=True)
    native.to_csv(EXP/'native_steps_per_flight.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for ax,metric,label in zip(axes.flat,METRICS,LABELS):
        for model,color in zip(MODELS,COLORS):
            f=native[(native.model==model)&(native.native_step<=10)]
            g=f.groupby(['flight_id','native_step'])[metric].mean().groupby('native_step').agg(['mean','std'])
            ax.plot(g.index,g['mean'],marker='.',color=color,label=model)
            ax.fill_between(g.index,np.maximum(0,g['mean']-g['std']),g['mean']+g['std'],color=color,alpha=.1)
        ax.set(xlabel='原生递推步数（使用各窗口实际dt）',ylabel=label); ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('同一批36,970验证窗口的前10个原生步骤；第11步仅54窗口，完整数值另存CSV')
    fig.tight_layout(); fig.savefig(figs/'native_step_growth.png',dpi=170); plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(15,7))
    for row,h in enumerate((100,200)):
        for col,(metric,label) in enumerate(zip(METRICS,LABELS)):
            ax=axes[row,col]
            vals=[flight[(flight.model==m)&(flight.horizon_ms==h)].groupby('flight_id')[metric].mean().to_numpy() for m in MODELS]
            ax.boxplot(vals,tick_labels=['Baseline','NoCtrl','Control'],showfliers=True)
            for i,v in enumerate(vals): ax.scatter(np.full(len(v),i+1)+np.linspace(-.1,.1,len(v)),v,s=9,color=COLORS[i],alpha=.6)
            ax.set(title=f'{h} ms · {label}',ylabel='单航次平均端点误差'); ax.grid(axis='y',alpha=.2)
    fig.tight_layout(); fig.savefig(figs/'flight_distributions.png',dpi=170); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    training=[]
    for model,color in zip(MODELS[1:],COLORS[1:]):
        for seed in config['seeds']:
            folder=EXP/model/f'seed_{seed}'
            log=pd.read_json(folder/'training.jsonl',lines=True)
            training.append(json.loads((folder/'summary.json').read_text()))
            axes[0].plot(log.epoch,log.train_window_loss,color=color,alpha=.65,label=f'{model} s{seed}')
            axes[1].plot(log.epoch,log.validation_flight_loss,color=color,alpha=.65,label=f'{model} s{seed}')
    for ax,title in zip(axes,['训练窗口等权损失','验证航次等权损失']):
        ax.set(xlabel='Epoch',ylabel='归一化多步损失',title=title); ax.grid(alpha=.2)
    axes[1].legend(fontsize=7); fig.tight_layout(); fig.savefig(figs/'training_curves.png',dpi=170); plt.close(fig)
    pd.DataFrame(training).to_csv(EXP/'training_budget.csv',index=False)
    ex=np.load(EXP/'trajectory_examples.npz'); metadata=pd.read_csv(EXP/'trajectory_example_windows.csv')
    fig=plt.figure(figsize=(14,9))
    for i in range(6):
        ax=fig.add_subplot(2,3,i+1,projection='3d'); mask=ex['mask'][i]
        start=ex['initial'][i,:3]
        true=np.vstack((start,ex['target'][i,mask,:3]))
        ax.plot(true[:,1],true[:,0],-true[:,2],color='black',lw=2,label='观测')
        for model,color in zip(MODELS,COLORS):
            seed=-1 if model==MODELS[0] else 17
            p=np.vstack((start,ex[model.replace(' ','_')+f'_seed_{seed}'][i,mask,:3]))
            ax.plot(p[:,1],p[:,0],-p[:,2],color=color,label=model)
        ax.set(xlabel='东 (m)',ylabel='北 (m)',zlabel='上 (m)',title=f'航次 {metadata.flight_id.iloc[i][:10]}')
    fig.axes[0].legend(fontsize=7); fig.suptitle('预先固定六航次的中间窗口：200 ms 真实与预测轨迹；展示种子17')
    fig.tight_layout(); fig.savefig(figs/'trajectory_examples.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(15,6))
    for row,i in enumerate((1,4)):
        mask=ex['mask'][i]; t=np.r_[0,np.cumsum(ex['dt'][i])[mask]]*1000
        values=[np.vstack((ex['initial'][i],ex['target'][i,mask]))]
        for m in MODELS:
            seed=-1 if m==MODELS[0] else 17
            values.append(np.vstack((ex['initial'][i],ex[m.replace(' ','_')+f'_seed_{seed}'][i,mask])))
        for value,name,color in zip(values,['观测']+MODELS,['black']+COLORS):
            for col,j in enumerate((3,5,10,11)):
                axes[row,col].plot(t,value[:,j],color=color,label=name)
        for col,label in enumerate(['v_N (m/s)','v_D (m/s)','ω_x (rad/s)','ω_y (rad/s)']):
            axes[row,col].set(xlabel='实际时间 (ms)',ylabel=label,title=str(metadata.flight_id.iloc[i])[:10]); axes[row,col].grid(alpha=.2)
    axes[0,0].legend(fontsize=7); fig.tight_layout(); fig.savefig(figs/'state_examples.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    for ax,h in zip(axes,(100,200)):
        g=groups[groups.horizon_ms==h].copy(); g['label']=g.grouping+':'+g.group
        no=g[g.model=='GRU-NoControl'].set_index('label'); co=g[g.model=='GRU-Control'].set_index('label')
        fields=[m+'_mean' for m in METRICS]
        a=100*(no[fields]-co[fields])/no[fields]
        im=ax.imshow(a.to_numpy(),cmap='RdYlGn',vmin=-20,vmax=20,aspect='auto')
        ax.set_xticks(range(4),['位置','速度','姿态','角速度']); ax.set_yticks(range(len(a)),a.index)
        ax.set_title(f'{h} ms: 加入控制的相对改善 (%)')
        for y in range(len(a)):
            for x in range(4): ax.text(x,y,f'{a.iloc[y,x]:.1f}',ha='center',va='center',fontsize=8)
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.7); fig.savefig(figs/'condition_groups.png',dpi=170,bbox_inches='tight'); plt.close(fig)
    norm=json.loads((EXP/'normalization.json').read_text()); contract=json.loads((EXP/'evaluation_contract.json').read_text())
    manifest=json.loads((EXP/'run_manifest.json').read_text()); audit=json.loads((EXP/'data_audit.json').read_text())
    lines=['# 第一轮 Velocity & Angular-Rate Multi-Step Dynamics Learning', '',
           '本报告只涉及 train/validation；测试 Parquet 未加载、未哈希、未评价。所有预测都是自主递推。', '',
           (EXP/'interpretation_zh.md').read_text(), '',
           '## 主要结果', '',
           '下表为验证集 **端点向量误差模长 / SO(3)测地角的均值 ± 航次间样本标准差**。先在每航次内平均窗口、再平均三个种子，最后对 27 航次等权；不是逐轴RMSE，也不是将重叠窗口当独立飞行。', '']
    rows=[]
    for h in (100,200):
        for model in MODELS:
            s=summary[(summary.model==model)&(summary.horizon_ms==h)].set_index('metric')
            rows.append([h,model]+[f'{s.loc[m,"flight_mean"]:.4f} ± {s.loc[m,"flight_std"]:.4f}' for m in METRICS])
    lines += [table(pd.DataFrame(rows,columns=['时域ms','模型',*LABELS])),'',
              '窗口等权与随机种子间波动分别见 `summary.csv` 的 `window_mean/window_std` 和 `seed_mean/seed_std`；原始逐航次见 `per_flight.csv`。Baseline不训练，因此只有一份结果。','',
              '### 配对比较', '', '负的误差差值表示候选更好。区间为按航次配对bootstrap 95%区间（5000次）；同日期航次仍可能相关，不能把它当严格独立飞行试验的显著性结论。','']
    c=comparisons[(comparisons.candidate=='GRU-Control')].copy()
    c['改善%']=c.relative_improvement_pct.map(lambda x:f'{x:.2f}')
    c['更好航次']=c.better_flights.astype(str)+'/27'
    c['差值95%区间']=c.apply(lambda r:f'[{r.bootstrap_low:.5f}, {r.bootstrap_high:.5f}]',axis=1)
    lines += [table(c[['horizon_ms','reference','metric','改善%','更好航次','差值95%区间']]),'',
              '## 数据、因果边界和模型', '',
              f'- 数据版本 `{config["dataset_id"]}`，组 `main_50hz`；manifest SHA256 `{audit["manifest_sha256"]}`。注册条目和全部 train/validation 文件哈希见 `data_audit.json`。',
              '- 94训练航次110,863窗口，27验证航次36,970窗口；500ms历史、200ms训练预测、100ms起点步长。保留全部原生步长，历史26/27点，未来10/11步；未重采样或丢弃11步窗口。',
              '- 100/200ms评估选最近原生目标，偏差最多12ms，不插值观测。辅助曲线点容差15ms（160ms处112个窗口超过12ms，最大12.170ms）；没有因此筛掉窗口。所有模型与时域用同一批200ms窗口；逐步原生时间和误差均已保存。',
              '- 输入使用历史NED速度、FRD角速度、FRD到NED旋转矩阵前两列、有效相位sin/cos、频率、显式mask和实际dt。绝对NED位置不进入网络；四元数符号不进入姿态特征。',
              '- 历史相位保持该日志原点下的有效相对相位，不声称跨日志绝对零位已标定。未来频率取起点有效历史频率并保持；相位按该频率线性推进。起点频率或相位无效时屏蔽未来相位。无效频率置零并带mask。',
              '- 训练有效历史相位/频率覆盖率98.93%，有1,181个窗口起点无效；验证覆盖率100%，所以本轮未实证验证缺相位情况下的泛化性能。mask代码有合成变异测试，覆盖率详见mask_coverage.json。',
              '- GRU64历史编码器 + GRUCell64递推器 + 6维线性头，采用当前v/ω的identity skip预测下一步v/ω，不训练加速度标签。位置梯形积分，姿态右乘机体系平均角速度的SO(3)指数，Hamilton wxyz每步归一化。',
              '- 控制模型历史额外读取四路命令；每个未来状态区间内的完整ZOH事件段按顺序送入GRU16（4路命令+持续时间），并在原生状态边界预测。边界事件属于右侧区间。没有把整个区间命令均值冒充起点因果输入。事件编码是条件序列特征，并不声称区间内部状态动力学已被解析积分。',
              '- NoControl完全不访问历史四路命令、事件内容、事件数量或控制变化标记；相同非控制输入、相同主干和相同参数初始化。为严密配对，保留相同张量形状与不活动的控制参数，其梯度不更新。',
              '- 未来真实状态、真实相位/频率只属于监督/诊断，forward接口不接收它们；无teacher forcing。未来实际dt作为共同采样时刻条件，并非已验证在线时钟预测。',
              '- 闭环日志中的未来控制是已记录命令；控制器可能响应未来扰动/状态，因此该对照检验控制历史与未来命令的总体预测信息，不能单独证明反事实响应、独立因果作用或控制可用性。', '',
              '## 损失与训练预算', '',
              f'四项权重均为1；每步用平方向量模长/平方测地角除以训练集RMS尺度平方，再对每窗口有效原生步等权平均、对训练窗口等权平均。位置用相对初始位置的位移。尺度依次为 {norm["loss_scales"]}（m, m/s, rad, rad/s），来自训练窗口恒定运动学残差；输入和输出尺度也仅由训练集估计。没有强制修改EKF观测以满足积分。', '',
              'AdamW，初始学习率0.001，weight decay=0.0001，24轮余弦调度，batch=1024，梯度裁剪5；每轮遍历全部训练窗口。两模型同一种子同一轮使用完全一致的随机置换。验证按航次等权的归一化多步损失选checkpoint，patience=5、min_delta=1e-4。', '',
              table(pd.DataFrame(training)[['model','seed','parameters','gradient_parameters','epochs','updates','best_epoch','best_validation_flight_loss','seconds','stop_reason']].round(5)), '',
              '两者均分配37,590参数。NoControl的事件GRU（1,104参数）不执行；历史控制输入列768参数及递推事件输入列3,072参数对输出无作用，因此功能上参与预测的是32,646参数，Control为37,590。`gradient_parameters`只是拥有梯度张量的参数数（仍包括零输入列），不是精确有效容量。主干宽度与预算保持一致，但不能声称有效容量完全相等。三种子全部完成，不挑选最好种子作主结果。各模型各种子的best.pt、训练日志和哈希均已保存。', '',
              f'训练代码提交 `{manifest["training_code_commit"]}`，起点 `{manifest["base_commit"]}`。GPU与显存信息见 `runtime.json`、各训练summary。GPU smoke含有限梯度、自主递推、平移不变性和NoControl控制隔离；测试覆盖坐标、四元数符号、零误差梯度、mask、边界事件以及数据打包与官方loader一致性。', '',
              '## 观测积分残差与归因', '',
              '以下诊断从与模型完全相同的可用初态出发，使用未来真实v/ω做同一运动学积分。它包含EKF更新、异步姿态/角速度观测与离散积分误差，属于观测链一致性参照，不是传感器精度、严格预测误差下界或唯一根因。','']
    rows=[]
    for h in (100,200):
        o=oracle[oracle.horizon_ms==h]
        for field in ('position_m','attitude_deg'):
            r=[h,field,f'{o[field].mean():.5f} ± {o[field].std(ddof=1):.5f}']
            for model in MODELS:
                d=dynamics[(dynamics.horizon_ms==h)&(dynamics.model==model)].groupby('flight_id')[field].mean()
                r.append(f'{d.mean():.5f} ± {d.std(ddof=1):.5f}')
            rows.append(r)
    lines += [table(pd.DataFrame(rows,columns=['时域ms','指标','观测积分→观测',*[m+'→观测积分' for m in MODELS]])), '',
              '位置向量上有精确分解：预测−观测 = (预测−观测积分) + (观测积分−观测)，但两部分可抵消，模长不能直接相加或当作独立方差占比。姿态只能比较旋转距离并用三角不等式解释；没有将两类误差当作独立噪声。v/ω端点误差本身来自对日志目标的预测；积分额外影响p/R。', '',
              '## 航次与工况差异', '',
              '所有分组均为事后诊断，不作为模型输入或调参依据。控制变化强度是历史+未来事件四通道峰峰值的L2范数，低/中/高阈值只由训练窗口三分位数确定。高动态和REVIEW采用数据集既有全窗口标记。组内先按航次等权；各组样本数与航次数见 `group_metrics.csv`，少航次组不可过度解读。', '',
              '## 图表', '',
              '![误差增长](figures/error_growth.png)', '', '![航次分布](figures/flight_distributions.png)', '',
              '![原生逐步误差](figures/native_step_growth.png)', '',
              '![工况分组](figures/condition_groups.png)', '', '![训练曲线](figures/training_curves.png)', '',
              '![真实与预测轨迹](figures/trajectory_examples.png)', '', '![状态预测](figures/state_examples.png)', '',
              '轨迹展示固定选六个按数据顺序均匀分布航次的中间窗口，未按模型误差挑选；展示种子17，其余种子的同窗预测也保存在trajectory_examples.npz。', '',
              '## 复现与归档', '',
              '独立复算见 `verification.json`：全部验证窗口基线使用NumPy/SciPy闭式积分核对；六个checkpoint各复跑182窗口（含全部54个11步窗口）；逐窗口→逐航次汇总独立复核。固定batch=2048时checkpoint指标最大差异约4.3e-6；航次汇总差异在1e-15量级。25项定向测试通过，2项原有全数据QA未启用，未触及测试Parquet。', '',
              '数值复现必须保持评估batch=2048及本轮TF32设置。改变batch形状时，GPU内核的数值路径不同；所查窗口的四项指标最大变化依次约为6.96e-5m、9.09e-4m/s、0.00623°、0.00118rad/s，已记录于verification.json。所有主比较使用相同固定batch协议；不能混用不同数值设置解释极小的平动增益。', '',
              '使用 `/home/zn/anaconda3/envs/flap-train-gpu/bin/python`，依次执行 `scripts/prepare_velocity_rate_learning.py`、`scripts/train_velocity_rate_learning.py --smoke-only`、`scripts/train_velocity_rate_learning.py`、`scripts/evaluate_velocity_rate_learning.py`、`scripts/report_velocity_rate_learning.py`。正式训练目录非空时拒绝覆盖；复现实验应在新的输出目录/独立checkout进行。', '',
              '缓存NPZ位于被git忽略的artifacts目录，可以由注册数据重建；其哈希已归档。逐窗口端点与逐原生步误差以Parquet保存；逐航次、种子、分组、配对比较为CSV；checkpoint均随本地实验提交。原数据、历史实验和主工作区未改动。未推送。', '']
    (EXP/'report_zh.md').write_text('\n'.join(lines))

if __name__=='__main__': main()
