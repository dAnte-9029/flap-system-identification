#!/usr/bin/env python3
"""Derive the Step 2 report from a completed frozen benchmark (no new rollout)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib-main-v2-free-running')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from system_identification.training.trajectory_main_v1 import assemble_history_trajectory_windows
from system_identification.evaluation.main_v2_free_running import HORIZONS,PHYSICAL_FIELDS,magnitude


def table(frame):
    rows=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    for row in frame.itertuples(index=False,name=None):
        rows.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root',type=Path,default=ROOT/'docs/analysis/results/main_v2_free_running_5s')
    parser.add_argument('--report',type=Path,default=ROOT/'docs/results/2026-09-14-main-v2-free-running.md')
    args=parser.parse_args();root=args.results_root
    s=json.loads((root/'summary.json').read_text())
    if s['smoke_only']:
        raise ValueError('full report requires full benchmark')
    if s['window_count'] != 722 or s['stride_steps'] != 50:
        raise ValueError('this frozen-cohort report requires the 722-start, 50-step-stride contract')
    for path,digest in s['output_sha256'].items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'result changed: {path}')
    for path,digest in s['trajectory_files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'trace changed: {path}')
    dataset=ROOT/'dataset/trajectory_v1_august_f5_c4/samples_validation.parquet'
    if hashlib.sha256(dataset.read_bytes()).hexdigest()!=s['source_hashes'][str(dataset.relative_to(ROOT))]:
        raise ValueError('validation samples changed')
    output=root/'review'
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('refusing nonempty review directory')
    if args.report.exists():
        raise FileExistsError('refusing to overwrite existing report')
    output.mkdir(parents=True,exist_ok=True)
    windows=pd.read_csv(root/'windows.csv')
    batch=assemble_history_trajectory_windows(pd.read_parquet(dataset),windows,history_steps=26)
    files=sorted(Path(p) for p in s['trajectory_files'] if p.endswith('.npz'))
    loaded=[np.load(f) for f in files]
    ids=np.concatenate([f['window_ids'] for f in loaded])
    np.testing.assert_array_equal(ids,windows.window_id.to_numpy())
    pred={k:np.concatenate([f[k] for f in loaded]) for k in loaded[0].files if k not in {'window_ids'}}
    for f in loaded: f.close()
    variation=[]
    for horizon,k in HORIZONS.items():
        for name,key in [('velocity','velocity_n'),('body_rate','angular_velocity_b'),('frequency','flap_frequency_hz')]:
            a=np.std(pred[key][:,:k+1],axis=1);b=np.std(getattr(batch.trajectory.truth,key)[:,:k+1],axis=1)
            if a.ndim==2: a=magnitude(a);b=magnitude(b)
            ratio=a/np.maximum(b,1e-8)
            variation.append(dict(horizon_s=horizon,state=name,ratio_median=float(np.median(ratio)),
                                  ratio_p10=float(np.quantile(ratio,.1)),ratio_p90=float(np.quantile(ratio,.9)),
                                  below_half_count=int((ratio<.5).sum()),count=len(ratio)))
    variation=pd.DataFrame(variation);variation.to_csv(output/'prefix_variation.csv',index=False)
    env=json.loads((root/'observed_envelope.json').read_text())['train']
    support=[]
    for name,key in [('body_rate','angular_velocity_b'),('acceleration','acceleration_b')]:
        a=pred[key][:,1:] if name=='body_rate' else pred[key]
        a=magnitude(a);low=a<env[name]['min'];high=a>env[name]['max']
        anylow=low.any(axis=1)
        first=np.take_along_axis(np.cumsum(pred['dt_s'],axis=1),np.argmax(low,axis=1)[:,None],axis=1)[:,0]
        support.append(dict(state=name,below_train_min_rollouts=int(anylow.sum()),above_train_max_rollouts=int(high.any(axis=1).sum()),
                            earliest_low_time_s=float(first[anylow].min()),median_first_low_time_s=float(np.median(first[anylow])),
                            predicted_min=float(a.min()),predicted_max=float(a.max()),train_min=env[name]['min'],train_max=env[name]['max']))
    support=pd.DataFrame(support);support.to_csv(output/'support_failure_direction.csv',index=False)
    ph=pd.read_csv(root/'per_horizon.csv');pf=pd.read_csv(root/'per_flight.csv')
    per=pd.read_csv(root/'per_rollout.csv');tr=pd.read_csv(root/'trajectory_diagnostics.csv')
    warm=ph[ph.initialization=='warm26'].set_index('horizon_s')
    warm_rows=per[per.initialization=='warm26'];terminal=warm_rows[warm_rows.horizon_s==5]
    baseline=pd.read_csv(root/'constant_twist_baseline.csv').set_index('horizon_s')
    trace=tr[tr.initialization=='warm26']
    rubric={'all_5s_numeric_pass':not terminal.numerical_failed.any(),
            'all_5s_observed_support_pass':not terminal.envelope_failed.any(),
            'all_5s_no_clipping':not terminal.clipping_failed.any()}
    for name in ['velocity_m_s','attitude_deg','body_rate_rad_s','frequency_hz']:
        rubric['beats_constant_twist_every_horizon_'+name]=bool((warm[name+'_equal_log_rmse']<baseline[name+'_equal_log_rmse']).all())
    rubric['no_finite_but_wrong_suspects']=not trace[['unsupported_equilibrium_suspect','angular_damping_suspect','energy_growth_suspect']].any().any()
    rubric['external_absolute_accuracy_acceptance_met']=False
    rubric['counterfactual_action_response_validated']=False
    readiness=dict(structural_percent=100,dynamics_fidelity_percent=10*sum(rubric.values()),
                   rl_ready=False,fidelity_checklist={k:bool(v) for k,v in rubric.items()},
                   score_meaning='predeclared evidence checklist, not probability or safety certification')
    (output/'readiness.json').write_text(json.dumps(readiness,indent=2))
    # Plot each segment separately: no colored interpolation across invalid gaps.
    for log_id,g in warm_rows.groupby('log_id'):
        fig,axes=plt.subplots(4,1,figsize=(12,10),sharex=True)
        for ax,name in zip(axes,('velocity_m_s','attitude_deg','body_rate_rad_s','position_m')):
            norm=Normalize(float(g[name].min()),float(g[name].max()))
            for _,segment in g.groupby('segment_id'):
                pivot=segment.pivot(index='horizon_s',columns='start_time_s',values=name)
                times=pivot.columns.to_numpy()
                xedges=np.r_[times-.5,times[-1]+.5] if len(times)==1 else np.r_[times[0]-.5,(times[1:]+times[:-1])/2,times[-1]+.5]
                y=np.array([.05,.35,.75,1.5,2.5,4.,6.])
                mesh=ax.pcolormesh(xedges,y,pivot.to_numpy(),norm=norm,shading='flat')
            ax.set(ylabel='Horizon (s)',title=name,yticks=list(HORIZONS),ylim=(.05,6.))
            fig.colorbar(mesh,ax=ax)
        axes[-1].set_xlabel('Log timestamp (s); white gaps have no eligible start')
        fig.suptitle(log_id);fig.tight_layout();fig.savefig(output/f'heatmap_{Path(log_id).stem}.png',dpi=150);plt.close(fig)
    # Illustrative paired traces: choose worst 5s velocity error deterministically.
    worst=terminal.sort_values('velocity_m_s',ascending=False).iloc[0]
    idx=int(np.flatnonzero(ids==worst.window_id)[0]);t=np.r_[0,np.cumsum(pred['dt_s'][idx])]
    fig,axes=plt.subplots(4,1,figsize=(10,10),sharex=True)
    for ax,key,label in zip(axes,['velocity_n','angular_velocity_b','flap_frequency_hz','relative_phase_rad'],['NED velocity m/s','FRD body rate rad/s','Frequency Hz','Relative phase rad']):
        predicted=pred[key][idx]
        observed=getattr(batch.trajectory.truth,key)[idx]
        if predicted.ndim == 2:
            names=('x','y','z') if key=='velocity_n' else ('p','q','r')
            for axis,name in enumerate(names):
                ax.plot(t,predicted[:,axis],color=f'C{axis}',label=f'{name} predicted')
                ax.plot(t,observed[:,axis],color=f'C{axis}',linestyle='--',alpha=.65,label=f'{name} truth')
        else:
            ax.plot(t,predicted,label='predicted')
            ax.plot(t,observed,linestyle='--',alpha=.7,label='truth')
        ax.legend(ncol=3,fontsize=8,loc='best')
        ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[-1].set_xlabel('Elapsed native time (s)');fig.suptitle(worst.window_id);fig.tight_layout()
    fig.savefig(output/'worst_velocity_trace.png',dpi=150);plt.close(fig)
    horizon_table=warm[[n+'_equal_log_rmse' for n in ['position_m','velocity_m_s','attitude_deg','body_rate_rad_s','frequency_hz','phase_rad']]].reset_index()
    horizon_table.columns=['horizon s','position m','velocity m/s','attitude deg','body rate rad/s','frequency Hz','phase rad']
    dist=[]
    for n in ['position_m','velocity_m_s','attitude_deg','body_rate_rad_s','frequency_hz','phase_rad']:
        dist.append(dict(metric=n,**{k:float(warm.loc[5,n+'_'+k]) for k in ['mean','median','p90','p95','max']}))
    gate=pd.DataFrame([dict(horizon=h,valid=v['n_valid'],failed=v['n_failed'],failure_pct=v['failure_pct'],numeric_failure_pct=v['numerical_failure_pct'],truth_support_failure_pct=v['truth_envelope_failure_pct']) for h,v in s['gates'].items()])
    ablation=ph[ph.horizon_s.isin([.2,.5,1.,2.])][['initialization','horizon_s']+[n+'_equal_log_rmse' for n in ['position_m','velocity_m_s','attitude_deg','body_rate_rad_s']]]
    flights=pf[(pf.initialization=='warm26')&(pf.horizon_s==5)][['log_id','n_rollouts','failure_pct']+[n+'_equal_log_rmse' for n in ['position_m','velocity_m_s','attitude_deg','body_rate_rad_s']]]
    regimes=pd.read_csv(root/'regime_bins.csv');regimes=regimes[regimes.horizon_s==5][['variable','bin','n_rollouts','velocity_m_s_equal_log_rmse','attitude_deg_equal_log_rmse']]
    envelope_table=pd.DataFrame([dict(partition=partition,signal=name,**{k:v[k] for k in ['min','max','p1','p99']}) for partition,d in json.loads((root/'observed_envelope.json').read_text()).items() for name,v in d.items()])
    text=f'''# Main V2 stateful simulator：0.2–5 s validation 结果

2026-09-14。HEAD `{s['git_head']}`；冻结 Main V2 权重未改变。
本轮完成 stateful 包装、续跑一致性验证与正式 validation benchmark；未训练、未改网络/损失、未打开 sealed test。

## 七个明确回答

**Q1：能否包装成真正 stateful simulator？YES。** 最小充分状态已经显式化，固定权重/配置下，state + command + dt 唯一决定下一状态。warm reset、显式恢复、CPU tensor snapshot、step 均已实现。五条真实验证日志起点的连续 250 步与 100+150 步保存/恢复结果，所有物理状态、GRU、drive、tail、phase anchor 逐位一致；与原 forward 的六类物理输出最大差为 0。未来标签全替换 NaN 后结果逐位不变。

**Q2：5 s 是否数值稳定？YES，限本次 722 条 warm26 验证轨迹。** 非有限值和四元数范数 gate 失败均为 0；frequency、derivative、drive、residual 四类裁剪的事件数均为 0。不能外推到任意新控制策略或更长时域。

**Q3：5 s 动力学是否可信？NO，作为可供 RL 使用的动力学模型尚不可信。** 5 s 姿态 RMSE 45.53°、速度 RMSE 5.05 m/s，姿态 p95 88.65°；所有起点末段角速度变化均触发过度衰减诊断。722 条中 255 条（35.32%）超出训练 min/max 支持域，benchmark 明确 FAIL（退出码 2），并不是 NaN 爆炸。

**Q4：哪个 horizon 开始明显恶化？** 0.5→1→2 s 位置 RMSE 0.204→0.571→1.936 m，速度 0.691→1.156→2.096 m/s，姿态 6.43→10.51→18.01°；1–2 s 已出现明显累计恶化，2–5 s 扩大到位置 11.81 m、速度 5.05 m/s、姿态 45.53°。速度和姿态基本持续增长，不能声称存在一个已证实的突然发散时刻。没有给定绝对精度验收阈值，因此不能把 1 s 或 2 s 宣布为安全时域。

**Q5：最先坏掉的状态？** 若“坏”指匹配实飞动态的衰减，短至 0.2 s 已能看到速度变化不足（prefix std 比中位数 0.346），到 0.5 s 角速度 prefix std 比降至 0.450，1 s 为 0.348，末 1 s 仅 0.074。若按硬支持门槛，最早的失败状态/时间见下表，主要是角速度和加速度低于训练最小值，完全不是速度/角速度向上爆炸。频率 RMSE 在 0.5–5 s 约 0.19–0.20 Hz，不是首要增长源。相位 circular RMSE 从 0.226 增到 1.626 rad，说明周期同步也逐渐丢失。以上不构成某个状态“导致”另一状态漂移的因果证明。

**Q6：最差工况？** 使用连续变量分桶而不擅自命名机动：高 motor-command 组（86 起点）的 5 s 速度/姿态 RMSE 为 6.59 m/s / 59.69°；高 tail-command-norm 组（220 起点）为 6.01 / 55.51°；高控制变化组（241 起点）为 6.00 / 54.88°。较大 |roll| 组也较差（5.50 / 49.90°）。最差整条日志为 log_22（140 起点），6.54 m/s / 58.39°。这些是相关分层，存在 flight/控制/姿态共线，不能解释为单个控制通道的因果效应。

**Q7：下一步优先级：A > D > E > C > B。**

1. **A，保持架构做训练目标的受控对照**：当前权重只以 1 s 多步目标训练，1–5 s 的刚体漂移与角速度幅度压低突出。后续优先比较多 horizon 目标及角运动动态保真；本轮不训练，不能保证改 loss 后改善。
2. **D，检查/改进 aerodynamic 或 state-transition 建模**：频率已稳定但刚体误差持续积累，需区分导数偏差、周期动态损失与姿态反馈旋转误差；不直接跳到 Main V3。
3. **E，针对高 motor/tail、控制变化与跨日志分布补证据/数据**：这些组误差较大，且本次 replay 未验证新策略反事实动作响应；本轮不给未经验证的激励幅值。
4. **C，执行器模型不是首个改动**：频率误差没有随 horizon 明显增加，改 actuator 是否能修复刚体误差没有证据。
5. **B，不优先加长历史或新增 latent**：13 点与26点已很接近，冷启动差距随 horizon 缩小，而5 s大漂移仍在。

## 状态/API 与一致性

代码：`src/system_identification/models/main_v2_simulator.py`。
状态：p_n、v_n、q_nb、ω_b、frequency、relative phase、64维 GRU hidden、标量 drive proxy、三维 tail proxy、phase anchor。不重复存 body velocity、重力投影、相位 sin/cos 或导数。tail proxy 顺序为 symmetric/differential/rudder，不能称作实测舵角。

```python
sim = MainV2Simulator(frozen_model.eval())
state = sim.reset(**warm_inputs)       # history through t0 only
state, diagnostics = sim.step(state, command, dt)  # batched command [B,4], dt [B]
torch.save(state.snapshot(), "state.pt")
restored = SimulatorState.from_snapshot(
    torch.load("state.pt", weights_only=True), device=device)
state = sim.reset(state=restored)      # no history encoding or re-anchoring
```

Snapshot 需配合同一模型权重、归一化及固定时间常数。simulator 外部负责时钟；模型输出也包含观察状态，不需要单独复制 observation。旧 forward、训练缓存语义、checkpoint schema 均未修改。

## 样本与时间契约

722 个起点，五条 validation flight，完整26点历史，同一 segment 内250个未来控制/251个状态；每50个原生样本滚动一次。所有初始化方式用完全相同起点。不随机切分，不跨段，不插值。训练数据只用于观测范围/分桶，没有更新归一化或权重。

实际 dt min/max = {s['actual_dt_s']['min']:.6f}/{s['actual_dt_s']['max']:.6f} s，p1/p99 = {s['actual_dt_s']['p1']:.6f}/{s['actual_dt_s']['p99']:.6f}，median = {s['actual_dt_s']['median']:.6f}，mean = {s['actual_dt_s']['mean']:.9f}。可见局部约10/30 ms，并非每步严格20 ms。
250步实际时长 {s['gates']['5.0']['actual_horizon_s']['min']:.6f}–{s['gates']['5.0']['actual_horizon_s']['max']:.6f} s；历史实际约 {s['history_duration_s']['min']:.6f}–{s['history_duration_s']['max']:.6f} s。原 warm history filter 固定0.02 s保持不变，以维持旧评估语义。

已有八月2秒报告是3920个起点；本轮是满足完整历史与5秒未来的新722个共同起点。因此不能直接把两张表的微小差异解释为模型改善。本轮所有模型比较都使用这722个相同起点。重叠起点不是722次独立飞行。

## 误差曲线和尾部

以下为先每日志 RMSE、再等权平均；失败但有限的轨迹仍包含在统计中。

{table(horizon_table)}

5 s 逐起点分布（pooled，不是日志等权）：

{table(pd.DataFrame(dist))}

各 horizon 的 mean/median/p90/p95/max、horizontal/vertical position、vx/vy/vz、p/q/r 与各自 RMSE 均在 `per_horizon.csv`、`per_flight.csv`、`per_rollout.csv`。姿态为 geodesic，phase 为 atan2(sin Δφ, cos Δφ) 的绝对 circular error。

原 constant-twist 参考在5 s位置/速度/姿态/角速度/频率等权RMSE为18.73 m / 6.58 m/s /126.79° /1.152 rad/s /0.434 Hz。Main V2在全部horizon均优于这个基线，但这不等于满足绝对保真需求。

## Gate 与“有限但假”

{table(gate)}

支持域门槛取训练 min/max，不是适航安全范围；p1/p99越界单独记录。真实 validation 自身5 s越界比例为0.139%（1条），模型35.32%。本轮超域均来自低角速度/低加速度：

{table(support)}

低于原生差分加速度的最小值本身并不证明违反物理定律，噪声会抬高差分统计；所以这里使用“支持域失败”，不称“飞机不可能”。没有任何超过训练最高速度、最高角速度、最高频率/加速度/角加速度的模型路径。裁剪事件全部为0，因此不能归因于 clamp 勉强维持有限。

训练与validation实际观测范围：

{table(envelope_table)}

末1秒 predicted/truth variation ratio 中位数：velocity {trace.velocity_variation_ratio.median():.3f}、body rate {trace.body_rate_variation_ratio.median():.3f}、frequency {trace.frequency_variation_ratio.median():.3f}。角速度衰减 suspect {int(trace.angular_damping_suspect.sum())}/722；unsupported equilibrium suspect {int(trace.unsupported_equilibrium_suspect.sum())}/722；低控制变化下的 energy-growth suspect {int(trace.energy_growth_suspect.sum())}/722。

这些为事先声明的描述性筛查：不把 speed²/rate² 代理当作真实机械能，也不把实测信号的所有高频变化都当作真实气动振荡（可能包含估计/采样噪声）。两个 equilibrium suspect 仅指末段近稳态且落在训练支持域外，没有证明异常吸引子；11个energy-growth suspect没有伴随上包线或裁剪失败，不能单凭低command variation断言无能量输入。连续比值、漂移、能量代理增长与具体window ID见 `trajectory_diagnostics.csv`。

角速度 prefix variation 中位数逐渐减少，表明存在实测动态被抹平的系统性迹象，而非仅少数异常窗口：

{table(variation[variation.state=='body_rate'])}

## 历史依赖与冷启动

{table(ablation)}

13点覆盖约0.24 s；在0.2/0.5/1/2 s全部核心等权RMSE与26点的差异不超过1.6%。5点（约0.08 s）在0.2 s位置/速度/姿态退化约7.2%/7.1%/6.4%；single-point cold分别退化约9.6%/12.3%/12.6%。cold-zero短期更差，重复26次t0不能代替真实历史（0.2 s速度退化约22%）。

因此，在本checkpoint/五条验证日志上，**约0.24 s真实历史已接近0.5 s历史，历史主要改善最初0.2–0.5 s**；不能称0.5 s是必需。2 s后single/zero与warm的主要指标多在约1%内，但这是共享的长期误差主导，绝不是冷启动已解决动力学。标准benchmark仍采用warm26；未来RL reset可考虑burn-in，但本轮尚不启动RL。没有评估>26点历史，因此不宣称全局最优历史长度。

## Flight / 连续状态分桶

{table(flights)}

训练三分位边界（bin0低、bin1中、bin2高）：

```json
{json.dumps(s['bin_edges_train_tertiles'],indent=2)}
```

NED vertical speed为正向下。|body yaw rate|不是严格航迹转弯率，因此不把它直接命名为turning。5 s分桶结果如下；0.5/1/2 s对照保存在 `regime_bins.csv`。

{table(regimes)}

控制变化组按future command tape总变差离线划分，阈值记录在summary；这些未来量仅用于离线分析，绝不进入初始化/模型特征。相关分桶不能隔离因果效应或保证新控制策略安全。

## Readiness

- **Simulator structural readiness: {readiness['structural_percent']}%**，仅按本轮十项stateful API清单计分，不代表生产级RL系统完整度。
- **Dynamics fidelity readiness: {readiness['dynamics_fidelity_percent']}%**，按实验前声明的十项证据清单计分：数值通过、无裁剪、四类状态全horizon优于constant-twist共六项通过；全路径支持域、无有限但假suspect、外部绝对精度验收、反事实动作验证四项未通过/缺失。
- **RL-ready: NO。** 分数不是安全概率；绝对精度目标与反事实验证不能由相对基线收益代替。

## 复现、验证与产物

协议：`docs/contracts/2026-09-14_main_v2_stateful_benchmark.md`。结果目录：`{root}`。
完整warm轨迹与pause snapshot在summary的trajectory_files中，带SHA256；旧checkpoint/历史结果保持不变。
复现到新目录（benchmark预期以2退出表示门槛FAIL，结果已落盘；其他非零码应排查）：

```bash
/home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/run_main_v2_free_running.py \\
  --output-root /tmp/main-v2-reproduce/results --trajectory-root /tmp/main-v2-reproduce/traces
/home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest -q \\
  tests/test_main_v2_simulator.py tests/test_main_v2_free_running.py \\
  tests/test_trajectory_main_v1.py tests/test_trajectory_main_v2.py
```

默认完整基准会自动生成结果目录内的report.md；独立report脚本仅供已有且尚无review产物的完整结果使用。
本机CUDA不可用，实际在指定环境CPU执行；没有安装依赖或换解释器。相关trajectory/september回归56项通过；git diff --check另行记录于交付检查。

图：五项 `error_vs_horizon_*.png`；每flight的 `review/heatmap_*.png`显式保留segment间白色空档（优先看review版本）；`review/worst_velocity_trace.png`显示最差速度起点的预测与实测时序，选择规则固定为5 s最大velocity error。
'''
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(text)
    provenance=dict(summary_sha256=hashlib.sha256((root/'summary.json').read_bytes()).hexdigest(),
        report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        report_sha256=hashlib.sha256(args.report.read_bytes()).hexdigest(),
        output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()},
        sealed_test_opened=False)
    (output/'manifest.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps(readiness,indent=2));print(args.report)


if __name__=='__main__':
    main()
