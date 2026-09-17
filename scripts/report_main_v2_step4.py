#!/usr/bin/env python3
"""Produce evidence-scoped Step 4 findings; no checkpoint promotion."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-main-v2-step4')
import argparse,json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from main_v2_step4_tools import harmonic_design


def md(f):
    s=['| '+' | '.join(map(str,f.columns))+' |','| '+' | '.join(['---']*len(f.columns))+' |']
    for row in f.itertuples(index=False,name=None):s.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(s)


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);args=pa.parse_args();r=args.output
    assert (r/'probe_complete.json').exists()
    h=pd.read_csv(r/'harmonic_regression.csv');c=pd.read_csv(r/'conditional_variance.csv');t=pd.read_csv(r/'teacher_aggregate.csv');p=pd.read_csv(r/'probe_metrics.csv');sen=pd.read_csv(r/'command_sensitivity.csv');tel=pd.read_csv(r/'telemetry_fields.csv');integ=pd.read_csv(r/'integration_consistency.csv');dur=pd.read_csv(r/'actual_durations.csv');off=pd.read_csv(r/'train_phase_offsets.csv');phase=pd.read_csv(r/'teacher_phase_component.csv')
    # Equal-log summaries; three diagnostic seeds remain visible in full CSV.
    harm=h[h.order==3].groupby(['split','representation','signal'])[['fit_r2','heldout_r2']].mean().reset_index();harm.to_csv(r/'harmonic_summary.csv',index=False)
    ps=p.groupby(['scope','model','evaluation_target','signal'])[['rmse','r2','pred_std','truth_std']].mean().reset_index();ps.to_csv(r/'probe_summary.csv',index=False)
    pc=p.groupby(['scope','model','seed','evaluation_target','signal']).rmse.mean().reset_index();changes=[]
    for scope in ['train_temporal_holdout','validation_unknown_offset']:
        for ca,base in [('B1','B0'),('B2','B0'),('B3','B1')]:
            target='increment2' if ca in ['B1','B3'] else 'D0_raw'
            for sig in ['linear','angular']:
                g=pc[(pc.scope==scope)&(pc.evaluation_target==target)&(pc.signal==sig)].pivot(index='seed',columns='model',values='rmse');v=100*(g[ca]/g[base]-1)
                changes.append(dict(scope=scope,candidate=ca,reference=base,target=target,signal=sig,mean_change_pct=v.mean(),seed_min=v.min(),seed_max=v.max()))
    changes=pd.DataFrame(changes);changes.to_csv(r/'probe_paired_changes.csv',index=False)
    # Raw synchronous teacher prediction curves on every log, no selected best case.
    for split in ['train','validation']:
        m=pd.read_csv(r/(split+'_points.csv'));a=np.load(r/(split+'_diagnostic_arrays.npz'))
        for log,g in m.groupby('log_id'):
            ix=g.index.to_numpy();phi=g.phase.to_numpy();X=harmonic_design(phi,3);grid=np.linspace(0,2*np.pi,200);G=harmonic_design(grid,3)
            cy=np.linalg.lstsq(X,a['target_D0_raw'][ix],rcond=None)[0];cp=np.linalg.lstsq(X,a['prediction'][ix],rcond=None)[0]
            fig,axes=plt.subplots(2,3,figsize=(11,6))
            for j,ax in enumerate(axes.flat):ax.plot(grid,(G@cy)[:,j],label='measured phase mean');ax.plot(grid,(G@cp)[:,j],label='teacher Main V2');ax.set(title=['ax_n','ay_n','az_n','p_dot','q_dot','r_dot'][j],xlabel='within-log phase rad')
            axes[0,0].legend(fontsize=7);fig.suptitle('Descriptive all-row harmonic curves: '+Path(log).stem);fig.tight_layout();fig.savefig(r/(split+'_'+Path(log).stem+'_teacher_phase.png'),dpi=130);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for ax,sig in zip(axes,['linear','angular']):
        f=c[(c.split=='validation')&(c.scope=='train_only')&(c.k==20)&(c.signal==sig)].groupby(['representation','target']).conditional_variance_ratio.mean().unstack()
        f[['D0_raw','D1_lp6','increment2']].plot.bar(ax=ax);ax.set(title=sig,ylabel='Conditional / unconditional train variance');ax.tick_params(axis='x',rotation=60)
    fig.tight_layout();fig.savefig(r/'conditional_variance.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,sig in zip(axes,['velocity','omega']):
        f=integ[(integ.split=='validation')&(integ.signal==sig)]
        for name,g in f.groupby('target'):
            z=g.groupby('steps').rmse.mean();ax.plot(z.index*.02,z,'o-',label=name)
        ax.set(title=sig,xlabel='Nominal integration window s',ylabel='Integrated increment RMSE');ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(r/'derivative_integration_consistency.png',dpi=150);plt.close(fig)
    ss=[]
    for keys,g in sen.groupby(['split','channel','step','output']):
        a=g.sensitivity.to_numpy();ss.append(dict(split=keys[0],channel=keys[1],step=keys[2],output=keys[3],mean=a.mean(),median=np.median(a),p1=np.quantile(a,.01),p99=np.quantile(a,.99),max_abs=np.max(abs(a)),near_zero_fraction=np.mean(abs(a)<1e-6),positive_fraction=np.mean(a>1e-6)))
    ss=pd.DataFrame(ss);ss.to_csv(r/'sensitivity_summary.csv',index=False)
    useful=ss[(ss.split=='validation')&((ss.channel=='motor')&(ss.output=='frequency_dot')|(ss.channel=='tail_sym')&(ss.output=='q_dot')|(ss.channel=='tail_diff')&(ss.output=='p_dot')|(ss.channel=='rudder')&(ss.output=='r_dot'))]
    table_rates=tel[tel.instance==0].groupby('topic').agg(rate_min=('rate_hz','min'),rate_max=('rate_hz','max'),max_recorded_gap_s=('max_gap_s','max')).reset_index()
    causal=phase.groupby(['split','signal'])[['component_heldout_r2','component_truth_std','component_pred_std']].mean().reset_index()
    same=c[(c.k==20)&(c.signal=='angular')&c.target.isin(['D0_raw','D1_lp6','increment2'])].groupby(['split','scope','representation','target'])[['conditional_variance_ratio','knn_r2','distance_median']].mean().reset_index()
    validation_same=same[same.split=='validation']
    load=pd.read_csv(r/'load_channel_information.csv');load=load[(load.split=='validation')&(load.signal=='angular')&(load.target=='D0_raw')].groupby('representation')[['conditional_variance_ratio','knn_rmse']].mean().reset_index()
    duration=dur.groupby(['split','steps']).agg(min_s=('min','min'),max_s=('max','max'),mean_s=('mean','mean')).reset_index()
    selected=ps[(ps.scope!='fit')&ps.evaluation_target.isin(['D0_raw','increment2'])]
    filt=integ[(integ.signal=='omega')&integ.steps.isin([2,5,10,25])].groupby(['split','target','steps'])[['rmse','r2']].mean().reset_index()
    report=['# Step 4 — Dynamics / Observability / Phase Audit',
        '**唯一第一优先级：P2 改 derivative/state-transition target。第二优先级：P1 修 phase representation，但必须保留/解决 per-log unknown zero，不能直接把日志phase当统一机械phase。**',
        '理由：2-step increment监督的小型受控对照在3个seed的validation上改善同一increment指标；直接加入经train offset调整的phase在未知validation offset下反而退化。强phase同步证明信号可重复，不等于已证明phase替换是主要解法。没有训练Main V3、没有修改Main V2架构/权重、没有打开sealed test或启动RL。',
        '## 输入与证据范围',
        '首先完整读取Step 3报告及loss审计。精确源码路径见 `docs/audits/2026-09-15_main_v2_phase_observability_audit.md`。所有Step 3文件hash保留并复核；本轮结果单独保存。',
        '**口径纠正：3.30 rad/s²是recursive全时域预测std，不是teacher-state一步std。** 本轮每个合法时点用真实当前状态和26点合法过去reset，只step一次；train 43,057点、validation 40,085点，要求两侧至少25步且不跨segment。validation逐点预测std约12.08、raw target约30.20 rad/s²，仍偏弱，但不能以3.30描述一步预测。',
        'Linear acceleration指NED速度差分，不是IMU specific force；angular acceleration为body-FRD角速度差分。统计结论是对已记录/估计的状态，不把phase-locked estimator vibration自动升级为全部真实刚体运动。',
        '实际duration范围（每个segment原生时间；过滤诊断才使用20ms网格）：\n'+md(duration),
        'history5/13/26分别覆盖4/12/25个实际转移，约0.08/0.24/0.50秒；不是5/13/26乘20ms。完整dt、increment时长在actual_durations.csv。',
        '## Q1：多少phase-synchronous component？',
        '按每个log独立3阶phase harmonic regression，前半时间拟合、后半时间检验，中间purge。下表为原生采样及真实训练窗口上的日志等权R²。cycle-domain24bins/完整周期分析另列，不能将重采样平滑后的R²冒充raw R²。',
        md(harm[harm.representation.isin(['within_log_native','t0_relative'])][['split','representation','signal','heldout_r2']]),
        '三维导数的逐点phase-only heldout R²与Main V2自身phase component：\n'+md(causal),
        'validation：p/q/r分别约0.439/0.567/0.036，ax/ay/az约−0.106/−0.029/0.930，p_dot/q_dot/r_dot约0.251/0.638/0.240（各log R²等权，不是合并方差百分比）。负值表示该固定phase模板没有跨时间预测能力。线性导数整体phase-only R²约0.798、角导数约0.404；不同采样权重下数字不能混用。',
        '1×/2×/3×阶数完整结果在harmonic_regression.csv；q_dot在cycle域的validation heldout R²从1阶约0.096升到2阶0.625、3阶0.703，证明不能仅保留fundamental。每log均值/std及十周期block bootstrap CI在phase_curves.csv，图为phase_domain_train/validation.png。CI只描述日志内block重复性，不覆盖跨flight未知offset或非平稳环境。',
        '## Q2：t0-relative是否破坏physical phase consistency？ YES',
        'encoder total_count → 4096×FLAP_RATIO换算 → log-local stored phase（原点为首个有限对齐encoder count）→ 每个history减t0 phase → rollout保持该anchor → sin(phi−anchor)/cos(phi−anchor)。stored phase不是per-segment归零，但network phase确实per-window归零。',
        '同一物理phase在不同anchor窗口中不保证同sin/cos；所有t0 sin/cos都为(0,1)，可以对应完全不同wing位置。对跨log还额外存在unknown mechanical zero。Q1表中t0-relative R²接近0，相比日志内phase显著丢失直接同步位置；但历史状态仍能间接承载phase，因此不能据此称全部观测信息被删除。',
        '只在train日志早半周期估计常数offset，再在晚半周期检查跨log导数曲线一致性：\n'+md(off[['log_id','phase_offset_rad','heldout_cross_log_dispersion_before','heldout_cross_log_dispersion_after']]),
        '离散15°offset使晚半周期跨log平均方差从184.88降至117.92，约36.2%。仍有明显剩余差异；这个训练内改善不是validation推广。validation没有拟合任何部署offset；对其单独phase回归只用于Q1描述性重复性分析。',
        '## Q3：raw变化多少是物理、多少是噪声？',
        '不能严格分成两个百分比。至少有大量可跨时间预测的phase-synchronous记录分量，尤其q_dot二/三阶谐波；大于10Hz不能全当noise。约40%的三维角导数heldout预测解释度是“实测可重复性”的证据，不是独立传感器确认的物理variance下界；剩余约60%混合非平稳、higher harmonics、未建模状态和测量/差分噪声。',
        'raw与各离线target的总体std、teacher-state误差：\n'+md(t[t.signal=='angular']),
        'D1为4阶Butterworth zero-phase，uniform20ms内部网格后回原生timestamp再差分，cutoff4/6/8/12Hz；边界剔除25步。**仅用于离线target diagnosis，不能作为simulator inference输入。** D2_center2/5/10用于居中导数积分分析；increment2/5/10为向前真实状态差/真实duration，两者的时间定位不同，不混称。',
        '## Q4：优先哪种supervision target？',
        '**优先短时间2-step state increment（约0.04秒），保留pointwise state约束，先做与Main V2架构固定的target实验。** 不是要求NN std达到raw30，不建议把6Hz低通导数直接设为新的唯一真值。',
        '积分一致性结果如下，velocity结果也保存在CSV：\n'+md(filt),
        'D0积分精确等于原状态增量是望远镜求和的代数恒等式，不证明无噪声。D1_lp6在validation约0.04/0.2/0.5秒角增量R²仅0.366/0.532/0.633，lp12约0.864/0.907/0.894；强低通确实删掉了可积累的state变化。2/5/10step平均若被当瞬时导数再积分也有平滑/时移误差。2step监督的优势由下面同输入、同架构、同预算模型对照支持；未证明可直接改善5秒simulator。',
        '## Q5：当前input/history是否近似可辨识？ PARTIAL',
        '固定train reference stride5、每log200个等间隔query、k5/20；同log排除±2秒，跨log单独检验。全部标准化只用train；neighbor_ids.csv保存每个query与邻居身份。',
        md(validation_same[['representation','target','conditional_variance_ratio','knn_r2','distance_median']]),
        'GRU26相对instant改善filtered/increment的邻域预测：validation raw R²约0.127→0.218，lp6约0.077→0.540，increment2约0.324→0.393。13点通常优于26点flattened history，不能据此说“history无用”，也不能说history越长越能解决missing state。不同维数邻域距离有集中效应，未做相同流形维数证明。',
        '严格“几乎相同”输入证据不足：validation k20、RMS标准化距离≤0.25时所有表示均无query；≤0.5时GRU仅30/1000个。不能从剩余conditional variance直接证明非Markov或必须latent。same_log/other_log结果有差异但邻域密度不同；date与split完全混杂，不能识别wind/date因果。',
        '## Q6：最可能缺什么？按证据排序',
        '1. **可靠phase reference/一致phase表示**：同步结构强且确有t0重锚、跨log零点不确定。它是有直接证据的表示缺口，但受控probe尚未证明简单替换能泛化。\n2. **airspeed/wind**：日志实际存在Pitot速度与estimated wind，不是完全缺传感器；额外context使raw角conditional ratio仅约0.532→0.522、kNN R²0.218→0.228，证据弱，不能宣布它主导误差。\n3. **actual servo state**：没有feedback，只有命令和输出命令；高tail误差关联见actuator_associations.csv，不能区分lag/load/saturation/dead-zone，更不能据此改time constants。\n4. **motor/load**：pack current/voltage、raw RPM存在，但不能当真实motor torque/wing deformation。额外信息的邻域指标改善很小。\n5. **wing/aero memory及其他latent**：本轮没有直接传感器或可识别的因果证据，不以剩余方差当作证明。',
        '可用telemetry的原生publication rate和全日志最大gap（不能把全日志gap都当飞行核心掉线；core freshness另列）：\n'+md(table_rates[table_rates.topic.isin(['airspeed','airspeed_validated','wind','battery_status','rpm','vehicle_angular_velocity','vehicle_acceleration','actuator_outputs'])]),
        'airspeed_quality_input的source=1且valid=1，validated多为sensor1；validation最后一log有约7.6% source=-1。字段存在/valid只表示发布契约，不证明校准精度；wind是估计量，GPS与local velocity来自导航链，均非独立空气速度真值。wind最长发布gap可约100秒，全日志range需结合core freshness。',
        'servo_feedback、ESC report/status在11日志均不存在。battery current为pack ADC量；日志间current均值有差异，validation一log约7.61A，其他约2.5–3.5A，不能假定同频率等于同负载或把电流差直接归于负载。',
        '新增load channels的validation角动态kNN：\n'+md(load),
        '## Q7：command sensitivity可信？ PARTIAL，尚无真实Jacobian验证',
        '固定600 train和500 validation operating points（每log100等间隔），±0.001 normalized command中心差分。sym同时改变左右，diff左右反向；没有用结果挑点。完整输出和motor/tail/speed/body-rate/frequency分桶见command_sensitivity.csv、sensitivity_bins.csv。',
        md(useful[['channel','step','output','median','p1','p99','max_abs','near_zero_fraction','positive_fraction']]),
        'step0导数对command严格零，原因是proxy在step末更新。step1 motor→frequency-dot为正且非零；此时motor→rigid-body仍零（结构只直接作用frequency），后续只通过状态间接影响。sym→q-dot、rudder→r-dot大多同号但少数变号，diff→p-dot有明显符号分布。没有经过机械舵面符号/真实响应实验标定，不能把正/负直接判wrong sign。零cross-axis响应多由mask强制，不是从数据证明该通道真实不存在；这些限制不支持宣称控制因果已学会。',
        '## 小型受控对照（先审计后训练）',
        'B0 current26-point input→raw；B1同输入→2step increment/duration；B2更换history phase为log-phase加train-only offset→raw；B3 phase+increment。相同420→64→64→6 MLP、30epoch、batch512、AdamW lr0.001、seed415/416/417、同训练rows/原始target scales；没有预训练GRU权重作为probe输入。第一70%每train log训练、末30%purge2s作训练内诊断，正式train/validation flight分配没有变化。validation offset未知设0，绝不调其标签offset。',
        md(selected),
        '同target、配对seed百分比变化（负数改善；seed范围不是跨flight置信区间）：\n'+md(changes),
        'B0本身已能在validation预测线性raw动态R²约0.835、角动态约0.411，明显强于冻结Main V2一步的约0.079/0.265。这说明合法history中有更多可用信息，不能直接判state整体不可观测或数据全噪声；但probe同时改变训练任务/映射结构，不能单独归因GRU容量。B1对共同2step target优于B0；B2对未知zero的validation退化，无法支持当前直接phase替换。所有probe只做局部回归，没有5秒simulator评估，不作候选晋级。',
        '## Q8：长期漂移最有证据的前三个来源',
        '1. **当前训练得到的局部transition没有利用可预测动态变化**：teacher-state已衰减，phase均值幅值偏弱；小模型在同合法history上能预测更多变化。原因可能包括监督对象与recurrent representation，不能仅凭此判容量不足。\n2. **phase位置表示不一致及跨log reference不确定**：有直接源码与phase-domain证据；但history可补部分信息，故没有证明其为唯一或首要可修复因果。\n3. **raw native差分含大量快速变化且监督时间尺度不匹配**：滤波/积分与increment probe显示target选择重要；噪声与高阶可重复分量不能严格分离。',
        '**下一步只选P2为第一优先级**：短时state-increment物理监督的固定Main V2实验，继续保留raw/filtered/phase同期诊断；P1为第二优先，必须先解决可部署的reference契约。P3/P4/P5没有足够提升证据，P6可为跨log机械相位和独立舵面/气流验证服务，但当前不优先替换网络或启动RL。',
        '## Readiness / 限制',
        'Simulator structural readiness: **100%**\n\nDynamics fidelity readiness: **60%**\n\nRL-ready: **NO**\n\n沿用Step 3十项证据清单，不因诊断或局部probe提高分数。本轮没有证明长期fidelity改善，没有确认独立物理真值，也没有完成可靠command因果标定。',
        '## 一条完整复现命令',
        '```bash\n/home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/run_main_v2_step4_all.py \\\n  --output /tmp/main-v2-step4/results --artifacts /tmp/main-v2-step4/probes --device cuda:1\n```',
        '使用空目录，先phase/derivative/telemetry/teacher/NN/sensitivity审计，再补充诊断、12个小模型、报告和回归。所有过滤参数、source hashes、IDs、seed与step数均保存。测试和git diff检查结果见verification.json。']
    (r/'report.md').write_text('\n\n'.join(report))
    summary=dict(first_priority='P2 derivative/state-transition target',second_priority='P1 phase representation with deployable reference',phase_consistency_broken='YES',identifiability='PARTIAL',command_sensitivity_trusted='PARTIAL',structural_readiness=100,dynamics_fidelity_readiness=60,rl_ready=False,sealed_test_opened=False,step3_changed=False,teacher_angular_pred_std_validation=float(t[(t.split=='validation')&(t.target=='D0_raw')&(t.signal=='angular')].pred_std.iloc[0]),probe_changes=changes.to_dict('records'),limitations=['phase synchronous measurement is not independent physical truth','no nearly-identical k20 queries at RMS radius0.25','validation offsets unknown','local probes not autonomous simulators'])
    (r/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='probe_changes'},indent=2))

if __name__=='__main__':main()
