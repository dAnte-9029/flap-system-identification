"""Report the complete bounded replay, including failed absolute accuracy gates."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/flap-v40-matplotlib')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_control_windows_v2 as base
from run_coupled_replay_v40 import OUT,DATA,PARENTS,read
from coupled_replay_v40 import VARIANTS
from audit_controller_support_v26 import verify_parent


def table(frame):
    rows=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    for row in frame.itertuples(index=False,name=None):
        rows.append('| '+' | '.join(f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(rows)


def main():
    assert read(OUT/'verification.json')['status']==read(OUT/'independent_verification.json')['status']=='passed'
    protocol=read(OUT/'protocol.json');gates=read(OUT/'gates.json')
    states=pd.read_csv(OUT/'state_summary.csv');actions=pd.read_csv(OUT/'action_summary.csv')
    state_table=states[(states.kind=='wave')&(states.horizon==50)].groupby(['variant','stage','metric']).rmse.mean().unstack().reset_index()
    action_table=actions[actions.comparison=='log'].groupby(['variant','stage','axis']).rmse.mean().unstack()
    action_table.columns=['motor0','servo0','servo1','servo2']
    teacher=pd.read_csv(OUT/'teacher_summary.csv').pivot(index='variant',columns='axis',values='rmse')
    teacher.columns=['motor0','servo0','servo1','servo2']
    absolute=pd.read_csv(OUT/'absolute_gates.csv')
    abs_table=absolute[(absolute.variant=='warm2_plus')&(absolute.stage==35)].groupby('metric')[['rmse','limit']].mean().reset_index()
    support=pd.read_csv(OUT/'support_summary.csv');supp=support[support.stage==35].copy()
    for k in supp.columns[2:]:supp[k]*=100
    gate_table=pd.DataFrame(gates)[['variant','q_improvement_percent','improving_flights','protections_passed','action_protections_passed','absolute_passed']]
    # Time curves are descriptive summaries of exactly the saved full cohort.
    curves=[];sensitivity=[]
    with np.load(DATA/'windows.npz') as z:
        target={k:z['label_'+k].astype(float) for k in ['rates_b','quaternion','velocity_n']}
        log=z['input_future_command'].astype(float)
    for fold in range(5):
        ids=read(DATA/f'fold{fold}/manifest.json')['query_indices'];dest=OUT/f'fold{fold}'
        with np.load(dest/'warm2_plus_teacher.npz') as z:tc=z['commands']
        for stage in [30,35]:
            for seed in [17,23,42]:
                with np.load(dest/f'warm2_plus_v{stage}_seed{seed}.npz') as z:a={k:z[k] for k in z.files}
                for metric,err in [('p',a['rates_b'][:,:,0]-target['rates_b'][ids,:,0]),
                                   ('q',a['rates_b'][:,:,1]-target['rates_b'][ids,:,1]),
                                   ('servo0',a['commands'][:,:,1]-log[ids,:,1]),
                                   ('servo1',a['commands'][:,:,2]-log[ids,:,2])]:
                    for k,value in enumerate(np.sqrt((err**2).mean(0))):curves.append(dict(fold=fold,stage=stage,seed=seed,metric=metric,step=k+1,rmse=value))
                for name,_,_ in VARIANTS[1:]:
                    with np.load(dest/f'{name}_v{stage}_seed{seed}.npz') as z:
                        for kind in ['commands','rates_b']:
                            delta=z[kind].astype(float)-a[kind].astype(float)
                            for axis,value in enumerate(np.sqrt((delta**2).mean((0,1)))):
                                sensitivity.append(dict(fold=fold,seed=seed,stage=stage,variant=name,kind=kind,axis=axis,rmse=value))
        for axis in [1,2]:
            for k,value in enumerate(np.sqrt(((tc[:,:,axis]-log[ids,:,axis])**2).mean(0))):
                curves.append(dict(fold=fold,stage=0,seed=0,metric=f'servo{axis-1}',step=k+1,rmse=value))
    curves=pd.DataFrame(curves);curves.to_csv(OUT/'time_curves_per_flight.csv',index=False)
    avg=curves.groupby(['stage','metric','step']).rmse.mean().reset_index()
    pd.DataFrame(sensitivity).to_csv(OUT/'sensitivity_per_flight.csv',index=False)
    sensitivity_summary=pd.DataFrame(sensitivity).groupby(['stage','variant','kind','axis']).rmse.mean().reset_index()
    sensitivity_summary.to_csv(OUT/'sensitivity_summary.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for ax,metric in zip(axes.flat,['p','q','servo0','servo1']):
        for stage,label,color in [(30,'v30 coupled','#b55a30'),(35,'v35 coupled','#2370a0'),(0,'measured-state controller','#777777')]:
            g=avg[(avg.stage==stage)&(avg.metric==metric)]
            if len(g):ax.plot((g.step-(1 if metric.startswith('servo') else 0))*.02,g.rmse,label=label,color=color)
        ax.set_title(metric);ax.set_xlabel('Time from origin (s)');ax.set_ylabel('RMSE (normalized command)' if metric.startswith('servo') else 'RMSE (rad/s)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('warm2_plus diagnostic — same 407 parents, equal flight / seed weighting')
    fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
    next_plan=dict(status='prepared_not_executed',objective='Frozen state-to-command error propagation audit, before further training',
        data='Same407 originaltrain parents and30 frozen models; retain4clock/state hypotheses, no new validation/sealed/reserved.',
        tasks=[
            'Reconstruct controller internal states from the saved model-state trajectories and condition tapes; first reproduce existing actions exactly.',
            'Freeze cross-comparisons of model versus measured attitude/rates at identical controller internal state, and model-history versus teacher-history controller states at identical plant state. Report time-dependent action changes, source saturation and coupling.',
            'Use separately named offline oracle replacements for p/q/r and roll/pitch; retain all windows and variants. Do not treat future-truth substitutions as deployable corrections or add nonlinear effects as if independent.',
            'Localize action-error onset by horizon/flight/mode and original support; decide whether local state-error learning or controller initialization/scheduling warrants the next controlled experiment. No automatic tuning, architecture search, window removal or real flight.'
        ], rationale='v35 relative benefit persists, but p/q/attitude absolute gates fail and generated surface commands are much less faithful than matched teacher commands.',
        no_training=True,no_controller_tuning=True,no_real_flight=True)
    base.write(OUT/'next_execution_plan.json',next_plan)
    decision=dict(status='bounded_coupled_replay_complete',coupled_window_runs=9768,teacher_window_runs=1628,
        all407_retained=True,nonfinite_failures=0,relative_gate_passed_variants=sum(g['relative_gate'] for g in gates),variants=4,
        absolute_passed_each_variant=[g['absolute_passed'] for g in gates],absolute_total_each_variant=12,
        exact_firmware_replay=False,independent_control_effectiveness_verified=False,control_ready=False,model_promoted=False,
        baseline='A1600 retained; v35 remains a research candidate',validation_accessed=False,sealed_test_accessed=False,reserved_accessed=False)
    base.write(OUT/'decision.json',decision)
    base.write(OUT/'test_results.json',dict(passed=3,command='pytest -q tests/test_coupled_replay_v40.py',
        unchanged_test_result_reused=True,strict_numpy_scalar_json_roundtrip_passed=True))
    report=f'''# c9af＋GRU受限闭环重放 v40

已完成30个冻结v30/v35 OOF模型、全部407个原train父窗、4种预先冻结初始化/时钟/反馈假设的**9768条闭环轨迹**，另完成1628条匹配调度的真实状态驱动控制器对照。所有轨迹有限，未删窗、重训或额外限制模型动作。**v35相对收益在本轮近似闭环中保持，但绝对精度和动作复现仍不足，不能认定控制可用。**

## 怎样闭环

数据仍为5个原train飞行的274 Stabilized＋133 Mission父窗，每窗H26初始化、K50预测1秒。模型参数和归一化保持原样，五折OOF已反复用于开发，不能称作独立测试。没有读取开发验证/sealed/reserved。

控制器起点前用原始日志的因果2秒历史预热；整个预热严格早于起点。保留warm2_plus、warm2_minus、warm2_20ms、warm2_delayed_feedback，分别覆盖HPF全正/全负、20ms替代时钟和上一合成分配残差反馈。两个符号探针并非全部初态不确定性边界，四分支均报告，不选最低误差配方。

从起点以后，姿态/角速度由GRU状态提供，c9af控制器产生力矩和分配命令，再由命令驱动GRU前进一步。模型闭环没有未来实测状态或未来日志执行命令入口；日志姿态参考、油门、manual yaw只作为条件带。Mission导航/TECS未模拟，这些条件也不是外生随机激励。

400Hz控制器与50Hz模型采用明确的分段保持近似：模型状态在20ms内保持，rate环每2.5ms更新、姿态环每10ms更新；模型只消费区间起点生成的命令，全部中间命令另存。20ms分支两个环均每20ms更新。由此产生了50Hz飞机输入保持，**不是还原了原生高频执行波形**。v35主要分支中，内部高频请求与模型所用命令的servo0/1 RMS差约0.00938/0.00968，不能视为零；20ms分支该差为0是定义导致的，不是其更真实。

控制器空速采用起点有效CAS（无效/过期则源码trim回退），作为固定的新鲜合成观测；GRU沿用原生固定扑频/空速和phase推进。真实三轴D=0已核验，因此控制器角加速度输入0；未来模式/空中/rate/attitude标志仅审计窗口适用性，运行时固定起点模式且landed=false，无未来积分/增益/reset回灌。没有增加舵面/油门归零、安全退出或额外动作clamp。

匹配的真实状态对照使用同样初始控制器、20ms保持、参考带和固定CAS，只将状态换为实测值。它与v38原先按日志姿态发布更新、使用未来实测空速的对照不同，必须分别比较；本轮不将它的误差从模型误差中直接相减。

## 状态与动作结果

下表先飞行内求RMSE，再五飞行等权、三种子等权。p/q/r为rad/s，速度为m/s，姿态为rad。`logged_command_conditioned`是此前用未来日志执行命令的条件预测，仅作边界不同的对照。

{table(state_table)}

warm2_plus中，v35完整1秒q从v30的0.367175降至0.331321rad/s，改善9.7648%；但比它自身日志命令条件预测的0.322464稍差。p/r为0.510303/0.183921，速度0.416773，姿态0.098555。闭环与日志命令条件预测的差别包含控制器/调度近似和反馈变化，不能唯一归因于裸机模型。

生成命令对真实日志命令的RMSE如下。四列是归一化命令，不是舵角或推力。

{table(action_table.reset_index())}

匹配真实状态驱动控制器的命令RMSE：

{table(teacher.reset_index())}

v35 warm2_plus的servo0/1误差约0.09377/0.09335，而匹配真实状态对照约0.03730/0.03678。生成动作还有显著偏差；不能只用“状态RMSE变化不大”宣称闭环已忠实复现。motor0由已知条件油门驱动，它在两种模型/teacher之间相同，不证明电机模型有效。

![状态与命令误差随时间变化](comparison.png)

所有四种假设的逐飞行配对轨迹差另存[sensitivity_summary.csv](sensitivity_summary.csv)；这些是有限的敏感性探针，不是误差置信区间。

## 冻结验收

沿用v29/v35主门槛：完整1秒q改善至少5%、三个种子均改善、至少4/5飞行改善；18项保护为每种子p/r完整1秒及速度/姿态0.5秒/1秒端点退化不超过5%。新增的动作保护在运行前冻结：每种子每个归一化命令RMSE相对v30退化不超过5%，共12项，仅是相对诊断。

{table(gate_table)}

四种假设均5/5飞行改善，状态保护18/18、动作保护12/12。**绝对门槛均仍只有3/12通过，三个种子的速度通过，p/q和姿态均失败。**下面是warm2_plus的v35三个种子平均值，正式门槛仍逐种子判定：

{table(abs_table)}

因此，相对收益可保留为研究证据，不能把原先要求放宽为“闭环有限”或仅比较平均误差。保留A1600及v35研究候选，不晋升模型，不设计或部署新控制器。

## 饱和与覆盖

仅用各折四个fit飞行的命令与历史状态建立逐通道min/max范围。在线标记使用当时生成命令和当前生成状态，没有用未来真值选择窗口。下表为v35百分比，先飞行/种子等权；fraction是时刻比例，any是出现过该事件的模型窗口比例。

{table(supp)}

warm2_plus约0.776%的时刻命令超出fit逐通道范围，约1.788%的时刻状态至少一列超范围；原生分配表面限幅约0.304%的控制微步出现。所有这些轨迹仍保留。逐通道范围内不保证联合输入/状态有支持，也不是可靠置信度；范围外不等同真实飞行危险判据。

完整计数见[support_window_counts.csv](support_window_counts.csv)，逐飞行、模式、种子和窗口指标全部保存。9768条是重复模型/假设运行，不是9768个独立真实操纵事件。

## 核验与限制

3项耦合测试通过：区间起点命令与中间命令区分、参考输入前缀因果性、重复独立会话一致，以及不支持时钟/D项的拒绝。全部162800个参考条件点从原ULog独立重建，publication和sample均满足因果边界；全部模型同一假设的第一步命令一致。另独立重算273504个窗口指标、488400个当前状态支持标记和全部命令支持，确认实际模型输入恰为区间起点命令。详见[verification.json](verification.json)、[independent_verification.json](independent_verification.json)。

首次准备阶段被严格JSON拒绝NumPy布尔标志，尚未运行模型耦合；未完成产物及当时源码保留于`control_windows_v40_preparation_serialization`。修复仅将NumPy标量无损转换为JSON原生标量，未改波形、数据或门槛，重新冻结后完成本轮。旧v29/v30/v35/v38/v39产物及检查点哈希均保持。

下一步先冻结本轮模型，定位角速度/姿态误差与控制器内部历史分别怎样变成动作偏差：在相同控制器状态下对照模型状态与实测状态，再分离积分/压缩历史差异，按时间、飞行和原支持分层。需要用到未来真值的替换明确标为离线oracle，不作为可部署补偿。具体合同见[next_execution_plan.json](next_execution_plan.json)；本轮未实施该后续诊断，也未据此认定必须补飞。
'''
    (OUT/'report_zh.md').write_text(report)
    for parent in PARENTS:verify_parent(parent)
    for name,digest in protocol['code'].items():assert base.sha(base.ROOT/name)==digest
    code=dict(protocol['code'])
    for name in ['scripts/evaluate_coupled_replay_v40.py','scripts/verify_coupled_outputs_v40.py','scripts/report_coupled_replay_v40.py']:code[name]=base.sha(base.ROOT/name)
    base.write(OUT/'status.json',dict(status='complete_verified',coupled_window_runs=9768,control_ready=False))
    base.write(OUT/'delivery_manifest.json',dict(artifacts={str(p.relative_to(OUT)):base.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'},
        code=code,model_promoted=False,control_ready=False,validation_accessed=False,sealed_test_accessed=False,reserved_accessed=False))
    verify_parent(OUT)
    print(decision,flush=True)


if __name__=='__main__':main()
