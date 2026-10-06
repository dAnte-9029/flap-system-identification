"""Report state/history separation without claiming deployable oracle gains."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/flap-v41-matplotlib')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_control_windows_v2 as base
from run_action_propagation_v41 import OUT,PARENT,PARENTS,read
from action_propagation_v41 import BRANCHES
from audit_controller_support_v26 import verify_parent


def table(frame):
    rows=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    for row in frame.itertuples(index=False,name=None):
        rows.append('| '+' | '.join(f'{x:.6f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |')
    return '\n'.join(rows)


def main():
    assert read(OUT/'verification.json')['status']==read(OUT/'independent_verification.json')['status']=='passed'
    protocol=read(OUT/'protocol.json')
    summary=pd.read_csv(OUT/'summary_rmse.csv')
    selected=summary[(summary.variant=='warm2_plus')&(summary.stage==35)&(summary.horizon==50)]
    log=selected[selected.target=='log'].pivot(index='branch',columns='axis',values='rmse').loc[BRANCHES]
    log.columns=['motor0','servo0','servo1','servo2']
    teacher=selected[selected.target=='teacher'].pivot(index='branch',columns='axis',values='rmse').loc[BRANCHES]
    teacher.columns=log.columns
    flight=pd.read_csv(OUT/'metrics_per_flight.csv')
    comparisons=flight[(flight.stage==35)&(flight.horizon==50)&(flight.target=='log')&flight.axis.isin([1,2])].groupby(['variant','fold','seed','branch']).rmse.mean().unstack()
    rows=[]
    for variant,g in comparisons.groupby(level='variant'):
        row=dict(variant=variant,groups=len(g),roll_positive_groups=int((g.roll<g.mm).sum()),attitude_positive_groups=int((g.attitude<g.mm).sum()))
        for name in ['mm','mt','tm','tt','p','q','roll','pitch','attitude','integral','compression']:row[name]=float(g[name].mean())
        rows.append(row)
    consistency=pd.DataFrame(rows);consistency.to_csv(OUT/'surface_oracle_consistency.csv',index=False)
    moments=pd.read_csv(OUT/'factor_moments_per_flight.csv')
    m=moments[(moments.stage==35)&(moments.horizon==50)].groupby(['variant','axis']).mean(numeric_only=True)
    signed=m[['plant_delta','history_delta','reverse_plant_delta','reverse_history_delta']].div(m.delta_sq.replace(0,np.nan),axis=0)*100
    signed=signed.dropna().reset_index();signed.to_csv(OUT/'signed_factor_shares.csv',index=False)
    strata=pd.read_csv(OUT/'strata_per_flight.csv');strata['rmse']=np.sqrt(strata.mse)
    st=strata[(strata.variant=='warm2_plus')&(strata.stage==35)&(strata.target=='log')&strata.axis.isin([1,2])].groupby(['family','layer','branch']).rmse.mean().unstack()
    st=st[['mm','roll','attitude','mt','tm','tt']]
    st.to_csv(OUT/'surface_strata_summary.csv')
    clip=[]
    for fold in range(5):
        for name in ['warm2_plus','warm2_minus','warm2_20ms','warm2_delayed_feedback']:
            for stage in [30,35]:
                for seed in [17,23,42]:
                    with np.load(OUT/f'fold{fold}/{name}_v{stage}_seed{seed}_metrics.npz') as z:
                        values=z['surface_clip_fraction'].mean(0)
                    for branch,value in zip(BRANCHES,values):clip.append(dict(fold=fold,variant=name,stage=stage,seed=seed,branch=branch,left_tick_clip_fraction=value))
    pd.DataFrame(clip).to_csv(OUT/'oracle_clipping_per_flight.csv',index=False)
    curves=pd.read_csv(OUT/'time_curves_per_flight.csv')
    curves=curves[(curves.variant=='warm2_plus')&(curves.stage==35)&(curves.target=='log')].groupby(['branch','axis','step']).rmse.mean().reset_index()
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    show=['mm','p','q','roll','pitch','attitude','integral','compression','mt','tm','tt']
    for col,axis in enumerate([1,2]):
        ax=axes[0,col];ax.bar(show,log.iloc[:,axis].loc[show],color=['#bd6041' if b=='mm' else '#297fa5' for b in show])
        ax.tick_params(axis='x',rotation=55);ax.set_ylabel('Full1s command RMSE');ax.set_title(f'servo{axis-1} — one-tick oracles')
        ax=axes[1,col]
        for branch,label in [('mm','factual'),('roll','true roll only'),('attitude','true roll+pitch'),('mt','true current plant'),('tt','teacher')]:
            g=curves[(curves.branch==branch)&(curves.axis==axis)]
            ax.plot(g.step*.02,g.rmse,label=label)
        ax.set_xlabel('Time from origin (s)');ax.set_ylabel('Command RMSE');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('v35 warm2_plus: fixed-history, single-tick substitutions (NOT propagated)')
    fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
    next_plan=dict(status='prepared_not_executed',objective='Prepare a controlled attitude-supervision training comparison',
        rationale='Current roll/pitch state errors dominate the observed surface-action discrepancy under all4diagnostic variants; currentrate-only or hidden-history-only substitution is much weaker.',
        baseline='Frozen v35 joint-increment model, with v30 and all v40/v41 artifacts retained',
        stages=[
            'Before selecting a loss, verify saved quaternion/roll/pitch trajectories by native midpoint integration and inspect whether accumulated body-rate error or kinematic/label residual drives roll/pitch drift. Truth-rate substitutions are offline diagnostics only.',
            'Audit the existing loss and fit-only gradient/loss scales. Prepare one explicit roll/pitch attitude-supervision term; freeze coefficient and normalization using fit data only, not by searching OOF gains.',
            'Keep architecture,40070parameters,H26/K50,all407parents,5folds3seeds,initialization,sampling and1600update budget fixed. First reproduce the zero-weight v35 baseline exactly.',
            'Freeze paired state accuracy/absolute gates plus generated-action metrics under all4v40coupling variants; retain p/q/r/velocity/attitude protections. No target-only easy-window evaluation.',
            'Only after preparation checks pass execute the specified controlled training. No controller gain tuning, new observation channels, action restriction or validation/sealed/reserved access.'
        ],no_training_started=True,no_controller_tuning=True,no_real_flight=True)
    base.write(OUT/'next_execution_plan.json',next_plan)
    decision=dict(status='state_to_command_audit_complete',all407_retained=True,source_model_trajectories=9768,
        controller_states_reconstructed=488400,oracle_branches=14,oracle_queries=488400*14,
        roll_substitution_improves_all_flight_seed_groups_per_variant=bool((consistency.roll_positive_groups==consistency.groups).all()),
        priority='Prepare attitude-focused model learning comparison; do not first tune controller initialization/gains',
        not_a_unique_physical_or_training_root_cause=True,oracle_propagated=False,model_promoted=False,control_ready=False,
        validation_accessed=False,sealed_test_accessed=False,reserved_accessed=False)
    base.write(OUT/'decision.json',decision)
    base.write(OUT/'test_results.json',dict(passed=3,command='pytest -q tests/test_action_propagation_v41.py',unchanged_test_result_reused=True))
    native=read(OUT/'independent_verification.json')
    report=f'''# 状态误差怎样变成控制命令偏差：v41

已完成v40全部407父窗、30个冻结模型来源、4种假设的控制器历史重建和14类当前时刻对照。**主要线索是滚转/俯仰姿态误差，而不是当前p/q角速度误差或单独的积分/压缩历史。**这支持下一轮优先准备姿态监督对照；仍没有证明唯一物理/训练根因，也没有取得可部署的真值补偿。

## 先验证没有改变原来的重放

9768条模型状态轨迹、1628条teacher轨迹的原动作、力矩、角速度参考、全部控制微步和分配限幅均逐值复现。记录488400份模型控制器更新前33维状态，teacher状态另存；全部407窗保留，包括Mission、Stabilized和超出原支持范围的时刻。

本轮不运行新GRU轨迹、不训练、不改权重/归一化/控制器参数。仅读取同一批原train数据及冻结预测，没有开发验证/sealed/reserved访问。v40的400Hz/50Hz保持、固定CAS等近似全部保留；teacher隐藏状态是重放的近似状态，不是实机真实隐藏状态。

## 对照如何定义

每个20ms区间起点，在姿态环更新前复制控制器状态，仅做一次对应2.5ms或20ms控制计算。对照输出随后丢弃，原两条控制器历史继续沿各自原状态推进所有微步；**任何真值替换都不回写后续积分、压缩器或飞机轨迹**。

| 分支 | 控制器内部历史 | 当前姿态/角速度 |
| --- | --- | --- |
| mm | 原模型重放 | 原模型状态，复现v40动作 |
| mt | 原模型重放 | 全部换为实测值 |
| tm | teacher重放 | 保留模型状态 |
| tt | teacher重放 | 实测值，复现v40 teacher |
| p/q/r | 原模型重放 | 仅该角速度轴换真值 |
| roll/pitch | 原模型重放 | 仅该欧拉角换真值 |
| rates/attitude | 原模型重放 | 全部角速度/滚转俯仰角换真值 |
| integral | 仅积分换为teacher | 保留模型状态 |
| compression | 压缩器内部及缓存增益、HPF、能量、上一effort换为teacher | 保留模型状态 |
| yaw_memory | 姿态环的旧yaw率记忆等换为teacher | 保留模型状态 |

姿态替换直接发生在控制器实际使用的roll/pitch量上，不构造并注入新的GRU四元数。压缩器先用缓存增益输出、再更新，因此当前单步中真正直接改变命令的是缓存增益；HPF/energy替换的后续效应没有传播。相同固定CAS使两条历史的空速滤波逐值相同，已核验。

## 命令误差结果

以下是v35 warm2_plus完整1秒命令RMSE：先每飞行求RMSE，再五飞行/三种子等权。命令为归一化motor0/servo0/servo1/servo2，不是舵角或气动力。全部14分支与v30、其余假设均保留在结果表中。

对真实日志命令：

{table(log.reset_index())}

对匹配teacher命令（是近似控制器对照，不是实机真值）：

{table(teacher.reset_index())}

几个直接结论：

- 原servo0/1误差0.093766/0.093346。只替换roll降为0.054511/0.051319；替换roll＋pitch降为0.043870/0.041518。保留模型控制器历史但全部换为实测姿态/角速度（mt）为0.042778/0.039875，接近但不等于teacher的0.037297/0.036784。
- 只替换当前p或q，升降副翼误差几乎不降，个别通道反而增大；因此不能把角速度RMSE大直接等同于它对当前动作偏差的贡献最大。这不意味着p/q预测不重要：它们的过去误差仍可能经运动学积分造成当前姿态误差。
- 只替换全部内部历史（tm）仍有0.084514/0.085890；单独积分、压缩、yaw记忆的改善也弱于姿态替换。隐藏历史有影响，不能解释为全部都来自初态或压缩器。
- 方向舵与当前r的关系更明显：仅r替换将servo2误差0.031954降为0.026347；仅姿态替换为0.027470。不同轴不能合并成单一结论。

![误差对照和随时间变化](comparison.png)

## 四种假设、模式和支持分层

下表是servo0/1 RMSE的算术平均，只便于比较两舵面，不是合并平方后开根。每种假设的15组为5飞行×3种子；roll/attitude替换在各组均改善，但这些相关对照不能当作15次独立实飞。

{table(consistency)}

v35主要分支按原模式与v40当时的支持标记分层，仍比较真实日志命令：

{table(st.reset_index())}

mode3为Mission、15为Stabilized。support0为当前状态和命令均在原fit逐通道min/max内，1为仅状态超范围，2为仅命令超范围，3为两者均超范围。分层标记始终来自原v40轨迹，不随oracle替换重新筛窗；按该层有数据的飞行/种子等权，点数另存[strata_per_flight.csv](strata_per_flight.csv)。范围内也存在明显动作误差，不能只归因于少量超范围时刻。边际范围不证明联合支持。

## 分解与交互

定义D=mm−tt、P=mm−mt、H=mt−tt，逐点严格满足D=P＋H。P是在模型内部历史下的当前状态效应，H是在实测当前状态下的内部历史效应；反过来的路径也保留，交互项为mm−mt−tm＋tt。全部1953600个动作标量恒等式核验通过。

为描述平方差如何分配，另报告E[P·D]/E[D²]、E[H·D]/E[D²]，先对各飞行/种子矩等权再求比，符号可以为负。这些是**指定分解路径下的有符号统计份额，不是独立因果百分比**。主要分支两升降副翼的当前状态项约88.44%/88.61%，内部历史约11.56%/11.39%；反向路径得到的当前状态项约87.94%/89.94%。完整四假设及yaw结果、交互项保留，不把各单轴替换效果相加。

## 为什么当前姿态会如此影响动作

在这批c9af实际参数下，滚转/俯仰姿态时间常数均约0.4s，rate环滚转P约0.04而参考前馈FF约0.8，俯仰P约0.06、FF约0.5。源码输出在限幅前包含：

`gain × [scale² × (P × (rate_sp − rate) + integral) + scale × FF × rate_sp] + trim`

姿态误差先改变rate参考，再同时经过P项和较强的参考前馈；当前角速度误差主要直接进入P项。这是现有配置下姿态偏差能显著影响命令的一条源码解释，与本轮替换结果一致，不构成调小FF/PID的建议。这里仍未区分姿态偏差究竟由持续角速度偏差、周期相位、运动学/标签不一致还是其他学习问题造成。

## 核验与决定

3项测试通过：克隆隔离和交叉分解恒等式、单次调用与完整区间起点动作一致、33维状态恢复及HPF对当前/下一tick的区别。独立重新计算全部{native['all_window_metric_values_recomputed']}个窗口误差量，并从保存状态调用c9af原API复查每条轨迹固定0/0.48/0.98s的全部14分支，共{native['saved_state_native_calls_verified']}次，命令与限幅逐值一致。v40原动作指标、原支持分层重组、父产物和源码哈希保持。

本轮没有降低原v40绝对门槛，也没有改善任何真实可部署预测：原p/q/姿态门槛仍失败，模型未晋升。下一步优先准备**针对滚转/俯仰姿态持续误差的受控训练对照**，先核对运动学一致性与现有损失量级，再用fit-only依据冻结单项姿态监督权重，保持网络、数据、初始化、抽样、预算和评价不变；先复现零权重基线。不先调控制器、不新增观测通道，也不因oracle有效而宣称训练一定有效。

具体准备任务见[next_execution_plan.json](next_execution_plan.json)，本轮未启动该新训练，未判断必须补飞。
'''
    (OUT/'report_zh.md').write_text(report)
    for root in PARENTS:verify_parent(root)
    for name,digest in protocol['code'].items():assert base.sha(base.ROOT/name)==digest
    code=dict(protocol['code'])
    for name in ['scripts/evaluate_action_propagation_v41.py','scripts/verify_action_propagation_v41.py','scripts/report_action_propagation_v41.py']:code[name]=base.sha(base.ROOT/name)
    base.write(OUT/'status.json',dict(status='complete_verified',model_promoted=False))
    base.write(OUT/'delivery_manifest.json',dict(artifacts={str(p.relative_to(OUT)):base.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'},
        code=code,model_promoted=False,validation_accessed=False,sealed_test_accessed=False,reserved_accessed=False))
    verify_parent(OUT)
    print(decision,flush=True)


if __name__=='__main__':main()
