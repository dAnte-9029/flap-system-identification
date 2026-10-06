"""Report frozen tilt-loss comparison without automatic model promotion."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/flap-v43-matplotlib')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_control_windows_v2 as base
from train_attitude_v42 import OUT,PREP,DATA,read
from evaluate_attitude_v43 import V35,V40
from audit_controller_support_v26 import verify_parent


def table(frame):
    rows=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    for row in frame.itertuples(index=False,name=None):
        rows.append('| '+' | '.join(f'{x:.6f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |')
    return '\n'.join(rows)


def main():
    for mode in ['conditional','coupled']:assert read(OUT/mode/'verification.json')['status']=='passed'
    frozen=read(OUT/'evaluation_protocol.json')
    for name,digest in frozen['code'].items():assert base.sha(base.ROOT/name)==digest
    for parent in [PREP,DATA,V35,V40]:verify_parent(parent)
    cond=read(OUT/'conditional/gates.json')[0];coupled=read(OUT/'coupled/gates.json')
    cp=pd.read_csv(OUT/'conditional/comparison.csv');bp=pd.read_csv(OUT/'coupled/comparison.csv')
    summary=cp[(cp.horizon==50)&(cp.kind=='wave')].groupby('metric')[['baseline','candidate']].mean()
    summary['improvement_percent']=100*(1-summary.candidate/summary.baseline)
    primary=bp[(bp.variant=='warm2_plus')&(bp.horizon==50)&(bp.kind=='wave')].groupby('metric')[['baseline','candidate']].mean()
    primary['improvement_percent']=100*(1-primary.candidate/primary.baseline)
    actions=pd.read_csv(OUT/'coupled/action_summary.csv');a=actions[(actions.variant=='warm2_plus')&(actions.comparison=='log')]
    a=a.groupby(['axis','arm']).rmse.mean().unstack();a['improvement_percent']=100*(1-a.candidate/a.baseline)
    support=pd.read_csv(OUT/'coupled/support_summary.csv')
    abs_cond=pd.read_csv(OUT/'conditional/absolute_gates.csv').groupby('metric')[['rmse','limit']].mean()
    abs_coupled=pd.read_csv(OUT/'coupled/absolute_gates.csv');abs_main=abs_coupled[abs_coupled.variant=='warm2_plus'].groupby('metric')[['rmse','limit']].mean()
    failures=pd.concat([pd.read_csv(OUT/'conditional/state_protections.csv').assign(evaluation='conditional'),
                        pd.read_csv(OUT/'coupled/state_protections.csv').assign(evaluation='coupled')])
    failures=failures[~failures.passed];failures.to_csv(OUT/'protection_failures.csv',index=False)
    gateframe=pd.DataFrame([dict(variant=g['variant'],tilt_improvement_percent=g['tilt']['improvement_percent'],
        surface_improvement_percent=g['surface_improvement_percent'],surface_seeds=g['surface_improving_seeds'],surface_flights=g['surface_improving_flights'],
        state_protections=f"{g['protections_passed']}/27",action_protections=f"{g['action_protections_passed']}/12",absolute=f"{g['absolute_passed']}/12",
        coupled_relative_gate=g['coupled_relative_gate']) for g in coupled])
    relative=cond['tilt_with_protections'] and all(g['coupled_relative_gate'] for g in coupled)
    absolute=cond['absolute_passed']==12 and all(g['absolute_passed']==12 for g in coupled)
    decision=dict(status='complete_verified',completed_trainings=15,steps_each=1600,conditional_tilt_target=cond['tilt']['passed'],
        conditional_relative_gate=cond['tilt_with_protections'],all_coupled_relative_gates=all(g['coupled_relative_gate'] for g in coupled),
        all_frozen_relative_gates=relative,all_absolute_gates=absolute,control_ready=False,model_promoted=False,
        baseline_retained='A1600 and v35 frozen research baseline',validation_accessed=False,sealed_test_accessed=False,reserved_accessed=False)
    base.write(OUT/'decision.json',decision)
    # The following investigation is a proposal, never an automatic extra sweep.
    plan=dict(status='proposed_not_started',objective='Localize remaining prediction/action errors before further model changes',
        evidence='Use v43 fit/OOF and four coupled comparisons, including every failed protection; do not retune lambda on these repeated OOF results.',
        steps=['Freeze all15 checkpoints and all407 parents; compare fit-vs-OOF tilt/rates and errors by flight/mode/time within the1s horizon.',
               'Pair candidate andv35 trajectories under identical reference/initial states; separate conditional changes from generated-action changes, preserving difficult windows.',
               'Use measured-rate kinematics and reference/action evidence only as labelled diagnostics; select at most one next hypothesis after identifying whether residuals are fit capacity, cross-flight transfer or accumulated dynamics.',
               'Keep original absolute gates and sealed boundaries; no controller changes or candidate deployment based on this comparison.'],
        training=False,controller_tuning=False,new_flight=False)
    base.write(OUT/'next_execution_plan.json',plan)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,title,frame,labels in [(axes[0],'Conditional full1s',summary,['roll','pitch','tilt']),
                                (axes[1],'Coupled warm2_plus full1s',primary,['roll','pitch','tilt'])]:
        x=np.arange(3);ax.bar(x-.18,frame.loc[labels,'baseline'],.36,label='v35');ax.bar(x+.18,frame.loc[labels,'candidate'],.36,label='v43')
        ax.set_xticks(x,labels);ax.set_ylabel('RMSE (rad)');ax.set_title(title);ax.legend();ax.grid(axis='y',alpha=.2)
    x=np.arange(4);axes[2].bar(x-.18,a.baseline,.36,label='v35');axes[2].bar(x+.18,a.candidate,.36,label='v43')
    axes[2].set_xticks(x,['motor','servo0','servo1','servo2']);axes[2].set_title('Coupled action vs log');axes[2].set_ylabel('Command RMSE');axes[2].legend();axes[2].grid(axis='y',alpha=.2)
    fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
    cv=read(OUT/'conditional/verification.json');bv=read(OUT/'coupled/verification.json')
    weights=read(PREP/'weights.json');fit=pd.read_csv(OUT/'conditional/fit_wave_metrics.csv')
    fit=fit.groupby(['arm','metric']).rmse.mean().unstack(0);fit['improvement_percent']=100*(1-fit.candidate/fit.baseline)
    text=f'''# 倾斜姿态监督训练与闭环评价 v43

**15组×1600步训练、全部407父窗条件预测、4种受限闭环重放及独立核验已完成。**

条件预测倾斜误差目标通过：{cond['tilt']['passed']}；含27项状态保护：{cond['tilt_with_protections']}。四种闭环的全部相对门槛通过：{decision['all_coupled_relative_gates']}；全部绝对门槛通过：{absolute}。本轮不晋升控制模型，保留A1600与v35研究基线。

## 实验合同

严格执行v42预先冻结方案：原v35完整递推＋局部增量损失保留，新增单位重力方向平方差/0.1²。五折系数分别为{', '.join(f"{w['weight']:.6f}" for w in weights)}，按fit初始梯度确定，未在本轮调权。H26/K50、40070参数、初始参数、父窗与local offset采样、归一化、AdamW和1600更新预算保持；从原初值训练，不是微调v35。

上一轮fold0/seed17完整1600步零权重复现已逐值通过，本轮复用证据。15个正权重候选均完成。数据是原5个训练飞行407父窗的留一飞行OOF，已经多轮用于开发，**不是未开封的独立测试**；开发验证、sealed test、reserved未访问。原数据为专门登记的9月日志命令数据，沿用v29时间/相位/扑频合同，不混入canonical旧数据或DeLaurier先验。

## 条件预测

给定未来日志分配命令的完整1秒波形RMSE，先飞行内RMSE，再五飞行、三seed等权；角度rad、角速度rad/s、速度m/s：

{table(summary.reset_index())}

倾斜改善{cond['tilt']['improvement_percent']:.3f}%，改善seed {cond['tilt']['improving_seeds']}/3、飞行{cond['tilt']['improving_flights']}/5；原q条件另计，改善{cond['legacy_q']['improvement_percent']:.3f}%，通过{cond['legacy_q']['passed']}。27项状态保护通过{cond['protections_passed']}/27。新增tilt通过不等于q条件或全姿态绝对门槛通过。

活动窗0.5秒端点，三seed均值仅供阅读，实际逐seed验收通过{cond['absolute_passed']}/12：

{table(abs_cond.reset_index())}

fit完整1秒波形结果（每模型内各fit飞行等权，再模型等权；与OOF来源/用途不同）：

{table(fit.reset_index())}

## 同一控制器的生成动作重放

复用v40逐值核验的因果2秒控制器预热、参考带、固定起点CAS及原生分配，使用候选状态生成未来动作，GRU不读取未来实测状态或日志执行命令。模型20ms推进，控制器2.5ms/姿态10ms和20ms替代调度保持；模型消费区间起点动作，内部微步单独记录。参考来自日志条件，Mission不包含完整导航/TECS复现；固定空速/扑频及50Hz输入保持仍是近似。

四种假设均保留全部407窗×3seed，共4884条新模型轨迹，另复用4884条冻结v35轨迹及1628条匹配teacher。没有删困难片段、增加动作clamp或调整控制参数。两升降副翼指标是servo0与servo1各自RMSE的算术平均，不与电机/方向舵混合。

{table(gateframe)}

warm2_plus完整1秒状态误差：

{table(primary.reset_index())}

warm2_plus分配命令对实测日志RMSE（axis0电机、1/2升降副翼、3方向舵）：

{table(a.reset_index())}

warm2_plus活动窗0.5秒端点：

{table(abs_main.reset_index())}

命令/状态越出fit逐通道范围和原生分配限幅的比例：

{table(support)}

范围仅为fit边际min/max，不是联合支持或校准置信度。完整mode分层、每飞行/seed结果、匹配teacher命令误差与内部微步差异均已保存；这些指标不用于事后删除窗口。

![对照](comparison.png)

## 验证与结论边界

10项针对性测试通过，包括原v42损失、耦合接口及本轮评价。全部15检查点的fit和OOF预测逐值重放；原始初始化、buffer、采样哈希、参数数目及前缀因果性通过。独立公式复算条件预测{cv['state_metric_values_recomputed']}个窗口指标，闭环{bv['state_metric_values_recomputed']}个状态指标和{bv['action_metric_values_recomputed']}个动作指标，核验{bv['causal_support_points_recomputed']}个当前状态支持点；所有相对/绝对门槛独立复算一致。父产物及源码哈希不变。

见[条件预测核验](conditional/verification.json)、[闭环核验](coupled/verification.json)、[完整保护失败清单](protection_failures.csv)。`control_ready=false`；即使某些相对门槛通过，也不替代独立控制响应证据和真实闭环验证。

下一步仅提出[残余误差定位计划](next_execution_plan.json)：冻结本轮结果，比较fit/OOF及条件/生成动作之间的误差变化，先确定剩余问题再选择一个新假设。本轮没有追加权重搜索、控制器调参或实飞。
'''
    (OUT/'report_zh.md').write_text(text)
    base.write(OUT/'status.json',decision)
    code=dict(frozen['code']);code['scripts/report_attitude_v43.py']=base.sha(base.ROOT/'scripts/report_attitude_v43.py')
    base.write(OUT/'delivery_manifest.json',dict(artifacts={str(p.relative_to(OUT)):base.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'},
        code=code,model_promoted=False,control_ready=False,validation_accessed=False,sealed_test_accessed=False,reserved_accessed=False))
    verify_parent(OUT);print(decision,flush=True)


if __name__=='__main__':main()
