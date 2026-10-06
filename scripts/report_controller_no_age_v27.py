"""Report primary zero-control and secondary full-history comparisons."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import train_control_windows_v2 as base
from controller_no_age_v27 import OUT,V25


def table(frame):
    def fmt(v):return f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v)
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])


def main():
    e=json.loads((OUT/'evaluation.json').read_text());verify=json.loads((OUT/'verification.json').read_text());assert verify['status']=='passed'
    preflight=json.loads((OUT/'preflight.json').read_text());assert preflight['full1600step_zero_control_exact']
    c=pd.read_csv(OUT/'comparison.csv');full=pd.read_csv(OUT/'secondary_full_history.csv')
    selected=full[(full.horizon==50)&(full.kind=='wave')]
    columns=['zero_history_control','observed_history_control','no_age_history_control']
    overall=selected[selected.cohort=='oof'].groupby('metric')[columns].mean()
    overall['vs_zero_improvement_percent']=100*(1-overall.no_age_history_control/overall.zero_history_control)
    overall['vs_full_improvement_percent']=100*(1-overall.no_age_history_control/overall.observed_history_control)
    fit=selected[selected.cohort=='fit'].groupby('metric')[columns].mean()
    fit['vs_zero_improvement_percent']=100*(1-fit.no_age_history_control/fit.zero_history_control)
    seed=selected[(selected.cohort=='oof')&(selected.metric=='q')][['seed']+columns+['improvement_percent','no_age_vs_full_percent']]
    flights=pd.read_csv(OUT/'metrics_per_flight.csv');old=pd.read_csv(V25/'metrics_per_flight.csv')
    flights=pd.concat([flights,old[old.arm=='observed_history_control']],ignore_index=True)
    q=flights[(flights.cohort=='oof')&(flights.metric=='q')&(flights.horizon==50)&(flights.kind=='wave')]
    q=q.groupby(['log_id','arm']).rmse.mean().unstack()
    q['vs_zero_improvement_percent']=100*(1-q.no_age_history_control/q.zero_history_control)
    q['vs_full_improvement_percent']=100*(1-q.no_age_history_control/q.observed_history_control)
    q.reset_index().to_csv(OUT/'q_per_flight.csv',index=False)
    protection=c[(c.cohort=='oof')&(((c.kind=='wave')&(c.horizon==50)&c.metric.isin(['p','r']))|
        ((c.kind=='endpoint')&c.horizon.isin([25,50])&c.metric.isin(['velocity','attitude'])))]
    protection.to_csv(OUT/'protection_gate.csv',index=False)
    absolute=pd.read_csv(OUT/'absolute_active_gate.csv');absolute=absolute[absolute.arm=='no_age_history_control'][['seed','metric','rmse','limit','passed']]
    response=pd.read_csv(OUT/'reference_hold_diagnostic.csv')
    response=response.groupby(['seed','arm','metric','log_id'])[['factual_mse','held_mse','effect_mse']].mean().pow(.5).groupby(['arm','metric']).mean().reset_index()
    response=response.rename(columns={'factual_mse':'factual_rmse','held_mse':'held_rmse','effect_mse':'effect_rms'})
    response.to_csv(OUT/'reference_hold_summary.csv',index=False)
    base.write(OUT/'decision.json',dict(**e,status='complete',full_407_refit=False,age_columns_normalized_zero=True,
        v25_control_reused_after_exact1600step_reproduction=True,secondary_q_vs_full_percent=float(overall.loc['q','vs_full_improvement_percent']),
        next_step_status='not_started',
        next_step='Retain baseline; stop adding controller-history channels on this evidence. Audit post-allocation actuator-command semantics/timing/coverage on the same407windows before deciding an actuator-conditioned model boundary; no new training yet.',
        limits='This ablation changes information available during training; it does not identify causal contributions of telemetry age or prove control readiness.'))
    report=f'''# 去除发布年龄的历史状态消融 v27

完成15/15候选训练（5折×3seed，各1600步），另完成1次零历史对照1600步精确复现。原407个train父窗口全部保留，无开发验证/sealed/reserved访问。相对零历史组，q完整1s OOF改善百分比为{e['primary_q_improvement_percent']:.4f}%（负数表示退化）；相对完整8维历史组为{overall.loc['q','vs_full_improvement_percent']:.4f}%。主门槛及保护联合结果：{'通过' if e['numerical_gate'] else '未通过'}。保留A1600，本轮不晋升或部署控制器。

去掉年龄后，log_17从相对零组退化6.16%变为改善0.46%，但log_19仍退化3.74%。3seed中17/23仅改善0.284%/0.048%，42退化2.773%；只2/5飞行改善。训练内q改善1.294%，也未转化为整体OOF收益。因此，年龄屏蔽缓解了部分跨飞行问题，但不足以证明剩余6维历史有稳定收益；本轮不采用这项扩展，不把年龄当作全部误差的原因。

## 唯一干预与复用条件

执行v26冻结方案：保留3积分、3压缩增益历史，屏蔽2发布年龄。编码器33输入、41606参数均不变，通过将原始年龄替换为同折训练均值，使其归一化后严格为0；未修改数据文件或重拟合归一化。年龄权重两列从零初始化且训练后仍为0；其余6列可获得梯度。所有新增观测均在过去26点，未来递推没有控制器状态。

与v25相同的407原窗口（274Stabilized、133Mission）、H26/K50、50Hz、初值、1600×32保存抽样、AdamW3e-4/weight_decay1e-4/clip1和原完整1s损失。只使用最终1600步检查点，无挑选早停或扫描参数。五折归一化仍只来自fit飞行。

核验Python/Torch/Numpy/Pandas版本一致；先用新封装的全零开关完整复现fold0/seed17零组，所有参数和buffer与v25最终检查点逐值一致，才复用冻结零组与完整8列组。15组初值/批次均匹配原对照；无需重新训练30个已有对照。

数据属于September自定义reference_dynamics_data_v1，父合同/清单/样本哈希见protocol.json；不使用April canonical。沿用原机械相位和扑频，September比值7.909090995788574，相位物理零位仍未确认。模型以未来日志rate参考为条件，包含rate内环/分配/执行器/飞机、排除姿态环；不等同于自主姿态参考模型。

## 完整1秒OOF波形

每飞行内先平均平方误差后开方，再飞行/种子等权。角速度rad/s，速度是分量平均RMSE（m/s），姿态是测地角（rad）。zero为主对照，observed为完整8列，no_age为本轮6列；改善百分比越大越好。

{table(overall.reset_index())}

q主门槛要求相对zero≥5%改善、3seed均正、至少4/5飞行改善；本次种子全部为正：{e['all_seeds_positive']}，改善飞行{e['positive_flights']}/5。与完整8列的比较为次要指标，不能取代主门槛。

{table(seed)}

{table(q.reset_index())}

## 训练内与保护

{table(fit.reset_index())}

保护项为每seed的p/r完整1s波形、速度/姿态0.5/1s端点，退化不能超过5%。通过{e['protection_passed']}/{e['protection_total']}；最差改善百分比{protection.improvement_percent.min():.4f}%。全18项见protection_gate.csv。

原活动窗口0.5s绝对精度门槛通过{e['absolute_active_passed']}/{e['absolute_active_total']}。姿态表内限值3°=0.05236rad：

{table(absolute)}

全部20ms/100ms/0.5s/1s波形与端点、fit/OOF和模式结果见comparison.csv、mode_comparison.csv及逐窗/逐飞行表。训练内窗口跨折重复出现，不是额外独立样本。

## 参考保持诊断

将未来rate3/thrust保持为起点值，reset不变；列分别是原参考误差、保持参考误差和两预测差RMS。这是冻结的模型敏感性诊断，不是真实反事实或舵效验证。

{table(response)}

## 核验及解释范围

3项针对性测试通过：非零读出下与显式年龄屏蔽等价、输入不被原地修改、年龄变化不影响预测/梯度、六列仍有梯度、参考前缀因果性。15个候选检查点完整fit/OOF/held预测逐值重放；改变所有历史年龄后全部OOF预测逐值不变；每seed恰好407个唯一OOF窗口。主/保护/绝对指标独立从预测数组复算，父产物与冻结代码哈希一致。原对照在本轮精确复现之后复用，详见preflight.json和verification.json。

这检验的是训练时是否提供年龄信息，不能直接识别传感器时序或气动因果作用。即使优于完整8列，也必须优于零历史组并达到绝对精度，才有理由继续推进。全部5飞行OOF已用于多轮开发，不是未开封最终测试；独立控制响应与闭环证据仍缺失。没有全407重训或部署。

模型位于fold0–4/no_age_history_control/seed17、23、42/final.pt；同目录包含训练曲线、初值/抽样哈希和预测数组。数据仍在v23/v24、对照仍在v25，不复制或更改原始窗口。delivery_manifest.json固定全部本轮产物与代码哈希。

## 后续建议（尚未执行）

保留现有基线，暂不继续叠加控制器历史通道。下一步先在原407窗核验各实际分配后执行命令的通道语义、PWM映射、发布时间及覆盖，准备把稳定内环与执行器/飞机响应分开研究的输入合同；先审计，不直接启动另一轮训练。已有局部偏航链路核验可以复用，仍需覆盖其他通道。执行命令不是实测舵角或气动力；未来日志命令是反馈生成的条件输入，不能把新边界下的预测改善当成原参考驱动模型的自主控制收益。
'''
    (OUT/'report_zh.md').write_text(report)
    code=dict(json.loads((OUT/'protocol.json').read_text())['code_pins'])
    for name in ['scripts/verify_controller_no_age_v27.py','scripts/report_controller_no_age_v27.py']:code[name]=base.sha(base.ROOT/name)
    base.write(OUT/'delivery_manifest.json',dict(status='complete',stage='v27',artifacts={str(p.relative_to(OUT)):base.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'},
        code=code,completed_candidate_trainings=15,control_reproduction_trainings=1,model_promoted=False,validation_accessed=False,sealed_test_accessed=False))


if __name__=='__main__':main()
