"""Summarize frozen paired results without changing evaluation or training."""
import json
import numpy as np
import pandas as pd
import train_control_windows_v2 as base
from controller_history_training_v25 import OUT,ARMS


def table(frame):
    lines=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    for row in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(lines)


def main():
    result=json.loads((OUT/'evaluation.json').read_text())
    verify=json.loads((OUT/'verification.json').read_text());assert verify['status']=='passed'
    comparison=pd.read_csv(OUT/'comparison.csv');flights=pd.read_csv(OUT/'metrics_per_flight.csv')
    chosen=comparison[(comparison.cohort=='oof')&(comparison.kind=='wave')&(comparison.horizon==50)]
    overall=chosen.groupby('metric')[ARMS].mean()
    overall['改善百分比']=100*(1-overall[ARMS[1]]/overall[ARMS[0]])
    overall=overall.rename(columns={ARMS[0]:'零历史对照',ARMS[1]:'真实历史'}).reset_index()
    qseed=chosen[chosen.metric=='q'][['seed']+ARMS+['improvement_percent']]
    qf=flights[(flights.cohort=='oof')&(flights.kind=='wave')&(flights.horizon==50)&(flights.metric=='q')]
    qf=qf.groupby(['log_id','arm']).rmse.mean().unstack()
    qf['improvement_percent']=100*(1-qf[ARMS[1]]/qf[ARMS[0]])
    qf.reset_index().to_csv(OUT/'q_per_flight.csv',index=False)
    fit=comparison[(comparison.cohort=='fit')&(comparison.kind=='wave')&(comparison.horizon==50)].groupby('metric')[ARMS].mean()
    fit['improvement_percent']=100*(1-fit[ARMS[1]]/fit[ARMS[0]])
    protection=comparison[(comparison.cohort=='oof')&(((comparison.kind=='wave')&(comparison.horizon==50)&comparison.metric.isin(['p','r']))|
        ((comparison.kind=='endpoint')&comparison.horizon.isin([25,50])&comparison.metric.isin(['velocity','attitude'])))]
    protection.to_csv(OUT/'protection_gate.csv',index=False)
    absolute=pd.read_csv(OUT/'absolute_active_gate.csv');absolute=absolute[absolute.arm==ARMS[1]][['seed','metric','rmse','limit','passed']]
    held=pd.read_csv(OUT/'reference_hold_diagnostic.csv')
    held=held.groupby(['seed','arm','metric','log_id'])[['factual_mse','held_mse','effect_mse']].mean().pow(.5).groupby(['arm','metric']).mean().reset_index()
    held=held.rename(columns={'factual_mse':'factual_rmse','held_mse':'held_rmse','effect_mse':'effect_rms'})
    held.to_csv(OUT/'reference_hold_summary.csv',index=False)
    secondary=pd.read_csv(OUT/'secondary_v23.csv');secondary=secondary[(secondary.cohort=='oof')&(secondary.kind=='wave')&(secondary.horizon==50)]
    secondary=secondary.groupby('metric')[['v23',ARMS[0]]].mean()
    secondary['zero_vs_v23_percent']=100*(1-secondary[ARMS[0]]/secondary.v23)
    decision=dict(**result,stage_status='complete',full_407_candidate_refit=False,
        selected_checkpoint='none',training_scope='Original407train parents only;5flight folds x3seeds x2arms',
        conclusion='Joint past controller observation block evaluated against same enlarged zero-input architecture. See paired gates; no deployment.',
        next_step_status='not_started',
        next_step='Freeze these30models. Audit train-flight support/distribution of the8past controller channels and locate q regression within original frequency/mode strata; no new training, window removal, threshold search or validation/test access.')
    base.write(OUT/'decision.json',decision)
    text=f'''# 历史控制器状态配对训练 v25

完成30/30组训练，每组1600步。原407个非重叠父窗口全部保留，5飞行×3种子×2组；开发验证、sealed/reserved均未访问。数值主门槛与保护联合结果：{'通过' if result['numerical_gate'] else '未通过'}。保留A1600；本轮没有全407重训候选或控制器部署。

新增历史块未解决q问题：训练内完整1s q改善{fit.loc['q','improvement_percent']:.4f}%，OOF改善为{result['primary_q_improvement_percent']:.4f}%（负数为退化），全部3种子和5飞行都未改善。seed42的0.5s姿态端点退化5.2952%，触发保护失败。该结果不支持直接采用这8维联合历史块，也不能据此认定所有控制器状态信息无用或必须补飞。

## 试验边界与数据

本轮执行v24已冻结方案。数据来自September自定义reference_dynamics_data_v1合同，祖先清单见protocol.json，不使用April canonical。407窗中274个Stabilized、133个Mission，总父窗口支撑618.64s；H26/K50、50Hz不变。mechanical_phase_rad为唯一机械相位，September频率比7.909090995788574；物理相位零位仍未确认，不能套用April ratio8合同。

角速度参考模型保留固定rate内环、分配、执行器与飞机，排除姿态环。只在过去26点历史编码器增加8列：三轴更新后积分、三轴已发布压缩增益、两组发布年龄。所有发布不晚于各历史查询时刻；无未来积分、增益、执行器输入。增益发布稀疏，发布年龄不是物理时延，新增块不等于恢复完整控制器内部状态。v24代数反推空速缩放不进入本轮训练。

两组同为GRU64、41606参数，编码器25→33；新增权重列零初始化，其余初值、1600×32抽样、AdamW设置、完整1s损失均相同。零组在归一化后清零新增8列，真实组使用过去观测。五折控制器归一化仅用各折fit飞行唯一历史行；其他归一化沿用v23。冻结v23只作次要参考，主比较是本轮重训的零历史组。

## 完整1秒OOF波形

先在每飞行内平均平方误差后开方，再对飞行与种子等权。p/q/r单位rad/s，velocity是三分量平均的RMSE（m/s），attitude是四元数测地角（rad）。正改善百分比表示真实历史更好。

{table(overall)}

q主指标改善{result['primary_q_improvement_percent']:.4f}%，要求≥5%；全部种子为正：{result['all_seeds_positive']}；改善飞行{result['positive_flights']}/5，要求至少4/5。

{table(qseed)}

{table(qf.reset_index())}

## 训练内与保护指标

训练内同口径结果用于判断是否只存在跨飞行泛化问题，不用于选检查点：

{table(fit.reset_index())}

冻结保护为每seed的p/r完整1s波形和velocity/attitude在0.5/1s端点退化均不超过5%。通过{result['protection_passed']}/{result['protection_total']}项；最差改善百分比{protection.improvement_percent.min():.4f}%。逐项记录在protection_gate.csv。

原活动窗口0.5s绝对门槛通过{result['absolute_active_passed']}/{result['absolute_active_total']}项（姿态表内单位rad，3°对应0.05236rad）：

{table(absolute)}

各模式保持原窗口标签，详见mode_comparison.csv；Mission飞行支持有限，不能把模式分层当独立新验证集。20/100ms及0.5/1s完整波形与端点见comparison.csv，逐窗和逐飞行明细完整保存。

## 参考保持诊断与次要对照

未来rate3/thrust保持为起点值、reset不变。下表分别为原参考误差、保持参考误差、两预测之差RMS。这是预声明的模型敏感性诊断，不是真实反事实或气动舵效。

{table(held)}

冻结v23与本轮零组对照如下。独立核验15个零组模型的fit/OOF共30组完整预测数组，与v23逐值完全一致（最大差0），确认零输入扩展在本次训练中复现了旧行为。主结论始终来自同结构两组，不用v23取代零组。

{table(secondary.reset_index())}

## 验证与结论边界

6项针对性测试通过（含v24合同3项）；检验非零读出下旧模型一致性、零组不依赖新增观测/无新增权重梯度、真实组可学习、未来参考前缀因果性。15对初始权重与抽样一致；30个检查点完成完整OOF和参考保持预测精确重放，每seed/arm恰好覆盖407个不同窗口。独立从预测复算主门槛、保护和绝对门槛，核验冻结代码、父清单与产物哈希。详细见verification.json、preflight.json、training_checks.json及delivery_manifest.json。

这检验的是过去积分/压缩增益/年龄联合观测块，不能分离各通道的因果贡献。未来日志角速度参考由姿态反馈产生，可能携带未来状态或扰动信息；条件预测改善不能等同于原姿态参考自主预测收益，更不能直接证明新控制器有效。当前绝对精度、独立控制响应和闭环证据仍需分别验收；保留A1600，不因模型内参考敏感性而晋升。这5个飞行的交叉验证结果已用于多轮开发，属于开发证据，不能重新称为未开封最终测试。

## 产物位置

- 历史张量沿用../control_windows_v24/history_controller.npz；参考/目标沿用../control_windows_v23/windows.npz，未再次筛窗。
- 30个模型：fold0–4/{{zero_history_control,observed_history_control}}/seed{{17,23,42}}/final.pt。
- 每模型同目录保存训练曲线、初值/批次哈希、fit/oof_predictions.npz及oof_held.npz。
- protocol.json固定设计与源码；evaluation.json和decision.json给出结论；delivery_manifest.json固定交付哈希。

## 下一步（尚未执行）

先冻结本轮30个模型，审计新增8维历史的跨飞行覆盖与分布，并按原有扑频/模式层定位q退化。归一化与支持阈值只在各折fit飞行确定，查询飞行只作诊断；检查积分、压缩增益和发布年龄是否主要编码飞行差异，比较训练内轻微收益为何未转移到留出飞行。先做这一有限诊断，不直接继续扩网络、扫描损失权重、删除困难窗口或开启新验证集。关联结果仍不能唯一证明退化原因；是否拆分输入块训练，需要诊断后另行冻结对照。
'''
    (OUT/'report_zh.md').write_text(text)
    protocol=json.loads((OUT/'protocol.json').read_text())
    code=dict(protocol['code_pins'])
    for path in ['scripts/verify_controller_history_v25.py','scripts/report_controller_history_v25.py']:
        code[path]=base.sha(base.ROOT/path)
    artifacts={str(path.relative_to(OUT)):base.sha(path) for path in sorted(OUT.rglob('*'))
        if path.is_file() and path.name!='delivery_manifest.json'}
    base.write(OUT/'delivery_manifest.json',dict(stage='v25',status='complete',artifacts=artifacts,code=code,
        completed_trainings=30,verification='verification.json',promoted=False,control_ready=False,
        validation_accessed=False,sealed_test_accessed=False))


if __name__=='__main__':main()
