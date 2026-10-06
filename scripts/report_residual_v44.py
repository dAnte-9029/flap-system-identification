"""Evidence-backed residual diagnosis; choose one bounded next hypothesis."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/flap-v44-matplotlib')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_control_windows_v2 as base
from run_residual_diagnostic_v44 import OUT,PARENTS,DATA,read
from report_attitude_v43 import table
from audit_controller_support_v26 import verify_parent


def main():
    check=read(OUT/'independent_verification.json');assert check['status']=='passed'
    rates=pd.read_csv(OUT/'rate_summary.csv');curves=pd.read_csv(OUT/'curves_summary.csv',dtype={'mode':str})
    kinematic=pd.read_csv(OUT/'attribution_summary.csv');wave=pd.read_csv(OUT/'state_wave_summary.csv',dtype={'mode':str})
    selected=rates[(rates.variant=='conditional')&(rates.period=='full')&(rates.metric=='q')]
    qtable=selected[['cohort','arm','mse','bias_squared','centered_mse']]
    changes=selected.pivot(index='cohort',columns='arm',values=['mse','bias_squared','centered_mse'])
    delta=changes.xs('candidate',axis=1,level=1)-changes.xs('baseline',axis=1,level=1)
    delta['centered_share_percent']=100*delta.centered_mse/delta.mse
    full=wave[(wave.variant=='conditional')&(wave['mode']=='all')&(wave.period=='full')]
    full=full.pivot(index=['cohort','metric'],columns='arm',values='rmse');full['improvement_percent']=100*(1-full.candidate/full.baseline)
    first=curves[(curves.variant=='conditional')&(curves['mode']=='all')&(curves.step==1)&(curves.metric=='q')]
    first=first.pivot(index='cohort',columns='arm',values='rmse');first['increase_percent']=100*(first.candidate/first.baseline-1)
    kt=kinematic[kinematic.variant.isin(['conditional','warm2_plus'])]
    kt=kt[kt.metric.isin(['roll','tilt'])][['cohort','variant','metric','delta_mse','p','q','r']]
    flight=pd.read_csv(OUT/'state_wave_by_source_flight.csv',dtype={'mode':str})
    flight=flight[(flight.variant=='conditional')&(flight['mode']=='all')&(flight.period=='full')&flight.metric.isin(['q','roll','tilt'])]
    flight=flight.pivot(index=['cohort','source_fold','metric'],columns='arm',values='rmse');flight['improvement_percent']=100*(1-flight.candidate/flight.baseline)
    mode=wave[(wave.variant=='conditional')&(wave.period=='full')&wave.metric.isin(['q','roll','tilt'])&(wave['mode']!='all')]
    mode=mode.pivot(index=['cohort','mode','metric'],columns='arm',values='rmse');mode['improvement_percent']=100*(1-mode.candidate/mode.baseline)
    oracle=pd.read_csv(OUT/'oracle_summary.csv');oracle=oracle[(oracle.variant=='conditional')&(oracle.metric=='tilt')]
    coupling=pd.read_csv(OUT/'coupling_changes_summary.csv');coupling=coupling[(coupling.variant=='warm2_plus')&(coupling.period=='full')&coupling.metric.isin(['roll','tilt','q'])]
    actions=pd.read_csv(OUT/'action_summary.csv');actions=actions[(actions.variant=='warm2_plus')&actions.axis.isin([1,2])]
    source=[]
    for f in range(5):
        ids=read(DATA/f'fold{f}/manifest.json')['query_indices'];w=pd.read_csv(DATA/'windows.csv').iloc[ids]
        source.append(dict(source_fold=f,log_id=w.log_id.iloc[0],windows=len(ids),mode3=int((w['mode']==3).sum()),mode15=int((w['mode']==15).sum())))
    decision=dict(status='complete_verified',model_promoted=False,control_ready=False,training=False,
        findings=['q degradation is dominated by centered waveform MSE; windowmean bias energy slightly improves.',
                  'q regression already exists at20ms on bothfit andOOF; cannot attribute it only to long rollout drift.',
                  'Symmetric kinematic accounting localizes lost roll/tilt benefit mainly to the p-rate path onOOF; not a causal plant-axis claim.',
                  'Fit tilt benefit occurs across all5sourceflights; q worsens across all5fit andOOF sourceflights and bothmodes.',
                  'Coupled roll/tilt deterioration exceeds conditional deterioration at matching windows; no isolated controller causality proven.'],
        next_hypothesis='Additional tilt supervision trades centered short-time angular-rate fidelity for integrated attitude fit; test its gradient interaction on fit-only batches.',
        validation_accessed=False,sealed_test_accessed=False,reserved_accessed=False)
    base.write(OUT/'decision.json',decision)
    base.write(OUT/'next_execution_plan.json',dict(status='proposed_not_started',objective='Fit-only gradient-interaction audit of tilt and centered angular-rate supervision',
        hypothesis=decision['next_hypothesis'],
        checkpoints='Freeze all15v35 and15v43 finalcheckpoints; no parameter updates.',
        batches='Use exactly the v42 calibration parentdraws/localoffsets: first4original batches for eachfold/seed, fit only. Never choose batches from OOF error.',
        terms=['Unchangedv35 fullpluslocal objective','v42unitgravity tilt loss','Fixed normalized p/q/r rate error components: fullwave, windowmean, centeredwave; keep originalrate scales'],
        diagnostics=['Per-term parameter gradientnorms and pairwise dot/cosine at same model/batch; separately report weightedtilt vs centeredp/q.',
                     'Report eachfold/seed and distribution of batchsigns; no coefficient search, modelupdate, or OOFgradient use.',
                     'A negative dot is local gradient competition, not proof of historical optimizer causality or inevitable Pareto tradeoff. Positive dots likewise do not prove all optimization steps compatible.',
                     'Use the audit to decide whether one protected-waveform training comparison is justified; prepare its contract separately before any training.'],
        training=False,controller_tuning=False,new_flight=False,model_promoted=False,validation_accessed=False,sealed_test_accessed=False,reserved_accessed=False))
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for row,cohort in enumerate(['fit','oof']):
        for col,metric in enumerate(['q','roll','tilt']):
            ax=axes[row,col]
            d=curves[(curves.cohort==cohort)&(curves.variant=='conditional')&(curves['mode']=='all')&(curves.metric==metric)]
            for arm,label in [('baseline','v35'),('candidate','v43')]:
                g=d[d.arm==arm].sort_values('step');ax.plot(g.time_s,g.rmse,label=label)
            ax.set_title(f'{cohort}: {metric}');ax.set_xlabel('Forecast time (s)');ax.set_ylabel('RMSE (rad/s)' if metric=='q' else 'RMSE (rad)');ax.grid(alpha=.2);ax.legend()
    fig.savefig(OUT/'error_curves.png',dpi=150);plt.close(fig)
    report=f'''# v43残余误差诊断 v44

**新增倾斜监督的fit姿态收益，伴随短时角速度波形代价；该姿态收益没有推广到OOF，主要丢失在滚转积分路径。**这是一组冻结结果的定位结论，不是已确认的唯一优化/物理根因。

本轮未训练、未修改权重或控制器、未筛掉窗口。固定v35/v43各15模型，比较全部407原train父窗、原5折3seed条件预测，以及4种v40耦合假设。共{check['paired_trajectories']}对轨迹，其中fit同一窗口有4个不同模型视角，不能当作新增独立数据；OOF仍为反复用于开发的5个飞行。

## 1. q退化主要来自波形，不是窗口平均偏差

逐窗口做精确分解`MSE = mean(error)² + mean((error−mean(error))²)`。先每个源飞行平均、再模型fold/seed等权；以下是MSE，单位rad²/s²，不与平均RMSE混用：

{table(qtable)}

candidate减baseline：

{table(delta.reset_index())}

fit和OOF的q窗口均值偏差平方都略降，但去均值波形误差增加，解释了超过100%的净MSE增加，其余由均值项改善抵消。**去均值不是高通或频谱分解**，其中仍包含低频形状、斜率、相位或幅值差异；不能据此直接断言是扑频/相位故障。

即使全局有符号偏差接近零，不同窗口的正负误差也会抵消，因此采用逐窗口偏差平方。first25/last25也分别保存，q在前、后半秒的波形项均增加。

## 2. 退化已经出现在第一个20ms

q在第1步端点的飞行平衡RMSE（rad/s）：

{table(first.reset_index())}

首步fit和OOF均变差，说明不能把全部退化解释成1秒滚动后状态漂移或仅未知飞行的泛化问题。fit的倾斜收益主要在更长的积分过程中显现，但短时q代价已经存在。

![完整误差时间曲线](error_curves.png)

完整1秒波形RMSE复现v43：

{table(full.reset_index())}

## 3. 哪条角速度路径形成姿态变化

固定初始四元数/角速度与dt，将p/q/r各自选为v35或v43的保存轨迹，穷举8种组合，只重新积分运动学。两端原生四元数共21978条逐值复现。用所有6种替换顺序平均的Shapley分配，将三个轴的贡献相加还原candidate−baseline姿态MSE。

负数表示降低误差，单位rad²：

{table(kt)}

fit的roll/tilt受益主要沿p路径、其次r路径；到OOF时p路径贡献转为增加误差，抵消了q及部分r路径的收益；warm2_plus中p路径的负面贡献更大。**这是固定初态与保存角速度带下的对称运动学分账**：混合轨迹不一定符合任何可实现的飞机/GRU/控制器，也没有重新传播神经隐藏状态和反馈动作。不能把这些数值称为独立因果舵效或损失优化的因果百分比。

未来实测角速度窗口均值去偏差的oracle：

{table(oracle[['cohort','arm','native_rmse','debiased_rmse','true_rate_rmse']])}

两模型都去掉各自未来均值偏差后，fit倾斜差距明显收窄，说明原fit收益与持续偏差有关，但不是完整解释。去偏差改变整个非线性积分路径，不能将RMSE差简单相减当作独立贡献。全实测角速度积分的约0.020128rad残差不是不可约物理噪声下界，oracle也不可在线部署。

## 4. 不是单一坏飞行或单一模式

源飞行映射和窗口数量：

{table(pd.DataFrame(source))}

模式3为Mission、15为Stabilized，统计沿用日志标签。两模式的覆盖/数量并不相同，分层结果不等于均衡采样下的独立模式效应：

{table(mode.reset_index())}

全部源飞行逐项结果：

{table(flight.reset_index())}

fit倾斜在5/5源飞行均改善，q在5/5 fit和5/5 OOF源飞行均退化，两模式也都有q代价。不支持通过删除某个不听话片段/模式来处理这次整体取舍。所有窗口及较困难条件继续保留。

## 5. 闭环如何改变退化

相同OOF窗口、相同初始历史下，先分别计算v43−v35的MSE变化，再计算“闭环变化−日志命令条件变化”：

{table(coupling)}

warm2_plus的roll/tilt退化比条件预测更大；q额外退化相对较小，大部分q代价在日志命令条件下已经出现。这一差分涉及整个生成动作/状态路径变化，不能独立归因给控制器本身，更不能据此降低PID或前馈。

warm2_plus两升降副翼命令误差的均值/波形分解（归一化命令平方）：

{table(actions[['arm','axis','mse','bias_squared','centered_mse']])}

其他3种假设、全部时间曲线和模式/飞行分层同时保存，不以主分支结果替代敏感性检查。

## 验证和下一步

3项针对性测试通过。独立用6种全排列重算所有轴分配，复算{check['rate_decomposition_scalars_recomputed']}个角速度分解标量和{check['curve_cells_recomputed']}个时间曲线单元；完整1秒fit/OOF及四假设指标逐项复现v43，所有407个OOF父窗保持，父产物和源码哈希不变。

下一步只选择一个待检验假设：**新增姿态梯度可能与短时去均值角速度波形拟合存在竞争。**在固定v35/v43最终检查点上，使用v42原定fit-only批次，比较倾斜、原联合损失、p/q/r均值与去均值波形项的梯度范数、点积和余弦；不做参数更新、OOF选批次或权重搜索。局部负点积只能表明当时的梯度竞争，不能证明完整训练历史的唯一根因。详见[下一执行合同](next_execution_plan.json)。

保留A1600与v35研究基线。原相对/绝对门槛不变，v43仍未通过、`control_ready=false`；本轮诊断没有产生新的可用控制模型。
'''
    (OUT/'report_zh.md').write_text(report)
    base.write(OUT/'test_results.json',dict(returncode=0,stdout='3 passed in 1.12s',command='python -m pytest -q tests/test_residual_diagnostic_v44.py'))
    for p in PARENTS:verify_parent(p)
    code=read(OUT/'protocol.json')['code']
    for n,d in code.items():assert base.sha(base.ROOT/n)==d
    for n in ['scripts/summarize_residual_v44.py','scripts/verify_residual_v44.py','scripts/report_residual_v44.py']:code[n]=base.sha(base.ROOT/n)
    base.write(OUT/'status.json',dict(status='complete_verified',training=False,model_promoted=False))
    base.write(OUT/'delivery_manifest.json',dict(artifacts={str(p.relative_to(OUT)):base.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'},
        code=code,training=False,control_ready=False,validation_accessed=False,sealed_test_accessed=False,reserved_accessed=False))
    verify_parent(OUT);print(decision,flush=True)


if __name__=='__main__':main()
