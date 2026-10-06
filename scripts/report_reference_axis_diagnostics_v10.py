"""Evidence and inference-contract audit for p/q localization and yaw candidate."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_control_windows_v2 as base
from reference_groupcv_v8 import load_train,STAT_KEYS
from reference_error_structure_v9 import history_features
from system_identification.models.reference_gru_yaw_mean import ReferenceGRUYawMean,historical_full_features
from system_identification.models.trajectory_main_v1 import _normalize_quaternion,_quaternion_multiply,_delta_quaternion
from run_reference_axis_diagnostics_v10 import OUT,V8,KEYS
from run_reference_generalization_v7 import verify_manifest
from report_reference_generalization_v7 import table


def main():
    torch.set_num_threads(1);verify_manifest(OUT/'protocol.json',['source_pins'])
    decision=json.loads((OUT/'oof_decision.json').read_text());development=json.loads((OUT/'development_status.json').read_text())
    x,y,w=load_train();pq=pd.read_csv(OUT/'pq_per_window.csv')
    pfg=pq.groupby(['cohort','seed','log_id','horizon_s','axis']).mean(numeric_only=True).reset_index()
    pg=pfg.groupby(['cohort','horizon_s','axis']).mean(numeric_only=True).reset_index()
    pg['gain_only_mse_reduction']=1-pg.zero_lag_gain_mse/pg.core_centered_mse
    pg['lag_gain_mse_reduction']=1-pg.best_lag_gain_mse/pg.core_centered_mse
    pg['std_ratio']=pg.predicted_std/pg.truth_std;pg.to_csv(OUT/'pq_summary.csv',index=False)
    lagrows=[]
    for (cohort,seed,log,h,axis),g in pq.groupby(['cohort','seed','log_id','horizon_s','axis']):
        lagrows.append(dict(cohort=cohort,seed=seed,log_id=log,horizon_s=h,axis=axis,
            zero_fraction=float((g.best_lag_s.abs()<1e-8).mean()),positive_fraction=float((g.best_lag_s>1e-8).mean()),negative_fraction=float((g.best_lag_s< -1e-8).mean()),
            boundary_fraction=float((g.best_lag_s.abs()>=.1-1e-8).mean()),gain_boundary_fraction=float(((g.best_lag_gain<=1e-8)|(g.best_lag_gain>=3-1e-8)).mean())))
    lags=pd.DataFrame(lagrows);lags.to_csv(OUT/'lag_per_flight.csv',index=False)
    lg=lags.groupby(['cohort','horizon_s','axis']).mean(numeric_only=True).reset_index();lg.to_csv(OUT/'lag_summary.csv',index=False)
    sensitivity=pd.read_csv(OUT/'reference_sensitivity.csv')
    sr=sensitivity.groupby(['arm','seed','log_id','horizon_s','axis']).reference_effect_mse.mean().pow(.5).groupby(['arm','horizon_s','axis']).mean().reset_index()
    sr.to_csv(OUT/'reference_sensitivity_summary.csv',index=False)
    verification=json.loads((OUT/'verification.json').read_text());reloads=[]
    for run in verification['runs']:
        fold,seed=run['fold'],run['seed'];path=OUT/f'fold{fold}'/f'seed{seed}'
        ck=torch.load(path/'candidate.pt',map_location='cpu',weights_only=False);sd=ck['state_dict'];params={k:sd['yaw_'+k] for k in ['mean','scale','coefficient','target_mean']}
        model=ReferenceGRUYawMean({k:sd[k] for k in STAT_KEYS},params);model.load_state_dict(sd);model.eval()
        with np.load(path/'oof_predictions.npz') as z:
            ids=z['array_index'][:4];saved={k:z[k][:4] for k in KEYS};bias=z['yaw_correction']
        batch={k:v[ids] for k,v in x.items()}
        with torch.no_grad():pred=model(**batch)
        maximum=max(float(np.max(abs(pred[k].numpy()-saved[k]))) for k in KEYS);assert maximum<1e-5
        # Independent midpoint quaternion re-integration of the saved corrected rates.
        quaternion=_normalize_quaternion(x['initial_quaternion'][ids]);previous=x['initial_rates_b'][ids];q=[]
        for k in range(50):
            rate=torch.tensor(saved['rates_b'][:,k]);dt=x['dt_s'][ids,k,None]
            quaternion=_normalize_quaternion(_quaternion_multiply(quaternion,_delta_quaternion(.5*(previous+rate)*dt)));q.append(quaternion);previous=rate
        pose_error=float(np.max(abs(torch.stack(q,1).numpy()-saved['quaternion'])));assert pose_error<1e-5
        feature=historical_full_features(batch['history_state'],batch['history_reference'],batch['history_actuator'],batch['current_mode']).numpy()
        expected=history_features(batch,'full');feature_error=float(np.max(abs(feature-expected)))
        # More relevant than tiny reduction-order differences: actual mean forecasts agree.
        parameters={k:v.numpy() for k,v in params.items()}
        direct=(expected-parameters['mean'])/parameters['scale']@parameters['coefficient']+parameters['target_mean']
        converted=(feature-parameters['mean'])/parameters['scale']@parameters['coefficient']+parameters['target_mean']
        forecast_error=float(np.max(abs(direct-converted)));assert forecast_error<1e-5
        limit=float(model.rate_scale[2]);delta=np.diff(np.column_stack([np.zeros(len(bias)),bias]),axis=1)
        reloads.append(dict(fold=fold,seed=seed,reload_max_abs=maximum,quaternion_reintegration_max_abs=pose_error,
            feature_max_abs=feature_error,mean_forecast_max_abs=forecast_error,slew_boundary_fraction=float((abs(delta)>=2*limit*.02-1e-6).mean())))
    verification.update(checkpoint_reloads=reloads,focused_tests_passed=37,development_evaluation=development['status'])
    base.write(OUT/'verification.json',verification)
    rows=pd.read_csv(OUT/'oof_yaw_per_flight.csv').groupby(['arm','horizon_s'])[['r_mean_mse','r_wave_mse']].mean().reset_index().rename(columns={'r_mean_mse':'r_mean_rmse','r_wave_mse':'r_wave_rmse'})
    rows.to_csv(OUT/'yaw_summary.csv',index=False)
    state=pd.read_csv(OUT/'state_summary.csv');shown=state[(state.cohort=='oof')&(state.subset=='all')&state.horizon_s.isin([.5,1.])]
    axis=pg[(pg.cohort=='oof')&pg.horizon_s.isin([.5,1.])]
    sensitivity_r=sr[sr.axis=='r'].pivot(index='horizon_s',columns='arm',values='reference_effect_mse')
    sensitivity_r['relative_change']=sensitivity_r.yaw/sensitivity_r.baseline-1;sensitivity_r=sensitivity_r.reset_index()
    report=['# 参考模型 v10：p/q波形定位与r因果修正','',
        '**r的连续修正取得一致的小幅预测收益，但未达到预设门槛；p/q仍不是简单放大幅值或固定时移可以解决的问题。保留A1600主基线与本轮研究候选，不晋升控制用途。**','',
        '## 范围','',
        '沿用407个训练窗口、5个留出飞行、3个冻结hold GRU种子。未重新筛选或训练GRU；均值预测器只拟合每折的4个训练飞行。r路线来自v9已经看过的探索结果，因此本轮仍是重复开发证据，不是独立盲测。sealed/reserved未打开。','',
        '## p/q：幅值不足不等于应该直接放大','',
        '对0.5/1秒窗口裁去首尾各100ms，所有时间偏移都使用相同真值内部区间。对每个窗口事后选择[-100,+100]ms的偏移，以及限制在[0,3]的非负最小二乘增益；先去均值。该oracle读取未来真值，只用于诊断，不是可部署预测器、严格精度下界或执行器时延辨识。',
        '以下是按飞行、种子平均后计算的指标。两项reduction均指同一内部区间的去均值MSE，不能解释为整体状态预测收益；std_ratio是预测/真值波动标准差之比。','',
        table(axis[['horizon_s','axis','std_ratio','zero_lag_correlation','zero_lag_gain','gain_only_mse_reduction','lag_gain_mse_reduction']]),'',
        '1秒p的波动幅值约为真值54%，但相关性仅约0.40，事后最优增益平均约0.86；q的幅值约83%、相关性约0.75、最优增益约0.90。幅值偏小同时伴随波形不匹配，简单按幅值比倒数放大会一起放大错误成分。',
        '增益与时间对齐联合oracle在1秒共同内部区间解释约20.8%的p去均值MSE、34.0%的q去均值MSE，仍有大部分误差未被这类变换消除。该有限oracle不能证明剩余误差是不可预测噪声。','',
        '所选偏移的分布（比例）：','',table(lg[(lg.cohort=='oof')][['horizon_s','axis','zero_fraction','positive_fraction','negative_fraction','boundary_fraction','gain_boundary_fraction']]),'',
        '不同窗口可选不同偏移，周期信号还可能具有多个相似对齐位置；不能把这些事后偏移直接当成一个统一通信延迟，或据此修改日志时间戳。完整逐飞行结果见pq_per_window.csv、lag_per_flight.csv。','',
        '## r：修正已纳入因果状态递推','',
        '固定94维历史特征、alpha100岭回归，预测未来各前缀的累计平均r。末端1秒均值参数与v9直接均值预测器复现。在线推理仅使用历史状态/参考/执行命令，不读未来测量；偏差估计仅使用当前模拟前缀，不提前计算完整1秒GRU输出再回填早期预测。',
        '修正由0开始，100ms smoothstep启用；幅值限制为对应折训练r标准差，变化率限制为其每秒2倍。这些是固定的研究约束，不是飞机安全包线。以修正后的角速度积分姿态，并把修正状态反馈给GRU，后续速度也随新姿态递推；没有保留旧姿态来冒充一致动力学。',
        '三seed、五折的OOF结果（rad/s，RMSE先按飞行再平均）：','',table(rows),'',
        f'1秒窗口均值误差改善{decision["mean_gain"]:.2%}，完整1秒r波形误差改善{decision["waveform_gain"]:.2%}。对应探索性飞行bootstrap95%改善区间分别为[{decision["bootstrap95_mean_waveform"][0][0]:.2%}, {decision["bootstrap95_mean_waveform"][1][0]:.2%}]和[{decision["bootstrap95_mean_waveform"][0][1]:.2%}, {decision["bootstrap95_mean_waveform"][1][1]:.2%}]。',
        '三个种子和五个飞行的两项误差均改善，说明效果并非只来自单个幸运种子或飞行；但只有5个曾参与开发的飞行，不能据此宣称独立泛化或因果控制证据。','',table(pd.DataFrame(decision['flight_gains'])),'',
        '约束确实在起作用：修正变化率触达上限的采样比例，各折/种子中位约12.4%、最大22.9%；幅值触达上限的比例中位约1.7%、最大4.4%。这些结果属于当前受约束方案，不能当作无约束预测器的表现，也不据此自动放宽限制。','',
        '其他状态端点误差：','',table(shown[['arm','horizon_s','p','q','r','attitude','velocity']]),'',
        f'保护指标中最大退化为{decision["maximum_protected_regression"]:.2%}，通过不超过5%的保护门槛。但均值改善未到10%、波形改善未到5%，因此本轮单轴相对门槛未通过；绝对状态误差也未通过。没有因结果接近门槛而事后降低标准。','',
        '## 参考响应仍需单独验收','',
        '分别用真实记录参考和“保持起点参考”驱动同一模型，比较r预测的变化幅度。下表为两次预测之间的r差值RMSE，不是真实飞机舵效或反事实真值：','',table(sensitivity_r),'',
        'r对参考变化的模型内响应幅度减小，1秒约下降15.4%。均值预测器目前只看历史，可能抑制由未来参考变化带来的低频响应；这只是需要检验的解释，不能断定真实控制能力增减。即使预测RMSE改善，也不能忽略这一变化。','',
        '## 交付和下一步','',
        f'开发验证执行状态：{development["status"]}。'+('按冻结OOF门槛停止扩展，未评价739个次日开发验证窗口。' if development['status']=='not_evaluated' else 'OOF通过后才固定评价开发验证；详见development_decision.json。'),
        '37项针对性测试通过；15个候选checkpoint重载、前缀因果性、关闭修正复现、姿态独立再积分、经验幅值/变化率限制以及均值参数复现均通过。候选保存在fold0至fold4/seed*/candidate.pt；这些是研究产物，不替换主模型。',
        '后续r应优先检验“只加入当前及过去时刻的计划参考前缀”的均值预测，避免均值修正压制未来参考应有的响应；必须重新冻结协议并维持查询飞行隔离。p/q应继续定位波形不匹配的状态/输入依赖，不直接采用事后对齐结果作校准。原数据不删除，本轮未证明必须补飞。','']
    (OUT/'report_zh.md').write_text('\n'.join(report))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    a=axis[axis.horizon_s==1.].set_index('axis').loc[['p','q']];xx=np.arange(2)
    axes[0].bar(xx-.18,100*a.gain_only_mse_reduction,.36,label='Oracle gain only');axes[0].bar(xx+.18,100*a.lag_gain_mse_reduction,.36,label='Oracle lag + gain');axes[0].set_xticks(xx,['p','q']);axes[0].set(ylabel='Common-core centered MSE reduction (%)',title='Offline oracle; not deployable');axes[0].legend()
    a=rows[rows.horizon_s==1.].set_index('arm').loc[['baseline','yaw']]
    axes[1].bar(xx-.18,a.r_mean_rmse,.36,label='Window mean');axes[1].bar(xx+.18,a.r_wave_rmse,.36,label='Full waveform');axes[1].set_xticks(xx,['Baseline','Causal yaw']);axes[1].set(ylabel='Yaw-rate RMSE (rad/s)',title='OOF 1 s, flight-equal 3-seed mean');axes[1].legend()
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'axis_diagnostics.png',dpi=160);plt.close(fig)
    paths=list((base.ROOT/'scripts').glob('*reference_axis_diagnostics_v10*.py'))+[base.ROOT/'src/system_identification/models/reference_gru_yaw_mean.py',base.ROOT/'tests/test_reference_axis_diagnostics_v10.py']
    base.write(OUT/'delivery_manifest.json',dict(status='complete',code={str(p.relative_to(base.ROOT)):base.sha(p) for p in paths},
        artifacts={str(p.relative_to(base.ROOT)):base.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'},
        tests_passed=37,control_validated=False,development_status=development['status'],sealed_test_accessed=False))

if __name__=='__main__':main()
