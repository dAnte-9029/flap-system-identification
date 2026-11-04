"""Summarize fixed outputs; no fitting or parameter selection."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(sys.argv[1]);c=pd.read_csv(p/'coverage.csv');pred=pd.read_csv(p/'cycle_prediction.csv');m=pd.read_csv(p/'motion_preservation.csv');r=pd.read_csv(p/'causal_runtime.csv')
common=pred[pred.common_observations==pred.n_observations].copy()
f=common.groupby(['split','flight_id','date','channel'])[['trend_mse','A_mse','B_mse','B_trend_mse','A_fast_rms','B_fast_rms','raw_std']].mean().reset_index()
f['A_relative_MSE_improvement']=1-f.A_mse/f.trend_mse;f['B_relative_MSE_improvement']=1-f.B_mse/f.B_trend_mse;f['B_vs_A_improvement']=1-f.B_mse/f.A_mse
summary=f.groupby(['split','channel']).mean(numeric_only=True);summary.to_csv(p/'flight_equal_summary.csv');f.groupby(['split','date','channel']).mean(numeric_only=True).to_csv(p/'date_summary.csv')
text=['# 快慢分离固定方案可行性验证','', '本报告为观测分离研究，不是物理真值或控制因果验证。仅 train/validation，无模型训练。', '', '## 方法与边界', '', 'A 为三阶谐波加局部线性趋势的整拍等权最小二乘。B 为每拍时间加权均值趋势补偿加三阶跨拍 LSF；51拍、Q=8半格网格固定。半格网格避免强制查询翼拍边界；缺失单元不填补，三阶拟合至少8拍且两侧各2拍。趋势补偿和零均值为项目适配，不属于 Abboud 原始方法。', '', '速度、角速度使用各自目标原生时间戳及相位；按翼拍内相邻观测插值。每拍均值与代表时间使用同一中点分区权重。每5拍留出1拍，所有拟合均排除其运动观测；评价使用A/B共同有效范围。', '', 'B零均值仅针对指定离散时间权重，不宣称连续积分精确守恒，也不保证SO(3)净旋转守恒。因果模板以最近已完整到达的51拍拟合；当前相位以截至决策时刻已到达Hall记录作局部速率外推。20ms保护不是实测延迟上界。', '', '## 覆盖', '', c.groupby(['split','sampling_group','channel'])[['valid_cycles','windows_51']].sum().to_string(), '', '窗口高度重叠；统计独立单位是航次。覆盖表每行是一航次的一通道，不可将六通道翼拍数直接相加。', '', '## 整拍留出预测：航次等权', '', summary.to_string(), '', '相对改善先在航次内计算，再等权平均。MSE单位分别为(m/s)^2和(rad/s)^2。仅证明跨拍观测预测能力，不证明Slow真值、未来反事实或闭环控制收益。', '', '## 非周期保留', '', m[m.kind.isin(['linear','smooth_step'])].groupby('kind')[['A_fast_leak_rms','B_fast_leak_rms','B_weighted_fast_mean','reconstruction_max']].agg(['mean','max']).to_string(), '', '合成变化仅加到vN：0.5 m/s²线性趋势；0.5s内0.3m/s平滑阶跃。真实事件阈值为train非零控制差分P90，冻结应用validation。事件是闭环相关性。', '', '## 因果差异与运行时间', '',r.groupby(['split','channel'])[['fast_slow_difference_rmse','teacher_fast_rms','history_seconds','template_update_ms','observation_median_ms','observation_p95_ms']].mean().to_string(), '', 'Slow误差与Fast误差大小相同，因为两者从同一原始观测相减。初始化计时只包括已有51拍摘要的模板初始化，不包括历史采集和全部数据预处理。单次更新计时为本机Python CPU，不是飞控部署保证。', '', '## 尚未完成的验证与结论限制', '', '本实现尚未完成因果事件滞后与初始化完整数据准备耗时。事件半变化到达时间仅为描述量，不是系统辨识时延。vBx沿用冻结数据集的FRD到NED姿态R，以R转置乘NED速度；姿态与速度观测的时间差仍是该辅助诊断限制。合成阶跃当前仅评价目标拍，不能代表整个阶跃区间的最大泄漏。因果当前趋势外推已计算但尚无独立趋势误差指标。故不得据此宣布任务全部完成或建议正式标签生产。', '', '总体结论：**证据不足**。该分类包含验证未完成的限制，不等同于方法已被否定。', '', '## 参考', '', '- Abboud et al.: https://doi.org/10.1016/j.ymssp.2021.108600', '- Qian et al.: https://arxiv.org/abs/2504.20809', '- SparseTSF: https://github.com/lss-1138/SparseTSF', '', '本次未实现SparseTSF，也不从其参数量推断本任务收益。']
(p/'report_zh.md').write_text('\n'.join(text))
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,split in zip(axes,['train','validation']):
    ss=summary.loc[split];x=np.arange(len(ss));ax.bar(x-.16,ss.A_relative_MSE_improvement,.32,label='A');ax.bar(x+.16,ss.B_relative_MSE_improvement,.32,label='B');ax.set_xticks(x,ss.index);ax.set_title(split+' flight-equal relative MSE gain');ax.legend();ax.axhline(0,color='k',lw=.5)
fig.tight_layout();fig.savefig(p/'prediction_comparison.png',dpi=140)
print(summary.to_string())

fig,axes=plt.subplots(2,1,figsize=(10,6))
for ax,ch in zip(axes,['vD','q']):
    z=pd.read_csv(p/f'curve_validation_{ch}.csv');t=z.time_s-z.time_s.iloc[0]
    for col in ['raw','slow','fast']:ax.plot(t,z[col],label=col)
    ax.set_title(ch);ax.legend();ax.set_xlabel('time (s)')
fig.tight_layout();fig.savefig(p/'representative_curves.png',dpi=140)

# Explicit paired scope and observational effect sizes, independent of per-point counts.
with (p/'report_zh.md').open('a') as handle:
    handle.write('\n\n## 共同评价范围与运行统计\n\n')
    handle.write('候选留出通道翼拍记录：%d；A/B共同有效记录：%d。\n\n' % (len(pred),len(common)))
    handle.write('相位单元缺失比例（有效航次通道中位数）：%.4f。\n\n' % c.loc[c.valid_cycles>0,'phase_cell_missing_fraction'].median())
    handle.write('```text\n'+r[['initialization_ms','template_update_ms','observation_median_ms','observation_p95_ms']].quantile([.5,.95]).to_string()+'\n```\n')
    handle.write('单观测汇总列是拍内中位数/P95，再跨拍汇总；不是所有观测池化的精确P95。\n')
    handle.write('\n事件记录按通道重复，不能当作独立控制试验数。\n')
    event=m[m.kind=='control_event']
    handle.write('```text\n'+event.groupby(['split','channel'])[['raw_change','slow_change','fast_change','raw_half_change_delay_s','slow_half_change_delay_s']].mean().to_string()+'\n```\n')
    handle.write('\n## 五个问题的当前判断\n\n1. 周期可预测性：vD、p、q有明显支持；水平速度B存在条件收益，但其总误差优势主要含局部趋势改进。r收益弱于p/q且依赖方法。\n2. B改善：速度和r总误差下降，不能全部归因于LSF周期模板；本轮没有额外消融，无法唯一归因。q不存在跨split一致的胜者。\n3. 非周期保留：线性扰动检验通过；局部阶跃和真实闭环事件证据仍有限，不能证明真实Slow。\n4. 因果差距：计算开销小，但约51拍启动支持和通道相关信息差距存在；不能仅靠代码优化解决。\n5. 下一阶段训练：证据不足，暂不推荐将本轮分离结果作为正式监督标签或直接进入GRU配对训练。\n')

report=p/'report_zh.md'
report.write_text('\n'.join(line.rstrip() for line in report.read_text().splitlines())+'\n')
