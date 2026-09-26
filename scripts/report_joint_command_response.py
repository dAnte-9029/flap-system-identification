import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/flap-paper-mpl')
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_joint_command_response import OUT,ROOT,m

def md(f):
 return '| '+' | '.join(f.columns)+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+'\n'.join('| '+' | '.join(f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |' for row in f.itertuples(index=False,name=None))

def main():
 a=pd.read_csv(OUT/'summary.csv');ev=pd.read_csv(OUT/'error_evolution.csv');g=pd.read_csv(OUT/'origin_groups.csv');c=pd.read_csv(OUT/'coverage_per_flight.csv');amp=pd.read_csv(OUT/'amplitude_summary.csv');trace=pd.read_csv(OUT/'representative_traces.csv')
 coverage=c.groupby(['mode','pattern']).agg(n_origins=('n_origins','sum'),n_flights=('flight_id','nunique')).reset_index();coverage.to_csv(OUT/'coverage_summary.csv',index=False)
 tests=subprocess.run(['/home/zn/anaconda3/envs/flap-train-gpu/bin/python','-m','pytest','-q','tests/test_joint_command_response.py'],cwd=ROOT,capture_output=True,text=True);assert tests.returncode==0
 m.write_json(OUT/'tests.json',dict(exit_code=tests.returncode,stdout=tests.stdout,stderr=tests.stderr))
 for mode in ['Stabilized','Mission']:
  fig,axs=plt.subplots(1,3,figsize=(13,4))
  for ax,metric in zip(axs,['velocity_m_s','attitude_deg','body_rate_rad_s']):
   for pattern in ['ALL','sustained','reversing','mixed_change']:
    z=ev[(ev['mode']==mode)&(ev.pattern==pattern)&(ev.metric==metric)].sort_values('step')
    if not len(z):continue
    ax.plot(z.elapsed_median_s,z['mean'],label=pattern);ax.fill_between(z.elapsed_median_s,z['mean']-z['std'],z['mean']+z['std'],alpha=.08)
   ax.set(xlabel='Median native elapsed time [s]',ylabel=metric);ax.grid(alpha=.2)
  axs[0].legend(fontsize=7);fig.suptitle(mode+' / actual joint commands, 3-seed mean +/- SD');fig.tight_layout()
  for ext in ['png','pdf']:fig.savefig(OUT/f'error_evolution_{mode}.{ext}',dpi=160)
  plt.close(fig)
 for (mode,pattern),z in trace.groupby(['mode','pattern']):
  z=z.sort_values('step');fig,axs=plt.subplots(4,1,figsize=(11,11),sharex=True)
  for ch in ['drive','common','differential','rudder']:axs[0].step(z.elapsed_s.iloc[:50],z[ch].iloc[:50],where='post',label=ch)
  for axis in ['vx','vy','vz']:
   line,=axs[1].plot(z.elapsed_s,z[axis+'_true'],label=axis+' true');axs[1].plot(z.elapsed_s,z[axis+'_pred'],'--',color=line.get_color(),label=axis+' pred')
  for axis in 'pqr':
   line,=axs[2].plot(z.elapsed_s,z[axis+'_true'],label=axis+' true');axs[2].plot(z.elapsed_s,z[axis+'_pred'],'--',color=line.get_color(),label=axis+' pred')
  axs[3].plot(z.elapsed_s,z.roll_true,label='roll true');axs[3].plot(z.elapsed_s,z.roll_pred,'--',label='roll pred')
  for ax,label in zip(axs,['Normalized commands','Velocity NED [m/s]','Body rate [rad/s]','Roll [deg]']):ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=3)
  axs[-1].set_xlabel('Native elapsed time [s]');fig.suptitle(mode+' / '+pattern+' / fixed median identity / seed17');fig.tight_layout()
  for ext in ['png','pdf']:fig.savefig(OUT/f'case_{mode}_{pattern}.{ext}',dpi=160)
  plt.close(fig)
 def table(h):
  z=a[(a['mode'].isin(['Stabilized','Mission','ALL_original']))&(a.pattern=='ALL')&(a.cohort=='ALL')&(a.horizon==h)&a.metric.isin(['position_m','velocity_m_s','attitude_deg','body_rate_rad_s'])].copy()
  z['mean ± SD']=z.apply(lambda r:f'{r["mean"]:.4f} ± {r["std"]:.4f}',axis=1)
  return md(z.pivot(index='mode',columns='metric',values='mean ± SD').reset_index())
 p=a[(a['mode'].isin(['Stabilized','Mission']))&(a.pattern.isin(['sustained','reversing','mixed_change','lower_change']))&(a.horizon=='500ms')&(a.cohort=='ALL')&a.metric.isin(['velocity_m_s','attitude_deg','body_rate_rad_s'])]
 p=p.pivot(index=['mode','pattern'],columns='metric',values='mean').reset_index();p.to_csv(OUT/'paper_pattern_500ms.csv',index=False)
 ratios=amp[(amp['mode'].isin(['Stabilized','Mission']))&(amp.pattern=='ALL')][['mode','interval','axis','true_amplitude','pred_amplitude','ratio']]
 text='''# 实际联合执行器命令下的模型响应：Stabilized与Mission

## 目标与结论边界
本轮使用模型真正接收的四通道最终归一化命令，比较冻结Standard GRU64/H26/K50在实际命令回放下与实飞的差异。遥控杆、任务目标和控制器转矩不是本模型四维输入，不用于筛选；Stabilized与Auto Mission全部纳入，互不冒充。没有训练、推理、修改模型/原数据，也没有访问Sep8/Sep19。

原验证2582个origin、17flight、三个seed17/23/42全部复用。完整51状态模式15为Stabilized、3为Mission；其余保留Other_or_transition，ALL_original仍覆盖全部原点。Mission包括任务中的正常转弯/姿态修正，不等于直线定常飞行。

## 控制语义与固定分组
实际输入是motor、left elevon、right elevon、rudder的postallocation软件命令；不是实测舵角、PWM或实际扑频。独立坐标common=(left+right)/2、differential=(left-right)/2只用于分析，递推始终使用原四通道。

用前25命令相对u0的RMS变化和原train尺度/q75确定各通道较大变化。lower_change无高活动通道；sustained要求全部高活动通道在后25命令至少80%仍同向，且同向均值≥前段均值幅度的一半；reversing至少一个高活动通道后段反向且满足同样幅度/比例条件；其他为mixed_change。其他控制通道不限，multi_high单独报告多个通道协同变化，组之间不作为独立随机干预比较。

这些标签在看本轮误差前固定，只使用实际命令，未按模型表现调整。“持续”是相对origin的同向变化，不保证恒定舵量；“反向”不保证回到trim；origin不一定是动作发生时刻。不能将误差曲线直接解释为阶跃传递函数或准确执行器延迟。

## 覆盖
'''+md(coverage)+'''

## 500ms端点：三seed均值±sample SD
'''+table('500ms')+'''

## 1s端点：三seed均值±sample SD
'''+table('1s')+'''

## 命令分组500ms
下表仅为mean，完整seed SD、Sep7/Sep17、各flight、四horizon、两个区间保留CSV。不同组初态/工况不同，不以组间误差大小直接推断输入因果效应。

'''+md(p)+'''

## 响应幅度是否随递推弱化
原始未滤波的p/q/r相对真实t0变化，分别在早段1..5、后段16..25、41..50按native dt计算RMS。先flight聚合，再seed平均；ratio=预测幅度宏平均/真实幅度宏平均。幅度相近不保证波形相位、符号和因果响应相近；相对t0的变化也不等于控制输入引起的净变化。

'''+md(ratios)+'''

## 完整过程与案例
error_evolution.csv保留1–50每一步误差、native累计dt中位数与5/95分位，图横轴仅代表中位时刻，未将所有origin伪装成相同时间。误差先flight内RMSE、再flight等权、再三seedmean/sample SD；三seedSD不是预测置信区间。位置/速度/角速度为向量RMSE，姿态使用四元数测地RMS；wrapped Euler roll仅作辅助。

每种模式的持续/反向/混合组均按window_id排序取中位identity，固定seed17。图同时展示实际命令、速度、p/q/r、roll真实与预测。所有6个案例保留，未按模型好坏选图；代表案例不替代总体统计。原始同步/滤波定义不改，本轮不使用上轮平滑后的较好分数。

## 工程检查
3项测试检查联合模式允许多通道变化、flight等权而非pooled、四元数符号不改变roll。90个ALL_original统计单元与旧评分复核一致；全部origin顺序、native dt、预测有限性、四元数单位范数与输入hash通过。首次尝试缺Path导入，在protocol/预测分析前停止，修复记录见engineering_notes.md，研究选择没有变化。

## 主要发现与判断
1. 原2582origin中，Stabilized1445个/17flight，Mission1125个/12flight，其他12个完整保留。持续同向组462/597个；不再要求其他通道安静后，此类联合输入并不稀缺。因此没有依据优先要求专门rudder激励。
2. 500ms时Stabilized velocity/attitude/rate为0.4341m/s、4.5641°、0.6188rad/s；Mission为0.2966、3.8582°、0.5546。1s分别增长到0.7474、7.1186°、0.6469，以及0.5836、6.6299°、0.5957。两模式的初态、动作、flight组成不同，不能将差距归因于模式本身。
3. 两种模式在约300–500ms段，p变化幅度比约0.84–0.85、q约0.93–0.95、r约0.87–0.88；约800–1000ms段p约0.83、q约0.95、r约0.79–0.81。预测对p/r变化的RMS偏弱，q更接近；这是幅度描述，不是控制增益辨识或波形全部匹配。
4. 持续同向组也有相同现象：Stabilized/Mission的r幅度比由0.884/0.877变为0.821/0.804，p由0.851/0.859变为0.837/0.846，q由0.942/0.943变为0.947/0.954。并不是所有轴都随时间衰减。r的早段比值还低于500ms，因此不能把局部后段下降写成全程单调失效。
5. 这些结果将问题定位为“已记录联合输入下，部分运动变化幅度不足，速度/姿态随递推累积偏差”。尚不能区分全部误差来自观测带宽、状态递推还是输入利用机制。此前p快速误差的带宽敏感性仍是背景证据，不能直接从当前RMS中扣除一个噪声比例。

## 下一步边界
先根据两模式完整误差过程定位偏差，不因某mode更差就删除、不自动训练。若偏差集中于滚转快速成分，应继续区分观测带宽和预测波形；若同向命令阶段速度/姿态误差持续扩大，则检查状态自由递推与联合输入建模。仅当正常联合操纵的关键工况缺少覆盖时才考虑补充roll参考操纵，而不是优先做孤立rudder激励。

本轮只回答“日志中的实际联合命令回放是否贴近实飞”。闭环命令由状态反馈产生，不能证明任意改变命令后的反事实轨迹、控制器优化收益或长期仿真有效。该验证集已经用于开发，不是新的独立测试。
'''
 (OUT/'report.md').write_text(text);(OUT/'review_report.md').write_text(text)
 m.verify_pins(m.read_json(OUT/'protocol.json')['pins'])
 m.write_json(OUT/'completion.json',dict(status='complete',tests_passed=3,heldout_accessed=False,training=0,new_inference=0,outputs={m.rel(f):m.file_hash(f) for f in OUT.iterdir() if f.is_file() and f.name!='completion.json'}))
 print(coverage.to_string(index=False));print(table('500ms'));print(table('1s'))

if __name__=='__main__':main()
