import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/flap-paper-mpl')
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_joint_command_local_reset import OUT,ROOT,m

def md(f):
 return '| '+' | '.join(f.columns)+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+'\n'.join('| '+' | '.join(f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |' for row in f.itertuples(index=False,name=None))
def main():
 a=pd.read_csv(OUT/'summary.csv');pf=pd.read_csv(OUT/'paired_flights.csv');ps=pd.read_csv(OUT/'paired_seeds.csv')
 piv=a.pivot(index=['mode','pattern','step','cohort','metric'],columns='condition',values='mean').reset_index()
 piv['reduction_pct']=100*(piv.continuous-piv.local100ms)/piv.continuous.replace(0,np.nan);piv.to_csv(OUT/'comparison.csv',index=False)
 directions=[]
 for frame,unit in [(ps,'seed'),(pf,'flight_three_seed_mean')]:
  if unit!='seed':frame=pd.concat([frame,frame.assign(cohort='ALL')])
  for key,g in frame.groupby(['mode','pattern','step','cohort','metric']):directions.append(dict(zip(['mode','pattern','step','cohort','metric'],key),unit=unit,n=len(g),local_better=int((g.local_minus_continuous<0).sum()),local_worse=int((g.local_minus_continuous>0).sum())))
 dirs=pd.DataFrame(directions);dirs.to_csv(OUT/'direction_counts.csv',index=False)
 for mode in ['Stabilized','Mission']:
  fig,axs=plt.subplots(1,3,figsize=(12,4))
  for ax,metric in zip(axs,['velocity_m_s','attitude_deg','body_rate_rad_s']):
   for condition in ['continuous','local100ms']:
    z=a.query('mode==@mode and metric==@metric and pattern=="sustained" and cohort=="ALL" and condition==@condition').sort_values('step')
    ax.errorbar(z.step,z['mean'],yerr=z['std'],marker='o',label=condition)
   ax.set(xlabel='Common endpoint native step',ylabel=metric);ax.legend();ax.grid(alpha=.2)
  fig.suptitle(mode+' / sustained joint commands; local receives newer observations');fig.tight_layout()
  for ext in ['png','pdf']:fig.savefig(OUT/f'comparison_{mode}.{ext}',dpi=160)
  plt.close(fig)
 res=subprocess.run(['/home/zn/anaconda3/envs/flap-train-gpu/bin/python','-m','pytest','-q','tests/test_joint_command_local_reset.py'],cwd=ROOT,capture_output=True,text=True);assert res.returncode==0
 m.write_json(OUT/'tests.json',dict(exit_code=res.returncode,stdout=res.stdout))
 key=piv.query('pattern=="sustained" and cohort=="ALL" and step in [25,50]')
 primary=key[key.metric.isin(['velocity_m_s','attitude_deg','body_rate_rad_s'])]
 rates=key[key.metric.isin(['p_rad_s','q_rad_s','r_rad_s'])]
 direction=dirs.query('pattern=="sustained" and cohort=="ALL" and step in [25,50] and metric in ["velocity_m_s","attitude_deg","body_rate_rad_s"]')
 text='''# 持续联合命令：连续递推与短时实测重初始化

本轮复用冻结三个seed的全部原始预测和既有local100ms预测，不训练、不新推理、不访问Sep8/Sep19。沿用Stabilized/Mission及上一轮持续同向命令规则，不按误差筛选：主组分别462origin/17flight和597origin/12flight；两模式全部窗口的结果也保留。

## 比较含义
continuous从原t0用真实历史/初始状态出发，之后自主递推。local100ms在各offset0/5/…/45用该时刻真实状态和真实H26历史重新编码，再自主预测5步。共同终点为5/10/…/50，同一实测标签、同一段实际四通道命令与native dt。

500ms终点的local大约从400ms重启；1s终点大约从900ms重启。它获得额外观测且预测年龄更短，**不是公平的主模型排行榜，也不是只替换一个状态量的因果实验**。重初始化同时更新观测状态、历史隐表示及相位/频率参考，不能将误差下降比例解释为“纯积分漂移占比”。每个5步预测内部无teacher forcing，多个local段不拼接成一条自主轨迹。

## 持续组主结果
物理单位分别m/s、deg、rad/s。先flight内RMSE，再flight等权，再seed均值；完整sample SD见summary.csv，正reduction_pct表示局部预测评分较低。

'''+md(primary)+'''

## p/q/r各轴
'''+md(rates)+'''

## seed与flight方向
flight方向先对每架flight的三seed误差平均，再比较，不能解释为所有window都改善。

'''+md(direction)+'''

## 实际发现与下一步
两模式持续组在500ms同终点，速度误差下降59.26%（Mission）/64.14%（Stabilized），姿态下降66.06%/69.46%，body-rate仅下降7.07%/6.76%。三seed均同方向。速度和姿态在12/12 Mission及17/17 Stabilized flights（三seed平均）均下降；body-rate分别11/12与15/17，保留不改善的flight。

到1s终点，速度和姿态下降约80%，角速度下降10.17%/14.36%。p仍是主要局部残差：500ms同终点的local p RMSE为0.4512/0.4999rad/s，较连续仅低5.81%/4.54%；1s也仍有0.4706/0.4650rad/s。因此当前证据不支持“只修长时积分漂移就能解决全部操纵响应”。

下一步优先定位既有100ms预测的角速度误差，尤其p的平均偏差与波形/幅度误差，并与此前测量带宽敏感性结果一起解释；不直接调rudder、actuator tau或控制器。速度/姿态较大的恢复与近期观测/较短递推有益一致，但本轮尚未分离哪个内部状态造成累积偏差。继续沿用联合命令及两种模式，不回到孤立rudder筛选。

## 解释限制
速度/姿态若随着重初始化显著恢复，说明引入近期真实观测和缩短预测长度有价值，与自由递推累积偏差相容；不能由此确定是积分器、GRU隐藏状态或控制映射哪一个根因。角速度若局部预测仍有较大残差，则仅靠周期重置并未解决其快速/局部波形问题，也不能自动归结为传感器噪声。

局部误差是100ms预测条件下的误差，不是不可消除噪声下限。全部数据是已开放validation，任何新设计都属于开发，不能称独立确认。没有修改当前H26/K50主模型，不开展闭环试验。

## 核验
三个测试覆盖终点索引、配对RMSE、不配对数组拒绝；逐offset核验全部local父origin/标签/controls/dt精确一致，offset0与原连续预测前5步数值一致，所有预测有限，输入hash前后不变。图横轴为原生step，实际endpoint和local年龄中位数保存在per_flight.csv，不假设严格50Hz。
'''
 (OUT/'report.md').write_text(text);(OUT/'review_report.md').write_text(text)
 m.verify_pins(m.read_json(OUT/'protocol.json')['pins'])
 m.write_json(OUT/'completion.json',dict(status='complete',tests_passed=3,training=0,new_predictions=0,heldout_accessed=False,outputs={m.rel(f):m.file_hash(f) for f in OUT.iterdir() if f.is_file() and f.name!='completion.json'}))
 print(primary.to_string(index=False));print(rates.to_string(index=False))
if __name__=='__main__':main()
