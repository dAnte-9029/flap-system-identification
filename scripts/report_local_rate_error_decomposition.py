import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/flap-paper-mpl')
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_local_rate_error_decomposition import OUT,ROOT,m

def main():
 a=pd.read_csv(OUT/'summary.csv');z=a.query('pattern=="sustained" and offset==-1 and cohort=="ALL"')
 cols=['mode','axis','rmse_mean','bias_mean','bias_share_pct','amplitude_share_pct','shape_share_pct','centered_amplitude_ratio','correlation_mean']
 table='| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'
 for row in z[cols].itertuples(index=False,name=None):table+='| '+' | '.join(f'{x:.4f}' if isinstance(x,float) else str(x) for x in row)+' |\n'
 fig,axs=plt.subplots(1,2,figsize=(10,4))
 for ax,mode in zip(axs,['Stabilized','Mission']):
  g=z[z['mode']==mode].set_index('axis').loc[list('pqr')];bottom=np.zeros(3)
  for term in ['bias','amplitude','shape']:
   v=g[term+'_share_pct'].to_numpy();ax.bar(list('pqr'),v,bottom=bottom,label=term);bottom+=v
  ax.set(title=mode,ylabel='Share of equal-flight mean MSE [%]');ax.legend()
 fig.suptitle('Existing 100ms forecasts / sustained joint commands / no lag or filter adjustment');fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(OUT/f'decomposition.{ext}',dpi=170)
 plt.close(fig)
 result=subprocess.run(['/home/zn/anaconda3/envs/flap-train-gpu/bin/python','-m','pytest','-q','tests/test_local_rate_error_decomposition.py'],cwd=ROOT,capture_output=True,text=True);assert result.returncode==0
 m.write_json(OUT/'tests.json',dict(exit_code=result.returncode,stdout=result.stdout))
 text='''# 100ms局部角速度预测误差分解

## 结论
Stabilized与Mission的持续联合命令组给出一致图景：**p主要缺少快速变化幅度和波形匹配；r主要是每个短段内平均水平偏离；q的中心化波形相关性较高。** 这些描述并未定位网络/测量链的唯一物理根因，不能用一个固定offset或简单加大控制增益解决。

本轮无训练、无新推理、无原始日志或Sep8/Sep19访问。复用原三个seed的local100ms预测、原始标签、native dt与上一轮固定组别，Stabilized462origin/17flight、Mission597origin/12flight。每个parent有10个独立重初始化的5步小段，始终按parent/flight聚合，不把十倍小段视为独立样本。

## 精确定义
每个100ms预测段的第1–5未来状态（排除初始共同零误差），使用实际dt归一化权重。μ表示加权均值、σ表示加权时间标准差：

MSE = (μ预测−μ真实)² + (σ预测−σ真实)² + 2(σ预测σ真实−Cov(预测,真实))。

三项分别为短段均值误差、中心化变化幅度误差、中心化波形误差。恒等式精确成立。没有搜索时移，没有滤波或调整幅值；波形项不能直接等同物理相位延迟或噪声。即使瞬时预测初值正确，之后5个状态的平均值仍可偏离。

all_offsets先在parent内等权平均十个小段MSE，再flight内平均后开方，再flight等权、再三seedmean/sample SD。份额用等权flight MSE的比值，不用宏平均RMSE平方作分母。各offset及Sep7/Sep17结果全部保留CSV。

centered_amplitude_ratio为sqrt(宏平均预测时间方差/宏平均真实时间方差)，不是上一轮相对t0总变化幅度比。correlation是5点加权去均值相关性，常量波形不可定义并计数；5点很短，不能从中估计可靠延迟。相关性均值与能量加权误差份额也不是相同统计量。

## 持续组结果：全部局部offset
RMSE及bias均值单位rad/s；bias_share等为MSE百分比；mean是三seed均值，std见summary.csv。

'''+table+'''

## 解释
- p：Stabilized/Mission的段内均值项约23%/20%，幅度项约34%/35%，波形项约43%/45%。中心化幅度比约0.53/0.55，相关性约0.24/0.30。表明100ms内实测快速变化没有被充分复现；长期积分漂移不是全部解释。
- q：中心化幅度比约0.88/0.89、相关性约0.92/0.94；不能据此说无误差，短段均值项仍约48%–53%。
- r：段内均值项约66%；中心化幅度比约0.61。这里的“均值偏差”是每段内的平均误差，不是整批恒定零偏。三seed宏平均有符号bias只有约0.0097/0.0032rad/s，不同段可相互抵消。不能直接加全局offset修正。

此前共同测量时刻/带宽分析已显示p对快速波形处理敏感，但这不能证明当前波形项是测量噪声。本次也不能判定存在一个固定相位滞后。原始数据与预测没有被改动，所有分解项都不是模型性能提升。

## 下一步建议（未执行）
不先调rudder、控制器或actuator tau，也不只靠周期重置掩盖p误差。若继续定位，应把局部p残差与已记录扑翼相位、频率及原生采样间隔关联，检查是否为可重复的相位相关遗漏或采样链效应；保留原始评分，不拟合一个“最好看”的时移。r应检查局部平均角加速度/状态转移偏差与初态/联合命令的关系，而不是先加入固定偏置。任何后续模型修改仍需单独冻结实验，不在本轮自动实施。

## 工程检查
四项测试：native-dt恒等分解、纯常量偏差、纯幅度缩放、常量相关性不可定义。所有原始预测/标签/分组哈希与上轮一致；各offset端点RMSE逐flight与上轮复核通过，误差恒等式残差及配对单元数见checks.json。未commit/push。
'''
 (OUT/'report.md').write_text(text);(OUT/'review_report.md').write_text(text)
 m.write_json(OUT/'completion.json',dict(status='complete',tests_passed=4,training=0,new_inference=0,heldout_accessed=False,outputs={m.rel(p):m.file_hash(p) for p in OUT.iterdir() if p.is_file() and p.name!='completion.json'}))
if __name__=='__main__':main()
