"""Finalize the observation scoring sensitivity, preserving original scores."""
import os
os.environ.setdefault("MPLCONFIGDIR","/tmp/flap-paper-mpl")
import json
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_observation_scoring_sensitivity import OUT,ART,ROOT,m

def md(f):
    return '| '+' | '.join(f.columns)+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+'\n'.join('| '+' | '.join(f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |' for row in f.itertuples(index=False,name=None))

def main():
    agg=pd.read_csv(OUT/'summary.csv');per=pd.read_csv(OUT/'per_seed.csv');fl=pd.read_csv(OUT/'per_flight.csv')
    tests=subprocess.run(['/home/zn/anaconda3/envs/flap-train-gpu/bin/python','-m','pytest','-q','tests/test_observation_scoring_sensitivity.py','tests/test_angular_rate_measurement_audit.py'],cwd=ROOT,capture_output=True,text=True)
    m.write_json(OUT/'tests.json',dict(command=tests.args,exit_code=tests.returncode,stdout=tests.stdout,stderr=tests.stderr));assert tests.returncode==0
    # Validate original interval scores against previous diagnostic, not heldout files.
    old=pd.read_csv(ROOT/'docs/analysis/results/response_timescale_diagnostic_v1/component_per_seed.csv')
    old=old.query('tau_s==0.1 and component=="raw"').copy();old['metric']=old.axis+'_rad_s';old=old.rename(columns={'interval':'horizon'})
    keys=['metric','horizon','group','cohort','seed']
    cmp=per.query('condition=="original"').merge(old,on=keys)
    assert len(cmp)==162
    delta=float(np.max(np.abs(cmp.value-cmp.rmse)));assert delta<1e-7
    checks=m.read_json(OUT/'sanity_checks.json');checks.update(previous_original_interval_parity_cells=len(cmp),previous_original_interval_max_abs=delta)
    m.write_json(OUT/'sanity_checks.json',checks)
    direction=[]
    for frame,unit in [(per,'seed'),(fl.groupby(['condition','metric','horizon','kind','group','flight','cohort']).value.mean().reset_index(),'flight_three_seed_mean')]:
        # flight results include separate dates and equal-flight ALL direction counts.
        if unit!='seed':frame=pd.concat([frame,frame.assign(cohort='ALL')])
        for k,g in frame.groupby(['metric','horizon','group','cohort']):
            idcol='seed' if unit=='seed' else 'flight'
            p=g.pivot(index=idcol,columns='condition',values='value')
            for condition in ['synchronized','synchronized_lowpass']:
                if condition not in p:continue
                d=p[condition]-p.original
                direction.append(dict(metric=k[0],horizon=k[1],group=k[2],cohort=k[3],condition=condition,unit=unit,n=len(d),lower=int((d<0).sum()),higher=int((d>0).sum()),equal=int((d==0).sum())))
    directions=pd.DataFrame(direction);directions.to_csv(OUT/'scoring_direction_counts.csv',index=False)
    labels={'original':'Original','synchronized':'Sample-time synchronized','synchronized_lowpass':'Synchronized + fixed lowpass'}
    for group in ['ALL','high']:
        fig,axs=plt.subplots(1,3,figsize=(12,3.6))
        for ax,metric in zip(axs,['p_rad_s','q_rad_s','r_rad_s']):
            for condition,label in labels.items():
                q=agg.query('cohort=="ALL" and group==@group and kind=="endpoint" and metric==@metric and condition==@condition').copy()
                q['time']=q.horizon.map({'100ms':.1,'200ms':.2,'500ms':.5,'1s':1});q=q.sort_values('time')
                ax.errorbar(q.time,q['mean'],yerr=q['std'],marker='o',label=label)
            ax.set(title=metric[0],xlabel='Nominal forecast horizon [s]',ylabel='Axis RMSE [rad/s]');ax.grid(alpha=.2)
        axs[0].legend(fontsize=7);fig.suptitle(group+' / same frozen predictions; scoring sensitivity only');fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(OUT/f'endpoint_sensitivity_{group}.{ext}',dpi=170)
        plt.close(fig)
    def table(horizon,group):
        f=agg.query('cohort=="ALL" and group==@group and horizon==@horizon').copy()
        f['mean ± seed SD']=f.apply(lambda r:f'{r["mean"]:.4f} ± {r["std"]:.4f}',axis=1)
        return md(f.pivot(index='condition',columns='metric',values='mean ± seed SD').fillna('not filtered').reset_index())
    contrasts=[]
    for (metric,horizon,group,cohort),g in agg.groupby(['metric','horizon','group','cohort']):
        values=g.set_index('condition')['mean'];base=values['original']
        for c,v in values.items():
            contrasts.append(dict(metric=metric,horizon=horizon,group=group,cohort=cohort,condition=c,original=base,value=v,scoring_change=v-base,scoring_reduction_pct=100*(base-v)/base if base else np.nan))
    pd.DataFrame(contrasts).to_csv(OUT/'scoring_changes.csv',index=False)
    report='''# 固定预测的观测链评分敏感性

## 问题与设置
本轮只问：同一批冻结预测，评分是否对测量时间基准和带宽处理敏感？没有生成新预测、训练或修改模型/归一化/原始标签。保留全部2582个validation origins、17架flight、3个seed。Sep8/Sep19没有访问。

先写protocol再加载原始日志与计算新的评分。沿用500ms控制活动分组（ALL/low/middle/high），没有重新计算阈值。原始评分始终并列保留。

三个条件：
1. original：冻结发布时间网格标签及原预测。
2. synchronized：原预测保持时间轴不变，原日志使用timestamp_sample在同一网格时刻插值；角速度线性插值，姿态SLERP。只调整评分参照，不修改初始状态/历史/预测。插值可能使用之后才发布的测量，因此仅限离线参考，不能输入模型或宣称部署可用。
3. synchronized_lowpass：同步实测角速度先在原约100Hz采样时间节点上做固定连续时间双一阶低通，再取评价时刻；预测在自身native节点上使用同一滤波器。两者均按分段线性输入精确积分，不是先把实测降到50Hz再平滑。姿态不滤波，避免新增不明确的四元数滤波定义。

滤波H(s)=1/(1+tau*s)^2，tau=1/(2π×10)s；每极点10Hz，组合−3dB约6.44Hz，25Hz理论衰减约17.21dB。这是固定降带宽诊断，只降低混叠风险，不能保证无混叠、恢复原100Hz日志已丢失的高频，或修复已训练模型的输入混叠。没有扫描截止频率或搜索最优时移。

滤波器两状态在各自t0值初始化，不用实测标签初始化预测滤波器。保留起始瞬态，并分开报告0–500与500–1000ms。同步参照在t0可能与原预测初值不同，详见initial_mismatch.csv；不通过平移预测强行消除该差异。

“共同测量时刻”仅指sample timestamp标签同步，不能补偿未知滤波/估计器物理群延迟。原预测本来学的是异步标签，这也不是重新训练后的同步模型性能。

## 500ms端点（ALL）
'''+table('500ms','ALL')+'''

## 0–500ms区间波形（ALL）
'''+table('0_500ms','ALL')+'''

## 高控制变化组0–500ms区间波形
'''+table('0_500ms','high')+'''

## 高控制变化组500–1000ms区间波形
'''+table('500_1000ms','high')+'''

## 实际结果与判断
高控制变化组0–500ms：同步后的p/q/r区间RMSE分别比原评分低4.36%、4.63%、0.47%；进一步固定低通后，相对原评分分别低40.24%、20.90%、12.28%。这些是评分变化，不能写成模型性能提升或噪声占比。

同步后的p、q评分在3/3seed及17/17flight（三seed平均）下降；r只有10/17flight下降、7/17上升。姿态评分反而从3.3942°增至3.5366°（增加4.20%），3/3seed、17/17flight同方向。两个cohort及所有时域结果均保留在summary.csv，不能只报告滚转变好。

双低通后高控制组p仍有0.3176rad/s、r仍有0.1423rad/s区间误差；500–1000ms分别为0.3314、0.1707rad/s。残差没有消失，偏航并非只表现为可由本低通去除的快速差异。

同步t0参考与原初值的向量RMS差为0.2796rad/s（这里为全部origin pooled的初值诊断量，不是flight宏平均性能指标）。因此极短时结果还包含初始观测定义不一致；没有偷偷重置预测初值。线性插值自身也有平滑效应，不能把同步条件的下降全部归因于纠正物理时间差。

这轮支持“滚转误差对快速波形/观测带宽敏感”，不支持“时间同步即可解决持续操纵响应”，也不支持“偏航问题主要由测量快速噪声造成”。不建议据此调模型或控制器。下一步优先制定专门输入激励与同步高频记录方案，尤其补足rudder正负方向的孤立变化；需要真实输入响应证据再判断模型缺陷。这里只提出方案方向，不启动实飞或训练。

## 统计与边界
轴误差单位rad/s，aggregate rate为向量RMSE，姿态为四元数测地RMS度。区间按真实dt加权每origin的误差平方；flight内等权origin后开方，再flight等权，再三seed均值与sample SD。t0不计入区间/端点指标。不同处理后的误差不能解释为模型获得了同等幅度的提升，三seed SD不是测量不确定性或置信区间。

全部原点保留；原始标签逐值复核；同一协议无外插，插值括号最大间隔≤50ms，否则停止而不筛掉样本。原始区间分数与上轮162个配对统计单元复核通过。9项单元测试通过（本轮5项和先前对齐4项），包含连续系统解析解、常量、前缀、插点不变、四元数符号、无外插/输入不变等。完整结果、cohort及seed/flight方向见CSV，未只展示有利指标。

## 解释限制与下一步
同步本身是否减小误差与低通后的误差下降必须分开讨论。低通降低误差只说明差异在被衰减的频段较大，不能证明这些频段是噪声；它也削弱真实快速运动。同步若使某指标变差，应保留，不能搜索反向时移修饰结果。

本轮不重新检验任意控制输入的因果响应，也不证明持续操纵模拟正确。下一步应根据完整结果决定是否需要同步高频记录和专门小幅正负输入激励；未执行新实验、控制器修改或实飞。既有主模型与独立测试结论保持不变。
'''
    (OUT/'report.md').write_text(report);(OUT/'review_report.md').write_text(report)
    comp=m.read_json(OUT/'completion.json');comp.update(status='complete',tests_passed=9,output_sha256={m.rel(p):m.file_hash(p) for p in OUT.iterdir() if p.is_file() and p.name!='completion.json'},reference_sha256=m.file_hash(ART/'scoring_references.npz'))
    m.write_json(OUT/'completion.json',comp)

if __name__=='__main__':main()
