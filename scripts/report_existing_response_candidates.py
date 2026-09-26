"""Finalize fixed-candidate audit; no model or heldout data access."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/flap-paper-mpl')
import subprocess
import numpy as np
import pandas as pd
from run_existing_response_candidates import OUT,ROOT,m

def md(f):
    return '| '+' | '.join(f.columns)+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+'\n'.join('| '+' | '.join(f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |' for row in f.itertuples(index=False,name=None))

def main():
    c=pd.read_csv(OUT/'candidate_review.csv');resp=pd.read_csv(OUT/'candidate_responses.csv');pairs=pd.read_csv(OUT/'opposite_sign_neighbors.csv');counts=pd.read_csv(OUT/'coverage_summary.csv')
    result=subprocess.run(['/home/zn/anaconda3/envs/flap-train-gpu/bin/python','-m','pytest','-q','tests/test_existing_response_candidates.py'],cwd=ROOT,text=True,capture_output=True)
    m.write_json(OUT/'tests.json',dict(command=result.args,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr));assert result.returncode==0
    # Deterministic descriptive response association AFTER matching was fixed on inputs only.
    rows=[];seen=set()
    for _,p in pairs.iterrows():
        key=(p.channel,p.kind,*sorted([p.window_id,p.opposite_window_id]))
        if key in seen:continue
        seen.add(key);axis='q' if p.channel=='common' else 'r'
        for interval in ['late_500ms','late_1s']:
            sub=resp.query('axis==@axis and interval==@interval').set_index('window_id')
            x=float(sub.loc[p.window_id,'delta_rate']);y=float(sub.loc[p.opposite_window_id,'delta_rate'])
            rows.append(dict(channel=p.channel,kind=p.kind,window_a=p.window_id,window_b=p.opposite_window_id,distance=p.distance,interval=interval,response_a=x,response_b=y,opposite_response_sign=x*y<0,zero_response=x*y==0))
    pr=pd.DataFrame(rows);pr.to_csv(OUT/'matched_response_pairs.csv',index=False)
    ps=pr.groupby(['channel','kind','interval']).agg(unique_pairs=('window_a','size'),opposite_response_sign=('opposite_response_sign','sum'),zero_response=('zero_response','sum'),distance_median=('distance','median')).reset_index();ps.to_csv(OUT/'matched_response_summary.csv',index=False)
    fl=pd.read_csv(OUT/'response_per_flight.csv');s=pd.read_csv(OUT/'response_summary.csv')
    chosen=s.query('interval in ["late_500ms","late_1s"] and ((channel=="common" and axis=="q") or(channel=="rudder" and axis=="r"))')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    # Re-render full set using ASCII flight basenames, preserving identities in CSV.
    traces=pd.read_csv(OUT/'all_traces.csv');order=c.sort_values(['channel','flight_id','origin_timestamp_us']).reset_index(drop=True)
    with PdfPages(OUT/'all_candidates.pdf') as pdf:
        for start in range(0,len(order),4):
            fig,axs=plt.subplots(4,2,figsize=(12,12))
            for row,i in enumerate(range(start,min(start+4,len(order)))):
                item=order.iloc[i];g=traces[traces.window_id==item.window_id].sort_values('step')
                for ch in ['drive','common','differential','rudder']:axs[row,0].step(g.elapsed_s.iloc[:50],g[ch].iloc[:50]-g[ch].iloc[0],where='post',label=ch)
                for axis in 'pqr':axs[row,1].plot(g.elapsed_s,g[axis]-g[axis].iloc[0],label=axis)
                axs[row,0].set_title(f'{i}: {item.channel} sign={item.sign:+.0f}',fontsize=9)
                axs[row,1].set_title(item.window_id.split('/')[-1],fontsize=7)
                axs[row,0].set_ylabel('Command departure');axs[row,1].set_ylabel('Rate departure [rad/s]')
                for ax in axs[row]:ax.axvline(.5,color='k',ls=':',lw=.5);ax.legend(fontsize=6);ax.set_xlabel('Actual elapsed time [s]');ax.grid(alpha=.2)
            for row in range(min(4,len(order)-start),4):
                for ax in axs[row]:ax.set_visible(False)
            fig.tight_layout();pdf.savefig(fig);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,ch,axis in zip(axs,['common','rudder'],['q','r']):
        for label,marker in [('late_500ms','o'),('late_1s','x')]:
            z=fl.query('channel==@ch and axis==@axis and interval==@label').sort_values('flight_id')
            ax.plot(np.arange(len(z)),z.sign_aligned_mean,marker=marker,label=label)
        ax.axhline(0,color='k',lw=.8);ax.set(title=ch,xlabel='Flight index',ylabel='Mean sign(command) × rate change [rad/s]');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Observed closed-loop association, not causal control effectiveness');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(OUT/f'response_per_flight.{ext}',dpi=170)
    plt.close(fig)
    text='''# 既有common/rudder候选响应复核

## 结论
现有日志含有可用响应信息，不能笼统说缺少控制变化；但这些候选大多不是前史命令稳定、输入持续保持、可作正负对照的实验。可继续用于短时预测和闭环日志分析，目前不能单靠它们验证持续操纵的因果响应。支持按具体缺口制定补充激励，不支持直接重建模型或丢弃现有数据。

本轮只使用原train缓存的68个common与38个rudder非重叠候选。未打开Sep8/Sep19、未训练、未生成模型预测。这里检验的是数据可比性和实测响应，不是模型与实飞响应对照已通过或失败。既有H26/K50保持不变。

## 固定方法与完整保留
候选直接来自上一轮nonoverlap_candidates.csv，identity和顺序在分析前固定。106个全部保留；3seed不是本轮统计对象，因为没有模型预测。原物理单位的全部p/q/r与四个控制通道轨迹保存在all_traces.csv；27页all_candidates.pdf展示每个候选，不按响应或模型误差选图。

候选基于前25个命令约500ms的变化量。新增描述性检查：后25命令中≥80%保持与前25平均变化同方向，后段同方向均值≥前段均值绝对值的一半，其他三通道后段RMS变化仍≤原train q25。前史稳定检查为H26目标通道相对origin的RMS≤前段变化量的一半。上述0.8/0.5是分析前固定的描述规则，不是物理稳态保证、质量准入门槛或满足/不满足可辨识性的定理。检查未通过的候选仍进入响应表。

保留native dt。late_500ms=状态16..25、late_1s=41..50；分别相对t0取实际dt加权平均角速度变化。它们是名义区间，实际每origin结束时间见candidate_review.csv。所有轴、早段与四端点均保留，未滤波、未拟合响应延迟或最佳时间常数。

## 持续性检查
'''+md(counts)+'''

因此common只有6/68、rudder只有3/38同时满足后半段同向和其他通道安静；再要求前史较稳定，两类各1个。不能把这两个片段单独拿出来宣称已完成重复实验，也不能反推剩余数据没用。

## 原始实测响应
下表先在每flight对候选的sign(输入变化)×角速度变化平均，再对flight等权汇总。负值只表示软件输入坐标与所观察变化反向，**不证明执行器物理符号错误**；闭环补偿、初始运动趋势与机体耦合都可能造成反向关联。

'''+md(chosen)+'''

common在约500ms时23/29flight为负关联，1s为24/29；rudder约500ms为22/24，1s为16/24。rudder符号关联随时间变化，不能拟合一条统一“持续命令增益”而忽略初始状态和其他输入。正负输入原样保留，逐candidate和flight结果可追溯。

## 相反方向是否拥有相近初态
仅根据起点速度NED三轴、角速度FRD三轴、重力方向三轴、扑频、四个初始命令选最近相反符号候选；14维采用已有train起点标准差做距离尺度，尺度写入protocol，绝不替换模型normalization。分别查同flight和其他flight，允许重用；匹配时不使用未来响应，没有最优结果筛选。

同flight有对向候选的common为42/68个、rudder9/38个；其最近距离中位数分别约1.27和1.11个标准化RMS单位。跨flight最近距离分别约0.83和0.99。这些距离不是0，不设事后合格阈值，也不能叫严格相同状态。没有匹配完整历史、相位、风、执行器状态和控制器积分状态。

下面为去掉双向重复identity对后的描述结果。opposite_response_sign表示正负输入候选的观测角速度变化是否反号；候选可在不同pair重用，不能把pairs当独立试验，反号也不证明因果；未设响应幅值显著性阈值，近零响应的符号尤其不可过度解释。

'''+md(ps)+'''

## 下一步：先制定有针对性的补充方案，不直接实飞
这次检查确认的缺口是“足够长且清楚记录的输入保持”“同工况正负重复”“其他通道及观测链可追溯”。不是数据总量少。因此保留当前全部训练数据和模型，开始将上一轮激励草案细化为：

1. 优先rudder：同一前飞工况成对正负输入，覆盖变化、保持和释放；记录实际最终命令与所有闭环补偿。不能用参考指令变化冒充舵机开环阶跃。
2. common：补充同工况双向和跨flight重复；先确认物理舵面映射、符号及实际舵角可观测性。
3. 观测：同步sample/publication时间、控制周期、最终命令、角速度与扑频/相位；如没有舵角反馈，只能辨识软件命令到机体的合成响应。
4. 幅值和执行方式需依据现场映射、已有批准飞行范围和可辨别响应来固定。本轮数据活动阈值不能直接换成实飞幅值；未填写未经依据的数值，未生成可执行任务。

可继续从现有数据开展完整闭环多输入辨识；孤立阶跃不是唯一可行方法。本轮没有比较所有辨识方法，因此“当前直接响应核验证据不足”不等于“这些日志在理论上不可辨识”。如果下一步要评价模型，可先固定这些候选，做原checkpoint同输入推理作为开发诊断；训练内误差不能替代独立因果验证。

## 检查
3项单元测试通过：持续性必须同时满足目标通道与其他通道条件、对向匹配只用初态/flight、native-dt加权正确。106个identity全部保留、命令/状态有限、dt为正、原cache与候选hash未变。所有产物是新增分析，旧报告和数据未修改，没有commit/push。
'''
    (OUT/'report.md').write_text(text);(OUT/'review_report.md').write_text(text)
    m.verify_pins(m.read_json(OUT/'protocol.json')['inputs'])
    m.write_json(OUT/'completion.json',dict(status='complete',tests_passed=3,training=0,new_predictions=0,heldout_accessed=False,outputs={m.rel(p):m.file_hash(p) for p in OUT.iterdir() if p.is_file() and p.name!='completion.json'}))
    print(ps.to_string(index=False))

if __name__=='__main__':main()
