"""Honest lifecycle report: prepared/blocked/running/evaluated, never invented results."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pacld_crm_v2 import OUT,LOCAL,BASE,write_json

REPORT=BASE/'crm_training_comparison_v2_zh.md'

def table(frame):
    def fmt(x):return f'{x:.6g}' if isinstance(x,(float,np.floating)) else str(x)
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+['| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])

def paired_interval(frame,metric,arm,reference,seed=17):
    wide=frame.pivot(index='date',columns='group',values=metric).dropna(subset=[arm,reference]);delta=(wide[arm]-wide[reference]).to_numpy()
    rng=np.random.default_rng(seed);boot=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
    return dict(arm=arm,reference=reference,metric=metric,date_mean_difference=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),dates=len(delta))

def main():
    contract=json.loads((OUT/'training_contract.json').read_text());pre=json.loads((OUT/'preflight.json').read_text()) if (OUT/'preflight.json').exists() else {}
    cal=json.loads((OUT/'loss_calibration.json').read_text()) if (OUT/'loss_calibration.json').exists() else {}
    sample=json.loads((OUT/'sampling_summary.json').read_text());status='prepared';body='';answers={str(i):'尚无正式训练/冻结评价结果，不能回答。' for i in range(1,8)}
    if cal.get('status')=='BLOCKED':
        status='blocked_before_formal_training'
        pd.DataFrame(columns=['group','update','cumulative_windows','full','macro','loss','M_normal','M_future100']).to_csv(OUT/'training_progress.csv',index=False)
        pd.DataFrame([dict(group=g,status='not_started_calibration_blocked',updates=0,windows=0) for g in ['B0','B1','B2']]).to_csv(OUT/'training_comparison.csv',index=False)
        for name,columns in [('moment_comparison',['group','split','control','date','M','M2']),('state_prediction_comparison',['group','split','component','horizon_s','rmse']),('cir_comparison',['group','time','gain_median']),('a2_comparison',['group','G_obs','G_PACLD','independent_response_score'])]:
            pd.DataFrame(columns=columns).to_csv(OUT/(name+'.csv'),index=False)
        diagnostics=pd.DataFrame(cal['diagnostics']);dy=diagnostics[diagnostics.module=='dynamics']
        body='## 校准阻碍\n\n**未启动B0/B1/B2正式训练。** 预先冻结的train校准停止规则已触发：\n\n'+'\n'.join('- '+r for r in cal['reasons'])+'\n\n'+table(dy[['batch','group','M','base_norm','moment_norm','lambda','weighted_ratio','cosine']])+'\n\n跨批Dynamics平方矩梯度方向：\n\n'+table(pd.DataFrame(cal['stability']))+'\n\n完整模块范数、余弦、加权比例及拟裁剪状态见 `loss_calibration.json`。停止依据是train校准，不是validation结果。未改变lambda、批量或采样来追求通过，也没有运行替代训练。H1/H2/H3仍未经过本轮训练检验。'
        fig,ax=plt.subplots(1,2,figsize=(11,4))
        for arm,g in dy.groupby('group'):
            ax[0].plot(g.batch,g.weighted_ratio,'o-',label=arm)
            st=pd.DataFrame(cal['stability']);v=st[st.group==arm];ax[1].plot(np.arange(len(v)),v.cosine,'o-',label=arm)
        ax[0].axhline(.25,color='gray');ax[0].axhspan(.0625,1.,color='gray',alpha=.1);ax[0].set(xlabel='Independent train calibration batch',ylabel='Weighted CRM / base Dynamics gradient norm');ax[1].axhline(0,color='gray');ax[1].set(xlabel='Batch pairs (0/1, 0/2, 1/2)',ylabel='CRM squared-moment gradient cosine')
        for a in ax:a.legend()
        fig.tight_layout();fig.savefig(OUT/'calibration_stability.png',dpi=160);plt.close(fig)
        body+='\n\n![校准稳定性](crm_training_comparison_v2/calibration_stability.png)\n'
        answers={str(i):'UNTESTED：校准停止，三组正式更新均为0；不作优劣、泛化或CIR变化判断。' for i in range(1,6)}|{'6':'当前校准未支持直接启动该固定CRM训练比较。需要先解释梯度校准不稳定，不能靠validation调权。','7':'本轮停止，不自动加大CRM权重或切换范式。现有证据不足以选择保留CRM或替代方法。'}
    elif (LOCAL/'evaluation_status.json').exists() and json.loads((LOCAL/'evaluation_status.json').read_text())['status']=='complete':
        status='evaluation_complete';progress=pd.read_csv(OUT/'training_progress.csv');mom=pd.read_csv(OUT/'moment_comparison.csv');state=pd.read_csv(OUT/'state_prediction_comparison.csv');cir=pd.read_csv(OUT/'cir_comparison.csv');a2=pd.read_csv(OUT/'a2_comparison.csv')
        training=progress.sort_values('update').groupby('group').tail(1);training.to_csv(OUT/'training_comparison.csv',index=False)
        m=mom[(mom.split=='validation')&(mom.date=='ALL')];s=state[state.date=='ALL'];c=cir[(cir.phase=='ALL')&(cir.multiplier==1)&cir.time.isin([.2,.5])]
        intervals=[]
        md=mom[(mom.split=='validation')&(mom.date!='ALL')&(mom.control=='normal')].copy();md['absolute_M']=md.M.abs()
        for reference in ['B0','B2']:intervals.append(paired_interval(md,'absolute_M','B1',reference))
        # No retrospective noninferiority margin or causal-gain target.
        def mv(arm):return float(m[(m.group==arm)&(m.control=='normal')].M.iloc[0])
        def sm(arm):return s[s.group==arm].set_index(['cohort','component','horizon_s']).rmse
        b1=sm('B1');b0=sm('B0');b2=sm('B2');common=b1.index.intersection(b0.index).intersection(b2.index)
        worse0=int((b1[common]>b0[common]).sum());worse2=int((b1[common]>b2[common]).sum())
        g20=c[c.time==.2].set_index('group').gain_median.to_dict()
        answers={'1':f'正常validation |M|：B1={abs(mv("B1")):.6g}，B0={abs(mv("B0")):.6g}；{worse0}/{len(common)}项物理RMSE高于B0。不能单凭矩减少宣布总体更好。','2':f'正常validation |M|：B1={abs(mv("B1")):.6g}，B2={abs(mv("B2")):.6g}；{worse2}/{len(common)}项物理RMSE高于B2。参见日期配对描述性区间。','3':'逐日期有符号矩及区间见下表和moment_comparison.csv；只有少量独立validation日期，窗口数不是独立样本量。','4':f'0.2秒内部CIR增益中位数：{g20}。变化不等于真实控制增益改善，增大也不自动是好事。','5':f'B1较B0有{worse0}/{len(common)}项物理RMSE数值上升；未预设非劣界，不宣称统计非劣。','6':'若B1仅降低矩而物理预测或CIR没有独特改善，则不支持独特控制学习收益；若B1/B2相近，反馈约束解释仍成立。所有判断仅为本seed固定预算观察性证据。','7':'不自动追加预算、权重、多种子或结构试验。依据配对结果决定后续设计；不把预算不足、噪声或结构限制中的任一项当成已证原因。'}
        body='## 固定第64次更新结果\n\n'+table(training[['group','update','cumulative_windows','cumulative_seconds','windows_per_second','peak_cuda_bytes','forward_counts']])+'\n\n### Validation矩\n\n'+table(m)+'\n\n日期配对的绝对局部矩差（不是总体M平方差的置信区间）：\n\n'+table(pd.DataFrame(intervals))+'\n\n### 真实状态预测\n\n'+table(s)+'\n\n### CIR\n\n'+table(c)+'\n\n### A2观察性条件增益与独立解释分数\n\n'+table(a2)
        fig,axs=plt.subplots(1,2,figsize=(11,4))
        for arm,g in progress.groupby('group'):axs[0].plot(g['update'],g.loss,label=arm);axs[1].plot(g['update'],g.M_normal,label=arm)
        axs[0].set(ylabel='Total objective',xlabel='Optimizer updates');axs[1].set(ylabel='Normal control moment',xlabel='Optimizer updates')
        for ax in axs:ax.legend()
        fig.tight_layout();fig.savefig(OUT/'training_curves.png',dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,4))
        for control,g in m.groupby('control'):ax.plot(g.group,g.M,'o-',label=control)
        ax.axhline(0,color='gray');ax.legend();ax.set_ylabel('Validation signed moment');fig.tight_layout();fig.savefig(OUT/'validation_moments.png',dpi=160);plt.close(fig)
        fig,axs=plt.subplots(2,2,figsize=(10,7))
        for ax,component in zip(axs.flat,['p','v','R','omega']):
            for arm,g in s[(s.cohort=='original_fixed156')&(s.component==component)].groupby('group'):ax.plot(g.horizon_s,g.rmse,'o-',label=arm)
            ax.set(title=component,xlabel='Horizon (s)',ylabel='Physical RMSE');ax.legend()
        fig.tight_layout();fig.savefig(OUT/'state_rmse.png',dpi=160);plt.close(fig)
        fig,axs=plt.subplots(1,2,figsize=(11,4))
        for arm,g in cir[(cir.phase=='ALL')&(cir.multiplier==1)&(cir.time<=.2)].groupby('group'):
            axs[0].plot(g.time,g.positive_median,'o-',label=arm);axs[1].plot(g.time,g.negative_median,'o-',label=arm)
        for ax,title in zip(axs,['Positive pulse','Negative pulse']):ax.set(title=title,xlabel='Time (s)',ylabel='Median roll rate difference (rad/s)');ax.legend()
        fig.tight_layout();fig.savefig(OUT/'cir_response.png',dpi=160);plt.close(fig)
        for name in ['training_curves','validation_moments','state_rmse','cir_response']:body+=f'\n\n![{name}](crm_training_comparison_v2/{name}.png)'
    elif (OUT/'training_progress.csv').exists():status='training';body='正式流程已启动，最终评价尚未完成。当前状态见本地各组status.json与training_progress.csv。'
    budget={arm:0 for arm in ['B0','B1','B2']}
    if (OUT/'training_progress.csv').exists():budget.update(pd.read_csv(OUT/'training_progress.csv').groupby('group')['update'].max().astype(int).to_dict())
    write_json(OUT/'result_summary.json',dict(status=status,formal_updates=budget,answers=answers,scientific_boundary='observational prediction association and model-internal CIR; no causal gain truth',calibration_reasons=cal.get('reasons',[]),single_seed=17,no_follow_on_runs=True))
    questions={'1':'B1是否比同预算B0更好？','2':'B1是否比未来控制负对照B2更好？','3':'正常Moment是否改善且跨日期稳定？','4':'内部CIR增益是否变化？','5':'普通状态预测精度是否受影响？','6':'是否支持继续研究CRM？','7':'下一步保留CRM还是改用其他范式？'}
    text=f'''# PACLD-CRM-v2 三组等预算控制相关训练判别实验

当前状态：**{status}**。实际正式更新：{budget}。未完成评价时不填写效果指标。

## 冻结合同

基线 `{contract['baseline']}`；共同checkpoint SHA256 `{contract['checkpoint_sha256']}`。三组结构相同，原模型源码不修改。优化器策略：{contract['optimizer']}；lr={contract['lr']}，lambda_macro=1，梯度裁剪={contract['max_grad_norm']}。生产精度={contract['precision']}。

预定三组各64次更新×512窗口=32,768次采样，seed17，共享完全一致的身份及顺序。训练仅CRM-v1的11,221窗口、114航次、17日期，使用原生0.5秒支持及0.2秒附近目标；**不同于原始0.2/0.5/1/2秒混合时域训练分布**。共同序列覆盖{sample['unique_windows']}个唯一窗口、{sample['repeated_draws']}次重复抽样；重复是有放回分层抽样的结果，不是新增独立数据。

主控制为前100ms差动尾舵变化，B2为250–350ms未来控制负对照。尺度严格复用CRM-v1完整精度：{contract['scales']}。未来控制残差只进入附加Loss。两遍微批量间参数固定，损失梯度为2M_B×mean(∇m_i)，不是mean(m_i²)或微批量矩平方。Moment完整初始Belief detach，FULL/Macro Encoder图保留。原FULL/Macro窗口归一化与无有效Macro时贡献0的规则保留。

预定比较保证**数据和优化更新预算相同**，不是GPU计算量相同。逻辑前向计数定义：每个FULL/Macro分支、每个CRM rollout（连同其必要历史编码）各计一次；B0每微批量FULL/Macro加无梯度Moment诊断共2次，B1/B2无梯度Moment、FULL/Macro、梯度Moment共3次。目标记录发生在optimizer更新后，但Loss/Moment对应该更新的前一步参数，不额外添加一次更新后回放。吞吐和ETA含实际已完成更新的窗口缓存准备及计算时间。

## 校准及定向验证

校准使用独立于正式采样流的三个512窗口批量；它们来自同一train日期/航次总体，可以共享窗口，不是三个独立实飞数据集。第0批固定权重使Dynamics上的CRM梯度为FULL+Macro的25%，另两批只检查稳定性。该25%是固定有限强度诊断，不是最优值。稳定性操作门槛在看结果前写入合同：任一批近零/非有限、lambda>10,000、加权梯度比不在[0.0625,1]，或任意两个批次Dynamics平方矩梯度余弦为负，则停止正式训练。这些是校准停止条件，不是模型效果筛选门槛。

preflight状态：{pre.get('status','not_run')}。本次32窗口可丢弃smoke耗时约16.19秒、1.98窗口/秒，allocated显存峰值约108.87 MiB，梯度范数17.57触发原阈值10裁剪；这不是正式训练吞吐。FP64完整图对微批量累积的FULL+Macro/正常M²/未来M²梯度相对L2误差分别约2.16e-15/4.83e-16/7.42e-16。完整图/微批量梯度等价、三组模型与Adam初态、未来残差不影响前向、Belief梯度隔离、Dynamics可达、GPU速度/显存数据见 `preflight.json`。smoke使用可丢弃副本，不计入正式预算；原冻结checkpoint不变。没有使用validation挑选校准权重或训练checkpoint。

{body}

## 七项问题

'''+ '\n\n'.join(f'{k}. **{questions[k]}** {v}' for k,v in answers.items())+'''

## 冻结评价与解释限制

只有B0/B1/B2全部完成64更新后，才打开validation。评价复用原PACLD156个窗口、A2固定身份和原状态校正/局部回归、CIR306窗口及正负差动pulse。CIR重新编码各模型但固定原选择、相位组与扰动协议，不按模型表现重筛。Validation条件模型只在train拟合；CIR是内部模型干预，G_obs是闭环观察性关联。验证日期少、单seed，日期bootstrap仅描述性；未预设非劣界，不能把数值接近称为证明状态性能不恶化。

正式训练、恢复和自动评价入口已实现并通过语法检查；若校准停止，这些后续路径没有获得端到端执行验证。若校准停止，所有后续评价维持未执行，不能制作虚构三组对比或把初始模型重复三次当成训练结果。

## 一键复现、恢复与发布

```bash
cd /home/zn/flap-system-identification/.worktrees/pacld-crm-training-comparison-v2
bash scripts/run_pacld_crm_v2.sh
# 仅在已启动任务中断后，严格恢复：
bash scripts/run_pacld_crm_v2.sh --resume
```

脚本串行执行B0/B1/B2，只使用CUDA_VISIBLE_DEVICES指定的一张GPU；结束后自动评价、生成报告、按公开allowlist提交推送并核对远端SHA。校准BLOCKED会拒绝启动，不能用resume绕过。已存在冻结校准结果时直接复用；BLOCKED状态不能靠重跑一键脚本覆盖。被排除的采样身份文件可由seed确定性重建并校验合同散列。逐窗口采样身份、控制/状态中间文件、checkpoint、RNG、优化器及梯度向量在`crm_training_comparison_v2/local/`或被.gitignore排除的采样文件中，仅本地保存。公开仅包含代码、合同、汇总统计与图表。
'''
    REPORT.write_text(text)

if __name__=='__main__':main()
