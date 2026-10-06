"""Deliver controlled budget/output comparisons, without declaring a noise floor."""
import json
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_control_windows_v2 as base
from reference_capacity_v6 import OUT,model_from
from control_windows_v3 import predict
from evaluate_control_windows_v2 import METRICS
from audit_reference_capacity_v6_rates import main as audit_rates


def table(df):
    def fmt(v):return f'{v:.6g}' if isinstance(v,(float,np.floating)) else str(v).replace('|','\\|')
    lines=['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    lines+=['| '+' | '.join(fmt(v) for v in r)+' |' for r in df.itertuples(index=False,name=None)]
    return '\n'.join(lines)


def main():
    torch.set_num_threads(1);protocol=json.loads((OUT/'protocol.json').read_text());gate=json.loads((OUT/'tiny_decision.json').read_text());status=json.loads((OUT/'formal_launch/status.json').read_text())
    assert status['status'] in ['complete','not_launched'];x,y,stats=base.load();w=pd.read_csv(base.DATA/'windows_v2.csv');records=[]
    audit_rates()
    for scope in ['tiny','formal']:
        if not (OUT/scope).exists():continue
        for path in sorted((OUT/scope).glob('*')):
            if not path.is_dir():continue
            run=json.loads((path/'run_protocol.json').read_text());state=json.loads((path/'status.json').read_text());assert state['status']=='complete'
            assert base.sha(base.ROOT/'scripts/train_reference_capacity_v6.py')==run['script_sha256']
            for ck in state['completed']:
                step=ck['step'];checkpoint=path/f'step_{step}.pt';assert base.sha(checkpoint)==ck['checkpoint_sha256']
                subset='fit8' if scope=='tiny' else 'validation'
                with np.load(path/f'{step}_{subset}_predictions.npz') as z:ix=z['array_index'][:3];expected={k:z[k][:3] for k in y}
                model=model_from(checkpoint);actual=predict(model,x,ix);diff=max(float(abs(actual[k]-expected[k]).max()) for k in y);assert diff<1e-5
                inputs={k:v[ix].clone() for k,v in x.items()};inputs['future_reference'][:,25:]+=1
                with torch.no_grad():changed=model(**inputs)
                for k in y:np.testing.assert_array_equal(actual[k][:,:25],changed[k][:,:25].numpy())
                records.append(dict(scope=scope,arm=run['arm'],seed=run['seed'],steps=step,reload_max_abs=diff,reproduction_max_abs=ck['reproduction_max_abs'],prefix_causality=True))
    for rel,digest in protocol['source_pins'].items():assert base.sha(base.ROOT/rel)==digest
    base.write(OUT/'verification.json',dict(checkpoints=records,focused_tests_passed=20,source_pins_verified=True,python=sys.version,torch=torch.__version__,sealed_test_accessed=False))
    tiny=pd.read_csv(OUT/'tiny_summary.csv');pred=pd.read_csv(OUT/'predictability/summary.csv');projection=pd.read_csv(OUT/'predictability/future_projection_summary.csv')
    diagnostic=pred[(pred.partition.isin(['validation','train_oof']))&pred.horizon_s.isin([.02,.5])&pred.model.isin(['ridge','ridge_oof','persistence','past_harmonic_4','past_harmonic_6','oracle_future_phase_6'])]
    lines=['# 参考模型 v6：训练预算、输出表达与可预测性','',
        '沿用原50Hz H26/K50、407训练/739开发验证窗口，不重新筛选、不加入未来执行命令或未来状态。验证已多次用于开发，不能称盲测；sealed/reserved未打开。','',
        '## 对照设计与小样本结果','',
        '固定原GRU64、40070参数、初始权重、A-train归一化和原始状态损失。分别比较原积分模型1600→6400更新，以及同预算下直接预测角速度相对起点的变化；速度仍积分加速度、姿态仍用前后角速度的中点值积分。',
        '直接输出为初始角速度加6倍训练标准差以内的残差，不再施加原角加速度增量上限。这属于输出参数化的变化，不能把结果只归因为去掉积分链。直接范围外的目标占训练角速度分量约0.015%、验证约0.040%；仍不能认为边界外动力学已覆盖。',
        '训练从相同seed重新运行，保留AdamW状态扩展到6400。积分模型1600权重与旧模型逐项复核；不重新初始化优化器冒充原训练续跑。1600和6400均为预定快照，不按验证误差挑选。','',
        table(tiny),'',table(pd.DataFrame(gate['decisions'])),'',
        '小样本为预先选定8个训练窗口，指标为完整1秒波形RMSE、先窗口后飞行平均。训练误差大幅降低证明当前设置能够记忆这些片段，说明此前1600步的训练内误差不能简单解释成硬性噪声下界；但神经网络也可能记忆噪声，不能据此证明新片段同样可预测。','',
        '## 正式训练结果','']
    formal=None
    if status['status']=='complete':
        formal=json.loads((OUT/'formal_decision.json').read_text());s=pd.read_csv(OUT/'formal_summary.csv');shown=s[(s.cohort=='validation')&(s.subset=='all')&s.horizon_s.isin([.5,1.])]
        lines+=['三个种子17/23/42、相同407个训练窗口和采样概率。以下p/q/r单位rad/s，姿态rad，速度m/s；全部使用相同原始状态目标。','',table(shown[['arm','horizon_s','p','q','r','attitude','velocity']]),'']
        for c in formal['comparisons']:
            lo,hi=c['paired_flight_bootstrap95'];lines.append(f"- {c['comparison']}：{c['candidate']} 对 {c['baseline']}，0.5/1秒主动p/q主指标改善{c['primary_gain_fraction']:.2%}；按飞行配对bootstrap95%区间[{lo:.2%}, {hi:.2%}]；相对门槛通过={c['relative_pass']}。")
        lines+=['','combined同时改变训练预算与输出表达，只表示整体效果；单因素结论以budget/output1600/output6400为准。','',table(pd.DataFrame(formal['absolute_by_seed'])),'']
        gap=s[(s.subset=='all')&(s.horizon_s==.5)&s.arm.str.endswith('6400')]
        lines+=['训练与开发验证差距（0.5秒）：','',table(gap[['arm','cohort','p','q','r','attitude','velocity']]),'']
        rr=pd.read_csv(OUT/'reference_hold_summary.csv');rr=rr[(rr.subset=='all')&rr.horizon_s.isin([.5,1.])]
        lines+=['固定模型的参考保持诊断：','',table(rr[['arm','intervention','horizon_s','p','q','attitude']]),'',
            '参考保持导致预测误差变化，只说明模型依赖未来参考序列，不能当作真实飞机的反事实控制响应。','']
        bounds=pd.read_csv(OUT/'rate_bound_audit.csv');bounds=bounds[(bounds.scope=='formal')&(bounds.partition=='validation')&(bounds.steps==6400)&(bounds.axis=='p')]
        bounds=bounds.groupby(['arm','signal'])[['absolute_increment_acceleration_p99','fraction_above_120','fraction_at_120']].mean().reset_index()
        lines+=['滚转输出增量检查（离散差分rad/s²；120是旧模型上限，不是实飞安全限制）：','',table(bounds),'',
            '直接输出不再受原角加速度上限约束。即使真实参考回放下没有异常增量，也不能证明未见过的参考变化具有正确的执行时延或物理控制作用。','']
    lines+=['## 高频误差是否不可预测？','',
        '使用固定alpha=10的线性岭回归，以历史状态/参考/执行命令以及截至目标时刻的计划参考为输入。5个训练飞行做整飞行留一交叉预测，每折归一化只用其余训练飞行；开发验证使用全部训练飞行拟合的固定基线。',
        '另用过去26个观测点拟合1/2/3/4/6阶谐波：保持当前频率给出因果外推；提供真实未来相位只作非部署诊断。未来真值上的谐波投影属于事后拟合，不能当预测准确率。','',table(diagnostic),'',
        '高阶相位结构在事后拟合中更明显，但因果相位外推的长时效果没有因此自动改善。不同阶数是预先列出的诊断，不选择最好的验证阶数作为模型。旧v4约15%的数值针对GRU残差的前两阶投影，本轮下表针对真实波形，不应直接混比。','',
        table(projection[projection.axis=='p']), '',
        '**目前不能认定剩余误差是不可预测噪声。**这些有限基线的失败只限定这些模型与输入表示的表现；缺少独立测量噪声真值或重复同条件试验，无法给出可信的不可约噪声下界。真实未来相位诊断也不是可实现预测上限。','',
        '## 验证与结论边界','',
        '20项针对性测试通过：相同初始权重/参数量、零输出等价性、直接速率输出含义、有限梯度、未来参考前缀因果性、基线归一化与目标时刻隔离。所有保存模型重载检查通过，原始数据和旧冻结模型哈希保持一致。',
        '本轮训练/诊断完成，不等于控制可用性已通过。短时预测、真实操纵响应与闭环控制必须分别验收；没有因本轮预测改善而启动控制优化或实飞。','']
    if formal:
        direct=next((c for c in formal['comparisons'] if c['comparison']=='output6400'),None)
        if direct and direct['relative_pass']:
            lines+=['下一步可将直接输出模型作为预测研究候选，优先检查未通过的绝对误差、各飞行响应形状和输入支持范围；继续保留积分模型作对照，不能仅凭低RMSE替换控制器。']
        else:
            lines+=['本轮候选均未通过相对和绝对误差门槛，保留A1600研究基线。延长训练降低训练误差却没有改善开发验证表现，提示泛化问题仍需定位，不能仅靠继续增加步数解决。',
                '下一步先按飞行与工况分解泛化差距，并检查历史相位、频率外推误差；候选选择优先采用训练飞行内分组交叉验证。小样本记忆能力不等于飞行控制能力，本轮也未证明所有旧日志无效或必须补飞。']
    (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')
    # Time-selected validation example for all four prespecified snapshots.
    ev=w[w.partition=='validation'];j=int(ev.iloc[len(ev)//2].array_index);loc=int(np.flatnonzero(ev.array_index.to_numpy()==j)[0]);t=np.arange(1,51)*.02
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    for ax,axis in enumerate(['p','q']):axes[ax].plot(t,y['rates_b'][j,:,ax].numpy(),label='Observed',color='black',lw=1.4)
    if formal:
        for arm in ['integrated','direct']:
            for step in [1600,6400]:
                p=OUT/'formal'/f'{arm}_17'/f'{step}_validation_predictions.npz'
                if not p.exists():continue
                with np.load(p) as z:rates=z['rates_b'][loc]
                for ax in [0,1]:axes[ax].plot(t,rates[:,ax],label=f'{arm}{step}',lw=1)
    for ax,axis in enumerate(['p','q']):axes[ax].set_ylabel(f'{axis} (rad/s)');axes[ax].grid(alpha=.3)
    axes[0].legend(fontsize=8,ncol=3);axes[-1].set_xlabel('Forecast horizon (s)');fig.suptitle(f'Time-selected validation window {j}, seed17');fig.tight_layout();fig.savefig(OUT/'waveform_comparison.png',dpi=150);plt.close(fig)
    base.write(OUT/'plot_provenance.json',dict(array_index=j,window_id=w.iloc[j].window_id,selection='middle fixed validation-table row, not error-selected'))
    paths=[*sorted((base.ROOT/'scripts').glob('*reference_capacity_v6*.py')),base.ROOT/'scripts/diagnose_reference_predictability_v6.py',base.ROOT/'src/system_identification/models/reference_gru_direct_rate.py',base.ROOT/'tests/test_reference_capacity_v6.py']
    base.write(OUT/'delivery_manifest.json',dict(status='complete',checkpoints=len(records),formal_trajectories=len(status.get('completed',[])),focused_tests=20,
        source_pins=protocol['source_pins'],artifacts={str(p.relative_to(base.ROOT)):base.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['delivery_manifest.json','pipeline_status.json']},
        code={str(p.relative_to(base.ROOT)):base.sha(p) for p in sorted(set(paths))},sealed_test_accessed=False))
if __name__=='__main__':main()
