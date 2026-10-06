"""Frozen-gate report for v9 decomposition and fold-isolated mean diagnostics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_control_windows_v2 as base
from run_reference_error_structure_v9 import OUT,V8,ARMS
from reference_groupcv_v8 import SEEDS
from run_reference_generalization_v7 import verify_manifest
from report_reference_generalization_v7 import table


def decisions(per):
    f=per[per.cohort=='oof'];flights=sorted(f.log_id.unique());cube=np.full((len(ARMS),3,5,2,3),np.nan)
    for ai,arm in enumerate(ARMS):
        for si,seed in enumerate(SEEDS):
            for fi,log in enumerate(flights):
                for hi,h in enumerate([.5,1.]):
                    for ki,axis in enumerate(['p','q','r']):
                        z=f[(f.predictor==arm)&(f.seed==seed)&(f.log_id==log)&(f.horizon_s==h)&(f.axis==axis)]
                        assert len(z)==1;cube[ai,si,fi,hi,ki]=z.iloc[0].normalized_mean_error_rmse
    assert np.isfinite(cube).all()
    rng=np.random.default_rng(909);resamples=rng.integers(5,size=(2000,5));result=[]
    axis_rows=[]
    for ai,arm in enumerate(ARMS[1:],1):
        seedscore=cube[[0,ai]].mean(axis=(2,3,4));gains=1-seedscore[1]/seedscore[0]
        flightscore=cube[[0,ai]].mean(axis=(1,3,4));flightgains=1-flightscore[1]/flightscore[0]
        axis_regression=cube[ai].mean(axis=(0,1))/cube[0].mean(axis=(0,1))-1
        gain=float(1-seedscore[1].mean()/seedscore[0].mean());boot=[]
        for ix in resamples:
            s=cube[[0,ai]][:,:,ix].mean(axis=(1,2,3,4));boot.append(1-s[1]/s[0])
        supported=bool(gain>=.10 and (gains>0).all() and (flightgains>0).sum()>=4 and (axis_regression<=.05).all())
        result.append(dict(predictor=arm,primary_gain=gain,seed_gains=gains.tolist(),flight_gains=flightgains.tolist(),
            bootstrap95=np.quantile(boot,[.025,.975]).tolist(),maximum_axis_horizon_regression=float(axis_regression.max()),followup_gate=supported))
        for h in [.5,1.]:
            for axis in ['p','q','r']:
                a=f[(f.predictor==arm)&(f.horizon_s==h)&(f.axis==axis)].groupby('log_id')[['mean_error_rmse','waveform_error_rmse']].mean().reindex(flights)
                b=f[(f.predictor=='zero')&(f.horizon_s==h)&(f.axis==axis)].groupby('log_id')[['mean_error_rmse','waveform_error_rmse']].mean().reindex(flights)
                bs=[]
                for ix in resamples:bs.append(1-a.iloc[ix].mean_error_rmse.mean()/b.iloc[ix].mean_error_rmse.mean())
                axis_rows.append(dict(predictor=arm,horizon_s=h,axis=axis,mean_gain=float(1-a.mean_error_rmse.mean()/b.mean_error_rmse.mean()),
                    rate_readout_gain=float(1-a.waveform_error_rmse.mean()/b.waveform_error_rmse.mean()),
                    positive_flights=int((a.mean_error_rmse<b.mean_error_rmse).sum()),bootstrap_low=float(np.quantile(bs,.025)),bootstrap_high=float(np.quantile(bs,.975))))
    pd.DataFrame(axis_rows).to_csv(OUT/'axis_gains.csv',index=False)
    base.write(OUT/'decision.json',dict(predictors=result,supported_followup_candidates=[r['predictor'] for r in result if r['followup_gate']],
        control_validated=False,rate_readout_is_not_a_dynamics_model=True,new_GRU_training_started=False,development_validation_used=False,sealed_test_accessed=False))
    return result,pd.DataFrame(axis_rows)


def main():
    assert json.loads((OUT/'status.json').read_text())['status']=='complete'
    verify_manifest(OUT/'protocol.json',['source_pins']);verify_manifest(V8/'delivery_manifest.json',['artifacts','code'])
    parts=pd.read_csv(OUT/'decomposition_summary.csv');per=pd.read_csv(OUT/'predictor_per_flight.csv')
    result,gains=decisions(per)
    d=parts[(parts.arm=='hold')&(parts.cohort=='oof')&parts.horizon_s.isin([.5,1.])].copy()
    d['ideal_constant_shift_rmse_gain']=1-np.sqrt(1-d.bias_share)
    d.to_csv(OUT/'mean_shift_scope.csv',index=False)
    p=pd.read_csv(OUT/'predictor_summary.csv');association=pd.read_csv(OUT/'association_summary.csv')
    flight=pd.read_csv(OUT/'decomposition_per_flight.csv');flight=flight[(flight.arm=='hold')&(flight.cohort=='oof')&(flight.horizon_s==1.)].groupby(['log_id','axis'])[['signed_bias','bias_mse']].mean().reset_index();flight['bias_rms']=np.sqrt(flight.bias_mse)
    support=any(r['followup_gate'] for r in result)
    fold=json.loads((OUT/'verification.json').read_text())
    assert len(fold['fold_provenance'])==15
    for item in fold['fold_provenance']:
        assert item['base_model_fit_indices']==item['calibrator_fit_indices']
        assert not set(item['base_model_fit_indices'])&set(item['query_indices'])
    fold.update(focused_tests_passed=32,development_validation_evaluated=False)
    base.write(OUT/'verification.json',fold)
    report=['# 参考模型 v9：留出飞行误差结构与历史信息诊断','',
        '**滚转p和俯仰q主要受波形误差限制；偏航r有更大的窗口均值误差。历史信息预测的有限对照不能等同于已验证的控制修正。**','',
        '## 数据与隔离','',
        '仅使用v8的5个原训练飞行、407窗口、3seed留一预测及同一个折模型的训练内预测，不重新训练GRU，不改变筛选，不评价739个开发验证窗口，未访问sealed/reserved。分解同时保留hold和phase结果；后续历史信息诊断固定使用hold。',
        '不能把所有v8 OOF误差拼起来，再留一训练修正器：其他折的GRU可能见过当前查询飞行。本轮每个折只使用同一个hold模型在其4个训练飞行上的残差作拟合标签，第5个飞行不参与GRU、修正回归或归一化。',
        '这种残差拟合标签仍是GRU训练内残差，通常比未见飞行乐观；严格隔离了外层查询，却不消除残差分布差异。失败不能证明误差不可预测。因此另设直接预测未来平均角速度的对照，其训练标签为4个拟合飞行的真实窗口均值，不依赖其他折的GRU输出。','',
        '## 误差分解','',
        r'对每个预测窗口，令e=预测−真值，则 MSE = mean(e)² + mean((e−mean(e))²)。',
        r'去均值误差再分成 (std(预测)−std(真值))² + 2[std(预测)std(真值)−cov(预测,真值)]。最后一项称波形/时序项，混合形状、延迟、相关性等，不能直接称纯相位误差或噪声。',
        '下表先按飞行求各平方误差分量，再等权平均飞行与种子后计算比例；不是单个窗口比例的中位数。窗口均值偏差会随片段变化，不等于全局常量偏移。','',
        table(d[['horizon_s','axis','bias_share','amplitude_share','shape_share','signed_bias','ideal_constant_shift_rmse_gain']]),'',
        'share以0–1表示；signed_bias单位rad/s。1秒内p/q均值项仅约3.5%/4.3%，即使事后知道每个窗口的完美常量偏移，按聚合MSE换算的RMSE最多降低约1.8%/2.2%；这个上限只针对保持波形不变的常量平移，不是模型精度上限。r的均值项约38%，因此更值得单独研究。',
        '不同窗口的正负偏差会抵消：全局有符号偏差接近零，并不代表窗口均值已经预测正确。每个飞行1秒均值偏差如下：','',table(flight[['log_id','axis','signed_bias','bias_rms']]),'',
        '## 历史信息能否解释均值？','',
        '固定alpha=100的岭回归，不根据留一结果扩大网格。state使用历史状态的末值、均值、标准差、线性趋势与当前模式，共58维；full再加入历史参考与执行命令的同类统计，共94维。均为过去26点，不使用未来参考、未来测量、飞行ID或预测误差作输入；标准化仅来自当前4个拟合飞行，并等权飞行。',
        '- zero：不修正，原GRU均值误差。',
        '- constant：从4个拟合飞行估计一个全局残差均值。',
        '- residual_state/full：预测同折GRU的窗口均值残差。',
        '- mean_state/full：直接预测真实未来窗口平均角速度，检验均值本身的可预测性。',
        '主指标是窗口均值误差的RMSE，以各折训练角速度标准差归一化，再等权p/q/r、0.5/1秒、飞行及种子。它与此前瞬时端点p/q主指标不同，不能直接比较百分比。','',
        table(pd.DataFrame([{k:r[k] for k in ['predictor','primary_gain','maximum_axis_horizon_regression','followup_gate']} for r in result])),'']
    for r in result:
        report.append(f'- {r["predictor"]}：主指标改善{r["primary_gain"]:.2%}；飞行配对bootstrap95%区间[{r["bootstrap95"][0]:.2%}, {r["bootstrap95"][1]:.2%}]；三个种子改善率为'+', '.join(f'{v:.2%}' for v in r['seed_gains'])+'。')
    report+=['','以上仅5个飞行、固定模型的探索性重采样，且训练集重叠，不能按窗口数或种子数充当独立飞行数。门槛为至少改善10%、全部种子改善、至少4/5飞行改善、任何轴/时域不退化超过5%。','',
        '原始单位下的OOF均值预测误差与代数速率读出误差（rad/s）：','',
        table(p[p.cohort=='oof'][['predictor','horizon_s','axis','mean_error_rmse','waveform_error_rmse']]),'',
        'waveform_error_rmse仅计算将每个窗口减去估计常量偏差后的速率误差，去均值波形完全不变。没有重算位置/姿态，也未限制起点连续性，因此不是已实现或物理一致的动力学修正，更不能称闭环收益。','',
        '按轴拆开的full历史直接均值预测结果（探索性分解，不按事后单轴收益改变门槛）：','',
        table(gains[gains.predictor=='mean_full'][['horizon_s','axis','mean_gain','rate_readout_gain','positive_flights','bootstrap_low','bootstrap_high']]),'',
        '## 波形误差与历史工况的关系','',
        'within_flight_associations.csv报告全部预定历史协变量与有符号均值偏差、去均值误差的逐飞行Spearman相关；下表为1秒去均值误差的相关中位数及范围。不据此删除弱响应窗口、不筛选“听话”样本，也不把相关关系当作控制因果效应。','',
        table(association[(association.horizon_s==1.)&(association.target=='centered_mse')][['axis','feature','median','min','max','count']]),'',
        '## 结论与交付','',
        ('有候选达到窗口均值研究门槛，见decision.json；仍需独立的轨迹一致性与控制响应验收。' if support else '**所有预定候选均未通过整体均值研究门槛，保留A1600基线，不添加全轴偏置修正。**'),
        'p/q下一步应围绕波形幅值与形状误差定位输入/状态表示的缺口，不能期待统一偏置解决主要误差。r的直接均值预测若有局部收益，可作为后续单轴研究线索，但本轮不因事后发现而改变预定门槛或宣布控制可用。有限线性对照的失败不证明误差不可约，也不证明必须补飞。',
        '32项针对性测试通过；逐窗口两级分解恒等式、150个辅助预测器重载、15组原GRU/修正器的拟合和查询来源核验通过。全部旧数据与模型保持原样。',
        '数据表：decomposition_per_window.csv、decomposition_per_flight.csv、decomposition_summary.csv；预测器：fold0至fold4/*.npz；评分：predictor_per_flight.csv、axis_gains.csv；冻结方案与结论：protocol.json、decision.json。','']
    (OUT/'report_zh.md').write_text('\n'.join(report))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,h in zip(axes,[.5,1.]):
        z=d[d.horizon_s==h].set_index('axis').loc[['p','q','r']];bottom=np.zeros(3)
        for name,label,color in [('bias_share','Window mean','tab:blue'),('amplitude_share','Amplitude','tab:orange'),('shape_share','Shape / timing','tab:green')]:
            ax.bar(['p','q','r'],100*z[name],bottom=bottom,label=label,color=color);bottom+=100*z[name].to_numpy()
        ax.set(title=f'OOF {h:g} s waveform error',ylabel='Share of aggregate MSE (%)',ylim=(0,100));ax.grid(axis='y',alpha=.2)
    axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=3,fontsize=9)
    fig.tight_layout();fig.savefig(OUT/'error_components.png',dpi=160,bbox_inches='tight');plt.close(fig)
    sources=[Path(__file__),base.ROOT/'scripts/run_reference_error_structure_v9.py',base.ROOT/'scripts/reference_error_structure_v9.py',base.ROOT/'tests/test_reference_error_structure_v9.py']
    base.write(OUT/'delivery_manifest.json',dict(status='complete',code={str(p.relative_to(base.ROOT)):base.sha(p) for p in sources},
        artifacts={str(p.relative_to(base.ROOT)):base.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'},
        tests_passed=32,new_neural_training=False,development_validation_used=False,sealed_test_accessed=False,control_validated=False))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
