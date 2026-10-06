"""Verify and report input support and channel-localized model sensitivity."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_control_windows_v2 as base
from run_reference_channels_v20 import OUT,V8,V18
from reference_channels_v20 import masked_reference,features,support,CHANNELS
from reference_groupcv_v8 import load_train,model_from,predict,SEEDS
from run_reference_generalization_v7 import verify_manifest
from report_reference_generalization_v7 import table


def main():
    torch.set_num_threads(1);verify_manifest(OUT/'protocol.json',['source_pins'])
    x,y,w=load_train();f=features(x);sw=pd.read_csv(OUT/'support_per_window.csv')
    checkpoints=[]
    for fold in range(5):
        info=json.loads((V8/f'fold{fold}/manifest.json').read_text());query=np.array(info['query_indices'])[:64]
        for seed in SEEDS:
            model=model_from(V18/f'fold{fold}/seed{seed}/final.pt')
            inp=dict(x);inp['future_reference']=masked_reference(x['future_reference'],31^16)
            p=predict(model,inp,query)['rates_b'][:,:,1]
            with np.load(OUT/f'fold{fold}/oof_aux_seed{seed}.npz') as z:
                np.testing.assert_array_equal(z['array_index'][:len(query)],query);diff=float(abs(p-z['hold_manual_yaw'][:len(query)]).max())
            assert diff<1e-6;checkpoints.append(dict(fold=fold,seed=seed,hold_manual_yaw_reload_max_abs=diff))
    for c in json.loads((OUT/'support_calibration.json').read_text()):
        fit=np.array(c['fit_indices']);query=np.array(c['query_indices']);a=support(f.iloc[fit][c['columns']],f.iloc[query][c['columns']],w.iloc[fit].log_id.to_numpy())
        np.testing.assert_allclose(a['mean'],c['mean'],atol=1e-12);np.testing.assert_allclose(a['scale'],c['scale'],atol=1e-12)
        assert abs(a['threshold']-c['threshold'])<1e-12
        r=sw[(sw.fold==c['fold'])&(sw.group==c['group'])&(sw['mask']=='factual')]
        np.testing.assert_array_equal(r.array_index,query);np.testing.assert_allclose(a['distance'],r.distance,atol=1e-12)
    d=pd.read_csv(OUT/'response_per_window.csv');pairs=pd.read_csv(OUT/'oof_channel_pairs.csv');sp=pd.read_csv(OUT/'support_per_flight.csv')
    high=pairs[pairs.log_id.str.contains('log_28_')&pairs.seed.isin([23,42])&(pairs.block=='full1s')&(pairs.stratum=='high')]
    log18=pairs[pairs.log_id.str.contains('log_18_')&(pairs.seed==23)&(pairs.block=='full1s')&(pairs.stratum=='all')]
    impact=[]
    actual=d[(d.cohort=='oof')&(d['mask']=='factual')&(d.block=='full1s')&d.log_id.str.contains('log_28_')]
    for seed in SEEDS:
        a=actual[(actual.seed==seed)&(actual.arm=='aux')].set_index('array_index');b=actual[(actual.seed==seed)&(actual.arm=='baseline')].set_index('array_index');delta=a.versus_held_mse-b.versus_held_mse
        impact.append(dict(seed=seed,window391_net_effect_energy_share=float(delta.loc[391]/delta.sum()),window391_aux_total_energy_share=float(a.loc[391,'versus_held_mse']/a.versus_held_mse.sum())))
    pd.DataFrame(impact).to_csv(OUT/'window391_concentration.csv',index=False)
    record=w.iloc[391].to_dict();record={k:(v.item() if isinstance(v,np.generic) else v) for k,v in record.items()}
    record.update(manual_yaw_initial=float(x['future_reference'][391,0,4]),manual_yaw_future_min=float(x['future_reference'][391,:,4].min()),manual_yaw_future_max=float(x['future_reference'][391,:,4].max()),origin_timestamp_us=int(w.iloc[391].start_timestamp_us)+520000)
    base.write(OUT/'window391.json',record)
    trace=sw[(sw.array_index==391)&sw['mask'].isin(['factual','held','only_manual_yaw','hold_manual_yaw'])];trace.to_csv(OUT/'window391_support.csv',index=False)
    corr=pd.read_csv(OUT/'bias_change_conditions.csv');g=corr[corr.log_id.str.contains('log_18_')&(corr.seed==23)]
    maximum=float(g.bias_square_change_spearman.abs().max())
    tests=Path('/tmp/reference-v20-tests.log').read_text();assert '3 passed' in tests and 'failed' not in tests
    (OUT/'tests.txt').write_text(tests)
    v=json.loads((OUT/'verification.json').read_text());v.update(focused_tests_passed=3,independent_model_reloads=checkpoints,support_calibration_and_distances_recomputed=True)
    base.write(OUT/'verification.json',v)
    base.write(OUT/'decision.json',dict(stage_complete=True,diagnostic_only=True,dominant_sensitivity_channel='manual_yaw',concentrated_window=391,
        original_and_held_reference_support_flagged=True,log18_not_explained_by_same_input_support_heuristic=True,
        log18_max_abs_bias_delta_feature_spearman=maximum,model_promoted=False,control_validated=False,no_training=True,sealed_test_accessed=False))
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    t=np.arange(50)*.02;axes[0].plot(t,x['future_reference'][391,:,4].numpy(),label='Logged reference');axes[0].axhline(float(x['future_reference'][391,0,4]),color='orange',label='Held initial value')
    axes[0].set(xlabel='Time from forecast origin (s)',ylabel='manual_yaw reference',title='Log28 window391');axes[0].legend(fontsize=8)
    for ax,seed in zip(axes[1:],[23,42]):
        z=np.load(OUT/f'fold4/oof_aux_seed{seed}.npz');i=int(np.flatnonzero(z['array_index']==391)[0])
        for name in ['factual','held','hold_manual_yaw']:ax.plot(t+.02,z[name][i],label=name)
        ax.plot(t+.02,y['rates_b'][391,:,1].numpy(),color='black',linestyle='--',label='Recorded q')
        ax.set(xlabel='Forecast age (s)',ylabel='q (rad/s)',title=f'Aux seed{seed}: model diagnostics');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'window391.png',dpi=160);plt.close(fig)
    support_orig=sp[(sp['mask']=='factual')&(sp.stratum=='all')]
    lines=['# 参考模型 v20：通道拆解、训练支持与持续q偏差','',
        '**log_28的参考敏感性增长主要来自manual_yaw，且几乎由单个窗口391驱动；该窗口原始参考和保持参考都缺少当前训练分布支持。log_18持续q偏差则未被同一覆盖问题或某个单独参考通道解释。模型与数据均未修改，仍未通过控制验收。**','',
        '## 固定范围与诊断含义','',
        '原407个非重叠父窗口、5外层飞行折×3种子、v8/v18配对冻结模型。原始参考、全部保持，以及每个通道“只允许它变化”/“只保持它”的12种设置均保留；相同参考张量复用预测。保持值是预测起点的首个计划值，历史状态/执行命令、模式等不改。未重训、筛除窗口、访问开发验证或sealed/reserved。',
        '人工组合参考没有对应的真实飞行结果。与日志真值的误差只用于诊断，不能把某个保持方案误差较低当成物理反事实成立；未对各通道RMS直接相加作舵效归因。已保存单通道预测和的非加性残差，避免忽略耦合。yaw_rate参考在这些片段没有变化，因此其保持/单通道对照无差异不能证明真实偏航控制无效。','',
        '## log_28主要是哪一通道','',
        table(high[['seed','mask','versus_held_rmse_baseline','versus_held_rmse_aux','versus_factual_rmse_aux']]),'',
        '表为原28个高参考活动窗口，单位rad/s。全部参考变化相对全部保持的q差异RMS，seed23为0.335→0.541，seed42为0.301→0.695。只允许manual_yaw变化已得到辅助模型0.517/0.672；只保持manual_yaw时降至0.147/0.167，接近基线的0.154/0.166。只固定pitch/roll/thrust均未消除主要增长。结论是模型的manual_yaw路径发生变化，不是真实手动偏航到俯仰气动作用已经被识别。','',
        '## 进一步集中到窗口391','',table(pd.DataFrame(impact)),'',
        '窗口391占log_28参考差异能量净增量约99.3%/101.0%/101.0%（三个种子）；超过100%表示其余窗口合计略有抵消。它在seed23/42辅助模型的全飞行参考差异总能量中占70.4%/80.0%。这细化了v19的“28个高活动窗口”结论，不表示其余27个窗口都有同样问题，也不授权删除391。',
        f"该窗口为Stabilized(mode15)，原始日志支撑{record['start_timestamp_us']/1e6:.6f}–{record['end_timestamp_us']/1e6:.6f}s，预测起点{record['origin_timestamp_us']/1e6:.6f}s（PX4时钟，不是日志文件相对零点）。manual_yaw首值约−0.19048，未来最小约−0.51190。其训练折其他飞行的manual_yaw首值范围约[−0.00238,0.20120]；训练未来轨迹确有负值到−0.45238，不能说训练从未见过负指令，缺口在幅值、起点和条件组合。",'',
        '![窗口391计划参考与模型诊断](window391.png)','',
        '保持参考会把起点−0.19048延续整秒，原始与人工保持方案均需检查支持。图中记录q只对应原始参考，不能用来验证保持方案的误差。','',
        '## 覆盖检查支持什么结论','',
        '使用20个计划参考摘要（每通道首值/范围/相邻变化RMS/净变化）、12个历史摘要。每折仅用4个训练飞行等权计算均值/标准差，计算标准化最近邻距离；阈值为训练窗口到其他训练飞行最近邻距离的等权95分位。参考、历史、联合三组分别报告，查询飞行不进入邻居库；训练边际min/max和1/99分位同时保留。','',
        table(trace[['mask','group','distance','threshold','outside']]),'',
        '391的原参考距离5.233远大于经验阈值0.740，保持参考也为4.127；历史摘要距离0.503则未超阈值0.937。联合距离同样超界。因而不能把这个对照的增长直接称作已经证实的控制响应退化。原冻结保护检查仍失败，不事后剔除此窗口改判通过。',
        '全飞行概况：','',table(support_orig[['log_id','group','windows','outside_fraction','median_distance','threshold']]),'',
        'log_28原参考/历史/联合分别8/81、22/81、12/81超经验阈值；高活动28窗分别8、10、10个。阈值只是选定摘要下的经验诊断，不是覆盖概率、可辨识性或物理安全范围。摘要会遗漏时序细节、模式和其他状态；未超界不能证明输入充分、模型准确或控制可用。人工组合支持也随所有mask保存，不假定单通道对照处于训练分布内。','',
        '## log_18偏差没有被同样原因解释','',
        table(log18[['mask','signed_bias_baseline','signed_bias_aux','bias_rmse_baseline','bias_rmse_aux','rmse_baseline','rmse_aux']]),'',
        'seed23的原始参考下，窗口平均q误差的RMS从0.1013升到0.1372rad/s，但全飞行带符号平均误差约0.00392→−0.00067，说明局部正负偏差抵消，不能统一减去一个常量。全部参考保持时偏差RMS仍为0.1027→0.1317，固定任一个通道也没有去除该差距；这些是对同一真实日志的模型诊断，不是保持参考的真实误差估计。',
        'log_18的78个原窗口在三组距离筛查中均未超经验阈值。这并不证明覆盖充分，但当前证据不支持把它简单归为log_28式的输入外推。对预先声明的32个输入摘要与“辅助−基线的窗口偏差平方变化”做逐飞行Spearman，seed23最大绝对值仅0.266，其他种子不一致；全部相关及常量特征空值已保留。没有据此训练阈值分类器或选删困难窗口。','',
        '## 验证与下一步','',
        '3项针对性测试通过；60组模型/队列的所有mask首步一致性检查、15组候选独立重载、15组输入支持标定与距离复算、查询飞行排除于最近邻库、407个OOF唯一性及v19原参考敏感性复现通过。汇总阶段修复了pandas的mask列与同名方法冲突，用显式列索引重跑；原始模型推理、权重、分层和实验判据未改变。',
        'review_windows.csv保留log_18和log_28全部相关窗口及所有通道对照；window391.json提供准确原日志位置。未生成新的删除集合。',
        '下一步优先回查窗口391的原始manual_control_setpoint、飞行模式和执行命令时间链，确认manual_yaw的含义、采样/保持语义及实际执行变化。只有核实这条输入链，才决定为支持不足的通道增加适用域限制还是修改表示；不应直接把manual_yaw清零来制造通过。log_18应作为独立的持续偏差问题继续研究，不能认为解释391就已修复整体模型。该原日志复核尚未执行，保留A1600和原验收失败结论。','']
    (OUT/'report_zh.md').write_text('\n'.join(lines))
    code=[base.ROOT/p for p in ['scripts/reference_channels_v20.py','scripts/run_reference_channels_v20.py','scripts/summarize_reference_channels_v20.py','scripts/report_reference_channels_v20.py','tests/test_reference_channels_v20.py']]
    base.write(OUT/'delivery_manifest.json',dict(artifacts={str(p.relative_to(base.ROOT)):base.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'},code={str(p.relative_to(base.ROOT)):base.sha(p) for p in code}))
    verify_manifest(OUT/'delivery_manifest.json',['artifacts','code']);print('v20 report and delivery complete')


if __name__=='__main__':main()
