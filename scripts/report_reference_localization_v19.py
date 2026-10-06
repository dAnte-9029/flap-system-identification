"""Report v19 localization without treating algebraic substitutions as models."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_control_windows_v2 as base
from run_reference_localization_v19 import OUT,V18
from reference_localization_v19 import BLOCKS
from run_reference_generalization_v7 import verify_manifest
from report_reference_generalization_v7 import table


def main():
    verify_manifest(OUT/'protocol.json',['source_pins'])
    q=pd.read_csv(OUT/'q_decomposition_summary.csv');af=pd.read_csv(OUT/'attitude_attribution_per_flight.csv')
    att=pd.read_csv(OUT/'attitude_attribution_summary.csv');rp=pd.read_csv(OUT/'reference_oof_pairs.csv')
    qf=pd.read_csv(OUT/'q_decomposition_per_flight.csv');v=json.loads((OUT/'verification.json').read_text())
    blocks=[b for b in BLOCKS if b!='full1s'];qoo=q[q.cohort=='oof']
    paired=[]
    for block in blocks+['full1s']:
        g=qoo[qoo.block==block].set_index('arm');start,end=BLOCKS[block]
        delta={c:float(g.loc['aux',c]-g.loc['baseline',c]) for c in ['total_mse','bias_mse','amplitude_mse','shape_mse']}
        paired.append(dict(block=block,baseline_rmse=g.loc['baseline','rmse'],aux_rmse=g.loc['aux','rmse'],gain=1-g.loc['aux','rmse']/g.loc['baseline','rmse'],
            weighted_mse_delta=delta['total_mse']*(end-start)/50,**delta))
    qp=pd.DataFrame(paired);qp.to_csv(OUT/'q_block_pairs.csv',index=False)
    focus_att=af[(af.cohort=='oof')&(af.seed==23)&(af.kind=='axis')&(af.horizon_s==1)]
    log18=focus_att[focus_att.log_id.str.contains('log_18_')].iloc[0]
    late18=af[(af.cohort=='oof')&(af.seed==23)&(af.kind=='time')&(af.horizon_s==1)&af.log_id.str.contains('log_18_')].iloc[0]
    focus_ref=rp[(rp.stratum=='all')&(rp.block=='full1s')&rp.seed.isin([23,42])]
    loc=rp[rp.log_id.str.contains('log_28_')&rp.seed.isin([23,42])]
    refshares={str(seed):float(g[g.log_id.str.contains('log_28_')].delta_effect_mse.iloc[0]/g.delta_effect_mse.sum()) for seed,g in focus_ref.groupby('seed')}
    # Review index retains every window of the implicated flights, all strata.
    raw=pd.read_csv(OUT/'reference_per_window.csv');raw=raw[(raw.cohort=='oof')&(raw.block=='full1s')]
    chosen=raw[((raw.seed==23)&raw.log_id.str.contains('log_18_'))|(raw.seed.isin([23,42])&raw.log_id.str.contains('log_28_'))].copy()
    chosen['scope']='posthoc diagnostic review only; original training/evaluation windows unchanged'
    chosen.to_csv(OUT/'review_windows.csv',index=False)
    # Compare source and derived artifact consistency before writing delivery.
    old=pd.read_csv(V18/'per_flight.csv')
    for horizon in [.5,1.]:
        now=af[(af.cohort=='oof')&(af.kind=='axis')&(af.horizon_s==horizon)]
        for arm in ['baseline','aux']:
            prior=old[(old.cohort=='oof')&(old.kind=='prefix')&(old.horizon_s==horizon)&(old.arm==arm)]
            merged=now.merge(prior,on=['fold','seed','log_id'],validate='one_to_one')
            assert abs(merged[arm+'_rmse']-merged.attitude).max()<2e-6
    tests=Path('/tmp/reference-v19-tests.log').read_text();assert '5 passed' in tests and 'failed' not in tests
    (OUT/'tests.txt').write_text(tests)
    v.update(focused_tests_passed=5,original_attitude_per_flight_reproduced=True,max_quaternion_reconstruction_angle=max(c['quaternion_max_angle'] for c in v['integration_and_reference_checks']))
    base.write(OUT/'verification.json',v)
    decision=dict(stage_complete=True,diagnostic_only=True,no_training=True,model_promoted=False,control_validated=False,
        seed23_log18_net_attitude_mse_share=float(log18.delta_mse/focus_att.delta_mse.sum()),
        seed23_log18_q_tape_share=float(log18.q/log18.delta_mse),seed23_log18_late_tape_share=float(late18.late_120_1000ms/late18.delta_mse),
        reference_delta_energy_log28_share=refshares,development_evaluated=False,sealed_test_accessed=False)
    base.write(OUT/'decision.json',decision)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for arm,label in [('baseline','Baseline'),('aux','Auxiliary q')]:
        g=qoo[qoo.arm==arm].set_index('block').loc[blocks];axes[0].plot(range(5),g.rmse,marker='o',label=label)
    labels=['20–40','60–100','120–200','220–500','520–1000']
    axes[0].set(xticks=range(5),xticklabels=labels,ylabel='q block RMSE (rad/s)',xlabel='Forecast block (ms)',title='Equal-flight OOF');axes[0].tick_params(axis='x',rotation=25);axes[0].legend(fontsize=8)
    aa=att[(att.cohort=='oof')&(att.kind=='axis')&(att.horizon_s==1)].iloc[0]
    axes[1].bar(['p','q','r'],[aa.p,aa.q,aa.r]);axes[1].set(title='Offline rate-tape attribution',ylabel='1s attitude MSE change (rad²)')
    for seed in [23,42]:
        g=loc[(loc.seed==seed)&(loc.stratum=='high')].set_index('block').loc[blocks]
        axes[2].plot(range(5),g.effect_ratio,marker='o',label=f'Seed {seed}')
    axes[2].axhline(1,color='gray',lw=1);axes[2].set(xticks=range(5),xticklabels=labels,xlabel='Forecast block (ms)',ylabel='Aux/baseline reference-effect RMS',title='Log28 high planned-ref activity');axes[2].tick_params(axis='x',rotation=25);axes[2].legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'localization.png',dpi=160);plt.close(fig)
    lines=['# 参考模型 v19：局部收益、姿态与参考敏感性定位','',
        '**q首步收益被后半秒新增误差抵消；姿态保护失败主要定位到seed23的log_18，参考敏感性增长主要定位到log_28的高参考变化层。独立积分与模型输出一致，未发现积分实现错误。此次仅诊断，不改变v18未晋升的决定。**','',
        '## 范围与方法','',
        '原407个非重叠父窗口、5外层飞行折、3种子，冻结v8 hold与v18辅助损失模型；fit和OOF均保留。固定时间段20–40、60–100、120–200、220–500、520–1000ms，另报完整1秒。没有重训、调权重、增加起点或按误差删除窗口，未评价开发验证、未访问sealed/reserved。OOF已多次研究，不称盲测。',
        '先对每窗口做精确MSE分解，再在每飞行/模型平均，fit重复模型按源飞行平均后等权汇总。归因在MSE空间保持可加性；报告的RMSE仍先逐飞行求根再平均，所以MSE贡献比例不是RMSE下降比例。短到2点的时间块可以作代数分解，但不能据此识别频率或物理时延。','',
        '## q收益为何没有保留到1秒','',table(qp),'',
        '表中total_mse/bias_mse/amplitude_mse/shape_mse为“辅助模型−基线”的变化，负值是改善；weighted_mse_delta按该时间块在50个目标点中的占比加权。20–40ms块RMSE 0.3204→0.2313，早期2点的MSE改善对全时域贡献约−0.00193(rad/s)²；520–1000ms的25点反而贡献约+0.00467，足以抵消前段收益。',
        '完整1秒新增MSE约0.00209，其中窗口均值偏差项增加约0.00193（约92%净增量）；这描述的是新增误差，不是原有总误差的92%。原有总体误差仍主要包含波形项。后半秒同时出现均值偏差和波形项增加，不能只按预测幅值放大或把它称为统一时延。',
        '偏差项是各窗口内的平均误差平方，不代表全数据有一个可以统一减去的常量；不同窗口会正负抵消。完整逐飞行和逐窗口分解保存在q_decomposition_per_flight.csv及q_decomposition_per_window.csv。','',
        '## 姿态积分与误差来源','',
        f"独立numpy float64中点积分从相同观测初值出发，重建全部60组模型/队列的姿态；与保存四元数的最大角距离为{v['max_quaternion_reconstruction_angle']:.3g}rad。因此没有发现姿态更新实现或索引不一致足以解释这次退化。用未来记录角速度积分的1秒姿态一致性RMSE约0.0212rad，仍有采样/估计链差异；它是使用未来真值的离线一致性检查，不能当成部署预测或不可约误差下界。",
        '将基线/辅助模型的预测角速度序列做全部8种p/q/r组合后重新积分，按Shapley平均分配非线性相互作用；再对前5点（20–100ms）和后45点做4种组合。仅替换已生成序列，不反馈到GRU，不构成可运行混合模型或网络因果机制识别。每次分配之和严格等于辅助与基线的姿态MSE差。','',
        table(focus_att[['log_id','baseline_rmse','aux_rmse','delta_mse','p','q','r','rmse_gain']]),'',
        'seed23的log_18完整1秒姿态RMSE 0.1127→0.1413rad（退化25.36%），贡献该种子5飞行净姿态MSE增量约91.4%。在这个飞行内，q序列归因约83.9%，后45个端点归因约86.2%。轴与时间是两种分别完备的分配，百分比不能相加。这说明后续q预测与该保护失败密切相关，不能仅归咎于修正了首20ms。',
        '全种子等权的1秒姿态新增MSE约0.000706rad²，q序列贡献约0.000585；后45个端点贡献约0.000589。各飞行也可能出现负贡献抵消，完整表保留所有种子，未只看seed23。','',
        '## 参考敏感性增长集中在哪里','',
        '仍比较原计划参考与所有5通道保持第一个计划值的预测。按每折训练归一化，把完整计划参考的相邻变化RMS分为低/中/高层，阈值只由该折4个fit飞行的等权三分位确定，OOF不参与定界。该分层使用已知完整计划，不使用未来运动结果；是离线描述，不能冒充只知道当前信息的在线可信度判据。','',
        table(focus_ref[['seed','log_id','effect_ratio','baseline_effect','aux_effect','delta_effect_mse','utility_change']]),'',
        'log_28在seed23/42的参考输出差RMS分别增至基线1.378/1.789倍；对应五飞行等权的差异能量净增量约79.0%/93.1%来自这个飞行。这里的能量分配是平方RMS空间，不是原参考保护中平均RMS比例的百分比分解。',
        'log_28共81窗口，高参考变化层28个、中层31个、低层22个，未丢弃任一层。主要放大在高层和后半秒：520–1000ms高层差异RMS比为seed23的1.643和seed42的2.429；中低层的变化较小，部分抵消。全数据分层每飞行/模型最小16窗口，计数见reference_coverage.csv。','',
        table(loc[(loc.stratum=='high')][['seed','block','windows','baseline_effect','aux_effect','effect_ratio','utility_change']]),'',
        '高层包含多种参考通道变化；本轮没有区分是哪一通道触发，也没有证明真实飞机需要更强或更弱的控制作用。对同一真值评价保持参考是模型诊断，日志里没有对应的真实反事实飞行。尤其首20ms两条预测的参考输入相同，首步精度改善不能直接证明未来参考变化的控制响应更真实。','',
        '![诊断图](localization.png)','',
        '## 可追溯片段与验证','',
        'review_windows.csv保留seed23/log_18的全部78个窗口，以及seed23/42/log_28的全部81个窗口和所有参考层，包含window_id、array_index、origin_index、原始起止timestamp_us、模式和配对指标。时间戳沿用PX4日志时钟，不转换成文件开头的相对秒数。该表只是事后复核索引，没有生成新的训练选择或删除集合；全部407窗口仍在原逐窗口表中。',
        '5项测试覆盖积分方向/初值/符号、含相互作用的精确分配、时间替换与输入活动度。60组独立姿态重建、15组v18参考保持预测复现、原参考指标与原姿态逐飞行指标复现、407个OOF唯一性、时间块MSE加权恒等式、Shapley可加性、后段替换不改变前100ms及fit分层阈值复算均通过。','',
        '## 下一步','',
        '继续保留A1600，v18仍是局部q改善候选。下一项应先把log_28高参考变化窗口的参考保持对照拆到滚转、俯仰、推力等单独通道，并核对它们在该折训练飞行中的支持范围；同时检查log_18的持续q偏差与参考/历史条件是否有关。所有窗口共同评价，局部对照只用于定位，不把这些飞行作为应当删除的异常。',
        '这样才能判断后续应约束长时均值、参考敏感性，还是处理输入支持与泛化。现在直接降低辅助权重、拼接预测或删掉这两个飞行，都不能算完成控制模型验收。该通道拆解尚未执行，本轮也没有证明必须补飞。','']
    (OUT/'report_zh.md').write_text('\n'.join(lines))
    code=[base.ROOT/p for p in ['scripts/reference_localization_v19.py','scripts/run_reference_localization_v19.py','scripts/summarize_reference_localization_v19.py','scripts/report_reference_localization_v19.py','tests/test_reference_localization_v19.py']]
    base.write(OUT/'delivery_manifest.json',dict(artifacts={str(p.relative_to(base.ROOT)):base.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'},code={str(p.relative_to(base.ROOT)):base.sha(p) for p in code}))
    verify_manifest(OUT/'delivery_manifest.json',['artifacts','code'])
    print('v19 report and delivery verification complete')


if __name__=='__main__':main()
