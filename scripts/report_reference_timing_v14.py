"""Report fixed-support timing evidence and limits, without model promotion."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_control_windows_v2 as base
from run_reference_timing_v14 import OUT,V8,V10,V12,V13
from run_reference_generalization_v7 import verify_manifest
from report_reference_generalization_v7 import table


def main():
    verify_manifest(OUT/'protocol.json',['source_pins'])
    amendment=json.loads((OUT/'audit_amendment.json').read_text());assert base.sha(base.ROOT/'scripts/audit_reference_timing_v14_corrected.py')==amendment['corrected_script_sha256']
    for src in [V8,V10,V12,V13]:verify_manifest(src/'delivery_manifest.json',['artifacts','code'])
    audit=json.loads((OUT/'source_timing_audit.json').read_text())
    for log,src in audit['source_logs'].items():assert base.sha(src['path'])==src['sha256']
    d=pd.read_csv(OUT/'block_drift_per_window.csv');d['abs_drift_s']=abs(d.lag_drift_s);d['abs_early_s']=abs(d.early_lag_s);d['abs_late_s']=abs(d.late_lag_s)
    pf=d.groupby(['cohort','axis','stratum','seed','log_id']).mean(numeric_only=True).reset_index();pf.to_csv(OUT/'block_drift_per_flight.csv',index=False)
    s=pf.groupby(['cohort','axis','stratum']).mean(numeric_only=True).reset_index();s['shared_oracle_mse_gain']=1-s.shared_oracle_mse/s.zero_mse;s['separate_oracle_mse_gain']=1-s.separate_oracle_mse/s.zero_mse;s.to_csv(OUT/'block_drift_summary.csv',index=False)
    fixed=pd.read_csv(OUT/'fixed_lag_per_window.csv');f=fixed.groupby(['cohort','axis','stratum','seed','log_id'])[['zero_mse','fixed_mse']].mean().pow(.5).groupby(['cohort','axis','stratum']).mean().reset_index();f['rmse_gain']=1-f.fixed_mse/f.zero_mse;f.to_csv(OUT/'fixed_lag_summary.csv',index=False)
    choice=pd.read_csv(OUT/'fit_lag_choices.csv');choice.groupby(['axis','stratum']).lag_s.agg(['min','median','max']).to_csv(OUT/'fit_lag_summary.csv')
    original=pd.read_csv(OUT/'fullprefix_oracles.csv');op=original.groupby(['cohort','axis','horizon_s','stratum','seed','log_id']).mean(numeric_only=True).groupby(['cohort','axis','horizon_s','stratum']).mean(numeric_only=True).reset_index();op['gain_only_mse_gain']=1-op.zero_lag_gain_mse/op.core_centered_mse;op['lag_gain_mse_gain']=1-op.best_lag_gain_mse/op.core_centered_mse;op.to_csv(OUT/'fullprefix_oracle_summary.csv',index=False)
    assign=pd.read_csv(V12/'strata.csv');assign=assign[(assign.cohort=='oof')&(assign.axis=='q')&(assign.condition=='frequency_mean')][['array_index','stratum']]
    phase=pd.read_csv(OUT/'phase_per_window.csv').merge(assign,on='array_index',validate='one_to_one');phasecols=[f'phase_error_{k}' for k in [12,25,37,50]]
    for k in phasecols:phase['abs_'+k]=abs(phase[k])
    ph=phase.groupby(['log_id','stratum'])[phasecols+['abs_'+c for c in phasecols]].mean().groupby('stratum').mean().reset_index();ph.to_csv(OUT/'phase_summary.csv',index=False)
    dd=d[d.cohort=='oof'].merge(phase[['array_index','late_minus_early_phase_error']],on='array_index',validate='many_to_one');dd['abs_phase_drift']=abs(dd.late_minus_early_phase_error)
    association=[]
    for (axis,seed,log),g in dd.groupby(['axis','seed','log_id']):
        association.append(dict(axis=axis,seed=seed,log_id=log,abs_drift_spearman=float(g[['abs_drift_s','abs_phase_drift']].corr(method='spearman').iloc[0,1]),signed_drift_spearman=float(g[['lag_drift_s','late_minus_early_phase_error']].corr(method='spearman').iloc[0,1])))
    assoc=pd.DataFrame(association);assoc.to_csv(OUT/'phase_lag_association.csv',index=False)
    aa=assoc.groupby('axis')[['abs_drift_spearman','signed_drift_spearman']].agg(['mean','min','max']);aa.to_csv(OUT/'phase_lag_association_summary.csv')
    timing=pd.read_csv(OUT/'source_timing_per_flight.csv');ts=timing.groupby('topic').mean(numeric_only=True).reset_index();ts.to_csv(OUT/'source_timing_summary.csv',index=False)
    tw=pd.read_csv(OUT/'source_timing_per_window.csv').merge(assign,on='array_index',validate='many_to_one');tc=tw.groupby(['topic','stratum','log_id'])[['mean_age_ms','last_model_history_age_at_origin_ms']].mean().groupby(['topic','stratum']).mean().reset_index();tc.to_csv(OUT/'source_timing_by_frequency.csv',index=False)
    tests=Path('/tmp/reference-v14-tests.log').read_text();assert '9 passed' in tests and 'failed' not in tests;(OUT/'tests.txt').write_text(tests)
    verification=json.loads((OUT/'verification.json').read_text());verification.update(focused_tests_passed=9,source_logs_verified=len(audit['checks']),windows_indices_verified=sum(r['windows'] for r in audit['checks']),
        full_source_hash_chain_verified=True,source_values_max_abs=max(r['cache_values_max_abs'] for r in audit['checks']),fixedlag_cases=len(choice),fit_query_disjoint=True,audit_implementation_amendment=amendment)
    base.write(OUT/'verification.json',verification)
    qcorr=float(assoc[assoc.axis=='q'].abs_drift_spearman.mean())
    base.write(OUT/'decision.json',dict(stage_complete=True,all_fit_lag_choices_zero=bool((choice.lag_s==0).all()),fixedshift_oof_gain=float(f[(f.cohort=='oof')&(f.axis=='q')&(f.stratum=='all')].rmse_gain.iloc[0]),
        q_abs_phase_lag_drift_spearman_mean=qcorr,physical_delay_identified=False,no_timestamp_change=True,model_promoted=False,development_evaluated=False,sealed_test_accessed=False))
    shown=s[(s.cohort=='oof')&(s.axis=='q')].copy()
    lines=['# 参考模型 v14：扑频分层的时间偏移与相位失配','',
        '**没有找到可统一修正的固定时移，也没有发现窗口索引或因果取样实现错误。q的逐窗口事后对齐有收益，但偏移不稳定；相位外推误差会增长，却尚不能解释q偏移变化。保留原基线，不改时间戳。**','',
        '## 范围与离线诊断边界','',
        '使用407个原训练窗口、5个整飞行留一折、3个v8 hold种子，以及v12仅由历史平均扑频确定的分层。未重训、未删窗口、未评价739个开发验证窗口，sealed/reserved未访问。OOF已用于多轮研究，不能称盲测。',
        '先复现v10完整0.5/1秒内部支撑的增益/时移诊断，再比较两个等长固定真值块：目标0.12–0.40秒与0.62–0.90秒，各15点；所有候选偏移都使用同一真值点，块内分别去均值。偏移−100到+100ms，每20ms一步；正偏移表示取模型更晚的输出对齐当前真值。',
        '该分辨率不能排除小于20ms的延迟。短块只有约一个扑翼周期，存在波形多义性；可见边界比例与相邻候选MSE差小于零偏移MSE的1%比例。事后偏移不是执行器时延，也不是可部署校准；正偏移还需要更晚的模拟输出。','',
        '## 从训练飞行选择的固定偏移','',
        '每折只用4个拟合飞行，在两个固定块上以飞行等权MSE选择一个单位增益偏移，再原样评价第5飞行；同时比较全工况及各扑频层。',
        f'全部{len(choice)}组（5折×3种子×p/q×全体及3层）都选择0ms。固定偏移的OOF收益为0，当前搜索范围/分辨率不支持公共时间平移。',
        '固定偏移比较使用相同30个真值采样点，表内两误差列已开方，为去均值RMSE（rad/s）：','',table(f[f.cohort=='oof'][['axis','stratum','zero_mse','fixed_mse','rmse_gain']]),'',
        '## 逐窗口偏移为何看起来有效','',
        '以下oracle读取查询窗口真值：一项为每窗口两个块共用一个最优偏移，另一项为两个块分别选择偏移；两者都固定单位增益，收益是相同支撑的去均值MSE减少比例，不是完整状态RMSE或可实现模型收益。','',
        table(shown[['stratum','abs_early_s','abs_late_s','abs_drift_s','same_lag','boundary','ambiguous','shared_oracle_mse_gain','separate_oracle_mse_gain']]),'',
        'q低频OOF的前/后块平均绝对偏移约17.1/18.7ms，前后偏移差绝对值约17.4ms；高频约5.3/11.4ms，差约8.1ms。低频窗口中前后同偏移约52.4%，边界命中约7.0%。带符号偏移会正负抵消，不能只按均值推断没有局部失配。',
        'q逐窗口共用偏移oracle解释约34%–37%的块内去均值MSE，独立分块约43%–44%；但从训练集学到的统一偏移仍为0。这说明局部对齐机会不等于可迁移的固定校准；独立块更灵活，本来就更容易拟合未来真值。',
        'p低频OOF前后偏移差绝对值约44.5ms，边界约23.7%，近似多解约34.5%；其对齐结果更不稳定，不支持对p套用一个固定延迟。v10全前缀结果已精确复现，分层表见fullprefix_oracle_summary.csv；不能把本轮两块MSE收益与旧完整前缀收益直接混比。','',
        '## 扑翼相位外推与偏移变化','',
        '仅用于离线诊断：比较缓存的展开相对相位与“保持起点扑频”积分相位；物理零点未确认，仍使用September ratio7.909090995788574，不引入April ratio8契约。下表为各飞行等权平均绝对相位差（rad），时刻为0.24/0.50/0.74/1.00秒。','',
        table(ph[['stratum']+['abs_'+c for c in phasecols]]),'',
        '低频层绝对相位差由约0.338rad增加到1.306rad，高频由约0.190增加到0.870rad。保持起点扑频会积累误差，但日志相位并非独立机械真值，本轮也没有确认频率估计波动的成因。',
        f'q每飞行/种子的“前后块相位误差变化绝对值”与“波形偏移变化绝对值”的Spearman平均仅{qcorr:.3f}，方向不一致；完整15项见phase_lag_association.csv。不能从两种误差都存在，就断定相位外推造成q波形失配。此前v7/v8改善相位预测未改善状态的结果仍需保留，不能据此重复宣称相位修正已获支持。','',
        '## 原始日志时间链核对','',
        '重新读取5个原训练ULog的所需消息，核对来源哈希，按max(publication timestamp, sample timestamp)≤网格时间选择最近消息。逐值复核角速度、姿态、位置速度、扑翼相位/频率、最终执行命令、姿态参考及manual yaw，再复算407个窗口所有模型输入和标签，均通过。',
        '每飞行中位数再等权平均的消息时间统计（ms）：','',table(ts[['topic','has_timestamp_sample','pub_age_median_ms','sample_age_median_ms','age_p95_ms','publication_period_median_ms','repeated_hold_fraction']]),'',
        'wing_phase与姿态参考没有独立timestamp_sample，表中相应sample age只是publication age回退，不能当成物理测量年龄。对其他话题，消息中记录的sample timestamp也不能替代独立硬件时延标定。',
        '角速度已知消息年龄中位约6.5ms、姿态约17.3ms、速度约14.1ms；相位发布年龄约5.1ms、参考发布年龄约14.5ms。执行命令缓存年龄约16.4ms，模型历史最后一项另有刻意的20ms索引偏移，不能误称同步舵角；manual原始发布周期约200ms，在50Hz网格上大部分时刻是保持值。',
        '这些消息不同步，但本轮未发现“未来消息倒灌”或错一格的实现错误。数据契约明确为历史状态/参考origin−25..origin、执行命令origin−26..origin−1、计划参考origin..origin49、目标origin+1..origin50。物理执行器到位时间与气动时延不在这些日志字段里，不能用发布间隔冒充。',
        '审计实现首次误把祖先manifest哈希与其子manifest直接比较，已按原有parent链修正并重跑；原脚本冻结保留，更正说明保存在audit_amendment.json，当前时间审计入口为scripts/audit_reference_timing_v14_corrected.py。该错误属于本轮审计代码，不是源日志或数据发生变化，未修改任何实验口径。','',
        '## 结论与下一步','',
        '本轮排除了当前20ms网格/±100ms范围内简单公共时间平移的收益，没有证明所有时序问题不存在；也没有建立相位误差导致q失配的因果证据。保留原基线，不添加固定延迟或事后分块对齐。',
        '下一步更适合在同一OOF模型上做短历史刷新诊断：对比完整1秒自主预测与每0.1/0.2秒用真实过去历史重新初始化的局部预测，按相同扑频层报告误差随预测年龄的变化。它可以区分持续传播与局部预测/初始状态表示的问题；刷新依赖后续观测，只是离线定位，不能作为无需观测的1秒模型成绩。出现可重复结果后再决定是否研究状态估计或递推结构，避免继续盲扫权重、相位或固定偏移。',
        '9项测试、v10数值复现、5份ULog哈希和消息因果时间、407窗口重提取及新产物哈希检查通过。报告、逐窗口曲线、逐飞行统计与source_timing_audit.json完整保留；没有模型晋升或控制部署。','']
    (OUT/'report_zh.md').write_text('\n'.join(lines))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5));xx=np.arange(3);a=shown.set_index('stratum').loc[['low','mid','high']]
    axes[0].bar(xx-.18,a.abs_early_s*1000,.36,label='Early core');axes[0].bar(xx+.18,a.abs_late_s*1000,.36,label='Late core');axes[0].set_xticks(xx,['Low freq','Middle','High freq']);axes[0].set(ylabel='Mean absolute oracle lag (ms)',title='q OOF: local alignment, not physical delay');axes[0].legend()
    for st in ['low','mid','high']:
        vals=ph.set_index('stratum').loc[st,['abs_'+c for c in phasecols]].to_numpy(dtype=float);axes[1].plot([.24,.5,.74,1.],vals,'o-',label=st)
    axes[1].set(xlabel='Prediction age (s)',ylabel='Absolute relative phase error (rad)',title='Held frequency vs logged phase');axes[1].legend()
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'timing_phase_comparison.png',dpi=150);plt.close(fig)
    code=list((base.ROOT/'scripts').glob('*timing_v14*.py'))+[base.ROOT/'tests/test_reference_timing_v14.py']
    base.write(OUT/'delivery_manifest.json',dict(status='complete',code={str(p.relative_to(base.ROOT)):base.sha(p) for p in code},artifacts={str(p.relative_to(base.ROOT)):base.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'},
        focused_tests_passed=9,model_promoted=False,development_evaluated=False,sealed_test_accessed=False))
    verify_manifest(OUT/'delivery_manifest.json',['artifacts','code']);print(json.dumps(dict(status='complete',q_phase_lag_drift_spearman=qcorr,artifact_count=len(json.loads((OUT/'delivery_manifest.json').read_text())['artifacts'])),indent=2))

if __name__=='__main__':main()
