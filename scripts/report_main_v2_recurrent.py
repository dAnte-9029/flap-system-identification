"""Evidence-bound Step 7 report generation; no fitting or inference."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def md(df):
    def value(v):return f'{v:.5f}' if isinstance(v,(float,np.floating)) else str(v)
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(value(v) for v in r)+' |' for r in df.itertuples(index=False,name=None))+'\n'


def report(out):
    def read(f):return pd.read_csv(out/f)
    s=read('rollout_mode_comparison.csv');ix=s.set_index(['mode','horizon_s'])
    hd=read('hidden_distance_vs_horizon.csv');hix=hd.set_index(['mode','horizon_s'])
    comparison=read('reencode_comparison.csv');paired=read('paired_flight_changes.csv')
    variation=read('variation_summary.csv');fail=read('failures.csv')
    hidden=read('hidden_ood.csv');physical=read('physical_ood.csv');probe=read('hidden_probe_metrics.csv')
    jac=read('recurrent_jacobian_summary.csv');hs=read('hidden_output_sensitivity.csv');xs=read('state_output_sensitivity.csv')
    manifest=json.loads((out/'manifest.json').read_text());reference=json.loads((out/'reference_contract.json').read_text())
    b_gap=max(abs(ix.loc[('B',h),'angular_acceleration_equal_log_rmse']/ix.loc[('A_fixed',h),'angular_acceleration_equal_log_rmse']-1) for h in [.2,.5,1.,2.,3.,5.])
    long=comparison[comparison.horizon_s.isin([2,3,5])&comparison['mode'].isin(['R13','R26'])]
    joint=any((g.velocity_m_s_change_pct<0).all() and (g.attitude_deg_change_pct<0).all() for _,g in long.groupby('mode'))
    if b_gap<.01 and hix.loc[('D',5.),'l2_median']>1 and not joint:result='R-B'
    else:raise RuntimeError('Evidence differs from reviewed Step 7 conclusion; report requires review')
    def tab(df):return '\n'+md(df)+'\n'
    text=f'''# Step 7 — Recurrent-State Drift and Transition Representation Audit

**结论 {result}：hidden 与真实轨迹对应 representation 明显失配，但没有发现它在真实 physical input 下独立累积退化；predicted-history re-encoding 未改善完整 autonomous simulator。** 下一步优先调查 **transition function / physical-state representation**，不据此增加 GRU、latent、Transformer 或 TCN。

原 Main V2 checkpoint 全程冻结；本轮不训练 dynamics 模型。仅训练一个固定 ridge=1 的小型线性信息 decoder，PCA/OOD/decoder 均只拟合 train。sealed test 未打开，未启动 RL，未修改 Step 1–6 或任何旧 checkpoint。

## 合同与对照口径

722 个共同 validation 起点、五条 flight、warm26、同 commands、250 个原生 dt。dt范围 {manifest['dt_min']:.6f}–{manifest['dt_max']:.6f}s，均值 {manifest['dt_mean']:.8f}s；250步实际时长 {manifest['duration_min']:.6f}–{manifest['duration_max']:.6f}s。0.2/0.5/1/2/3/5为10/25/50/100/150/250步的名义标签，积分没有改为固定20ms。

- **A_native（Full Teacher，NOT DEPLOYABLE）**：每t真实physical + 原始26点history重新以当前t归零编码，只预测一步。它是正式reset语义下的局部参考。
- **A_fixed（匹配anchor的Teacher控制，NOT DEPLOYABLE）**：同样真实history/physical，但history及当前feature均保持原t0 anchor，用于和B/C/D隔离hidden更新差异。
- **B（NOT DEPLOYABLE）**：只在t0编码；每t用真实physical输出一步，随后用 **真实x_(t+1)** 更新GRU。不能先用预测x更新GRU再仅覆盖physical，那不是本问题的B。
- **C（NOT DEPLOYABLE）**：physical自主递推，每t换入A_fixed hidden；anchor保持t0。另报C_native：原reset重锚teacher hidden并匹配当前teacher anchor，避免把两种坐标合同混为一谈。
- **D**：原始自主simulator；与旧S0逐字段重跑校验见baseline_parity.json。
- **R13/R26（autonomous frozen-weight probes）**：相同warm26 h0，此后每一步用最近13/26个自身预测feature重编码。保持原t0 anchor、actuator代理、weights、integration。早期buffer包含合法t0前真实history，t0之后只append预测；完全替换后是纯预测history。

所有模式的actuator proxies都由同一t0 history初始化并沿commands递推，不每步重置。B的teacher physical包含frequency/encoder phase，因此它同时阻断这些预测误差，不能把B结果仅归于velocity/body-rate。A/B报告的是各时点**一次native-step预测误差**，不是5s累计轨迹；C/D/R才是连续预测。导数指标取每horizon末10个native转移的向量RMSE，各flight先聚合再等权；endpoint同Step2合同。raw差分仍含快速变化，本轮不将其全部认定为独立物理真值。

teacher、free、reencode 的history控制对齐显式记录：legacy GRU更新为G(h_t,feature(x_(t+1)),u_t)，history encoder原样读取各历史row的command。当前base.use_controls=False，因此这一索引差不影响hidden，但actuator路径没有忽略command。

## Q1：Full Teacher一步预测随时间稳定吗？

**YES，指没有随rollout elapsed time累积恶化，不表示局部误差小。** A_native及A_fixed各horizon导数误差基本平稳；raw angular error仍约26 rad/s²。B与A应比较单步endpoint，而非把它们的小数误读为5秒准确轨迹。
'''
    cols=['mode','horizon_s','velocity_m_s_equal_log_rmse','attitude_deg_equal_log_rmse','body_rate_rad_s_equal_log_rmse','linear_acceleration_equal_log_rmse','angular_acceleration_equal_log_rmse']
    text+=tab(s[s['mode'].isin(['A_native','A_fixed','B'])][cols])
    times=read('teacher_flight_time_bins.csv')
    text+='另按真实flight timestamp分为五个等时长桶，先按(log,segment,sample)去除重叠起点造成的重复。不同regime下误差有变化，不能宣称飞行过程严格平稳；完整25桶见teacher_flight_time_bins.csv。\n'
    text+=tab(times.groupby('log_id')[['linear_rmse','angular_rmse']].agg(['min','max']).reset_index().pipe(lambda d:d.set_axis(['log_id','linear_min','linear_max','angular_min','angular_max'],axis=1)))
    text+=f'''## Q2：Teacher physical + recurrent hidden退化吗？

**NO，未观察到实质累积退化。** B相对A_fixed的angular RMSE最大绝对变化仅{100*b_gap:.4f}%。到5s，B与teacher hidden的L2中位数{hix.loc[('B',5.),'l2_median']:.4f}、cosine {hix.loc[('B',5.),'cosine_median']:.6f}；D对应{hix.loc[('D',5.),'l2_median']:.4f}、{hix.loc[('D',5.),'cosine_median']:.6f}。

B包含从t0开始的长期recurrent memory，A_fixed每次截断26点从零编码，二者不要求hidden完全相等。实际差异小且没有累积，反驳“只要反复GRUCell就会自己漂走”的强假设。
'''
    text+=tab(hd[hd['mode'].isin(['B','D'])&hd.horizon_s.isin([0,.2,.5,1,2,3,5])][['mode','horizon_s','l2_median','dimension_rmse_median','cosine_median']])
    text+='![Hidden distance](hidden_distance_vs_horizon.png)\n\n![Train-only PCA](hidden_pca.png)\n\n'
    text+='''## Q3：Predicted physical + teacher hidden是否明显优于D？

**NO。** C改善raw角加速度误差、恢复variation，但累计velocity/attitude更差；C_native也不改善。teacher hidden与预测physical的组合可能不自洽，这种oracle干预不是保证改善的数学上界，不能把恶化量解释为hidden“有益”的因果百分比。
'''
    text+=tab(s[s['mode'].isin(['C','C_native','D'])][cols])
    text+=tab(comparison[comparison['mode'].isin(['C','C_native'])&comparison.horizon_s.isin([2,3,5])][['mode','horizon_s','velocity_m_s_change_pct','attitude_deg_change_pct']])
    text+='C的5s omega variation prefix/last1s接近1，依然未改善轨迹，是另一项“幅值不是fidelity”的证据。\n'
    text+=tab(variation[variation['mode'].isin(['C','C_native','D'])&(variation.horizon_s==5)][['mode','scope','median']])
    text+='![Modes](mode_error_vs_horizon.png)\n\n'
    text+='''## Q4：hidden OOD与physical OOD谁先出现？

**当前距离阈值不能可靠给出先后顺序。** 必须区分两件事：D远离同一真实轨迹的teacher hidden，但未持续远离train hidden bank；相反kNN距离下降。衰减到平均动态附近，也可能更接近训练点云中心。

原生bank严格以当前t为anchor编码真实train history；native_hidden_ood/native_physical_ood另存。主图使用额外的gauge控制bank：同一真实train history加8个等间隔anchor坐标（不拟合offset、不改simulator），防止单纯phase坐标旋转被误读为动力学OOD。它是train-derived坐标控制，**不是声称原训练看过所有这些gauge**。两种bank均无validation样本。PCA也只fit这些train hidden，validation仅project。

k=5，标准化RMS欧氏距离；Mahalanobis用训练协方差+0.001I，除维数后开根号。原阈值来自train查询排除自身后的p99，有同log时间邻近/采样密度影响；补充 **leave-one-log-out train查询** 校准阈值，只变诊断刻度，不选择模型。所有阈值不是物理安全界限。
'''
    text+=tab(pd.concat([hidden[(hidden['mode'].isin(['teacher','D']))&hidden.horizon_s.isin([0,.2,1,5])].assign(space='hidden'),physical[(physical['mode'].isin(['teacher','D']))&physical.horizon_s.isin([0,.2,1,5])].assign(space='physical')])[['space','mode','horizon_s','knn_median','knn_outside_train_p99_mean','knn_outside_leave_log_p99_mean']])
    text+='初始状态已经存在跨日志距离偏移，不能称其在某个未来horizon“首次离开训练域”。ood_onset.csv的连续三个采样点阈值越界仅作描述，不作为起因排序。更强的诊断是：喂入真实physical的B保持teacher附近，而D明显偏离；支持偏移由预测physical输入所驱动，但不能排除state/hidden非线性反馈。\n'
    corr=read('ood_error_correlations.csv')
    text+='固定horizon、跨722起点的相关性如下；避免仅用共同时间趋势产生相关性。重叠起点不独立，未用该相关性做显著性或因果判断。\n'
    text+=tab(corr[corr.horizon_s==5])
    text+='![OOD](hidden_ood_vs_horizon.png)\n\n![Physical/hidden OOD](physical_vs_hidden_ood.png)\n\n'
    text+='''## Q5：hidden drift对derivative输出有多大影响？

固定真实physical，换入D hidden；反向固定teacher hidden，把physical换成D状态。两者均保留同actuator/command、fixed anchor。下表为各origin末10步输出差向量RMS的中位数，**不是truth误差，也不能相加为因果分解**。linear是NED，linear_body为模型body输出；角加速度单位rad/s²，频率导数Hz/s。
'''
    sens=pd.concat([hs.assign(intervention='hidden_only'),xs.assign(intervention='physical_only')])
    text+=tab(sens[['intervention','horizon_s','linear_median','linear_body_median','angular_median','frequency_median']])
    text+='hidden对角导数确实敏感，不能称hidden“无关”；但敏感性大不等于它是独立起因。B与C的直接干预不支持通过换hidden独立解决长期轨迹。\n\n'
    text+='''## Q6：GRU是否明显memory contraction？

局部64×64 Jacobian直接自动微分，固定输入，48个确定性train-bank点及每flight前2个origin的t=0/1/4.98s teacher/B/free，共138点。validation使用实际下一步feature（teacher为真实下一步，free为预测下一步）；train为记录history终点附近的固定输入局部map，不是训练新模型。IDs和全部谱值保存。
'''
    text+=tab(jac[['mode','horizon_s','n','largest_singular_median','largest_singular_p95','spectral_radius_median','median_singular_median']])
    text+='多数维度局部缩小、spectral radius约0.9，但最大singular通常接近或超过1，不能称所有方向严格contractive；时变Jacobian乘积也不能由单步特征值推断。真实teacher/B与free没有显示后期独有的强收缩。稳定受驱动GRU本来可以衰减旧扰动，同时由输入持续补充信息。B不衰退是关键反证，因此不把这一谱结果判为memory实现bug。\n\n'
    text+='''train-only ridge decoder结果（RMSE按列单位；phase用sin/cos二维、非absolute mechanical phase）：
'''
    text+=tab(probe[probe.horizon_s.isin([.2,1,5])&probe.target.isin(['phase_relative','frequency','body_rate','recent_delta_omega'])][['mode','horizon_s','target','rmse','r2']])
    text+='D丢失对**真实**rate/近期增量的解码能力，B接近teacher。该probe不能区分“忘记真实信息”与“忠实编码已经错误的自身physical”；也不是因果证明。within-log phase跨log zero未知，不作机械相位准确度。command-history decoder在teacher上本来就弱（完整CSV），base.use_controls=False，不将负R²解释为free-run才忘记command。\n\n'
    text+='''## Q7：predicted-history re-encode优于pure recurrence吗？

**NO。** 全722起点结果不支持这条修复路线。诊断训练gate（C长期v/att改善>5%或B angular相对A_fixed退化>5%）未触发；没有训练新representation。R13/R26是用户要求的两个冻结权重negative controls，不是通过gate的候选。
'''
    fullcols=['mode','horizon_s','position_m_equal_log_rmse','velocity_m_s_equal_log_rmse','attitude_deg_equal_log_rmse','body_rate_rad_s_equal_log_rmse','frequency_hz_equal_log_rmse','phase_rad_equal_log_rmse']
    text+=tab(s[s['mode'].isin(['D','R13','R26'])][fullcols])
    text+='相对D的配对flight变化（负数改善）；列举所有5条flight有放回的3125种bootstrap，不能把722重叠起点当独立样本，也不表示跨模型seed置信区间。\n'
    text+=tab(paired[paired['mode'].isin(['R13','R26'])&paired.horizon_s.isin([2,3,5])&paired.metric.isin(['velocity_m_s','attitude_deg'])])
    text+=tab(variation[variation['mode'].isin(['D','R13','R26'])][['mode','horizon_s','scope','median']])
    text+='variation有小幅恢复，但仍远低于1，且长期误差退化；不晋级，不需要用更复杂谱指标为一个未通过轨迹门槛的candidate寻找收益。完整6指标分布、每flight、失败保留在CSV。\n'
    text+=tab(fail[fail['mode'].isin(['D','R13','R26'])])
    text+='support gate沿用Step2 train observed envelope，不是适航安全界限；有限但失败轨迹没有从平均值中删掉。\n\n![Reencode](reencode_free_running.png)\n\n'
    text+='''## Q8：最终 R-B

1. B保持真实physical时，hidden和一步误差都稳定；不存在已证实的“GRU recurrence单独漂移”。
2. D与teacher hidden方向显著失配，hidden替换显著影响角导数；因此不能选择“hidden完全不相关”的强版本R-C。
3. C能恢复部分动态幅值，却恶化累计轨迹；autonomous R13/R26也失败。结合Step4的一步局部映射不足，最有证据的解释是 **physical预测误差改变输入，hidden跟随并反馈，单独换encoding不足以修复transition**。没有证明具体空气动力缺失变量，也没有证明加大网络会解决。

下一步：transition function / physical-state representation的受控调查。停止把GRU重复编码作为当前主要修复方向；不继续constant-offset搜索或增大increment权重。

Simulator structural readiness: **100%**

Dynamics fidelity readiness: **60%**

RL-ready: **NO**

沿用既有证据评分；本轮是失败分解，没有晋级模型，不提高readiness。

## 复现与工程验收

```bash
/home/zn/anaconda3/envs/flap-train-gpu/bin/python /home/zn/flap-system-identification/scripts/run_main_v2_recurrent_audit.py --output /tmp/main-v2-step7/results --artifacts /tmp/main-v2-step7/artifacts --device cuda:1
```

使用空目录。固定seed717；真实历史和oracle trace仅在独立诊断脚本处理，autonomous rolling类没有truth参数。snapshot保存physical/hidden、features、controls、mask；恢复不重新编码。K13/K26各250步与100+150步所有state/buffer逐位相同；future-label NaN回归继续验证。

manifest.json、historical_hashes.json、reference_contract.json、train_bank_rows.csv、operating_points.pt、reference_fit.npz及hidden_probe.npz保存来源、索引和拟合参数。verification.json记录旧结果hash、pytest、git diff检查与墙钟时间。原数据split不变；全部新文件独立保存。
'''
    (out/'report.md').write_text(text)
    summary=dict(result=result,n_origins=int(manifest['n_origins']),training_performed=False,
        max_B_vs_A_angular_relative_gap=b_gap,simulator_structural_readiness=100,dynamics_fidelity_readiness=60,rl_ready=False,
        hidden_distance_5s_median=float(hix.loc[('D',5.),'l2_median']),hidden_cosine_5s_median=float(hix.loc[('D',5.),'cosine_median']),
        C_5s_velocity_change_pct=float(comparison[(comparison['mode']=='C')&(comparison.horizon_s==5)].velocity_m_s_change_pct.iloc[0]),
        C_5s_attitude_change_pct=float(comparison[(comparison['mode']=='C')&(comparison.horizon_s==5)].attitude_deg_change_pct.iloc[0]),
        verdict_scope='S0 frozen Main V2; no claim about all recurrent architectures or arbitrary rolling cadence')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
