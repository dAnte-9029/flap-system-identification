"""Compact report and two fixed representative figures, no parameter selection."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def conclusion():
    return '''## 结论与方法建议

1. **vD、q 均建议 F 作为主候选。LSF没有同趋势预测优势。** F在train/validation两个通道均降低航次等权MSE；validation的F/L RMSE分别为vD 0.107184/0.112704 m/s、q 0.211514/0.223155 rad/s。LSF相对F的MSE反而增加10.57%、11.31%；F在vD的24/24、q的22/24个validation航次胜出。train中F分别胜出65/69、68/69个航次。两方法相对共同趋势均在全部有效航次改善。
2. **上轮B的vD优势主要来自趋势。** validation中旧A→B总MSE下降0.293266，其中趋势误差下降0.297562、周期增量记账为−0.004296；train对应0.342984、0.346489、−0.003505。q的validation总改善0.002212也小于趋势改善0.006578，周期增量为−0.004366；train的B总误差原本就更高。不能把旧B的总优势解释成LSF周期提取更好。
3. **固定0.5s合成阶跃被两者基本保留，F泄漏更小。** 所有train/validation航段中F的最大Fast泄漏为vD 0.000455 m/s、q 0.000355 rad/s；L为0.002572 m/s、0.001687 rad/s。F结束后幅值保留范围约99.994%–100.014%；F半幅到达相对延迟在当前采样分辨率下为0，L最大正延迟0.027441s。此结论仅适用于本次幅值与持续时间，不能外推到任意机动。
4. **真实vD时序问题仍在，不能宣布所有非周期运动都已验证保留。** vD的F/L相对越阈延迟中位数0.118/0.079s，P95为0.649/0.601s；净变化差绝对值P95为0.148/0.137 m/s。q的相对延迟中位数均为0，但净变化差P95仍为0.109/0.094 rad/s。L在这些描述性事件量上略小，故作为对照候选一起保留；事件窗口不是整拍积分，Fast净变化不必为零，原始运动也不是已知Slow真值。没有证据把这些差异直接定性为算法物理滞后或控制失效。
5. **可以进入下一阶段Slow/Fast GRU的设计讨论，不能直接把标签当正式真值启动训练。** 优先研究F标签，并保留L敏感性对照；设计必须保留原生双时间轴、有效掩码、航次划分及离线标签/因果输入边界，明确vD事件时序尚未解决。仅约35.7%的train和44.2%的validation原生记录有标签，边缘和质量排除不可伪填。本轮止于候选标签与证据，不启动训练。

'''


def write_report(out, comparison, motion, coverage, predictions, maneuvers, config, prior_events):
    fig, axes=plt.subplots(2,1,figsize=(10,7),constrained_layout=True)
    for ax,(ch,z) in zip(axes,predictions.items()):
        for j,name in enumerate(['raw','shared trend','trend + F','trend + L'],1):ax.plot(z[:,0],z[:,j],label=name)
        ax.set(xlabel='Time within held-out cycle (s)',ylabel=f'{ch} ({"m/s" if ch=="vD" else "rad/s"})');ax.legend()
    fig.savefig(out/'periodic_prediction.png',dpi=160);plt.close(fig)
    fig, axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for row,(ch,z) in zip(axes,maneuvers.items()):
        for j,name in enumerate(['injected','Slow F response','Slow L response'],1):row[0].plot(z[:,0],z[:,j],label=name)
        row[1].plot(z[:,0],z[:,4],label='Fast F leakage');row[1].plot(z[:,0],z[:,5],label='Fast L leakage')
        for ax in row:
            ax.axvline(0,color='gray',ls=':');ax.axvline(.5,color='gray',ls=':');ax.set(xlim=(-.5,1),xlabel='Time from maneuver start (s)',ylabel=f'{ch} ({"m/s" if ch=="vD" else "rad/s"})');ax.legend()
    fig.savefig(out/'maneuver_preservation.png',dpi=160);plt.close(fig)
    fair=comparison[comparison.aggregation=='flight_equal']
    syn=motion[(motion.kind=='synthetic')&(motion.stage=='full')].copy()
    syn['relative_delay_s']=syn.slow_half_change_delay_s-syn.raw_half_change_delay_s
    ev=motion[motion.kind=='control_event'].copy()
    ev['abs_raw_change']=ev.raw_change.abs();ev['abs_fast_change']=ev.fast_change.abs()
    ev['legacy_delay_s']=ev.legacy_slow_abs_half_delay_s-ev.legacy_raw_abs_half_delay_s
    cov=coverage.groupby(['split','channel','method'])[['n_native','n_valid']].sum();cov['coverage']=cov.n_valid/cov.n_native
    ss=syn.groupby(['split','channel','method']).agg(n=('run','size'),rmse_mean=('slow_error_rmse','mean'),leak_max=('fast_leak_max','max'),delay_median=('relative_delay_s','median'),delay_max=('relative_delay_s','max'))
    after=motion[(motion.kind=='synthetic')&(motion.stage=='after')].copy()
    after['retained_fraction']=after.slow_change/after.amplitude
    after_summary=after.groupby(['split','channel','method']).agg(retained_mean=('retained_fraction','mean'),retained_min=('retained_fraction','min'),retained_max=('retained_fraction','max'))
    es=ev.groupby(['channel','method']).agg(n=('run','size'),abs_raw_median=('abs_raw_change','median'),abs_fast_median=('abs_fast_change','median'),net_error_p95=('abs_net_change_error',lambda x:x.quantile(.95)),delay_median=('relative_delay_s','median'),delay_p95=('relative_delay_s',lambda x:x.quantile(.95)),timing_valid=('relative_delay_s','count'),legacy_delay_mean=('legacy_delay_s','mean'))
    def table(df):return '```text\n'+df.to_string(index=True,float_format=lambda x:f'{x:.6g}')+'\n```\n'
    text='''# vD/q 同趋势快慢分离补充实验

仅使用冻结 velocity_rate_multistep_v1 的 train/validation；未访问 test，未训练 GRU。vD 为 NED 向下速度(m/s)，q 为 FRD 俯仰角速度(rad/s)。

## 同趋势整拍留出比较

F/L 共用原 B 的每拍时间加权均值分段线性趋势，51拍双侧窗口；固定每5拍留出1拍。所有留出拍运动观测均排除；目标拍仅提供时间、相位与评价值。F 为残差截距加 H1–H3、整拍等权最小二乘，仅保留谐波；L 为Q=8半格网格、跨拍三阶LSF、至少8个单元且两侧各2拍。L沿用原实现，在网格时刻将插值运动减去共同趋势。两者在同一目标拍离散时间权重下去均值。

下表为先拍内MSE、再航次内平均、最后航次等权；RMSE为对应总体MSE的平方根。相对改善为汇总MSE之比；CSV另存航次相对改善的均值、每航次明细与整拍等权结果。窗口不是独立航次。MSE单位为各通道单位平方。

'''
    text+=table(fair[['split','channel','n_flights','trend_mse','trend_rmse','F_mse','F_rmse','L_mse','L_rmse','F_vs_trend_relative','L_vs_trend_relative','L_vs_F_abs','L_vs_F_relative','F_improved_flights','L_improved_flights','L_beats_F_flights','F_coverage','L_coverage']])
    text+='\n## 上轮 B 优势的拆解\n\n以同一批留出拍计算恒等分解：A误差−B误差 =（A趋势误差−共同趋势误差）+（B相对共同趋势的改善−A相对旧趋势的改善）。这只是误差记账，并非独立因果贡献率；F同时改变去趋势拟合和零均值约束，因此A−F也不能全部称为趋势贡献。\n\n'
    text+=table(fair[['split','channel','old_A_mse','old_B_mse','old_total_gain','old_trend_gain','old_incremental_periodic_gain','L_vs_F_abs']])
    text+='\n## 完整非周期机动\n\n每个足够长且中心前后覆盖完整的既有有效航段注入一次固定平滑阶跃；vD=0.3 m/s，q=0.2 rad/s，持续0.5s。重新分离注入前后数据，Slow差值应等于注入量；Fast差值为泄漏。表内覆盖开始前0.5s至结束后0.5s，CSV另有start/during/end/after分阶段误差、均值与最大值。不是飞机物理控制响应。\n\n'
    text+=table(ss)
    text+='\n结束后0.5s的Slow幅值保留比例（1为完整保留）：\n\n'+table(after_summary)
    text+='\n真实事件直接复用上一轮冻结P90筛选的validation事件时刻（电机及尾舵），不重新选阈值。净变化=事件后[0.5,1]s均值−事件前[-0.5,0)s均值。仅保留F/L共同连续有效且无缺口的完整事件窗口；事件CSV保留方向、控制量和航次，避免正负抵消。时序以原始净变化的一半为共同带方向门槛，并记录原始与Slow首次到达时间差；未到达为NaN。另保留上轮绝对偏离/各自半幅门槛的legacy描述量。\n\n'
    text+=table(es)
    text+=f'\n旧validation双通道事件记录 {len(prior_events)} 条；本轮共同有效事件记录 {len(ev)//2} 条（每条分别评估F/L，不作为独立因果试验）。\n'
    text+='\n闭环控制事件受反馈、周期运动与其他输入混杂；原始首次越阈可能由扑翼振荡触发，因此Slow较晚越阈不能直接判定为非周期响应的物理滞后，更不是反事实控制有效性证明。\n\n## 候选标签与适用范围\n\n保留F/L双候选以便按实测结果选择，均为研究级标签。Fast=P；Slow=raw−P，绝非共同趋势。每个通道分别保留自身原生timestamp_us，未强制同步。Parquet内raw、slow_F/fast_F/valid_F、slow_L/fast_L/valid_L及方法标识；无效位置为NaN和false，含窗口边缘、质量拒绝、稀疏航次等。原数据和split未改变。\n\n'
    text+=table(cov)
    text+=f'\n局部数值检查：改变所有留出拍运动观测1000后F/L预测完全不变；L与上一轮B及共同趋势逐留出拍误差一致。最大拍内加权Fast均值 {config["max_weighted_fast_mean"]:.3g}，最大重构误差 {config["max_reconstruction_error"]:.3g}。\n'
    text+='\n离线双侧标签使用未来观测；未来GRU的因果输入和自主rollout不得使用这些未来信息。Hall零点沿用工作假设，未独立校准绝对翼角；零均值仅针对拍内离散权重，不保证连续积分或净旋转守恒。候选Slow保留未解释残差和观测噪声，不是物理慢运动真值。覆盖不足的位置不能补伪标签。\n'
    text+='\n## 复现\n\n从该worktree运行（输出目录必须不存在；重现时可用 --output 指定新目录）：\n\n```bash\n/home/zn/anaconda3/envs/flap-train-gpu/bin/python experiments/slow_fast_separation_v1/two_channel_followup.py\n```\n'
    text+=f'\n基准代码版本 {config["code_revision"]} 加本轮methods.py/two_channel_followup.py/two_channel_report.py修改；固定数据split、相位合同及参数见configuration.json。运行耗时 {config["elapsed_s"]:.1f}s。两张图取首个满足条件的validation例子，未按效果挑选。\n'
    text=text.replace('## 同趋势整拍留出比较',conclusion()+'## 同趋势整拍留出比较',1)
    (out/'report_zh.md').write_text('\n'.join(line.rstrip() for line in text.splitlines())+'\n')
