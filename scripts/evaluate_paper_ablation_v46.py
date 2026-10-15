"""Descriptive OOF comparison and kinematic audit; never selects checkpoints."""
import argparse
import base64
import html
import io

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import paper_ablation_v46 as run
from evaluate_joint_increment_v35 import errors

OUT = run.OUT
ARMS = ('baseline',) + run.ARMS


def multiply(a, b):
    """Independent scalar-first quaternion multiplication, NumPy float64."""
    return np.concatenate([a[..., :1]*b[..., :1] - (a[..., 1:]*b[..., 1:]).sum(-1, keepdims=True),
        a[..., :1]*b[..., 1:] + b[..., :1]*a[..., 1:] + np.cross(a[..., 1:], b[..., 1:])], -1)


def distance(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    inverse = b.copy()
    inverse[..., 1:] *= -1
    relative = multiply(inverse, a)
    return 2*np.arctan2(np.linalg.norm(relative[..., 1:], axis=-1), np.abs(relative[..., 0]))


def integrate_rates(initial_quaternion, initial_rates, rates, dt):
    quat = initial_quaternion.astype(float).copy()
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    previous = initial_rates.astype(float)
    result = []
    for k in range(rates.shape[1]):
        vector = .5*(previous + rates[:, k])*dt[:, k, None]
        angle = np.linalg.norm(vector, axis=-1, keepdims=True)
        delta = np.concatenate([np.cos(angle/2), .5*np.sinc(angle/(2*np.pi))*vector], -1)
        quat = multiply(quat, delta)
        quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
        result.append(quat.copy())
        previous = rates[:, k]
    return np.stack(result, 1)


def prediction(arm, fold, seed, cohort='oof'):
    root = run.baseline_path(fold, seed) if arm == 'baseline' else run.destination(arm, fold, seed)
    with np.load(root / f'{cohort}_predictions.npz') as z:
        return z['array_index'], {k: z[k].astype(float) for k in ('rates_b', 'velocity_n', 'quaternion')}


def summarize():
    assert run.read(OUT / 'status.json')['status'] == 'training_complete'
    run.check_pins()
    windows = pd.read_csv(run.DATA / 'windows.csv')
    with np.load(run.DATA / 'windows.npz') as z:
        arrays = dict(z)
    with np.load(run.RESULTS / 'control_windows_v33/rolling_inputs.npz') as z:
        local_truth = z['label_rates_b'].astype(float)
    rows, decomposed, kinematic, local_rows, time_rows = [], [], [], [], []
    for fold in range(5):
        info = run.read(run.DATA / f'fold{fold}/manifest.json')
        for seed in run.SEEDS:
            for arm in ARMS:
                root = run.baseline_path(fold, seed) if arm == 'baseline' else run.destination(arm, fold, seed)
                if arm != 'baseline':
                    status = run.read(root / 'status.json')
                    assert status['status'] == 'complete' and status['step'] == 1600
                    assert run.base.sha(root / 'final.pt') == status['checkpoint_sha256']
                    assert status['reload_exact'] and status['buffers_unchanged']
                for cohort in ['fit', 'oof']:
                    ids, pred = prediction(arm, fold, seed, cohort)
                    np.testing.assert_array_equal(ids, info['fit_indices' if cohort == 'fit' else 'query_indices'])
                    true = {k: arrays['label_' + k][ids].astype(float) for k in pred}
                    meta = windows.iloc[ids][['array_index', 'log_id', 'active_reference', 'quality']].assign(
                        fold=fold, seed=seed, arm=arm, cohort=cohort)
                    for horizon in [1, 5, 25, 50]:
                        wave, endpoint = errors(pred, true, horizon)
                        for kind, values in [('wave', wave), ('endpoint', endpoint)]:
                            for metric, mse in values.items():
                                rows.append(meta.assign(horizon=horizon, kind=kind, metric=metric, mse=mse))
                    error = pred['rates_b'] - true['rates_b']
                    mean = error.mean(1)
                    centered = error - mean[:, None]
                    np.testing.assert_allclose((error**2).mean(1), mean**2 + (centered**2).mean(1), atol=1e-12)
                    for j, axis in enumerate(['p', 'q', 'r']):
                        pcenter = pred['rates_b'][:, :, j] - pred['rates_b'][:, :, j].mean(1, keepdims=True)
                        tcenter = true['rates_b'][:, :, j] - true['rates_b'][:, :, j].mean(1, keepdims=True)
                        denominator = np.sqrt((pcenter**2).sum(1)*(tcenter**2).sum(1))
                        corr = np.divide((pcenter*tcenter).sum(1), denominator,
                                         out=np.full(len(ids), np.nan), where=denominator > 1e-12)
                        decomposed.append(meta.assign(axis=axis, mean_error=mean[:, j], bias_mse=mean[:, j]**2,
                            centered_mse=(centered[:, :, j]**2).mean(1), correlation=corr,
                            amplitude_ratio=np.sqrt((pcenter**2).mean(1)/np.maximum((tcenter**2).mean(1), 1e-12))))
                    if cohort != 'oof':
                        continue
                    integrated = integrate_rates(arrays['input_initial_quaternion'][ids],
                        arrays['input_initial_rates_b'][ids], pred['rates_b'], arrays['input_dt_s'][ids])
                    gap = distance(pred['quaternion'], integrated)
                    integrated_error = distance(integrated, true['quaternion'])
                    for h in [25, 50]:
                        kinematic.append(meta.assign(horizon=h, gap_endpoint_mse=gap[:, h-1]**2,
                            integrated_attitude_endpoint_mse=integrated_error[:, h-1]**2))
                    # Horizon curves aggregate endpoint errors flight-first, not pooled windows.
                    point = dict(p=error[:, :, 0]**2, q=error[:, :, 1]**2, r=error[:, :, 2]**2,
                                 attitude=distance(pred['quaternion'], true['quaternion'])**2)
                    for metric, mse in point.items():
                        for k in range(50):
                            time_rows.append(dict(arm=arm, fold=fold, seed=seed, horizon=k+1,
                                                  metric=metric, rmse=float(np.sqrt(mse[:, k].mean()))))
                with np.load(root / 'rolling_predictions.npz') as z:
                    ids = z['array_index']
                    np.testing.assert_array_equal(ids, info['query_indices'])
                    mse = ((z['rates_b'].astype(float) - local_truth[ids])**2).mean(1)
                for j, axis in enumerate(['p', 'q', 'r']):
                    local_rows.append(windows.iloc[ids][['array_index', 'log_id']].assign(
                        arm=arm, fold=fold, seed=seed, axis=axis, mse=mse[:, j]))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(OUT / 'metrics_per_window.csv', index=False)
    flight = df.groupby(['seed', 'cohort', 'arm', 'horizon', 'kind', 'metric', 'log_id']).mse.mean().pow(.5).rename('rmse').reset_index()
    flight.to_csv(OUT / 'metrics_per_flight.csv', index=False)
    mean = flight.groupby(['seed', 'cohort', 'arm', 'horizon', 'kind', 'metric']).rmse.mean().reset_index()
    mean.to_csv(OUT / 'metrics_per_seed.csv', index=False)
    summary = mean.groupby(['cohort', 'arm', 'horizon', 'kind', 'metric']).rmse.mean().reset_index()
    summary.to_csv(OUT / 'summary.csv', index=False)
    pivot = mean.pivot(index=['seed', 'cohort', 'horizon', 'kind', 'metric'], columns='arm', values='rmse')
    baseline_old = pd.read_csv(run.RESULTS / 'control_windows_v35/comparison.csv').set_index(pivot.index.names)
    np.testing.assert_allclose(pivot.baseline, baseline_old.loc[pivot.index, 'joint'], rtol=1e-12, atol=1e-12)
    comparisons, gates = [], []
    for control in ['baseline', 'full']:
        for arm in run.ARMS:
            if arm == control:
                continue
            p = pivot.reset_index()
            p['gain_percent'] = 100*(1-p[arm]/p[control])
            comparisons.append(p[pivot.index.names].assign(arm=arm, control=control,
                                candidate_rmse=p[arm], control_rmse=p[control], gain_percent=p.gain_percent))
            q = p[(p.cohort == 'oof') & (p.horizon == 50) & (p.kind == 'wave') & (p.metric == 'q')]
            qflight = flight[(flight.cohort == 'oof') & (flight.horizon == 50) &
                (flight.kind == 'wave') & (flight.metric == 'q')].groupby(['log_id', 'arm']).rmse.mean().unstack()
            protect = p[(p.cohort == 'oof') & (((p.kind == 'wave') & (p.horizon == 50) & p.metric.isin(['p', 'r'])) |
                ((p.kind == 'endpoint') & p.horizon.isin([25, 50]) & p.metric.isin(['velocity', 'attitude'])))]
            protect.to_csv(OUT / f'protections_{arm}_vs_{control}.csv', index=False)
            gain = float(100*(1-q[arm].mean()/q[control].mean()))
            positive_seeds = int((q.gain_percent > 0).sum())
            positive_flights = int((qflight[arm] < qflight[control]).sum())
            protections = int((protect.gain_percent >= -5).sum())
            gates.append(dict(arm=arm, control=control, q_gain_percent=gain, positive_seeds=positive_seeds,
                positive_flights=positive_flights, protections_passed=protections, protections_total=len(protect),
                descriptive_gate=bool(gain >= 5 and positive_seeds == 3 and positive_flights >= 4 and protections == 18)))
    pd.concat(comparisons).to_csv(OUT / 'comparisons.csv', index=False)
    absolute = df[(df.cohort == 'oof') & (df.kind == 'endpoint') & (df.horizon == 25) & df.active_reference]
    absolute = absolute.groupby(['arm', 'seed', 'metric', 'log_id']).mse.mean().pow(.5).groupby(['arm', 'seed', 'metric']).mean().rename('rmse').reset_index()
    absolute['limit'] = absolute.metric.map({'p': .15, 'q': .15, 'attitude': np.deg2rad(3), 'velocity': .5})
    absolute = absolute[absolute.limit.notna()].copy()
    absolute['passed'] = absolute.rmse <= absolute.limit
    absolute.to_csv(OUT / 'absolute_gates.csv', index=False)
    for gate in gates:
        gate['absolute_passed'] = int(absolute[absolute.arm == gate['arm']].passed.sum())
        gate['absolute_total'] = 12
    pd.DataFrame(gates).to_csv(OUT / 'gates.csv', index=False)
    for name, frames in [('rate_decomposition', decomposed), ('kinematics', kinematic), ('local_rates', local_rows)]:
        pd.concat(frames, ignore_index=True).to_csv(OUT / f'{name}_per_window.csv', index=False)
    pd.DataFrame(time_rows).to_csv(OUT / 'horizon_curves.csv', index=False)
    run.base.write(OUT / 'evaluation.json', dict(gates=gates, baseline_metrics_reproduced=True,
        completed_runs=45, checkpoint_selection='final1600only', validation_accessed=False,
        sealed_test_accessed=False, reserved_accessed=False, control_ready=False,
        qualification='Repeated-development original-train OOF; descriptive paired comparisons only. No causal root-cause or closed-loop claim.'))
    print(pd.DataFrame(gates).to_string(index=False))


def metric_table(frame, value, group, square_root=True):
    f = frame.groupby(group + ['seed', 'log_id'])[value].mean()
    if square_root:
        f = f.pow(.5)
    return f.groupby(group + ['seed']).mean().groupby(group).mean()


def report():
    summary = pd.read_csv(OUT / 'summary.csv')
    main = summary[(summary.cohort == 'oof') & (summary.horizon == 50) & (summary.kind == 'wave')].pivot(index='arm', columns='metric', values='rmse').loc[list(ARMS)]
    main['attitude_deg'] = np.rad2deg(main.attitude)
    main = main[['p', 'q', 'r', 'velocity', 'attitude_deg']]
    gates = pd.read_csv(OUT / 'gates.csv')
    dec = pd.read_csv(OUT / 'rate_decomposition_per_window.csv')
    dec = dec[dec.cohort == 'oof']
    decomposition = pd.DataFrame({name: metric_table(dec, name, ['arm', 'axis'], square_root=name.endswith('mse'))
                                  for name in ['bias_mse', 'centered_mse', 'correlation', 'amplitude_ratio']})
    decomposition.to_csv(OUT / 'rate_decomposition_summary.csv')
    kin = pd.read_csv(OUT / 'kinematics_per_window.csv')
    ksummary = pd.DataFrame({name: metric_table(kin, name, ['arm', 'horizon'])*180/np.pi for name in
        ['gap_endpoint_mse', 'integrated_attitude_endpoint_mse']})
    ksummary.to_csv(OUT / 'kinematics_summary_deg.csv')
    local = pd.read_csv(OUT / 'local_rates_per_window.csv')
    lsummary = metric_table(local, 'mse', ['arm', 'axis']).unstack().loc[list(ARMS)]
    lsummary.to_csv(OUT / 'local_rates_summary.csv')
    # Fixed all-flight aggregate plot; seed band is min/max, not a confidence interval.
    curves = pd.read_csv(OUT / 'horizon_curves.csv').groupby(['arm', 'metric', 'seed', 'horizon']).rmse.mean().reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    colors = dict(zip(ARMS, ['#303030', '#e69f00', '#0072b2', '#009e73']))
    for ax, metric in zip(axes.flat, ['p', 'q', 'r', 'attitude']):
        for arm in ARMS:
            values = curves[(curves.arm == arm) & (curves.metric == metric)].pivot(index='horizon', columns='seed', values='rmse')
            if metric == 'attitude':
                values *= 180/np.pi
            times = values.index.to_numpy()*.02
            ax.plot(times, values.mean(axis=1), label=arm, color=colors[arm])
            ax.fill_between(times, values.min(axis=1), values.max(axis=1), color=colors[arm], alpha=.12)
        ax.set(title=metric, xlabel='Rollout time (s)', ylabel='RMSE (deg)' if metric == 'attitude' else 'RMSE (rad/s)')
        ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.savefig(OUT / 'horizon_curves.png', dpi=160)
    stream = io.StringIO()
    fig.savefig(stream, format='svg')
    svg = stream.getvalue()[stream.getvalue().index('<svg'):]
    plt.close(fig)
    conclusions = []
    for arm in run.ARMS:
        row = gates[(gates.arm == arm) & (gates.control == 'baseline')].iloc[0]
        conclusions.append(f'{arm}：q 波形改善 {row.q_gain_percent:+.2f}%；改善种子 {row.positive_seeds}/3、航次 {row.positive_flights}/5；保护项 {row.protections_passed}/18，绝对门槛 {row.absolute_passed}/12。')
    report_html = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>v46 论文结构消融</title>
<style>body{font:16px/1.7 system-ui,sans-serif;max-width:1200px;margin:36px auto;padding:0 24px;color:#172536}table{border-collapse:collapse;margin:20px 0;width:100%;font-variant-numeric:tabular-nums}th,td{padding:8px;border:1px solid #cbd5e1;text-align:right}th{background:#eef2f6}svg{max-width:100%;height:auto}.note{background:#eff6ff;padding:18px;border-radius:8px}code{background:#eef2f6;padding:2px 5px}</style>
<h1>v46：Rao 论文 multihead / decoupled 结构消融</h1>
<p>2026-10-15 · 原训练集 407 个窗口 / 5 航次留一法 / 3 种子 / 45 次新训练。各次 1600 更新，H26/K50；训练抽样、局部监督、损失及归一化沿用 v35。</p>
<div class="note">''' + '<br>'.join(map(html.escape, conclusions)) + '''<br>这些是反复用于开发的 OOF 数据，不能当作独立最终测试；未运行闭环控制。</div>
<h2>论文定义与本次适配</h2>
<p>论文 Figure 2 / IV-C 把线速度与角速度归为一个模块、姿态归为另一个模块。multihead 共享编码器、使用两个解码头；decoupled 使用两套编码器与解码头。</p>
<p><a href="https://github.com/arplaboratory/long-horizon-dynamics/blob/a8451a119b9096eda980f71b55dbb05012b8c47a/scripts/dynamics_learning/lighting.py">作者公开实现（固定 commit）</a>在训练及评估滚动时补入真实的另一组状态。本次所有分支只交换预测值，输入接口没有未来真实状态，因此是自主滚动的结构适配，不能与论文数值直接比较。</p>
<p>baseline 是 v35：由预测角速度积分姿态。full 改为一个头预测速度导数与姿态增量；multihead 把该头分开；decoupled 再拆分编码器/递归单元，并阻断互补预测状态的跨模块梯度。后两项相对 full 才是结构对照。姿态用三维旋转向量生成单位四元数，所有新组均相同。</p>
<p>参数量 baseline 40070、full 40265、multihead 39977、decoupled 39531（总量差异小于 1.5%）。decoupled 每分支宽度 42，其余编码器宽度 64；multihead 每头宽度 32。未做容量扫描。相同种子并不意味着不同结构的全部初值相同；full/multihead 编码器与递归单元初值逐张量配对 v35。</p>
<h2>1 秒波形 RMSE</h2><p>p/q/r：rad/s；velocity：m/s，三维分量平均口径；attitude：度。先每航次 RMSE，再航次等权、种子等权。</p>''' + main.to_html(float_format=lambda v:f'{v:.6f}') + '''
<h2>相对改善与保护门槛</h2><p>正值表示改善。主指标仍为 q 的 1 秒波形；保护项包括 p/r 波形、0.5/1 秒速度与姿态端点，每个种子不得退化超过 5%。绝对门槛为活跃输入窗口 0.5 秒 p/q≤0.15 rad/s、姿态≤3°、速度≤0.5 m/s。</p>''' + gates.to_html(index=False, float_format=lambda v:f'{v:.3f}') + '''
<h2>误差随自主滚动时间增长</h2><p>阴影为三个种子的最小到最大值，并非置信区间。</p>''' + svg + '''
<h2>误差组成</h2><p>bias_mse、centered_mse 列已开方：分别是片段均值偏差与去均值波形误差（rad/s）。correlation 为逐窗时间相关系数再航次等权；amplitude_ratio 为预测/真实去均值 RMS 比。相关系数为诊断，不替代误差指标。</p>''' + decomposition.to_html(float_format=lambda v:f'{v:.5f}') + '''
<h2>重新观察历史后的局部 20ms 角速度预测</h2><p>每次使用真实 H26 历史重新初始化，只预测下一步。不能当作 1 秒自由滚动成绩。</p>''' + lsummary.to_html(float_format=lambda v:f'{v:.6f}') + '''
<h2>姿态与预测角速度是否一致</h2><p>gap_endpoint_mse 列为独立姿态输出与积分预测角速度所得姿态的端点 RMSE（度）；integrated_attitude_endpoint_mse 为后者与记录姿态的端点 RMSE（度）。两列均已开方。独立姿态更准但该差距增大，不能说明角速度动力学问题解决。</p>''' + ksummary.to_html(float_format=lambda v:f'{v:.5f}') + '''
<h2>边界与可复核文件</h2><p>这次只改变模型结构/对应梯度连接，未改变记录频率、滤波、事件覆盖、训练样本比例或控制器。结果不能单独确定原始传感器噪声，也不能排除数据覆盖问题。decoupled 虽无跨模块参数梯度，仍共享全局梯度裁剪标量及同一训练计划；不是论文独立组件、真值补齐的复现。</p>
<p><a href="protocol.json">冻结协议</a> · <a href="runs.json">45 次训练清单</a> · <a href="summary.csv">汇总 CSV</a> · <a href="metrics_per_flight.csv">逐航次指标</a> · <a href="evaluation.json">门槛结果</a> · <a href="verification.json">独立复核</a></p></html>'''
    (OUT / 'report_zh.html').write_text(report_html)
    print(main.to_string())
    print(lsummary.to_string())
    print(ksummary.to_string())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['evaluate', 'report'], required=True)
    args = p.parse_args()
    summarize() if args.stage == 'evaluate' else report()
