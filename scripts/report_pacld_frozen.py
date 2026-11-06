#!/usr/bin/env python
"""Render measured frozen evaluation artifacts without reading flight data."""
import json,math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];out=root/'runs/pacld_frozen_eval_seed17/results'
r=json.loads((out/'evaluation.json').read_text());response=json.loads((out/'response.json').read_text());metrics=pd.read_csv(out/'physical_rmse.csv');curve=pd.read_csv(out/'validation_curve.csv');intra=pd.read_csv(out/'intracycle_summary.csv');pairs=pd.DataFrame(r['paired']);latent=r['latent']
ps=[]
for (mode,c),g in pairs.groupby(['mode','component']):
 mse=g.groupby(['flight_id','horizon_us']).mean_squared.mean().groupby('flight_id').mean().mean()
 ps.append(dict(mode=mode,component=c,rmse=np.sqrt(mse)*(180/math.pi if c=='R' else 1),pairs=int(g.pairs.sum()),windows=g.window_id.nunique(),flights=g.flight_id.nunique()))
pd.DataFrame(ps).to_csv(out/'paired_summary.csv',index=False)
phase=pd.read_csv(out/'phase_residuals.csv');fig,axs=plt.subplots(1,2,figsize=(10,4))
for ax,var in zip(axs,['q','vD']):
 for mode in ['FULL','BASE']:
  g=phase[(phase.variable==var)&(phase['mode']==mode)].groupby(['flight_id','horizon_us','bin']).error.mean().groupby(['flight_id','bin']).mean().groupby('bin').mean();ax.plot((g.index+.5)*2*math.pi/16,g,'o-',label=mode)
 ax.set(title=var,xlabel='Offline Hall phase (rad)',ylabel='Signed residual');ax.legend()
fig.tight_layout();fig.savefig(out/'phase_residuals.png')
slow=np.array([x['slow_std'] for x in latent]);amp=np.array([[m['envelope_amplitude_rms'] for m in x['modes']] for x in latent]);slope=np.array([[m['envelope_phase_slope_vs_phi'] if m['envelope_phase_slope_vs_phi'] is not None else np.nan for m in x['modes']] for x in latent]);explicit=np.array([x['explicit_acceleration_rms']+x['explicit_angular_acceleration_rms'] for x in latent])
diag=dict(slow_std_median= np.median(slow,axis=0).tolist(),envelope_rms_median=np.median(amp,axis=0).tolist(),envelope_phase_slope_median=np.nanmedian(slope,axis=0).tolist(),explicit_decoder_rms_median=np.median(explicit,axis=0).tolist());(out/'latent_summary.json').write_text(json.dumps(diag,indent=2))
def table(df):
 def fmt(v):return f'{v:.5f}' if isinstance(v,float) else str(v)
 return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))
text=f'''# PACLD v1 冻结模型综合评价

代码基线 `669c50042f9f50fdbea0ad6e8b875bd40c116b86`。Seed 17，mixed precision，Global Batch 32，342 updates / 10,944 训练窗口。本次未训练、未改权重/损失/数据、未访问 sealed test、未启动闭环仿真。

## 冻结身份与复现

工作树 `{root}`。模型 `runs/pacld_frozen_eval_seed17/model.pt`；SHA256 `9d82640692c0baec476e276696be4f89fd89a58854361bbcfdf3dbb84cb466c9`（此前复制及复核，本次只读）。完整来源、配置、数据身份和固定 validation 成员见同目录 `manifest.json`。本次从 checkpoint 的 validation_window_ids 取出固定 156 窗口，未重新抽样。

运行 `/home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/evaluate_pacld_frozen.py`，再运行 `scripts/report_pacld_frozen.py`。完整 JSON、CSV、PNG 位于 `{out}`，不提交模型及大型结果。

## 11 节点验证与训练充分性

{table(curve)}

FULL 最佳点为 342 步 39.848219，Macro 为 6.775553。192 步出现反弹，随后改善。最后 22 步 FULL 仍下降 1.328499，现有验证曲线不支持已达到平台期；继续训练可能改善验证目标，但不保证物理/控制收益。本次不继续训练。没有同口径训练曲线，不能认定过拟合。图：`results/validation_curve.png`。

## 四时域物理预测

NED 位置/速度，FRD 角速度，Hamilton wxyz 姿态。姿态采用 SO(3) 测地误差（表中度）；其余依次 m、m/s、rad/s。向量指标是三维误差平方模长先平均再开根号。每个请求时刻取 ±30 ms 内最近原生有效观测，不插值；不同指标覆盖独立报告。窗口先在航次/窗口时域内平均，再对时域组平均，最后航次等权。native_observations 是所选原生目标计数，跨窗口可能重复，不能视为独立样本数。

{table(metrics)}

FULL、BASE、运动学基线使用相同因果 Belief、相同窗口及目标。运动学基线保持初始惯性速度和机体角速度，以 SO(3) 积分姿态。BASE 仅禁用显式相位载波，仍保留包络、隐状态、控制和全部共享参数。图：`results/physical_rmse.png`。

GRU 未纳入公平比较：当前评价接口没有经核验的同一 156 窗口、四时域、相同因果输入/物理状态输出合同的冻结 GRU adapter。既有 FrozenRollResponse 的 GRU 对齐目标和来源合同不同，不能抄旧表拼接排名。因此“竞争力”结论仅相对本次同口径运动学基线，不涵盖 GRU。

## 拍内 FULL/BASE

{table(intra)}

按原生 q、vD 观测计算；control_change/stable 使用既有 diagnostics 阈值与回看区间。不是重新筛选控制事件。相位仅用于离线残差分组，从未进入 Encoder 或未来传播。图 `results/phase_residuals.png` 为 16 个机械相位箱内先航次/时域聚合、再航次等权的有符号残差；详细相位箱、航次、窗口见 `phase_residuals.csv`。相位相关只说明描述性结构，不代表高频真值重建。

## 跨拍净运动

{table(pd.DataFrame(ps))}

严格复用 valid_pairs 的真实跨拍配对；位置/速度/角速度比较物理增量，姿态比较相对旋转增量。RMSE 航次等权，配对高度重叠；pairs 是计数而非独立样本量，独立信息以 flights 为主。没有伪造 Macro 配对。

## 隐状态功能诊断

以下是跨窗口中位数，不是隐变量真值标签：

```json
{json.dumps(diag,indent=2)}
```

slow_std 为每窗口 12 维 Slow 沿未来轨迹的标准差；三组 envelope_rms 为二维包络模长 RMS；envelope_phase_slope 是包络角对自主传播相位的斜率，完全反向抵消参考值依次 -1/-2/-3。显式解码贡献在同一个 FULL 状态上计算 decode(FULL)-decode(BASE)，前三维为解码体轴加速度、后三维为角加速度。与 BASE 自主轨迹的差异不是同一概念。包络 RMS 中位数约 2.78–2.81，接近分量限幅 2 所对应的模长上限 2√2，提示包络接近饱和值得关注，但不能据此诊断物理塌缩。包络角斜率中位数接近 0，未显示整体 -1/-2/-3 反向抵消。完整逐窗口摘要见 evaluation.json；平滑、非零振幅或相位斜率均不足以证明物理快慢解耦或排除局部塌缩。

## 冻结控制响应

```json
{json.dumps(response,indent=2,ensure_ascii=False)}
```

保留原 FrozenRollResponse 主配对、航次、依赖分组；不重筛选事件。只有所有窗口身份均存在才运行；缺失明确记为 missing_windows。本次 validation 唯一配对缺少 1 个窗口，train 48 对缺少 26 个窗口，因此两者均未计算响应指标；不是模型响应失败，而是当前 PACLD admission index 与历史配对窗口不完全兼容。缺失身份见 response_missing_windows.json。零配对或缺失窗口时不输出伪造指标。查询时刻保持原协议的 decision-relative shift，严格检查 actual_horizon_s，超出支持直接拒绝。历史仅包含当时已发布观测，未来使用真实 ZOH 控制。

该协议自身使用既有原生轨迹插值对齐，这与上面的原生端点误差不同；不是额外构造原生真值，也不是反事实控制真值。Train 描述性结果与 validation 分开，不能作为独立泛化证据。仅一个 validation 配对即使可计算，也不足以支持可靠泛化结论。

## 正确性与证据边界

定向测试 `tests/test_pacld_frozen_eval.py`：CPU/GPU 两项通过。检查无 Macro 配对的强制 BASE、异步任意合法查询时刻、共享因果 Belief、超时域拒绝；批量 FULL/BASE 对单窗口 FP64 参考使用 rtol=1e-8、atol=2e-10。本次实际评价沿用训练 mixed precision；该测试证明 FP64 执行定义一致，不把 mixed 与 FP64 宣称为逐位相同。
'''
# Explicit measured comparisons, no causal claims.
pivot=metrics.pivot(index=['component','horizon_s'],columns='model',values='rmse');wins=int((pivot.FULL<pivot.kinematic).sum());fullbase=int((pivot.FULL<pivot.BASE).sum())
text+=f'''\n## 科研判断\n\n- **VERIFIED：预测精度。** FULL 在 {wins}/{len(pivot)} 个物理通道/时域组合优于运动学基线；在 {fullbase}/{len(pivot)} 个组合优于 BASE。具体优势与不足以四时域表为准，不能据此声称优于 GRU。\n- **SUPPORTED / UNRESOLVED：结构分工。** 所有 7 个可计算的 q/vD 分组均有 FULL 收益：q RMSE 降低约 8.6%–12.1%，vD 约 1.6%–1.9%。Fast 提供可观测额外预测价值，但垂向收益较小。跨拍 BASE 相比 FULL 的位置/速度/姿态/角速度 RMSE 分别约高 0.56%/0.99%/3.55%/0.03%，说明在这些冻结配对上大部分跨拍运动保留。是否保留跨拍运动应看相对误差，而非隐状态外观。不存在真实慢变标签，BASE 不是真实慢动力学，非零 Fast 贡献也不等于额外预测价值。\n- **UNRESOLVED：控制有效性。** 冻结配对覆盖及支持检查决定可计算范围；有限 validation 配对不足以确认可靠控制响应泛化。观察性配对不能提供反事实真值。\n- **SUPPORTED：训练尚有改善迹象。** 最后节点仍改善，未证实平台。下一阶段最关键的问题是物理预测收益能否转化为独立航次上的控制响应证据，而非增加隐状态可视化或宣称物理解耦。\n'''
(root/'experiments/phase_anchored_coupled_latent_v1/frozen_evaluation_zh.md').write_text(text)
