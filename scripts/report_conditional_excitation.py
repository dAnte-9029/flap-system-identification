"""Describe excitation evidence without equating predictability with identifiability."""
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_conditional_excitation import OUT,CHANNELS
from run_isaac_surrogate_straight import sha

def main():
    m=pd.read_csv(OUT/'per_log_metrics.csv');allm=m[m.log_id=='ALL'];time=pd.read_csv(OUT/'action_change_by_horizon.csv');t=time[time.log_id=='ALL'];local=pd.read_csv(OUT/'local_state_action_support.csv')
    rows=[]
    for channel in CHANNELS:
        get=lambda features,learner:allm[(allm.features==features)&(allm.learner==learner)&(allm.channel==channel)].iloc[0]
        a=get('state_history','mlp');b=get('state_and_control_history','ridge');h=t[t.channel==channel].set_index('lag_s');n=local[local.channel==channel]
        rows.append(dict(channel=channel,state_history_r2=a.r2,state_action_history_r2=b.r2,state_action_history_residual_rmse=b.residual_rmse,
            delta_0p1_rms=h.loc[.1].persistence_rmse,delta_0p5_rms=h.loc[.5].persistence_rmse,delta_1p0_rms=h.loc[1.].persistence_rmse,
            delta_1p0_over_action_std=h.loc[1.].ratio,neighbor32_median_width=n.conditional_width.median(),neighbor32_median_distance=n.neighbor32_rms_standardized_distance.median()))
    result=pd.DataFrame(rows);result.to_csv(OUT/'summary.csv',index=False)
    fig,ax=plt.subplots(1,2,figsize=(11,4));x=np.arange(4)
    ax[0].bar(x-.17,result.state_history_r2,.34,label='State history: MLP');ax[0].bar(x+.17,result.state_action_history_r2,.34,label='State + past actions: Ridge');ax[0].set_xticks(x,CHANNELS);ax[0].set_ylabel('Whole-log held-out R2');ax[0].legend(fontsize=8)
    for c in CHANNELS:
        g=t[t.channel==c];ax[1].plot(g.lag_s,g.ratio,'o-',label=c)
    ax[1].set(xlabel='Action change horizon (s)',ylabel='RMS action change / action std');ax[1].legend(fontsize=8);ax[1].grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'excitation_summary.png',dpi=160)
    text='''# 闭环数据条件激励审计

**结论：现有数据不能简单认定为“控制信号完全被状态决定，因此不可辨识”。** 舵面在 0.5–1 s 尺度上有明显变化；短时动作高度可预测，很大程度来自指令自身的时间连续性。现有证据支持先做控制条件联合训练对照，不支持仅凭这次审计就宣布必须补飞或数据已经充分。

## 方法

仅使用 registry 指定的 September v3 训练集：41 个日志，295259 个有效样本。条件动作预测每 5 个采样点取一行，共 58772 行。5 折按完整日志划分，特征标准化和回归器仅拟合各折训练日志，保留每个日志的结果。没有加载测试集。

- 状态特征包含体轴速度、角速度、重力方向、已有记录相位的 sin/cos 和扑翼频率；使用当前及过去 0.1/0.2 s 状态。
- 第二组另外输入过去 0.02/0.1/0.2 s 的动作，绝不输入当前预测目标动作。
- 使用已有环境的 PyTorch 实现 Ridge 和两层 64 单元 tanh MLP（25 epochs）作诊断。环境没有 sklearn，未安装新依赖。
- 分别检查 0.02/0.1/0.5/1 s 原始动作变化、残差谱、残差通道相关性，以及其他日志的 32 个近邻状态中的动作范围。跨日志/分段和无效间隔不拼接。

## 主要结果

下面 R² 分别取预先指定的状态历史 MLP 与状态+动作历史 Ridge，完整两种回归器结果见 `per_log_metrics.csv`；并非从多个网络中选择最有利结果。

| 通道 | 状态历史 R² | 加动作历史 R² | 加历史后的动作 RMSE | 1 s 动作变化 RMS / 动作标准差 |
|---|---:|---:|---:|---:|
'''
    for r in result.itertuples():text+=f'| {r.channel} | {r.state_history_r2:.3f} | {r.state_action_history_r2:.3f} | {r.state_action_history_residual_rmse:.4f} | {r.delta_1p0_over_action_std:.3f} |\n'
    text+='''
油门与状态（包括扑翼频率）的关系较强，但三个舵面仅凭这组状态历史不能准确预测。加入最近动作以后 R² 接近 1，并不意味着没有动态激励：0.02 s 指令变化很小，而 1 s 内差模变化 RMS 已达到其全局标准差的约 85%，共模约 57%，方向舵约 48%。

![审计结果](excitation_summary.png)

跨日志状态近邻的动作 10%–90% 宽度中位数分别为 motor 0.101、common 0.179、differential 0.509、rudder 0.242（归一化指令单位）。但最近邻状态的标准化距离并不为零：最近点的特征 RMS 距离中位数约 0.425，第 32 个近邻约 0.583。因此这只是存在局部动作差异的线索，不是严格的“相同状态、随机动作”证据。

状态+历史动作 MLP 残差相关矩阵最小特征值约 0.615，没有出现全局残差通道严格线性秩塌缩。相关矩阵经过逐通道归一化，不能以这个数证明残差信号强或辨识充分。残差谱来自约 10 Hz 抽样，未做抗混叠滤波，仅作为抽样序列描述，不用于判断物理高频激励；原始动作谱另由完整采样段计算。

## 能说明和不能说明什么

1. 没有证据支持所有通道都落在唯一、确定的状态反馈图上。不同日志的参考、积分状态、风和飞行条件也可能导致同一可观测状态下动作不同。
2. 动作预测残差混合了预测器误差、未观测控制器状态和噪声，不是外生激励，不能直接拿来作因果工具变量。状态预测器拟合不足时，也会夸大“独立变化”。部分单日志 R² 为负，尤其方向舵，完整结果已保留。
3. 高时间相关性不等于激励不足；0.5–1 s 的变化仍可能有辨识价值。但它是否激励到了相关动力学模式，还要看输入—响应的联合信息以及新模型的操纵响应验证。
4. 本轮没有据此改变样本权重、训练目标或划分；联合训练对照使用相同完整训练日志。没有给随机修改后的动作配原来的状态标签。

下一步由固定预算的联合模型对照检验：控制输入从训练开始进入主干，并联合更新主干与执行器分支，是否能同时改善短时真实轨迹预测与模型内操纵响应。即使改善，也不等于闭环辨识偏差已经消除；若仍不合理，再针对具体通道设计独立闭环激励。

## 产物与复现

`manifest.json`、`timescale_manifest.json` 记录数据/相位/频率契约与哈希。`folds.json` 记录完整日志划分，`crossfit_predictions.csv` 和 `per_log_metrics.csv` 保留逐行/逐日志结果；`action_change_by_horizon.csv`、`residual_psd.csv`、`action_psd.csv` 和 `local_state_action_support.csv` 保留尺度与邻域证据。

使用项目环境依次执行 `scripts/audit_conditional_excitation.py`、`scripts/audit_excitation_timescales.py`、`scripts/report_conditional_excitation.py`。前者默认要求新输出目录以避免覆盖既有审计。
'''
    (OUT/'report.md').write_text(text);(OUT/'report_manifest.json').write_text(json.dumps(dict(dataset=json.loads((OUT/'manifest.json').read_text())['dataset'],script_sha256=sha(__file__),input_sha256={n:sha(OUT/n) for n in ['per_log_metrics.csv','action_change_by_horizon.csv','local_state_action_support.csv']},test_opened=False),indent=2));print(result.to_string(index=False))
if __name__=='__main__':main()
