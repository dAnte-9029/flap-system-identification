"""Join the three experiments under their fixed cohorts and write the Chinese scientific report."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from workspace import *
from freeze import names

LABEL={'position_m':'位置','velocity_m_s':'速度','attitude_deg':'姿态','rate_rad_s':'角速度'}
UNITS={'position_m':'m','velocity_m_s':'m/s','attitude_deg':'°','rate_rad_s':'rad/s'}
ORDER=list(LABEL)
SETTINGS=('2s','matched_original2','extrapolated3','trained3')
CANDIDATE_IDS=['s2a','s2b','s2c','s3a','s3b','s3c']


def read(path):
    return json.loads(path.read_text())


def run_record(directory,stage):
    cfg=read(directory/'config.json')
    record=read(directory/'completed.json')
    epochs=pd.read_json(directory/'epochs.jsonl',lines=True)
    selected=epochs.loc[epochs.validation_score.idxmin()]
    return dict(stage=stage,model=directory.name,params=cfg['params'],seed=cfg['seed'],updates=record['updates'],
                epoch=int(selected.epoch),selected_updates=int(selected.updates),shared=float(selected.validation_score),
                gpu_seconds=record['seconds'],checkpoint=str(directory/'best.pt'))


def joint():
    allrows=[];paired=[]
    for setting in SETTINGS:
        f=pd.read_csv(RUN/f'{setting}_paired.csv')
        f=f[f.horizon_s>=.5].copy()
        f['setting']=setting
        paired.append(f)
        for (h,metric),sub in f.groupby(['horizon_s','metric']):
            by=sub.pivot(index='flight_id',columns='model',values='mse')
            win=sub.pivot(index='flight_id',columns='model',values='windows')
            assert (win.eq(win.gru,axis=0)).all().all() and np.isfinite(by).all().all()
            public='attitude_deg' if metric=='attitude_rad' else metric
            factor=180/np.pi if public=='attitude_deg' else 1
            rmse=by.mean().pow(.5)*factor
            best=rmse[list(BASELINES)].idxmin()
            for m,v in rmse.items():
                allrows.append(dict(setting=setting,horizon_s=h,metric=public,unit=UNITS[public],model=m,rmse=v,
                                    best_baseline=best,best_baseline_rmse=rmse[best],gap=v-rmse[best],
                                    wins_vs_best=int((by[m]<by[best]).sum()),flights=len(by),windows=int(win[m].sum())))
    pd.concat(paired,ignore_index=True).to_csv(RUN/'paired_flights.csv',index=False)
    full=pd.DataFrame(allrows)
    full.to_csv(RUN/'comparison.csv',index=False)
    return full


def ordered(tab):
    return tab.assign(sort=tab.metric.map({k:i for i,k in enumerate(ORDER)})).sort_values(['horizon_s','sort'])


def md_table(header,rows):
    return '\n'.join(['| '+' | '.join(header)+' |','|'+'|'.join(['---']*len(header))+'|']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def metric_table(setting,rep):
    frame=ordered(pd.read_csv(RUN/f'{setting}_comparison.csv'))
    rows=[]
    for _,v in frame.iterrows():
        spec='.2f' if v.metric=='attitude_deg' else '.4f'
        vals=[format(v[k],spec) for k in [*BASELINES,rep]]
        rows.append([f'{v.horizon_s:g}',f'{LABEL[v.metric]} ({UNITS[v.metric]})',*vals,
                     v.best_baseline,format(v.gap,'+'+spec),f'{int(v.wins_vs_best)}/{int(v.flights)}'])
    return md_table(['时域 s','RMSE','GRU','LSTM','TCN','Transformer','主模型','最强基线','主模型减最强','胜出航次'],rows)


def write_report():
    rule=read(RUN/'freeze.json')
    rep=rule['representative']
    full=joint()
    selected=full[full.model==rep]
    def wins(setting):
        return selected[selected.setting==setting].groupby('horizon_s').gap.agg(lambda s:int((s<0).sum()))
    w0,w4,w5=map(wins,('2s','extrapolated3','trained3'))
    hist=[run_record((OLD if n in ('slowfast','gru') else SUITE)/n,'historic_2s') for n in ['slowfast',*BASELINES]]
    hist.extend(run_record(STAGE1/n,'stage1') for n in ('c1','c2'))
    runs=[run_record(RUN/n,'stage2' if n.startswith('s2') else 'stage3') for n in CANDIDATE_IDS]
    three=[run_record(RUN/'trained3'/n,'stage5') for n in [rep,*BASELINES]]
    trials=pd.DataFrame([*hist,*runs,*three])
    trials.to_csv(RUN/'experiments.csv',index=False)
    total=sum(r['gpu_seconds'] for r in runs+three)
    methods={'s2a':'历史均值 + 受控目标弛豫','s2b':'额外持续输入历史编码的慢目标 MLP',
             's2c':'将慢更新从 100 ms 调到 40 ms','s3a':'起点参考系，三个时常的控制记忆',
             's3b':'S2B 加权相对端点损失 0.16','s3c':'门控转弯项及 0.05 相对端点损失'}
    attempts=[]
    for r in runs+three:
        model=r['model']
        learn2=r['stage']!='stage5'
        attempts.append([r['stage'][-1],model,methods[model] if learn2 else '共同 3 s 从头训练',
                         f"{r['params']:,}",f"{r['epoch']} / {r['updates']:,}",f"{r['shared']:.6f}",
                         f"{r['gpu_seconds']/60:.2f}"])
    matched=ordered(pd.read_csv(RUN/'matched_original2_comparison.csv'))
    ext=ordered(pd.read_csv(RUN/'extrapolated3_comparison.csv'))
    only2=ordered(pd.read_csv(RUN/'2s_comparison.csv'))
    cm=matched[matched.horizon_s==2][rep].to_numpy()
    cs=only2[only2.horizon_s==2][rep].to_numpy()
    ce=ext[ext.horizon_s==2][rep].to_numpy()
    contrast=md_table(['2 s 主模型 RMSE','原始 44,524 起点','限制为 43,987 起点，原 2 s 评分','同起点，扩展 3 s 评分'],
                      [[LABEL[k]+' ('+UNITS[k]+')',*(f'{v:.4f}' for v in (a,b,c))]
                       for k,a,b,c in zip(ORDER,cs,cm,ce)])
    f4=selected[(selected.setting=='extrapolated3')&(selected.horizon_s==3)].set_index('metric')
    f5=selected[(selected.setting=='trained3')&(selected.horizon_s==3)].set_index('metric')
    paired3=md_table(['3 s 末端','冻结 2 s 主模型','3 s 训练主模型','同训最强基线','主模型减最强','胜出航次'],
                     [[LABEL[k]+' ('+UNITS[k]+')',f'{f4.loc[k,"rmse"]:.4f}',f'{f5.loc[k,"rmse"]:.4f}',
                       f'{f5.loc[k,"best_baseline"]} {f5.loc[k,"best_baseline_rmse"]:.4f}',
                       f'{f5.loc[k,"gap"]:+.4f}',f'{f5.loc[k,"wins_vs_best"]}/39'] for k in ORDER])
    checks=[read(RUN/f'{arm}_pipeline.json')['checks'] for arm in ('2s','extrapolated3','trained3')]
    check_max={col:max(row[col] for check in checks for row in check.values()) for col in ('observation_removal','control_prefix','cut_to_2s')}
    result=f"""# 扑翼飞行受控轨迹预测 · 第二至第五阶段

**原 2 s 开发协议选出 {rep.upper()}。同一权重共领先 {w0.sum()}/12 项**，0.5/1/2 s 的领先项数为 {w0.loc[.5]}/4、{w0.loc[1]}/4、{w0.loc[2]}/4。第一阶段 C2 为 7/12。{rep.upper()} 的 2 s 角速度为 0.6648 rad/s，32/39 架次优于 TCN；2 s 位置 1.1554 m、姿态 11.333°，仍落后于 Transformer。三个中间路标中，1 s 角速度为 0.5751 rad/s，达到 ≤0.63；2 s 位置比 1.13 m 高 0.0254 m，姿态比 11.25° 高 0.083°。1 s 位置仅领先约 0.0012 m，21/39 架次胜出。单种子开发验证不支持所有时域全面领先的结论。

**3 s 末端：直接外推领先 {w4.loc[3]}/4 项，五模型同训至 3 s 后领先 {w5.loc[3]}/4 项。** 两组使用相同的 43,987 个验证起点与完整四路控制。四个时域合计分别领先 {w4.sum()}/16 和 {w5.sum()}/16。

{paired3}

## 冻结诊断和 2 s 设计选择

C2 的首个有效完整拍内快分支角速度瞬时 RMS 为 0.4929 rad/s，拍内有符号时间均值的 RMS 只有 0.0250 rad/s。两秒窗口内，预测减观测角速度的有符号时间均值逐轴 RMS 为 0.0621/0.0501/0.0794 rad/s，水平速度为 0.3466/0.3506 m/s。积分速度误差得到 1.1012 m 的末端位移差。快分支在拍内大部分抵消，跨拍的平均运动仍值得建模。

S2B 保留 C2 的事件级相位、两阶谐波及四元数积分。0.5 s 历史由 64 维 GRU 编码；48 维慢 GRU 每 100 ms 更新。一个 64 隐层单元的 MLP 持续读取历史编码、当前慢状态、递推状态与机体系重力方向，预测慢均值偏移。各轴按可训练正弛豫率追踪目标。控制事件按实际持续时间进入慢加速度和谐波系数演化，连续查询只使用事件以来已经过的时间。S2B 有 43,274 个参数，四个基线有 34,468–35,494 个参数，训练预算一致，容量有所增加。[论文结构图 PDF](network.pdf) · [结构图 PNG](network.png) · [交互结构图](network.html)

S2C 的 40 ms 慢更新较 S2B 的 100 ms 更新计算更多，共享分数较差。S3A 使用固定起点机体系和平滑控制记忆，2 s 速度退到 1.5262 m/s。S3B 加大相对端点权重，2 s 位置改善到 1.1345 m，但全程共享验证分数从 0.135240 变为 0.139938。S2B 预测转弯运输项的水平 RMS 为 2.05 m/s²，与尚缺的水平位移修正的未中心化余弦为 0.098；实测角速度积分闭合误差与模型姿态残差的余弦仅 0.014。最后的 S3C 只引入可学习的门控转弯项、减轻端点重权，也未超过 S2B 的共享分数。闭合误差的具体物理来源仍未确定。

## 0.5、1、2 s 原协议

四个基线逐指标取最小 RMSE，符号差值小于零表示主模型胜出。姿态取四元数测地角，其余为三轴平方范数。先在架次内平均窗口平方误差，再在航次间等权平均后开根。姿态以度展示，配对 CSV 保留 rad²。位置、速度为 NED，角速度为 FRD。开发验证共 39 架次。

{metric_table('2s',rep)}

## 3 s 共同起点和直接外推

在原 velocity_rate_multistep_v1 的训练/开发验证分区调用原生窗口构建器，保留 0.5 s 历史以及与第一阶段相同的起点身份。对完整 3 s 支持检查航段、重复与断流、复位、审阅区间、三类原生时钟容差、四通道控制和标签覆盖。训练 126,237 → 123,712 个窗口（121 架次），验证 44,524 → 43,987 个窗口（39 架次）。依次按原生 3 s 窗口不可用、扩展后命中审阅区间、决策时钟尾部支持不足排除训练 2,269 / 218 / 38 和验证 426 / 100 / 11 个起点。所有新合格窗口使用真实记录；填充行不参与评分。

{contrast}

两个数据管线重叠的有效原生标签、时间戳最大差均为 0。0.5/1 s 全部端点相同。2 s 时，新管线能看到原 2 s 标签支持之外仍在同一航段的记录，{read(RUN/'common_prefix_data.json')['changed_endpoints']['2.0']['pv']} 个起点的位置/速度标签、{read(RUN/'common_prefix_data.json')['changed_endpoints']['2.0']['rate']} 个起点的角速度标签选择有变化。原管线在最后可评分标签后保持末值，7,710 个起点的控制在最后 0.1 s 内与扩展数据有差异。上表中间列隔离窗口成员变化，右列另包含评分边界与控制支持变化。冻结外推各项结果如下。

{metric_table('extrapolated3',rep)}

## 所有模型从头训练至 3 s

第三阶段结束时先按 2 s 共享分数冻结主结构、四个基线和 3 s 规则。每个模型均从 seed 17 重新初始化，使用相同训练窗口、打乱顺序、batch 1024 和 2,976 次 AdamW 更新，约 24.63 个训练集等效轮次。归一化仅从对应训练分区拟合。共有选模分数由全程速度/角速度归一化误差和位置/姿态端点归一化误差等权构成；端点为 0.2/0.5/1/2/3 s。主模型延续首完整拍的慢趋势辅助损失和 1/2/3 s 位置与姿态端点损失。每个模型所有指标统一使用该轮次的同一 checkpoint。两种 3 s 设定同时改变了训练监督时长和训练起点集合及相应训练尺度，不能从两表差值分离这些因素。

{metric_table('trained3',rep)}

同训到 3 s 后，主模型在 0.5 s 和 1 s 各领先四项，2 s 只有角速度领先。1 s 姿态的相对名次上升，但在相同起点上主模型自身由 6.698° 变为 6.748°，同项最强 Transformer 由 6.603° 变为 7.031°。3 s 角速度只落后 TCN 约 0.00051 rad/s，39 架次中主模型赢 22 架；总体排序仍按事先约定的均方根判定。Transformer 的 3 s 姿态分别为 40.531°（外推）和 14.634°（同训），使两组 3 s 姿态基线冠军从 TCN 换为 Transformer。

[预测时域–误差图 PDF](rmse_horizons.pdf) · [PNG](rmse_horizons.png)

## 训练账目与核查

第二、三阶段各 3 次正式训练；第四阶段 0 次；第五阶段 5 次。共 {len(runs)+len(three)} 次、{(len(runs)+len(three))*2976:,} 次更新。各次占用 GPU 的墙钟时长合计约 {total/60:.1f} 分钟，含该运行的数据加载、图捕获和验证，两卡并行时会重叠。所有 epoch 按统一验证分数选择。历史基线和第一阶段的两次训练分别列在 experiments.csv 中。

{md_table(['阶段','模型','主要改动','参数','最佳轮次 / 总更新','共享分数','GPU 分钟'],attempts)}

另有三次初始化级技术失败没有执行参数更新：S2A 图捕获遇到高级索引；3 s TCN 与 Transformer 的全零四元数占位在被掩码后仍引发 atan2(0,0) 的反向梯度 NaN。修复为可捕获的张量操作和缺失行单位四元数占位。实测有效标签与掩码不变，第四阶段全部逐架次分数修复前后逐字节相同，五种模型均验证梯度有限。保留故障日志。

对六个新 2 s 候选及两种 3 s 设定下的五个模型，针对每架次第一合格起点重建原始预处理。固定控制、移除 T 后发布的全部观测，最大预测差 {check_max['observation_removal']:.2g}。在 T+667 ms 或 T+2,667 ms 的步内改变后续控制，所有先前输出最大差 {check_max['control_prefix']:.2g}。相同历史和控制前缀上，3 s 调用截到 2 s 与原短调用最大差 {check_max['cut_to_2s']:.2g}。这组因果检查覆盖每种设定 39 个代表起点；性能评分覆盖该设定全部合格窗口。

结果限定于单种子、开发验证集及给定已执行控制的条件轨迹预测。主模型经历多次开发尝试，四个基线沿用已选设计。多窗口同属同一飞行，航次胜数与 RMSE 同列，不能把窗口数当成独立实验数。封存测试和预留航次未用于开发；反事实控制响应与 MPC 闭环效果尚未检验。

## 可复用产物

- [统一对照表](comparison.csv) · [逐架次配对误差](paired_flights.csv) · [所有运行和权重的索引](experiments.csv) · [3 s 评估前冻结记录](freeze.json)
- [2 s 主模型权重]({rep}/best.pt) · [3 s 主模型权重](trained3/{rep}/best.pt) · [受控模型、数据构建、评分与制图代码](../../.worktrees/control-event-canonicalization-v1/experiments/controlled_multistage)
- 使用仓库 GPU 环境，从上述代码目录运行 `evaluate.py extrapolated3` 或 `evaluate.py trained3` 可分别从冻结权重复算两种设定。交互结构图通过 9 项交付检查，并以 Firefox 实际渲染核对；本机缺少 Chromium，Archify 的 Chromium 自动浏览器检查未执行。
- [共同起点核查](common_prefix_data.json) · [2 s 因果检查](2s_pipeline.json) · [3 s 外推因果检查](extrapolated3_pipeline.json) · [3 s 同训因果检查](trained3_pipeline.json)
"""
    (RUN/'report_zh.md').write_text(result)


if __name__=='__main__': write_report()
