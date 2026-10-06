"""Report rolling one-step evidence separately from parent-origin forecasts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_control_windows_v2 as base
from audit_controller_support_v26 import verify_parent
from audit_rolling_history_v33 import OUT,DATA,V23,V28,V30,V32
from rolling_history_v33 import VARIANTS
from rollout_trace_v32 import accumulated_rates


def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    for path,digest in protocol['code'].items():assert base.sha(base.ROOT/path)==digest
    for root in [DATA,V23,V28,V30,V32]:verify_parent(root)
    m=pd.read_csv(OUT/'per_window.csv');clip=pd.read_csv(OUT/'clipping.csv')
    cols=['local_mse','accumulated_mse','accumulated_end_mse','persistence_mse']
    flight=m.groupby(['arm','variant','axis','seed','log_id'])[cols].mean().pow(.5)
    s=flight.groupby(['arm','variant','axis']).mean();s.to_csv(OUT/'summary_rmse.csv')
    flight.groupby(['arm','variant','axis','log_id']).mean().to_csv(OUT/'by_flight_rmse.csv')
    meanbias=m.groupby(['arm','variant','axis','seed','log_id']).derivative_bias.mean().groupby(['arm','variant','axis']).mean()
    meanbias.to_csv(OUT/'signed_derivative_bias.csv')
    event=m[(m.array_index==97)&(m.arm=='actuator')];event.to_csv(OUT/'window97_metrics.csv',index=False)
    q=s.loc[('actuator','rolling_current','q')];r=s.loc[('actuator','rolling_current','r')]
    bq=s.loc[('baseline','rolling_current','q')];oldq=s.loc[('actuator','legacy_legacy','q')]
    focus=event[(event.seed==23)&(event.axis=='q')].set_index('variant')
    decision=dict(status='diagnostic_complete',windows_retained=407,rolling_anchors=20350,new_independent_training_windows=0,
        trained_models=0,models_evaluated=30,actuator_rolling_local_q_rmse=float(q.local_mse),
        baseline_rolling_local_q_rmse=float(bq.local_mse),previous_rate_q_rmse=float(q.persistence_mse),
        actuator_rolling_local_r_rmse=float(r.local_mse),previous_rate_r_rmse=float(r.persistence_mse),
        actuator_rolling_accumulated_q_rmse=float(q.accumulated_mse),legacy_accumulated_q_rmse=float(oldq.accumulated_mse),
        window97_seed23_q_mean_derivative_bias_before=float(focus.loc['legacy_legacy','derivative_bias']),
        window97_seed23_q_mean_derivative_bias_rolling=float(focus.loc['rolling_current','derivative_bias']),
        clipped_components=int(clip.clipped_components.sum()),model_promoted=False,baseline_retained='A1600',
        conclusion='Complete causal history re-encoding reduces some window97 anomalies but does not remove overall increment accumulation bias. Current decoder-context update alone has small overall effect; both model families show similar local/accumulation limitations. Offline crossed combinations do not establish unique physical cause.',
        next_step='Prepare and freeze a paired training protocol that retains the original full1s free-rollout loss and adds local increment supervision from rolling observed H26 histories in FIT flights only. Fix coefficient, parent sampling, local-anchor schedule and compute accounting before training; no weight sweep or window deletion. Full1s q and state-protection/absolute gates remain decisive, not local oracle improvement. Training is not started in v33.',
        validation_accessed=False,sealed_test_accessed=False,reserved_accessed=False)
    base.write(OUT/'decision.json',decision)
    # Human-readable trace: focus event was fixed by prior v31; no exclusions.
    with np.load(OUT/'rolling_inputs.npz') as z:
        true=z['label_rates_b'][97].astype(float);initial=z['input_initial_rates_b'][97,0].astype(float);dt=z['input_dt_s'][97,:,0].astype(float)
    with np.load(OUT/'fold1/actuator/seed23/trace.npz') as z:
        idx=np.flatnonzero(z['array_index']==97).item();trace={k:z[k][idx] for k in z.files if k!='array_index'}
    labels={'rolling_current':'new H26 + current context','legacy_current':'legacy hidden + current context',
            'rolling_legacy':'new H26 + legacy context','legacy_legacy':'legacy hidden + legacy context'}
    colors={'rolling_current':'tab:blue','legacy_current':'tab:orange','rolling_legacy':'tab:green','legacy_legacy':'tab:red'}
    t=np.arange(1,51)*.02;fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for ax in axes[0]:ax.plot(t,true[:,1],color='black',label='logged q')
    for name in VARIANTS:
        axes[0,0].plot(t,trace[name+'_rates_b'][:,1],color=colors[name],label=labels[name])
        a=accumulated_rates(initial[None],trace[name+'_derivative'][None],dt[None])[0,:,1]
        axes[0,1].plot(t,a,color=colors[name],label=labels[name])
    axes[0,0].set_title('Window97 seed23: repeated20ms q predictions')
    axes[0,1].set_title('Same event: offline sums of predicted increments')
    for ax in axes[0]:ax.set_xlabel('seconds after parent origin');ax.set_ylabel('rad/s');ax.legend(fontsize=7);ax.grid(alpha=.25)
    keys=['legacy_legacy','legacy_current','rolling_legacy','rolling_current'];ticks=['old H/old C','old H/new C','new H/old C','new H/new C']
    for ax,metric,title in [(axes[1,0],'local_mse','All407 windows: local q RMSE'),(axes[1,1],'accumulated_mse','All407 windows: offline accumulated q RMSE')]:
        for j,(arm,label) in enumerate([('baseline','v23'),('actuator','v30')]):
            ax.bar(np.arange(4)+(j-.5)*.35,[s.loc[(arm,k,'q'),metric] for k in keys],width=.35,label=label)
        ax.set_xticks(np.arange(4),ticks);ax.set_ylabel('rad/s');ax.set_title(title);ax.legend();ax.grid(axis='y',alpha=.25)
    axes[1,0].axhline(q.persistence_mse,color='black',linestyle='--',label='previous true rate');axes[1,0].legend(fontsize=8)
    fig.suptitle('v33: rolling observations are not a parent-origin1s forecast; sums are diagnostic only')
    fig.savefig(OUT/'rolling_comparison.png',dpi=160);fig.savefig(OUT/'rolling_comparison.svg');plt.close(fig)
    table=[]
    names={'rolling_current':'滚动H26＋当前上下文','legacy_current':'旧隐藏＋当前上下文','rolling_legacy':'滚动H26＋旧上下文','legacy_legacy':'旧隐藏＋旧上下文（v32）'}
    for arm,label in [('baseline','v23'),('actuator','v30')]:
        for name in ['legacy_legacy','legacy_current','rolling_legacy','rolling_current']:
            qr=s.loc[(arm,name,'q')];rr=s.loc[(arm,name,'r')]
            table.append(f'| {label} | {names[name]} | {qr.local_mse:.6f} | {rr.local_mse:.6f} | {qr.accumulated_mse:.6f} | {rr.accumulated_mse:.6f} |')
    events='\n'.join(f'| {names[name]} | {focus.loc[name,"derivative_bias"]:+.6f} | {np.sqrt(focus.loc[name,"local_mse"]):.6f} | {np.sqrt(focus.loc[name,"accumulated_end_mse"]):.6f} |' for name in ['legacy_legacy','legacy_current','rolling_legacy','rolling_current'])
    text=f'''# v33：完整H26历史重新编码的滚动单步对照

**完成30个冻结模型、原407窗内全部20350个滚动起点的诊断；未重训、删窗或晋升模型，保留A1600。**完整历史重编码减轻了窗口97的部分异常，但没有消除整体局部增量的持续偏差。不能把这一步当作控制可用模型已经得到。

## 观测和时间边界

沿用5个原train飞行、5折×3种子、原归一化与v23/v30检查点。每个父窗的原起点+k（k=0..49）建立26点历史，只预测其后20ms。历史状态/参考/当前命令为新起点−25..0，原滞后执行历史为−26..−1；预测输入为新起点当前命令，标签为+1。全部观测都位于原77点支持内，逐起点通过原valid/reset/时间间隔/模式检查。

从已哈希固定的9月缓存读取实际记录的相位、扑频和空速；相位仅相对各个新起点重设零，未推断物理零位，也未换成4月ratio8。单步解码器使用该起点最新已知扑频/空速，相对相位从0起步。角速度参考由v23已审计的源消息保持点重建，执行命令与v28网格逐值一致；不重新插值或拟合时延。

这些输入在每个**新的滚动时刻**可因果获得，但包含原父窗起点之后逐渐获得的观测，所以结果不是原起点自主预测1秒。50个滚动历史高度重叠，只是407个原父窗内的诊断观测，不计为20350个独立训练窗口，没有新划分。

## 交叉对照

主路径“滚动H26＋当前上下文”直接调用原模型的单步forward，重新编码完整观测历史。另三组仅作离线对照：交换新编码隐藏与v32真实状态递推隐藏，同时交换当前解码上下文与v32原起点固定相位推进/扑频/空速。物理状态和同刻控制保持一致，除了原代码四元数归一化的极小舍入差异。

新隐藏包含整段历史状态、历史命令和历史上下文的变化，不能把该因素叫作纯记忆作用；交叉组合也可能超出训练分布，不能直接部署。旧隐藏＋旧上下文逐步复现v32，提供两轮诊断之间的核对桥梁。

## 同一批观测下的结果

下表均先飞行内汇总平方误差并开方，再等权平均飞行和种子，单位rad/s。局部列是反复观测下的20ms预测；累计列是从父窗初始角速度累加各路径预测导数×dt，仅用于显示有符号增量误差，**不是自洽动力学轨迹或可部署1秒预测**。

| 模型 | 路径 | q局部RMSE | r局部RMSE | q离线累计RMSE | r离线累计RMSE |
|---|---|---:|---:|---:|---:|
{chr(10).join(table)}

同一批点保持上一真实角速度的q/r RMSE为{q.persistence_mse:.6f}/{r.persistence_mse:.6f}。主路径v30分别改善{100*(1-q.local_mse/q.persistence_mse):.2f}%/{100*(1-r.local_mse/r.persistence_mse):.2f}%；但v23与v30的局部q分别{bq.local_mse:.6f}/{q.local_mse:.6f}，非常接近，执行命令输入并未在这项检查中带来明显额外收益。

v30主路径q离线累计误差{q.accumulated_mse:.6f}，高于v32旧路径{oldq.accumulated_mse:.6f}；问题没有被完整历史重编码普遍解决。跨窗等权平均q导数偏差仅{meanbias.loc[('actuator','rolling_current','q')]:+.6f}rad/s²，但各窗正负抵消，累计端点RMSE仍为{q.accumulated_end_mse:.6f}rad/s。不能据总体平均接近0就认为局部无偏，也不能直接加一个全局常数修正。

仅更新当前解码上下文时，全局指标变化相对较小；改变历史编码后，累计误差仍明显。因此，上一轮的总体结论不能仅归因于固定扑频/空速或沿用了旧隐藏。不过这仍不排除相位质量、采样差异、局部状态支持或训练目标的影响。

## 窗口97：局部改善与总体问题并存

窗口97/seed23的q诊断如下。平均导数误差单位rad/s²，最后一列为1秒离线累计端点绝对误差rad/s：

| 路径 | 平均导数误差 | 20ms局部RMSE | 累计端点绝对误差 |
|---|---:|---:|---:|
{events}

完整滚动路径将此例平均导数偏差从+15.764369降到+3.596759，说明隐藏/上下文组合确实放大了上一轮这个例子的异常，不能把原oracle数值当作固有舵效错误。但改善不等于消除，其他窗口的偏差也仍在；全部窗口保留在汇总中。

![滚动单步与离线累计对照](rolling_comparison.png)

[矢量图](rolling_comparison.svg)。图中的累计曲线和底部累计柱状图均为诊断，不是新预测器效果。

## 核验与下一步

3项测试通过：滚动时序/相位重设/边界、未来记录不能进入此前输入、隐藏与上下文相同则交叉路径相同。407个原起点输入逐值复现，20350个标签逐值等于原数据；全部30模型的新主路径与原生forward一致，旧路径逐步复现v32的rawhead/导数/角速度，指标一致。每种子每组407窗唯一，检查点/父产物/代码哈希未变。所有分支输出限幅触发{int(clip.clipped_components.sum())}次。未读取开发验证、sealed或reserved。

下一步建议进入**联合监督训练对照的准备**：保持原完整1秒自由递推损失，增加来自fit飞行滚动H26观测历史的局部增量监督；先冻结系数、父窗采样、局部起点抽样和计算量口径，不扫描权重、不删困难窗。相同初始化与训练预算下，仍以完整1秒q、其他状态保护和绝对精度作为验收，不能以局部预测更好替代控制能力。旧v18的原起点首步q辅助损失不等同于此方案，也未证明该思路必然有效。

本轮没有准备最终训练系数或启动新训练；结果只是支持检验“局部增量与长递推目标是否兼容”这一假设，不是唯一根因证明。五飞行OOF已被多轮开发使用，后续仍需独立响应与闭环证据。

入口：[冻结协议](protocol.json)、[数据血缘](lineage.csv)、[数据合同](dataset_manifest.json)、[结论](decision.json)、[验证](verification.json)、[逐窗口](per_window.csv)、[逐飞行](by_flight_rmse.csv)。滚动输入位于rolling_inputs.npz；各模型内部结果位于fold*/{{baseline,actuator}}/seed*/trace.npz。
'''
    (OUT/'report_zh.md').write_text(text)
    code=dict(protocol['code']);code[str(Path(__file__).resolve().relative_to(base.ROOT))]=base.sha(__file__)
    base.write(OUT/'delivery_manifest.json',dict(artifacts={str(p.relative_to(OUT)):base.sha(p)
        for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'},code=code,
        models_evaluated=30,trained_models=0,model_promoted=False,validation_accessed=False,sealed_test_accessed=False))
    print(json.dumps(decision,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
