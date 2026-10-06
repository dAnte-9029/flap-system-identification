"""Five-stage receipt, with implementation erratum and unmet gates explicit."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_reference_unit_gravity import OUT,ART,PARENT,SEEDS,base


def main():
    summary=json.loads((OUT/'evaluation/summary.json').read_text())
    response=json.loads((OUT/'response/status.json').read_text())
    probe=json.loads((OUT/'action_probe/summary.json').read_text())
    selection=json.loads((PARENT/'selection_protocol.json').read_text())
    old=json.loads((base.DATA/'evaluation/summary.json').read_text())
    gates=dict(absolute_prediction=summary['absolute_control_candidate_passed'],
        observable_waveforms=response['G2_observational_thresholds_passed'],
        conservative_effect_to_error=bool(len(probe)==2 and all(r['error_to_effect_ratio']<=1 for r in probe)))
    decision=dict(prerequisites=gates,optimizer_started=False,hardware_or_flight_tests_started=False,
        status='prerequisites_failed' if not all(gates.values()) else 'optimizer_required_not_yet_executed',
        interpretation='error/effect test is conservative; shared error can cancel in action differences, so failing it is not proof of wrong derivatives or impossibility',
        sealed_test_accessed=False)
    base.write(OUT/'control_decision.json',decision)
    # A positive gate result must trigger subsequent work; never label it completed here.
    if all(gates.values()):raise RuntimeError('Offline optimizer stage now required; do not finalize yet')
    metrics=pd.read_csv(OUT/'evaluation/flight_equal_rmse.csv')
    diagnostics=pd.read_csv(OUT/'diagnosis/summary.csv')
    table=[]
    for arm in ['A','B','C','D','A_long']:
        score=summary['scores_by_arm'][arm]
        table.append(f"| {arm} | {' / '.join(f'{v:.5f}' for v in score)} | {np.mean(score):.5f} |")
    compare=[]
    for arm,value in summary['comparisons'].items():
        for baseline,result in value['versus'].items():
            lo,hi=result['flight_bootstrap_95ci']
            compare.append(f"| {arm} 对 {baseline} | {result['improvement_fraction']*100:+.2f}% | [{lo*100:+.2f}%, {hi*100:+.2f}%] |")
    absolute=summary['absolute_research_gate']['A_long']['seeds']
    absrows=[f"| {v['seed']} | {v['p']:.4f} | {v['q']:.4f} | {np.rad2deg(v['attitude_rad']):.2f} | {v['velocity']:.4f} | {v['passed']} |" for v in absolute]
    actions=[f"| {r['axis']} | {r['windows']} | {r['predicted_effect_rms']:.5f} | {r['observed_error_rms']:.5f} | {r['error_to_effect_ratio']:.1f} | {r['ridge_sign_agreement_fraction']*100:.1f}% |" for r in probe]
    diag=diagnostics[(diagnostics.arm=='A_long')&(diagnostics.subset=='all')&np.isclose(diagnostics.horizon_s,.5)]
    diagrows=[f"| {r.partition} | {r.intervention} | {r.p:.4f} | {r.q:.4f} | {r.velocity:.4f} |" for r in diag.itertuples()]
    improvement=1-summary['mean_by_arm']['A']/old['mean_by_arm']['A']
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    labels=['legacy A800','fixed A800','fixed A1600']
    values=[old['mean_by_arm']['A'],summary['mean_by_arm']['A'],summary['mean_by_arm']['A_long']]
    axes[0].bar(labels,values,color=['#a5a8ad','#437eb4','#36836e']);axes[0].set_title('Unit correction and update budget')
    for i,arm in enumerate(['A','B','C','D']):
        v=summary['scores_by_arm'][arm];axes[1].bar(i,np.mean(v),alpha=.7)
        axes[1].scatter(i+np.array([-.1,0,.1]),v,color='black',s=15,zorder=3)
    axes[1].set_xticks(range(4),['A all','B random','C selected','D weighted']);axes[1].set_title('Fixed units, same 800 updates')
    for ax in axes:ax.set_ylabel('Event-axis rate RMSE (rad/s)');ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
    text=f'''# 五步研究执行结果：控制导向数据与参考模型 v3

日期：2026-09-27；分支 `reference-dynamics-data-v1`。数据/诊断/训练/响应验收已执行；**第5步的控制优化与硬件试验未启动，因为前置证据未通过**。sealed/reserved 未打开。此前查看过的9个validation日志继续用于开发，不称新盲测。

## 最重要的修正

本轮找到并修复了之前参考模型实现中的特征单位错误：历史记录为单位重力方向，原解码器却使用9.80665倍的重力加速度，并套用同一归一化。详见[勘误](../gravity_units_erratum.md)。修复只改变解码器特征单位，保持40070参数架构、初始权重、输入、损失、数据与优化条件不变。

全部1146个预测起点的12维重建特征与记录历史一致性检查通过，新增回归测试覆盖重力方向及参考因果性。旧代码、旧checkpoint和旧报告保留作历史证据。**v2/v3 legacy 的负面结果不能被解释为旧日志已经没有控制辨识价值。**

同预算全量 A800 的主指标从旧实现 {old['mean_by_arm']['A']:.5f} 降至修复后 {summary['mean_by_arm']['A']:.5f} rad/s，改善 {improvement*100:.2f}%。这是实现修复收益，不能算成数据筛选收益。

## 第1步：定位瓶颈

先对原A模型复查train/validation与未来参考保持，再检查特征接口；单位错误由数据构建和解码器源代码及全部样本直接证实。修复后重做诊断，见[详细表](diagnosis/summary.csv)。A1600在0.5秒的全有效集结果如下，单位p/q为rad/s，速度为分量RMSE的m/s。

| 集合 | 未来参考 | p | q | 速度 |
|---|---|---:|---:|---:|
{chr(10).join(diagrows)}

参考保持仅检查网络使用输入的程度，不是实际飞机的反事实真值。[0.2秒因果平均角速度诊断](diagnosis/causal_200ms_mean_rate_rmse.csv)和[均值/波形误差分解](diagnosis/rate_error_components.csv)另行保存，未用滤波后的指标替换预注册原始状态门槛。滤波会改变延迟和建模对象。

当前扑频保持代理相对日志在0.5秒的相位偏差RMS约train0.606、validation0.552rad；频率变化RMS约0.247/0.206Hz。它们是诊断标签，不作为未来模型输入，也没有用于筛除难预测片段。相关性或这些误差本身不足以证明剩余误差全部来自相位。

## 第2步：产生新的筛选数据

沿用已审计的1146个非重叠H26/K50样本及14个同记录配置日志，保留原407train/739validation归属。没有重切窗口、移动验证飞行或读取封存日志。

评分只使用未来参考的变化范围和净变化/总变化比；单位归一化沿用每通道固定变化阈值。在每个train flight/mode内取前25%，时间顺序打破同分；不按输出方向、强弱或模型误差筛选。近限幅Q3从C分开，仍保留在A/D。

| 组 | 训练样本 | 规则 |
|---|---:|---|
| A | 407 | 全部有效、flight等权 |
| B | 105 | 匹配C的flight/mode配额随机取样 |
| C | 105 | 输入评分前25% |
| D | 407 | 全保留，C对应片段权重3、其余1；flight等权 |

B/C仅共享24个样本（22.9%），比v2的97.2%形成了明确差异。阈值仅从train输入确定，未因训练结果再调整。这是看过v2之后的新开发假设，不冒充未接触数据的确认性研究。

实际数据：`dataset/control_windows_v3/selected_C_train.npz`；来源、数组hash和统计口径见其 `manifest.json`。[C清单](../arm_C.csv)、[全部窗口与评分](../windows_v3.csv)、[冻结筛选协议](../selection_protocol.json)。训练使用原始冻结数组加索引清单，导出文件是同一105个样本的可独立读取副本。

## 第3步：受控训练与简单模型对照

修复后A/B/C/D均800次更新、batch32、AdamW lr3e-4、三个seed17/23/42；另有全量A1600。A800为同一1600步轨迹的固定中间checkpoint，推理不会消费训练随机数，等价于相同初始化的固定800/1600步比较；没有依据验证误差选epoch或seed。共15个修复版checkpoint，训练/评价均完成。

旧错误实现下的v3对照也保存，但主要评价以此`unit_gravity/`目录为准。[协议](training/protocol.json)、[完成状态](training/status.json)。checkpoint位于 `artifacts/control_windows_v3_unit_gravity/<arm>_<seed>/final.pt`，必须使用单位方向版本的模型加载，专用加载器检查`feature_contract=unit_gravity_v1`。

简单岭回归基线固定alpha=10，训练集拟合归一化和参数，按每个终点只读取截至该终点的参考；保存5折按训练flight留一预测。它在部分短时角速度指标上优于旧GRU，但长时速度较差；不是可直接积分的完整动力学模型，不能替代控制模拟器。[基线结果](ridge/summary.csv)。

主指标仍为活动roll的p与活动pitch的q、0.1/0.2/0.5秒、flight等权RMSE，再对轴/时域/seed等权；所有组使用同一739窗口。

| 模型 | seed17 / 23 / 42 | 均值rad/s |
|---|---|---:|
{chr(10).join(table)}

| 比较 | 改善（正数更好） | flight聚类95%区间 |
|---|---:|---|
{chr(10).join(compare)}

![修复、预算与数据对照](comparison.png)

[全部指标](evaluation/flight_equal_rmse.csv)、[逐flight指标](evaluation/per_flight_rmse.csv)、[门槛逐项结果](evaluation/summary.json)。数据策略的相对改善与模型控制资格分开判断；平均预测改善不自动等于动作效果已经学对。

实际对照显示：C对A退化约4.12%，对等数量随机B仅改善约1.03%，后者区间包含零改善；D对A退化约0.53%。因此这次输入排名筛选/加权不应替代全量基线。A1600的主指标改善约12.58%，主要来自俯仰预测；但全有效集短时r误差增加约3.41%，保持层增加约5.91%，分别超过预设3%和5%的保护门槛，不能只报告改善项。

## 第4步：绝对预测、完整响应和小幅动作证据

新训练前冻结的研究目标为0.5秒roll-active p与pitch-active q各≤0.15rad/s、全有效集姿态≤3°、速度分量RMSE≤0.5m/s，三个seed都需满足。这些是研究容差，不是已经验证的飞行安全边界。

| A1600 seed | p | q | 姿态° | 速度m/s | 通过 |
|---|---:|---:|---:|---:|---|
{chr(10).join(absrows)}

本轮对全部19个完整validation事件簇、5个flight做了完整上下文滚动与原1秒预测的分开报告。完整自由滚动最长约38秒，超出训练时域；其失败不能单独否定1秒内用途。所有弱/不可测响应保留，只为峰值评价标记可测性，不用于重新筛选训练。

[完整波形、幅值、峰值时刻、终段均值和交叉轴诊断](response/full_context_waveforms.csv)、[完整上下文汇总](response/full_context_summary.csv)、[上下文内原1秒样本](response/nominal_summary.csv)、[响应门槛](response/status.json)。混合参考自然记录中的峰值对齐不是独立舵机物理时延，也不是裸机因果舵效。完整事件的手动yaw/thrust覆盖不足，不能泛化到全部通道。

对满足训练边际参考范围的窗口，追加±1°、0.2秒渐入的滚转/俯仰参考敏感度诊断；不是控制优化，也没有实机动作。它比较模型预测的动作差异与原记录预测误差，并检查另一个模型族的方向一致性。

| 轴 | 窗口 | 预测动作差异RMS | 原响应误差RMS | 误差/差异 | 岭回归方向一致 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(actions)}

[逐窗口证据](action_probe/per_window.csv)。误差大于动作差异是保守的证据不足提示，**并不证明控制导数一定错误**：共同误差可能在动作差分中抵消，但当前记录尚未独立验证这种抵消。边际范围内的扰动也不等于已确认联合反事实支持。

## 第5步：控制设计是否启动

判定：[control_decision.json](control_decision.json)。绝对预测门槛={gates['absolute_prediction']}；完整可测波形门槛={gates['observable_waveforms']}；保守动作差异/误差检查={gates['conservative_effect_to_error']}。

由于前置证据未通过，未运行控制优化器、台架或实飞。已保存[受限外层参考优化协议](../control_research_protocol.json)，模型已包含PX4内环，不重复串入内环。不能通过在同一个模型上优化成功来绕过这些失败项。

本轮完成了可执行的筛选数据、实现修复、15个修复版模型、统一评价和完整事件诊断；尚未得到通过控制用途验收的模型。下一轮应针对这些具体失败项研究可预测的时间尺度、扑频/相位推进与控制响应误差，而不是继续按“响应听话”删数据，也不能把旧实现错误导致的结论当成必须补飞的证据。

验证：21项针对性测试通过；全部1146个起点特征一致；相同seed的修复前后初始参数逐项相同；修复版checkpoint重载预测与保存结果完全一致。见[验证记录](verification.json)。未安装依赖，未改动原始日志或解封test，也未向飞控发送控制命令。
'''
    (OUT/'report_zh.md').write_text(text)
    receipt=dict(stage1='complete, includes unit-contract correction and repeated diagnostics',stage2='complete105 selected training samples',
        stage3='complete15 corrected checkpoints plus legacy diagnostics',stage4='complete, acceptance gates not all met',
        stage5='not started: quantitative prerequisites failed',sealed_test_accessed=False,
        report_sha256=base.sha(OUT/'report_zh.md'))
    base.write(OUT/'execution_receipt.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
