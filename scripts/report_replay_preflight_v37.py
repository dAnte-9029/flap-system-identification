"""Publish replay entry evidence, partial-state contract and bounded next stage."""
from pathlib import Path
import numpy as np
import pandas as pd
import train_control_windows_v2 as base
from audit_replay_preflight_v37 import OUT,PARENTS,read
from audit_controller_support_v26 import verify_parent


def main():
    protocol=read(OUT/'protocol.json')
    assert read(OUT/'verification.json')['status']==read(OUT/'independent_verification.json')['status']=='passed'
    for name,digest in protocol['code'].items():assert base.sha(base.ROOT/name)==digest
    p=read(OUT/'parameters_by_flight.json');assert all(r['values']==p[0]['values'] for r in p)
    d=[p[0]['values'][f'FW_{axis}R_D'] for axis in 'RPY'];assert d==[0,0,0]
    a=pd.read_csv(OUT/'origin_observation_age.csv');c=pd.read_csv(OUT/'cadence_summary.csv')
    ages=a.groupby('topic').publication_age_ms.agg(['median','max'])
    table='\n'.join(f'| {r.topic} | {r.median_of_window_medians_ms:.3f} | {ages.loc[r.topic,"median"]:.3f} | {ages.loc[r.topic,"max"]:.3f} |' for r in c.itertuples())
    clock=pd.read_csv(OUT/'clock_summary.csv');ct='\n'.join(f'| {r.log_id.split("/")[-1]} | {r.axis} | {r.count} | {r.median*1000:.6f} |' for r in clock.itertuples())
    checks=read(OUT/'independent_verification.json')
    states=[
        ('固件与参数','identified','5日志c9af，相关参数一致，无变更；不是二进制/实物接线认证'),
        ('积分','partial','status为更新后，terms为更新前；起点最新记录仍有年龄，不是同周期完整状态'),
        ('压缩增益与能量LPF','partial','约100ms发布；最新观测可提取但不是起点真实内部状态'),
        ('压缩signed HPF','not_identifiable_from_field','日志getSpectralDamperHpf返回_hpf*_hpf，符号丢失'),
        ('压缩previous effort','not_directly_logged','_input_prev为压缩器上一周期输入control_u，不是任意最新PWM/限幅后torque'),
        ('空速滤波与缩放','partial','原始空速可用；1秒时间常数内部滤波值未直接记录，代数反推也只覆盖部分历史时刻'),
        ('姿态yaw欧拉率记忆','partial','roll/pitch使用上一周期yaw欧拉率，然后更新yaw；body rate setpoint不直接等价于此状态'),
        ('飞手yaw叠加','partial','Stabilized姿态链另加yaw杆量，约200ms日志；只给姿态四元数不足以完整复现'),
        ('控制器/分配时钟','not_exact','日志抽样缺少中间更新、uORB读取顺序和最后运行时刻；原生周期约2.5ms'),
        ('分配与反饱和','mapping_verified_state_partial','静态分配已有同sample验证；反馈状态约200ms，不能恢复所有周期的旗标'),
        ('角加速度','not_active_D_path','三轴D=0，此配置角加速度缺失不会通过D项产生力矩；不假定其他配置如此'),
        ('电池/轮控/执行器slew','inactive_under_frozen_config','FW_BAT_SCALE_EN=0、FW_W_EN=0、舵机slew=0；不是永久忽略这些路径'),
        ('模型持久状态','adapter_required','现接口整段forward；闭环需H26初始化一次并保持GRU hidden/物理状态/相位，同一命令带必须重现原50步'),
    ]
    base.write(OUT/'state_recovery_matrix.json',[dict(component=x,status=y,evidence=z) for x,y,z in states])
    next_plan=dict(status='prepared_not_executed',name='c9af controller component replay before learned-model closed loop',
        source_contract=str((OUT/'source_provenance.json').relative_to(base.ROOT)),
        origin_manifest=str((OUT/'origin_observations.json').relative_to(base.ROOT)),
        allowed_logs='Exactly5originaltrain sources pinned here; all407 parents retained; causal preceding raw history may be used for warmup',
        stages=[
            dict(id='source_adapter',work='Implement exact-c9af attitude/rate/compression/allocation adapter with explicit serializable state, manual yaw addition and publication/update ordering. Do not inject e624-only FLAP_SLOW_EN default.',verification='C++/Python component arithmetic on same supplied complete states and inputs; sign-sensitive HPF, pre/post integral, old/new compression gain and preceding allocation feedback tests.'),
            dict(id='teacher_state_replay',work='Drive controller from available logged actual state/reference history; compare rate setpoints, torque, allocated commands. Future measured states are allowed ONLY in this named controller-component diagnostic, not a model forecast.',verification='Full407 parents, perflight/peraxis errors and timing/coverage. Initial conditions explicit; cold/causal warmup differences recorded. Never inject future logged integrals/gains/torque as generated controller state or fit gains to targets.'),
            dict(id='state_clock_sensitivity',work='Before selecting a replay recipe freeze warmup durations, initial-state variants and clock approximations; report all variants rather than selecting best against targets.',verification='Signed HPF and previous-effort ambiguity, held sensor and reference updates, allocator feedback timing. dt inference is diagnostic only; no claim that resampling recovers missing cycles.'),
            dict(id='model_adapter',work='Expose persistent H26 initialized GRU step at20ms; causal model-generated future actions only. Do not reencode hidden from future real state or feed observed future phase/airspeed.',verification='With prerecorded identical command tape, every step matches frozen native50step output; save/restore and future-prefix causality tests. Tapes used here for adapter parity only.'),
            dict(id='conditional_closed_loop_gate',work='Only after controller reproduction and model-adapter evidence, freeze an explicit multirate approximation and acceptance criteria before learned-model replay.',verification='Recorded attitude/thrust/yaw tapes are conditioning, not Mission/TECS/navigation replay or independent exogenous excitation. Compare both states/actions; track causal past-state and marginal command support; do not use offline future-support bins as online confidence.')],
        no_new_training=True,no_new_controller_design=True,no_real_flight=True,
        exact_replay_not_available_from_partial_snapshots=True)
    base.write(OUT/'next_execution_plan.json',next_plan)
    base.write(OUT/'decision.json',dict(status='replay_entry_audit_complete',exact_closed_loop_entry_ready=False,
        approximate_closed_loop_started=False,partial_origin_snapshots=407,controller_adapter_compatible=False,
        missing_signed_hpf=True,missing_previous_compression_input=True,native_vs_model_clock_mismatch=True,
        next_stage='Implement c9af source-parity controller adapter, then logged-state controller-only replay and explicit initialization/clock sensitivity; no learned-model closed loop yet.',
        no_new_flight_requirement_inferred=True,model_promoted=False,baseline_retained='A1600',
        validation_accessed=False,sealed_test_accessed=False,reserved_accessed=False))
    state_table='\n'.join(f'| {x} | {y} | {z} |' for x,y,z in states)
    report=f'''# v37：同一PX4内环闭环重放入口审计

**完成5个原始日志、407起点、固件/参数/时间与内部状态的核查。完整闭环重放入口目前未通过：现有组件版本不匹配，日志不能直接给出完整同步内部状态，模型与控制器时钟也不同。本轮未运行控制器或神经模型闭环。**

已交付可追溯的407起点部分状态观测、c9af精确源码快照、缺口矩阵和下一阶段执行合同。未重训、删除窗口、访问开发验证/sealed/reserved，也没有据此要求新增飞行。

## 可以确认的部分

五ULog哈希与v23源清单逐一一致，均标记固件c9af571575af33d5403fbb479256848797c801e2；FW/CA/VT/TRIM/PWM_MAIN/FLAP相关参数五飞行一致，无参数变更。参数完整保存，源码按该修订导出并记录Git blob与SHA256，不用当前PX4分支替代。

已有v28同sample分配核验仍保持：30853组输出满足固定翼分配映射，最大残差约1.53e−7。故静态分配算式有可复用证据；这不意味着反饱和反馈在每个控制周期都已记录。

本地`px4_pitch.py`和`px4_flight_chain.py`声明e624固件。将本批真实参数交给现有FlightInnerLoop，五飞行均得到`KeyError: FLAP_SLOW_EN`；该参数在这批c9af日志不存在。不能补一个默认0就把版本差异当成已经验证。下一步应建c9af对应适配器，哪些算式可复用应逐项对源码验证。

## 已提取的起点状态不是完整restore快照

[origin_observations.json](origin_observations.json)包含407起点的18类最新可用话题，共7326条观测，逐条保留原始索引、发布时间、sample时间及年龄。选取时同时要求publication和sample≤origin；没有用起点之后的积分/滤波量回填。字段无sample时明确标记，不能把发布时间当传感器采样时间。

原始无效或未使用通道的NaN保存为null，并列入nonfinite_fields，不填零。首次导出被严格JSON拒绝非有限值，已将未完成产物保留于control_windows_v37_initial_serialization；修正的是序列化合同，未改变数据、状态选择或门槛。

| 状态/组件 | 恢复状态 | 证据及限制 |
|---|---|---|
{state_table}

**特别重要：`getSpectralDamperHpf()`返回的是`_hpf * _hpf`。**已知平方不能确定符号；即使LPF和增益数值都知道，下一次更新仍依赖带符号HPF和`_input_prev`。测试给出两组相同观测平方的隐藏状态，在同一后续输入下产生不同的下一步能量。这是字段信息不足的直接反例，不是按模型误差猜测。

压缩器input为限幅/trim之前的control_u，不能无条件拿最新已发布力矩或PWM替代。积分也有阶段区别：terms.i_term先用于输出，随后更新积分，再发布status。增益是先用getGains产生输出，再update并按约100ms发布；异步保持值不是输出周期的精确used_gain。

## 更新频率和年龄

下表间隔只计算原父窗内部相邻发布，不将窗口之间空隙算丢包；年龄来自407个起点。

| 话题 | 窗内发布间隔中位数ms | 起点年龄中位ms | 起点年龄最大ms |
|---|---:|---:|---:|
{table}

对时间相容的terms/status配对，在积分未贴限幅且分母充分非零时，用Pterm/Kp得到误差，再从单次积分增量反推dt。保留68538个轴级估计；要求落入源码2–40ms范围，假设该次没有额外reset/antiwindup干预。它只是条件时钟证据，不是完整原生周期序列或新的插值时钟。

| 飞行 | 轴 | 有效估计数 | dt中位ms |
|---|---|---:|---:|
{ct}

至少两轴同时可估的记录有{checks['multiaxis_clock_rows']}条，轴间dt极差中位{checks['multiaxis_dt_spread_median_us']:.4f}μs、P95={checks['multiaxis_dt_spread_p95_us']:.4f}μs。大部分约2.495ms，log_18约2.464ms，支持控制器约400Hz的判断；不能直接把约20ms日志间隔当成控制周期。gyro记录约10ms，增益压缩约102ms，分配状态约200ms，未记录的中间更新无法凭重采样变成真实观测。

神经模型固定50Hz/20ms。每20ms只更新一次控制器会改变内环；每20ms内细分约8次、保持同一模型状态也引入新的时序近似。两者均须明确标记、先用真实状态输入的控制器组件重放检验，不把近似称为同固件原生闭环保真。

本批三轴D=0，因此不能把角加速度缺失泛化为本配置D项障碍；电池油门缩放和轮控关闭、舵机slew为0，也已按真实配置区分。这些简化不能补齐压缩器、空速滤波、姿态yaw记忆及原生时钟缺口。

## 重放边界与下一步执行顺序

计划首先固定在“姿态参考/油门/yaw参考→同PX4姿态/角速度内环→分配→模型”的条件重放边界。Stabilized保留飞手yaw附加路径；Mission若使用记录的姿态/油门带，只是在固定外层参考下复现内环，不是完整Mission导航/TECS闭环或独立外生激励。未来执行命令必须由模型状态驱动控制链产生，未来日志执行命令只用于比较，不能回灌。

下一阶段的直接工作是**实现c9af同源最小控制链，然后以真实记录状态驱动，比较它生成的rate参考、力矩和分配命令**，先把控制器重放误差与神经动力学误差分开。方案已写入[next_execution_plan.json](next_execution_plan.json)：

1. c9af适配器拥有显式、可序列化内部状态，按实际调用顺序实现姿态yaw记忆、飞手yaw、积分、压缩与分配反馈；先做同完整状态下的源码算式对照。
2. 仅在明确命名的控制器组件诊断中使用未来真实状态；不得把其结果当神经模型预测。比较全部407父窗，不调PID或按输出误差挑初始化。
3. 初始化预热时长、缺失状态分支和时钟近似先冻结，再全量报告敏感性。预热后收敛需要测量，不能先假定；有符号HPF及前输入的估计仍不是精确restore。
4. 准备模型持久step适配器：只在起点编码H26，之后保持hidden/物理状态/相位。给同一预录命令带时必须复现原50步预测；这项接口对照可以用日志命令，正式闭环不能。
5. 上述证据齐备后再冻结多速率近似与数值验收标准，决定是否跑受限闭环。只用因果过去支持/已生成动作做在线范围检查，不能把v31离线未来支持当在线可信门。

本轮不执行未准备的闭环，不把新增动作限幅当作模型成功，也不设计新控制器。现有日志可以继续支持组件复现和显式不确定性的诊断，当前缺口尚不能推出“必须补飞”或“控制模型永远不可用”。

## 验证

3项测试通过：publication/sample双时间因果边界、积分dt反推和限幅拒绝、HPF平方观测不唯一的反例。独立从原始ULog复算全部7326条起点观测和68538个dt估计，非有限值语义保持。父产物与源码哈希不变；全部407父窗保留。入口决定见[decision.json](decision.json)，当前exact_closed_loop_entry_ready=false。
'''
    (OUT/'report_zh.md').write_text(report)
    for root in PARENTS:verify_parent(root)
    code=dict(protocol['code'])
    for path in ['scripts/report_replay_preflight_v37.py','scripts/verify_replay_preflight_v37.py']:code[path]=base.sha(base.ROOT/path)
    base.write(OUT/'delivery_manifest.json',dict(artifacts={str(p.relative_to(OUT)):base.sha(p)
        for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'},code=code,
        replay_started=False,model_promoted=False,validation_accessed=False,sealed_test_accessed=False,reserved_accessed=False))
    print(read(OUT/'decision.json'),flush=True)


if __name__=='__main__':main()
