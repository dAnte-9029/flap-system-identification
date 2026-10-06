"""Independent raw-message reconstruction and final actuator input contract."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pyulog import ULog
import train_control_windows_v2 as base
from audit_actuator_contract_v28 import OUT,V23,V27
from actuator_contract_v28 import CHANNELS,PORTS,stats


def table(frame):
    def fmt(v):return f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v)
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])


def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    for p,h in protocol['code'].items():assert base.sha(base.ROOT/p)==h
    parent=json.loads((V23/'dataset_manifest.json').read_text());w=pd.read_csv(V23/'windows.csv')
    with np.load(OUT/'actuator_grid.npz') as z:data={k:z[k] for k in z.files}
    checked=[]
    for entry in parent['sources']:
        log=entry['log_id'];source=entry['source'];assert source['partition']=='train' and base.sha(source['path'])==source['sha256']
        ids=w[w.log_id==log].array_index.to_numpy();q=data['query_us'][ids].ravel()
        u=ULog(source['path'],message_name_filter_list=['actuator_motors','actuator_servos','actuator_outputs']);ds={d.name:d.data for d in u.data_list if d.multi_id==0}
        raw=[]
        for j,topic in enumerate(['actuator_motors','actuator_servos']):
            d=ds[topic];ix=np.searchsorted(np.maximum(d['timestamp'],d['timestamp_sample']).astype(np.int64),q,side='right')-1;assert (ix>=0).all()
            np.testing.assert_array_equal(d['timestamp'][ix],data['publication_us'][ids,:,j].ravel())
            np.testing.assert_array_equal(d['timestamp_sample'][ix],data['sample_us'][ids,:,j].ravel())
            cols=['control[0]'] if j==0 else [f'control[{k}]' for k in range(3)]
            raw.extend(d[c][ix] for c in cols)
        np.testing.assert_array_equal(np.column_stack(raw).astype(np.float32),data['command_grid'][ids].reshape(-1,4))
        d=ds['actuator_outputs'];ix=np.searchsorted(d['timestamp'],q,side='right')-1
        np.testing.assert_array_equal(np.column_stack([d[f'output[{p-1}]'][ix] for p in PORTS]),data['pwm_grid'][ids].reshape(-1,4))
        checked.append(dict(log_id=log,points=len(q),raw_commands_times_pwm_exact=True))
    g=pd.read_csv(OUT/'grid_points.csv');c=pd.read_csv(OUT/'coverage_per_window.csv');t=pd.read_csv(OUT/'native_timing.csv')
    a=pd.read_csv(OUT/'allocation_exact_pairs.csv');pwm=pd.read_csv(OUT/'pwm_comparison.csv');n=pd.read_csv(OUT/'native_sample_counts.csv')
    ages=[]
    for name in ['motor_publication_us','servo_publication_us','motor_sample_us','servo_sample_us','output_publication_us']:
        ages.append(dict(field=name,**stats((g.query_us-g[name])*.001)))
    ages=pd.DataFrame(ages);ages.to_csv(OUT/'age_summary.csv',index=False)
    output_summary=[]
    for channel,h in pwm.groupby('channel'):
        output_summary.append(dict(channel=channel,**stats(h.residual_us),abs_p95_us=float(np.quantile(abs(h.residual_us),.95)),
            fraction_within_1us=float(np.mean(abs(h.residual_us)<=1))))
    output_summary=pd.DataFrame(output_summary);output_summary.to_csv(OUT/'pwm_summary.csv',index=False)
    native=t.groupby('topic').agg(windows_with_two_rows=('array_index','size'),median_of_window_medians_ms=('period_median_ms','median'),max_gap_ms=('period_max_ms','max')).reset_index()
    native.to_csv(OUT/'native_timing_summary.csv',index=False)
    cov=c.groupby('channel').agg(windows=('array_index','size'),constant_future_windows=('future_changes',lambda x:int((x==0).sum())),
        changes_min=('future_changes','min'),changes_median=('future_changes','median'),messages_min=('future_distinct_messages','min'),messages_median=('future_distinct_messages','median'),
        range_median=('future_range','median'),range_max=('future_range','max'),near_bound_windows=('near_bound_fraction',lambda x:int((x>0).sum()))).reset_index()
    cov.to_csv(OUT/'coverage_summary.csv',index=False)
    # Independently recover the configured formula, checking all joined samples.
    parameters=json.loads((OUT/'parameters_by_flight.json').read_text());params=parameters[0]['values']
    # CSV uses shortest float32 decimal representations; restore the source dtype
    # before double precision algebra, especially at half-microsecond rounding ties.
    raw_columns=['actuator_motors_control[0]']+[f'actuator_servos_control[{j}]' for j in range(3)]+[f'vehicle_torque_setpoint_xyz[{j}]' for j in range(3)]+['vehicle_thrust_setpoint_xyz[0]']
    source_values=a[raw_columns].astype(np.float32).astype(float)
    roll=source_values['vehicle_torque_setpoint_xyz[0]'];pitch=source_values['vehicle_torque_setpoint_xyz[1]'];yaw=source_values['vehicle_torque_setpoint_xyz[2]']
    expected=np.column_stack([source_values['vehicle_thrust_setpoint_xyz[0]'].clip(0,1),(.5*pitch-roll/(2*params['CA_SV_CS1_TRQ_R'])).clip(-1,1),
                              (.5*pitch+roll/(2*params['CA_SV_CS1_TRQ_R'])).clip(-1,1),yaw.clip(-1,1)])
    actual=source_values[['actuator_motors_control[0]']+[f'actuator_servos_control[{j}]' for j in range(3)]].to_numpy()
    assert np.max(abs(actual-expected))<2e-6
    allocation_summary=pd.DataFrame([dict(channel=ch,**stats(actual[:,j]-expected[:,j])) for j,ch in enumerate(CHANNELS)])
    allocation_summary.to_csv(OUT/'allocation_summary.csv',index=False)
    # Verify normal map independently from the already exported commands/parameters.
    assert params['THR_MDL_FAC']==0 and params['PWM_MAIN_REV']==17
    for ch,slope,offset in [('motor0',800,1100),('servo0',-600,1600),('servo1',600,1600),('servo2',-500,1300)]:
        h=pwm[pwm.channel==ch]
        np.testing.assert_array_equal(np.floor(offset+slope*h.command.to_numpy(dtype=np.float32).astype(float)+.5),h.predicted_pwm)
    g['allocator_pub_us']=(g.query_us-g.allocator_status_age_ms*1000).round().astype(np.int64)
    unallocated=g[g.torque_achieved==0]
    sat_windows=[]
    for row in w.itertuples():
        group=g[(g.log_id==row.log_id)&(g.query_us>=row.start_timestamp_us)&(g.query_us<=row.end_timestamp_us)]
        sat_windows.append(dict(array_index=row.array_index,log_id=row.log_id,limited_grid_points=int((group.torque_achieved==0).sum()),
            limited_status_messages=len(group[group.torque_achieved==0][['log_id','allocator_pub_us']].drop_duplicates())))
    pd.DataFrame(sat_windows).to_csv(OUT/'allocation_limits_per_window.csv',index=False)
    flags=dict(torque_unachieved_grid_points=len(unallocated),torque_unachieved_unique_status_messages=len(unallocated[['log_id','allocator_pub_us']].drop_duplicates()),
        affected_windows=sum(x['limited_grid_points']>0 for x in sat_windows),mixed_motor_servo_sample_grid_points=int((g.motor_sample_us!=g.servo_sample_us).sum()),
        failure_mask_nonzero=int((g.failure_mask!=0).sum()),stop_mask_nonzero=int((g.stop_mask!=0).sum()))
    base.write(OUT/'chain_flags.json',flags)
    independent=json.loads((OUT/'input_independence.json').read_text())
    independence=pd.DataFrame([dict(log_id=x['log_id'],motor_step_std=x['delta_std'][0],common_step_std=x['delta_std'][1],differential_step_std=x['delta_std'][2],yaw_step_std=x['delta_std'][3],singular_ratio=x['singular_min_max_ratio']) for x in independent])
    checked_json=json.loads((OUT/'verification.json').read_text());checked_json.update(independent_raw_reconstruction=checked,
        independent_allocation_and_pwm_formula=True,focused_tests_passed=3)
    base.write(OUT/'verification.json',checked_json)
    contract=dict(dataset_id='actuator_command_contract_v28',status='audited_conditional_inputs_not_trained',windows=407,partitions=['original_train'],
        parent_manifest=str((V23/'dataset_manifest.json').relative_to(base.ROOT)),parent_manifest_sha256=base.sha(V23/'dataset_manifest.json'),
        original_window_manifest=str((V23/'windows.csv').relative_to(base.ROOT)),windows_sha256=base.sha(V23/'windows.csv'),
        artifact='actuator_grid.npz',artifact_sha256=base.sha(OUT/'actuator_grid.npz'),
        channels=CHANNELS,units=['normalized_motor_thrust_request','normalized_servo_command','normalized_servo_command','normalized_servo_command'],
        history='command_grid[:,0:26] is origin-26..origin-1, exactly existing v23 past actuator history.',
        future='command_grid[:,26:76] is origin..origin+49, four left-endpoint controls paired with unchanged states origin+1..origin+50.',
        grid='query_us contains origin-26..origin+50,77points,dt20ms; finalcommand row is audit context, not an extra target/input.',
        timing='Separate causal hold by topic with publication AND sample<=query. Preserve timestamps and occasional mixed source cycles. No fitted lag/interpolation. 50Hz logging may omit intermediate commands.',
        outputs='pwm_grid is independently causal held actuator_outputs instance0 natural units microseconds for this hardware configuration; secondary chain check, not servo angle or measured actuator forces.',
        boundary='Normalized post-allocation command -> output mapping/driver/actuator/aircraft. Remove attitude/rate controllers and allocator from learned dynamics boundary. Firmware still matters for output mapping.',
        conditional_limit='Future flight-recorded actuator commands depend on feedback and disturbances. They are permissible ONLY for explicitly actuator-conditioned offline identification, not known inputs in a reference-driven autonomous prediction claim. Closed-loop use must generate actions from a controller rather than future logs.',
        identification_limit='Independent flight actuator position/force and actual high-rate command waveform are not reconstructed. Rich command variation and full numerical input rank do not prove independent causal effectiveness.',
        fold_authority='Unchanged v23 fold0..4 fit_indices/query_indices andseeds; no repartition. Fit-only unique-row normalization still to be prepared before training; no normalization fitted this audit.',
        phase_contract=parent['phase_contract'],frequency_contract=parent['frequency_contract'],validation_accessed=False,sealed_test_accessed=False)
    base.write(OUT/'input_contract.json',contract)
    base.write(OUT/'decision.json',dict(status='audit_complete',conditional_actuator_input_contract_available=True,windows_retained=407,
        complete_high_rate_actuator_waveform_recovered=False,physical_servo_angle_or_force_verified=False,
        training_started=False,model_promoted=False,baseline_retained='A1600',next_step='Prepare and freeze an actuator-conditioned same-window comparison: fold normalization, input substitution and capacity control first. No claim of autonomous reference response or deployment.',
        validation_accessed=False,sealed_test_accessed=False))
    report=f'''# 执行命令输入合同审计 v28

原407个train窗口全部保留，5个源ULog、配置和固件核验通过。已生成可供下一轮**执行命令条件模型**使用的输入合同和时间戳张量；本轮未训练或加载新模型，未访问开发验证/sealed/reserved，保留A1600。

**能恢复已记录的归一化命令，尚不能恢复完整高频执行波形，也不能把命令当成实测舵角或气动力。**30,853组同timestamp_sample记录符合源码分配，最大残差约1.53e-7；但约50Hz命令日志与约100Hz输出日志不同步，两路升降副翼PWM保持重建仍有8.31/8.39us RMS差异。因此下一轮只能声称离线条件辨识，不能直接声称控制可用。

## 通道与配置

五飞行相关CA/PWM_MAIN/THR_MDL参数完全相同，无参数变更，日志固件为c9af571575af33d5403fbb479256848797c801e2。快照来自这个精确修订，不用当前分支代替；版本字符串不是独立的二进制或实物接线认证。

| 数据通道 | 发布字段 | 输出端口 | 当前普通工作状态映射 |
| --- | --- | --- | --- |
| motor0 | actuator_motors.control[0] | MAIN3 / function101 | PWM=1100+800u，u范围[0,1] |
| servo0 | actuator_servos.control[0] | MAIN1 / function201 | PWM=1600−600u，反向 |
| servo1 | actuator_servos.control[1] | MAIN2 / function202 | PWM=1600+600u |
| servo2 | actuator_servos.control[2] | MAIN5 / function203 | PWM=1300−500u，反向 |

映射按正数lround四舍五入；这5飞行THR_MDL_FAC=0、无可逆电机位，没有有效PWM_CENT覆盖。舵机配置类型为LeftElevon、RightElevon、Rudder，指参数命名，不是机械舵角/气动力方向的独立验证。输出消息单位由具体驱动决定，本配置为us；不能把SITL输出[-1,1]合同套入。`actuator_outputs`发布内部当前输出值，没有实际舵角反馈，也不提供独立硬件脉冲测量。

CA_METHOD=2是AUTO，固定翼默认伪逆、RPY不归一化；单电机前向轴，固定翼关闭螺旋桨反扭矩列，推力列归一化。3舵面、零trim/flap/spoiler、零执行器slew配置下，限幅前：

- servo0 = 0.5×torque_pitch − torque_roll/(2×0.5500000119)
- servo1 = 0.5×torque_pitch + torque_roll/(2×0.5500000119)
- servo2 = torque_yaw；motor0 = thrust_x

之后按舵机[-1,1]/电机[0,1]限幅。这里是归一化控制需求，不是已测N、N·m。方程来自源码/参数，没有用响应拟合系数。

## 同周期分配核验

{table(allocation_summary)}

选取原父窗口内发布的原生消息，只连接四话题唯一的timestamp_sample；重复sample记录数为{int(n.duplicate_sample_rows.sum())}。motor/servo各30,933条、torque/thrust各30,934条，四方共同30,853条。缺少配对的边缘/不同记录不能视为分配失败；逐条结果保留在allocation_exact_pairs.csv。

共有{flags['torque_unachieved_grid_points']}个20ms网格点保持了torque_achieved=false，来自{flags['torque_unachieved_unique_status_messages']}条独立状态消息，涉及{flags['affected_windows']}个父窗；记录到升降副翼限幅，故不能宣称所有片段均未饱和。失败/停机mask均为0。这些窗口保留原样并打标签。状态日志约200ms，保持值不是每个20ms周期的独立饱和测量，也不能证明其间不存在短暂事件。

## 时序与记录能力

31339=407×77个网格点，跨度origin−26..origin+50，严格只取发布与sample都不晚于查询的最新记录。命令与原缓存逐值一致，过去26点张量与v23逐值一致。

{table(ages[['field','n','p50','p95','max']])}

上表单位ms；sample时间是产生该控制响应所依据的数据时刻，不是舵面开始运动时刻。执行命令发布年龄最大22.87ms，PWM发布年龄最大14.096ms；本轮没有插值成更高频测量或优化任何时移。电机和舵机各自保持后有{flags['mixed_motor_servo_sample_grid_points']}个网格点来自不同sample周期，保留这种可用信息状态，不偷偷前看或强制配成同周期。

每窗内部原生发布间隔，避免把窗口之间空隙算作丢包：

{table(native)}

period栏是逐窗中位间隔的中位数；max_gap为已选区间内最大间隔。ULog自身是否完整记录了所有生产者更新无法由这些间隔保证。

## PWM链路核验

在61868条原生输出时刻上，保持最近已发布命令，按固定配置预测PWM，并与instance0输出比较；其他实例在日志里仅有单条初始化记录。残差定义为日志PWM−预测PWM：

{table(output_summary[['channel','n','rms','abs_p95_us','min','max','fraction_within_1us']])}

非零残差可能来自不同发布/记录速率、未记录中间控制更新及输出更新时序；本审计没有把它拟合成舵机延迟或错误通道。映射用float64复算固定公式，不是C++逐次float32运算的位级仿真，1us附近还可能包含舍入差异。最大瞬时差比RMS大，不能仅以平均吻合宣称高频命令完整。原始命令、PWM和时间戳全部保留供后续建模边界检查。

## 变化幅度与输入覆盖

下表只统计每窗未来50点已有命令，变化阈值1e-6仅抑制浮点重复计数，不用于筛窗：

{table(cov)}

32窗电机命令完全恒定；舵面普遍有连续变化，但反馈控制本来就会随状态改变。四通道数值有变化并不等于独立激励充分，更不是四条物理舵效已被验证。

用每窗内部差分（不跨窗口拼接）转换为motor/common=(s0+s1)/2/differential=(s1−s0)/2/yaw。以下原归一化单位下的标准差与原始奇异值最小/最大比，仅为描述，不作认证或筛选：

{table(independence)}

完整相关矩阵见input_independence.json；输入较弱的电机分量影响条件数，不能把数值秩当因果可辨识性。

## 后续输入合同

input_contract.json与actuator_grid.npz给出固定顺序[motor0,servo0,servo1,servo2]。过去输入为grid[:,0:26]（origin−26..−1）；未来条件输入为grid[:,26:76]（origin..+49），预测标签继续用v23的origin+1..+50。最后一个grid行仅供审计，不能多取一个未来输入或标签。所有旧状态、模式、相位和扑频合同保持，原5折fit/query不变，下一轮归一化仍必须只用fit飞行唯一行。本轮尚未拟合新归一化或训练新模型。

边界明确改为“分配后命令→输出映射/驱动/执行器/飞机”，不再让网络同时学习姿态/角速度内环和分配。未来日志命令依赖反馈和扰动，只允许作为明确声明的离线条件输入，不能继续把它当成原参考驱动模型的外生计划或自主预测。闭环时必须由控制器产生后续动作，不能读取未来真实执行命令。

沿用September自定义数据：407原train父窗，274Stabilized/133Mission；扑频比例7.909090995788574，机械相位零位的物理姿态未确认，不使用April canonical或ratio8。未来状态/扑频/空速仍不得额外注入。命令日志重建不等于新控制器已安全可用，也不据此认定必须补飞。

3项测试、5日志31339网格点独立原始消息复算、历史张量逐值一致、全配对分配与PWM公式独立复算、源固件/配置/原始文件/缓存/父清单哈希核验通过。预检修正了该固件版本消息定义位于msg/versioned的路径，在协议生成前完成，未修改数据。独立CSV复核先恢复原始float32命令再做公式计算，避免最短十进制表示在半微秒附近引入1us假差异；未放宽容差。

下一步先据此准备并冻结同窗、同预算的执行命令条件对照：确定4维输入替换、归一化与结构容量对照，再训练和比较固定真实窗口。新的任务边界与原参考模型不同，不能把平均预测改善直接算作控制能力提升。本轮无训练、删窗、部署或新测试集访问。
'''
    (OUT/'report_zh.md').write_text(report)
    code=dict(protocol['code']);code[str(Path(__file__).relative_to(base.ROOT))]=base.sha(Path(__file__))
    base.write(OUT/'delivery_manifest.json',dict(status='complete',artifacts={str(p.relative_to(OUT)):base.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'},code=code,
        training_started=False,model_promoted=False,validation_accessed=False,sealed_test_accessed=False))


if __name__=='__main__':main()
