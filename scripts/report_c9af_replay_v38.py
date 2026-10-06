"""Seal the controller-component diagnostic without promoting a replay recipe."""
import numpy as np
import pandas as pd
import train_control_windows_v2 as base
from run_c9af_replay_v38 import OUT, PREV, DATA, VARIANTS, read
from audit_controller_support_v26 import verify_parent


def markdown(frame):
    header = '| ' + ' | '.join(map(str, frame.columns)) + ' |'
    separator = '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |'
    rows = []
    for row in frame.itertuples(index=False, name=None):
        rows.append('| ' + ' | '.join(f'{v:.6f}' if isinstance(v, (float, np.floating)) else str(v) for v in row) + ' |')
    return '\n'.join([header, separator, *rows])


def main():
    protocol = read(OUT / 'protocol.json')
    verification = read(OUT / 'verification.json')
    source = read(OUT / 'source_check/verification.json')
    assert verification['status'] == source['status'] == 'passed'
    assert verification['raw_target_metric_groups_recomputed'] == 17094
    assert verification['serialized_initial_hypotheses_restored'] == 5698
    assert read(OUT / 'test_results.json')['passed'] == 6
    for path, digest in protocol['code'].items():
        assert base.sha(base.ROOT / path) == digest, path
    for root in [PREV, DATA]:
        verify_parent(root)
        assert base.sha(root / 'delivery_manifest.json') == protocol['parents'][str(root.relative_to(base.ROOT))]
    windows = pd.read_csv(DATA / 'windows.csv')
    coverage = pd.read_csv(OUT / 'coverage.csv')
    assert len(coverage) == 5698
    assert coverage.unsupported_warmup_cycles.sum() == coverage.unsupported_future_cycles.sum() == 0
    summary = pd.read_csv(OUT / 'summary_rmse.csv')

    def table(kind, branch, labels):
        frame = summary[(summary.kind == kind) & (summary.branch == branch)].pivot(index='variant', columns='axis', values='rmse')
        frame = frame.loc[[v[0] for v in VARIANTS]]
        frame.columns = labels
        return markdown(frame.reset_index())

    # Descriptive stratification, never a selection or a numerical acceptance gate.
    modes = pd.read_csv(OUT / 'by_mode_flight.csv')
    modes = modes[(modes.variant == 'warm2_plus') & (modes.branch == 'attitude_chain') & (modes.kind == 'commands')]
    mode_table = modes.groupby(['mode', 'axis']).rmse.mean().unstack('axis')
    mode_table.columns = ['motor0', 'servo0', 'servo1', 'servo2']
    mode_table.insert(0, 'parents', windows.groupby('mode').size())
    mode_table.to_csv(OUT / 'warm2_plus_mode_summary.csv')
    perflight = pd.read_csv(OUT / 'metrics_per_flight.csv')
    perflight = perflight[(perflight.variant == 'warm2_plus') & (perflight.branch == 'attitude_chain') & (perflight.kind == 'commands')]
    flight_table = perflight.pivot(index='log_id', columns='axis', values='rmse')
    flight_table.columns = ['motor0', 'servo0', 'servo1', 'servo2']
    flight_table.to_csv(OUT / 'warm2_plus_flight_summary.csv')
    flight_table.index = [name.split('/')[1].split('_2026')[0] for name in flight_table.index]

    plan = dict(
        status='prepared_not_executed',
        next_stage='Persistent model step adapter parity; no controller or model tuning',
        parents=['control_windows_v30', 'control_windows_v35', 'control_windows_v38'],
        model_scope='Frozen v30/v35 5fold x3seed OOF models, same407 originaltrain parents; no developmentvalidation/sealed/reserved.',
        tasks=[
            'Expose H26 initialization and persistent20ms step with complete hidden, dynamic state, previous command, fixed phase/frequency and airspeed-proxy state; keep native operations unchanged.',
            'On identical prerecorded command tapes, compare all50 steps with each frozen native output; tapes are adapter arithmetic checks only, not autonomous forecasts. Freeze tolerances before execution.',
            'Verify checkpoint/statistics hashes, save/restore continuation, interleaved independent sessions and future-input-prefix causality. No future measured state reencoding.',
            'Before any learned-model closed loop, freeze explicit400Hz-controller/50Hz-model coupling, reference tapes, airspeed/acceleration/phase assumptions and controller initialization. Retain multiple clock/state hypotheses; do not select20ms by lowest error.',
            'Define comparison of generated commands and states, support excursions and sensitivity against the measured-state controller baseline. Keep physical limits and record interventions. Set acceptance criteria before model replay; numerical completion alone is not control readiness.'
        ],
        controller_reproduction_limit='Component arithmetic verified; exact initial state and native scheduler unavailable. Current teacher errors are a diagnostic baseline, not a passed fidelity gate.',
        learned_model_replay_ready=False,
        no_training=True, no_controller_design=True, no_real_flight=True)
    base.write(OUT / 'next_execution_plan.json', plan)
    decision = dict(
        status='controller_component_replay_complete',
        source_component_arithmetic_passed=True,
        all407_retained=True, windows=407, flights=5, replay_runs=5698,
        measured_state_controller_replay_completed=True,
        exact_initial_controller_state_recovered=False,
        exact_native_scheduler_recovered=False,
        output_fidelity_gate_predefined=False,
        controller_fidelity_certified=False,
        replay_recipe_selected=False,
        model_replay_started=False, model_promoted=False,
        baseline='A1600 retained; v35 research candidate remains unpromoted',
        validation_accessed=False, sealed_test_accessed=False, reserved_accessed=False,
        next_action=plan['next_stage'])
    base.write(OUT / 'decision.json', decision)
    report = f'''# c9af 同源控制器组件重放 v38

已实现独立的 c9af 姿态→角速度→增益压缩→控制分配适配器，并完成原5个train飞行、407父窗的全部7种初始化/时钟假设×2种输入分支，共5698次真实状态驱动重放。**组件公式核对通过，历史预热明显改善命令复现；尚未恢复精确控制器初态或原生调度，因此没有认证完整闭环保真，也没有晋升动力学模型。**

## 范围与数据边界

- 固件固定为 `{protocol['firmware']}`，使用v37的精确源码快照、各ULog实际参数；没有向旧e624封装补FLAP_SLOW_EN默认值。
- 原407个父窗全部保留，274 Stabilized、133 Mission，包含弱响应/分配受限窗口。仍是此前已反复开发使用的5个train飞行，不是独立未开封测试。没有读取开发验证、sealed test或reserved，没有训练、删窗或部署。
- 比较网格为起点+0..49×20ms的50个命令点。两分支均使用当时因果可用的**未来实测运动状态**驱动控制器，这是隔离控制器实现误差的离线诊断，不是从起点自主预测飞机。
- `rate_input` 使用日志rate参考和油门；`attitude_chain` 使用日志姿态参考、油门和manual yaw并重新生成rate参考。记录的Mission参考只是外层条件带，不复现导航/TECS，也不视为外生随机激励。
- 每种假设只在各自启动时读取当时最新积分、压缩器观测等；随后不回灌未来日志积分、增益或力矩。publication和sample时间都必须不晚于当前合成时刻。
- [dataset_manifest.json](dataset_manifest.json)记录5份原始ULog/缓存哈希和v37父清单哈希。继承9月数据自身的mechanical phase及7.909090995788574频率比合同，未切换到4月canonical ratio8；相位物理零位仍未确认。

## 实现与核对

新增 `src/system_identification/control/px4_c9af.py`，保留旧e624组件。适配器显式保存积分、带符号HPF、能量、上一effort、内部/输出缓存增益、空速滤波、yaw记忆、分配残差及饱和标志，支持完整状态序列化和恢复。roll/pitch使用旧yaw欧拉率、随后更新yaw；manual yaw在body yaw限幅后叠加；输出使用更新前积分/缓存增益。

6项针对性测试通过。抽取6个未改写的c9af C++函数体，以最小host类型/数学垫片编译，512组相同输入/状态与Python比较通过，最大绝对差 `{source['max_abs_error']:.12g}`（预设rtol1e−5/atol2e−6）。[源码核对](source_check/verification.json)针对所选组件算术，不覆盖完整PX4/uORB运行、所有模式或位级一致性；Python使用float64。

独立从原始ULog重建目标，复算全部17094组轴向指标；5698份初始化假设全部可序列化/恢复。恢复的是明确假设的内部状态，不是证明它们等于实机隐藏状态。所有预热/评分周期均落在实现支持的模式内，unsupported计数为0。详见[verification.json](verification.json)。

## 冻结的七种假设

| 假设 | 初始化与预热 | rate时钟 | 分配反馈 |
| --- | --- | --- | --- |
| cold | 源码冷启动：状态0、增益1；无预热 | 2.5ms | 无中间发布假设 |
| partial_plus | 起点部分观测；三轴HPF均取正平方根 | 2.5ms | 同上 |
| partial_minus | 起点部分观测；三轴HPF均取负平方根 | 2.5ms | 同上 |
| warm2_plus | 起点前2秒部分观测初始化，三轴正号，再沿实测历史预热 | 2.5ms | 同上 |
| warm2_minus | 同上，三轴负号 | 2.5ms | 同上 |
| warm2_20ms | 2秒预热、正号 | 20ms | 同上 |
| warm2_delayed_feedback | 2秒预热、正号 | 2.5ms | 上次合成分配残差 |

部分观测初始化使用最新可用post-integral、gain/energy、±sqrt(HPF平方)；以原始CAS近似内部滤波，previous effort/yaw记忆设0，均明确为假设。两种全轴同号只是两个探针，不覆盖八种符号组合或所有未观测状态，不能解释为完整置信区间。

时钟为锚定起点的固定网格；传感器和参考因果保持，记录姿态索引变化时先更新姿态环再运行rate环。v37推断约400Hz只支持量级，不能恢复丢失的原生调用序列；2.5ms与20ms均是这里的显式近似。

c9af非VTOL路径两处`Subscription.update()`指向同一instance0，第一次用于差动推力反馈、第二次用于舵面反馈。若两次之间没有新发布，第一次已消费generation，第二次没有新消息。本轮主要假设采用这种串行语义，另保留上一合成分配残差反馈分支。实际线程间是否插入发布无法从低频日志恢复；该检查**不构成已证实的实机调度行为或固件故障结论**。[Subscription源码及出处](subscription_source_provenance.json)已保存。

## 结果

以下均先在各飞行内对所有父窗/50时刻求MSE、开根，再对5飞行等权平均。命令/力矩为归一化量，rate为rad/s；命令不是实测舵角或推力。数值不是相对百分比误差。

姿态参考分支的分配命令RMSE：

{table('commands', 'attitude_chain', ['motor0', 'servo0', 'servo1', 'servo2'])}

日志rate参考分支的分配命令RMSE：

{table('commands', 'rate_input', ['motor0', 'servo0', 'servo1', 'servo2'])}

姿态参考分支的rate参考RMSE：

{table('rates', 'attitude_chain', ['p_sp', 'q_sp', 'r_sp'])}

姿态参考分支的力矩需求RMSE：

{table('torque', 'attitude_chain', ['roll', 'pitch', 'yaw'])}

日志rate参考分支的力矩需求RMSE：

{table('torque', 'rate_input', ['roll', 'pitch', 'yaw'])}

`rate_input`的rate参考零误差是直接使用目标输入的恒等关系，不是姿态环验证成绩。motor来自日志油门条件，误差很小不能证明电机/飞机动力学正确。

1. 冷启动两升降副翼误差约0.223/0.247；部分初始化降低到约0.05，再增加2秒预热后约0.028。起点历史对本批重放影响很大，冷启动误差不能全部归于动力学模型。
2. 姿态链与直接rate输入的命令误差接近，重新生成的rate参考误差也较小；残余命令差异并非主要只来自是否重新生成姿态环参考。但不能由此唯一确定隐藏状态、日志时间或其他近似各自贡献。
3. 20ms分支部分命令RMSE更低，却将yaw参考RMSE提高至约0.0118。不能按最低误差挑20ms并宣称等同原生约400Hz控制器；本轮不选择或晋升任何重放配方。
4. 两种HPF符号预热后结果接近但不相同；替代分配反馈也有影响。有限探针不能证明全部初态/调度不确定性已覆盖。

为方便查阅，预先定义的`warm2_plus`分支按模式/飞行展开如下；这只是一个具名假设的描述，不是按结果选出的获胜配方。全部假设的逐飞行/逐模式结果保留在[metrics_per_flight.csv](metrics_per_flight.csv)和[by_mode_flight.csv](by_mode_flight.csv)。

{markdown(mode_table.reset_index())}

mode3为Mission、15为Stabilized；每个模式先在该飞行该模式内部求RMSE，再在拥有该模式的飞行间等权，不能拿模式表直接重组总体表。

{markdown(flight_table.reset_index())}

## 验收结论与下一步

本轮已完成v37计划的c9af适配器、真实状态驱动重放和冻结的状态/时钟敏感性实验。源码组件算术和重放指标复核通过；**真实命令复现没有预先定义的数值验收门槛，本轮只能报告测得误差，不能事后宣布保真通过。**没有运行任何神经模型闭环，A1600保留、v35仍为研究候选。

下一步先做与控制器误差独立的工程工作：为冻结v30/v35模型实现持久H26初始化/20ms step接口，保存完整GRU与动力学状态，在相同命令带下逐步复现原生50步输出，并测试中断恢复和输入前缀因果性。这一阶段不需要新训练或补飞。

随后才冻结约400Hz控制器与50Hz模型如何交换状态/动作、空速和相位如何推进、哪些初态假设共同保留，以及状态/动作/支持范围的对比指标。真实状态驱动误差作为诊断基线保留，不能当作可从模型误差中直接相减的噪声下界。没有通过控制器和模型接口证据前，不宣布已能做真实新控制器；即使受限闭环数值稳定也仍需既定的动力学精度、控制响应和独立验证证据。

下一阶段具体范围见[next_execution_plan.json](next_execution_plan.json)，本轮未实施该模型接口或闭环阶段。原始日志未改动；[decision.json](decision.json)与[delivery_manifest.json](delivery_manifest.json)记录决定及交付哈希。
'''
    (OUT / 'report_zh.md').write_text(report)
    status = read(OUT / 'status.json')
    status['status'] = 'complete_verified'
    status['model_replay_started'] = False
    base.write(OUT / 'status.json', status)
    code = dict(protocol['code'])
    for path in ['scripts/evaluate_c9af_replay_v38.py', 'scripts/report_c9af_replay_v38.py']:
        code[path] = base.sha(base.ROOT / path)
    base.write(OUT / 'delivery_manifest.json', dict(
        artifacts={str(p.relative_to(OUT)): base.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'delivery_manifest.json'},
        code=code, model_replay_started=False, model_promoted=False,
        validation_accessed=False, sealed_test_accessed=False, reserved_accessed=False))
    verify_parent(OUT)
    print(decision, flush=True)


if __name__ == '__main__':
    main()
