"""Independent saved-output audit and delivery of the persistent GRU adapter."""
import numpy as np
import torch
import train_control_windows_v2 as base
from actuator_training_v30 import model_from
from audit_controller_support_v26 import verify_parent
from verify_actuator_step_v39 import OUT, DATA, PARENTS, read
from system_identification.models.actuator_command_step import ActuatorCommandStepper


def main():
    torch.set_num_threads(1)
    p = read(OUT/'protocol.json')
    v = read(OUT/'verification.json')
    assert v['status'] == 'passed' and v['checkpoints'] == 30
    for name,digest in p['code'].items():
        assert base.sha(base.ROOT/name) == digest
    restored = 0; snapshots = 0; scalars = 0; maximum = 0.; seen = {}
    for run in p['runs']:
        stage, fold, seed = [run[key] for key in ['stage','fold','seed']]
        dest = OUT/f'v{stage}_fold{fold}_seed{seed}'
        assert read(dest/'verification.json')['status'] == 'passed'
        assert base.sha(base.ROOT/run['checkpoint']) == run['checkpoint_sha256']
        assert base.sha(base.ROOT/run['saved_predictions']) == run['saved_predictions_sha256']
        with np.load(base.ROOT/run['saved_predictions']) as native, np.load(dest/'step_predictions.npz') as step:
            ids = read(DATA/f'fold{fold}/manifest.json')['query_indices']
            np.testing.assert_array_equal(step['array_index'], ids)
            np.testing.assert_array_equal(native['array_index'], ids)
            seen.setdefault((stage,seed), []).extend(ids)
            for key in ['rates_b','velocity_n','quaternion']:
                np.testing.assert_array_equal(step[key], native[key])
                maximum = max(maximum, float(np.abs(step[key]-native[key]).max()))
                scalars += step[key].size
        adapter = ActuatorCommandStepper(model_from(base.ROOT/run['checkpoint']))
        snap_ids = []
        for record in read(dest/'snapshots.json').values():
            assert base.sha(dest/record['path']) == record['sha256']
            payload = torch.load(dest/record['path'], map_location='cpu', weights_only=True)
            state = adapter.restore(payload)
            assert state.steps == 17 and state.model_id == read(dest/'verification.json')['model_id']
            assert len(state.hidden) == len(record['array_index'])
            resaved = adapter.snapshot(state)
            for key,value in payload.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(resaved[key], value, rtol=0, atol=0)
                else:
                    assert resaved[key] == value
            restored += len(state.hidden); snapshots += 1
            snap_ids.extend(record['array_index'])
        np.testing.assert_array_equal(snap_ids, ids)
    for ids in seen.values():
        np.testing.assert_array_equal(sorted(ids), np.arange(407))
    assert restored == 2442 and scalars == 2442*50*10 and maximum == 0
    base.write(OUT/'independent_verification.json', dict(status='passed',
        scalar_outputs_compared=scalars, max_abs_error=maximum, snapshot_files=snapshots,
        restored_window_states=restored, unique407_per_stage_seed=True,
        checked_against_unchanged_preexisting_native_archives=True))
    base.write(OUT/'test_results.json', dict(passed=7,
        command='/home/zn/anaconda3/envs/flap-train-gpu/bin/python -m pytest -q tests/test_actuator_command_step.py tests/test_actuator_inputs_v29.py',
        reused_unchanged_test_result=True))
    decision = dict(status='persistent_model_step_adapter_verified',
        exact_cpu_native_step_parity=True, save_restore_continuation_verified=True,
        independent_sessions_verified=True, input_prefix_causality_verified=True,
        model_predictions_changed=False, weights_or_normalization_changed=False,
        controller_model_coupled=False, model_promoted=False,
        baseline='A1600 retained; v35 remains a research candidate',
        next='Freeze multirate diagnostic coupling and evaluation contract, then bounded controller-model replay; no new training.',
        validation_accessed=False, sealed_test_accessed=False, reserved_accessed=False)
    base.write(OUT/'decision.json', decision)
    base.write(OUT/'next_execution_plan.json', dict(status='prepared_not_executed',
        objective='Bounded reference-conditioned c9af + learned-model closed-loop diagnostic',
        prerequisites_completed=['c9af component/source parity and teacher-state error baseline v38',
            'H26 persistent20ms adapter/native50step parity and restore v39'],
        limitation='v38 does not certify exact controller initial state or scheduling; diagnostic execution cannot be promoted as faithful full firmware replay.',
        stages=[
            'Before any coupled run, freeze400Hz controller/50Hz model exchange order and held-state/command approximation; preserve native20ms model command semantics and do not choose clocks by lowest error.',
            'Specify origin initialization, causal2s warmup and retainedHPF/clock/allocation-feedback sensitivity branches; use matched teacher-state controller baselines. Do not read future controller hidden state.',
            'Freeze attitude/thrust/manual-yaw condition tapes and input timestamps; separately specify airspeed/phase/frequency/acceleration proxies. No future measured plant states or future logged actuator commands may drive the model.',
            'Retain all407 parents and frozenv30/v35 OOF models; report states AND generated actions perflight/mode/seed plus failures, physical clamps, command/support excursions and sensitivity. Support uses past/current quantities only.',
            'Use existing frozen state-accuracy/protection gates unchanged; explicitly separate descriptive coupled errors and controller-approximation sensitivity. Define any additional acceptance criteria before replay, never from its outcomes.',
            'Run only after source/temporal/causality checks pass. A completed finite trajectory is diagnostic evidence, not independent control effectiveness or flight certification.'
        ], no_training=True, no_controller_tuning=True, no_real_flight=True,
        no_developmentvalidation_sealed_reserved=True))
    report = f'''# 持久 GRU 单步接口 v39

已实现并核验冻结v30/v35模型的H26初始化、20ms单步推进及完整状态保存恢复接口。**30个检查点、每个模型组/种子的全部407个OOF父窗，都与原生50步预测逐值相同，最大差为0。**接口准备完成；模型精度、控制器复现误差和控制可用性结论均未因此改变。

## 范围

仍使用v29自有数据合同的5个原train飞行、407父窗（274 Stabilized/133 Mission），v30/v35各5折×3种子，共30检查点。每个stage/seed按5折合起来恰好407窗，累计2442条模型窗口轨迹、122100个原命令单步；另有等量122100步扰动命令会话用于状态隔离与因果核验，不是新增真实试验。

没有加载状态标签、开发验证/sealed/reserved，没有训练或修改检查点/归一化，没有控制器连接或实飞。本批OOF已反复用于开发，不能称作独立未开封验证。所有父产物、检查点、normalization和源码哈希均核对。[protocol.json](protocol.json)在正式30模型运行前冻结，容差始终rtol=atol=0；批次与既有预测保持相同的最多64窗布局。

## 新接口

代码：`src/system_identification/models/actuator_command_step.py`。原模型forward、40070个参数及训练代码均未修改。

```python
adapter = ActuatorCommandStepper(frozen_model)
state = adapter.initialize(**history_inputs)  # 仅起点H26历史和初始状态
state, prediction = adapter.step(state, command_now, dt_s)  # dt_s为每条轨迹的0.02s
snapshot = adapter.snapshot(state)
state = adapter.restore(snapshot)
```

每个会话显式保存GRU hidden、NED速度、四元数、FRD角速度、phase/anchor、固定扑频、空速代理、模式、已执行步数与上一命令；上一命令仅作追踪，原生decoder没有该额外输入槽。初始化时刻的phase和anchor均为0，随后按原生固定扑频推进；没有引入后续实测相位、空速或状态重新编码。

每步只消费一个四通道分配后命令（motor0/servo0/servo1/servo2），沿用当前命令origin+k→下一状态origin+k+1的时序。模式、扑频和两列空速上下文保持原生行为。50步上限及20ms步长明确校验，不能把模型直接调用为400Hz动力学。

快照绑定模型参数、归一化、计算标志和接口版本的摘要；缺字段、错误模型、非有限值、形状错误或非20ms步长拒绝。张量独立复制，调用方修改输入、输出或导出的快照不会污染已有会话；不同会话不共享hidden。

## 核验结果

| 核验 | 覆盖 | 结果 |
| --- | --- | --- |
| 原生forward与持久step的全部状态输出 | 2442窗口×50步×10标量 | 完全相同 |
| 原生GRUCell隐藏状态 | 每个窗口全部50步、64维 | 完全相同 |
| 与此前保存的OOF预测文件比较 | {scalars}个输出标量 | 最大绝对差0 |
| 第17步保存到文件、由新adapter恢复并继续 | {restored}个窗口状态，{snapshots}份批次文件 | 后续33步完全相同 |
| 两个独立会话交错推进 | 原命令与第25步以后+0.125的命令带 | 分别等于各自原生forward |
| 前缀因果性 | 后25步命令改变 | 前25步逐值不变，后段确有响应 |
| 测试 | 新接口4项＋v29已有合同3项 | 7项通过 |

`state_step17_batch*.pt`保留中途快照，`step_predictions.npz`保留单步接口输出，逐批次结果见[parity.csv](parity.csv)。独立交付脚本重新读取所有输出，与此前冻结的预测逐标量比较，并重新加载全部快照验证字段保持；见[independent_verification.json](independent_verification.json)。

工程调试中发现：将复制模型参数的requires_grad标志改为False，会使CPU GRU编码器选择不同计算路径，即便在no_grad下也出现约1e−7的舍入差异。最终保留原标志，所有接口通过no_grad推理且没有optimizer；正式冻结前已修复，未放宽容差。模型副本仍与外部调用方权重隔离。数值一致证据针对记录的CPU/float32软件环境和批次布局，不声称所有GPU/编译器/批次尺寸都位级相同。

## 能够得到什么结论

现在可以在不改变既有预测的前提下，逐步输入控制器生成的命令，并随时保存/恢复完整模型会话。原生条件预测与新接口之间的迁移误差为0，后续若出现闭环偏离，可以排除本轮验证范围内的接口拆分误差。

这没有改善模型自身误差，也没有解除v38控制器初态/调度不确定性。命令扰动分支只检验程序输入依赖，没有真实反事实标签，不能当作因果舵效验证；本轮仍保留A1600及v35研究候选，不晋升控制可用模型。

下一步进入受限的同控制链闭环诊断：先冻结400Hz控制器与50Hz模型之间的状态保持、命令交换和更新顺序，明确空速/相位等代理及初态敏感性分支，然后由模型状态驱动控制器生成未来命令，同时评价状态与动作。未来日志命令只在本轮接口核验中作相同测试输入，不能进入自主闭环动力学；Mission参考带仍仅是外层条件，不等于导航/TECS复现。

下一阶段必须沿用既定状态精度/保护门槛，单独报告控制器近似误差、支持范围和初态/时钟敏感性，在运行前确定额外验收条件。数值跑完或低于某个重放误差都不自动证明新控制器可实飞。[next_execution_plan.json](next_execution_plan.json)列出执行合同待冻结项；本轮未运行控制器与神经模型耦合。
'''
    (OUT/'report_zh.md').write_text(report)
    base.write(OUT/'status.json', dict(status='complete_verified', completed_models=30, model_closed_loop=False))
    for parent in PARENTS:
        assert base.sha(parent/'delivery_manifest.json') == p['parents'][str(parent.relative_to(base.ROOT))]
        verify_parent(parent)
    code = dict(p['code'])
    code['scripts/report_actuator_step_v39.py'] = base.sha(base.ROOT/'scripts/report_actuator_step_v39.py')
    base.write(OUT/'delivery_manifest.json', dict(
        artifacts={str(path.relative_to(OUT)):base.sha(path) for path in sorted(OUT.rglob('*')) if path.is_file() and path.name != 'delivery_manifest.json'},
        code=code, model_closed_loop=False, model_promoted=False,
        validation_accessed=False, sealed_test_accessed=False, reserved_accessed=False))
    verify_parent(OUT)
    print(dict(status='complete_verified', compared_scalars=scalars, restored_windows=restored, max_abs_error=maximum), flush=True)


if __name__ == '__main__':
    main()
