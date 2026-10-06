"""Freeze actuator-conditioned data/recipe and preflight; never optimize weights."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

import train_control_windows_v2 as base
from actuator_inputs_v29 import replace_inputs, unique_fit_commands, command_stats
from audit_controller_support_v26 import verify_parent
from reference_groupcv_v8 import predict
from system_identification.models.actuator_command_gru import ActuatorCommandGRU
from system_identification.models.reference_gru_unit_gravity import ReferenceGRUUnitGravity

OUT = base.ROOT / 'docs/analysis/results/control_windows_v29'
V23 = OUT.parent / 'control_windows_v23'
V28 = OUT.parent / 'control_windows_v28'
V8 = OUT.parent / 'control_windows_v8'


def read(path):
    return json.loads(path.read_text())


def fingerprint(model):
    return hashlib.sha256(b''.join(p.detach().numpy().tobytes()
                                  for p in model.parameters())).hexdigest()


def main():
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    for root in [V23, V28]: verify_parent(root)
    parent = read(V23 / 'dataset_manifest.json'); contract = read(V28 / 'input_contract.json')
    assert base.sha(V23 / 'dataset_manifest.json') == contract['parent_manifest_sha256']
    assert base.sha(V28 / contract['artifact']) == contract['artifact_sha256']
    assert base.sha(V23 / 'windows.npz') == parent['arrays_sha256']
    for source in parent['sources']:
        assert source['source']['partition'] == 'train'
        assert base.sha(source['source']['path']) == source['source']['sha256']
        assert base.sha(base.ROOT / source['cache']) == source['cache_sha256']
    w = pd.read_csv(V23 / 'windows.csv')
    assert len(w) == 407 and w.partition.eq('train').all() and w.log_id.nunique() == 5
    np.testing.assert_array_equal(w.array_index, np.arange(407))
    with np.load(V23 / 'windows.npz') as z: old = {k: z[k] for k in z.files}
    with np.load(V28 / 'actuator_grid.npz') as z:
        grid = z['command_grid']; times = z['query_us']
        np.testing.assert_array_equal(z['array_index'], w.array_index)
        assert (z['publication_us'] <= times[..., None]).all()
        assert (z['sample_us'] <= times[..., None]).all()
    assert (np.diff(times, axis=1) == 20000).all()
    np.testing.assert_array_equal(times[:, 0], w.start_timestamp_us)
    np.testing.assert_array_equal(times[:, -1], w.end_timestamp_us)
    for _, group in w.groupby('log_id'):
        ids = group.sort_values('start_timestamp_us').array_index.to_numpy()
        assert (times[ids[1:], 0] > times[ids[:-1], -1]).all()
    arrays = replace_inputs(old, grid)
    for k in old:
        if 'reference' not in k: np.testing.assert_array_equal(arrays[k], old[k])
    assert not old['input_history_reference'][..., 4].any()
    assert not old['input_future_reference'][..., 4].any()
    paths = [Path(__file__), base.ROOT / 'scripts/actuator_inputs_v29.py',
             base.ROOT / 'tests/test_actuator_inputs_v29.py',
             base.ROOT / 'src/system_identification/models/actuator_command_gru.py',
             base.ROOT / 'src/system_identification/models/reference_gru_unit_gravity.py',
             base.ROOT / 'src/system_identification/models/trajectory_main_v1.py',
             base.ROOT / 'scripts/train_control_windows_v2.py',
             base.ROOT / 'scripts/reference_groupcv_v8.py',
             base.ROOT / 'scripts/audit_controller_support_v26.py']
    pins = {str(p.relative_to(base.ROOT)): base.sha(p) for p in paths}
    OUT.mkdir(exist_ok=False)
    protocol = dict(
        stage='v29 actuator-conditioned preparation', status='frozen_before_training',
        source_pins=pins,
        parents={str(p.relative_to(base.ROOT)): base.sha(p / 'delivery_manifest.json') for p in [V23, V28]},
        scope='Original 407 train windows only; preserve all windows, five flight folds, modes, labels and non-control features. No validation/sealed/reserved access.',
        boundary=contract['boundary'], conditional_limit=contract['conditional_limit'],
        substitution='Replace both history reference5 and future reference5 by motor0/servo0/servo1/servo2 plus structural zero. No controller history8. This is a whole-boundary comparison, not an isolated future-channel ablation.',
        history='state and new command history origin-25..0; retained actuator history origin-26..-1. Duplicate past commands intentionally preserve encoder slots; no new sensors.',
        future='commands origin..+49 are left-endpoint inputs for labels origin+1..+50. Row+50 is excluded. Each topic uses causal publication/sample hold inherited from v28; no shift/interpolation.',
        architecture=dict(name='ActuatorCommandGRU', hidden=64, encoder_input=25, recurrent_input=21,
                          trainable_parameters=40070, padding='fifth slot always0, mean0/scale1; v23 reset was also always0'),
        normalization='Each fold fits command mean/std on unique (flight,query_us) rows origin-25..+49 from its four fit flights only; std floor1e-4. Copy all other v23 stats exactly. The retained lagged actuator block keeps its older past-only normalization. Both control blocks use the same physical units but different fitting supports.',
        steps=1600, batch=32, seeds=[17, 23, 42], folds=5, candidate_runs=15,
        optimizer=dict(name='AdamW', lr=3e-4, weight_decay=1e-4, gradient_clip=1.),
        loss='Unchanged base.loss_fn over complete 50-step rates+velocity+0.2 quaternion loss. No auxiliary loss. Fixed final1600 checkpoint; no early stopping or outcome-selected checkpoint.',
        pairing='Same trainable initialization as v23/v8, same stored1600x32 draws per fold/seed. Reuse15 frozenv23 controls after exact fit/OOF inference replay; no retraining baseline required because its implementation/data/optimizer are unchanged.',
        evaluation='Save all fit and OOF predictions and per-window/per-flight metrics. Equal-flight then equal-seed aggregation, full1s p/q/r/velocity/attitude waveforms and20ms/.1s/.5s/1s endpoints; same labels and original active-window mask as v23. Verify every seed covers407uniqueOOF windows.',
        descriptive_gate=dict(q_relative_improvement_min=.05, all_three_seeds_improve=True,
            minimum_improving_flights=4, protections='For each seed, p/r full1s and velocity/attitude .5s/1s endpoints regress <=5% (18 values)',
            absolute='Original active-window0.5s p/q<=.15rad/s, attitude<=3deg, velocity<=.5m/s (12 values); reuse original metric definitions/masks'),
        diagnostics='After training, hold all4futurecommands at origin within each query window; compare conditional predictions. Artificial hold is a sensitivity diagnostic without counterfactual ground truth, not an accuracy/control gate.',
        limitations='Five flights have repeatedly informed development, not an unopened final test. Logged future commands are feedback-conditioned. No independent actuator positions/forces or high-rate waveform. Even all numerical gates passing cannot promote to causal control or deployment.',
        next_gates='Conditional prediction first; independently supported direction/delay/amplitude/persistence/coupling second; model-state-driven actual PX4 control/allocation closed-loop replay third. Novel-controller robustness and independent evidence required beyond these development folds.',
        training_started=False, model_promoted=False, baseline_retained='A1600',
        phase_contract=contract['phase_contract'], frequency_contract=contract['frequency_contract'],
        validation_accessed=False, sealed_test_accessed=False, reserved_accessed=False)
    base.write(OUT / 'protocol.json', protocol)
    np.savez_compressed(OUT / 'windows.npz', **arrays)
    shutil.copyfile(V23 / 'windows.csv', OUT / 'windows.csv')
    x = {k[6:]: torch.tensor(v) for k, v in arrays.items() if k.startswith('input_')}
    y = {k[6:]: torch.tensor(v) for k, v in arrays.items() if k.startswith('label_')}
    oldx = {k[6:]: torch.tensor(v) for k, v in old.items() if k.startswith('input_')}
    checks = []; runs = []; counts = []
    for fold in range(5):
        info = read(V23 / f'fold{fold}/manifest.json'); fit = info['fit_indices']; query = info['query_indices']
        assert set(w.iloc[fit].log_id).isdisjoint(w.iloc[query].log_id)
        assert sorted(fit + query) == list(range(407))
        values, keys = unique_fit_commands(grid, times, w.log_id.to_numpy(), fit)
        with np.load(V23 / f'fold{fold}/normalization.npz') as z: stats = {k: z[k] for k in z.files}
        stats['reference_mean'], stats['reference_scale'] = command_stats(values)
        # Independent oracle: original cache indices, not timestamp dedup helper.
        oracle = {}
        for i in fit:
            for j in range(1, 76):
                key = (w.iloc[i].log_id, int(w.iloc[i].origin_index) - 26 + j)
                if key in oracle: np.testing.assert_array_equal(oracle[key], grid[i, j])
                oracle[key] = grid[i, j]
        independent = np.stack([oracle[k] for k in sorted(oracle)]).astype(float)
        np.testing.assert_allclose(stats['reference_mean'][:4], independent.mean(0), rtol=0, atol=1e-12)
        np.testing.assert_allclose(stats['reference_scale'][:4], independent.std(0).clip(1e-4), rtol=0, atol=1e-12)
        assert len(oracle) == len(keys) and not (set(k[0] for k in keys) & set(w.iloc[query].log_id))
        dest = OUT / f'fold{fold}'; dest.mkdir()
        np.savez(dest / 'normalization.npz', **stats)
        pd.DataFrame(keys, columns=['log_id', 'query_us']).to_csv(dest / 'normalization_rows.csv', index=False)
        base.write(dest / 'manifest.json', dict(fold=fold, fit_indices=fit, query_indices=query,
            source_manifest_sha256=base.sha(V23 / f'fold{fold}/manifest.json'),
            normalization_sha256=base.sha(dest / 'normalization.npz'),
            normalization_rows_sha256=base.sha(dest / 'normalization_rows.csv'), unique_command_rows=len(keys)))
        counts.append(dict(fold=fold, fit_windows=len(fit), query_windows=len(query), unique_command_rows=len(keys)))
        for seed in [17, 23, 42]:
            baseline_dir = V23 / f'fold{fold}/seed{seed}'
            previous = read(baseline_dir / 'run_protocol.json')
            draws_path = V8 / f'fold{fold}/hold_{seed}/sample_draws.npz'
            assert base.sha(draws_path) == previous['draws_sha256']
            with np.load(draws_path) as z: draws = z['array_index']
            assert draws.shape == (1600, 32) and set(draws.ravel()) <= set(fit)
            torch.manual_seed(seed); model = ActuatorCommandGRU(stats)
            digest = fingerprint(model)
            assert digest == previous['initial_parameter_sha256']
            assert sum(p.numel() for p in model.parameters()) == 40070
            # Forward/backward preflight only: no optimizer and no parameter updates.
            ids = draws[0]; pred = model(**{k: v[ids] for k, v in x.items()})
            loss = base.loss_fn(pred, {k: v[ids] for k, v in y.items()}, model)
            assert torch.isfinite(loss); loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            assert fingerprint(model) == digest
            # The existing baseline uses the same unit-gravity forward with no schedule.
            ck = torch.load(baseline_dir / 'final.pt', weights_only=False, map_location='cpu')
            replay = ReferenceGRUUnitGravity(stats); replay.load_state_dict(ck['state_dict']); replay.eval()
            for cohort, indices in [('fit', fit), ('oof', query)]:
                predicted = predict(replay, oldx, indices)
                with np.load(baseline_dir / f'{cohort}_predictions.npz') as z:
                    np.testing.assert_array_equal(z['array_index'], indices)
                    for k, v in predicted.items(): np.testing.assert_array_equal(v, z[k])
            runs.append(dict(fold=fold, seed=seed, initial_parameter_sha256=digest,
                draws_path=str(draws_path.relative_to(base.ROOT)), draws_sha256=base.sha(draws_path),
                control_checkpoint=str((baseline_dir / 'final.pt').relative_to(base.ROOT)),
                control_checkpoint_sha256=base.sha(baseline_dir / 'final.pt'),
                normalization_sha256=base.sha(dest / 'normalization.npz'), steps=1600, batch=32))
            checks.append(dict(fold=fold, seed=seed, matched_initial_parameters=True,
                exact_fit_oof_baseline_replay=True, draws_fit_only=True,
                finite_forward_backward=True, parameters_unchanged=True))
        print(f'fold {fold}: {len(fit)} fit, {len(query)} query, {len(keys)} normalization rows; 3 controls replayed', flush=True)
    base.write(OUT / 'training_plan.json', dict(status='ready_not_started', runs=runs,
        protocol_sha256=base.sha(OUT / 'protocol.json'), formal_training_runs_completed=0))
    base.write(OUT / 'dataset_manifest.json', dict(dataset_id='actuator_conditioned_train_v29',
        partition='original_train_only', windows=407, flights=5, history_steps=26, rollout_steps=50,
        authority=str((OUT / 'protocol.json').relative_to(base.ROOT)), protocol_sha256=base.sha(OUT / 'protocol.json'),
        parent_manifest_sha256=base.sha(V23 / 'dataset_manifest.json'),
        command_contract_sha256=base.sha(V28 / 'input_contract.json'),
        arrays_sha256=base.sha(OUT / 'windows.npz'), windows_sha256=base.sha(OUT / 'windows.csv'),
        shapes={k: list(v.shape) for k, v in arrays.items()}, sources=parent['sources'],
        channel_order=contract['channels'], units=contract['units'],
        phase_contract=contract['phase_contract'], frequency_contract=contract['frequency_contract'],
        validation_accessed=False, sealed_test_accessed=False))
    tests = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_actuator_inputs_v29.py'],
                           cwd=base.ROOT, text=True, capture_output=True)
    base.write(OUT / 'test_results.json', dict(command=tests.args, returncode=tests.returncode,
        stdout=tests.stdout, stderr=tests.stderr))
    assert tests.returncode == 0, tests.stdout + tests.stderr
    base.write(OUT / 'verification.json', dict(status='passed', checks=checks, normalization=counts,
        non_control_inputs_and_labels_unchanged=True, independent_normalization_recomputed=True,
        all407_windows_retained=True, nonoverlapping_support=True, final_audit_command_excluded=True,
        command_timestamps_causal=True, source_and_parent_hashes_verified=True,
        training_started=False, optimizer_steps=0))
    report = '''# 执行命令条件模型准备 v29

本轮完成数据、纯推理接口、五折归一化、训练对照协议和预检。没有启动正式训练，没有新模型收益结论；保留 A1600。

## 数据与输入

沿用原 5 个 train 飞行的 407 个不重叠父窗口（274 Stabilized、133 Mission），支持区间每窗1.52秒，共618.64秒；不删弱响应、饱和或困难窗口，不新增划分。windows.npz 可供下一轮直接加载：状态历史407×26×14、对齐命令历史407×26×4、原滞后命令历史407×26×4、未来命令407×50×4及原有标签。四通道依次 motor0/servo0/servo1/servo2，均为归一化分配后命令，不是实测舵角或推力。

命令历史用 origin−25..0；原有执行历史仍用 origin−26..−1。未来用 origin..+49，对应状态标签 origin+1..+50。origin+50 的命令仅用于父审计，本模型不读入。保留双时间戳因果保持与不同电机/舵机消息周期，不平移、不插值。9月相位/扑频合同原样继承，不改为4月ratio8。

## 对照与归一化

ActuatorCommandGRU 使用原 GRU64 和同一积分器，四命令后补恒零第五槽，保持25维编码器输入、21维递推输入和40070参数。原v23第五reset槽同样恒零。保留重复的滞后执行历史是为了维持结构；两组历史参考和未来参考一并替换，因此不能把差异唯一归因于未来输入。历史控制器8列不加入。

每折只从4个fit飞行收集 origin−25..+49 的唯一(飞行,查询时刻)命令，按行等权拟合mean/std，std下限1e−4；第五槽mean0/scale1。状态、旧执行历史和损失尺度完全继承v23。旧执行历史与新命令槽物理单位相同，但前者归一化仍基于原历史支持；不混用两套统计。normalization_rows.csv记录全部拟合行。

冻结5折×3种子17/23/42，共15次候选训练，各1600步、batch32，AdamW lr3e−4/weight_decay1e−4/梯度裁剪1，完整1秒原损失，固定最后检查点。逐组复用原抽样并验证相同可训练参数初值；15个v23冻结对照的fit/OOF预测全部逐值重放成功，不重复训练未改变的对照。

## 验收与边界

在相同标签及原活动窗口上比较：主描述门槛为完整1秒q的飞行/种子等权RMSE至少改善5%，三个种子均改善、至少4/5飞行改善；18项状态保护限制退化不超过5%；另报告原0.5秒p/q≤0.15rad/s、姿态≤3度、速度≤0.5m/s的12项绝对门槛。不得换成筛后的容易窗口。训练后固定未来命令到起点值仅作敏感性诊断，没有真实反事实标签。

这些是“给定未来日志执行命令”的条件预测门槛。反馈命令包含状态/扰动相关信息，输入边界已经改变；不能称为原参考自主预测改进。五飞行已反复用于开发，也不是未开封最终测试。即使全部数值门槛通过，仍须独立控制响应（方向、延迟、幅值、持续与耦合）证据和由模型状态驱动真实控制链的闭环评价；本轮不晋升、不部署。

## 检查与交付

三项针对性测试通过，覆盖时序/最后审计行排除、fit-only去重与查询扰动不影响统计、相同容量/补零等价/非零输出头下前缀因果性及四通道梯度。全部5折统计独立复算；15组初值、批次、有限前后向和不改权重检查通过，15个对照的fit/OOF共30组预测逐值一致。源日志/缓存与父产物哈希核验通过。没有optimizer step、开发验证/sealed/reserved访问或窗口剔除。

下一步按 training_plan.json 执行15次候选训练并同窗评价。protocol.json、dataset_manifest.json、verification.json及delivery_manifest.json共同固定本轮数据、代码与边界；正式训练尚未开始。
'''
    (OUT / 'report_zh.md').write_text(report)
    base.write(OUT / 'decision.json', dict(status='preparation_complete', training_started=False,
        model_promoted=False, baseline_retained='A1600', next_step='Execute frozen15candidate runs and matched conditional prediction evaluation; do not equate gate pass with causal control readiness.'))
    for root in [V23, V28]: verify_parent(root)
    for name, digest in pins.items(): assert base.sha(base.ROOT / name) == digest
    base.write(OUT / 'delivery_manifest.json', dict(
        artifacts={str(p.relative_to(OUT)): base.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()},
        code=pins, training_started=False, model_promoted=False, validation_accessed=False, sealed_test_accessed=False))
    print('v29 prepared: 407 windows, 5 folds, 15 planned runs; zero optimizer steps', flush=True)


if __name__ == '__main__': main()
