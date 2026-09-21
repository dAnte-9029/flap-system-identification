"""Frozen Step 3: reuse B2 seed17; train seeds23/42, then evaluate automatically.

No model, optimizer, data loader or metric implementation is changed. Only the
seed orchestration and cross-seed reporting are new. Test partitions are denied.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
import torch
import yaml

from run_paper_baseline_comparison import (
    STATS, _FrequencyLossAdapter, build_model as original_build_model,
    causality_check, write_json,
)
from system_identification.data.september_trajectory import file_hash
from system_identification.evaluation.paper_baselines import HORIZONS, METRIC_MAP, aggregate_flights, endpoint_metrics
from system_identification.models.trajectory import TrajectoryPrediction
from system_identification.models.trajectory_main_v1 import CausalHistoryTrajectoryModel
from system_identification.training.main_v2_increment import train_increment_stage, increment_terms
from system_identification.training.trajectory_main_v1 import (
    _model_call, assemble_history_trajectory_windows, predict_history_trajectory_model,
    trajectory_rollout_loss,
)

BASE = ROOT / 'docs/analysis/results/paper_baseline_comparison_v1'
OLD = ROOT / 'artifacts/paper_baseline_comparison_v1'
OUT = ROOT / 'docs/analysis/results/paper_standard_gru_multiseed_v1'
ART = ROOT / 'artifacts/paper_standard_gru_multiseed_v1'
MODEL = 'B2_StandardGRU'
SEEDS = (17, 23, 42)
DEVICE = 'cuda:1'
METRICS = ['position_rmse_m', 'velocity_rmse_m_s', 'attitude_error_deg', 'body_rate_rmse_rad_s']
SCRIPT = Path(__file__).resolve()


def read_json(path):
    return json.loads(path.read_text())


def rel(path):
    return str(path.relative_to(ROOT))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def configure():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stages(seed):
    require(seed in SEEDS, 'undeclared seed')
    return [('base', 40, seed, .0003, False), ('continuation', 25, seed + 12, .0005, True)]


def build_model(seed, stats):
    seed_all(seed)
    return CausalHistoryTrajectoryModel(hidden_size=64, use_controls=True, **stats)


def checkpoint_path(seed):
    return OLD / f'{MODEL}.pt' if seed == 17 else ART / f'seed{seed}' / 'model.pt'


def normalization_hash(stats):
    digest = hashlib.sha256()
    for key in STATS:
        array = np.asarray(stats[key], dtype='<f4')
        digest.update(json.dumps([key, list(array.shape), '<f4']).encode())
        digest.update(array.tobytes(order='C'))
    return digest.hexdigest()


def same_tree(a, b):
    """Exact equality of cached and freshly assembled histories/targets/identities."""
    if is_dataclass(a):
        for field in fields(a):
            same_tree(getattr(a, field.name), getattr(b, field.name))
    elif isinstance(a, np.ndarray):
        require(a.dtype == b.dtype and np.array_equal(a, b), 'batch differs from Step 1')
    else:
        require(a == b, 'batch metadata differs from Step 1')


def subset(batch, n):
    def cut(value):
        if is_dataclass(value):
            return replace(value, **{f.name: cut(getattr(value, f.name)) for f in fields(value)})
        return value[:n] if isinstance(value, np.ndarray) else value
    return cut(batch)


def verify_pins(pins):
    for path, expected in pins.items():
        require(file_hash(ROOT / path) == expected, f'frozen input changed: {path}')


def load_inputs():
    protocol = read_json(OUT / 'protocol.json')
    verify_pins(protocol['frozen_input_sha256'])
    require(file_hash(ART / 'prepared.pt') == protocol['prepared_sha256'], 'prepared cache changed')
    prepared = torch.load(ART / 'prepared.pt', map_location='cpu', weights_only=False)
    cp = torch.load(checkpoint_path(17), map_location='cpu', weights_only=False)
    stats = {key: cp['state_dict'][key].numpy() for key in STATS}
    require(normalization_hash(stats) == protocol['normalization_sha256'], 'normalization changed')
    return protocol, prepared, stats


def validate_rows(rows, batch, seed):
    require(set(rows.seed) == {seed} and set(rows.model) == {MODEL}, 'wrong model or seed')
    require(set(rows.horizon_s) == set(HORIZONS), 'wrong horizons')
    require(len(rows) == 2582 * 4, 'prediction count mismatch')
    require(not rows.duplicated(['window_id', 'horizon_s']).any(), 'duplicate origin/horizon')
    for horizon, group in rows.groupby('horizon_s'):
        indexed = group.set_index('window_id').loc[batch.trajectory.window_ids]
        require(len(group) == 2582, 'missing origins')
        np.testing.assert_array_equal(indexed.log_id, batch.trajectory.log_ids)
        np.testing.assert_allclose(indexed.observed_horizon_s,
                                   batch.trajectory.dt_s[:, :HORIZONS[horizon]].sum(1), atol=1e-12, rtol=0)
    require(np.isfinite(rows[list(METRIC_MAP)].to_numpy()).all(), 'nonfinite metrics')


def audit():
    require(not (OUT / 'protocol.json').exists(), 'protocol already exists; do not overwrite frozen audit')
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    base = read_json(BASE / 'protocol.json')
    complete = read_json(BASE / 'completion.json')
    require(complete['status'] == 'complete', 'Step 1 incomplete')
    pins = {}
    def pin(path, expected=None):
        actual = file_hash(path)
        require(expected is None or actual == expected, f'hash mismatch: {path}')
        pins[rel(path)] = actual
    # Historical completed result hashes and original executable definitions.
    for name, expected in complete['artifact_sha256'].items():
        pin(BASE / name, expected)
    pin(BASE / 'completion.json')
    for path, expected in base['source_sha256'].items():
        pin(ROOT / path, expected)
    pin(checkpoint_path(17), base['checkpoint_sha256'][MODEL])
    step2 = read_json(ROOT / 'docs/analysis/results/paper_control_conditioned_eval_v1/protocol.json')
    for name in [f'{MODEL}_predictions.npz', 'per_origin.csv']:
        path = OLD / name
        pin(path, step2['frozen_input_sha256'][rel(path)])
    registry_path = ROOT / 'configs/data/trajectory_dataset_registry.yaml'
    pin(registry_path, base['registry_sha256'])
    registry = yaml.safe_load(registry_path.read_text())
    require(registry['default_dataset_id'] == base['dataset_id'], 'dataset registry changed')
    entry = registry['datasets'][base['dataset_id']]
    require(entry['manifest_sha256'] == base['manifest_sha256'], 'registry manifest changed')
    manifest_path = ROOT / entry['manifest_path']
    pin(manifest_path, base['manifest_sha256'])
    manifest = read_json(manifest_path)
    assignments = manifest['split_contract']['assignments']
    require(not manifest['split_contract']['sealed_test_opened'], 'test marked open')
    forbidden = set(assignments['sealed_test']) | set(assignments['reserved_evaluation'])
    require(not set(assignments['train']) & set(assignments['validation']), 'split overlap')
    require(not (set(assignments['train']) | set(assignments['validation'])) & forbidden, 'sealed flight overlap')
    cache = torch.load(OLD / 'prepared.pt', map_location='cpu', weights_only=False)
    require(cache['identity']['manifest_sha256'] == base['manifest_sha256'], 'historical cache identity changed')
    prepared = {}
    for part, count in [('train', 28293), ('validation', 2582)]:
        require(assignments[part] == base[f'{part}_flights'], 'split changed')
        # Strict four-file allowlist. No test Parquet, raw logs, or test results.
        for kind in ['samples', 'windows']:
            filename = f'{kind}_{part}.parquet'
            pin(manifest_path.parent / filename, base['artifact_sha256'][filename])
        samples = pd.read_parquet(manifest_path.parent / f'samples_{part}.parquet')
        windows = pd.read_parquet(manifest_path.parent / f'windows_{part}.parquet')
        origins = pd.read_csv(BASE / f'{part}_origins.csv')
        require(set(samples.log_id) == set(assignments[part]) and set(samples.split) == {part}, 'sample split changed')
        require(windows.equals(origins[windows.columns]), 'window identities/order changed')
        require(len(windows) == count and not windows.window_id.duplicated().any(), 'window count mismatch')
        batch = assemble_history_trajectory_windows(samples, windows, history_steps=26)
        same_tree(batch, cache['batches'][part])
        np.testing.assert_array_equal(batch.trajectory.window_ids, origins.window_id)
        require(batch.history_mask.all(), 'incomplete history')
        prepared[part] = batch
        del samples
    cp = torch.load(checkpoint_path(17), map_location='cpu', weights_only=False)
    stats = {key: cp['state_dict'][key].numpy() for key in STATS}
    for key in STATS:
        np.testing.assert_array_equal(stats[key], np.asarray(base['normalization']['values'][key], dtype=np.float32))
    model = build_model(17, stats)
    original = original_build_model(MODEL, stats)
    for key, value in original.state_dict().items():
        require(torch.equal(value, model.state_dict()[key]), 'seed17 initialization parity failed')
    model.load_state_dict(cp['state_dict'], strict=True)
    require(sum(p.numel() for p in model.parameters()) == 21383, 'parameter count changed')
    archive = np.load(OLD / f'{MODEL}_predictions.npz', allow_pickle=False)
    np.testing.assert_array_equal(archive['window_ids'], prepared['validation'].trajectory.window_ids)
    prediction = TrajectoryPrediction(**{key: archive[key] for key in vars(prepared['validation'].trajectory.truth)})
    recomputed = endpoint_metrics(prediction, prepared['validation'].trajectory, model=MODEL, seed=17)
    old_rows = pd.read_csv(OLD / 'per_origin.csv')
    old_rows = old_rows[old_rows.model == MODEL].reset_index(drop=True)
    validate_rows(old_rows, prepared['validation'], 17)
    order = ['window_id', 'horizon_s']
    for frame in [recomputed, old_rows]:
        frame.sort_values(order, inplace=True, ignore_index=True)
    np.testing.assert_allclose(recomputed[list(METRIC_MAP)], old_rows[list(METRIC_MAP)], rtol=1e-11, atol=1e-10)
    per, summary, _ = aggregate_flights(old_rows)
    old_summary = pd.read_csv(BASE / 'summary.csv').query('model == @MODEL').sort_values(['cohort', 'horizon_s'])
    np.testing.assert_allclose(summary[METRICS], old_summary[METRICS], rtol=1e-12, atol=1e-12)
    old_rows.to_csv(ART / 'seed17_per_origin.csv', index=False)
    for stage, epochs, *_ in stages(17):
        path = OLD / f'{MODEL}_{stage}_history.csv'
        pin(path)
        validate_history(pd.read_csv(path), epochs)
    torch.save(prepared, ART / 'prepared.pt')
    for path in [SCRIPT, ROOT / 'scripts/report_paper_standard_gru_multiseed.py', ROOT / 'tests/test_paper_standard_gru_multiseed.py']:
        pin(path)
    protocol = {key: base[key] for key in [
        'branch', 'dataset_id', 'manifest_path', 'manifest_sha256', 'artifact_sha256',
        'train_flights', 'validation_flights', 'excluded_sealed_flights', 'frames',
        'model_phase', 'model_phase_source_detail', 'phase_contract', 'frequency_contract',
        'history_steps', 'nominal_history_span_s', 'nominal_dt_s', 'horizons_s',
        'primary_horizon_s', 'timing_contract', 'checkpoint_rule', 'future_known', 'future_forbidden',
    ]}
    protocol.update(
        experiment='paper_standard_gru_multiseed_v1', source_baseline_commit='cdf7625af4301642df61a085c3998383bbdbbe62',
        source_baseline_result=rel(BASE), git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        main_model=MODEL, architecture='CausalHistoryTrajectoryModel(hidden_size=64,use_controls=True); GRUCell(16,64), head Linear(80,64)/Tanh/Linear(64,7)',
        parameter_count=21383, seeds=list(SEEDS), trained_seeds=[23,42], reused_seed17=True,
        stage_seed_rule='initialization/base=s; continuation=s+12, preserving historical 17->29',
        training_schedule=base['training_schedule'], increment_scales=base['ours_training_protocol']['scales'],
        increment_weights=base['ours_training_protocol']['weights'], epochs=65, batch_size=256,
        normalization=dict(source='six frozen float32 B2 seed17 buffers; never fitted in Step 3', values=base['normalization']['values']),
        normalization_sha256=normalization_hash(stats), normalization_hash_encoding='for ordered STATS: JSON [key,shape,<f4] then little-endian float32 C-order bytes',
        train_window_sha256=pins[rel(BASE/'train_origins.csv')],
        validation_origin_sha256=pins[rel(BASE/'validation_origins.csv')],
        training_windows=28293, validation_origins=2582, validation_flights_count=17,
        preprocessing_parity='all freshly assembled train/validation history, controls, dt, state, target and identity arrays exactly equal Step1 cache; no stats fitting',
        seed17_prediction_parity='frozen NPZ hash verified against Step2; metrics recomputed and match Step1; no seed17 retraining or full re-inference',
        optimizer=dict(name='AdamW', betas=[.9,.999], eps=1e-8, weight_decay=1e-5, amsgrad=False,
                       reset_each_stage=True, lr_schedule='constant .0003 for40 then .0005 for25; no scheduler', gradient_clip_norm=5.),
        sampling='CPU torch.Generator(stage_seed); full randperm of same28293 windows everyepoch;111 batches;drop_last=False;no DataLoader/workers',
        training_objective='unchanged trajectory_rollout_loss + wv*increment_terms[0]+ww*increment_terms[1], lag2; continuation adds .2*frequency_MSE via existing zero-regularizer adapter',
        checkpoint_selection='last epoch65; no best_epoch, no validation-based selection; final validation loss calculated for reporting only',
        metrics={**base['metrics'], 'uncertainty':'sample SD(ddof=1) across three seed-specific equal-flight macro means; no windows as independent samples'},
        runtime_policy='same4torchthreads/deterministic_algorithms=True/cudnn.benchmark=False/CUBLAS_WORKSPACE_CONFIG=:4096:8; other defaults recorded, unchanged',
        seed17_rng_provenance='torch CPU/CUDA init17;base17;continuation29;samplers17/29. Python/NumPy not explicitly seeded historically; unused by stochastic training path. Historical runtime versions not recorded; do not claim bitwise cross-version reproducibility.',
        robustness_screen=dict(description='predeclared descriptive screen, not significance test or model selection',
            max_primary_cv=.20, max_1s_cv=.20, require_all_1s_errors_above_100ms=True,
            require_consistent_primary_cohort_gap_sign=True),
        frozen_input_sha256=pins, prepared_sha256=file_hash(ART / 'prepared.pt'),
        sealed_test_opened_this_run=False, reserved_evaluation_opened_this_run=False,
        sealed_test_status='Only explicit train/validation files read; excluded names are manifest metadata only.',
    )
    # Remove misleading old stage seeds; exact per-replicate mapping follows.
    protocol['training_schedule'] = {**protocol['training_schedule'], 'sampling_seeds': {str(s): [s,s+12] for s in SEEDS}}
    write_json(OUT / 'protocol.json', protocol)
    write_json(OUT / 'audit.json', dict(status='passed', seed17_reusable=True, train_windows=28293,
        validation_origins=2582, checkpoint_sha256=pins[rel(checkpoint_path(17))],
        normalization_sha256=normalization_hash(stats), fresh_assembly_exact_cache_parity=True,
        seed17_initialization_exact_parity=True, seed17_prediction_metric_parity=True,
        sealed_test_opened_this_run=False, reserved_evaluation_opened_this_run=False))
    (OUT / 'git_status_at_launch.txt').write_text(subprocess.check_output(['git','status','--short'],text=True))
    print('Audit passed: frozen seed17 reusable; exact train/validation array parity; no normalization fit.', flush=True)


def validate_history(history, epochs):
    require(list(history.epoch) == list(range(1, epochs + 1)), 'incomplete training history')
    require(np.isfinite(history.select_dtypes('number').to_numpy()).all(), 'nonfinite training history')
    require(int(history.optimizer_steps.iloc[-1]) == 111 * epochs, 'training sample count changed')


def runtime(seed):
    return dict(seed=seed, python_seed=[seed,seed+12], numpy_seed=[seed,seed+12],
        torch_seed=[seed,seed+12], cuda_seed_all_devices=[seed,seed+12],
        initialization_seed=seed, dataloader='not used; CPU torch.Generator per stage', sampler_seeds=[seed,seed+12],
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'), threads=torch.get_num_threads(),
        device=DEVICE, device_name=torch.cuda.get_device_name(DEVICE), torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
        python_version=sys.version, numpy_version=np.__version__)


def val_loss(model, batch, protocol):
    total = 0.
    model.to(DEVICE).eval()
    with torch.no_grad():
        for start in range(0,len(batch.trajectory.window_ids),128):
            ids = np.arange(start,min(start+128,len(batch.trajectory.window_ids)))
            p,t = _model_call(model,batch,ids,use_history=True,rollout_steps=50,device=torch.device(DEVICE))
            iv,iw = increment_terms(p,t,protocol['increment_scales'])
            loss = trajectory_rollout_loss(p,t,objective_steps=50) + .2*(p.flap_frequency_hz[:,1:51]-t.flap_frequency_hz[:,1:51]).square().mean()
            loss = loss + protocol['increment_weights'][0]*iv + protocol['increment_weights'][1]*iw
            require(bool(torch.isfinite(loss)), 'nonfinite final validation loss')
            total += float(loss)*len(ids)
    model.cpu()
    return total/len(batch.trajectory.window_ids)


def preflight():
    protocol, prepared, stats = load_inputs()
    require(torch.cuda.is_available(), 'CUDA unavailable; no CPU fallback')
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'CUBLAS setting differs')
    rt = runtime(23)
    small = subset(prepared['train'], 4)
    model = build_model(23, stats)
    checks = []
    for stage, _, seed, lr, frequency in stages(23):
        seed_all(seed)
        wrapped = _FrequencyLossAdapter(model) if frequency else model
        trained, history = train_increment_stage(wrapped, small, scales=protocol['increment_scales'],
            weights=protocol['increment_weights'], device=DEVICE, epochs=1, seed=seed,
            learning_rate=lr, actuator=frequency, steps=50, batch_size=256)
        model = trained.model if frequency else trained
        require(np.isfinite(history.select_dtypes('number').to_numpy()).all(), 'preflight nonfinite')
        checks.append(dict(stage=stage,loss=float(history.loss.iloc[-1]),optimizer_steps=1))
    require(normalization_hash({k:model.state_dict()[k].numpy() for k in STATS}) == protocol['normalization_sha256'], 'preflight normalization mutated')
    leakage = causality_check(model, subset(prepared['validation'],3), DEVICE)
    write_json(OUT/'preflight.json', dict(status='passed', runtime=rt, stages=checks, checks=leakage,
        disposable_four_window_model=True, no_experiment_checkpoint_written=True,
        protocol_sha256=file_hash(OUT/'protocol.json')))
    print('GPU preflight passed for both training stages and label-poisoning/prefix parity.',flush=True)


def worker(seed):
    require(seed in (23,42), 'seed17 must never be retrained')
    folder = ART / f'seed{seed}'
    folder.mkdir(exist_ok=False)
    write_json(folder/'status.json',dict(status='starting',pid=os.getpid(),seed=seed))
    try:
        protocol, prepared, stats = load_inputs()
        write_json(folder/'runtime.json',runtime(seed))
        model = build_model(seed,stats)
        for stage,epochs,stage_seed,lr,frequency in stages(seed):
            seed_all(stage_seed)
            wrapped = _FrequencyLossAdapter(model) if frequency else model
            def callback(_,history):
                history.to_csv(folder/f'{stage}_history.csv',index=False)
                write_json(folder/'status.json',dict(status='training',pid=os.getpid(),seed=seed,
                    stage=stage,epoch=int(history.epoch.iloc[-1]),loss=float(history.loss.iloc[-1])))
            trained,history = train_increment_stage(wrapped,prepared['train'],scales=protocol['increment_scales'],
                weights=protocol['increment_weights'],device=DEVICE,epochs=epochs,seed=stage_seed,
                learning_rate=lr,actuator=frequency,steps=50,batch_size=256,callback=callback)
            model = trained.model if frequency else trained
            validate_history(history,epochs)
            torch.save(dict(state_dict=model.state_dict(),seed=seed,stage=stage,
                protocol_sha256=file_hash(OUT/'protocol.json'),normalization_sha256=protocol['normalization_sha256']),folder/f'{stage}.pt')
        torch.save(dict(state_dict=model.state_dict(),seed=seed,final_epoch=65,
            protocol_sha256=file_hash(OUT/'protocol.json'),normalization_sha256=protocol['normalization_sha256']),folder/'model.pt')
        write_json(folder/'status.json',dict(status='training_complete',seed=seed,final_epoch=65,
            checkpoint_sha256=file_hash(folder/'model.pt')))
    except BaseException:
        write_json(folder/'status.json',dict(status='failed',seed=seed,error=traceback.format_exc(),automatic_retry=False))
        raise


def evaluate():
    protocol, prepared, stats = load_inputs()
    rows, runs, checks = [], [], {}
    for seed in SEEDS:
        path = checkpoint_path(seed)
        cp = torch.load(path,map_location='cpu',weights_only=False)
        model = build_model(seed,stats)
        model.load_state_dict(cp['state_dict'],strict=True)
        require(normalization_hash({k:cp['state_dict'][k].numpy() for k in STATS}) == protocol['normalization_sha256'], 'checkpoint normalization changed')
        if seed != 17:
            require(cp['protocol_sha256'] == file_hash(OUT/'protocol.json') and cp['final_epoch'] == 65, 'checkpoint provenance mismatch')
        checks[str(seed)] = causality_check(model,subset(prepared['validation'],3),DEVICE)
        if seed == 17:
            frame = pd.read_csv(ART/'seed17_per_origin.csv')
        else:
            prediction = predict_history_trajectory_model(model,prepared['validation'],use_history=True,batch_size=128,device=DEVICE)
            np.savez_compressed(ART/f'seed{seed}'/'predictions.npz',**vars(prediction),window_ids=prepared['validation'].trajectory.window_ids.astype(str))
            frame = endpoint_metrics(prediction,prepared['validation'].trajectory,model=MODEL,seed=seed)
            frame.to_csv(ART/f'seed{seed}'/'per_origin.csv',index=False)
        validate_rows(frame,prepared['validation'],seed)
        rows.append(frame)
        histories = []
        for stage,epochs,*_ in stages(seed):
            hist_path = OLD/f'{MODEL}_{stage}_history.csv' if seed == 17 else ART/f'seed{seed}'/f'{stage}_history.csv'
            history = pd.read_csv(hist_path)
            validate_history(history,epochs)
            histories.append(history)
        final_val = val_loss(model,prepared['validation'],protocol)
        checks[str(seed)].update(checkpoint_load=True,normalization_identical=True,finite_losses=True,
            n_origins=2582,endpoint_rows=len(frame),horizon_timing_identical=True)
        runs.append(dict(seed=seed,checkpoint_path=rel(path),checkpoint_sha256=file_hash(path),final_epoch=65,
            best_epoch=None,final_train_loss=float(histories[-1].loss.iloc[-1]),final_val_loss=final_val,
            training_time_s=sum(float(h.wall_time_s.iloc[-1]) for h in histories),
            status='reused_complete' if seed == 17 else 'complete'))
    all_rows = pd.concat(rows,ignore_index=True)
    per, summary, aggregate = aggregate_flights(all_rows)
    require(len(summary) == 36 and len(per) == 204, 'aggregation count mismatch')
    summary.to_csv(OUT/'per_seed_summary.csv',index=False)
    per.to_csv(OUT/'per_flight_wide.csv',index=False)
    per.rename(columns={'log_id':'flight_id'}).melt(
        id_vars=['seed','flight_id','cohort','horizon_s','n_windows'],value_vars=list(METRIC_MAP.values()),
        var_name='metric',value_name='value').to_csv(OUT/'per_flight_per_seed.csv',index=False)
    aggregate.to_csv(OUT/'across_flights_per_seed.csv',index=False)
    pd.DataFrame(runs).to_csv(OUT/'training_runs.csv',index=False)
    write_json(OUT/'sanity_checks.json',dict(status='passed',models=checks,seed17_reused=True,
        same_train_windows=True,same_validation_origins=True,normalization_refit=False,
        sealed_test_opened_this_run=False,reserved_evaluation_opened_this_run=False))
    from report_paper_standard_gru_multiseed import generate
    generate(OUT,protocol)
    verify_pins(protocol['frozen_input_sha256'])
    (OUT/'git_status_at_completion.txt').write_text(subprocess.check_output(['git','status','--short'],text=True))
    hashes = {rel(p):file_hash(p) for p in OUT.iterdir() if p.is_file() and p.name!='completion.json'}
    for seed in (23,42):
        hashes.update({rel(p):file_hash(p) for p in (ART/f'seed{seed}').iterdir() if p.is_file()})
    write_json(OUT/'completion.json',dict(status='complete',completed_at_unix=time.time(),
        training_completed_seeds=[23,42],reused_seeds=[17],training_runs=runs,
        artifact_sha256=hashes,normalization_sha256=protocol['normalization_sha256'],
        train_window_sha256=protocol['train_window_sha256'],validation_origin_sha256=protocol['validation_origin_sha256'],
        tests=read_json(OUT/'tests.json'),sanity_checks='sanity_checks.json',
        robustness=read_json(OUT/'interpretation.json'),
        sealed_test_opened_this_run=False,reserved_evaluation_opened_this_run=False,ablation_started=False))


def run():
    write_json(ART/'status.json',dict(status='running',pid=os.getpid(),started_at_unix=time.time()))
    try:
        processes = []
        for seed in (23,42):
            log = (ART/f'seed{seed}.log').open('w')
            child = subprocess.Popen([sys.executable,str(SCRIPT),'--phase','worker','--seed',str(seed)],
                                     cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            log.close()
            processes.append((seed,child))
        write_json(ART/'workers.json',{str(seed):child.pid for seed,child in processes})
        # Block on completion; no epoch polling or metric-based intervention.
        exits = {str(seed):child.wait() for seed,child in processes}
        require(all(code == 0 for code in exits.values()),f'worker failure; no automatic retries: {exits}')
        evaluate()
        write_json(ART/'status.json',dict(status='complete',pid=os.getpid(),completion=rel(OUT/'completion.json')))
    except BaseException:
        error = traceback.format_exc()
        write_json(ART/'status.json',dict(status='failed',error=error,automatic_retry=False))
        write_json(OUT/'completion.json',dict(status='failed',error=error,automatic_retry=False,
            sealed_test_opened_this_run=False,reserved_evaluation_opened_this_run=False))
        raise


def launch():
    require(not (ART/'launch.json').exists(), 'already launched; inspect status, never silently rerun')
    require(read_json(OUT/'tests.json')['returncode'] == 0, 'tests not passed')
    preflight()
    log = (ART/'run.log').open('w')
    child = subprocess.Popen([sys.executable,str(SCRIPT),'--phase','run'],cwd=ROOT,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    log.close()
    write_json(ART/'launch.json',dict(pid=child.pid,command=[sys.executable,str(SCRIPT),'--phase','run'],
        launched_at_unix=time.time(),protocol_sha256=file_hash(OUT/'protocol.json'),
        automatic_evaluation_and_report=True,epoch_monitoring=False))
    print(json.dumps(read_json(ART/'launch.json')),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['audit','preflight','launch','run','worker','evaluate'],required=True)
    parser.add_argument('--seed',type=int,choices=[23,42])
    args = parser.parse_args()
    configure()
    if args.phase == 'worker':
        require(args.seed is not None,'worker requires seed')
        worker(args.seed)
    else:
        globals()[args.phase]()


if __name__ == '__main__':
    main()
