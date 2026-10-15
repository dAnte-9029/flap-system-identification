"""Frozen, original-train-only Rao predictor ablations against v35.

Run prepare once, then launch. No validation/test loader, tuning, early stopping,
or future complementary state is used. Reuses v34 exact parent/offset plans.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

import train_control_windows_v2 as base
from train_joint_increment_v34 import load_inputs, fingerprint
from joint_increment_v34 import objective, select_local
from actuator_training_v30 import ActuatorCommandGRU, model_from as baseline_from
from audit_controller_support_v26 import verify_parent
from reference_groupcv_v8 import predict
from system_identification.models.paper_predictors import PaperPredictor

ROOT = base.ROOT
RESULTS = ROOT / 'docs/analysis/results'
OUT = RESULTS / 'control_windows_v46'
DATA = RESULTS / 'control_windows_v29'
PREP = RESULTS / 'control_windows_v34'
ARMS = PaperPredictor.ARMS
SEEDS = (17, 23, 42)
STATS = ('state_mean', 'state_scale', 'reference_mean', 'reference_scale',
         'actuator_mean', 'actuator_scale', 'rate_scale', 'velocity_scale')


def read(path):
    return json.loads(Path(path).read_text())


def baseline_path(fold, seed):
    return RESULTS / f'control_windows_v35/fold{fold}/joint_increment/seed{seed}'


def destination(arm, fold, seed):
    return OUT / arm / f'fold{fold}/seed{seed}'


def model_from(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    assert ck['input_contract'] == 'post_allocation_motor0_servo012_v29'
    model = PaperPredictor({k: ck['state_dict'][k] for k in STATS}, ck['arm'])
    model.load_state_dict(ck['state_dict'])
    return model.eval()


def prepare():
    torch.set_num_threads(1)
    assert not OUT.exists()
    registry_path = ROOT / 'configs/data/trajectory_dataset_registry.yaml'
    registry = yaml.safe_load(registry_path.read_text())
    dataset_id = registry['default_dataset_id']
    assert dataset_id == 'trajectory_v3_september_expanded'
    entry = registry['datasets'][dataset_id]
    assert base.sha(ROOT / entry['manifest_path']) == entry['manifest_sha256']
    contract = read(DATA / 'dataset_manifest.json')
    assert contract['partition'] == 'original_train_only'
    paths = [registry_path, ROOT / entry['manifest_path'], DATA / 'dataset_manifest.json',
             DATA / 'windows.npz', DATA / 'windows.csv', PREP / 'training_plan.json',
             RESULTS / 'control_windows_v33/rolling_inputs.npz']
    for version in [29, 33, 34, 35]:
        parent = RESULTS / f'control_windows_v{version}'
        verify_parent(parent)
        paths.append(parent / 'delivery_manifest.json')
        paths.extend(ROOT / p for p in read(parent / 'delivery_manifest.json')['code'])
    # Verify all materialized sample artifacts in the explicitly requested train
    # contract. The registry's other partitions are neither loaded nor hashed.
    for source in contract['sources']:
        assert source['source']['partition'] == 'train'
        assert base.sha(source['source']['path']) == source['source']['sha256']
        assert base.sha(ROOT / source['cache']) == source['cache_sha256']
        paths.extend([Path(source['source']['path']), ROOT / source['cache']])
    paths.extend([ROOT / 'src/system_identification/models/paper_predictors.py', Path(__file__),
                  ROOT / 'tests/test_paper_predictors_v46.py'])
    paper = ROOT / 'docs/papers/Rao 等 - 2024 - Learning Long-Horizon Predictions for Quadrotor Dynamics.pdf'
    paths.append(paper)
    windows = pd.read_csv(DATA / 'windows.csv')
    assert len(windows) == 407 and windows.partition.eq('train').all()
    old_runs = read(PREP / 'training_plan.json')['runs']
    runs = []
    for fold in range(5):
        info = read(DATA / f'fold{fold}/manifest.json')
        fit, query = info['fit_indices'], info['query_indices']
        assert set(windows.iloc[fit].log_id).isdisjoint(windows.iloc[query].log_id)
        paths.extend([DATA / f'fold{fold}/manifest.json', DATA / f'fold{fold}/normalization.npz'])
        with np.load(DATA / f'fold{fold}/normalization.npz') as z:
            stats = dict(z)
        for seed in SEEDS:
            old = next(r for r in old_runs if (r['fold'], r['seed']) == (fold, seed))
            assert base.sha(PREP / old['sample_plan']) == old['sample_plan_sha256']
            assert base.sha(DATA / f'fold{fold}/normalization.npz') == old['normalization_sha256']
            paths.append(PREP / old['sample_plan'])
            paths.extend(baseline_path(fold, seed) / n for n in
                         ['final.pt', 'fit_predictions.npz', 'oof_predictions.npz', 'rolling_predictions.npz'])
            torch.manual_seed(seed)
            baseline = ActuatorCommandGRU(stats)
            assert fingerprint(baseline) == old['initial_parameter_sha256']
            for arm in ARMS:
                torch.manual_seed(seed)
                model = PaperPredictor(stats, arm)
                if arm != 'decoupled':
                    for name, p in model.named_parameters():
                        if name.startswith(('encoder.', 'cell.')):
                            torch.testing.assert_close(p, dict(baseline.named_parameters())[name], rtol=0, atol=0)
                runs.append(dict(arm=arm, fold=fold, seed=seed, sample_plan=old['sample_plan'],
                    sample_plan_sha256=old['sample_plan_sha256'],
                    normalization_sha256=old['normalization_sha256'],
                    initial_parameter_sha256=fingerprint(model),
                    parameter_count=sum(p.numel() for p in model.parameters())))
    OUT.mkdir()
    pins = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): base.sha(p)
            for p in sorted(set(paths))}
    protocol = dict(status='frozen_before_training', version='paper_predictor_v46',
        dataset_id=dataset_id, registry_entry=entry, parent_data_contract=contract, pins=pins,
        scope='407 original train windows; 5 leave-one-flight-out folds; seeds17,23,42. Repeated-development OOF, not an untouched test.',
        paper=dict(pdf=str(paper.relative_to(ROOT)), doi='10.1109/IROS58592.2024.10801793',
            repository='https://github.com/arplaboratory/long-horizon-dynamics',
            inspected_commit='a8451a119b9096eda980f71b55dbb05012b8c47a',
            source_files={name: base.sha('/tmp/rao-' + name + '.py') for name in ['lighting', 'gru', 'eval']},
            definition='Figure2/SectionIV-C: z=(linear velocity, angular velocity) versus attitude; multihead shares encoder; decoupled uses separate encoders/decoders.',
            code_caveat='lighting.py unroll_step AND eval_trajectory substitute true complementary states. We do not: our primary protocol jointly rolls all predicted states.',
            adaptation='GRU H26/K50 at50Hz; body derivative velocity integration as v35; attitude exp-map3 instead of unconstrained quaternion4. No paper TCN, H20/U10/100Hz, or paper metric reproduction.'),
        architecture=dict(baseline='Frozen v35 physical quaternion integration,40070parameters.',
            full='Shared GRU64 plus head64->9:6 velocity derivatives and3 attitude tangent corrections. Controls change of quaternion update.',
            multihead='Shared GRU64 plus separate32wide heads6 and3. Same attitude update as full.',
            decoupled='Two GRU42 plus42wide heads6 and3. Stop gradients through complementary predicted states, including rotation for acceleration; own recurrent graph retained.',
            attitude='q_next=q*Exp((observed_initial_rate+clamp(tangent,-6,6))*dt); unit quaternion normalized. Same initial constant-rate predictor in all arms.',
            capacity='39531/39977/40265 total parameters within1.5percent of40070. Equal total budget, not equal per-module width. No capacity tuning.',
            gradient='Decoupled parameter losses are separated by stop-gradient; AdamW hyperparameters and global norm clip1 stay fixed. Global clipping still shares a scalar between modules.'),
        training=dict(updates=1600, batch=32, local_batch=32, lr=3e-4, weight_decay=1e-4,
            clip=1., optimizer='AdamW', local_weight=1.,
            objective='Unchanged v35 full normalized rate+velocity+.2attitude, plus local normalized rate. Exact v34 parent/offset plans; final checkpoint only.',
            initialization='Same seeds; full/multihead encoder and cell tensors equal v35 at initialization; output topology differs. Decoupled independent branches, width42, recorded fingerprints.',
            repeats='Exactly3newarms*5folds*3seeds=45runs. No failed-result retuning.'),
        evaluation=dict(primary='v35 original gate: full1s qRMSE >=5percent gain, eachseedpositive, >=4/5flightpositive; report both newarms vs full as structural contrast.',
            protections='18cells: eachseed p/r1swave and velocity/attitude .5/1sendpoint, <=5percent regression.',
            absolute='12cells: active .5sendpoint p/q<=.15rad/s, velocity<=.5m/s, attitude<=3degrees.',
            diagnostics='p/q/r wave, segmentmeanbias, centeredwave, Pearson correlation, local fresh-history20ms rate, quaternion-to-integrated-predicted-rate discrepancy; all407preserved.',
            promotion='No control-ready or physical-causal promotion from these reused-development OOF metrics; no closed-loop trial.'),
        environment=dict(python=sys.executable, torch=torch.__version__, numpy=np.__version__, platform=platform.platform()),
        validation_accessed=False, sealed_test_accessed=False, reserved_accessed=False, control_ready=False)
    base.write(OUT / 'protocol.json', protocol)
    base.write(OUT / 'runs.json', runs)
    base.write(OUT / 'status.json', dict(status='ready', total_runs=len(runs)))
    print({arm: next(r['parameter_count'] for r in runs if r['arm'] == arm) for arm in ARMS})


def check_pins(code_only=False):
    for name, digest in read(OUT / 'protocol.json')['pins'].items():
        if code_only and not name.endswith('.py'):
            continue
        assert base.sha(ROOT / name) == digest, name


def rolling(model, lx, ids):
    values = {k: v[ids].flatten(0, 1) for k, v in lx.items()
              if k not in ['history_reference', 'future_reference']}
    result = predict(model, values, np.arange(len(ids) * 50))
    return {k: v.reshape(len(ids), 50, v.shape[-1]) for k, v in result.items()}


def train(arm, fold, seed):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    check_pins(code_only=True)
    run = next(r for r in read(OUT / 'runs.json') if (r['arm'], r['fold'], r['seed']) == (arm, fold, seed))
    root = destination(arm, fold, seed)
    assert not (root / 'final.pt').exists()
    assert base.sha(PREP / run['sample_plan']) == run['sample_plan_sha256']
    assert base.sha(DATA / f'fold{fold}/normalization.npz') == run['normalization_sha256']
    with np.load(DATA / f'fold{fold}/normalization.npz') as z:
        stats = dict(z)
    torch.manual_seed(seed)
    model = PaperPredictor(stats, arm)
    assert fingerprint(model) == run['initial_parameter_sha256']
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    x, y, lx, ly = load_inputs()
    info = read(DATA / f'fold{fold}/manifest.json')
    with np.load(PREP / run['sample_plan']) as z:
        draws, offsets = z['array_index'], z['offset']
    assert draws.shape == offsets.shape == (1600, 32)
    assert set(draws.ravel()) <= set(info['fit_indices'])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    started = time.monotonic()
    curve = []
    for step, (ids, offs) in enumerate(zip(draws, offsets), 1):
        bx, by = {k: v[ids] for k, v in x.items()}, {k: v[ids] for k, v in y.items()}
        local_x, local_y = select_local(lx, ly, ids, offs, info['fit_indices'])
        opt.zero_grad(set_to_none=True)
        loss, full, local = objective(model, bx, by, local_x, local_y)
        assert torch.isfinite(loss)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        if step == 1 or step % 400 == 0:
            row = dict(step=step, loss=float(loss.detach()), full=float(full.detach()),
                local=float(local.detach()), gradient_norm=float(norm), elapsed_s=time.monotonic()-started)
            curve.append(row)
            pd.DataFrame(curve).to_csv(root / 'curve.csv', index=False)
            base.write(root / 'status.json', dict(status='training', **row))
            print(row, flush=True)
    for k, v in model.named_buffers():
        torch.testing.assert_close(v, buffers[k], rtol=0, atol=0)
    torch.save(dict(state_dict=model.state_dict(), arm=arm, steps=1600, fold=fold, seed=seed,
        protocol_sha256=base.sha(OUT / 'protocol.json'),
        input_contract='post_allocation_motor0_servo012_v29'), root / 'final.pt')
    model.eval()
    for cohort, ids in [('fit', info['fit_indices']), ('oof', info['query_indices'])]:
        np.savez_compressed(root / f'{cohort}_predictions.npz', array_index=ids, **predict(model, x, ids))
    np.savez_compressed(root / 'rolling_predictions.npz', array_index=info['query_indices'],
                        **rolling(model, lx, info['query_indices']))
    restored = model_from(root / 'final.pt')
    with np.load(root / 'oof_predictions.npz') as z:
        reproduced = predict(restored, x, info['query_indices'])
        for k, v in reproduced.items():
            np.testing.assert_array_equal(v, z[k])
    # Recheck each original baseline once (15 total), without retraining it.
    if arm == 'full':
        baseline = baseline_from(baseline_path(fold, seed) / 'final.pt')
        with np.load(baseline_path(fold, seed) / 'oof_predictions.npz') as z:
            for k, v in predict(baseline, x, info['query_indices']).items():
                np.testing.assert_array_equal(v, z[k])
    base.write(root / 'status.json', dict(status='complete', step=1600, elapsed_s=time.monotonic()-started,
        checkpoint_sha256=base.sha(root / 'final.pt'), reload_exact=True, buffers_unchanged=True,
        baseline_replay_exact=True if arm == 'full' else None))


def launch(workers):
    check_pins()
    runs = read(OUT / 'runs.json')
    base.write(OUT / 'status.json', dict(status='running', workers=workers, total_runs=len(runs), pid=os.getpid()))

    def child(run):
        arm, fold, seed = run['arm'], run['fold'], run['seed']
        root = destination(arm, fold, seed)
        root.mkdir(parents=True, exist_ok=False)
        base.write(root / 'run_protocol.json', dict(**run, protocol_sha256=base.sha(OUT / 'protocol.json')))
        with (root / 'train.log').open('w') as log:
            subprocess.run([sys.executable, __file__, '--stage', 'train', '--arm', arm,
                            '--fold', str(fold), '--seed', str(seed)], cwd=ROOT,
                           stdout=log, stderr=subprocess.STDOUT, check=True,
                           env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
        print('complete', arm, fold, seed, flush=True)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(child, runs))
    except Exception as exc:
        base.write(OUT / 'status.json', dict(status='failed', error=str(exc)))
        raise
    check_pins(code_only=True)
    base.write(OUT / 'status.json', dict(status='training_complete', total_runs=len(runs)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['prepare', 'launch', 'train'], required=True)
    p.add_argument('--arm', choices=ARMS)
    p.add_argument('--fold', type=int)
    p.add_argument('--seed', type=int)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    if args.stage == 'train':
        train(args.arm, args.fold, args.seed)
    elif args.stage == 'launch':
        launch(args.workers)
    else:
        prepare()
