"""Fresh baseline inference on explicit development partitions; no test access."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

import run_paper_rollout_horizon_ablation as source
from system_identification.evaluation.paper_baselines import aggregate_flights, endpoint_metrics
from system_identification.training.trajectory_main_v1 import predict_history_trajectory_model

m = source.m
OUT = m.ROOT / 'docs/analysis/results/control_dynamics_realdata_v1/baseline'


def run(device):
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'protocol.json').exists():
        raise FileExistsError('Immutable run exists; do not overwrite')
    m.configure()
    sp, batches, stats = source.inputs()
    registry_path = m.ROOT / 'configs/data/trajectory_dataset_registry.yaml'
    registry = yaml.safe_load(registry_path.read_text())
    dataset_id = registry['default_dataset_id']
    entry = registry['datasets'][dataset_id]
    assert dataset_id == sp['dataset_id']
    assert entry['manifest_path'] == sp['manifest_path']
    manifest = m.ROOT / entry['manifest_path']
    assert m.file_hash(manifest) == entry['manifest_sha256'] == sp['manifest_sha256']
    pins = {m.rel(registry_path): m.file_hash(registry_path), m.rel(manifest): m.file_hash(manifest)}
    for split in ('train', 'validation'):
        for kind in ('samples', 'windows'):
            name = f'{kind}_{split}.parquet'
            path = manifest.parent / name
            assert m.file_hash(path) == sp['artifact_sha256'][name]
            pins[m.rel(path)] = sp['artifact_sha256'][name]
    original_runs = pd.read_csv(m.OUT / 'training_runs.csv').set_index('seed')
    completion = m.read_json(m.OUT / 'completion.json')
    for seed in m.SEEDS:
        cp = m.checkpoint_path(seed)
        assert m.file_hash(cp) == original_runs.loc[seed, 'checkpoint_sha256']
        pins[m.rel(cp)] = m.file_hash(cp)
        pp = source.predpath(seed)
        expected = sp['frozen_input_sha256'].get(m.rel(pp), completion['artifact_sha256'].get(m.rel(pp)))
        assert expected and m.file_hash(pp) == expected
        pins[m.rel(pp)] = expected
    for path in [m.ART/'prepared.pt', m.OUT/'protocol.json', Path(__file__),
                 m.ROOT/'scripts/run_paper_rollout_horizon_ablation.py',
                 m.ROOT/'scripts/run_paper_standard_gru_multiseed.py']:
        pins[m.rel(path)] = m.file_hash(path)
    # Pin all current implementation files without opening any test artifacts.
    for path in sorted((m.ROOT/'src/system_identification').rglob('*.py')):
        pins[m.rel(path)] = m.file_hash(path)
    protocol = dict(experiment='control_dynamics_realdata_v1_baseline',
        head=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'], text=True).strip(),
        created_utc=pd.Timestamp.now(tz='UTC').isoformat(), dataset_id=dataset_id,
        dataset_entry=entry, manifest_contract=json.loads(manifest.read_text()),
        partitions=['train','validation'], seeds=list(m.SEEDS), device=device,
        heldout_accessed=False, prior_heldout_results_known=True, training=False,
        normalization_sha256=sp['normalization_sha256'], pins=pins,
        parity_tolerance=dict(rtol=1e-5, atol=1e-4),
        metric='native dt; per-flight RMS then equal-flight mean; all original validation origins',
        purpose='baseline reproduction only; no control readiness claim')
    m.write_json(OUT/'protocol.json', protocol)
    rows, checks = [], {}
    batch = batches['validation']
    for seed in m.SEEDS:
        m.write_json(OUT/'status.json', dict(status='running', seed=seed))
        model = m.build_model(seed, stats)
        cp = torch.load(m.checkpoint_path(seed), map_location='cpu', weights_only=False)
        model.load_state_dict(cp['state_dict'], strict=True)
        checks[str(seed)] = m.causality_check(model, m.subset(batch, 3), device)
        pred = predict_history_trajectory_model(model, batch, use_history=True, batch_size=128, device=device)
        differences = {}
        with np.load(source.predpath(seed), allow_pickle=False) as old:
            np.testing.assert_array_equal(old['window_ids'], batch.trajectory.window_ids)
            for key, value in vars(pred).items():
                np.testing.assert_allclose(value, old[key], rtol=1e-5, atol=1e-4)
                differences[key] = float(np.max(np.abs(value-old[key])))
        checks[str(seed)]['max_absolute_difference'] = differences
        rows.append(endpoint_metrics(pred, batch.trajectory, model='StandardGRU_H26_K50', seed=seed))
        m.write_json(OUT/'checks.json', checks)
        print(f'seed {seed} reproduced: {differences}', flush=True)
    frame = pd.concat(rows, ignore_index=True)
    flight, seed_summary, _ = aggregate_flights(frame)
    flight.to_csv(OUT/'per_flight.csv', index=False)
    seed_summary.to_csv(OUT/'per_seed.csv', index=False)
    metrics = m.METRICS
    seed_summary.groupby(['cohort','horizon_s'])[metrics].agg(['mean','std']).to_csv(OUT/'summary.csv')
    m.verify_pins(pins)
    m.write_json(OUT/'completion.json', dict(status='complete', heldout_accessed=False,
        input_hashes_unchanged=True, seeds=list(m.SEEDS), origins=len(batch.trajectory.window_ids),
        outputs={m.rel(p):m.file_hash(p) for p in OUT.iterdir() if p.suffix in ('.csv','.json') and p.name not in ('status.json','completion.json')}))
    m.write_json(OUT/'status.json', dict(status='complete'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    run(parser.parse_args().device)
