"""Independent scalar metric and provenance verification for the v46 delivery."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch

import paper_ablation_v46 as run


def main():
    torch.set_num_threads(1)
    run.check_pins()
    evaluation_pin = run.read(run.OUT / 'evaluation_protocol.json')
    for name, digest in evaluation_pin['code'].items():
        assert run.base.sha(run.ROOT / name) == digest
    plans = run.read(run.OUT / 'runs.json')
    assert len(plans) == 45 and len({(r['arm'], r['fold'], r['seed']) for r in plans}) == 45
    assert len(list(run.OUT.glob('*/fold*/seed*/final.pt'))) == 45
    protocol_sha = run.base.sha(run.OUT / 'protocol.json')
    for row in plans:
        root = run.destination(row['arm'], row['fold'], row['seed'])
        status = run.read(root / 'status.json')
        assert status['status'] == 'complete' and status['step'] == 1600
        assert status['reload_exact'] and status['buffers_unchanged']
        if row['arm'] == 'full':
            assert status['baseline_replay_exact']
        checkpoint = torch.load(root / 'final.pt', weights_only=False, map_location='cpu')
        assert checkpoint['protocol_sha256'] == protocol_sha and checkpoint['steps'] == 1600
        assert checkpoint['arm'] == row['arm'] and checkpoint['seed'] == row['seed'] and checkpoint['fold'] == row['fold']
        assert run.base.sha(root / 'final.pt') == status['checkpoint_sha256']
        assert run.base.sha(run.PREP / row['sample_plan']) == row['sample_plan_sha256']
        with np.load(run.DATA / f"fold{row['fold']}/normalization.npz") as z:
            for key in run.STATS:
                np.testing.assert_array_equal(checkpoint['state_dict'][key].numpy(), np.asarray(z[key], dtype=np.float32))
        curve = pd.read_csv(root / 'curve.csv')
        assert curve.step.tolist() == [1, 400, 800, 1200, 1600]
        assert np.isfinite(curve.to_numpy()).all()
    with np.load(run.DATA / 'windows.npz') as z:
        data = {k: z[k].astype(float) for k in z.files if k.startswith('label_')}
    recorded = pd.read_csv(run.OUT / 'metrics_per_window.csv')
    key_columns = ['arm', 'fold', 'seed', 'cohort', 'array_index', 'horizon', 'kind', 'metric']
    recorded = recorded.set_index(key_columns).mse.sort_index()
    assert not recorded.index.has_duplicates
    independent = []
    quaternion_norm_max_error = 0.
    for fold in range(5):
        info = run.read(run.DATA / f'fold{fold}/manifest.json')
        for seed in run.SEEDS:
            for arm in ('baseline',) + run.ARMS:
                root = run.baseline_path(fold, seed) if arm == 'baseline' else run.destination(arm, fold, seed)
                for cohort, index_key in [('fit', 'fit_indices'), ('oof', 'query_indices')]:
                    with np.load(root / f'{cohort}_predictions.npz') as z:
                        ids = z['array_index']
                        np.testing.assert_array_equal(ids, info[index_key])
                        rate = z['rates_b'].astype(float) - data['label_rates_b'][ids]
                        vel = z['velocity_n'].astype(float) - data['label_velocity_n'][ids]
                        quat = z['quaternion'].astype(float)
                    assert np.isfinite(rate).all() and np.isfinite(vel).all() and np.isfinite(quat).all()
                    quaternion_norm_max_error = max(quaternion_norm_max_error,
                        float(np.abs(np.linalg.norm(quat, axis=-1)-1).max()))
                    # Independent implementation of the frozen published metric,
                    # intentionally preserving its float64 dot/clip convention.
                    cosine = np.einsum('ijk,ijk->ij', quat, data['label_quaternion'][ids])
                    angle_squared = (2*np.arccos(np.minimum(1., np.abs(cosine))))**2
                    squared = {'p': rate[:, :, 0]**2, 'q': rate[:, :, 1]**2, 'r': rate[:, :, 2]**2,
                               'velocity': np.einsum('ijk,ijk->ij', vel, vel)/3, 'attitude': angle_squared}
                    for horizon in [1, 5, 25, 50]:
                        for metric, matrix in squared.items():
                            for kind, values in [('wave', np.sum(matrix[:, :horizon], axis=1)/horizon),
                                                 ('endpoint', matrix[:, horizon-1])]:
                                independent.extend((arm, fold, seed, cohort, int(idx), horizon, kind, metric, float(val))
                                                   for idx, val in zip(ids, values))
    expected = pd.DataFrame(independent, columns=key_columns+['mse']).set_index(key_columns).mse.sort_index()
    assert len(expected) == len(recorded)
    np.testing.assert_allclose(recorded.to_numpy(), expected.loc[recorded.index].to_numpy(), rtol=1e-10, atol=1e-12)
    assert quaternion_norm_max_error < 2e-7
    run.base.write(run.OUT / 'verification.json', dict(status='passed', completed_training_runs=45,
        independent_scalar_metrics=len(expected), max_quaternion_norm_error=quaternion_norm_max_error,
        all_checkpoint_metadata_hashes_and_buffers_verified=True, unchanged_parent_pins_verified=True,
        baseline_oof_prediction_replays=15, candidate_oof_reloads=45,
        baseline_aggregate_metrics_reproduced=run.read(run.OUT / 'evaluation.json')['baseline_metrics_reproduced'],
        evaluation_sources_frozen_before_results=True,
        validation_accessed=False, sealed_test_accessed=False, reserved_accessed=False))
    code = [run.ROOT / p for p in [
        'src/system_identification/models/paper_predictors.py',
        'scripts/paper_ablation_v46.py', 'scripts/evaluate_paper_ablation_v46.py',
        'scripts/verify_paper_ablation_v46.py', 'tests/test_paper_predictors_v46.py',
        'tests/test_paper_ablation_evaluation_v46.py']]
    artifacts = sorted(p for p in run.OUT.rglob('*') if p.is_file() and p.name != 'delivery_manifest.json')
    run.base.write(run.OUT / 'delivery_manifest.json', dict(
        artifacts={str(p.relative_to(run.OUT)): run.base.sha(p) for p in artifacts},
        code={str(p.relative_to(run.ROOT)): run.base.sha(p) for p in code}))
    print(run.read(run.OUT / 'verification.json'))


if __name__ == '__main__':
    main()
