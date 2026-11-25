"""Fit and evaluate the circular calibration using only train and development encoder data."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from module import ARMS, HERE, PRIOR, build_design, circular_factors, complex_unit

SOURCE = Path('/home/zn/flap-system-identification')
spec = importlib.util.spec_from_file_location('clock_labels', PRIOR / 'clock/calibrate.py')
labels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(labels)


def extract(x, base, target, t, b):
    ramp = t + .2 * np.expm1(-t / .2)
    conditions = x[..., 1:] / ramp[..., None]
    phase = 2 * np.pi * (target - base - x @ b)
    unit = np.stack((np.cos(phase), np.sin(phase)), -1)
    design = build_design(torch.from_numpy(t.astype('float64')), torch.from_numpy(conditions)).numpy()
    return design, unit


def fit(source):
    labels.DATA = source / 'dataset/velocity_rate_multistep_v1'
    labels.MEM = source / 'outputs/history_slow_fast_20261119_cache'
    labels.OLD = source / 'outputs/autopilot_flight_dynamics_20261124'
    meta = {r['flight_id']: r for r in json.loads((labels.DATA / 'flight_provenance.json').read_text())
            if r['split'] in ('train', 'validation')}
    b = np.array(json.loads((PRIOR / 'clock/clock_fit.json').read_text())['coefficients'])
    names = ('global', 'conditional', 'sham')
    dims = [3, 45, 45]
    grams = {name: np.zeros((d, d)) for name, d in zip(names, dims)}
    rhs = {name: np.zeros((d, 2)) for name, d in zip(names, dims)}
    counts = {}
    rng = np.random.default_rng(17)
    for fid, x, base, target, mask, t in labels.rows('train', meta):
        design, unit = extract(x, base, target, t.astype(np.float64), b)
        sham = unit.copy()
        for q in range(t.shape[1]):
            for c in range(t.shape[2]):
                good = mask[:, q, c]
                v = unit[good, q, c]
                sham[good, q, c] = v[rng.permutation(len(v))]
        counts[fid] = int(mask.sum())
        real = unit[mask] - [1., 0.]
        permuted = sham[mask] - [1., 0.]
        for arm, d in zip(names, dims):
            a = design[mask, :d]
            grams[arm] += a.T @ a / len(a)
            rhs[arm] += a.T @ (permuted if arm == 'sham' else real) / len(a)
    solved, residual = {}, {}
    for arm in names:
        g = grams[arm]
        normal = g + .001 * np.diag(np.diag(g))
        w = np.linalg.solve(normal, rhs[arm])
        solved[arm] = torch.from_numpy(w)
        residual[arm] = np.max(np.abs(normal @ w - rhs[arm])) / np.max(np.abs(rhs[arm]))
    assert len(counts) == 121 and max(residual.values()) < 1e-9
    np.savez(HERE / 'clock.npz', **{k: v.numpy() for k, v in solved.items()})
    records = {}
    covers = {}
    for fid, x, base, target, mask, t in labels.rows('validation', meta):
        design, unit = extract(x, base, target, t.astype(np.float64), b)
        estimates = circular_factors(torch.from_numpy(design), solved)
        estimates['global'] = estimates.pop('global_')
        estimates['legacy'] = torch.from_numpy(np.stack((np.exp(-t / 2), np.zeros_like(t)), -1))
        row, cov = {}, {}
        for group, valid in {'early': mask & (t <= .5), 'mid': mask & (t > .5) & (t <= 1),
                             'late': mask & (t > 1) & (t <= 2.1)}.items():
            assert valid.any()
            obs = torch.from_numpy(unit[valid])
            cov[group] = int(valid.sum())
            for arm, estimate in estimates.items():
                v = estimate[valid]
                values = dict(loss=(v-obs).square().sum(-1).mean().item(),
                              alignment=(complex_unit(v)*obs).sum(-1).mean().item(),
                              magnitude=torch.linalg.vector_norm(v, dim=-1).mean().item())
                row.setdefault(group, {})[arm] = values
        records[fid] = row
        covers[fid] = cov
    assert len(records) == 39 and not (set(counts) & set(records))
    rng = np.random.default_rng(17)
    draw = rng.integers(0, len(records), (10000, len(records)))
    summary = {}
    for group in ('early', 'mid', 'late'):
        grouped = [records[fid][group] for fid in sorted(records)]
        row = {arm: {k: float(np.mean([f[arm][k] for f in grouped])) for k in ('loss','alignment','magnitude')}
               for arm in (*ARMS, 'legacy')}
        for ref, arm in [('global','conditional'), ('sham','conditional'), ('legacy','conditional'),
                         ('global','magnitude'), ('direction','conditional'),
                         ('global','direction'), ('magnitude','conditional')]:
            delta = np.array([v[ref]['loss'] - v[arm]['loss'] for v in grouped])
            row[f'{arm}_vs_{ref}'] = {'loss_reduction': delta.mean(), 'interval95': np.quantile(
                delta[draw].mean(-1), [.025,.975]).tolist(), 'improved_flights': int(sum(delta > 0))}
        summary[group] = row
    (HERE / 'clock_flights.json').write_text(json.dumps(dict(
        train_counts=counts, dev_counts=covers, dev_scores=records), ensure_ascii=False) + '\n')
    (HERE / 'clock_summary.json').write_text(json.dumps(dict(
        train_flights=len(counts), train_pairs=sum(counts.values()), dev_flights=len(records),
        normal_residual=residual, scores=summary), indent=2) + '\n')
    print('late', summary['late'], flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-project', type=Path, default=SOURCE)
    torch.set_num_threads(2)
    fit(ap.parse_args().source_project)
