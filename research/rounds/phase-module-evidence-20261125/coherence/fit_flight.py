"""Fit the same convex readout to each carrier using the original eight actual query times."""
import argparse
import json
from pathlib import Path
import runpy

import numpy as np
import torch

from module import ARMS, HERE, PRIOR, MomentPredictor, prior

NAMES = ('v05', 'w05', 'v1', 'w1', 'v2', 'w2')
SEEDS = (17, 29, 43)
RIDGE = 1e-5


def load_data(project, phase):
    with np.load(project / 'outputs/autopilot_flight_dynamics_20261124' / (phase + '.npz')) as z:
        raw = {k: torch.from_numpy(z[k]) for k in ('time', 'u0', 'pooled', 'mask', 'y', 'hist')}
        flights = z['flight']
    extra = np.load(project / 'outputs/autopilot_periodic_motion_20261125' / (phase + '_features.npy'))
    raw['hist'] = torch.cat((raw['hist'], torch.from_numpy(extra)), -1)
    return raw, flights


def batches(raw, device):
    for i in range(0, len(raw['time']), 1024):
        yield {k: v[i:i + 1024].to(device) for k, v in raw.items()}


def setup(seed, device):
    artifact = PRIOR / 'readout/original_runs' / f'incumbent{seed}/corrected.pt'
    state = torch.load(artifact, weights_only=True, map_location='cpu')
    model = MomentPredictor(state['mean'], state['std'])
    legacy = prior.FlightPredictor(state['mean'], state['std'])
    legacy.load_state_dict(state)
    missing, extra = model.load_state_dict(state, strict=False)
    assert set(missing) == {'time_moment', 'conditional_moment', 'sham_moment'} and not extra
    with np.load(HERE / 'clock.npz') as moments:
        for key, target in model.coefficients().items():
            target.copy_(torch.from_numpy(moments[key]))
    return model.to(device).eval(), legacy.to(device).eval()


def solve(model, train, device):
    n = 3 * 47
    grams = torch.zeros(len(ARMS), 2, n, n, device=device, dtype=torch.float64)
    rhs = torch.zeros(len(ARMS), 2, n, 3, device=device, dtype=torch.float64)
    counts = torch.zeros(2, device=device, dtype=torch.long)
    for b in batches(train, device):
        h, t, u, pool = (b[k] for k in ('hist','time','u0','pooled'))
        x = model.context(h).double()
        base = model.core(h[:, :model.mean.numel()], pool, u, t)
        remaining = (b['y'] - base).double()
        basis = model.bases(h, pool, u, t)
        counts += b['mask'].sum((0,1))
        for c in (0, 1):
            good = b['mask'][:, :, c].reshape(-1)
            y = remaining[:,:,c,:].reshape(-1,3)
            for j, arm in enumerate(ARMS):
                z = (basis[arm][:,:,c,:,None].double() * x[:,None,None,:]).reshape(-1,n)
                z = z * good[:,None]
                grams[j,c].addmm_(z.T, z)
                rhs[j,c].addmm_(z.T, y)
    diag = grams.diagonal(dim1=-2, dim2=-1)
    normal = grams + torch.diag_embed(RIDGE*diag + (diag == 0))
    weights = torch.linalg.solve(normal, rhs)
    resid = ((normal @ weights-rhs).abs().amax((-2,-1))/rhs.abs().amax((-2,-1))).cpu().numpy()
    assert np.max(resid) < 1e-9
    return weights.reshape(len(ARMS), 2, 3, 47, 3).float(), resid.tolist(), counts.tolist()


def evaluate(model, legacy, weights, dev, ids, device):
    scores = {name: [] for name in ('legacy', *ARMS)}
    for b in batches(dev, device):
        h, t, u, pool = (b[k] for k in ('hist','time','u0','pooled'))
        args = (h, pool, u, t)
        pred = {'legacy': legacy(*args)}
        base = model.core(h[:, :model.mean.numel()], pool, u, t)
        x = model.context(h)
        basis = model.bases(*args)
        for j, name in enumerate(ARMS):
            coef = torch.einsum('bh,ckha->bcka', x, weights[j])
            pred[name] = base + torch.einsum('bqck,bcka->bqca', basis[name], coef)
        for name, values in pred.items():
            errors = (values - b['y']).square().sum(-1)[:, [1,3,7]].reshape(-1,6)
            scores[name].append(errors.cpu().numpy())
    valid = dev['mask'][:, [1,3,7]].reshape(-1,6).numpy()
    scores = {arm: np.concatenate(parts) for arm, parts in scores.items()}
    fids = np.unique(ids)
    return {arm: {key: {fid: float(a[(ids==fid) & valid[:,j], j].mean()) for fid in fids}
                  for j,key in enumerate(NAMES)} for arm,a in scores.items()}


@torch.inference_mode()
def main(source, device):
    train, fit_flights = load_data(source, 'train')
    dev, dev_flights = load_data(source, 'validation')
    assert len(np.unique(fit_flights)) == 121 and len(np.unique(dev_flights)) == 39
    assert set(fit_flights).isdisjoint(dev_flights)
    scores, fit_summary = {}, {}
    for seed in SEEDS:
        model, legacy = setup(seed, device)
        weights, residual, counts = solve(model, train, device)
        out = evaluate(model, legacy, weights, dev, dev_flights, device)
        source_scores = json.loads((PRIOR / 'readout/original_runs' / f'incumbent{seed}/flight_mse.json').read_text())
        max_difference = max(abs(x-source_scores['corrected'][k][fid]) for k in NAMES for fid,x in out['legacy'][k].items())
        assert max_difference < 3e-6, max_difference
        scores[str(seed)] = out
        fit_summary[str(seed)] = dict(equation_residual=residual, training_target_counts=counts,
                                      max_legacy_flight_mse_difference=max_difference)
        model.weights.copy_(weights[ARMS.index('conditional')])
        # Save the inference-ready conditional model and optional matched refitted heads.
        torch.save(model.cpu().state_dict(), HERE / f'conditional_seed{seed}.pt')
        np.savez(HERE / f'heads_seed{seed}.npz', **{
            arm: weights[j].cpu().numpy() for j, arm in enumerate(ARMS)})
        print(seed, {arm:float(np.sqrt(np.mean(list(s['w2'].values())))) for arm,s in out.items()}, flush=True)
    (HERE / 'flight_mse.json').write_text(json.dumps(scores) + '\n')
    aggregate = runpy.run_path(str(PRIOR / 'analyze_accuracy.py'))['aggregate']
    results = aggregate(scores, [('legacy','global'), ('legacy','conditional'), ('global','conditional'),
        ('sham','conditional'), ('global','magnitude'), ('direction','conditional'), ('global','direction'),
        ('magnitude','conditional'), ('legacy','direction'), ('sham','direction')])
    results.update(readout_parameters=846, readout_ridge=RIDGE, fit=fit_summary,
                   train_windows=len(fit_flights), dev_windows=len(dev_flights), seeds=SEEDS)
    (HERE / 'flight_summary.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-project', type=Path, default=Path('/home/zn/flap-system-identification'))
    ap.add_argument('--device', default='cuda:1')
    args = ap.parse_args()
    torch.set_num_threads(2)
    main(args.source_project, args.device)
