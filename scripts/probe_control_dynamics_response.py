"""Train-supported local command sensitivity; model evidence, not causal truth."""
from __future__ import annotations

import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch

import run_paper_rollout_horizon_ablation as source
from system_identification.training.trajectory_main_v1 import predict_history_trajectory_model

m = source.m
OUT = m.ROOT/'docs/analysis/results/control_dynamics_realdata_v1/response_probe'
CHANNELS = ['drive', 'common', 'differential', 'rudder']
DIRECTIONS = np.array([[1,0,0,0], [0,1,1,0], [0,1,-1,0], [0,0,0,1]], dtype=float)


def coordinates(u):
    return np.stack([u[...,0], (u[...,1]+u[...,2])/2, (u[...,1]-u[...,2])/2, u[...,3]], axis=-1)


def run():
    assert not OUT.exists(), 'immutable experiment already exists'
    OUT.mkdir(parents=True)
    m.configure()
    baseline = m.read_json(OUT.parent/'baseline/protocol.json')
    m.verify_pins(baseline['pins'])
    sp, batches, stats = source.inputs()
    train = batches['train']
    batch = copy.deepcopy(batches['validation'])
    # Fixed log-balanced identity subset, chosen without prediction errors.
    ids = batch.trajectory.log_ids
    chosen = np.concatenate([np.flatnonzero(ids == log)[np.linspace(0, (ids == log).sum()-1,
        min(16, (ids == log).sum()), dtype=int)] for log in sorted(set(ids))])
    from dataclasses import fields, is_dataclass, replace
    def take(x):
        if is_dataclass(x):
            return replace(x, **{f.name: take(getattr(x, f.name)) for f in fields(x)})
        return x[chosen].copy() if isinstance(x, np.ndarray) else x
    batch = take(batch)
    raw = train.trajectory.controls.reshape(-1, 4)
    lower, upper = np.quantile(raw, [.01, .99], axis=0)
    eps = .1 * coordinates(raw).std(axis=0)
    assert np.all(eps > 0)
    # Marginal bounds are a necessary screen only, not a joint-support guarantee.
    hs = train.history_state_features[:, -1]
    slo, shi = np.quantile(hs, [.01,.99], axis=0)
    inside_state = ((batch.history_state_features[:, -1] >= slo) & (batch.history_state_features[:, -1] <= shi)).all(1)
    u0 = batch.trajectory.controls[:, 0].copy()
    batch.trajectory.controls[:] = u0[:, None, :]
    pins = dict(baseline['pins'])
    pins[m.rel(Path(__file__))] = m.file_hash(Path(__file__))
    m.write_json(OUT/'protocol.json', dict(experiment='local_command_seed_disagreement',
        hypothesis='similar logged prediction errors may conceal inconsistent intervention sensitivity across seeds',
        created_utc=pd.Timestamp.now(tz='UTC').isoformat(), pins=pins,
        selection='up to16 identity-spaced origins per validation flight; no error selection',
        channels=CHANNELS, raw_lower=lower.tolist(), raw_upper=upper.tolist(),
        epsilon=eps.tolist(), state_lower=slo.tolist(), state_upper=shi.tolist(),
        command='hold current command throughout50 steps; +/-0.1 train coordinate std for all future steps',
        history='identical observed H26 history and t0 for both signs; future state unobserved',
        supported='both signs inside raw train q01..q99 and initial history-state feature box; marginal only, not joint density',
        reported='central derivative at steps5/10/25/50; p/q/r and vx/vy/vz; seed sign agreement, gain dispersion',
        decision='retain as failure diagnostic if seed sign disagreement exists within support; no truth sign or delay claim',
        heldout_accessed=False, training=False, seeds=list(m.SEEDS)))
    rows = []
    checks = {}
    for seed in m.SEEDS:
        model = m.build_model(seed, stats)
        model.load_state_dict(torch.load(m.checkpoint_path(seed), map_location='cpu', weights_only=False)['state_dict'])
        for j, channel in enumerate(CHANNELS):
            delta = eps[j]*DIRECTIONS[j]
            supported = inside_state & ((u0-delta >= lower) & (u0-delta <= upper) & (u0+delta >= lower) & (u0+delta <= upper)).all(1)
            outputs = []
            for sign in [-1,1]:
                b = copy.deepcopy(batch)
                b.trajectory.controls[:] += sign*delta[None,None,:]
                p = predict_history_trajectory_model(model, b, use_history=True, batch_size=128, device='cpu')
                a = np.concatenate([p.velocity_n, p.angular_velocity_b], axis=-1)
                assert np.isfinite(a).all()
                outputs.append(a)
            gain = (outputs[1]-outputs[0])/(2*eps[j])
            for horizon, step in [(.1,5),(.2,10),(.5,25),(1.,50)]:
                for i, wid in enumerate(batch.trajectory.window_ids):
                    for axis, value in zip(['vx','vy','vz','p','q','r'], gain[i,step]):
                        rows.append(dict(seed=seed, window_id=wid, log_id=batch.trajectory.log_ids[i],
                            channel=channel, horizon_s=horizon, axis=axis, supported=bool(supported[i]), gain=float(value)))
            print(seed, channel, 'supported', int(supported.sum()), flush=True)
        # Actual future commands after step10 must not affect the first10 states.
        small = m.subset(batch,3)
        other = copy.deepcopy(small)
        other.trajectory.controls[:,10:] += .123
        a = predict_history_trajectory_model(model, small, use_history=True, batch_size=3, device='cpu')
        b = predict_history_trajectory_model(model, other, use_history=True, batch_size=3, device='cpu')
        for key in vars(a): np.testing.assert_array_equal(getattr(a,key)[:,:11], getattr(b,key)[:,:11])
        checks[str(seed)] = dict(future_command_prefix_invariant=True, finite=True)
    f = pd.DataFrame(rows)
    f.to_csv(OUT/'per_origin_seed.csv', index=False)
    idx = ['window_id','log_id','channel','horizon_s','axis','supported']
    g = f.pivot(index=idx, columns='seed', values='gain').reset_index()
    gains = g[list(m.SEEDS)].to_numpy()
    g['all_seed_sign_agree'] = (gains > 0).all(1) | (gains < 0).all(1)
    g['absolute_gain_mean'] = np.abs(gains).mean(1)
    g['gain_seed_std'] = gains.std(1, ddof=1)
    g['near_zero_any_seed'] = (np.abs(gains)<1e-4).any(1)
    g.to_csv(OUT/'cross_seed.csv', index=False)
    keys = ['channel','horizon_s','axis']
    per = g[g.supported].groupby(keys+['log_id']).agg(sign_agreement=('all_seed_sign_agree','mean'),
        absolute_gain_mean=('absolute_gain_mean','mean'), gain_seed_std=('gain_seed_std','mean'),
        near_zero_fraction=('near_zero_any_seed','mean'), n_origins=('window_id','size')).reset_index()
    per.to_csv(OUT/'per_flight.csv', index=False)
    summary = per.groupby(keys).agg(sign_agreement=('sign_agreement','mean'),
        absolute_gain_mean=('absolute_gain_mean','mean'),gain_seed_std=('gain_seed_std','mean'),
        near_zero_fraction=('near_zero_fraction','mean'),n_flights=('log_id','nunique'),n_origins=('n_origins','sum')).reset_index()
    summary.to_csv(OUT/'summary.csv', index=False)
    m.verify_pins(pins)
    m.write_json(OUT/'checks.json', checks)
    m.write_json(OUT/'completion.json', dict(status='complete',heldout_accessed=False,
        outputs={m.rel(p):m.file_hash(p) for p in OUT.iterdir() if p.is_file()}))


if __name__ == '__main__':
    run()
