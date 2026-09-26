"""Frozen validation coverage and ensemble-spread audit; no calibration or tuning."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch

import run_paper_rollout_horizon_ablation as source
from system_identification.integration.control_dynamics_candidate import JointSupport,support_vector
from system_identification.models.trajectory_main_v1 import _state_features

m=source.m
ROOT=m.ROOT/'docs/analysis/results/control_dynamics_realdata_v1'
OUT=ROOT/'credibility'


def run():
    OUT.mkdir(parents=True,exist_ok=False);m.configure()
    ep=m.read_json(ROOT/'ensemble_candidate/protocol.json');m.verify_pins(ep['pins'])
    for experiment in ['ensemble_candidate','px4_guarded_closed_loop']:
        m.verify_pins(m.read_json(ROOT/experiment/'completion.json')['outputs'])
    sp,batches,stats=source.inputs();batch=batches['validation'];t=batch.trajectory;n=len(t.window_ids)
    support_path=ROOT/'px4_guarded_closed_loop/support.npz'
    with np.load(support_path,allow_pickle=False) as z:
        support=JointSupport(**{k:z[k] for k in ['reference','mean','scale','lower','upper','distance_limit']})
    pins=dict(ep['pins'])
    for path in [Path(__file__),support_path,ROOT/'ensemble_candidate/predictions.npz',
        m.ROOT/'src/system_identification/integration/control_dynamics_candidate.py']:
        pins[m.rel(path)]=m.file_hash(path)
    m.write_json(OUT/'protocol.json',dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),pins=pins,
        experiment='frozen_candidate_credibility',heldout_accessed=False,training=False,
        support='same train-only joint support as closure; every predicted transition entry and exit with provided logged command tape; cumulative prefix acceptance',
        scoring='all origins retained; supported subset selection never uses future true states/errors',
        spread='sample std of3 independently rolled original frozen seeds; not shared-state derivative std; same-data/model dependence remains',
        uncertainty='report empirical |candidate-truth|<=2*seed_std coverage, not a claimed95%interval; no calibration fitted',
        purpose='measure bias/coverage limits; do not expand support thresholds or select horizons based on desired outcome'))
    with np.load(ROOT/'ensemble_candidate/predictions.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['window_ids'],t.window_ids)
        pred={k:z[k].copy() for k in vars(t.truth)}
    member=[]
    for seed in m.SEEDS:
        with np.load(source.predpath(seed),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['window_ids'],t.window_ids)
            member.append(np.concatenate([z['velocity_n'],z['angular_velocity_b']],-1))
    spread=np.std(np.stack(member),axis=0,ddof=1)
    estimate=np.concatenate([pred['velocity_n'],pred['angular_velocity_b']],-1)
    truth=np.concatenate([t.truth.velocity_n,t.truth.angular_velocity_b],-1)
    def features(k):
        def ten(a):return torch.as_tensor(a,dtype=torch.float32)
        return _state_features(ten(pred['velocity_n'][:,k]),ten(pred['quaternion_nb'][:,k]),ten(pred['angular_velocity_b'][:,k]),
            ten(pred['relative_phase_rad'][:,k]),ten(pred['relative_phase_rad'][:,0]),ten(pred['flap_frequency_hz'][:,k])).numpy()
    past=batch.history_controls[:,:-1].copy()
    prefix=np.ones(n,bool);first_exit=np.full(n,51,int);accepted={};dist=[]
    for k in range(50):
        command=t.controls[:,k]
        pre=support.query(support_vector(features(k),command,past[:,-5]))
        past=np.concatenate([past[:,1:],command[:,None]],axis=1)
        post=support.query(support_vector(features(k+1),command,past[:,-5]))
        stepok=pre['accepted']&post['accepted']
        first_exit[prefix&~stepok]=k+1;prefix&=stepok
        if k+1 in [5,10,25,50]:accepted[k+1]=prefix.copy()
        dist.append(float(np.median(post['distance'])))
    pd.DataFrame(dict(window_id=t.window_ids,log_id=t.log_ids,first_failed_transition=first_exit)).to_csv(OUT/'support_per_origin.csv',index=False)
    rows=[]
    for k,horizon in [(5,.1),(10,.2),(25,.5),(50,1.)]:
        for i in range(n):
            for a,axis in enumerate(['vx','vy','vz','p','q','r']):
                error=float(estimate[i,k,a]-truth[i,k,a]);sd=float(spread[i,k,a])
                rows.append(dict(window_id=t.window_ids[i],log_id=t.log_ids[i],horizon_s=horizon,axis=axis,
                    prefix_supported=bool(accepted[k][i]),error=error,seed_std=sd,inside_2std=abs(error)<=2*sd))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'per_origin.csv',index=False)
    grouped=[]
    for group,data in [('ALL',frame),('prefix_supported',frame[frame.prefix_supported])]:
        for key,g in data.groupby(['horizon_s','axis','log_id']):
            grouped.append(dict(group=group,horizon_s=key[0],axis=key[1],log_id=key[2],n=len(g),
                rmse=float(np.sqrt(np.mean(g.error**2))),abs_error_p95=float(np.quantile(np.abs(g.error),.95)),
                mean_seed_std=float(g.seed_std.mean()),coverage_2std=float(g.inside_2std.mean())))
    f=pd.DataFrame(grouped);f.to_csv(OUT/'per_flight.csv',index=False)
    f.groupby(['group','horizon_s','axis']).agg(n_origins=('n','sum'),n_flights=('log_id','nunique'),rmse=('rmse','mean'),
        mean_seed_std=('mean_seed_std','mean'),coverage_2std=('coverage_2std','mean'),flight_abs_p95_mean=('abs_error_p95','mean')).to_csv(OUT/'summary.csv')
    m.verify_pins(pins)
    m.write_json(OUT/'completion.json',dict(status='complete',heldout_accessed=False,
        uncertainty_calibrated=False,support_proves_causality=False,
        outputs={m.rel(p):m.file_hash(p) for p in OUT.iterdir() if p.is_file()}))


if __name__=='__main__':run()
