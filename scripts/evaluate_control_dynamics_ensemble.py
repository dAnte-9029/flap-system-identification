"""Frozen equal-derivative candidate; all selection gates declared before inference."""
from pathlib import Path
import copy
import numpy as np
import pandas as pd
import torch

import run_paper_rollout_horizon_ablation as source
from system_identification.models.control_ensemble import ControlDerivativeEnsemble
from system_identification.evaluation.paper_baselines import endpoint_metrics, aggregate_flights
from system_identification.evaluation.future_control_diagnostic import excitation
from system_identification.training.trajectory_main_v1 import predict_history_trajectory_model
from system_identification.models.trajectory import TrajectoryPrediction

m=source.m
OUT=m.ROOT/'docs/analysis/results/control_dynamics_realdata_v1/ensemble_candidate'


def load_members(stats):
    models=[]
    for seed in m.SEEDS:
        model=m.build_model(seed,stats)
        model.load_state_dict(torch.load(m.checkpoint_path(seed),map_location='cpu',weights_only=False)['state_dict'])
        models.append(model)
    return models


def run():
    OUT.mkdir(parents=True,exist_ok=False);m.configure()
    baseline=m.read_json(OUT.parent/'baseline/protocol.json');m.verify_pins(baseline['pins'])
    sp,batches,stats=source.inputs();batch=batches['validation']
    pins=dict(baseline['pins'])
    for path in [Path(__file__),m.ROOT/'src/system_identification/models/control_ensemble.py',m.ROOT/'tests/test_control_derivative_ensemble.py']:
        pins[m.rel(path)]=m.file_hash(path)
    threshold=float(np.quantile(excitation(batches['train'].trajectory.controls,stats['control_std'],25),.75))
    high=excitation(batch.trajectory.controls,stats['control_std'],25)>threshold
    m.write_json(OUT/'protocol.json',dict(experiment='frozen_equal_derivative_ensemble',pins=pins,
        created_utc=pd.Timestamp.now(tz='UTC').isoformat(),heldout_accessed=False,training=False,
        members=list(m.SEEDS),aggregation='equal physical body acceleration, angular acceleration, frequency derivative at one shared state; separate member hidden states updated with shared next state',
        uncertainty='member derivative spread uncalibrated and not independent-model truth; retains common bias',
        baseline='mean physical metric over existing17/23/42 independent GRU predictions; no best-seed choice',
        groups=['ALL','high_change'],high_change_train_q75=threshold,cohorts=['ALL','Sep7','Sep17'],
        prediction_gate='all group x cohort x100/200ms main four metrics degrade <=1%; at least one ALL/ALL100/200ms velocity/attitude/rate improves >=1%',
        horizons=[.1,.2,.5,1.],long_horizon='stress report only, not candidate selection',
        control_gate='not evaluated here; must run support/response/independent-member PX4 closure and exploitation audit',
        candidate='fixed members, weights, clipping and integrator; no coefficient sign regularization'))
    model=ControlDerivativeEnsemble(load_members(stats)).eval()
    checks=m.causality_check(model,m.subset(batch,3),'cpu')
    pred=predict_history_trajectory_model(model,batch,use_history=True,batch_size=128,device='cpu')
    np.savez_compressed(OUT/'predictions.npz',**vars(pred),window_ids=batch.trajectory.window_ids.astype(str))
    # Self-contained frozen candidate weights; configs needed for member construction retained.
    torch.save(dict(format='control_derivative_ensemble_v1',member_seeds=list(m.SEEDS),hidden_size=64,
        use_controls=True,member_state_dicts=[x.state_dict() for x in model.members],
        normalization_sha256=sp['normalization_sha256']),OUT/'model.pt')
    frames=[]
    for name,seed,p in [('ensemble',0,pred)]+[("baseline",s,None) for s in m.SEEDS]:
        if p is None:
            with np.load(source.predpath(seed),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['window_ids'],batch.trajectory.window_ids)
                p=TrajectoryPrediction(**{k:z[k] for k in vars(batch.trajectory.truth)})
        scores=endpoint_metrics(p,batch.trajectory,model=name,seed=seed)
        high_ids=set(batch.trajectory.window_ids[high])
        for group,mask in [('ALL',np.ones(len(scores),bool)),('high_change',scores.window_id.isin(high_ids).to_numpy())]:
            flight,summary,_=aggregate_flights(scores[mask])
            frames.append(flight.assign(group=group))
    f=pd.concat(frames,ignore_index=True);f.to_csv(OUT/'per_flight.csv',index=False)
    # Average independent baseline seeds inside each flight, then equal flights.
    metrics=m.METRICS
    per=f.groupby(['model','group','cohort','log_id','horizon_s'])[metrics].mean().reset_index()
    per=pd.concat([per,per.assign(cohort='ALL')])
    summary=per.groupby(['model','group','cohort','horizon_s'])[metrics].mean().reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    b=summary[summary.model=='baseline'].drop(columns='model')
    c=summary[summary.model=='ensemble'].drop(columns='model')
    paired=b.merge(c,on=['group','cohort','horizon_s'],suffixes=('_baseline','_ensemble'))
    rows=[]
    for row in paired.to_dict('records'):
        for metric in metrics:
            base=row[metric+'_baseline'];candidate=row[metric+'_ensemble']
            rows.append(dict(group=row['group'],cohort=row['cohort'],horizon_s=row['horizon_s'],metric=metric,
                baseline=base,ensemble=candidate,relative_gain=(base-candidate)/base))
    compare=pd.DataFrame(rows);compare.to_csv(OUT/'comparison.csv',index=False)
    short=compare[compare.horizon_s.isin([.1,.2])]
    primary=short[(short.group=='ALL')&(short.cohort=='ALL')&(short.metric!='position_rmse_m')]
    no_regression=bool((short.relative_gain>=-.01).all());improved=bool((primary.relative_gain>=.01).any())
    m.verify_pins(pins)
    m.write_json(OUT/'checks.json',checks)
    m.write_json(OUT/'completion.json',dict(status='complete',heldout_accessed=False,
        prediction_gate_passed=no_regression and improved,nonregression_passed=no_regression,meaningful_improvement=improved,
        control_ready=False,outputs={m.rel(p):m.file_hash(p) for p in OUT.iterdir() if p.is_file()}))
    print(compare[(compare.group=='ALL')&(compare.cohort=='ALL')].to_string(index=False),flush=True)
    print('prediction gate',no_regression and improved,flush=True)


if __name__=='__main__':run()
