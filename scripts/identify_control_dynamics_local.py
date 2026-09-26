"""Grouped development-only linear response contrast; observational, not causal ID."""
from __future__ import annotations

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
from pathlib import Path
import json
import numpy as np
import pandas as pd

import run_paper_rollout_horizon_ablation as source
from probe_control_dynamics_response import coordinates

m = source.m
OUT = m.ROOT/'docs/analysis/results/control_dynamics_realdata_v1/local_identification'
HORIZONS = (1, 5, 10, 25)
AXES = ('vx','vy','vz','p','q','r')
STATE_LAGS = (0,1,2,5,10,25)
PAST_CONTROL_LAGS = (1,2,5,10,25)
EDGES = (0,1,2,5,10,25)


def design(batch, horizon):
    """Strictly observed history + declared future command tape, no future states."""
    assert batch.history_mask.all()
    state = batch.history_state_features[:,[-1-j for j in STATE_LAGS]].reshape(len(batch.history_mask),-1)
    past = coordinates(batch.history_controls[:,[-1-j for j in PAST_CONTROL_LAGS]]).reshape(len(state),-1)
    base = np.concatenate([state,past],1)
    tape = coordinates(batch.trajectory.controls)
    bins = [(lo,min(hi,horizon)) for lo,hi in zip(EDGES[:-1],EDGES[1:]) if lo<horizon]
    planned = np.concatenate([np.mean(tape[:,lo:hi],axis=1) for lo,hi in bins],1)
    return base, np.concatenate([base,planned],1), bins


def fit(x,y,ids,penalty=.01):
    """Flight-equal weighted ridge. Moments and all normalization are fit-only."""
    _, inverse, count = np.unique(ids,return_inverse=True,return_counts=True)
    weight = 1/count[inverse]
    weight /= weight.sum()
    mean = weight@x
    scale = np.sqrt(weight@((x-mean)**2)).clip(1e-6)
    z = (x-mean)/scale
    ym = weight@y
    coef = np.linalg.solve(z.T@(z*weight[:,None])+penalty*np.eye(z.shape[1]), z.T@((y-ym)*weight[:,None]))
    return dict(mean=mean,scale=scale,coef=coef,intercept=ym)


def predict(model,x):
    return (x-model['mean'])/model['scale']@model['coef']+model['intercept']


def labels(batch,k):
    t = batch.trajectory.truth
    return np.concatenate([t.velocity_n[:,k]-t.velocity_n[:,0],t.angular_velocity_b[:,k]-t.angular_velocity_b[:,0]],1)


def flight_scores(y,p,ids,**meta):
    rows=[]
    for log in sorted(set(ids)):
        use=ids==log
        for axis,error in zip(AXES,np.sqrt(np.mean((y[use]-p[use])**2,0))):
            rows.append(dict(**meta,log_id=log,axis=axis,rmse=float(error),n=int(use.sum())))
    return rows


def run():
    OUT.mkdir(parents=True,exist_ok=False)
    m.configure()
    baseline=m.read_json(OUT.parent/'baseline/protocol.json')
    m.verify_pins(baseline['pins'])
    sp,batches,_=source.inputs()
    pins=dict(baseline['pins']);pins[m.rel(Path(__file__))]=m.file_hash(Path(__file__))
    # Whole flights, balanced by number of origins, deterministic tie ordering.
    ids=batches['train'].trajectory.log_ids
    logs,counts=np.unique(ids,return_counts=True)
    load=np.zeros(5);assign={}
    for log,count in sorted(zip(logs,counts),key=lambda p:(-p[1],p[0])):
        j=int(np.argmin(load));assign[log]=j;load[j]+=count
    fold=np.array([assign[log] for log in ids])
    m.write_json(OUT/'protocol.json',dict(
        experiment='local_response_information_contrast',created_utc=pd.Timestamp.now(tz='UTC').isoformat(),pins=pins,
        hypotheses=['planned commands add predictive information beyond state and past commands',
                    'observational sustained-command gain direction may be unstable across whole-flight folds'],
        source_dataset=sp['dataset_id'],partitions=['train','validation'],heldout_accessed=False,
        folds=assign,horizons_steps=HORIZONS,state_lags=STATE_LAGS,past_command_lags=PAST_CONTROL_LAGS,
        feature='body velocity, rates, gravity, relative phase sin/cos, frequency at six observed lags; past command coordinates at five strictly past lags',
        future_command='bin means of supplied postallocation command tape over [0,1,2,5,10,25] clipped at horizon; not future measurements',
        target='physical NED velocity and body angular-velocity change from current observation at k; native dt distribution reported',
        fit='fixed ridge penalty0.01 on standardized predictors, flight-equal loss and moments; no hyperparameter search',
        primary='q at step5: flight-macro OOF RMSE change; retain information hypothesis only if positive gain and improvement on >=60% train flights',
        gain_stability='same sign all five training folds required to call coefficient stable; never proof of causal correctness',
        validation='one evaluation of full-train fitted models; no validation fitting/selection',
        controls='known recorded future commands for prediction contrast; endogenous feedback command not random excitation',
        comparison='history only vs history+planned command, same origin/target/fold/penalty; coefficients conditional on correlated historical features'))
    scores=[];gains=[];excitation=[];timing=[]
    for k in HORIZONS:
        x0,x1,bins=design(batches['train'],k);y=labels(batches['train'],k)
        xv0,xv1,_=design(batches['validation'],k);yv=labels(batches['validation'],k)
        for part in ('train','validation'):
            elapsed=batches[part].trajectory.dt_s[:,:k].sum(1)
            timing.append(dict(partition=part,steps=k,min=float(elapsed.min()),p05=float(np.quantile(elapsed,.05)),median=float(np.median(elapsed)),p95=float(np.quantile(elapsed,.95)),max=float(elapsed.max())))
        for name,x,xv in [('history',x0,xv0),('planned',x1,xv1)]:
            oof=np.full_like(y,np.nan)
            for j in range(5):
                tr=fold!=j;ho=~tr
                assert not set(ids[tr])&set(ids[ho])
                model=fit(x[tr],y[tr],ids[tr]);oof[ho]=predict(model,x[ho])
                if name=='planned':
                    physical=model['coef']/model['scale'][:,None]
                    # Perturb every future bin in one channel; observed history fixed.
                    for c,channel in enumerate(['drive','common','differential','rudder']):
                        total=physical[x0.shape[1]+c::4].sum(0)
                        for a,axis in enumerate(AXES):gains.append(dict(steps=k,fold=j,channel=channel,axis=axis,gain=float(total[a])))
                        for b,(lo,hi) in enumerate(bins):
                            for a,axis in enumerate(AXES):gains.append(dict(steps=k,fold=j,channel=channel,axis=axis,gain=float(physical[x0.shape[1]+4*b+c,a]),bin_start=lo,bin_stop=hi))
            assert np.isfinite(oof).all()
            scores.extend(flight_scores(y,oof,ids,partition='train_oof',steps=k,variant=name))
            model=fit(x,y,ids);vp=predict(model,xv)
            scores.extend(flight_scores(yv,vp,batches['validation'].trajectory.log_ids,partition='validation',steps=k,variant=name))
            np.savez_compressed(OUT/f'{name}_k{k}.npz',**model)
            if name=='planned':
                np.savez_compressed(OUT/f'predictions_k{k}.npz',train_oof=oof,validation=vp,
                    train_ids=batches['train'].trajectory.window_ids.astype(str),validation_ids=batches['validation'].trajectory.window_ids.astype(str))
        # Conditional command predictability using separate fold-trained history regression.
        u=x1[:,x0.shape[1]:];up=np.full_like(u,np.nan)
        for j in range(5):
            tr=fold!=j;ho=~tr;up[ho]=predict(fit(x0[tr],u[tr],ids[tr]),x0[ho])
        for log in sorted(set(ids)):
            use=ids==log
            for b,(lo,hi) in enumerate(bins):
                for c,channel in enumerate(['drive','common','differential','rudder']):
                    idx=4*b+c;v=u[use,idx];r=v-up[use,idx]
                    excitation.append(dict(log_id=log,steps=k,bin_start=lo,bin_stop=hi,channel=channel,
                        residual_rms=float(np.sqrt(np.mean(r*r))),command_std=float(np.std(v)),
                        positive_fraction=float(np.mean(r>0)),negative_fraction=float(np.mean(r<0))))
        print('completed horizon',k,flush=True)
    f=pd.DataFrame(scores);f.to_csv(OUT/'per_flight.csv',index=False)
    paired=f.pivot(index=['partition','steps','log_id','axis'],columns='variant',values='rmse').reset_index()
    paired['relative_gain']=(paired.history-paired.planned)/paired.history
    paired['improved']=paired.planned<paired.history
    paired.to_csv(OUT/'paired_flights.csv',index=False)
    summary=paired.groupby(['partition','steps','axis']).agg(history_rmse=('history','mean'),planned_rmse=('planned','mean'),improved_fraction=('improved','mean'))
    summary['macro_relative_gain']=(summary.history_rmse-summary.planned_rmse)/summary.history_rmse
    summary.to_csv(OUT/'summary.csv')
    pd.DataFrame(gains).to_csv(OUT/'gain_by_fold.csv',index=False)
    pd.DataFrame(excitation).to_csv(OUT/'conditional_command_residual.csv',index=False)
    pd.DataFrame(timing).to_csv(OUT/'elapsed_time.csv',index=False)
    m.verify_pins(pins)
    primary=summary.loc[('train_oof',5,'q')]
    m.write_json(OUT/'completion.json',dict(status='complete',heldout_accessed=False,
        primary_information_hypothesis_retained=bool(primary.macro_relative_gain>0 and primary.improved_fraction>=.6),
        physical_causal_gain_identified=False,outputs={m.rel(p):m.file_hash(p) for p in OUT.iterdir() if p.is_file()}))


if __name__=='__main__':run()
