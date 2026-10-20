"""Frozen-checkpoint validation, flight-equal metrics and observation diagnostics."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'))
from system_identification.models.velocity_rate_gru import VelocityRateGRU,baseline,errors,integrate
from system_identification.data.velocity_rate_dataset import write_json,file_hash
from train_velocity_rate_learning import load,batch,window_loss
from prepare_velocity_rate_learning import EXP,CACHE
METRICS=['position_m','velocity_m_s','attitude_deg','rate_rad_s']


def oracle(initial,target,dt):
    s=initial; out=[]
    for k in range(dt.shape[1]):
        s=integrate(s,target[:,k,3:6],target[:,k,10:13],dt[:,k]); out.append(s)
    return torch.stack(out,1)


def endpoint_frame(err,times,mask,meta,model,seed):
    # Same native future rows for every model. No synthetic fixed-dt transitions.
    frames=[]
    for horizon in range(20,201,20):
        distance=np.where(mask,abs(times-horizon/1000),np.inf)
        index=distance.argmin(1)
        assert (distance[np.arange(len(index)),index] <= .012001).all()
        f=meta[['window_id','flight_id','review','high_dynamics','mode_change','control_change']].copy()
        f['horizon_ms']=horizon; f['actual_time_s']=times[np.arange(len(index)),index]
        f['native_step']=index+1; f['model']=model; f['seed']=seed
        for j,name in enumerate(METRICS): f[name]=err[np.arange(len(index)),index,j]
        frames.append(f)
    return pd.concat(frames,ignore_index=True)


def summarize(all_endpoints,thresholds):
    keys=['model','seed','horizon_ms','flight_id']
    flight=all_endpoints.groupby(keys)[METRICS].mean().reset_index()
    flight['n_windows']=all_endpoints.groupby(keys).size().to_numpy()
    flight.to_csv(EXP/'per_flight.csv',index=False)
    seed=flight.groupby(['model','seed','horizon_ms'])[METRICS].mean().reset_index()
    seed.to_csv(EXP/'per_seed.csv',index=False)
    rows=[]
    for model,g in flight.groupby('model'):
        for horizon,gh in g.groupby('horizon_ms'):
            # Average repeated seeds within flight before between-flight SD.
            f=gh.groupby('flight_id')[METRICS].mean()
            s=seed[(seed.model==model)&(seed.horizon_ms==horizon)]
            windows=all_endpoints[(all_endpoints.model==model)&(all_endpoints.horizon_ms==horizon)]
            # Seeds are repeats, not independent windows or flights.
            window_averaged=windows.groupby('window_id')[METRICS].mean()
            for metric in METRICS:
                rows.append(dict(model=model,horizon_ms=horizon,metric=metric,
                    flight_mean=f[metric].mean(),flight_std=f[metric].std(ddof=1),
                    seed_mean=s[metric].mean(),seed_std=s[metric].std(ddof=1) if len(s)>1 else None,
                    window_mean=window_averaged[metric].mean(),window_std=window_averaged[metric].std(ddof=1),
                    n_flights=len(f),n_windows=len(window_averaged),n_seeds=len(s)))
    pd.DataFrame(rows).to_csv(EXP/'summary.csv',index=False)
    paired=[]; rng=np.random.default_rng(20261020)
    for h in (100,200):
        f=flight[flight.horizon_ms==h].groupby(['model','flight_id'])[METRICS].mean()
        for comparator in ('Kinematic Baseline','GRU-NoControl'):
            for candidate in ('GRU-NoControl','GRU-Control'):
                if comparator==candidate: continue
                for metric in METRICS:
                    ref=f.loc[comparator,metric]; val=f.loc[candidate,metric].reindex(ref.index)
                    d=(val-ref).to_numpy()
                    ci=np.quantile(d[rng.integers(len(d),size=(5000,len(d)))].mean(1),[.025,.975])
                    paired.append(dict(horizon_ms=h,reference=comparator,candidate=candidate,metric=metric,
                        delta_mean=d.mean(),relative_improvement_pct=100*(ref.mean()-val.mean())/ref.mean(),
                        better_flights=int((d<0).sum()),flights=len(d),bootstrap_low=ci[0],bootstrap_high=ci[1]))
    pd.DataFrame(paired).to_csv(EXP/'paired_comparisons.csv',index=False)
    frames=[]
    endpoint=all_endpoints[all_endpoints.horizon_ms.isin([100,200])].copy()
    for field in ('review','high_dynamics','mode_change','control_change'):
        f=endpoint.copy()
        f['group']=pd.cut(f.control_change,[-np.inf,*thresholds,np.inf],labels=['low','medium','high']).astype(str) if field=='control_change' else f[field].astype(str)
        g=f.groupby(['model','seed','horizon_ms','group','flight_id'])[METRICS].mean().reset_index()
        g=g.groupby(['model','horizon_ms','group','flight_id'])[METRICS].mean().reset_index()
        means=g.groupby(['model','horizon_ms','group'])[METRICS].agg(['mean','std'])
        means.columns=['_'.join(c) for c in means.columns]; means=means.reset_index()
        means['n_flights']=g.groupby(['model','horizon_ms','group']).size().to_numpy()
        counts=f.groupby(['model','horizon_ms','group']).window_id.nunique()
        means['n_windows']=counts.reindex(pd.MultiIndex.from_frame(means[['model','horizon_ms','group']])).to_numpy()
        means['grouping']=field; frames.append(means)
    pd.concat(frames,ignore_index=True).to_csv(EXP/'group_metrics.csv',index=False)


@torch.no_grad()
def main():
    torch.set_num_threads(4); torch.backends.cudnn.deterministic=True; torch.backends.cuda.matmul.allow_tf32=True
    device=torch.device('cuda:0'); data=load('validation',device)
    config=json.loads((EXP/'config.json').read_text()); norm=json.loads((EXP/'normalization.json').read_text())
    meta=pd.read_parquet(EXP/'validation_windows.parquet')
    train_meta=pd.read_parquet(EXP/'train_windows.parquet')
    thresholds=train_meta.control_change.quantile([1/3,2/3]).tolist()
    write_json(EXP/'group_contract.json',dict(control_change_thresholds_train_only=thresholds,
        control_change_definition='L2 of per-channel peak-to-peak over history and future event tape; normalized command units',
        high_dynamics_definition='dataset native flag; may use future observations for diagnostic grouping only',
        review_definition='dataset full-window REVIEW overlap; never a model feature'))
    times=data['dt'].cumsum(1).cpu().numpy(); mask=data['future_mask'].cpu().numpy()
    choices=[]
    fids=meta.flight_id.unique()
    for fid in fids[np.linspace(0,len(fids)-1,6,dtype=int)]:
        ids=np.flatnonzero(meta.flight_id.to_numpy()==fid); choices.append(int(ids[len(ids)//2]))
    examples=dict(indices=np.array(choices),initial=data['initial'][choices].cpu().numpy(),
        target=data['target'][choices].cpu().numpy(),dt=data['dt'][choices].cpu().numpy(),mask=mask[choices])
    models=[('Kinematic Baseline',-1,None)]
    for name in ('GRU-NoControl','GRU-Control'):
        for seed in config['seeds']:
            path=EXP/name/f'seed_{seed}/best.pt'
            if not path.exists(): raise FileNotFoundError(path)
            ckpt=torch.load(path,map_location=device,weights_only=False)
            m=VelocityRateGRU(ckpt['normalization'],control=name=='GRU-Control').to(device)
            m.load_state_dict(ckpt['model']); m.eval(); models.append((name,seed,m))
    result=[]; diagnostics=[]; selection=[]
    metric_dir=EXP/'metrics'; metric_dir.mkdir(exist_ok=True)
    oracle_all=torch.cat([oracle(data['initial'][ix],data['target'][ix],data['dt'][ix])
                         for ix in torch.arange(len(meta),device=device).split(2048)])
    oracle_err=errors(oracle_all,data['target']).cpu().numpy(); oracle_err[...,2]*=180/np.pi
    oracle_frame=endpoint_frame(oracle_err,times,mask,meta,'Observed v/w integration',-1)
    oracle_frame[oracle_frame.horizon_ms.isin([100,200])].to_parquet(metric_dir/'observation_integration.parquet',index=False)
    oracle_frame.groupby(['horizon_ms','flight_id'])[METRICS].mean().reset_index().to_csv(EXP/'observation_per_flight.csv',index=False)
    for name,seed,model in models:
        start=time.time(); predictions=[]; losses=[]
        for ix in torch.arange(len(meta),device=device).split(config['evaluation_batch_size']):
            pred=baseline(data['initial'][ix],data['dt'][ix]) if model is None else model(batch(data,ix))
            assert torch.isfinite(pred).all()
            losses.append(window_loss(pred,data['target'][ix],data['future_mask'][ix],torch.tensor(norm['loss_scales'],device=device),torch.tensor(config['loss_weights'],device=device)))
            predictions.append(pred)
        pred=torch.cat(predictions); loss=torch.cat(losses).cpu().numpy()
        flight_loss=pd.DataFrame(dict(flight_id=meta.flight_id,loss=loss)).groupby('flight_id').loss.mean().mean()
        selection.append(dict(model=name,seed=seed,validation_flight_loss=flight_loss,validation_window_loss=loss.mean()))
        err=errors(pred,data['target']).cpu().numpy(); err[...,2]*=180/np.pi
        endpoint=endpoint_frame(err,times,mask,meta,name,seed); result.append(endpoint)
        slug=name.replace(' ','_')+f'_seed_{seed}'
        endpoint[endpoint.horizon_ms.isin([100,200])].to_parquet(metric_dir/f'{slug}_endpoints.parquet',index=False)
        i,k=np.where(mask)
        steps=pd.DataFrame(dict(window_id=meta.window_id.to_numpy()[i],flight_id=meta.flight_id.to_numpy()[i],
                                 native_step=k+1,actual_time_s=times[i,k]))
        for j,metric in enumerate(METRICS): steps[metric]=err[i,k,j]
        steps.to_parquet(metric_dir/f'{slug}_native_steps.parquet',index=False)
        dynamic=errors(pred,oracle_all).cpu().numpy(); dynamic[...,2]*=180/np.pi
        df=endpoint_frame(dynamic,times,mask,meta,name,seed)
        diagnostics.append(df.groupby(['model','seed','horizon_ms','flight_id'])[['position_m','attitude_deg']].mean().reset_index())
        examples[slug]=pred[choices].cpu().numpy()
        print(f'evaluated {name} seed={seed}: flight loss={flight_loss:.6f}, seconds={time.time()-start:.1f}',flush=True)
    pd.DataFrame(selection).to_csv(EXP/'recomputed_selection.csv',index=False)
    all_endpoints=pd.concat(result,ignore_index=True)
    summarize(all_endpoints,thresholds)
    pd.concat(diagnostics,ignore_index=True).to_csv(EXP/'dynamics_vs_oracle_per_flight.csv',index=False)
    np.savez_compressed(EXP/'trajectory_examples.npz',**examples)
    meta.iloc[choices].to_csv(EXP/'trajectory_example_windows.csv',index=False)
    write_json(EXP/'evaluation_contract.json',dict(partition='validation',windows=len(meta),flights=27,
        evaluation_horizons_ms=list(range(20,201,20)),endpoint_policy='nearest actual native timestamp; <=12ms; no state interpolation',
        actual_time_100ms_s=all_endpoints.loc[all_endpoints.horizon_ms==100,'actual_time_s'].describe().to_dict(),
        actual_time_200ms_s=all_endpoints.loc[all_endpoints.horizon_ms==200,'actual_time_s'].describe().to_dict(),
        metrics='Euclidean vector error magnitude, SO3 geodesic angle; endpoint not waveform RMSE',
        curve='same 200ms windows at every horizon; primary flight equal then seed mean',
        std='sample ddof=1; across flights after seed averaging; separate across-seed SD in summary',
        bootstrap='paired flights after seed averaging, 5000 replicates seed 20261020; exploratory, flight dependence within date remains',
        oracle='true future v/w integrated from exactly the same available initial state; observation-chain diagnostic, not model input or lower bound'))

if __name__=='__main__': main()
