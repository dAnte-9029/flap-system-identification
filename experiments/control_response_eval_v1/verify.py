"""Independent raw-target, quaternion and flight-aggregation recomputation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from evaluate import HERE,OUT,ROOT,STATE,TARGET,file_hash,write_json

def mul(a,b):
    out=np.empty_like(a,dtype=float)
    out[...,0]=a[...,0]*b[...,0]-np.sum(a[...,1:]*b[...,1:],axis=-1)
    out[...,1:]=a[...,:1]*b[...,1:]+b[...,:1]*a[...,1:]+np.cross(a[...,1:],b[...,1:])
    return out

def unit(q):return q/np.linalg.norm(q,axis=-1,keepdims=True)
def conj(q):return q*np.array([1,-1,-1,-1])
def angle(q):return 2*np.arctan2(np.linalg.norm(q[...,1:],axis=-1),np.abs(q[...,0]))*180/np.pi

def main():
    cfg=json.loads((HERE/'config.json').read_text());rm=json.loads((OUT/'run_manifest.json').read_text())
    for name,h in rm['checkpoints'].items():assert file_hash(ROOT/name)==h
    for name,h in rm['data']['checked_artifact_sha256'].items():
        path=Path(name) if name.startswith('/') else ROOT/'dataset/velocity_rate_multistep_v1'/name
        assert file_hash(path)==h
    cohorts=pd.read_parquet(OUT/'cohorts.parquet');metrics=pd.read_parquet(OUT/'response_metrics.parquet')
    summary=pd.read_csv(OUT/'response_summary.csv');source_checks=0;scalar_checks=0;worst=0.;quatworst=0.
    for horizon in [500,1000]:
        z=np.load(OUT/f'predictions_{horizon}ms.npz');meta=pd.read_parquet(OUT/f'windows_{horizon}ms.parquet');a,b=z['i'],z['j'];P=len(a)
        for split in ['train','validation']:
            needed=meta[meta.split==split]
            raw=pd.read_parquet(ROOT/f'dataset/velocity_rate_multistep_v1/samples_{split}.parquet',columns=['flight_id','state_measurement_us']+STATE+TARGET)
            raw={f:g.reset_index(drop=True) for f,g in raw[raw.flight_id.isin(needed.flight_id)].groupby('flight_id')}
            for i,m in needed.iterrows():
                s=raw[m.flight_id];o=int(m.origin_sample);e=int(m.end_sample);p0=s.loc[o,STATE[:3]].to_numpy(float)
                truth=s.loc[o+1:e,TARGET].to_numpy(float);truth[:,:3]-=p0
                np.testing.assert_allclose(z['native_truth'][i,:len(truth)],truth,atol=0,rtol=0)
                t0=s.loc[o,TARGET].to_numpy(float);t0[:3]-=p0
                np.testing.assert_allclose(z['truth_initial'][i],t0,atol=0,rtol=0);source_checks+=1
        for model in ['GRU-Control','GRU-NoControl']:
            for seed in [17,29,43]:
                pred=z[f'{model}_{seed}'].astype(float);truth=z['truth'];initial=z['initial'].astype(float);truth0=z['truth_initial']
                for metric,cols in [('velocity',slice(3,6)),('rate',slice(10,13)),('position',slice(0,3)),('attitude',None)]:
                    if cols is not None:
                        tp=(truth[a,:,cols]-truth0[a,None,cols])-(truth[b,:,cols]-truth0[b,None,cols])
                        pp=(pred[a,:,cols]-initial[a,None,cols])-(pred[b,:,cols]-initial[b,None,cols])
                        errors=np.sqrt(np.sum((tp-pp)**2,axis=-1));amplitude=np.sqrt(np.sum(tp**2,axis=-1))
                    else:
                        tq=unit(truth[:,:,6:10]);pq=unit(pred[:,:,6:10]);t0=unit(truth0[:,None,6:10]);p0=unit(initial[:,None,6:10])
                        ti=mul(tq,conj(t0));pi=mul(pq,conj(p0));tc=mul(ti[a],conj(ti[b]));pc=mul(pi[a],conj(pi[b]))
                        errors=angle(mul(conj(pc),tc));amplitude=angle(tc)
                    reported=metrics[(metrics.horizon_ms==horizon)&(metrics.model==model)&(metrics.seed==seed)&(metrics.metric==metric)].drop_duplicates(['pair_uid','point_ms']).set_index(['pair_uid','point_ms'])
                    for i,idx in enumerate(z['pair_index']):
                        for j,pt in enumerate(z['point_ms']):
                            r=reported.loc[(f'{horizon}:{idx}',pt)]
                            delta=max(abs(r.error-errors[i,j]),abs(r.true_amplitude-amplitude[i,j]))
                            if metric=='attitude':quatworst=max(quatworst,delta)
                            else:worst=max(worst,delta)
                            assert delta<1e-8,(metric,delta);scalar_checks+=1
    # Independent endpoint-wise flight mean, not the analysis script's edge-weight formula.
    max_aggregate=0.
    for key,g in metrics.groupby(['horizon_ms','point_ms','cohort','split','model','metric','seed']):
        incident={}
        for r in g.itertuples():
            for f in [r.flight_a,r.flight_b]:incident.setdefault(f,[]).append(r.squared_error)
        value=np.sqrt(np.mean([np.mean(v) for v in incident.values()]))
        selected=summary
        for k,v in zip(['horizon_ms','point_ms','cohort','split','model','metric','seed'],key):selected=selected[selected[k]==(str(v) if k=='seed' else v)]
        assert len(selected)==1
        delta=abs(selected.response_rmse.iloc[0]-value);max_aggregate=max(max_aggregate,delta);assert delta<1e-12
    # Every original control stream equals its common-grid refinement as a function of time.
    refinement_checks=0
    for horizon in [500,1000]:
        z=np.load(OUT/f'predictions_{horizon}ms.npz')
        for n,recipient in enumerate(z['recipient']):
            for k,dt in enumerate(z['dt'][recipient]):
                if dt==0:continue
                old=z['events'][recipient,k,:z['event_length'][recipient,k]]
                new=z['common_events'][n,k,:z['common_event_length'][n,k]]
                end=np.cumsum(old[:,4],dtype=float);starts=np.r_[0,np.cumsum(new[:,4],dtype=float)[:-1]]
                middle=starts+new[:,4]/2;indices=np.searchsorted(end,middle).clip(0,len(old)-1)
                np.testing.assert_allclose(new[:,:4],old[indices,:4],atol=0,rtol=0)
                np.testing.assert_allclose(new[:,4].sum(),dt,atol=1e-7);refinement_checks+=1
    write_json(OUT/'independent_verification.json',dict(raw_target_windows=source_checks,response_scalar_values=scalar_checks,
        max_vector_difference=worst,max_quaternion_difference_deg=quatworst,max_flight_aggregate_difference=max_aggregate,
        equivalent_control_intervals=refinement_checks,checkpoint_and_dataset_hashes_rechecked=True,
        code_sha256=file_hash(Path(__file__))))
    print('Independent verification passed:',source_checks,'raw windows;',scalar_checks,'response values;',refinement_checks,'control intervals')

if __name__=='__main__':main()
