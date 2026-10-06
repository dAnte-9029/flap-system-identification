"""Finite causal baselines and phase oracles; no claim of irreducible noise."""
import json
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from reference_capacity_v6 import OUT


def harmonic_basis(phase,order):
    return np.stack([np.ones_like(phase)]+[v for k in range(1,order+1) for v in [np.sin(k*phase),np.cos(k*phase)]],axis=-1)


def features(x,steps):
    past=torch.cat([x['history_state'],x['history_reference'],x['history_actuator']],-1).flatten(1)
    current=x['history_reference'][:,-1];future=x['future_reference'][:,:steps]
    return torch.cat([past,x['current_mode'],future.mean(1)-current,future[:,-1]-current],-1).numpy().astype(float)


def fit_ridge(train_x,train_y,query_x,logs,alpha=10.):
    _,codes,counts=np.unique(logs,return_inverse=True,return_counts=True);w=1/counts[codes];w=w/w.sum()
    mean=np.sum(train_x*w[:,None],0);scale=np.sqrt(np.sum((train_x-mean)**2*w[:,None],0)).clip(1e-4);ym=np.sum(train_y*w[:,None],0)
    xx=(train_x-mean)/scale;sw=np.sqrt(w*len(w));weighted=xx*sw[:,None];yy=(train_y-ym)*sw[:,None]
    coefficient=weighted.T@np.linalg.solve(weighted@weighted.T+alpha*np.eye(len(w)),yy)
    params=dict(mean=mean,scale=scale,target_mean=ym,coefficient=coefficient)
    return (query_x-mean)/scale@coefficient+ym,params


def main():
    torch.set_num_threads(1);x,y,stats=base.load();w=pd.read_csv(base.DATA/'windows_v2.csv');dest=OUT/'predictability';dest.mkdir(exist_ok=False)
    protocol=json.loads((OUT/'protocol.json').read_text());spec=protocol['predictability'];steps=spec['horizons_steps'];rows=[];projection=[]
    phasefile=base.ROOT/'docs/analysis/results/control_windows_v4/diagnostic_future_phase.npz';m=json.loads((phasefile.parent/'delivery_manifest.json').read_text());assert base.sha(phasefile)==m['artifacts'][str(phasefile.relative_to(base.ROOT))]
    with np.load(phasefile) as z:future_phase=z['phase'];np.testing.assert_array_equal(z['array_index'],w.array_index)
    tr=np.flatnonzero(w.partition.to_numpy()=='train');val=np.flatnonzero(w.partition.to_numpy()=='validation');logs=w.log_id.to_numpy();truth=y['rates_b'].numpy()
    def add(label,part,ix,k,pred):
        for n,j in enumerate(ix):
            err=(pred[n]-truth[j,k-1])**2
            rows.append(dict(model=label,partition=part,array_index=int(j),log_id=logs[j],horizon_s=.02*k,p=err[0],q=err[1],r=err[2]))
    for k in steps:
        fx=features(x,k);oof=np.zeros((len(w),3));oof.fill(np.nan)
        for li,log in enumerate(sorted(set(logs[tr]))):
            ti=tr[logs[tr]!=log];qi=tr[logs[tr]==log];pp,pars=fit_ridge(fx[ti],truth[ti,k-1],fx[qi],logs[ti],spec['linear_ridge_alpha']);oof[qi]=pp
            np.savez_compressed(dest/f'ridge_k{k}_fold{li}.npz',train_indices=ti,query_indices=qi,prediction=pp,**pars)
        add('ridge_oof','train_oof',tr,k,oof[tr]);pp,pars=fit_ridge(fx[tr],truth[tr,k-1],fx[val],logs[tr],spec['linear_ridge_alpha']);add('ridge','validation',val,k,pp)
        np.savez_compressed(dest/f'ridge_k{k}_trainfit.npz',train_indices=tr,query_indices=val,prediction=pp,**pars)
        for part,ix in [('train',tr),('validation',val)]:add('persistence',part,ix,k,x['initial_rates_b'][ix].numpy())
    for order in spec['harmonic_orders']:
        predictions=[];oracles=[]
        for j in range(len(w)):
            hs=x['history_state'][j].numpy();hp=np.arctan2(hs[:,9],hs[:,10]);design=harmonic_basis(hp,order)
            penalty=np.eye(1+2*order)*spec['harmonic_ridge'];penalty[0,0]=0
            coef=np.linalg.solve(design.T@design+penalty,design.T@hs[:,3:6])
            extrap=np.arange(1,51)*.02*2*np.pi*hs[-1,11];predictions.append(harmonic_basis(extrap,order)@coef);oracles.append(harmonic_basis(future_phase[j],order)@coef)
            # Descriptive projection fits future labels; never enters a forecast or gate.
            b=harmonic_basis(future_phase[j],order);fit=b@np.linalg.lstsq(b,truth[j],rcond=None)[0]
            for axis,name in enumerate(['p','q','r']):
                variance=np.mean((truth[j,:,axis]-truth[j,:,axis].mean())**2)
                frac=1-np.mean((truth[j,:,axis]-fit[:,axis])**2)/max(variance,1e-12)
                projection.append(dict(array_index=j,partition=w.iloc[j].partition,log_id=logs[j],axis=name,order=order,future_in_sample_fraction=float(frac)))
        for name,pp in [('past_harmonic',np.array(predictions)),('oracle_future_phase',np.array(oracles))]:
            for part,ix in [('train',tr),('validation',val)]:
                for k in steps:add(f'{name}_{order}',part,ix,k,pp[ix,k-1])
    a=pd.DataFrame(rows);a.to_csv(dest/'per_window_squared_errors.csv',index=False)
    np.sqrt(a.groupby(['model','partition','log_id','horizon_s'])[['p','q','r']].mean()).groupby(['model','partition','horizon_s']).mean().to_csv(dest/'summary.csv')
    p=pd.DataFrame(projection);p.to_csv(dest/'future_projection.csv',index=False);p.groupby(['partition','axis','order']).future_in_sample_fraction.agg(['median','mean']).to_csv(dest/'future_projection_summary.csv')
    base.write(dest/'manifest.json',dict(status='complete',script_sha256=base.sha(__file__),parent_protocol_sha256=base.sha(OUT/'protocol.json'),
        phase_source_sha256=base.sha(phasefile),learned_statistics='ridge eachfold trainonly; harmonic coefficients observed historyonly',
        oracle_and_projection='future phase or future-label regression are nondeployable diagnostics, not identified causal effects or strict achievable performance bounds',
        sealed_test_accessed=False,independent_repeated_trials_available=False,irreducible_noise_proved=False))
if __name__=='__main__':main()
