"""Reuse frozen free-run/local100ms predictions at paired endpoints."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/flap-paper-mpl')
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import torch
from run_joint_command_response import error_array,METRICS
import run_model_flight_response_diagnostic as r
m=r.m;ROOT=r.ROOT;OUT=ROOT/'docs/analysis/results/joint_command_local_reset_v1'

def endpoint_indices(offsets):return np.array(offsets)+5

def paired_rmse(a,b):
    if a.shape!=b.shape:raise ValueError('unpaired arrays')
    return np.sqrt(a.mean(0)),np.sqrt(b.mean(0))

def run():
    assert not OUT.exists();OUT.mkdir(parents=True)
    sp,batches,stats=r.source.inputs();t=batches['validation'].trajectory;n=len(t.window_ids)
    gp=ROOT/'docs/analysis/results/joint_command_response_v1/origin_groups.csv';groups=pd.read_csv(gp);np.testing.assert_array_equal(groups.window_id,t.window_ids)
    jp=m.read_json(ROOT/'docs/analysis/results/joint_command_response_v1/protocol.json');assert m.file_hash(gp)==m.read_json(ROOT/'docs/analysis/results/joint_command_response_v1/completion.json')['outputs'][m.rel(gp)]
    completion=m.read_json(r.OUT/'completion.json');pins={m.rel(p):m.file_hash(p) for p in [gp,r.ART/'local_batch.pt',r.ART/'local_origins.csv',m.ART/'prepared.pt',Path(__file__),ROOT/'tests/test_joint_command_local_reset.py']}
    for seed in [17,23,42]:
        ap=r.source.predpath(seed);lp=r.ART/f'local_seed{seed}.npz'
        assert m.file_hash(ap)==jp['pins'][m.rel(ap)]
        assert m.file_hash(lp)==completion['prediction_sha256'][m.rel(lp)]
        pins[m.rel(ap)]=m.file_hash(ap);pins[m.rel(lp)]=m.file_hash(lp)
    m.write_json(OUT/'protocol.json',dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),pins=pins,
        model='frozen H26 K50 seeds17/23/42',heldout_accessed=False,training=False,new_inference=False,
        groups='unchanged Stabilized/Mission and ALL/sustained from preceding joint-command analysis; compare same origins within each group',
        free='original actual joint-command prediction from origin t0; same fixed50step trajectory',
        local='existing independent5step forecasts reinitialized from measured state and true H26 history at offsets0,5,..45. No updates within5step forecast; not state-only reset, not stitched autonomous trajectory.',
        endpoints='offset+5 -> steps5,10,..50. Same true endpoint, control sub-tape and native dt. Primary steps25 and50.',
        metrics=METRICS,aggregation='flight RMSE -> equal flight mean -> seed mean/sample SD. Paired seed deltas local-free; flight direction after3seed mean.',
        interpretation='Extra recent observations AND shorter forecast age; cannot causally separate integrator drift, hidden-state drift, initial-state bias, phase/frequency error or input mapping. No model promotion or causal response proof.'))
    lb=torch.load(r.ART/'local_batch.pt',map_location='cpu',weights_only=False);lo=pd.read_csv(r.ART/'local_origins.csv')
    np.testing.assert_array_equal(lo.window_id,lb.trajectory.window_ids)
    for block,offset in enumerate(r.OFFSETS):
        sl=slice(block*n,(block+1)*n)
        np.testing.assert_array_equal(lo.parent_window_id.iloc[sl],t.window_ids)
        assert (lo.offset.iloc[sl]==offset).all()
        np.testing.assert_array_equal(lb.trajectory.controls[sl],t.controls[:,offset:offset+5]);np.testing.assert_array_equal(lb.trajectory.dt_s[sl],t.dt_s[:,offset:offset+5])
        for k in vars(t.truth):np.testing.assert_array_equal(getattr(lb.trajectory.truth,k)[sl],getattr(t.truth,k)[:,offset:offset+6])
    rows=[]
    for seed in [17,23,42]:
        with np.load(r.source.predpath(seed),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['window_ids'],t.window_ids);ap={k:z[k] for k in vars(t.truth)}
        with np.load(r.ART/f'local_seed{seed}.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(z['window_ids'],lb.trajectory.window_ids);lp={k:z[k] for k in vars(t.truth)}
        for k in ap:np.testing.assert_allclose(lp[k][:n],ap[k][:,:6],rtol=1e-6,atol=2e-5)
        ae=error_array(ap,t);le=error_array(lp,lb.trajectory).reshape(10,n,5,len(METRICS))
        assert np.isfinite(ae).all() and np.isfinite(le).all()
        for mode in ['Stabilized','Mission']:
            for pattern in ['ALL','sustained']:
                mask=(groups['mode']==mode).to_numpy()&((groups.command_pattern=='sustained').to_numpy() if pattern=='sustained' else np.ones(n,bool))
                for block,step in enumerate(endpoint_indices(r.OFFSETS)):
                    for flight in sorted(set(t.log_ids[mask])):
                        take=mask&(t.log_ids==flight);a,b=paired_rmse(ae[take,step-1],le[block,take,4])
                        for condition,values in [('continuous',a),('local100ms',b)]:
                            for metric,value in zip(METRICS,values):rows.append(dict(seed=seed,mode=mode,pattern=pattern,step=int(step),flight_id=flight,cohort=r.cohort(flight),n_origins=int(take.sum()),condition=condition,metric=metric,value=value,
                                endpoint_elapsed_median_s=float(np.median(t.dt_s[take,:step].sum(1))),local_age_median_s=float(np.median(t.dt_s[take,step-5:step].sum(1)))))
    f=pd.DataFrame(rows);f.to_csv(OUT/'per_flight.csv',index=False)
    ex=pd.concat([f,f.assign(cohort='ALL')]);keys=['mode','pattern','step','cohort','condition','metric']
    per=ex.groupby(keys+['seed']).value.mean().reset_index();per.to_csv(OUT/'per_seed.csv',index=False)
    agg=per.groupby(keys).value.agg(['mean','std']).reset_index();agg.to_csv(OUT/'summary.csv',index=False)
    paired=per.pivot(index=['mode','pattern','step','cohort','seed','metric'],columns='condition',values='value').reset_index()
    paired['local_minus_continuous']=paired.local100ms-paired.continuous;paired.to_csv(OUT/'paired_seeds.csv',index=False)
    fp=f.groupby(['mode','pattern','step','flight_id','cohort','condition','metric']).value.mean().reset_index().pivot(index=['mode','pattern','step','flight_id','cohort','metric'],columns='condition',values='value').reset_index()
    fp['local_minus_continuous']=fp.local100ms-fp.continuous;fp.to_csv(OUT/'paired_flights.csv',index=False)
    m.verify_pins(pins);m.write_json(OUT/'checks.json',dict(status='passed',all_frozen_groups_unchanged=True,local_parent_identity_labels_dt_controls_exact=True,offset0_prefix_parity=True,predictions_finite=True,heldout_accessed=False))

if __name__=='__main__':run()
