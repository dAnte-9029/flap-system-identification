"""Frozen H26 actual-command response by flight mode and joint-command pattern."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/flap-paper-mpl')
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import run_model_flight_response_diagnostic as r
from system_identification.evaluation.future_control_diagnostic import squared_errors
from types import SimpleNamespace
m=r.m;ROOT=r.ROOT;OUT=ROOT/'docs/analysis/results/joint_command_response_v1'
METRICS=['position_m','velocity_m_s','attitude_deg','body_rate_rad_s','p_rad_s','q_rad_s','r_rad_s','roll_deg']

def patterns(u,scale,q75):
    coord=r.coordinates(u);departure=coord-coord[:,:1]
    a=np.sqrt(np.mean((departure[:,:25]/scale)**2,axis=1));active=a>q75
    first=departure[:,:25].mean(1);late=departure[:,25:];sign=np.sign(first)
    forward=(np.mean(late*sign[:,None]>0,axis=1)>=.8)&(sign*late.mean(1)>=.5*np.abs(first))
    reverse=(np.mean(late*sign[:,None]<0,axis=1)>=.8)&(-sign*late.mean(1)>=.5*np.abs(first))
    sustained=active.any(1)&np.all(~active|forward,axis=1)
    reversing=np.any(active&reverse,axis=1)
    label=np.where(~active.any(1),'lower_change',np.where(sustained,'sustained',np.where(reversing,'reversing','mixed_change')))
    return label,active,a

def roll(q):
    w,x,y,z=np.moveaxis(q,-1,0);return np.rad2deg(np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)))

def error_array(pred,t):
    base=squared_errors(SimpleNamespace(**pred),t.truth)
    rate=(pred['angular_velocity_b'][:,1:]-t.truth.angular_velocity_b[:,1:])**2
    diff=(roll(pred['quaternion_nb'][:,1:])-roll(t.truth.quaternion_nb[:,1:])+180)%360-180
    return np.concatenate([base,rate,diff[...,None]**2],axis=-1)

def flight_macro(values,ids,mask):
    return np.mean([np.sqrt(values[mask&(ids==f)].mean(0)) for f in sorted(set(ids[mask]))],axis=0)

def run():
    assert not (OUT/'protocol.json').exists();OUT.mkdir(parents=True,exist_ok=True)
    sp,batches,stats=r.source.inputs();t=batches['validation'].trajectory
    origin=pd.read_csv(m.BASE/'validation_origins.csv');np.testing.assert_array_equal(origin.window_id,t.window_ids)
    manifest=ROOT/sp['manifest_path'];data=manifest.parent
    samples=data/'samples_validation.parquet';assert m.file_hash(samples)==sp['artifact_sha256'][samples.name]
    f=pd.read_parquet(samples);frames={key:g.reset_index(drop=True) for key,g in f.groupby(['log_id','segment_id'])}
    modes=[]
    for _,w in origin.iterrows():
        seg=frames[(w.log_id,w.segment_id)];start=np.flatnonzero(seg.sample_in_segment.to_numpy()==w.start_sample_in_segment);assert len(start)==1
        s=seg.iloc[start[0]:start[0]+51];assert len(s)==51
        np.testing.assert_allclose(np.diff(s.timestamp_us)*1e-6,t.dt_s[len(modes)],rtol=0,atol=1e-12)
        v=s.nav_state.to_numpy();modes.append('Stabilized' if (v==15).all() else 'Mission' if (v==3).all() else 'Other_or_transition')
    mode=np.array(modes);thpath=ROOT/'docs/analysis/results/response_timescale_diagnostic_v1/channel_thresholds.csv';th=pd.read_csv(thpath)
    labels,active,activity=patterns(t.controls,th.scale.to_numpy(),th.train_q75.to_numpy())
    origin['mode']=mode;origin['command_pattern']=labels;origin['n_high_channels']=active.sum(1)
    for j,ch in enumerate(r.CHANNELS):origin[ch+'_activity']=activity[:,j];origin[ch+'_high']=active[:,j]
    origin.to_csv(OUT/'origin_groups.csv',index=False)
    pins={m.rel(p):m.file_hash(p) for p in [manifest,samples,m.ART/'prepared.pt',thpath,m.BASE/'validation_origins.csv',Path(__file__),ROOT/'tests/test_joint_command_response.py']}
    old=m.read_json(r.OUT/'protocol.json')['frozen_input_sha256']
    for seed in [17,23,42]:
        p=r.source.predpath(seed);h=m.file_hash(p);assert h==old[m.rel(p)];pins[m.rel(p)]=h
    cases=[]
    for name in ['Stabilized','Mission']:
        for kind in ['sustained','reversing','mixed_change']:
            z=origin[(mode==name)&(labels==kind)].sort_values('window_id')
            if len(z):cases.append(dict(mode=name,pattern=kind,window_id=z.iloc[len(z)//2].window_id,seed=17))
    m.write_json(OUT/'representative_selection.json',cases)
    m.write_json(OUT/'protocol.json',dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),pins=pins,
        model='StandardGRU64/H26/K50 seeds17/23/42; frozen existing actual-command predictions',origins=2582,heldout_accessed=False,new_inference=False,training=False,
        modes='all51 ground truth state rows nav_state15=Stabilized;3=Auto Mission; remainder retained separately and in ALL_original. Not RC/mode flags freshness filtered.',
        control='model actual motor/left/right/rudder normalized postallocation commands; common=(L+R)/2,differential=(L-R)/2 only analysis coordinates; no physical-deflection claim',
        patterns='First25 RMS departure/previous frozen scales >trainq75 per coordinate defines high channels. lower_change=no high channel. sustained=all high channels maintain sign in >=80% late25 and signed late mean>=half absolute earlymean. reversing=any high channel opposite >=80% and magnitude>=half. remainder=mixed_change. Other channels unrestricted; coactivation explicitly counted. Not step/trim/steady-state identification.',
        phase='groups use whole command tape offline; errors at states5/10/25/50, dtweighted1..25 and26..50. Origin is not necessarily action onset; reversing not necessarily release to trim.',
        metrics='physical endpoint and dtweighted interval RMS; flight RMSE -> equal flights ->3seed mean/sampleSD. Quaternion geodesic main attitude, wrapped Euler roll secondary. Axis waveform amplitude=RMS(change from same true t0), macro ratio not averaged window percentages.',
        cases='identity median per mode x pattern seed17; selected before prediction access',
        claims='Logged joint-input response fidelity, not causal intervention/control readiness; no pattern ranking threshold or selected bestseed.'))
    groups={}
    for name in ['ALL_original','Stabilized','Mission','Other_or_transition']:
        mm=np.ones(len(mode),bool) if name=='ALL_original' else mode==name
        for kind in ['ALL','lower_change','sustained','reversing','mixed_change','multi_high']:
            mask=mm&(np.ones(len(mode),bool) if kind=='ALL' else active.sum(1)>=2 if kind=='multi_high' else labels==kind)
            if mask.any():groups[(name,kind)]=mask
    coverage=[]
    for (name,kind),mask in groups.items():
        for flight in sorted(set(t.log_ids[mask])):coverage.append(dict(mode=name,pattern=kind,flight_id=flight,cohort=r.cohort(flight),n_origins=int((mask&(t.log_ids==flight)).sum())))
    pd.DataFrame(coverage).to_csv(OUT/'coverage_per_flight.csv',index=False)
    rows=[];evolution=[];amps=[]
    for seed in [17,23,42]:
        with np.load(r.source.predpath(seed),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['window_ids'],t.window_ids)
            pred={k:z[k].astype(float) for k in ['position_n','velocity_n','quaternion_nb','angular_velocity_b','relative_phase_rad','flap_frequency_hz']}
        assert all(np.isfinite(v).all() for v in pred.values())
        assert np.max(np.abs(np.linalg.norm(pred['quaternion_nb'],axis=-1)-1))<1e-5
        err=error_array(pred,t)
        for (name,kind),mask in groups.items():
            for horizon,k0,k1 in [('100ms',5,5),('200ms',10,10),('500ms',25,25),('1s',50,50),('interval_0_500ms',1,25),('interval_500_1000ms',26,50)]:
                value=err[:,k1-1] if k0==k1 else np.sum(err[:,k0-1:k1]*t.dt_s[:,k0-1:k1,None],axis=1)/t.dt_s[:,k0-1:k1].sum(1)[:,None]
                for flight in sorted(set(t.log_ids[mask])):
                    take=mask&(t.log_ids==flight);v=np.sqrt(value[take].mean(0))
                    for metric,num in zip(METRICS,v):rows.append(dict(seed=seed,mode=name,pattern=kind,horizon=horizon,flight_id=flight,cohort=r.cohort(flight),n_origins=int(take.sum()),metric=metric,value=num))
            ev=flight_macro(err,t.log_ids,mask);elapsed=np.cumsum(t.dt_s,axis=1)
            for k in range(50):
                for j,metric in enumerate(METRICS):evolution.append(dict(seed=seed,mode=name,pattern=kind,step=k+1,elapsed_median_s=np.median(elapsed[mask,k]),elapsed_p05_s=np.quantile(elapsed[mask,k],.05),elapsed_p95_s=np.quantile(elapsed[mask,k],.95),metric=metric,value=ev[k,j]))
            for label,lo,hi in [('early',1,5),('late500',16,25),('late1s',41,50)]:
                w=t.dt_s[:,lo-1:hi];true=t.truth.angular_velocity_b[:,lo:hi+1]-t.truth.angular_velocity_b[:,:1];pr=pred['angular_velocity_b'][:,lo:hi+1]-t.truth.angular_velocity_b[:,:1]
                tm=np.sum(true**2*w[:,:,None],1)/w.sum(1)[:,None];pm=np.sum(pr**2*w[:,:,None],1)/w.sum(1)[:,None]
                tv=flight_macro(tm,t.log_ids,mask);pv=flight_macro(pm,t.log_ids,mask)
                for j,axis in enumerate('pqr'):amps.append(dict(seed=seed,mode=name,pattern=kind,interval=label,axis=axis,true_amplitude=tv[j],pred_amplitude=pv[j]))
        if seed==17:
            export=[]
            for case in cases:
                i=int(np.flatnonzero(t.window_ids==case['window_id'])[0]);tt=np.r_[0,np.cumsum(t.dt_s[i])];coord=r.coordinates(t.controls[i])
                for k in range(51):
                    row=dict(**case,step=k,elapsed_s=tt[k],roll_true=roll(t.truth.quaternion_nb[i,k]),roll_pred=roll(pred['quaternion_nb'][i,k]))
                    for j,axis in enumerate('pqr'):row[axis+'_true']=t.truth.angular_velocity_b[i,k,j];row[axis+'_pred']=pred['angular_velocity_b'][i,k,j]
                    for j,axis in enumerate(['vx','vy','vz']):row[axis+'_true']=t.truth.velocity_n[i,k,j];row[axis+'_pred']=pred['velocity_n'][i,k,j]
                    if k<50:
                        for j,ch in enumerate(r.CHANNELS):row[ch]=coord[k,j]
                    export.append(row)
            pd.DataFrame(export).to_csv(OUT/'representative_traces.csv',index=False)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'per_flight.csv',index=False)
    ex=pd.concat([frame,frame.assign(cohort='ALL')]);keys=['mode','pattern','horizon','cohort','metric']
    per=ex.groupby(keys+['seed']).value.mean().reset_index();per.to_csv(OUT/'per_seed.csv',index=False)
    per.groupby(keys).value.agg(['mean','std']).reset_index().to_csv(OUT/'summary.csv',index=False)
    ev=pd.DataFrame(evolution);ev.to_csv(OUT/'error_evolution_per_seed.csv',index=False);ev.groupby(['mode','pattern','step','metric']).agg(mean=('value','mean'),std=('value','std'),elapsed_median_s=('elapsed_median_s','first'),elapsed_p05_s=('elapsed_p05_s','first'),elapsed_p95_s=('elapsed_p95_s','first')).reset_index().to_csv(OUT/'error_evolution.csv',index=False)
    a=pd.DataFrame(amps);a.to_csv(OUT/'amplitude_per_seed.csv',index=False);a=a.groupby(['mode','pattern','interval','axis'])[['true_amplitude','pred_amplitude']].mean().reset_index();a['ratio']=a.pred_amplitude/a.true_amplitude.replace(0,np.nan);a.to_csv(OUT/'amplitude_summary.csv',index=False)
    # Reproduce all-original metric definitions exactly, independently of mode grouping.
    oldscores=pd.read_csv(ROOT/'docs/analysis/results/observation_scoring_sensitivity_v1/per_seed.csv').query('condition=="original" and group=="ALL" and cohort=="ALL"')
    mapping={'body_rate_rad_s':'rate_rad_s','attitude_deg':'attitude_deg','p_rad_s':'p_rad_s','q_rad_s':'q_rad_s','r_rad_s':'r_rad_s'}
    new=per.query('mode=="ALL_original" and pattern=="ALL" and cohort=="ALL"').copy();new['metric']=new.metric.map(mapping)
    new['horizon']=new.horizon.replace({'interval_0_500ms':'0_500ms','interval_500_1000ms':'500_1000ms'})
    compare=new.merge(oldscores,on=['seed','metric','horizon']);assert len(compare)==90
    residual=float(np.max(np.abs(compare.value_x-compare.value_y)));assert residual<1e-6
    m.verify_pins(pins);m.write_json(OUT/'checks.json',dict(all_origins_retained=True,heldout_accessed=False,new_predictions=0,parity_cells=len(compare),parity_max_abs=residual,source_hashes_unchanged=True))

if __name__=='__main__':run()
