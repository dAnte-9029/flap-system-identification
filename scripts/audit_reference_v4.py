"""Frozen-data contract, waveform and small-cohort capacity diagnostics."""
import json
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation
from pyulog import ULog
import train_control_windows_v2 as base
from train_reference_unit_gravity import ART,model_from
from control_windows_v3 import predict
from system_identification.data.reference_windows import extract_window
from system_identification.models.reference_gru_unit_gravity import ReferenceGRUUnitGravity,unit_state_features
from evaluate_control_windows_v2 import error_rows,summarize
from audit_control_reference_semantics import past_values
OUT=base.ROOT/'docs/analysis/results/control_windows_v4'

def basis(phi):
    return np.stack([np.ones_like(phi),np.sin(phi),np.cos(phi),np.sin(2*phi),np.cos(2*phi)],-1)

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    OUT.mkdir(exist_ok=False)
    x,y,stats=base.load(); w=pd.read_csv(base.DATA/'windows_v2.csv')
    train=w[w.partition=='train']; small=train.iloc[np.linspace(0,len(train)-1,8,dtype=int)]
    protocol=dict(stage='contract and training-capacity audit',source_manifest_sha256=base.sha(base.DATA/'manifest.json'),
        script_sha256=base.sha(__file__),sealed_test_accessed=False,partitions=['train','validation'],
        validation_status='repeatedly viewed development evaluation, not blind test',
        tiny_cohort=small.window_id.tolist(),tiny_seed=17,tiny_updates=1600,tiny_batch='all8',
        tiny_optimizer='AdamW3e-4, weight_decay1e-4, clip1; unchanged loss/normalization',
        tiny_purpose='memorization diagnostic only; no candidate selection from validation',
        waveform='phase harmonic1+2 regression of centered residual; future phase allowed only in offline attribution, not operational inputs',
        conditional_next_experiment='if train median harmonic fraction of p residual exceeds0.5, compare causal history-fitted p harmonic prior with A1600, same407windows/3seeds/1600updates; otherwise stop to report diagnosis',
        primary_horizons_s=[.5,1.],long_rollout='stress only, not primary acceptance',
        candidate_acceptance='>=5percent mean improvement on active p/q at0.5/1s across flights; all3seeds improve; no metric >5percent worse; absolute0.5s active p/q<=.15rad/s attitude<=3deg velocity<=.5m/s eachseed',
        response='nominal waveform amplitude/time and reference ablation; neither supplies measured counterfactual truth; no optimizer unless absolute and response evidence passes')
    base.write(OUT/'protocol.json',protocol)
    cm=json.loads((base.ROOT/'docs/analysis/results/reference_dynamics_data_v1/training_contract/manifest.json').read_text())
    parent=json.loads((base.ROOT/'docs/analysis/results/reference_dynamics_data_v1/training_ready/manifest.json').read_text())
    rawmanifest=json.loads((base.ROOT/parent['parent_manifest']).read_text())
    checks=[]; ages=[]; native=[]; phase=np.zeros((len(w),50)); pins={}
    for cache,group in w.groupby('cache'):
        assert base.sha(base.ROOT/cache)==cm['cache_sha256'][cache]
        with np.load(base.ROOT/cache) as z:a={k:z[k] for k in z.files}
        selected=np.zeros(len(a['timestamp_us']),bool)
        maximum=0.
        for row in group.itertuples():
            i=int(row.origin_index); ix=int(row.array_index)
            xx,yy=extract_window(a,i)
            for k in x:maximum=max(maximum,float(np.max(np.abs(xx[k]-x[k][ix].numpy()))))
            for k in y:maximum=max(maximum,float(np.max(np.abs(yy[k]-y[k][ix].numpy()))))
            selected[i-26:i+51]=True
            phase[ix]=a['phase_rad'][i+1:i+51]-a['phase_rad'][i]
        assert maximum<1e-5
        q=a['quaternion'][selected]; rot=Rotation.from_quat(q[:,[1,2,3,0]])
        s=a['state_features'][selected]
        gravity=np.max(np.abs(rot.inv().apply(np.tile([0,0,1],(len(q),1)))-s[:,6:9]))
        vel=np.max(np.abs(rot.inv().apply(a['velocity_n'][selected])-s[:,:3]))
        assert gravity<1e-5 and vel<1e-4
        checks.append(dict(log_id=group.log_id.iloc[0],windows=len(group),tensor_max_abs=maximum,gravity_max_abs=gravity,body_velocity_max_abs=vel))
        for j,name in enumerate(a['age_topic_names']):
            vals=a['age_s'][selected,j]; finite=vals[np.isfinite(vals)]
            ages.append(dict(log_id=group.log_id.iloc[0],topic=str(name),p50_s=float(np.median(finite)),p95_s=float(np.quantile(finite,.95)),max_s=float(finite.max()),nonfinite=int((~np.isfinite(vals)).sum())))
        source=rawmanifest['sources'][group.log_id.iloc[0]];assert base.sha(source['path'])==source['sha256'];pins[source['path']]=source['sha256']
        topics=['vehicle_attitude','vehicle_angular_velocity','vehicle_attitude_setpoint','wing_phase']
        u=ULog(source['path'],message_name_filter_list=topics)
        for topic in [d for d in u.data_list if d.multi_id==0]:
            d=topic.data;t=d['timestamp'].astype(np.int64); ts=np.asarray(d.get('timestamp_sample',t),np.int64)
            _,pa,sa=past_values(d,['timestamp'],a['timestamp_us'][selected]); assert (pa>=0).all() and (sa>=0).all()
            # Select native messages inside the selected full-support windows, never bridge gaps.
            native_mask=np.zeros(len(t),bool)
            for row in group.itertuples():native_mask|=(t>=row.start_timestamp_us)&(t<=row.end_timestamp_us)
            adjacent=native_mask[:-1]&native_mask[1:]
            intervals=np.diff(t)[adjacent]*1e-6;latency=(t-ts)[native_mask]*1e-6
            native.append(dict(log_id=group.log_id.iloc[0],topic=topic.name,has_sample_timestamp='timestamp_sample' in d,
                publication_interval_p50_s=float(np.median(intervals)),publication_interval_p95_s=float(np.quantile(intervals,.95)),
                publication_minus_sample_p50_s=float(np.median(latency)),publication_minus_sample_p95_s=float(np.quantile(latency,.95)),
                sample_age_p95_s=float(np.quantile(sa,.95)),publication_age_p95_s=float(np.quantile(pa,.95))))
    h=x['history_state'][:,-1]; zero=torch.zeros(len(h))
    ff=unit_state_features(x['initial_velocity_n'],x['initial_quaternion'],x['initial_rates_b'],zero,zero,h[:,11])
    assert (ff-h[:,:12]).abs().max()<1e-5
    pd.DataFrame(checks).to_csv(OUT/'contract_checks.csv',index=False);pd.DataFrame(ages).to_csv(OUT/'cache_ages.csv',index=False);pd.DataFrame(native).to_csv(OUT/'native_timing.csv',index=False)
    base.write(OUT/'audit.json',dict(status='passed',source_pins=pins,windows=len(w),history_steps=26,dt_s=.02,
        caveat='native publication/sample times are logging timing, not physical actuator delay; no physical phase-zero certification'))
    np.savez_compressed(OUT/'diagnostic_future_phase.npz',phase=phase,array_index=w.array_index.to_numpy())
    model=model_from(ART/'A_long_17/final.pt'); pred=predict(model,x,w.array_index.to_numpy());rows=[];harmonic=[]
    for row in w.itertuples():
        j=row.array_index;err=pred['rates_b'][j]-y['rates_b'][j].numpy(); design=basis(phase[j]); centered=err-err.mean(0)
        fit=design@np.linalg.lstsq(design,err,rcond=None)[0]; remaining=err-fit
        hp=x['history_state'][j,:,9:11].numpy(); histphi=np.arctan2(hp[:,0],hp[:,1]);hd=basis(histphi)
        coef=np.linalg.solve(hd.T@hd+np.diag([0,.001,.001,.001,.001]),hd.T@x['history_state'][j,:,3:6].numpy())
        future=2*np.pi*float(h[j,11])*np.arange(1,51)*.02
        for ax,name in enumerate(['p','q','r']):
            frac=1-np.mean(remaining[:,ax]**2)/max(np.mean(centered[:,ax]**2),1e-12)
            rows.append(dict(array_index=j,log_id=row.log_id,partition=row.partition,axis=name,harmonic_fraction=float(frac)))
        for label,phi in [('causal_held_frequency',future),('oracle_logged_phase_diagnostic',phase[j])]:
            rate=basis(phi)@coef;rate+=x['initial_rates_b'][j].numpy()-basis(np.array(0.))@coef
            for step in [25,50]:
                er=(rate[step-1]-y['rates_b'][j,step-1].numpy())**2
                harmonic.append(dict(array_index=j,log_id=row.log_id,partition=row.partition,model=label,horizon_s=step*.02,p=er[0],q=er[1],r=er[2]))
    df=pd.DataFrame(rows);df.to_csv(OUT/'waveform_projection.csv',index=False)
    trainfrac=float(df[(df.partition=='train')&(df.axis=='p')].harmonic_fraction.median())
    base.write(OUT/'candidate_trigger.json',dict(train_p_median_harmonic_fraction=trainfrac,trigger_passed=trainfrac>.5,attribution='in-sample descriptive phase projection, not causal explained variance or achievable forecast accuracy'))
    hh=pd.DataFrame(harmonic);hh.to_csv(OUT/'harmonic_forecast_squared_errors.csv',index=False)
    np.sqrt(hh.groupby(['model','partition','horizon_s','log_id'])[['p','q','r']].mean()).groupby(['model','partition','horizon_s']).mean().to_csv(OUT/'harmonic_forecast_summary.csv')
    # Tiny fitting check: train only, preselected by chronological index, no response selection.
    ix=small.array_index.to_numpy();torch.manual_seed(17);tiny=ReferenceGRUUnitGravity(stats)
    optimizer=torch.optim.AdamW(tiny.parameters(),lr=3e-4,weight_decay=1e-4);curve=[]
    xx={k:v[ix] for k,v in x.items()}; yy={k:v[ix] for k,v in y.items()}
    for step in range(1,1601):
        optimizer.zero_grad(set_to_none=True);loss=base.loss_fn(tiny(**xx),yy,tiny);assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_(tiny.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if step==1 or step%100==0:
            curve.append(dict(step=step,loss=float(loss.detach())));print(json.dumps(curve[-1]),flush=True)
    torch.save(dict(state_dict=tiny.state_dict(),feature_contract='unit_gravity_v1',protocol_sha256=base.sha(OUT/'protocol.json')),OUT/'tiny_fit.pt')
    pd.DataFrame(curve).to_csv(OUT/'tiny_training_curve.csv',index=False)
    frames=[]
    for label,mod in [('A1600_seed17',model),('tiny8_seed17',tiny.eval())]:
        for subset,meta in [('fit8',small),('other_train',train[~train.array_index.isin(ix)])]:
            jj=meta.array_index.to_numpy();pp=predict(mod,x,jj);truth={k:v[jj].numpy() for k,v in y.items()}
            frame=summarize(error_rows(pp,truth,meta,label,17));frame['cohort']=subset;frames.append(frame)
    pd.concat(frames).to_csv(OUT/'tiny_per_flight.csv',index=False)
    base.write(OUT/'audit_status.json',dict(status='complete',trigger=trainfrac>.5,sealed_test_accessed=False))

if __name__=='__main__':main()
