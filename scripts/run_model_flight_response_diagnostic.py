"""Frozen H26/K50 logged-command response matching; development data only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json,copy,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import run_paper_rollout_horizon_ablation as source
from system_identification.training.trajectory_main_v1 import assemble_history_trajectory_windows,predict_history_trajectory_model
from system_identification.evaluation.future_control_diagnostic import excitation,assign_groups,squared_errors
from system_identification.evaluation.paper_baselines import cohort
m=source.m;ROOT=m.ROOT
OUT=ROOT/'docs/analysis/results/model_flight_response_diagnostic_v1'
ART=ROOT/'artifacts/model_flight_response_diagnostic_v1'
OFFSETS=tuple(range(0,50,5));AXES=('vx','vy','vz','p','q','r')
CHANNELS=('drive','common','differential','rudder')

def coordinates(u):
    return np.stack([u[...,0],(u[...,1]+u[...,2])/2,(u[...,1]-u[...,2])/2,u[...,3]],-1)

def state_axes(truth):return np.concatenate([truth.velocity_n,truth.angular_velocity_b],axis=-1)

def activity(u,std):
    scale=np.array([std[0],np.hypot(std[1],std[2])/2,np.hypot(std[1],std[2])/2,std[3]])
    c=coordinates(u)
    return np.sqrt(np.mean(((c[:,:25]-c[:,:1])/scale)**2,axis=1)),scale

def weighted_mean(x,dt):return np.sum(x*dt[...,None],axis=1)/dt.sum(1)[:,None]

def local_windows(origins):
    frames=[]
    for offset in OFFSETS:
        w=origins.copy();w['parent_window_id']=w.window_id;w['offset']=offset
        w['window_id']=w.window_id+f':local{offset}'
        w['start_sample_in_segment']+=offset;w['state_sample_count']=6
        # Timestamp columns describe the old full window and must not masquerade as local metadata.
        w=w.drop(columns=[c for c in w.columns if 'timestamp' in c or c=='history_span_s'])
        frames.append(w)
    return pd.concat(frames,ignore_index=True)

def prepare():
    assert not (OUT/'protocol.json').exists()
    OUT.mkdir(parents=True);ART.mkdir(parents=True)
    p,batches,stats=source.inputs();train=batches['train'];val=batches['validation']
    prior=m.read_json(ROOT/'docs/analysis/results/paper_rollout_horizon_ablation_v1/protocol.json')
    # Reuse only its explicit K50/train/validation identity pins, not its results.
    pins={k:v for k,v in prior['frozen_input_sha256'].items() if not k.startswith('scripts/')}
    m.verify_pins(pins)
    threshold_path=ROOT/'docs/analysis/results/paper_dynamics_input_diagnostic_v1/control_group_thresholds.csv'
    pins[m.rel(threshold_path)]=m.file_hash(threshold_path)
    th=pd.read_csv(threshold_path).query('step == 25').iloc[0]
    train_score=excitation(train.trajectory.controls,stats['control_std'],25)
    np.testing.assert_allclose(np.quantile(train_score,[.25,.75]),[th.train_q25,th.train_q75],rtol=1e-12)
    origins=pd.read_csv(m.BASE/'validation_origins.csv')
    score=excitation(val.trajectory.controls,stats['control_std'],25)
    selection=origins.copy();selection['activity']=score;selection['activity_group']=assign_groups(score,th.train_q25,th.train_q75)
    ta,scales=activity(train.trajectory.controls,stats['control_std']);va,_=activity(val.trajectory.controls,stats['control_std'])
    thresholds=[]
    for j,ch in enumerate(CHANNELS):
        q=float(np.quantile(ta[:,j],.75));selection[f'{ch}_activity']=va[:,j];selection[f'{ch}_high']=va[:,j]>q
        thresholds.append(dict(channel=ch,train_q75=q,scale=float(scales[j])))
    delta=state_axes(train.trajectory.truth)[:,1:26]-state_axes(train.trajectory.truth)[:,:1]
    floors=np.quantile(np.abs(weighted_mean(delta,train.trajectory.dt_s[:,:25])),.25,axis=0)
    cases=[]
    for c in ('Sep7','Sep17'):
        for g in ('low','high'):
            x=selection[(selection.cohort==c)&(selection.activity_group==g)].sort_values(['log_id','segment_id','start_sample_in_segment'])
            if len(x):cases.append(dict(cohort=c,group=g,window_id=x.iloc[len(x)//2].window_id,seed=17))
    selection.to_csv(OUT/'origin_selection.csv',index=False)
    pd.DataFrame(thresholds).to_csv(OUT/'channel_thresholds.csv',index=False)
    coverage=[]
    for name,mask in masks(selection).items():
        for log,g in selection[mask].groupby('log_id'):
            coverage.append(dict(group=name,flight_id=log,cohort=g.cohort.iloc[0],n_origins=len(g)))
    pd.DataFrame(coverage).to_csv(OUT/'per_flight_coverage.csv',index=False)
    m.write_json(OUT/'representative_selection.json',cases)
    for path in [Path(__file__),ROOT/'tests/test_model_flight_response_diagnostic.py']:
        pins[m.rel(path)]=m.file_hash(path)
    protocol=dict(experiment='model_flight_response_diagnostic_v1',frozen_at_utc=pd.Timestamp.now(tz='UTC').isoformat(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),model='Standard GRU64/H26/K50',seeds=[17,23,42],
        source_protocol=m.rel(m.OUT/'protocol.json'),normalization_sha256=p['normalization_sha256'],
        frozen_input_sha256=pins,train_origins=28293,validation_origins=2582,
        controls='motor,left,right,rudder post-allocation normalized commands, pre-PWM reversal; not actual angles or frequency',
        coordinates='drive=motor; common=(left+right)/2; differential=(left-right)/2; rudder=rudder',
        activity='500ms frozen Step5 RMS standardized raw-command departure; low<=trainq25; middle<=trainq75; high>trainq75',
        channel_activity='RMS coordinate departure over25 commands / fixed scale derived from frozen raw control_std; common/diff scale=hypot(stdL,stdR)/2, a weighting convention, NOT fitted physical variance',
        channel_groups='each channel > its TRAIN q75; overlapping sets, not isolated interventions; all6 velocity/rate axes reported',
        local_offsets=list(OFFSETS),local_horizon_steps=5,local_definition='true H26 history and measured state at offset; independent5step forecasts; no update inside each forecast; offsets remain inside original50future window',
        local_comparison='matched endpoint times vs original continuous forecast; local has newer observations and shorter forecast age; not a fair main-model leaderboard or causal decomposition',
        response='dt-weighted waveform change from observed t0; windows1..5,6..15,16..25,26..50; axis RMSE and predicted/truth RMS amplitude, sign of dt-weighted mean change',
        direction_floor=dict(zip(AXES,map(float,floors))),direction_floor_rule='train q25 absolute weighted mean change over1..25; same per-axis floor at all intervals; only direction denominator excludes weak real changes, never main RMSE samples',
        lag='descriptive waveform alignment ONLY: causal trailing5sample mean, common truth indices5..44 and prediction index+lag for lag=-5..5; max demeaned correlation, ties prefer zero; constant waveforms undefined; positive lag means prediction later; native time difference, not actuator delay',
        aggregation='origin MSE->sqrt(mean within flight)->equal flights->seed mean/sampleSD; amplitude ratio=macro predicted amplitude/macro true amplitude, not average window ratio',
        representative='median identity in each cohort x low/high group, seed17; frozen before prediction inspection',
        inference_device='cuda:1',prior_heldout_results_known=True,heldout_data_accessed_this_run=False,
        interpretation='response-associated logged-trajectory fidelity, not identified causal actuator effects; no threshold for declaring model control-ready')
    m.write_json(OUT/'protocol.json',protocol)
    print('RULES FROZEN before loading predictions',flush=True)

def masks(selection):
    return {'ALL':np.ones(len(selection),bool),**{g:(selection.activity_group==g).to_numpy() for g in ('low','middle','high')},
            **{ch+'_high':selection[ch+'_high'].to_numpy(bool) for ch in CHANNELS}}

def infer():
    p=m.read_json(OUT/'protocol.json');m.verify_pins(p['frozen_input_sha256']);sp,batches,stats=source.inputs()
    samples=pd.read_parquet((ROOT/sp['manifest_path']).parent/'samples_validation.parquet')
    origins=pd.read_csv(m.BASE/'validation_origins.csv');w=local_windows(origins)
    local=assemble_history_trajectory_windows(samples,w,history_steps=26);n=len(origins);assert local.history_mask.all()
    for i,offset in enumerate(OFFSETS):
        part=slice(i*n,(i+1)*n)
        np.testing.assert_array_equal(local.trajectory.controls[part],batches['validation'].trajectory.controls[:,offset:offset+5])
        np.testing.assert_array_equal(local.trajectory.dt_s[part],batches['validation'].trajectory.dt_s[:,offset:offset+5])
        for key in vars(local.trajectory.truth):np.testing.assert_array_equal(getattr(local.trajectory.truth,key)[part],getattr(batches['validation'].trajectory.truth,key)[:,offset:offset+6])
    for key in ('history_state_features','history_controls','history_mask'):
        np.testing.assert_array_equal(getattr(local,key)[:n],getattr(batches['validation'],key))
    torch.save(local,ART/'local_batch.pt');w.to_csv(ART/'local_origins.csv',index=False)
    checks={}
    for seed in (17,23,42):
        cp=torch.load(m.checkpoint_path(seed),map_location='cpu',weights_only=False)
        model=m.build_model(seed,stats);model.load_state_dict(cp['state_dict'],strict=True)
        assert m.normalization_hash({k:cp['state_dict'][k].numpy() for k in m.STATS})==sp['normalization_sha256']
        pred=predict_history_trajectory_model(model,local,use_history=True,batch_size=128,device='cuda:1')
        for key,v in vars(pred).items():assert np.isfinite(v).all() and v.shape==getattr(local.trajectory.truth,key).shape
        assert np.max(np.abs(np.linalg.norm(pred.quaternion_nb,axis=-1)-1))<1e-5
        with np.load(source.predpath(seed),allow_pickle=False) as f:
            np.testing.assert_array_equal(f['window_ids'],origins.window_id)
            for key,v in vars(pred).items():np.testing.assert_allclose(v[:n],f[key][:,:6],rtol=1e-6,atol=2e-5)
        np.savez_compressed(ART/f'local_seed{seed}.npz',**vars(pred),window_ids=local.trajectory.window_ids.astype(str))
        small=m.subset(local,3);poisoned=copy.deepcopy(small)
        for v in vars(poisoned.trajectory.truth).values():v[:,1:]+=100
        a=predict_history_trajectory_model(model,small,use_history=True,batch_size=3,device='cuda:1')
        b=predict_history_trajectory_model(model,poisoned,use_history=True,batch_size=3,device='cuda:1')
        for key in vars(a):np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
        checks[str(seed)]=dict(finite=True,unit_quaternions=True,origin0_actual_prefix_parity=True,future_label_poisoning=True,local_forecasts=len(w))
        print('INFERRED seed',seed,flush=True)
    m.verify_pins(p['frozen_input_sha256'])
    m.write_json(OUT/'sanity_checks.json',dict(status='passed',seeds=checks,native_dt_labels_controls_exact=True,H26_origin0_exact=True,heldout_data_accessed_this_run=False))

if __name__=='__main__':
    m.configure()
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='infer':infer()
