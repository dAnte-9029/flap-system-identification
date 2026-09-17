#!/usr/bin/env python3
"""Step 7: frozen Main V2; train-only banks/probes; validation oracle isolation."""
from __future__ import annotations
import argparse
from dataclasses import fields, replace
import json, os, sys, time, subprocess
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-step7')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import pandas as pd
import torch
from run_main_v2_free_running import load_simulator, make_initial, sha
from system_identification.training.trajectory_main_v1 import assemble_history_trajectory_windows
from system_identification.evaluation.trajectory import POSITION_COLUMNS,VELOCITY_COLUMNS,QUATERNION_COLUMNS,BODY_RATE_COLUMNS
from system_identification.data.trajectory_dataset import CONTROL_COLUMNS
from system_identification.models.trajectory_main_v1 import _state_features, _rotation_body_to_ned
from system_identification.models.rolling_history_simulator import RollingHistorySimulator,state_features
from system_identification.evaluation.recurrent_representation import teacher_physical,teacher_recurrent_step
from system_identification.evaluation.main_v2_free_running import HORIZONS,PHYSICAL_FIELDS,endpoint_errors,aggregate,stability

DATA=ROOT/'dataset/trajectory_v1_august_f5_c4'
OLD=ROOT/'docs/analysis/results'
CHECKPOINT=ROOT/'artifacts/trajectory_main_v2/models/main_v2_drive_tail_gated.pt'
COLS=dict(position_n=POSITION_COLUMNS,velocity_n=VELOCITY_COLUMNS,quaternion_nb=QUATERNION_COLUMNS,
          angular_velocity_b=BODY_RATE_COLUMNS,relative_phase_rad=['relative_flap_phase_rad'],flap_frequency_hz=['flap_frequency_hz'])
MODES=['A_native','A_fixed','B','C','C_native','D']
SELECT=np.unique(np.r_[np.arange(0,251,10),25,249]).astype(int)

def npy(t): return t.detach().cpu().numpy()
def tensor(a,device): return torch.as_tensor(a,dtype=torch.float32,device=device)
def save_json(p,v): p.write_text(json.dumps(v,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
def log(s): print(time.strftime('%H:%M:%S'),s,flush=True)

def histories(samples, origins, length, future, device):
    groups={k:g.sort_values('sample_in_segment') for k,g in samples.loc[samples.valid_core].groupby(['log_id','segment_id'])}
    values={k:[] for k in COLS}; controls=[]; stamps=[]
    for r in origins.itertuples():
        g=groups[(r.log_id,r.segment_id)]
        start=int(r.start_sample_in_segment)
        a=g.iloc[start-length+1:start+future+1]
        assert len(a)==length+future and np.array_equal(a.sample_in_segment,np.arange(start-length+1,start+future+1))
        for name,col in COLS.items():
            v=a[list(col)].to_numpy()
            values[name].append(v[:,0] if len(col)==1 else v)
        controls.append(a[list(CONTROL_COLUMNS)].to_numpy());stamps.append(a.timestamp_us.to_numpy()*1e-6)
    values={k:tensor(np.stack(v),device) for k,v in values.items()}
    return values,tensor(np.stack(controls),device),np.stack(stamps)

def features_at(p,sl,anchor):
    n=p['velocity_n'][:,sl].shape[:2]
    a=anchor[:,None].expand(n).reshape(-1)
    return _state_features(p['velocity_n'][:,sl].reshape(-1,3),p['quaternion_nb'][:,sl].reshape(-1,4),
        p['angular_velocity_b'][:,sl].reshape(-1,3),p['relative_phase_rad'][:,sl].reshape(-1),a,
        p['flap_frequency_hz'][:,sl].reshape(-1)).reshape(*n,12)

def encoding(base,p,c,step,anchor):
    f=features_at(p,slice(step,step+26),anchor)
    return base._encode_history(f,c[:,step:step+26],torch.ones(f.shape[:2],device=f.device,dtype=torch.bool))

def output_vector(before,after,diag,dt):
    a=torch.einsum('bij,bj->bi',_rotation_body_to_ned(before.quaternion_nb),diag.acceleration_b)
    return torch.cat((a,diag.angular_acceleration_b,((after.flap_frequency_hz-before.flap_frequency_hz)/dt)[:,None],diag.acceleration_b),1)

def trace(states,diags):
    return ({f.name:np.stack([s[f.name] for s in states],1) for f in fields(states[0]['_type'])},
            {f.name:np.stack([d[f.name] for d in diags],1) for f in fields(diags[0]['_type'])})
def pack(s):return {**{f.name:npy(getattr(s,f.name)) for f in fields(s)},'_type':type(s)}

def run_modes(sim,batch,p,c,out,art,windows,device):
    n=len(windows); base=sim.model.base_model
    dt=tensor(batch.trajectory.dt_s,device);cmd=tensor(batch.trajectory.controls,device)
    init=make_initial(sim,batch,np.arange(n),'warm26',device)
    state={m:init for m in MODES}; history={m:[pack(init)] for m in MODES}; diagnostic={m:[] for m in MODES}
    hteacher=[];h_native=[];sens_h=[];sens_x=[];op=[]
    with torch.no_grad():
        for k in range(250):
            now={name:p[name][:,25+k] for name in PHYSICAL_FIELDS}
            nex={name:p[name][:,26+k] for name in PHYSICAL_FIELDS}
            hf=encoding(base,p,c,k,init.phase_anchor);hn=encoding(base,p,c,k,now['relative_phase_rad'])
            if k==0: hf=hn=init.gru_hidden
            hteacher.append(npy(hf));h_native.append(npy(hn))
            ref=replace(teacher_physical(state['D'],now),gru_hidden=hf)
            ref_next,ref_diag=sim.step(ref,cmd[:,k],dt[:,k]);refout=output_vector(ref,ref_next,ref_diag,dt[:,k])
            hp=replace(ref,gru_hidden=state['D'].gru_hidden)
            hp_next,hp_diag=sim.step(hp,cmd[:,k],dt[:,k])
            xp=replace(state['D'],gru_hidden=hf)
            xp_next,xp_diag=sim.step(xp,cmd[:,k],dt[:,k])
            sens_h.append(npy(output_vector(hp,hp_next,hp_diag,dt[:,k])-refout))
            sens_x.append(npy(output_vector(xp,xp_next,xp_diag,dt[:,k])-refout))
            old_free_hidden=state['D'].gru_hidden
            old_b_hidden=state['B'].gru_hidden
            for m in MODES:
                s=state[m]
                if m=='B':
                    state[m],pred,d=teacher_recurrent_step(sim,s,cmd[:,k],dt[:,k],now,nex)
                else:
                    if m.startswith('A'):
                        s=teacher_physical(s,now)
                        s=replace(s,gru_hidden=hn if m=='A_native' else hf,
                                  phase_anchor=now['relative_phase_rad'] if m=='A_native' else init.phase_anchor)
                    elif m.startswith('C'):
                        s=replace(s,gru_hidden=hn if m=='C_native' else hf,
                                  phase_anchor=now['relative_phase_rad'] if m=='C_native' else init.phase_anchor)
                    pred,d=sim.step(s,cmd[:,k],dt[:,k]);state[m]=pred
                history[m].append(pack(replace(pred,gru_hidden=state[m].gru_hidden)));diagnostic[m].append(pack(d))
            if k in SELECT:
                op.append(dict(step=k,teacher=npy(hf),free=npy(old_free_hidden),b=npy(old_b_hidden),
                    z_teacher=npy(base._model_input(state_features(teacher_physical(ref,nex)),cmd[:,k])),
                    z_free=npy(base._model_input(state_features(state['D']),cmd[:,k]))))
            if k%50==0:log(f'A-D transitions {k}/250 ({n} shared origins)')
        # h at t=250 needed for hidden distance / information probe.
        hteacher.append(npy(encoding(base,p,c,250,init.phase_anchor)))
        h_native.append(npy(encoding(base,p,c,250,p['relative_phase_rad'][:,-1])))
    traces={}
    for m in MODES:
        pred,diag=trace(history[m],diagnostic[m]);traces[m]=(pred,diag)
        np.savez_compressed(art/f'{m}.npz',**pred,**{'diag_'+k:v for k,v in diag.items()})
    np.savez_compressed(art/'interventions.npz',teacher=np.stack(hteacher,1),native=np.stack(h_native,1),
                        hidden_effect=np.stack(sens_h,1),state_effect=np.stack(sens_x,1))
    torch.save(op,art/'operating_points.pt')
    return traces

def evaluate(traces,batch,windows,out):
    rows=[];var=[]
    env=json.loads((OLD/'main_v2_free_running_5s/observed_envelope.json').read_text())['train']
    truth=batch.trajectory.truth;dt=batch.trajectory.dt_s
    da=np.diff(truth.velocity_n,axis=1)/dt[...,None];dw=np.diff(truth.angular_velocity_b,axis=1)/dt[...,None]
    for m,(pred,diag) in traces.items():
        _,flags=stability(pred,diag,dt,env)
        # For A/B predictions are independent one-step endpoints, not a continuous trajectory.
        current_q=truth.quaternion_nb[:,:-1] if m.startswith('A') or m=='B' else pred['quaternion_nb'][:,:-1]
        from scipy.spatial.transform import Rotation
        q=current_q.reshape(-1,4)
        rot=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix().reshape(*current_q.shape[:2],3,3)
        pa=np.einsum('btij,btj->bti',rot,diag['acceleration_b'])
        for sec,k in HORIZONS.items():
            errors=endpoint_errors(pred,truth,k)
            # Native last 10 transitions near each horizon (not ever-longer cumulative derivative average).
            lo=max(0,k-10)
            errors['linear_acceleration']=np.sqrt(np.mean(np.sum((pa[:,lo:k]-da[:,lo:k])**2,axis=-1),axis=1))
            errors['angular_acceleration']=np.sqrt(np.mean(np.sum((diag['angular_acceleration_b'][:,lo:k]-dw[:,lo:k])**2,axis=-1),axis=1))
            for i,r in enumerate(windows.itertuples()):
                rows.append(dict(mode=m,deployable=m in ('D','R13','R26'),window_id=r.window_id,log_id=r.log_id,
                    horizon_s=sec,actual_duration_s=dt[i,:k].sum(),failed=flags['failed'][i,:k].any(),
                    numerical_failed=flags['numerical'][i,:k].any(),support_failed=flags['envelope'][i,:k].any(),
                    clipping_failed=flags['clipping'][i,:k].any(),**{name:v[i] for name,v in errors.items()}))
            if not m.startswith('A') and m!='B':
                for label,sl in [('prefix',slice(0,k+1))]+([('last1s',slice(200,251))] if k==250 else []):
                    ratio=np.linalg.norm(np.std(pred['angular_velocity_b'][:,sl],axis=1),axis=1)/np.maximum(np.linalg.norm(np.std(truth.angular_velocity_b[:,sl],axis=1),axis=1),1e-9)
                    for i,r in enumerate(windows.itertuples()):var.append(dict(mode=m,log_id=r.log_id,window_id=r.window_id,horizon_s=sec,scope=label,ratio=ratio[i]))
    frame=pd.DataFrame(rows);frame.to_csv(out/'per_rollout.csv',index=False)
    metrics=list(errors)
    summary=aggregate(frame,['mode','horizon_s'],metrics)
    summary['deployable']=summary['mode'].isin(['D','R13','R26'])
    summary['interpretation']=np.where(summary['deployable'],'autonomous','NOT DEPLOYABLE diagnostic')
    summary.to_csv(out/'rollout_mode_comparison.csv',index=False)
    summary[summary['mode'].isin(['D','R13','R26'])].to_csv(out/'free_running_per_horizon.csv',index=False)
    aggregate(frame,['mode','log_id','horizon_s'],metrics).to_csv(out/'per_flight.csv',index=False)
    pd.DataFrame(var).to_csv(out/'variation_per_rollout.csv',index=False)
    pd.DataFrame(var).groupby(['mode','horizon_s','scope']).ratio.agg(['median','mean','min','max']).reset_index().to_csv(out/'variation_summary.csv',index=False)
    return summary

def run_rolling(sim,batch,device,k):
    with torch.no_grad():
        init=make_initial(sim,batch,np.arange(len(batch.trajectory.dt_s)),'warm26',device)
        rolling=RollingHistorySimulator(sim,k)
        s=rolling.reset(init,tensor(batch.history_state_features,device),tensor(batch.history_controls,device),torch.as_tensor(batch.history_mask,device=device))
        states=[pack(s.physical)];diags=[]
        for j in range(250):
            s,d=rolling.step(s,tensor(batch.trajectory.controls[:,j],device),tensor(batch.trajectory.dt_s[:,j],device))
            states.append(pack(s.physical));diags.append(pack(d))
    return trace(states,diags)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=OLD/'main_v2_recurrent_representation')
    ap.add_argument('--artifacts',type=Path,default=ROOT/'artifacts/main_v2_recurrent_representation')
    ap.add_argument('--device',default='cuda:1');ap.add_argument('--smoke',action='store_true');ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();out=args.output;art=args.artifacts
    for d in (out,art):
        if d.exists() and any(d.iterdir()) and not args.resume:raise FileExistsError(d)
        d.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.manual_seed(717);np.random.seed(717)
    torch.use_deterministic_algorithms(True)
    if args.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('requested GPU unavailable')
    started=time.time()
    protected=[CHECKPOINT,DATA/'manifest.json',DATA/'samples_train.parquet',DATA/'samples_validation.parquet']
    for dirname in ('main_v2_free_running_5s','main_v2_training_objective_ablation','main_v2_dynamics_observability_phase','main_v2_increment_supervision','main_v2_phase_reference'):
        protected.extend(p for p in (OLD/dirname).rglob('*') if p.is_file())
    protected.extend((ROOT/'artifacts/main_v2_increment_supervision').glob('*/model.pt'))
    hashes={str(p.relative_to(ROOT)):sha(p) for p in protected}
    if args.resume and (out/'historical_hashes.json').exists():
        assert json.loads((out/'historical_hashes.json').read_text())==hashes,'protected files changed since first launch'
    save_json(out/'historical_hashes.json',hashes)
    windows=pd.read_csv(OLD/'main_v2_free_running_5s/windows.csv')
    assert len(windows)==722 and sha(OLD/'main_v2_free_running_5s/windows.csv')=='c4ce42a6dbfac148aa790a85ead3d3e4a473c79b595e2d26d632832949641929'
    if args.smoke:windows=windows.groupby('log_id',sort=False).head(2).reset_index(drop=True)
    windows.to_csv(out/'origins.csv',index=False)
    manifest=json.loads((DATA/'manifest.json').read_text());assert not manifest['split_contract']['sealed_test_opened']
    validation=pd.read_parquet(DATA/'samples_validation.parquet')
    assert set(validation.log_id)==set(manifest['split_contract']['assignments']['validation'])
    batch=assemble_history_trajectory_windows(validation,windows,history_steps=26)
    p,c,timestamps=histories(validation,windows,26,250,args.device)
    sim=load_simulator(CHECKPOINT,args.device)
    save_json(out/'manifest.json',dict(seed=717,device=args.device,n_origins=len(windows),checkpoint_sha256=sha(CHECKPOINT),
        step_contract='250 native steps, no resampling',dt_min=float(batch.trajectory.dt_s.min()),dt_max=float(batch.trajectory.dt_s.max()),
        dt_mean=float(batch.trajectory.dt_s.mean()),duration_min=float(batch.trajectory.dt_s.sum(1).min()),duration_max=float(batch.trajectory.dt_s.sum(1).max()),
        teacher_modes='A_native/A_fixed/B/C/C_native NOT DEPLOYABLE',
        anchor_control='A_native reanchors each history. A_fixed/B/C/D preserve original t0 anchor. C_native supplementary changes hidden and anchor.',
        proxies='all modes same causal command-driven proxies; no reinitialization after t0',
        physical_teacher='all six physical fields including frequency and encoder phase; NOT DEPLOYABLE',
        bank='real train histories stride10, eight equally spaced anchor gauges; no validation fitting',
        ood='train-standardized k5 RMS Euclidean and ridge covariance Mahalanobis; no certified safety interpretation',
        probe='train-only ridge1 with train-only scales, fixed small linear information probe; not simulator training',
        jacobian='full64x64 frozen-input local Jacobian at deterministic per-log points; singular norm and spectral radius distinct',
        resumed=bool(args.resume),source_hashes={str(q.relative_to(ROOT)):sha(q) for q in [Path(__file__),ROOT/'src/system_identification/models/rolling_history_simulator.py',ROOT/'src/system_identification/evaluation/recurrent_representation.py',ROOT/'scripts/analyze_main_v2_recurrent.py',ROOT/'scripts/report_main_v2_recurrent.py',ROOT/'tests/test_recurrent_representation.py']}))
    if args.resume and (art/'interventions.npz').exists():
        traces={}
        for m in MODES:
            z=np.load(art/f'{m}.npz');traces[m]=({k:z[k] for k in z.files if not k.startswith('diag_')},{k[5:]:z[k] for k in z.files if k.startswith('diag_')})
    else:traces=run_modes(sim,batch,p,c,out,art,windows,args.device)
    reference=np.load(ROOT/'artifacts/main_v2_increment_supervision/S0/validation_rollout.npz')
    ref_ids={v:i for i,v in enumerate(reference['window_ids'])}
    take=[ref_ids[v] for v in windows.window_id]
    parity={k:float(np.max(np.abs(traces['D'][0][k]-reference[k][take]))) for k in traces['D'][0]}
    save_json(out/'baseline_parity.json',parity)
    for key in PHYSICAL_FIELDS:
        np.testing.assert_allclose(traces['D'][0][key],reference[key][take],rtol=2e-4,atol=2e-4)
    summary=evaluate(traces,batch,windows,out)
    # Predeclared diagnostic trigger: >5% C improvement in either long-horizon state error,
    # or B derivative deterioration >5% vs matched A_fixed. This is NOT promotion.
    idx=summary.set_index(['mode','horizon_s'])
    c_gain=max(1-idx.loc[('C',h),metric]/idx.loc[('D',h),metric] for h in (2.,3.,5.) for metric in ('velocity_m_s_equal_log_rmse','attitude_deg_equal_log_rmse'))
    b_gap=max(idx.loc[('B',h),'angular_acceleration_equal_log_rmse']/idx.loc[('A_fixed',h),'angular_acceleration_equal_log_rmse']-1 for h in (2.,3.,5.))
    gate=bool(c_gain>.05 or b_gap>.05)
    save_json(out/'reencode_gate.json',dict(triggered=gate,max_c_improvement=c_gain,max_b_angular_gap=b_gap,rule='C long improvement >5% OR B angular gap >5%; diagnostic only'))
    save_json(out/'experiment_decision.json',dict(training_authorized_by_evidence=gate,training_performed=False,
        frozen_reencode_reason='User-required predicted-history negative controls K13/K26; NOT a positive gate or new training'))
    for k in (13,26):
        log(f'Autonomous predicted-history R{k}; diagnostic trigger={gate}')
        path=art/f'R{k}.npz'
        if args.resume and path.exists():
            z=np.load(path);traces[f'R{k}']=({x:z[x] for x in z.files if not x.startswith('diag_')},{x[5:]:z[x] for x in z.files if x.startswith('diag_')})
        else:
            pred,diag=run_rolling(sim,batch,args.device,k);traces[f'R{k}']=(pred,diag)
            np.savez_compressed(path,**pred,**{'diag_'+x:v for x,v in diag.items()})
    summary=evaluate(traces,batch,windows,out)
    from analyze_main_v2_recurrent import analyze
    from report_main_v2_recurrent import report
    analyze(sim,batch,p,c,windows,traces,out,art,args.device,args.smoke)
    report(out)
    unchanged=all(sha(ROOT/k)==v for k,v in hashes.items())
    assert unchanged,'historical file changed'
    tests=subprocess.run([sys.executable,'-m','pytest','-q',*[str(p.relative_to(ROOT)) for p in sorted((ROOT/'tests').glob('test_*.py')) if p.name.startswith(('test_main_v2','test_trajectory')) or p.name in ('test_recurrent_representation.py','test_september_trajectory.py')]],cwd=ROOT,capture_output=True,text=True)
    (out/'pytest.txt').write_text(tests.stdout+tests.stderr)
    diff=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True)
    new_checks={}
    for path in [Path(__file__),ROOT/'scripts/analyze_main_v2_recurrent.py',ROOT/'scripts/report_main_v2_recurrent.py',
                 ROOT/'src/system_identification/models/rolling_history_simulator.py',ROOT/'src/system_identification/evaluation/recurrent_representation.py',ROOT/'tests/test_recurrent_representation.py']:
        check=subprocess.run(['git','diff','--no-index','--check','/dev/null',str(path)],capture_output=True,text=True)
        new_checks[str(path.relative_to(ROOT))]=dict(exit_code=check.returncode,output=check.stdout+check.stderr,
            clean=check.returncode in (0,1) and not (check.stdout+check.stderr).strip())
    save_json(out/'verification.json',dict(historical_hashes_unchanged=unchanged,pytest_exit=tests.returncode,git_diff_exit=diff.returncode,
        git_diff_output=diff.stdout+diff.stderr,new_file_diff_checks=new_checks,wall_seconds=time.time()-started,wall_time_scope='this invocation',trace_cache_reused=bool(args.resume),
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'scripts/analyze_main_v2_recurrent.py',ROOT/'scripts/report_main_v2_recurrent.py',ROOT/'tests/test_recurrent_representation.py']}))
    if tests.returncode or diff.returncode or not all(v['clean'] for v in new_checks.values()):raise RuntimeError('verification failed')
    log('Completed Step 7 report and verification')

if __name__=='__main__':main()
