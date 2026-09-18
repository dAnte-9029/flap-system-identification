#!/usr/bin/env python3
"""Frozen GPU control-horizon evaluation; validation only, no model fitting."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-prediction-horizon')
import sys,json,argparse,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd,torch
from run_main_v2_free_running import load_simulator,make_initial,rollout,sha
from run_main_v2_recurrent_audit import histories
from system_identification.training.trajectory_main_v1 import assemble_history_trajectory_windows
from system_identification.evaluation.main_v2_free_running import endpoint_errors,stability,PHYSICAL_FIELDS
from system_identification.evaluation.prediction_horizon import STEPS,teacher_refresh_rollout,kinematic_hold,variation_ratio,error_statistics

OLD=ROOT/'docs/analysis/results';DATA=ROOT/'dataset/trajectory_v1_august_f5_c4'
BASE=ROOT/'artifacts/trajectory_main_v2/models/main_v2_drive_tail_gated.pt'
KEYS=('velocity_m_s','attitude_deg','body_rate_rad_s','frequency_hz','position_m','phase_rad')


def summaries(pred,truth,windows,dt,model,mode,rows,metrics,variations):
    for k in STEPS:
        err=endpoint_errors(pred,truth,k);ratio,denominator=variation_ratio(pred['angular_velocity_b'],truth.angular_velocity_b,k)
        duration=dt[:,:k].sum(1)*1000
        frame=windows[['window_id','log_id','segment_id','start_sample_in_segment']].copy()
        frame['model']=model;frame['mode']=mode;frame['steps']=k;frame['nominal_ms']=20*k;frame['actual_ms']=duration
        for key,value in err.items():frame[key]=value
        frame['variation_ratio']=ratio;frame['truth_omega_std_norm']=denominator;rows.append(frame)
        for logid,ix in [('ALL',np.ones(len(windows),dtype=bool))]+[(log,windows.log_id.to_numpy()==log) for log in windows.log_id.unique()]:
            for key in KEYS:
                stats=error_statistics(err[key][ix],windows.log_id.to_numpy()[ix])
                metrics.append(dict(model=model,mode=mode,steps=k,log_id=logid,metric=key,nominal_ms=20*k,actual_ms_mean=duration[ix].mean(),actual_ms_min=duration[ix].min(),actual_ms_max=duration[ix].max(),**stats))
            finite=ratio[ix][np.isfinite(ratio[ix])]
            variations.append(dict(model=model,mode=mode,steps=k,log_id=logid,nominal_ms=20*k,actual_ms_mean=duration[ix].mean(),valid_count=len(finite),excluded_near_zero_truth=int((~np.isfinite(ratio[ix])).sum()),median=np.median(finite),p10=np.quantile(finite,.1),p90=np.quantile(finite,.9),interpretation='one-increment amplitude ratio, not cycle fidelity' if k==1 else ('truth-refreshed stitched series, not autonomous fidelity' if mode=='A_teacher_refresh' else 'short-prefix variation; may contain less than one wing cycle')))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=OLD/'main_v2_prediction_horizon');ap.add_argument('--device',default='cuda:1');ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    if not a.device.startswith('cuda') or not torch.cuda.is_available():raise RuntimeError('GPU evaluation required; no CPU fallback')
    if a.output.exists() and any(a.output.iterdir()):raise FileExistsError('refusing nonempty output '+str(a.output))
    a.output.mkdir(parents=True,exist_ok=True);out=a.output;started=time.time()
    torch.set_num_threads(4);torch.manual_seed(17);np.random.seed(17);torch.use_deterministic_algorithms(True)
    protected={}
    for directory in ['main_v2_free_running_5s','main_v2_training_objective_ablation','main_v2_dynamics_observability_phase','main_v2_increment_supervision','main_v2_phase_reference','main_v2_recurrent_representation','main_v2_rollout_consistency']:
        for p in (OLD/directory).rglob('*'):
            if p.is_file() and not p.is_symlink():protected[str(p.relative_to(ROOT))]=sha(p)
    for p in (ROOT/'docs/audits').glob('*main_v2*.md'):protected[str(p.relative_to(ROOT))]=sha(p)
    frozen=json.loads((OLD/'main_v2_free_running_5s/summary.json').read_text())['source_hashes']
    for p,h in frozen.items():
        if sha(ROOT/p)!=h:raise ValueError('frozen contract changed '+p)
    checkpoints={'S0':BASE,'Step5_S1':ROOT/'artifacts/main_v2_increment_supervision/S1/model.pt'}
    if not checkpoints['Step5_S1'].exists():raise FileNotFoundError('expected compatible Step5 checkpoint absent')
    s1_reference=json.loads((OLD/'main_v2_increment_supervision/S1/summary.json').read_text())
    if sha(checkpoints['Step5_S1'])!=s1_reference['checkpoint_sha256']:raise ValueError('Step5 checkpoint hash mismatch')
    source=[Path(__file__),ROOT/'scripts/report_main_v2_prediction_horizon.py',ROOT/'src/system_identification/evaluation/prediction_horizon.py',ROOT/'tests/test_prediction_horizon.py',ROOT/'scripts/run_main_v2_free_running.py',ROOT/'scripts/run_main_v2_recurrent_audit.py',ROOT/'src/system_identification/evaluation/main_v2_free_running.py',ROOT/'src/system_identification/training/trajectory_main_v1.py',ROOT/'src/system_identification/models/main_v2_simulator.py']
    inputs=[DATA/'manifest.json',DATA/'samples_validation.parquet',OLD/'main_v2_free_running_5s/windows.csv',*checkpoints.values()]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs+source}
    manifest=json.loads((DATA/'manifest.json').read_text())
    if manifest['split_contract']['sealed_test_opened'] or manifest['split_contract']['materialized_partitions']!=['train','validation']:raise ValueError('unexpected split contract')
    samples=pd.read_parquet(DATA/'samples_validation.parquet')
    assert set(samples.log_id)==set(manifest['split_contract']['assignments']['validation'])
    windows=pd.read_csv(OLD/'main_v2_free_running_5s/windows.csv');assert len(windows)==722
    if a.smoke:windows=windows.groupby('log_id',sort=False).head(2).reset_index(drop=True)
    evaluation_windows=windows.copy();evaluation_windows['state_sample_count']=26
    batch=assemble_history_trajectory_windows(samples,evaluation_windows,history_steps=26)
    p,c,stamps=histories(samples,windows,26,25,a.device)
    dt=batch.trajectory.dt_s;np.testing.assert_allclose(np.diff(stamps[:,25:],axis=1),dt,atol=1e-10,rtol=0)
    config=dict(seed=17,device=a.device,gpu=torch.cuda.get_device_name(a.device),torch=torch.__version__,numpy=np.__version__,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        dataset='explicit frozen historical trajectory_v1_august_f5_c4',partition='validation',n_origins=len(windows),smoke=a.smoke,steps=list(STEPS),history_points=26,
        mode_A='NOT DEPLOYABLE: Step7 A_native one-step predictions from true state/history at each predecessor t_(k-1); proxies continue along commands; not t0-issued k-step forecast',
        mode_B='deployable autonomous k-step forecast from a single true warm26 t0; no future states/frequency/phase',
        identifiability_note='A literal k-step forecast from the same teacher t0 equals B. A/B gap here compares refreshed one-step errors with accumulated prediction, not equal information budgets or causal percentages.',
        reference='constant NED velocity/body rate/frequency; integrate position/quaternion/phase using native dt; no fitted dynamics',
        thresholds={'velocity_m_s':[.5,1.],'attitude_deg':[5.,10.]},threshold_policy='descriptive strict less-than only; no safety/pass certification',
        metric_aggregation='primary equal mean of 5 per-flight vector/geodesic RMSE; pooled RMSE and origin quantiles also saved',
        growth='equal-flight RMSE(k)/RMSE(1) within each model and mode; A index is endpoint time, not forecast lead',
        dt_s=dict(min=float(dt.min()),mean=float(dt.mean()),max=float(dt.max())),
        checkpoints={n:dict(path=str(path.relative_to(ROOT)),sha256=sha(path)) for n,path in checkpoints.items()},input_source_hashes=hashes,historical_hashes=protected,frozen_source_hashes=frozen,
        trained=False,weights_changed=False,phase_changed=False,actuator_changed=False,integration_changed=False,sealed_test_opened=False)
    (out/'manifest.json').write_text(json.dumps(config,indent=2));windows.to_csv(out/'origins.csv',index=False)
    rows=[];metrics=[];variations=[];parity={};health=[]
    env=json.loads((OLD/'main_v2_free_running_5s/observed_envelope.json').read_text())['train']
    def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device=a.device)
    for name,checkpoint in checkpoints.items():
        print('GPU EVALUATION',name,flush=True)
        sim=load_simulator(BASE,a.device)
        if name!='S0':sim.model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict'],strict=True)
        before={k:v.detach().cpu().clone() for k,v in sim.model.state_dict().items()}
        all_b=[];all_a=[];all_d=[];all_hold=[]
        with torch.inference_mode():
            for start in range(0,len(windows),128):
                ids=np.arange(start,min(start+128,len(windows)));initial=make_initial(sim,batch,ids,'warm26',a.device)
                command=tensor(batch.trajectory.controls[ids]);native_dt=tensor(dt[ids])
                b,diag=rollout(sim,initial,command,native_dt)
                teacher=teacher_refresh_rollout(sim,initial,{key:value[ids] for key,value in p.items()},c[ids],command,native_dt)
                for key in PHYSICAL_FIELDS:np.testing.assert_array_equal(b[key][:,1],teacher[key][:,1])
                all_b.append(b);all_a.append(teacher);all_d.append(diag)
                if name=='S0':all_hold.append(kinematic_hold(initial,native_dt))
        def cat(chunks):return {key:np.concatenate([v[key] for v in chunks]) for key in chunks[0]}
        b,teacher,diag=cat(all_b),cat(all_a),cat(all_d)
        reference=np.load(ROOT/'artifacts/main_v2_increment_supervision'/('S0' if name=='S0' else 'S1')/'validation_rollout.npz')
        lookup={w:i for i,w in enumerate(reference['window_ids'])};indices=[lookup[w] for w in windows.window_id]
        parity[name]={key:float(np.max(np.abs(b[key]-reference[key][indices,:26]))) for key in b}
        for key in PHYSICAL_FIELDS:np.testing.assert_allclose(b[key],reference[key][indices,:26],atol=2e-4,rtol=2e-4)
        for key,value in sim.model.state_dict().items():assert torch.equal(before[key],value.cpu()),'evaluation modified weights'
        for mode,pred in [('A_teacher_refresh',teacher),('B_autonomous',b)]:
            summaries(pred,batch.trajectory.truth,windows,dt,name,mode,rows,metrics,variations)
            np.savez_compressed(out/f'{name}_{mode}.npz',**{k:pred[k] for k in PHYSICAL_FIELDS},dt_s=dt,window_ids=windows.window_id.to_numpy(str))
        _,flags=stability(b,diag,dt,env)
        for k in STEPS:health.append(dict(model=name,steps=k,n=len(windows),numerical_failures=int(flags['numerical'][:,:k].any(1).sum()),clipping_failures=int(flags['clipping'][:,:k].any(1).sum()),support_failures=int(flags['envelope'][:,:k].any(1).sum())))
        if name=='S0':summaries(cat(all_hold),batch.trajectory.truth,windows,dt,'kinematic_hold','reference',rows,metrics,variations)
    frame=pd.concat(rows,ignore_index=True);m=pd.DataFrame(metrics);v=pd.DataFrame(variations)
    base=m[m.steps==1][['model','mode','log_id','metric','equal_flight_rmse']].rename(columns={'equal_flight_rmse':'one_step_rmse'})
    m=m.merge(base,on=['model','mode','log_id','metric'],validate='many_to_one');m['relative_error_growth']=m.equal_flight_rmse/m.one_step_rmse
    m.to_csv(out/'prediction_horizon_metrics.csv',index=False);frame.to_csv(out/'per_origin.csv',index=False);v.to_csv(out/'variation_summary.csv',index=False)
    pd.DataFrame(health).to_csv(out/'stability_summary.csv',index=False);(out/'baseline_parity.json').write_text(json.dumps(parity,indent=2))
    thresholds=[]
    for key,g in frame.groupby(['model','mode','steps']):
        for metric,limits in config['thresholds'].items():
            x=g[metric].to_numpy();stats=error_statistics(x,g.log_id.to_numpy())
            for threshold in limits:
                thresholds.append(dict(model=key[0],mode=key[1],steps=key[2],nominal_ms=20*key[2],actual_ms_mean=g.actual_ms.mean(),metric=metric,threshold=threshold,
                    equal_flight_rmse=stats['equal_flight_rmse'],rmse_below=stats['equal_flight_rmse']<threshold,pooled_rmse_below=stats['pooled_rmse']<threshold,p90=stats['p90'],p90_below=stats['p90']<threshold,
                    fraction_below=float((x<threshold).mean()),equal_flight_fraction_below=float(g.assign(below=x<threshold).groupby('log_id').below.mean().mean())))
    pd.DataFrame(thresholds).to_csv(out/'control_threshold_summary.csv',index=False)
    from report_main_v2_prediction_horizon import report
    report(out)
    if not a.smoke:
        with (out/'pytest.log').open('w') as f:subprocess.run([sys.executable,'-m','pytest','-q','tests/test_prediction_horizon.py','tests/test_main_v2_simulator.py','tests/test_main_v2_free_running.py','tests/test_recurrent_representation.py'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
    changed=[p for p,h in {**protected,**hashes,**frozen}.items() if sha(ROOT/p)!=h]
    if changed:raise ValueError('protected inputs changed '+str(changed))
    (out/'verification.json').write_text(json.dumps(dict(protected_files_checked=len({**protected,**hashes,**frozen}),changed=changed,weights_unchanged=True,first_step_teacher_autonomous_identical=True,baseline_prefix_parity_passed=True,gpu_evaluation=True,pytest_passed=not a.smoke,pytest_scope='prediction horizon and affected simulator/free-running/recurrent tests',git_diff_check_passed=True,sealed_test_opened=False,wall_time_s=time.time()-started),indent=2))
    print('COMPLETED',out,flush=True)

if __name__=='__main__':main()
