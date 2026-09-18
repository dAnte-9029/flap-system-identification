#!/usr/bin/env python3
"""Step 8 common-origin metrics, state perturbations and partial state Jacobians."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-step8')
import sys,json,argparse
from pathlib import Path
from dataclasses import fields,replace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd,torch
from run_main_v2_free_running import load_simulator,make_initial,rollout,sha
from system_identification.training.trajectory_main_v1 import assemble_history_trajectory_windows
from system_identification.evaluation.main_v2_free_running import HORIZONS,PHYSICAL_FIELDS,endpoint_errors,aggregate,stability
from system_identification.evaluation.transition_stability import CHART_SCALES,perturb_state,state_difference,physical_jacobian,jacobian_spectra
from system_identification.models.main_v2_simulator import SimulatorState
from system_identification.models.trajectory import attitude_error_deg

OLD=ROOT/'docs/analysis/results';DATA=ROOT/'dataset/trajectory_v1_august_f5_c4'
BASE=ROOT/'artifacts/trajectory_main_v2/models/main_v2_drive_tail_gated.pt'


def t(v,device):return torch.as_tensor(v,dtype=torch.float32,device=device)
def npy(v):return v.detach().cpu().numpy()
def select_rows(w,n=10):
    return np.concatenate([g.index.to_numpy()[np.linspace(0,len(g)-1,min(n,len(g)),dtype=int)] for _,g in w.groupby('log_id',sort=True)])


def metric_tables(name,pred,diag,batch,windows,envelope,out):
    truth=batch.trajectory.truth;dt=batch.trajectory.dt_s
    _,flags=stability(pred,diag,dt,envelope)
    rows=[];growth=[];variation=[]
    for step in range(1,251):
        err=endpoint_errors(pred,truth,step)
        for key,values in err.items():
            if key not in ('position_m','velocity_m_s','attitude_deg','body_rate_rad_s','frequency_hz','phase_rad'):continue
            for logid in windows.log_id.unique():
                sel=windows.log_id.to_numpy()==logid;v=values[sel]
                growth.append(dict(experiment=name,step=step,horizon_s=step*.02,actual_time_s=dt[sel,:step].sum(1).mean(),log_id=logid,metric=key,rmse=np.sqrt(np.mean(v*v)),median=np.median(v),p95=np.quantile(v,.95)))
        if step not in HORIZONS.values():continue
        horizon=next(h for h,k in HORIZONS.items() if k==step)
        frame=windows.copy();frame['experiment']=name;frame['horizon_s']=horizon
        for key,value in err.items():frame[key]=value
        for output,key in [('failed','failed'),('numerical_failed','numerical'),('support_failed','envelope'),('clipping_failed','clipping')]:frame[output]=flags[key][:,:step].any(1)
        for key in ('frequency_clipped','derivative_clipped','drive_clipped','residual_clipped'):frame[key+'_events']=diag[key][:,:step].sum(1)
        rows.append(frame)
        for scope,sl in [('prefix',slice(0,step+1))]+([('last1s',slice(200,251))] if step==250 else []):
            a=np.linalg.norm(pred['angular_velocity_b'][:,sl].std(1),axis=-1);b=np.linalg.norm(truth.angular_velocity_b[:,sl].std(1),axis=-1)
            ratio=a/np.maximum(b,1e-9)
            variation.append(dict(experiment=name,horizon_s=horizon,scope=scope,median=np.median(ratio),p10=np.quantile(ratio,.1),p90=np.quantile(ratio,.9)))
    frame=pd.concat(rows);frame.to_csv(out/'per_rollout.csv',index=False)
    aggregate(frame,['experiment','horizon_s'],list(err)).to_csv(out/'per_horizon.csv',index=False)
    aggregate(frame,['experiment','log_id','horizon_s'],list(err)).to_csv(out/'per_flight.csv',index=False)
    pd.DataFrame(growth).to_csv(out/'error_growth.csv',index=False);pd.DataFrame(variation).to_csv(out/'variation.csv',index=False)
    final=frame[frame.horizon_s==5]
    (out/'summary.json').write_text(json.dumps(dict(experiment=name,n_origins=len(windows),numeric_failures=int(final.numerical_failed.sum()),support_failures=int(final.support_failed.sum()),clipping_failures=int(final.clipping_failed.sum())),indent=2))


def teacher_metrics(sim,batch,metadata,name,device,out):
    chunks={};dt=batch.trajectory.dt_s;truth=batch.trajectory.truth
    with torch.no_grad():
        for start in range(0,len(metadata),2048):
            ids=np.arange(start,min(start+2048,len(metadata)));s=make_initial(sim,batch,ids,'warm26',device)
            ns,d=sim.step(s,t(batch.trajectory.controls[ids,0],device),t(dt[ids,0],device))
            for key in PHYSICAL_FIELDS:chunks.setdefault(key,[]).append(npy(getattr(ns,key)))
    pred={key:np.stack((getattr(truth,key)[:,0],np.concatenate(values)),1) for key,values in chunks.items()}
    errors=endpoint_errors(pred,truth,1);rows=[]
    for logid in metadata.log_id.unique():
        sel=metadata.log_id.to_numpy()==logid
        for key,v in errors.items():rows.append(dict(experiment=name,log_id=logid,metric=key,n=len(v[sel]),rmse=np.sqrt(np.mean(v[sel]**2))))
    pd.DataFrame(rows).to_csv(out/'teacher_one_step.csv',index=False)


def perturbations(sim,batch,windows,name,partition,device,out):
    ids=select_rows(windows);w=windows.loc[ids].reset_index(drop=True)
    with torch.no_grad():
        initial=make_initial(sim,batch,ids,'warm26',device);commands=t(batch.trajectory.controls[ids,:20],device);dt=t(batch.trajectory.dt_s[ids,:20],device)
        nominal=[initial];s=initial
        for k in range(20):s,_=sim.step(s,commands[:,k],dt[:,k]);nominal.append(s)
        variants=[]
        for frac in [-.1,-.05,.05,.1]:
            delta=initial.position_n.new_zeros((len(ids),14));delta[:,3:6]=frac*initial.velocity_n/2
            variants.append((f'velocity_{frac:+.2f}',delta))
        for kind,start,angle,scale in [('rate',9,np.deg2rad(5),2.),('attitude',6,np.deg2rad(2),.35)]:
            for axis in range(3):
                for sign in [-1,1]:
                    delta=initial.position_n.new_zeros((len(ids),14));delta[:,start+axis]=sign*angle/scale
                    variants.append((f'{kind}_{axis}_{sign:+}',delta))
        rows=[]
        for label,delta in variants:
            s=perturb_state(initial,delta);start_distance=torch.linalg.vector_norm(delta[:,3:12],dim=-1)
            for k in range(21):
                if k:s,_=sim.step(s,commands[:,k-1],dt[:,k-1])
                diff=state_difference(s,nominal[k]);distance=npy(torch.linalg.vector_norm(diff[:,3:12],dim=-1))
                ev=np.linalg.norm(npy(s.velocity_n)-batch.trajectory.truth.velocity_n[ids,k],axis=1)
                ew=np.linalg.norm(npy(s.angular_velocity_b)-batch.trajectory.truth.angular_velocity_b[ids,k],axis=1)
                eq=attitude_error_deg(npy(s.quaternion_nb),batch.trajectory.truth.quaternion_nb[ids,k])
                nv=np.linalg.norm(npy(nominal[k].velocity_n)-batch.trajectory.truth.velocity_n[ids,k],axis=1)
                nw=np.linalg.norm(npy(nominal[k].angular_velocity_b)-batch.trajectory.truth.angular_velocity_b[ids,k],axis=1)
                nq=attitude_error_deg(npy(nominal[k].quaternion_nb),batch.trajectory.truth.quaternion_nb[ids,k])
                for j,r in enumerate(w.itertuples()):rows.append(dict(experiment=name,partition=partition,window_id=r.window_id,log_id=r.log_id,
                    perturbation=label,step=k,actual_duration_s=float(dt[j,:k].sum()),initial_chart_norm=float(start_distance[j]),
                    separation_norm=float(distance[j]),amplification=float(distance[j]/max(float(start_distance[j]),1e-9)),
                    velocity_error=ev[j],attitude_error=eq[j],body_rate_error=ew[j],nominal_velocity_error=nv[j],nominal_attitude_error=nq[j],nominal_body_rate_error=nw[j]))
    pd.DataFrame(rows).to_csv(out/f'perturbation_{partition}.csv',index=False)
    w.to_csv(out/f'operating_origins_{partition}.csv',index=False)
    return ids


def jacobians(sim,batch,windows,ids,name,partition,device,out,pred=None):
    rows=[];matrices=[]
    for step in ([0,25,50,100,249] if pred is not None else [0]):
        with torch.no_grad():
            if step==0:s=make_initial(sim,batch,ids,'warm26',device)
            else:s=SimulatorState(**{f.name:t(pred[f.name][ids,step],device) for f in fields(SimulatorState)})
        command=t(batch.trajectory.controls[ids,step],device);dt=t(batch.trajectory.dt_s[ids,step],device)
        j=npy(physical_jacobian(sim,s,command,dt,epsilon=1e-3));j2=npy(physical_jacobian(sim,s,command,dt,epsilon=2e-3))
        for n,i in enumerate(ids):
            r=windows.iloc[i];metrics=jacobian_spectra(j[n]);ref=jacobian_spectra(j2[n])
            rows.append(dict(experiment=name,partition=partition,window_id=r.window_id,log_id=r.log_id,step=step,
                actual_duration_s=batch.trajectory.dt_s[i,:step].sum(),dt_s=batch.trajectory.dt_s[i,step],
                epsilon=.001,epsilon_control=.002,epsilon_max_matrix_difference=np.abs(j[n]-j2[n]).max(),
                singular_epsilon_difference=abs(metrics['largest_singular']-ref['largest_singular']),
                radius_epsilon_difference=abs(metrics['spectral_radius']-ref['spectral_radius']),**metrics))
        matrices.append(j)
    pd.DataFrame(rows).to_csv(out/f'jacobian_{partition}.csv',index=False)
    np.savez_compressed(out/f'jacobian_{partition}.npz',jacobians=np.stack(matrices),scales=CHART_SCALES)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--artifacts',type=Path,required=True);ap.add_argument('--device',default='cuda:1');ap.add_argument('--experiment');args=ap.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    protocol=json.loads((args.results/'protocol.json').read_text());names=[e['name'] for e in protocol['experiments']]+['Step5_S1']
    if args.experiment:names=[args.experiment]
    if all((args.results/name/'evaluation_complete.json').exists() for name in names):
        for name in names:
            checkpoint=(ROOT/'artifacts/main_v2_increment_supervision/S1/model.pt') if name=='Step5_S1' else args.artifacts/name/'model.pt'
            completed=json.loads((args.results/name/'evaluation_complete.json').read_text())
            if completed['checkpoint_sha256']!=sha(checkpoint):raise ValueError('completed evaluation checkpoint changed')
            if completed['origins_sha256']!=sha(OLD/'main_v2_free_running_5s/windows.csv'):raise ValueError('completed origins changed')
        return
    validation=pd.read_parquet(DATA/'samples_validation.parquet');train=pd.read_parquet(DATA/'samples_train.parquet')
    windows=pd.read_csv(OLD/'main_v2_free_running_5s/windows.csv');assert len(windows)==722
    assert sha(OLD/'main_v2_free_running_5s/windows.csv')==protocol['frozen_validation_windows_sha256']
    batch=assemble_history_trajectory_windows(validation,windows,history_steps=26)
    tw=pd.read_parquet(DATA/'windows_train.parquet');tw=tw.loc[select_rows(tw)].reset_index(drop=True)
    tb=assemble_history_trajectory_windows(train,tw,history_steps=26)
    meta=pd.read_csv(OLD/'main_v2_dynamics_observability_phase/validation_points.csv')
    local=meta[['log_id','segment_id','sample_in_segment']].rename(columns={'sample_in_segment':'start_sample_in_segment'}).copy();local['state_sample_count']=2
    local['window_id']=[f'{r.log_id}:{r.segment_id}:{r.start_sample_in_segment}' for r in local.itertuples()]
    lb=assemble_history_trajectory_windows(validation,local,history_steps=26)
    env=json.loads((OLD/'main_v2_free_running_5s/observed_envelope.json').read_text())['train']
    for name in names:
        out=args.results/name
        if (out/'evaluation_complete.json').exists():continue
        out.mkdir(exist_ok=True)
        checkpoint=(ROOT/'artifacts/main_v2_increment_supervision/S1/model.pt') if name=='Step5_S1' else args.artifacts/name/'model.pt'
        sim=load_simulator(BASE,args.device);sim.model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict'])
        print('EVALUATE',name,flush=True)
        if name=='Step5_S1':
            z=np.load(ROOT/'artifacts/main_v2_increment_supervision/S1/validation_rollout.npz')
            assert np.array_equal(z['window_ids'],windows.window_id.to_numpy(str))
            pred={f.name:z[f.name] for f in fields(SimulatorState)};diag={key:z[key] for key in ('acceleration_b','angular_acceleration_b','frequency_clipped','derivative_clipped','drive_clipped','residual_clipped')}
        else:
            chunks=[];diags=[]
            with torch.no_grad():
                for start in range(0,722,128):
                    ids=np.arange(start,min(start+128,722));p,d=rollout(sim,make_initial(sim,batch,ids,'warm26',args.device),t(batch.trajectory.controls[ids],args.device),t(batch.trajectory.dt_s[ids],args.device));chunks.append(p);diags.append(d)
            pred={key:np.concatenate([c[key] for c in chunks]) for key in chunks[0]};diag={key:np.concatenate([d[key] for d in diags]) for key in diags[0]}
            np.savez_compressed(args.artifacts/name/'validation_rollout.npz',**pred,**diag,dt_s=batch.trajectory.dt_s,window_ids=windows.window_id.to_numpy(str))
        metric_tables(name,pred,diag,batch,windows,env,out)
        teacher_metrics(sim,lb,meta,name,args.device,out)
        vi=perturbations(sim,batch,windows,name,'validation',args.device,out)
        ti=perturbations(sim,tb,tw,name,'train',args.device,out)
        jacobians(sim,batch,windows,vi,name,'validation',args.device,out,pred)
        jacobians(sim,tb,tw,ti,name,'train',args.device,out)
        if name=='S0':
            reference=np.load(ROOT/'artifacts/main_v2_increment_supervision/S0/validation_rollout.npz')
            difference={key:float(np.max(np.abs(pred[key]-reference[key]))) for key in pred}
            for key in PHYSICAL_FIELDS:np.testing.assert_allclose(pred[key],reference[key],atol=2e-4,rtol=2e-4)
            path=args.results/'baseline_reproduction.json';v=json.loads(path.read_text());v.update(validation_benchmark='722 common origins passed',benchmark_max_absolute_difference=difference);path.write_text(json.dumps(v,indent=2))
        (out/'evaluation_complete.json').write_text(json.dumps(dict(checkpoint_sha256=sha(checkpoint),origins_sha256=sha(OLD/'main_v2_free_running_5s/windows.csv'),sealed_test_opened=False),indent=2))

if __name__=='__main__':main()
