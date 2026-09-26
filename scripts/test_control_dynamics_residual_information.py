"""Causal past-forecast residual screen and train-only empirical error envelopes.

The endpoint correction is an information diagnostic, not an integrated plant.
"""
from pathlib import Path
import copy
import numpy as np
import pandas as pd
import torch

import run_paper_rollout_horizon_ablation as source
from system_identification.integration.control_dynamics_candidate import load_candidate
from system_identification.training.trajectory_main_v1 import assemble_history_trajectory_windows,predict_history_trajectory_model

m=source.m
ROOT=m.ROOT/'docs/analysis/results/control_dynamics_realdata_v1'
OUT=ROOT/'past_residual_information'
AXES=('vx','vy','vz','p','q','r')


def preceding_windows(origins):
    w=origins.copy()
    if np.any(w.start_sample_in_segment<5):raise ValueError('insufficient causal past')
    w['start_sample_in_segment']-=5
    w['state_sample_count']=6
    w['window_id']=w.window_id+':past5'
    return w.drop(columns=[c for c in w if 'timestamp' in c or c=='history_span_s'])


def axes(pred):return np.concatenate([pred.velocity_n,pred.angular_velocity_b],-1)


def fit_gain(x,y,ids):
    _,inv,count=np.unique(ids,return_inverse=True,return_counts=True)
    weight=1/count[inv];weight/=weight.sum()
    # Fixed 1% variance-relative ridge, no intercept, no validation tuning.
    xx=np.sum(weight[:,None]*x*x,0)
    return np.sum(weight[:,None]*x*y,0)/np.maximum(1.01*xx,1e-12)


def run():
    OUT.mkdir(parents=True,exist_ok=False);m.configure()
    ep=m.read_json(ROOT/'ensemble_candidate/protocol.json');m.verify_pins(ep['pins'])
    done=m.read_json(ROOT/'ensemble_candidate/completion.json');m.verify_pins(done['outputs'])
    cp=ROOT/'ensemble_candidate/model.pt';model=load_candidate(cp,done['outputs'][m.rel(cp)])
    sp,batches,stats=source.inputs()
    pins=dict(ep['pins'])
    fold_path=ROOT/'local_identification/protocol.json';folds=m.read_json(fold_path)['folds']
    for path in [Path(__file__),fold_path,cp,m.ROOT/'src/system_identification/integration/control_dynamics_candidate.py']:
        pins[m.rel(path)]=m.file_hash(path)
    m.write_json(OUT/'protocol.json',dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),pins=pins,
        experiment='causal_past_residual_information',heldout_accessed=False,base_model_training=False,
        past='H26 at t-5 with observed commands/dt through t0; residual at t0=current measured state minus old forecast; no state newer than t0 used as feature',
        padding='original earliest origins may have<26 history at t-5; standard past-only masked left padding, count reported; all original origins retained',
        forecast='current H26 candidate forecast independently initialized at t0; frozen ensemble; 5/10/25/50 steps',
        correction='endpoint diagnostic y_corrected=y_pred+alpha * residual_past * future_elapsed/past_elapsed; separate6 axes and horizons; NOT a physically integrated model or control-ready checkpoint',
        estimators=['zero_baseline','persistent_gain_clipped_0_1','signed_diagnostic'],
        fit='flight-equal 1% variance-relative diagonal ridge without intercept; same5 whole-train-flight folds; full-train fit once for validation',
        oof_boundary='ONLY correction coefficients cross-fitted; frozen base model has trained on all original train flights, so these are not OOF base predictions',
        hypothesis='persistent disturbance: primary100ms q bounded positive correction improves OOF>=5% and>=60%flights; then validation must improve both dates and high-change group without any six-axis100/200ms >1% regression',
        signed='negative/unbounded gain is diagnostic, not automatic bias-observer or control candidate',
        envelope='per axis/horizon max of train-flight95th percentile absolute baseline error; fixed raw threshold, no formal conformal/exchangeability guarantee, no counterfactual coverage claim',
        uncertainty_scope='base-model in-sample train residuals; empirical validation coverage explicitly measured; no seed-std rescaling'))
    residual={};errors={};elapsed={};preds={};checks={};padded={}
    for split in ('train','validation'):
        current=batches[split];origin=pd.read_csv(m.BASE/f'{split}_origins.csv')
        assert np.array_equal(origin.window_id,current.trajectory.window_ids)
        mp=m.ROOT/sp['manifest_path'];path=mp.parent/f'samples_{split}.parquet'
        assert m.file_hash(path)==sp['artifact_sha256'][path.name]
        samples=pd.read_parquet(path)
        old=assemble_history_trajectory_windows(samples,preceding_windows(origin),history_steps=26)
        # This old forecast ends EXACTLY at current t0, for each physical field.
        for k in vars(current.trajectory.truth):
            np.testing.assert_allclose(getattr(old.trajectory.truth,k)[:,-1],getattr(current.trajectory.truth,k)[:,0],rtol=0,atol=0)
        np.testing.assert_array_equal(old.trajectory.controls,current.history_controls[:,-6:-1])
        past=predict_history_trajectory_model(model,old,use_history=True,batch_size=128,device='cpu')
        residual[split]=axes(old.trajectory.truth)[:,-1]-axes(past)[:,-1]
        elapsed[split]=old.trajectory.dt_s.sum(1)
        padded[split]=int((~old.history_mask.all(1)).sum())
        if split=='train':
            future=predict_history_trajectory_model(model,current,use_history=True,batch_size=128,device='cpu')
            np.savez_compressed(OUT/'train_predictions.npz',**vars(future),window_ids=current.trajectory.window_ids.astype(str))
        else:
            from system_identification.models.trajectory import TrajectoryPrediction
            with np.load(ROOT/'ensemble_candidate/predictions.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(z['window_ids'],current.trajectory.window_ids)
                future=TrajectoryPrediction(**{k:z[k] for k in vars(current.trajectory.truth)})
        preds[split]=axes(future);errors[split]=axes(current.trajectory.truth)-preds[split]
        np.savez_compressed(OUT/f'{split}_past_residual.npz',residual=residual[split],elapsed=elapsed[split],window_ids=current.trajectory.window_ids.astype(str))
        # Future-label corruption cannot change either frozen current inference or old inputs.
        small=m.subset(current,3);poison=copy.deepcopy(small)
        for a in vars(poison.trajectory.truth).values():a[:,1:]=np.nan
        a=predict_history_trajectory_model(model,small,use_history=True,batch_size=3,device='cpu')
        b=predict_history_trajectory_model(model,poison,use_history=True,batch_size=3,device='cpu')
        for key in vars(a):np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
        checks[split]=dict(past_endpoint_equals_current=True,past_commands_exact=True,future_labels_invariant=True,padded_origins=padded[split])
        print('prepared',split,'origins',len(origin),'padded past histories',padded[split],flush=True)
    ids=batches['train'].trajectory.log_ids;fold=np.array([folds[log] for log in ids])
    rows=[];coeff=[];envelopes=[];coverage=[]
    from system_identification.evaluation.future_control_diagnostic import excitation
    threshold=np.quantile(excitation(batches['train'].trajectory.controls,stats['control_std'],25),.75)
    for steps in [5,10,25,50]:
        x=residual['train']*(batches['train'].trajectory.dt_s[:,:steps].sum(1)/elapsed['train'])[:,None]
        y=errors['train'][:,steps]
        cross=np.zeros_like(y);signed=np.zeros_like(y)
        for j in range(5):
            tr=fold!=j;ho=~tr;gain=fit_gain(x[tr],y[tr],ids[tr])
            cross[ho]=x[ho]*gain.clip(0,1);signed[ho]=x[ho]*gain
            for axis,g in zip(AXES,gain):coeff.append(dict(steps=steps,fold=j,axis=axis,gain=g,bounded_gain=np.clip(g,0,1)))
        gain=fit_gain(x,y,ids)
        for axis,g in zip(AXES,gain):coeff.append(dict(steps=steps,fold='full_train',axis=axis,gain=g,bounded_gain=np.clip(g,0,1)))
        # Frozen max of within-flight quantiles, to resist domination by long logs.
        bound=np.max(np.stack([np.quantile(np.abs(y[ids==log]),.95,axis=0) for log in sorted(set(ids))]),axis=0)
        for axis,v in zip(AXES,bound):envelopes.append(dict(steps=steps,axis=axis,absolute_error_bound=float(v)))
        for split in ('train','validation'):
            b=batches[split];logids=b.trajectory.log_ids;e=errors[split][:,steps]
            xx=residual[split]*(b.trajectory.dt_s[:,:steps].sum(1)/elapsed[split])[:,None]
            candidates={'zero_baseline':np.zeros_like(e),'persistent_gain_clipped_0_1':cross if split=='train' else xx*gain.clip(0,1),
                        'signed_diagnostic':signed if split=='train' else xx*gain}
            high=excitation(b.trajectory.controls,stats['control_std'],25)>threshold
            for log in sorted(set(logids)):
                for group,mask in [('ALL',logids==log),('high_change',(logids==log)&high)]:
                    if not mask.any():continue
                    for name,c in candidates.items():
                        for axis,rmse in zip(AXES,np.sqrt(np.mean((e[mask]-c[mask])**2,axis=0))):
                            rows.append(dict(partition='train_correction_oof' if split=='train' else split,steps=steps,log_id=log,group=group,variant=name,axis=axis,rmse=float(rmse),n=int(mask.sum())))
                    for axis,value in zip(AXES,np.mean(np.abs(e[mask])<=bound,axis=0)):
                        coverage.append(dict(partition=split,steps=steps,log_id=log,group=group,axis=axis,coverage=float(value),n=int(mask.sum())))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'per_flight.csv',index=False)
    summary=frame.groupby(['partition','steps','group','variant','axis']).rmse.mean().reset_index();summary.to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(coeff).to_csv(OUT/'coefficients.csv',index=False);pd.DataFrame(envelopes).to_csv(OUT/'empirical_envelope.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'envelope_coverage_per_flight.csv',index=False)
    primary=frame[(frame.partition=='train_correction_oof')&(frame.steps==5)&(frame.group=='ALL')&(frame.axis=='q')]
    p=primary.pivot(index='log_id',columns='variant',values='rmse')
    improvement=float(1-p.persistent_gain_clipped_0_1.mean()/p.zero_baseline.mean())
    fraction=float((p.persistent_gain_clipped_0_1<p.zero_baseline).mean())
    m.verify_pins(pins);m.write_json(OUT/'checks.json',checks)
    m.write_json(OUT/'completion.json',dict(status='complete',heldout_accessed=False,
        persistent_residual_train_gate=bool(improvement>=.05 and fraction>=.6),primary_relative_improvement=improvement,primary_improved_flight_fraction=fraction,
        integrated_candidate_delivered=False,envelope_is_formal_guarantee=False,
        outputs={m.rel(p):m.file_hash(p) for p in OUT.iterdir() if p.is_file()}))


if __name__=='__main__':run()
