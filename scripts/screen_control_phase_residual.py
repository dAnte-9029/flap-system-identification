"""Past-only periodic-rate residual information screen, with train-only selection."""
from pathlib import Path
import numpy as np
import pandas as pd

import run_paper_rollout_horizon_ablation as source
from test_control_dynamics_residual_information import fit_gain

m=source.m
BASE=m.ROOT/'docs/analysis/results/control_dynamics_realdata_v1'
OUT=BASE/'phase_residual_screen'


def harmonic_increment(history,phase_delta,harmonics):
    """Fit current/past rates to phase harmonics and trend; return zero-at-t0 change.

    history uses inherited relative-to-origin sin/cos. No physical phase-zero claim.
    Normalized sample-index trend is a nuisance covariate, not a physical clock.
    """
    n,h,_=history.shape
    phi=np.arctan2(history[:,:,9],history[:,:,10])
    terms=[np.ones((n,h)),np.broadcast_to(np.linspace(-1,0,h),(n,h))]
    for k in range(1,harmonics+1):terms.extend([np.sin(k*phi),np.cos(k*phi)])
    design=np.stack(terms,-1)
    penalty=np.eye(design.shape[-1])*.01*h;penalty[0,0]=0
    coef=np.linalg.solve(np.einsum('nhi,nhj->nij',design,design)+penalty,
        np.einsum('nhi,nhj->nij',design,history[:,:,3:6]))
    future=[]
    for k in range(1,harmonics+1):future.extend([np.sin(k*phase_delta),np.cos(k*phase_delta)-1])
    return np.einsum('nti,nij->ntj',np.stack(future,-1),coef[:,2:])


def run():
    OUT.mkdir(parents=True,exist_ok=False);m.configure()
    for name in ['ensemble_candidate','past_residual_information']:
        m.verify_pins(m.read_json(BASE/name/'completion.json')['outputs'])
    ep=m.read_json(BASE/'ensemble_candidate/protocol.json');m.verify_pins(ep['pins'])
    sp,batches,_=source.inputs();fold_path=BASE/'local_identification/protocol.json'
    folds=m.read_json(fold_path)['folds']
    pins=dict(ep['pins'])
    for p in [Path(__file__),fold_path,BASE/'past_residual_information/train_predictions.npz',BASE/'ensemble_candidate/predictions.npz',
        m.ROOT/'scripts/test_control_dynamics_residual_information.py']:
        pins[m.rel(p)]=m.file_hash(p)
    m.write_json(OUT/'protocol.json',dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),pins=pins,
        experiment='explicit_past_harmonic_residual',heldout_accessed=False,base_model_training=False,
        hypothesis='phase-locked components inferred from past observations explain future short-horizon rate residuals better than persistent bias',
        observed='all26 past/current feature rows, causal mask complete; fit rawp/q/r against intercept,normalized sample-index trend and1or2 relative phase harmonics',
        fit='per-origin6-or4-column ridge0.01*26, intercept unpenalized; no future observations',
        future_phase='frozen ensemble predicted phase minus current phase; NEVER future logged phase/frequency',
        correction='diagnostic endpoint increment alpha_axis*(harmonic_prediction(t)-harmonic_prediction(t0)), no initial-state jump; alpha clipped[-1,1]',
        alpha='single per-axis coefficient fit jointly to100/200ms residuals, equal horizons and training flights; coefficients five-fold cross-fitted on train only',
        base_oof_caveat='base weights saw all train flights, only residual gain cross-fitted',
        selection='choose1or2 harmonics ONLY by trainingOOF meanq100/200ms; requires>=5% improvement,>=60%train flights improve, no p/r horizon>1% regression; otherwise retain baseline',
        validation='selected arm evaluated once on original full validation; all diagnostic arms retained without reselection',
        gate_scope='information screen only, not an integrated dynamics/control candidate; do not substitute corrected rate without re-integrating attitude and state feedback'))
    data={}
    for split in ['train','validation']:
        b=batches[split];assert b.history_mask.all()
        path=BASE/('past_residual_information/train_predictions.npz' if split=='train' else 'ensemble_candidate/predictions.npz')
        with np.load(path,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['window_ids'],b.trajectory.window_ids)
            phase=z['relative_phase_rad'];rate=z['angular_velocity_b']
        delta=phase-phase[:,:1]
        data[split]=dict(error=b.trajectory.truth.angular_velocity_b-rate,
            features={arm:harmonic_increment(b.history_state_features,delta,arm) for arm in [1,2]})
    # No validation outcome is used in fit or selection below.
    ids=batches['train'].trajectory.log_ids;fold=np.array([folds[x] for x in ids])
    rows=[];coeff=[];full={};results={};proposals=[]
    def record(split,arm,variant,error,logids):
        for h in [5,10,25,50]:
            for log in sorted(set(logids)):
                mask=logids==log
                for axis,value in zip('pqr',np.sqrt(np.mean(error[mask,h]**2,0))):
                    rows.append(dict(partition=split,harmonics=arm,variant=variant,steps=h,log_id=log,axis=axis,rmse=float(value)))
    err=data['train']['error'];target=np.concatenate([err[:,5],err[:,10]])
    base_metrics=np.array([np.sqrt(np.mean(err[ids==log][:,[5,10],1]**2,axis=0)) for log in sorted(set(ids))])
    record('train_correction_oof',0,'baseline',err,ids)
    for arm in [1,2]:
        feature=data['train']['features'][arm];x=np.concatenate([feature[:,5],feature[:,10]])
        doubled_ids=np.tile(ids,2);doubled_fold=np.tile(fold,2)
        corrected=err.copy()
        for j in range(5):
            train=doubled_fold!=j;gain=fit_gain(x[train],target[train],doubled_ids[train]).clip(-1,1)
            corrected[fold==j]-=feature[fold==j]*gain
            for axis,v in zip('pqr',gain):coeff.append(dict(harmonics=arm,fold=j,axis=axis,gain=v))
        gain=fit_gain(x,target,doubled_ids).clip(-1,1);full[arm]=gain
        for axis,v in zip('pqr',gain):coeff.append(dict(harmonics=arm,fold='full_train',axis=axis,gain=v))
        record('train_correction_oof',arm,'harmonic_diagnostic',corrected,ids)
        scores=np.array([np.sqrt(np.mean(corrected[ids==log][:,[5,10],1]**2,axis=0)) for log in sorted(set(ids))])
        improvement=float(1-scores.mean()/base_metrics.mean());fraction=float((scores.mean(1)<base_metrics.mean(1)).mean())
        guard=True
        for h in [5,10]:
            for axis in [0,2]:
                old=np.mean([np.sqrt(np.mean(err[ids==log,h,axis]**2)) for log in sorted(set(ids))])
                new=np.mean([np.sqrt(np.mean(corrected[ids==log,h,axis]**2)) for log in sorted(set(ids))])
                guard&=new<=1.01*old
        proposals.append(dict(harmonics=arm,improvement=improvement,improved_flight_fraction=fraction,nonregression=bool(guard),
            eligible=bool(improvement>=.05 and fraction>=.6 and guard)))
    eligible=[p for p in proposals if p['eligible']]
    selected=max(eligible,key=lambda p:p['improvement'])['harmonics'] if eligible else 0
    m.write_json(OUT/'selection.json',dict(selected_harmonics=selected,candidates=proposals,validation_used_for_selection=False))
    val=batches['validation'];err=data['validation']['error']
    record('validation',0,'baseline',err,val.trajectory.log_ids)
    for arm in [1,2]:record('validation',arm,'harmonic_diagnostic',err-data['validation']['features'][arm]*full[arm],val.trajectory.log_ids)
    f=pd.DataFrame(rows);f.to_csv(OUT/'per_flight.csv',index=False)
    f.groupby(['partition','harmonics','variant','steps','axis']).rmse.mean().reset_index().to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(coeff).to_csv(OUT/'coefficients.csv',index=False)
    m.verify_pins(pins)
    m.write_json(OUT/'completion.json',dict(status='complete',heldout_accessed=False,selected_harmonics=selected,
        integrated_candidate=False,outputs={m.rel(p):m.file_hash(p) for p in OUT.iterdir() if p.is_file()}))


if __name__=='__main__':run()
