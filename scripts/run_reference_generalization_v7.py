"""Frozen v7 audit: flight shift, phase extrapolation, nested train-flight selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from reference_capacity_v6 import model_from
from evaluate_control_windows_v2 import error_rows, summarize, METRICS
from reference_generalization_v7 import (CANDIDATES, phase_prediction, phase_loss,
    choose_candidate, scheduled_rollout, held_start_schedule)

OUT = base.ROOT/'docs/analysis/results/control_windows_v7'
V6 = OUT.parent/'control_windows_v6'
PHASE = OUT.parent/'control_windows_v4/diagnostic_future_phase.npz'


def verify_manifest(path, groups):
    manifest = json.loads(path.read_text())
    for group in groups:
        for rel, digest in manifest[group].items():
            assert base.sha(base.ROOT/rel) == digest, rel


def freeze():
    OUT.mkdir(exist_ok=False)
    pins = [base.DATA/'manifest.json', base.DATA/'windows_v2.csv',
        base.ROOT/'dataset/control_windows_v2/windows.npz',
        base.ROOT/'dataset/control_windows_v2/normalization.npz', PHASE,
        V6/'delivery_manifest.json', Path(__file__),
        base.ROOT/'scripts/reference_generalization_v7.py',
        base.ROOT/'tests/test_reference_generalization_v7.py']
    protocol = dict(stage='flight/condition generalization and phase extrapolation diagnosis',
        authority='control_windows_v2/manifest.json; same407train739developmentvalidation; no screening or split changes',
        phase_contract='September logged relative wing phase; physical zero unconfirmed; ratio7.909090995788574, not April canonical ratio8',
        data='50Hz H26/K50; future phase only supplies training forecast labels or explicitly marked offline oracle',
        source_pins={str(p.relative_to(base.ROOT)):base.sha(p) for p in pins},
        stratification='all4 v6 arms x3seeds; all windows retained; mode, quality, initial frequency/speed/history-rate-RMS; terciles fit training only; flight means before cohort means',
        support='descriptive training1/99percentile ranges of initial/history covariates, not safe envelope; no response-based selection',
        phase_candidates=CANDIDATES,
        candidate_details='hold current frequency; mean last5; phase slope6/11/26; damped frequency trend11 withtau.15; train-flight-balanced ridge residual trajectory alpha10/100 on15history features',
        selection='nested leave-one-training-flight-out: choose on inner LOO of4flights then evaluate fifth; final fixed candidate chosen by full5flightLOO. No validation labels for selection.',
        phase_primary='mean across flights of sqrt(mean harmonic1/2 chord squared error over full1s)); fixed order breaks ties',
        phase_support_gate='nestedOOF>=10percent improvement overhold, >=4of5flights positive, noflight>5percent worse',
        frozen_model='A1600 integrated3seeds, exact original weights and normalization. Replay originalphase, selected causalphase and oracleloggedphase, keep frequency context unchanged to isolate phase schedule. No retraining or optimizer.',
        inference='start-of-step phase0 then forecast/logged phase atsteps1..49; hold replay must match original inference; each timestep sees current/past planned references only',
        candidate_evaluation='evaluate final train-selected phase model once on development validation even if phase gate fails, as diagnosis; no promotion without gates',
        state_gate='activep/q .5/1s >=5percent flight-equal improvement, all3seedspositive, allmetricregression<=5percent; absolute .5s p/q<=.15rad/s attitude<=3deg velocity<=.5m/s eachseed',
        causality_limit='phase intervention changes frozen decoder input distribution; oracle improvement does not identify rootcause or achievable bound. References/response and independent closedloop gates remain required.',
        validation_status='previously viewed development data, not blind test',sealed_test_accessed=False)
    base.write(OUT/'protocol.json', protocol)


def primary_comparison(frame, candidate):
    v = frame[(frame.partition=='validation') & frame.horizon_s.isin([.5, 1.])]
    flights = sorted(v.log_id.unique())
    cube = np.full((2, 3, len(flights), 4), np.nan)
    for ai, arm in enumerate(['hold', candidate]):
        for si, seed in enumerate([17, 23, 42]):
            for fi, log in enumerate(flights):
                for ki, (subset, axis, h) in enumerate([(s, ax, t) for s, ax in [('roll_active', 'p'), ('pitch_active', 'q')] for t in [.5, 1.]]):
                    z = v[(v.arm==arm)&(v.seed==seed)&(v.log_id==log)&(v.subset==subset)&(v.horizon_s==h)]
                    if len(z): cube[ai, si, fi, ki] = z.iloc[0][axis]
    scores = np.nanmean(np.nanmean(cube, 2), 2)
    gains = 1-scores[1]/scores[0]
    rng = np.random.default_rng(907); boot = []
    for _ in range(2000):
        ix = rng.integers(len(flights), size=len(flights))
        s = np.nanmean(np.nanmean(cube[:, :, ix], 2), axis=(1, 2)); boot.append(1-s[1]/s[0])
    allm = v[v.subset=='all'].groupby(['arm','seed','horizon_s'])[METRICS].mean()
    reg = allm.loc[candidate]/allm.loc['hold']-1
    gain = float(1-scores[1].mean()/scores[0].mean())
    absolute = []
    for seed in [17, 23, 42]:
        a = v[(v.arm==candidate)&(v.seed==seed)&(v.horizon_s==.5)]
        p, q = float(a[a.subset=='roll_active'].p.mean()), float(a[a.subset=='pitch_active'].q.mean())
        other = a[a.subset=='all'][['attitude','velocity']].mean()
        absolute.append(dict(seed=seed, p=p, q=q, attitude_deg=float(np.rad2deg(other.attitude)),
            velocity=float(other.velocity), passed=bool(p<=.15 and q<=.15 and other.attitude<=np.deg2rad(3) and other.velocity<=.5)))
    return dict(candidate=candidate, primary_gain=gain, seed_gains=gains.tolist(),
        bootstrap95=np.quantile(boot,[.025,.975]).tolist(), maximum_regression=float(reg.to_numpy().max()),
        relative_pass=bool(gain>=.05 and (gains>0).all() and (reg<=.05).all().all()), absolute=absolute)


def audit_cohorts(x, y, w, phase):
    h = x['history_state'].numpy()
    cov = w[['array_index','log_id','partition','mode','quality']].copy()
    cov['frequency'] = h[:, -1, 11]
    cov['speed'] = np.linalg.norm(h[:, -1, :3], axis=1)
    cov['history_rate_rms'] = np.sqrt(np.mean(h[:, :, 3:6]**2, axis=(1,2)))
    cov['roll_reference'] = x['history_reference'][:, -1, 0].numpy()
    cov['pitch_reference'] = x['history_reference'][:, -1, 1].numpy()
    cov['history_actuator_rms'] = np.sqrt(np.mean(x['history_actuator'].numpy()**2, axis=(1,2)))
    tr = w.partition.to_numpy()=='train'; bins = {}; support = []
    for c in ['frequency','speed','history_rate_rms','roll_reference','pitch_reference','history_actuator_rms']:
        edges = np.quantile(cov.loc[tr,c], [1/3,2/3]); bins[c] = edges.tolist()
        cov[c+'_bin'] = np.searchsorted(edges, cov[c].to_numpy(), side='right')
        low, high = np.quantile(cov.loc[tr,c], [.01,.99])
        cov[c+'_outside'] = (cov[c]<low)|(cov[c]>high)
        for part, g in cov.groupby('partition'):
            support.append(dict(covariate=c, partition=part, low=low, high=high,
                outside_fraction=float(g[c+'_outside'].mean()), median=float(g[c].median())))
    cov.to_csv(OUT/'window_covariates.csv', index=False)
    pd.DataFrame(support).to_csv(OUT/'support_summary.csv',index=False)
    base.write(OUT/'training_bins.json',bins)
    all_errors=[]
    for arm in ['integrated','direct']:
        for seed in [17,23,42]:
            for step in [1600,6400]:
                for part in ['train','validation']:
                    meta=w[w.partition==part];ix=meta.array_index.to_numpy()
                    with np.load(V6/'formal'/f'{arm}_{seed}'/f'{step}_{part}_predictions.npz') as z:
                        np.testing.assert_array_equal(z['array_index'],ix)
                        pred={k:z[k] for k in y}
                    e=error_rows(pred,{k:v[ix].numpy() for k,v in y.items()},meta,f'{arm}{step}',seed)
                    all_errors.append(e)
    err=pd.concat(all_errors,ignore_index=True)
    err.to_csv(OUT/'frozen_per_window_squared_errors.csv',index=False)
    group=['arm','seed','partition','log_id','horizon_s']
    pf=np.sqrt(err.groupby(group)[METRICS].mean()).reset_index()
    pf.groupby(['arm','partition','log_id','horizon_s'])[METRICS].mean().to_csv(OUT/'frozen_per_flight.csv')
    joined=err.merge(cov.drop(columns=['log_id','partition','mode','quality']),on='array_index',validate='many_to_one')
    summaries=[]
    for c in ['mode','quality','frequency_bin','speed_bin','history_rate_rms_bin']:
        g=np.sqrt(joined.groupby(group+[c])[METRICS].mean()).reset_index()
        g=g.groupby(['arm','partition','horizon_s',c])[METRICS].mean().reset_index().rename(columns={c:'stratum'})
        g['condition']=c;summaries.append(g)
    pd.concat(summaries).to_csv(OUT/'condition_summary.csv',index=False)
    # Descriptive phase/state association, clustered within each flight; no causal attribution.
    hp,_=phase_prediction('hold',h); loss=phase_loss(hp,phase)
    a=err[(err.arm=='integrated1600')&(err.horizon_s==1.)].groupby('array_index')[['p','q']].mean()
    a=a.join(cov.set_index('array_index')[['partition','log_id']]);a['phase_loss']=loss[a.index]
    rows=[]
    for (part,log),g in a.groupby(['partition','log_id']):
        for ax in ['p','q']:
            rows.append(dict(partition=part,log_id=log,axis=ax,n=len(g),
                spearman=float(g[['phase_loss',ax]].corr(method='spearman').iloc[0,1])))
    pd.DataFrame(rows).to_csv(OUT/'phase_error_association.csv',index=False)


def select_phase(x,w,phase):
    h=x['history_state'].numpy();tr=np.flatnonzero(w.partition.to_numpy()=='train');logs=w.log_id.to_numpy()
    nested=np.full((len(tr),50),np.nan); decisions=[]
    for log in sorted(set(logs[tr])):
        fit=tr[logs[tr]!=log];qi=np.flatnonzero(logs[tr]==log);query=tr[qi]
        winner,scores=choose_candidate(h[fit],phase[fit],logs[fit])
        pp,params=phase_prediction(winner,h[query],h[fit],phase[fit],logs[fit]);nested[qi]=pp
        np.savez_compressed(OUT/f'phase_outer_fold{len(decisions)}.npz',fit_indices=fit,query_indices=query,prediction=pp,**params)
        hold,_=phase_prediction('hold',h[query])
        b=float(np.sqrt(phase_loss(hold,phase[query]).mean()));c=float(np.sqrt(phase_loss(pp,phase[query]).mean()))
        decisions.append(dict(log_id=log,winner=winner,inner_scores=scores,hold=b,candidate=c,gain=1-c/b))
    winner,scores=choose_candidate(h[tr],phase[tr],logs[tr])
    # Freeze the selection and nested-OOF gate BEFORE evaluating validation labels.
    gains=np.array([d['gain'] for d in decisions]);mean_gain=1-np.mean([d['candidate'] for d in decisions])/np.mean([d['hold'] for d in decisions])
    gate=bool(mean_gain>=.10 and (gains>0).sum()>=4 and (gains>=-.05).all())
    selection=dict(winner=winner,full_train_loo_scores=scores,outer_folds=decisions,nested_gain=float(mean_gain),phase_gate=gate,
        validation_used_for_selection=False,selection_scope='phase forecasting only; not cross-validated GRU state training')
    base.write(OUT/'phase_selection.json',selection)
    pp,params=phase_prediction(winner,h,h[tr],phase[tr],logs[tr]);np.savez_compressed(OUT/'phase_candidate.npz',array_index=w.array_index,prediction=pp,**params)
    np.savez_compressed(OUT/'nested_oof_phase.npz',array_index=tr,prediction=nested)
    rows=[]
    for name,pred in [('hold',phase_prediction('hold',h)[0]),('selected',pp)]:
        for part in ['train','validation']:
            for log in sorted(set(logs[w.partition==part])):
                ix=np.flatnonzero(logs==log)
                delta=np.angle(np.exp(1j*(pred[ix]-phase[ix])))
                for steps in [5,25,50]:
                    rows.append(dict(model=name,partition=part,log_id=log,horizon_s=steps*.02,
                        circular_rmse=float(np.sqrt(np.mean(delta[:,steps-1]**2))),
                        full_prefix_harmonic_rmse=float(np.sqrt(phase_loss(pred[ix,:steps],phase[ix,:steps]).mean()))))
    pd.DataFrame(rows).to_csv(OUT/'phase_per_flight.csv',index=False)
    return pp,selection


def replay(x,y,w,phase,candidate):
    frames=[];checks=[]
    for seed in [17,23,42]:
        model=model_from(V6/'formal'/f'integrated_{seed}'/'step_1600.pt')
        for part in ['train','validation']:
            meta=w[w.partition==part];ix=meta.array_index.to_numpy();truth={k:v[ix].numpy() for k,v in y.items()}
            for arm,end_phase in [('hold',None),('selected',candidate),('oracle',phase)]:
                outputs={k:[] for k in y}
                for start in range(0,len(ix),64):
                    ids=ix[start:start+64];xx={k:v[ids] for k,v in x.items()}
                    if arm=='hold':schedule=held_start_schedule(xx)
                    else:schedule=torch.as_tensor(np.column_stack([np.zeros(len(ids)),end_phase[ids,:-1]]),dtype=torch.float32)
                    pred=scheduled_rollout(model,xx,schedule)
                    for k in outputs:outputs[k].append(pred[k].numpy())
                pred={k:np.concatenate(v) for k,v in outputs.items()}
                np.savez_compressed(OUT/f'{arm}_{seed}_{part}.npz',array_index=ix,**pred)
                if arm=='hold':
                    with np.load(V6/'formal'/f'integrated_{seed}'/f'1600_{part}_predictions.npz') as z:
                        error=max(float(np.max(abs(pred[k]-z[k]))) for k in y)
                    assert error<1e-5, error
                    checks.append(dict(seed=seed,partition=part,replay_max_abs=error))
                f=summarize(error_rows(pred,truth,meta,arm,seed));f['partition']=part;frames.append(f)
    frame=pd.concat(frames);frame.to_csv(OUT/'replay_per_flight.csv',index=False)
    frame.groupby(['arm','partition','subset','horizon_s'])[METRICS].mean().to_csv(OUT/'replay_summary.csv')
    comparisons=[primary_comparison(frame,c) for c in ['selected','oracle']]
    base.write(OUT/'replay_decision.json',dict(comparisons=comparisons,control_validated=False,controller_optimization_started=False,
        oracle='nondeployable diagnostic; neither independent response truth nor achievable bound'))
    base.write(OUT/'verification.json',dict(held_replay=checks,sealed_test_accessed=False,phase_future_used_only_as_training_label_or_oracle=True))


def main():
    torch.set_num_threads(1)
    freeze()
    verify_manifest(V6/'delivery_manifest.json',['artifacts','code','source_pins'])
    parent=json.loads((PHASE.parent/'delivery_manifest.json').read_text());assert base.sha(PHASE)==parent['artifacts'][str(PHASE.relative_to(base.ROOT))]
    x,y,_=base.load();w=pd.read_csv(base.DATA/'windows_v2.csv')
    with np.load(PHASE) as z:phase=z['phase'];np.testing.assert_array_equal(z['array_index'],w.array_index)
    base.write(OUT/'status.json',dict(status='running',stage='nested train-flight phase selection'))
    candidate,selection=select_phase(x,w,phase)
    print(json.dumps(selection,ensure_ascii=False),flush=True)
    base.write(OUT/'status.json',dict(status='running',stage='flight/condition error audit'))
    audit_cohorts(x,y,w,phase)
    base.write(OUT/'status.json',dict(status='running',stage='frozen state replay'))
    replay(x,y,w,phase,candidate)
    base.write(OUT/'status.json',dict(status='complete',sealed_test_accessed=False))

if __name__=='__main__':main()
