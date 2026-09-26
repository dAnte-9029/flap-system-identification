"""Different-flight, state/history-matched logged command contrasts.

Selection uses observed past/current state and known command tapes only.
These observational pairs do not remove hidden confounding or establish causality.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import run_paper_rollout_horizon_ablation as source
from system_identification.integration.control_dynamics_candidate import command_coordinates

m=source.m
BASE=m.ROOT/'docs/analysis/results/control_dynamics_realdata_v1'
OUT=BASE/'matched_contrasts'
CHANNELS=('drive','common','differential','rudder')
AXES=('vz','q','p','r')
CALIPERS=(.25,.5,1.)


def context(batch):
    # All state lags are observed by t0. Relative phase is included at past lags.
    states=batch.history_state_features[:,[-1,-6,-11,-26]].reshape(len(batch.history_mask),-1)
    controls=command_coordinates(batch.history_controls[:,[-2,-6,-11]]).reshape(len(states),-1)
    return np.concatenate([states,controls],1)


def nonoverlapping_origins(origin,horizon):
    """Disjoint history+forecast sample intervals within each log/segment."""
    keep=[]
    for _,g in origin.groupby(['log_id','segment_id'],sort=True):
        end=-1
        for i,row in g.sort_values('start_sample_in_segment').iterrows():
            start=int(row.start_sample_in_segment)
            if start-25>end:
                keep.append(i);end=start+horizon
    return np.array(keep,dtype=int)


def current_modes(samples,origin):
    valid=samples.loc[samples.valid_core].set_index(['log_id','segment_id','sample_in_segment']).nav_state
    if not valid.index.is_unique:raise ValueError('duplicate valid sample identity')
    index=pd.MultiIndex.from_frame(origin[['log_id','segment_id','start_sample_in_segment']].rename(columns={'start_sample_in_segment':'sample_in_segment'}))
    result=valid.reindex(index).to_numpy()
    if not np.isfinite(result).all():raise ValueError('missing current valid mode')
    return result


def choose_pairs(z,commands,dose,elapsed,logids,modes,dose_min,command_scale,channel,caliper):
    """Greedy nonreused pairs; no labels/predictions are accepted by this function."""
    candidates=[]
    for mode in sorted(set(modes)):
        idx=np.flatnonzero(modes==mode)
        if len(idx)<2:continue
        tree=cKDTree(z[idx]);dist,near=tree.query(z[idx],k=min(65,len(idx)))
        for row,i in enumerate(idx):
            for distance,local in zip(np.atleast_1d(dist[row]),np.atleast_1d(near[row])):
                j=idx[int(local)]
                if i>=j or logids[i]==logids[j]:continue
                distance=float(distance/np.sqrt(z.shape[1]))
                if distance>caliper:continue
                contrast=float(dose[i,channel]-dose[j,channel])
                if abs(contrast)<dose_min:continue
                other=[k for k in range(4) if k!=channel]
                differences=np.sqrt(np.mean(((commands[i,:,other]-commands[j,:,other])/command_scale[other,None])**2,axis=1))
                if np.max(differences)>.25:continue
                if abs(elapsed[i]-elapsed[j])>.05*(elapsed[i]+elapsed[j])/2:continue
                high,low=(i,j) if contrast>0 else (j,i)
                candidates.append((distance,high,low,float(abs(contrast)),float(np.max(differences))))
    chosen=[];used=set()
    for row in sorted(candidates):
        _,hi,lo,_,_=row
        if hi in used or lo in used:continue
        used.update([hi,lo]);chosen.append(row)
    return chosen


def run():
    OUT.mkdir(parents=True,exist_ok=False);m.configure()
    for name in ['ensemble_candidate','past_residual_information']:
        c=m.read_json(BASE/name/'completion.json');m.verify_pins(c['outputs'])
    ep=m.read_json(BASE/'ensemble_candidate/protocol.json');m.verify_pins(ep['pins'])
    sp,batches,_=source.inputs();train=batches['train']
    ctx=context(train);mean=ctx.mean(0);scale=ctx.std(0).clip(1e-5)
    command_scale=command_coordinates(train.trajectory.controls.reshape(-1,4)).std(0).clip(1e-5)
    dose_min={};floors={}
    for h in [5,10,25]:
        c=command_coordinates(train.trajectory.controls[:,:h]);w=train.trajectory.dt_s[:,:h]
        dose=(c*w[:,:,None]).sum(1)/w.sum(1)[:,None]-command_coordinates(train.history_controls[:,-2])
        dose_min[h]=np.maximum(np.quantile(dose,.75,axis=0)-np.quantile(dose,.25,axis=0),.1*command_scale)
        q=train.trajectory.truth
        target=np.column_stack([q.velocity_n[:,h,2]-q.velocity_n[:,0,2],q.angular_velocity_b[:,h,1]-q.angular_velocity_b[:,0,1],
            q.angular_velocity_b[:,h,0]-q.angular_velocity_b[:,0,0],q.angular_velocity_b[:,h,2]-q.angular_velocity_b[:,0,2]])
        floors[h]=np.quantile(np.abs(target),.25,axis=0)
    pins=dict(ep['pins'])
    for path in [Path(__file__),BASE/'past_residual_information/train_predictions.npz',BASE/'ensemble_candidate/predictions.npz']:
        pins[m.rel(path)]=m.file_hash(path)
    m.write_json(OUT/'protocol.json',dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),pins=pins,
        hypothesis='some logged action contrasts may be resolved more tightly than2*global absolute forecast error bound',
        heldout_accessed=False,training=False,partitions=['train','validation'],
        context='current/past5/10/25 complete12 state features + strictly past1/5/10 four command coordinates; train normalization',
        treatment='native-dt weighted future mean command minus last pre-t0 command; channel contrast >=max(train dose IQR,0.1 global command std)',
        dose_min={str(k):v.tolist() for k,v in dose_min.items()},direction_floor={str(k):v.tolist() for k,v in floors.items()},
        matching='same current nav_state; different flights; top64 nearest candidate contexts; rms normalized context calipers0.25/0.5/1; other-channel full command-tape RMS differences<=0.25 train std; elapsed difference<=5%',
        disjoint='history25+forecast prefix nonoverlapping within log/segment; greedily sorted context-distance pairs, no origin reuse within each channel/horizon/caliper',
        selection='pairs frozen BEFORE reading future labels/predictions for scoring; all calipers reported, no winning-caliper selection',
        metric='difference of observed state increments vs difference of predicted state increments; model error difference directly, no2B substitution',
        data_gate='>=20 train pairs covering>=10 train flights and>=10 validation pairs covering>=5 validation flights; descriptive sufficiency only',
        limitation='not randomization or causal identification; imperfect context match and hidden airflow/actuator/controller states remain; overlapping caliper/horizon analyses not independent'))
    selected=[];candidate_counts=[];cache={}
    for split in ['train','validation']:
        b=batches[split];origin=pd.read_csv(m.BASE/f'{split}_origins.csv')
        np.testing.assert_array_equal(origin.window_id,b.trajectory.window_ids)
        samples_path=(m.ROOT/sp['manifest_path']).parent/f'samples_{split}.parquet'
        assert m.file_hash(samples_path)==sp['artifact_sha256'][samples_path.name]
        samples=pd.read_parquet(samples_path,columns=['log_id','segment_id','sample_in_segment','nav_state','valid_core'])
        modes=current_modes(samples,origin)
        z=(context(b)-mean)/scale
        for h in [5,10,25]:
            ix=nonoverlapping_origins(origin,h);command=command_coordinates(b.trajectory.controls[ix,:h]);dt=b.trajectory.dt_s[ix,:h]
            dose=(command*dt[:,:,None]).sum(1)/dt.sum(1)[:,None]-command_coordinates(b.history_controls[ix,-2])
            for ch,channel in enumerate(CHANNELS):
                for caliper in CALIPERS:
                    pairs=choose_pairs(z[ix],command,dose,dt.sum(1),b.trajectory.log_ids[ix],modes[ix],dose_min[h][ch],command_scale,ch,caliper)
                    logs=set()
                    for distance,hi,lo,contrast,other in pairs:
                        a,c=ix[hi],ix[lo];logs.update([b.trajectory.log_ids[a],b.trajectory.log_ids[c]])
                        selected.append(dict(partition=split,steps=h,channel=channel,axis=AXES[ch],caliper=caliper,
                            high_index=int(a),low_index=int(c),high_window_id=b.trajectory.window_ids[a],low_window_id=b.trajectory.window_ids[c],
                            high_log=b.trajectory.log_ids[a],low_log=b.trajectory.log_ids[c],context_distance=distance,command_contrast=contrast,other_command_distance=other,current_nav_state=int(modes[a])))
                    candidate_counts.append(dict(partition=split,steps=h,channel=channel,caliper=caliper,available_nonoverlapping_origins=len(ix),pairs=len(pairs),flights=len(logs)))
        print('pair selection frozen',split,flush=True)
    selection=pd.DataFrame(selected)
    selection.to_csv(OUT/'pairs.csv',index=False)
    counts=pd.DataFrame(candidate_counts);counts.to_csv(OUT/'coverage.csv',index=False)
    m.write_json(OUT/'selection.json',dict(pair_sha256=m.file_hash(OUT/'pairs.csv'),future_outcomes_used_for_selection=False))
    score(selection,batches,floors,pins)


def score(selection,batches,floors,pins):
    scored=[]
    for split in ['train','validation']:
        path=BASE/('past_residual_information/train_predictions.npz' if split=='train' else 'ensemble_candidate/predictions.npz')
        b=batches[split];truth=b.trajectory.truth
        with np.load(path,allow_pickle=False) as archive:
            # Materialize each compressed array once, not once per matched pair.
            p={k:archive[k] for k in ['window_ids','velocity_n','angular_velocity_b']}
            np.testing.assert_array_equal(p['window_ids'],b.trajectory.window_ids)
            for row in selection[selection.partition==split].to_dict('records'):
                i,j,h=row['high_index'],row['low_index'],row['steps'];ch=CHANNELS.index(row['channel'])
                field,axis=('velocity_n',2) if ch==0 else ('angular_velocity_b',[1,0,2][ch-1])
                target=getattr(truth,field)
                real=(target[i,h,axis]-target[i,0,axis])-(target[j,h,axis]-target[j,0,axis])
                estimate=(p[field][i,h,axis]-p[field][i,0,axis])-(p[field][j,h,axis]-p[field][j,0,axis])
                scored.append(dict(**row,observed_contrast=float(real),predicted_contrast=float(estimate),
                    contrast_error=float(estimate-real),direction_eligible=abs(real)>floors[h][ch],direction_agrees=real*estimate>0,
                    apparent_observed_gain=float(real/row['command_contrast']),apparent_predicted_gain=float(estimate/row['command_contrast'])))
    f=pd.DataFrame(scored);f.to_csv(OUT/'scored_pairs.csv',index=False)
    rows=[]
    for key,g in f.groupby(['steps','channel','caliper']):
        train_g=g[g.partition=='train'];val=g[g.partition=='validation']
        ntr=len(train_g);nv=len(val)
        fltr=len(set(train_g.high_log)|set(train_g.low_log));flval=len(set(val.high_log)|set(val.low_log))
        bound=float(np.quantile(np.abs(train_g.contrast_error),.95)) if ntr else None
        for split,z in [('train',train_g),('validation',val)]:
            if not len(z):continue
            eligible=z[z.direction_eligible]
            rows.append(dict(steps=key[0],channel=key[1],caliper=key[2],partition=split,pairs=len(z),
                flights=len(set(z.high_log)|set(z.low_log)),context_median=float(z.context_distance.median()),
                contrast_rmse=float(np.sqrt(np.mean(z.contrast_error**2))),observed_contrast_rms=float(np.sqrt(np.mean(z.observed_contrast**2))),
                direction_n=len(eligible),direction_agreement=float(eligible.direction_agrees.mean()) if len(eligible) else None,
                train_contrast_error_p95=bound,coverage=float((z.contrast_error.abs()<=bound).mean()) if bound is not None else None,
                contrast_exceeds_error_fraction=float((z.predicted_contrast.abs()>bound).mean()) if bound is not None else None,
                descriptive_data_gate=ntr>=20 and nv>=10 and fltr>=10 and flval>=5))
    pd.DataFrame(rows).to_csv(OUT/'summary.csv',index=False)
    m.verify_pins(pins)
    m.write_json(OUT/'completion.json',dict(status='complete',heldout_accessed=False,causal_gain_identified=False,
        all_calipers_retained=True,outputs={m.rel(p):m.file_hash(p) for p in OUT.iterdir() if p.is_file()}))


def score_frozen():
    global OUT
    import shutil
    previous=BASE/'matched_contrasts'
    prior=m.read_json(previous/'protocol.json')
    assert m.file_hash(previous/'source_at_selection.py')==prior['pins'][m.rel(Path(__file__))]
    assert m.file_hash(previous/'pairs.csv')==m.read_json(previous/'selection.json')['pair_sha256']
    pins=dict(prior['pins']);pins[m.rel(Path(__file__))]=m.file_hash(Path(__file__))
    for p in [previous/'protocol.json',previous/'source_at_selection.py',previous/'pairs.csv',previous/'coverage.csv']:
        pins[m.rel(p)]=m.file_hash(p)
    m.verify_pins(pins)
    OUT=BASE/'matched_contrasts_scored';OUT.mkdir(parents=True,exist_ok=False)
    for name in ['pairs.csv','coverage.csv','selection.json']:shutil.copyfile(previous/name,OUT/name)
    protocol={**prior,'pins':pins,'resumed_from':m.rel(previous/'protocol.json'),
        'resume_reason':'explicitly interrupted scoring due to repeated NPZ decompression; identical frozen pair selection and scoring math; no matching rerun'}
    m.write_json(OUT/'protocol.json',protocol)
    _,batches,_=source.inputs()
    floors={int(k):np.asarray(v) for k,v in prior['direction_floor'].items()}
    score(pd.read_csv(OUT/'pairs.csv'),batches,floors,pins)


if __name__=='__main__':
    import sys
    if '--score-frozen' in sys.argv:score_frozen()
    else:run()
