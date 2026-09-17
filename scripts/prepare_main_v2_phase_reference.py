#!/usr/bin/env python3
"""Train-only reference, causal resets, then separately labelled alignment oracle."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4');os.environ.setdefault('OMP_NUM_THREADS','4')
import sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd
from system_identification.models.phase_reference import interval_design,rotate_coefficients,estimate_windows,circular
from run_main_v2_free_running import sha
OLD=ROOT/'docs/analysis/results/main_v2_dynamics_observability_phase'
DATA=ROOT/'dataset/trajectory_v1_august_f5_c4'
LENGTHS=[12,25,50]


def segment_arrays(g):
    return (g.timestamp_us.to_numpy()*1e-6,g.relative_flap_phase_rad.to_numpy(),
        g[['angular_velocity_body_rad_s_'+s for s in 'xyz']].to_numpy(),g[['velocity_ned_m_s_'+s for s in 'xyz']].to_numpy())


def coefficients(groups,cuts=None):
    bylog={};samples={}
    for (log,seg),g in groups:
        t,p,w,v=segment_arrays(g);X=interval_design(p);y=np.column_stack([np.diff(w[:,1]),np.diff(v[:,2])])/np.diff(t)[:,None]
        sel=np.ones(len(y),bool) if cuts is None else t[1:]<cuts[log]-2
        if sel.sum()<10:continue
        bylog.setdefault(log,[]).append((X[sel],y[sel]));samples.setdefault(log,[]).append(y[sel])
    co={}
    for log,parts in bylog.items():
        x=np.concatenate([p[0] for p in parts]);y=np.concatenate([p[1] for p in parts]);c=np.linalg.lstsq(np.column_stack([np.ones(len(x)),x]),y,rcond=None)[0]
        co[log]=c[1:]
    scales=np.std(np.concatenate([np.concatenate(v) for v in samples.values()]),axis=0)
    return co,scales


def alignment_offset(c,template,scales):
    grid=np.arange(360)*2*np.pi/360;rot=rotate_coefficients(template,grid)
    # Positive gain per signal permits load/frequency-dependent amplitude.
    dot=np.sum(rot*c[None],axis=1);den=np.sum(rot*rot,axis=1)
    score=np.sum((np.sum(c*c,axis=0)[None]-np.maximum(dot,0)**2/np.maximum(den,1e-12))/scales**2,axis=1)
    return float(grid[score.argmin()])


def fit_template(co,scales):
    logs=sorted(co);template=co[logs[0]].copy();offsets={l:0. for l in logs}
    for _ in range(15):
        offsets={l:alignment_offset(co[l],template,scales) for l in logs}
        gauge=offsets[logs[0]];offsets={l:float((v-gauge)%(2*np.pi)) for l,v in offsets.items()}
        aligned=[rotate_coefficients(co[l],-offsets[l]) for l in logs]
        template=np.mean(aligned,axis=0)
    return template,offsets


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);a=pa.parse_args();out=a.output
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    roots=[ROOT/'docs/analysis/results'/n for n in ['main_v2_free_running_5s','main_v2_training_objective_ablation','main_v2_dynamics_observability_phase','main_v2_increment_supervision']]
    hashes={str(p.relative_to(ROOT)):sha(p) for root in roots for p in root.rglob('*') if p.is_file()}
    data={s:pd.read_parquet(DATA/f'samples_{s}.parquet') for s in ['train','validation']}
    meta={s:pd.read_csv(OLD/f'{s}_points.csv') for s in data}
    cuts={l:float(g.timestamp_s.quantile(.7)) for l,g in meta['train'].groupby('log_id')}
    groups={s:list(d[d.valid_core].groupby(['log_id','segment_id'],sort=False)) for s,d in data.items()}
    co,scales=coefficients(groups['train'],cuts);template,reference=fit_template(co,scales)
    (out/'canonical_template.json').write_text(json.dumps(dict(coefficients=template.tolist(),scales=scales.tolist(),train_reference_offsets=reference,
        gauge='first sorted train log has offset0; statistical canonical gauge, NOT verified mechanical zero',fit='first70percent minus2s of train logs only',cuts=cuts,
        order=3,grid_degrees=1,matching='positive per-channel gain and DC nuisance; inverse train variance weights',
        derivative='backward native state difference; template interval-average harmonics assumes linear phase within interval; no temporal filter',
        filter_delay='backward difference represents its measured interval, not its right endpoint; exact sinc correction under interval-linear phase',
        channels=['q_dot from body rates','az_n from navigation velocity, not accelerometer specific force']),indent=2))
    frames=[]
    for split in data:
        for (log,seg),g in groups[split]:
            m=meta[split];idx=m.index[(m.log_id==log)&(m.segment_id==seg)].to_numpy()
            if not len(idx):continue
            ends=m.loc[idx,'sample_in_segment'].to_numpy(int);arrays=segment_arrays(g)
            for length in LENGTHS:
                for method,channels in [('E1',(0,)),('E2',(0,1))]:
                    ans=estimate_windows(*arrays,ends,length,template,scales,channels)
                    f=m.loc[idx,['log_id','segment_id','sample_in_segment','timestamp_s']].copy();f['point_index']=idx;f['split']=split;f['method']=method;f['requested_history_steps']=length
                    for k,v in ans.items():f[k]=v
                    f['complete_history']=f.history_intervals==length
                    f['train_temporal_holdout']=(split=='train')&(f.timestamp_s>cuts.get(log,np.inf)+2)
                    f['reference_offset_error_rad']=np.abs(circular(f.offset_rad-reference[log])) if split=='train' else np.nan
                    # No validation oracle offsets are available at this stage.
                    frames.append(f)
            print('causal estimates',split,log,seg,flush=True)
    estimates=pd.concat(frames,ignore_index=True);estimates.to_csv(out/'phase_offset_estimation.csv',index=False)
    hold=estimates[estimates.train_temporal_holdout&estimates.complete_history]
    scores=hold.groupby(['method','requested_history_steps']).reference_offset_error_rad.agg(['median',lambda x:x.quantile(.9)])
    scores['selection_score']=scores.iloc[:,0]+scores.iloc[:,1];best=scores.selection_score.idxmin()
    (out/'selected_estimator.json').write_text(json.dumps(dict(method=best[0],history_steps=int(best[1]),selection='train temporal holdout median+p90 offset reference error; validation not used'),indent=2))
    rows=[]
    for keys,g in estimates.groupby(['split','method','requested_history_steps']):
        if keys[0]=='train':g=g[g.train_temporal_holdout]
        for bucket,z in [('all',g[g.complete_history]),('less_than_one_cycle',g[g.complete_history&(g.observed_cycles<1)]),('one_or_more_cycles',g[g.complete_history&(g.observed_cycles>=1)])]:
            if not len(z):continue
            row=dict(split=keys[0],method=keys[1],history_steps=keys[2],cycle_bin=bucket,n=len(z),duration_mean_s=z.duration_s.mean(),cycles_median=z.observed_cycles.median())
            for q in [.5,.75,.9,.95]:row['offset_abs_error_p'+str(int(q*100))]=z.reference_offset_error_rad.quantile(q) if keys[0]=='train' else np.nan
            rows.append(row)
    pd.DataFrame(rows).to_csv(out/'history_length_ablation.csv',index=False)
    # Oracle explicitly isolated: full-log truth used ONLY for diagnostic offsets.
    oracle=[]
    for split in data:
        full,_=coefficients(groups[split])
        for log,c in full.items():oracle.append(dict(split=split,log_id=log,offset_rad=alignment_offset(c,template,scales),contract='ORACLE_NOT_DEPLOYABLE_FULL_LOG_TRUTH'))
    pd.DataFrame(oracle).to_csv(out/'oracle_offsets_NOT_DEPLOYABLE.csv',index=False)
    protocol=dict(historical_hashes=hashes,dataset='trajectory_v1_august_f5_c4 historical Main V2 reproduction',
        dataset_hashes={str((DATA/f'samples_{s}.parquet').relative_to(ROOT)):sha(DATA/f'samples_{s}.parquet') for s in data},
        seeds=[415,416,417],probe_epochs=30,probe_batch_size=512,probe_lr=.001,probe_targets=['D0_raw','increment2'],
        probe_contracts=['P0','P1','P2','P_oracle'],probe_architecture='420 ->64 SiLU ->64 SiLU ->6; same as Step4',
        probe_gate='oracle angular increment RMSE >=5percent lower, no linear increment >5percent worse, all seeds angular improve; causal recovers >=50percent oracle angular gain and no linear >5percent worse; no complete-model training unless both pass',
        full_model_policy='F0 original objective, F1 causal phase only, F_oracle diagnostic if probe passes; F2 optional only after F1 improves long benchmark',
        oracle_scope='full log truth allowed ONLY for P_oracle diagnostic; template/estimator/normalization never fitted to validation',sealed_test_opened=False)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));print('PREPARE COMPLETE',best,flush=True)

if __name__=='__main__':main()
