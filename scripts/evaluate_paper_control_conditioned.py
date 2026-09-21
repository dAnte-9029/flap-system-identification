"""Regroup frozen Step 1 predictions by controls only; no training or inference."""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import torch
import yaml
from system_identification.data.trajectory_dataset import CONTROL_COLUMNS
from system_identification.evaluation.trajectory import assemble_trajectory_windows
from system_identification.evaluation.paper_baselines import HORIZONS,METRIC_MAP,endpoint_metrics,aggregate_flights
from system_identification.evaluation.control_conditioned import (
    CHANNELS,BIN_NAMES,activity,control_bins,attitude_rotation_vector_error,paired_statistics,
)
from system_identification.models.trajectory import TrajectoryPrediction

BASE=ROOT/'docs/analysis/results/paper_baseline_comparison_v1'
ART=ROOT/'artifacts/paper_baseline_comparison_v1'
OUT=ROOT/'docs/analysis/results/paper_control_conditioned_eval_v1'
MODELS={'B2':'B2_StandardGRU','B3':'B3_ActuatorAwareGRU'}
EXTRA={f'attitude_{axis}_component_error_deg':f'attitude_{axis}_component_rmse_deg' for axis in ('roll','pitch','yaw')}
METRICS={**METRIC_MAP,**EXTRA}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true',help='resume only an incomplete output, with the same frozen controls protocol')
    args=parser.parse_args()
    if OUT.exists() and any(OUT.iterdir()) and not args.resume:
        raise FileExistsError('refuse nonempty result directory')
    if (OUT/'completion.json').exists():
        raise FileExistsError('completed results must not be overwritten')
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    started=time.monotonic()
    base=json.loads((BASE/'protocol.json').read_text())
    complete=json.loads((BASE/'completion.json').read_text())
    if complete['status']!='complete' or base['status']!='seed17_pilot_complete_reported':
        raise ValueError('Step 1 pilot incomplete')
    frozen={}
    def pin(path,expected=None):
        h=sha(path)
        if expected is not None and h!=expected:raise ValueError(f'hash mismatch: {path}')
        frozen[str(Path(path).relative_to(ROOT))]=h
    for name,h in complete['artifact_sha256'].items():pin(BASE/name,h)
    pin(BASE/'completion.json')
    checkpoints={'B2':ART/'B2_StandardGRU.pt','B3':ROOT/base['ours_checkpoint']}
    for name,path in checkpoints.items():pin(path,base['checkpoint_sha256'][MODELS[name]])
    for name in MODELS:pin(ART/f'{MODELS[name]}_predictions.npz')
    pin(ART/'per_origin.csv')
    # Read normalization buffers only. No model is instantiated, evaluated or trained.
    cp=torch.load(checkpoints['B3'],map_location='cpu',weights_only=False)
    state=cp['state_dict']
    reference=np.r_[state['base_model.control_mean'][0].item(),state['tail_mean'].numpy()]
    scales=np.r_[state['base_model.control_std'][0].item(),state['tail_std'].numpy()]
    expected=base['normalization']['values']
    for key in ('control_mean','control_std'):
        if not np.array_equal(state['base_model.'+key].numpy(),np.array(expected[key],dtype=np.float32)):
            raise ValueError('normalization no longer matches frozen protocol')
    reg_path=ROOT/'configs/data/trajectory_dataset_registry.yaml';pin(reg_path)
    reg=yaml.safe_load(reg_path.read_text());entry=reg['datasets'][base['dataset_id']]
    mp=ROOT/entry['manifest_path'];pin(mp,base['manifest_sha256'])
    if entry['manifest_sha256']!=base['manifest_sha256']:raise ValueError('registry mismatch')
    manifest=json.loads(mp.read_text());assign=manifest['split_contract']['assignments']
    forbidden=set(assign['sealed_test'])|set(assign['reserved_evaluation'])
    if set(assign['validation'])&forbidden or set(assign['train'])&set(assign['validation']):
        raise ValueError('invalid split')
    # Only the two explicitly open validation files are permitted; no raw logs or train reads.
    for filename in ('samples_validation.parquet','windows_validation.parquet'):
        pin(mp.parent/filename,base['artifact_sha256'][filename])
    samples=pd.read_parquet(mp.parent/'samples_validation.parquet')
    windows=pd.read_parquet(mp.parent/'windows_validation.parquet')
    origins=pd.read_csv(BASE/'validation_origins.csv')
    if len(origins)!=2582 or origins.window_id.duplicated().any():raise ValueError('origin contract mismatch')
    if set(samples.log_id)!=set(base['validation_flights']) or set(samples.split)!={'validation'}:
        raise ValueError('unexpected sample partition')
    if not windows.equals(origins[windows.columns]):raise ValueError('registered windows differ from baseline origins')
    batch=assemble_trajectory_windows(samples,windows)
    if not np.array_equal(batch.window_ids,origins.window_id.to_numpy()):raise ValueError('origin order mismatch')
    lookup=samples[samples.valid_core].set_index(['log_id','segment_id','sample_in_segment'])
    for offset,column in [(0,'origin_timestamp_us'),(50,'end_timestamp_us')]:
        index=pd.MultiIndex.from_arrays([origins.log_id,origins.segment_id,origins.start_sample_in_segment+offset])
        if not np.array_equal(lookup.loc[index,'timestamp_us'].to_numpy(),origins[column].to_numpy()):
            raise ValueError('timestamp identity mismatch')
    activity_frames=[];thresholds={}
    for horizon,k in HORIZONS.items():
        a=activity(batch.controls[:,:k],reference,scales)
        frame=origins[['window_id','log_id','cohort','segment_id','start_sample_in_segment','origin_timestamp_us']].copy()
        frame['horizon_s']=horizon;frame['observed_horizon_s']=batch.dt_s[:,:k].sum(1)
        frame['total_activity']=a['total']
        for j,channel in enumerate(CHANNELS):
            for key in ('magnitude','delta','tv','normalized_delta'):frame[f'{channel}_{key}']=a[key][:,j]
        thresholds[str(horizon)]={}
        for channel in ('overall',*CHANNELS):
            score=a['total'] if channel=='overall' else a['normalized_delta'][:,CHANNELS.index(channel)]
            labels,top,th=control_bins(score)
            frame[f'{channel}_bin']=labels;frame[f'{channel}_top20']=top
            thresholds[str(horizon)][channel]=th
        activity_frames.append(frame)
    activity_frame=pd.concat(activity_frames,ignore_index=True)
    activity_frame.to_csv(OUT/'origin_activity.csv',index=False)
    protocol=dict(experiment='paper_control_conditioned_eval_v1',state='control_bins_frozen_before_error_loading',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        source_baseline_result=str(BASE.relative_to(ROOT)),source_baseline_commit='cdf7625af4301642df61a085c3998383bbdbbe62',
        checkpoints={n:dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for n,p in checkpoints.items()},
        origins=dict(count=2582,file=str((BASE/'validation_origins.csv').relative_to(ROOT)),
                     sha256=sha(BASE/'validation_origins.csv'),identity='window_id + log_id + segment_id + start sample + native timestamp'),
        controls=dict(raw_order=list(CONTROL_COLUMNS),logged_sources=['actuator_motors.control[0]','actuator_servos.control[0]','actuator_servos.control[1]','actuator_servos.control[2]'],
            independent_coordinates=['drive=motor','common=(left+right)/2','differential=(left-right)/2','rudder=rudder'],
            command_semantics='post-allocation normalized setpoints, pre-driver PWM reversal; not PWM, surface-angle feedback, or Hz commands',
            PWM='PWM_MAIN_REV=17 reverses MAIN1 and MAIN5 in the downstream output driver; extraction/model do not reapply reversal',
            B2='raw four channels standardized using frozen train control_mean/control_std',
            B3='motor normalized by backbone train stats; transformed tail normalized by frozen tail_mean/tail_std then drive/tail proxies; history-only backbone',
            reference=reference.tolist(),scale=scales.tolist(),scale_source='existing frozen B3 TRAIN history+future-control normalization buffers; no train/validation refit'),
        activity=dict(window='actual applied control tape u[0:K], excludes u[K] at endpoint',
            magnitude='sqrt(mean_k((coordinate[k]-train_mean)^2))',delta='max_k(abs(coordinate[k]-coordinate[0]))',
            tv='sum_k(abs(coordinate[k+1]-coordinate[k]))',channel_score='delta/train_std',
            total_score='sqrt(mean_channels((delta/train_std)^2))',
            primary='max departure from origin; magnitude and TV descriptive only, not alternative outcome-driven selectors'),
        bins=dict(quantiles=[.33,.67,.8],method='numpy linear quantile per horizon, pooled open validation controls; shared across cohorts',
            boundaries='Low <= q33; Medium q33 < score <= q67; High > q67; Top20 >= q80 AND score>0',
            ties='preserved, never broken with model errors or arbitrary origin ranks; actual counts reported',thresholds=thresholds),
        horizons=HORIZONS,primary_horizon_s=.5,timing_contract=base['timing_contract'],
        metrics=dict(existing=METRIC_MAP,additional=EXTRA,aggregation='per-flight RMSE then equal-flight mean, as in Step 1',
            attitude_components='principal Log(q_truth^-1*q_prediction) degrees, in truth-body coordinates; component RMS, not independent Euler angles',
            increments='common true initial state makes delta-v/omega errors identical to endpoint v/omega; not independent evidence'),
        paired=dict(sign='B3-B2; negative favors B3',origin_statistics='pooled mean, median and win fraction, descriptive only',
            flight_statistics='equal-flight mean of within-flight mean paired errors AND equal-flight mean of paired per-flight RMSE differences',
            bootstrap='10000 resamples of whole flights, within Sep7/Sep17 strata; fixed represented-flight counts and frozen control bins; percentile 95% CI',
            bootstrap_rng_seed=17023,training_seed_runs_added=0,
            limitations='only 17 flights/two days, fewer in some subsets; within-day dependence, bin-threshold and seed uncertainty not captured; no multiplicity-adjusted claims'),
        data_files_opened=[str((mp.parent/x).relative_to(ROOT)) for x in ('samples_validation.parquet','windows_validation.parquet')],
        sealed_test_opened=False,reserved_test_opened=False,inference_run=False,training_run=False,
        excluded_flights=base['excluded_sealed_flights'],frozen_input_sha256=frozen,
        origin_activity_sha256=sha(OUT/'origin_activity.csv'),
        interpretation='control-conditioned prediction accuracy in closed-loop logs, not causal identification; retain Step 1 training-family confounds')
    if args.resume and (OUT/'protocol.json').exists():
        previous=json.loads((OUT/'protocol.json').read_text())
        if previous['bins']!=protocol['bins'] or previous['origin_activity_sha256']!=protocol['origin_activity_sha256']:
            raise ValueError('resume would change control-only selection')
    write_json(OUT/'protocol.json',protocol)
    print('Control bins frozen; loading existing predictions for metric parity only.',flush=True)
    existing=pd.read_csv(ART/'per_origin.csv')
    error_frames={};parity={}
    for short,name in MODELS.items():
        with np.load(ART/f'{name}_predictions.npz',allow_pickle=False) as predfile:
            if not np.array_equal(predfile['window_ids'],batch.window_ids):raise ValueError('prediction origin IDs mismatch')
            prediction=TrajectoryPrediction(**{k:predfile[k] for k in vars(batch.truth)})
        frame=endpoint_metrics(prediction,batch,model=name)
        keys=['window_id','horizon_s']
        old=existing[existing.model==name].set_index(keys).sort_index()
        fresh=frame.set_index(keys).sort_index()
        if not old.index.equals(fresh.index):raise ValueError('prediction error key mismatch')
        difference=np.max(np.abs(old[list(METRIC_MAP)].to_numpy()-fresh[list(METRIC_MAP)].to_numpy()))
        if difference>1e-10:raise ValueError('frozen metrics not reproduced')
        parity[short]=float(difference)
        for h,k in HORIZONS.items():
            component=attitude_rotation_vector_error(prediction.quaternion_nb[:,k],batch.truth.quaternion_nb[:,k])
            for j,col in enumerate(EXTRA):frame.loc[frame.horizon_s==h,col]=np.abs(component[:,j])
        error_frames[short]=frame.set_index(keys)
    # Verify ALL unconditioned equal-flight means against published Step 1 table.
    joined=pd.concat([x.reset_index() for x in error_frames.values()],ignore_index=True)
    _,macro,_=aggregate_flights(joined)
    old_summary=pd.read_csv(BASE/'summary.csv')
    old_summary=old_summary[old_summary.model.isin(MODELS.values())]
    keys=['model','seed','cohort','horizon_s']
    if not np.allclose(macro.set_index(keys).sort_index()[list(METRIC_MAP.values())],
                       old_summary.set_index(keys).sort_index()[list(METRIC_MAP.values())],atol=1e-12,rtol=1e-12):
        raise ValueError('full-cohort macro parity failed')
    summaries=[];paired=[];flights=[];coverage=[]
    for h in HORIZONS:
        act=activity_frame[activity_frame.horizon_s==h]
        for channel in ('overall',*CHANNELS):
            for label in BIN_NAMES:
                select=act[f'{channel}_top20'] if label=='Top20' else act[f'{channel}_bin']==label
                selected=act.loc[select]
                for cohort in ('ALL','Sep7','Sep17'):
                    sub=selected if cohort=='ALL' else selected[selected.cohort==cohort]
                    counts=sub.log_id.value_counts()
                    common=dict(channel=channel,activity_bin=label,horizon_s=h,cohort=cohort)
                    coverage.append(dict(**common,n_origins=len(sub),n_flights=len(counts),
                        Sep7_origins=int((sub.cohort=='Sep7').sum()),Sep17_origins=int((sub.cohort=='Sep17').sum()),
                        max_flight_origin_fraction=float(counts.max()/len(sub)) if len(sub) else None))
                    if not len(sub):continue
                    index=pd.MultiIndex.from_arrays([sub.window_id,sub.horizon_s],names=['window_id','horizon_s'])
                    b2=error_frames['B2'].loc[index,list(METRICS)].to_numpy()
                    b3=error_frames['B3'].loc[index,list(METRICS)].to_numpy()
                    ps,fs=paired_statistics(b2,b3,sub.reset_index(drop=True),list(METRICS.values()))
                    for k,v in common.items():ps[k]=v
                    paired.append(ps)
                    for k,v in common.items():
                        if k!='cohort':fs[k]=v
                    fs['aggregate_cohort']=cohort
                    flights.append(fs)
                    for short in MODELS:
                        means=fs.groupby('metric')[short+'_error'].mean()
                        summaries.append(dict(**common,model=short,n_origins=len(sub),n_flights=len(counts),**means.to_dict()))
    pd.DataFrame(summaries).to_csv(OUT/'conditional_summary.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'bin_coverage.csv',index=False)
    pd.concat(paired,ignore_index=True).to_csv(OUT/'paired_summary.csv',index=False)
    perflight=pd.concat(flights,ignore_index=True)
    perflight=perflight[perflight.aggregate_cohort=='ALL'].drop(columns='aggregate_cohort')
    perflight.to_csv(OUT/'per_flight_all_bins.csv',index=False)
    perflight[perflight.activity_bin.isin(['High','Top20'])].to_csv(OUT/'per_flight_high_excitation.csv',index=False)
    # Useful for auditing pair direction/median/win fraction without new predictions.
    for name,frame in error_frames.items():
        frame[list(METRICS)].to_csv(OUT/f'{name}_frozen_endpoint_errors.csv')
    for path,h in frozen.items():
        if sha(ROOT/path)!=h:raise ValueError(f'frozen input changed: {path}')
    protocol['state']='statistics_complete'
    protocol['source_sha256']={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),
        ROOT/'src/system_identification/evaluation/control_conditioned.py',ROOT/'src/system_identification/evaluation/paper_baselines.py',
        ROOT/'src/system_identification/evaluation/trajectory.py',ROOT/'src/system_identification/data/trajectory_dataset.py',
        ROOT/'src/system_identification/models/trajectory_main_v2.py']}
    write_json(OUT/'protocol.json',protocol)
    write_json(OUT/'sanity_checks.json',dict(frozen_inputs_unchanged=True,origin_count=2582,
        same_origins_truth_horizons_metrics=True,full_cohort_macro_parity=True,metric_max_abs_parity_error=parity,
        quantiles_frozen_before_reading_errors=True,training_run=False,inference_run=False,
        normalization_changed=False,sealed_test_opened=False,reserved_test_opened=False,
        wall_time_s=time.monotonic()-started))
    print('Conditional and paired-flight statistics completed.',flush=True)


if __name__=='__main__':main()
