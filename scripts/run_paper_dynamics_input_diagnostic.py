"""Paper Step5: inference-only Actual versus current-command Hold replay."""
from __future__ import annotations
import argparse
from dataclasses import fields,is_dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import torch
from run_paper_baseline_comparison import write_json,causality_check,STATS
from run_paper_standard_gru_multiseed import subset,normalization_hash
from system_identification.data.september_trajectory import file_hash
from system_identification.models.trajectory import TrajectoryPrediction
from system_identification.models.trajectory_main_v1 import CausalHistoryTrajectoryModel
from system_identification.training.trajectory_main_v1 import predict_history_trajectory_model
from system_identification.evaluation.paper_baselines import HORIZONS,endpoint_metrics,aggregate_flights,cohort
from system_identification.evaluation.future_control_diagnostic import (
    METRICS,GROUPS,hold_controls,excitation,train_thresholds,assign_groups,squared_errors,
    interval_mse,flight_metrics,aggregate,paired_comparisons,
)

OUT=ROOT/'docs/analysis/results/paper_dynamics_input_diagnostic_v1'
ART=ROOT/'artifacts/paper_dynamics_input_diagnostic_v1'
BASE=ROOT/'docs/analysis/results/paper_baseline_comparison_v1'
S3=ROOT/'docs/analysis/results/paper_standard_gru_multiseed_v1'
A3=ROOT/'artifacts/paper_standard_gru_multiseed_v1'
HIST=ROOT/'docs/analysis/results/paper_history_length_ablation_v1'
SEEDS=(17,23,42)
DEVICE='cuda:1'
ATOL=2e-5
RTOL=1e-6
SOURCES=['src/system_identification/models/trajectory_main_v1.py','src/system_identification/models/trajectory.py',
    'src/system_identification/training/trajectory_main_v1.py','src/system_identification/evaluation/trajectory.py',
    'src/system_identification/evaluation/paper_baselines.py']


def read(path):return json.loads(path.read_text())
def rel(path):return str(path.relative_to(ROOT))
def require(ok,message):
    if not ok:raise ValueError(message)


def verify(pins):
    for name,value in pins.items():require(file_hash(ROOT/name)==value,f'required input hash mismatch: {name}')


def array_digest(value):
    h=hashlib.sha256()
    def visit(v):
        if is_dataclass(v):
            for f in fields(v):h.update(f.name.encode());visit(getattr(v,f.name))
        elif isinstance(v,np.ndarray):
            h.update(str(v.shape).encode());h.update(str(v.dtype).encode())
            h.update(json.dumps(v.tolist(),ensure_ascii=False).encode() if v.dtype.kind in 'OUS' else v.tobytes())
        else:h.update(repr(v).encode())
    visit(value)
    return h.hexdigest()


def stats_distribution(values):
    return dict(mean=float(np.mean(values)),std=float(np.std(values,ddof=1)) if len(values)>1 else 0.,
        min=float(np.min(values)),p05=float(np.quantile(values,.05)),p25=float(np.quantile(values,.25)),
        median=float(np.median(values)),p75=float(np.quantile(values,.75)),p95=float(np.quantile(values,.95)),max=float(np.max(values)))


def audit():
    require(not (OUT/'protocol.json').exists(),'protocol already frozen; no overwrite')
    OUT.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
    s3=read(S3/'protocol.json');complete=read(S3/'completion.json');bp=read(BASE/'protocol.json')
    hp=read(HIST/'protocol.json')
    require(complete['status']=='complete','H26 source incomplete')
    pins={}
    def pin(path,expected=None):
        actual=file_hash(path);require(expected is None or actual==expected,f'required hash mismatch: {path}')
        pins[rel(path)]=actual
    # Deliberately no broad historical source/registry audit; verify required interfaces only.
    for path in [S3/'protocol.json',S3/'completion.json',S3/'training_runs.csv',S3/'per_seed_summary.csv',
                 BASE/'protocol.json',BASE/'representative_selection.json',HIST/'protocol.json',HIST/'review_report.md']:
        pin(path)
    for source in SOURCES:pin(ROOT/source,bp['source_sha256'][source])
    manifest_path=ROOT/s3['manifest_path'];pin(manifest_path,s3['manifest_sha256'])
    manifest=read(manifest_path);assign=manifest['split_contract']['assignments']
    forbidden=set(assign['sealed_test'])|set(assign['reserved_evaluation'])
    require(not manifest['split_contract']['sealed_test_opened'],'test marked open')
    require(not set(assign['train'])&set(assign['validation']),'overlapping split')
    require(not (set(assign['train'])|set(assign['validation']))&forbidden,'forbidden flight')
    for part in ('train','validation'):
        require(assign[part]==s3[f'{part}_flights'],'flight list changed')
        for kind in ('samples','windows'):
            name=f'{kind}_{part}.parquet';pin(manifest_path.parent/name,s3['artifact_sha256'][name])
        pin(BASE/f'{part}_origins.csv',s3['train_window_sha256'] if part=='train' else s3['validation_origin_sha256'])
    pin(A3/'prepared.pt',s3['prepared_sha256'])
    batches=torch.load(A3/'prepared.pt',map_location='cpu',weights_only=False)
    checkpoints={};predictions={};runs=pd.read_csv(S3/'training_runs.csv')
    for seed in SEEDS:
        row=runs[runs.seed==seed].iloc[0];cp=ROOT/row.checkpoint_path;pin(cp,row.checkpoint_sha256)
        pred=ROOT/'artifacts/paper_baseline_comparison_v1/B2_StandardGRU_predictions.npz' if seed==17 else A3/f'seed{seed}'/'predictions.npz'
        expected=s3['frozen_input_sha256'][rel(pred)] if seed==17 else complete['artifact_sha256'][rel(pred)]
        pin(pred,expected);checkpoints[str(seed)]=rel(cp);predictions[str(seed)]=rel(pred)
    cp=torch.load(ROOT/checkpoints['17'],map_location='cpu',weights_only=False)
    stats={k:cp['state_dict'][k].numpy() for k in STATS}
    require(normalization_hash(stats)==s3['normalization_sha256'],'normalization changed')
    thresholds=[];activity=[];distributions=[];coverage=[];perflight=[]
    for part,n in [('train',28293),('validation',2582)]:
        b=batches[part];origins=pd.read_csv(BASE/f'{part}_origins.csv')
        require(len(b.trajectory.window_ids)==n,'origin count')
        np.testing.assert_array_equal(b.trajectory.window_ids,origins.window_id)
        np.testing.assert_array_equal(b.trajectory.log_ids,origins.log_id)
        require(set(origins.log_id)==set(assign[part]) and b.history_mask.all(),'origin/history contract')
        require(b.history_state_features.shape==(n,26,12),'H26 state shape')
        np.testing.assert_array_equal(b.history_controls[:,-1],b.trajectory.controls[:,0])
    for horizon,k in HORIZONS.items():
        train=excitation(batches['train'].trajectory.controls,stats['control_std'],k)
        low,high,valid=train_thresholds(train)
        thresholds.append(dict(horizon_s=horizon,step=k,train_q25=low,train_q75=high,grouping_available=valid,n_train_origins=len(train)))
        for part,values in [('train',train),('validation',excitation(batches['validation'].trajectory.controls,stats['control_std'],k))]:
            b=batches[part].trajectory
            cohorts=np.array([cohort(x) for x in b.log_ids]) if part=='validation' else np.full(len(values),'train')
            labels=assign_groups(values,low,high)
            frame=pd.DataFrame(dict(partition=part,window_id=b.window_ids,flight_id=b.log_ids,cohort=cohorts,
                horizon_s=horizon,step=k,excitation=values,group=labels))
            activity.append(frame)
            for name,group in [('ALL',frame),*list(frame.groupby('cohort'))]:
                distributions.append(dict(partition=part,cohort=name,step=k,horizon_s=horizon,**stats_distribution(group.excitation)))
            if part!='validation':continue
            for groupname in ['ALL',*GROUPS]:
                selected=frame if groupname=='ALL' else frame[frame.group==groupname]
                for c in ['ALL','Sep7','Sep17']:
                    g=selected if c=='ALL' else selected[selected.cohort==c]
                    coverage.append(dict(step=k,horizon_s=horizon,group=groupname,cohort=c,
                        grouping_available=valid,n_origins=len(g),n_flights=g.flight_id.nunique(),
                        **({f'E_{key}':value for key,value in stats_distribution(g.excitation).items()} if len(g) else {})))
                for log in sorted(set(b.log_ids)):
                    perflight.append(dict(step=k,horizon_s=horizon,group=groupname,flight_id=log,cohort=cohort(log),n_origins=int((selected.flight_id==log).sum())))
    activities=pd.concat(activity,ignore_index=True)
    activities.to_csv(ART/'origin_control_activity.csv',index=False)
    pd.DataFrame(thresholds).to_csv(OUT/'control_group_thresholds.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'control_group_coverage.csv',index=False)
    pd.DataFrame(perflight).to_csv(OUT/'control_group_per_flight.csv',index=False)
    pd.DataFrame(distributions).to_csv(OUT/'control_change_distribution.csv',index=False)
    origins=pd.read_csv(BASE/'validation_origins.csv')
    fixed=read(BASE/'representative_selection.json');cases=[dict(name='fixed_step1',window_id=fixed['window_id'],source_index=fixed['source_index'])]
    labels=activities.query('partition=="validation" and step==25').set_index('window_id').loc[origins.window_id]
    for c in ('Sep7','Sep17'):
        selected=origins[(labels.group.to_numpy()=='high')&(origins.cohort==c)].sort_values(['log_id','segment_id','start_sample_in_segment'])
        if len(selected):
            row=selected.iloc[len(selected)//2];cases.append(dict(name=f'{c}_high_median_identity',window_id=row.window_id,
                source_index=int(row.name),cohort=c,eligible_origins=len(selected)))
        else:cases.append(dict(name=f'{c}_high_median_identity',missing=True,eligible_origins=0))
    selection=dict(seed=17,rule='Step1 fixed origin plus each cohort high-E25 group sorted (log_id,segment_id,start_sample_in_segment), index floor(n/2); frozen before reading Hold or prediction errors',cases=cases)
    write_json(OUT/'representative_selection.json',selection)
    elapsed=batches['validation'].trajectory.dt_s.cumsum(axis=1)
    time_rows=[]
    for step in range(1,51):
        for group in ['ALL',*GROUPS]:
            mask=np.ones(len(origins),bool) if group=='ALL' else labels.group.to_numpy()==group
            for c in ['ALL','Sep7','Sep17']:
                selected=mask if c=='ALL' else mask&(origins.cohort.to_numpy()==c)
                if selected.any():time_rows.append(dict(step=step,group=group,cohort=c,n_origins=int(selected.sum()),
                    **{f'elapsed_{key}_s':value for key,value in stats_distribution(elapsed[selected,step-1]).items()}))
    pd.DataFrame(time_rows).to_csv(OUT/'native_elapsed_time.csv',index=False)
    own=['scripts/run_paper_dynamics_input_diagnostic.py','scripts/report_paper_dynamics_input_diagnostic.py',
         'src/system_identification/evaluation/future_control_diagnostic.py','tests/test_future_control_diagnostic.py']
    for name in own:pin(ROOT/name)
    for name in ['control_group_thresholds.csv','representative_selection.json','native_elapsed_time.csv']:
        pin(OUT/name)
    pin(ART/'origin_control_activity.csv')
    registry=ROOT/'configs/data/trajectory_dataset_registry.yaml'
    protocol=dict(experiment='paper_dynamics_input_diagnostic_v1',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),source_main_commit=hp['source_main_model_commit'],
        source_history_commit='c65d80fb2e87ee17ef8beab2fffa02d06daa9eb0',source_history_classification_unchanged='Mixed',
        checkpoints=checkpoints,actual_predictions=predictions,seeds=list(SEEDS),model=s3['architecture'],parameter_count=21383,
        normalization_sha256=s3['normalization_sha256'],control_std=stats['control_std'].tolist(),
        manifest_sha256=s3['manifest_sha256'],train_window_sha256=s3['train_window_sha256'],validation_origin_sha256=s3['validation_origin_sha256'],
        train_flights=s3['train_flights'],validation_flights=s3['validation_flights'],excluded_flights=s3['excluded_sealed_flights'],
        origins=dict(train=28293,validation=2582),history_steps=26,history_semantics=hp['history_semantics'],
        control_representation='[motor,left,right,rudder] normalized post-allocation pre-PWM commands, NOT measured flapping frequency or physical surface angles',
        conditions=dict(Actual='u_t,...,u_(t+49), reuse frozen full predictions',Hold='repeat u_t on all4channels for50commands; history/initial states/phase anchor/dt unchanged'),
        grouping=dict(formula='E_K=sqrt(mean_{k=0..K-1,j}(((u[t+k,j]-u[t,j])/frozen_control_std[j])**2))',
            thresholds='train origin E_K linear quantiles .25/.75 separately K5/10/25/50',
            bins='low E<=q25; middle q25<E<=q75; high E>q75; ties deterministic; q25==q75 disables groups, retains ALL',
            curve_groups='fixed K25 membership for all1..50steps',offline_only=True,error_independent=True),
        horizons=dict(HORIZONS),primary_horizon_s=.5,metrics=list(METRICS),
        aggregation='sqrt(mean_origins(error_squared)) within each flight, equal-flight mean, then3seed mean and sampleSD(ddof1); no window-level tests',
        interval_formula='origin MSE(K)=sum_{k=1..K}dt[k-1]*error_squared[k]/sum_{k=1..K}dt[k-1]; equal-origin mean within flight then sqrt; equal-flight mean then seed mean/SD; t0 excluded; right endpoint weights',
        gains='Hold-Actual; relative_gain_pct=100*(Hold-Actual)/Hold on reported aggregate; Hold==0 -> undefined, no epsilon',
        plotting_time='median origin cumulative native dt at each step; no resampling; full distribution and group coverage retained',
        tolerances=dict(reuse_and_first_step_atol=ATOL,reuse_and_first_step_rtol=RTOL,quaternion_norm_atol=1e-5,
            within_same_model_first_step='bitwise in test; full reused comparison uses above frozen tolerance',constant_control='bitwise same-batch prediction'),
        nonfinite_policy='save invalid identity/context, do not drop samples; stop aggregation; no checkpoint retry',
        scientific_pass_threshold=None,engineering_separate_from_scientific_result=True,
        inference_device=DEVICE,batch_size=128,runtime_settings='4torchthreads,deterministic_algorithms=True,cudnn.benchmark=False,CUBLAS=:4096:8; inherited',
        actual_parity_sample='first128 fixed origins for each seed; same batch size/order as source',
        case_selection=selection,frozen_input_sha256=pins,
        source_registry_observation=dict(current_sha256=file_hash(registry),historical_sha256=s3.get('frozen_input_sha256',{}).get(rel(registry)),
            policy='record only; required explicit manifest/cache identities are authority for this frozen analysis'),
        validation_array_sha256=array_digest(batches['validation']),
        sealed_test_opened_this_run=False,reserved_evaluation_opened_this_run=False,training_performed=False,
        causal_limitation='Both conditions compared with original Actual flight truth; Hold has no matched counterfactual real flight. Predictive information only, not causal control validation.')
    write_json(OUT/'protocol.json',protocol)
    (OUT/'git_status_at_launch.txt').write_text(subprocess.check_output(['git','status','--short'],text=True))
    print(pd.DataFrame(thresholds).to_string(index=False),flush=True)
    print(pd.DataFrame(coverage).query('step==25').to_string(index=False),flush=True)


def load_inputs():
    p=read(OUT/'protocol.json');verify(p['frozen_input_sha256'])
    batches=torch.load(A3/'prepared.pt',map_location='cpu',weights_only=False)
    return p,batches


def model_for(seed,p):
    cp=torch.load(ROOT/p['checkpoints'][str(seed)],map_location='cpu',weights_only=False)
    stats={k:cp['state_dict'][k].numpy() for k in STATS}
    require(normalization_hash(stats)==p['normalization_sha256'],'normalization mismatch')
    model=CausalHistoryTrajectoryModel(hidden_size=64,use_controls=True,**stats)
    model.load_state_dict(cp['state_dict'],strict=True)
    require(sum(x.numel() for x in model.parameters())==21383,'wrong architecture')
    return model.eval()


def frozen_prediction(seed,p,batch):
    with np.load(ROOT/p['actual_predictions'][str(seed)],allow_pickle=False) as f:
        np.testing.assert_array_equal(f['window_ids'],batch.trajectory.window_ids)
        return TrajectoryPrediction(**{k:f[k] for k in vars(batch.trajectory.truth)})


def validate_prediction(pred,batch,seed,condition):
    failed=np.zeros(len(batch.trajectory.window_ids),bool)
    for key,values in vars(pred).items():
        require(values.shape==getattr(batch.trajectory.truth,key).shape,f'bad shape {condition}/{key}')
        failed|=~np.isfinite(values).reshape(len(values),-1).all(1)
    failed|=(np.abs(np.linalg.norm(pred.quaternion_nb,axis=-1)-1)>1e-5).any(1)
    if failed.any():
        ids=np.flatnonzero(failed)
        pd.DataFrame(dict(window_id=batch.trajectory.window_ids[ids],flight_id=batch.trajectory.log_ids[ids],seed=seed,condition=condition)).to_csv(ART/f'failed_{condition}_seed{seed}.csv',index=False)
        np.savez_compressed(ART/f'failed_{condition}_seed{seed}_context.npz',controls=batch.trajectory.controls[ids],dt=batch.trajectory.dt_s[ids])
        raise ValueError(f'{failed.sum()} invalid predictions, no origins discarded; stopping')
    return endpoint_metrics(pred,batch.trajectory,model='B2_StandardGRU',seed=seed)


def run_inference():
    p,batches=load_inputs();batch=batches['validation'];held=hold_controls(batch)
    require(torch.cuda.is_available(),'CUDA required for formal inference')
    require(array_digest(batch)==p['validation_array_sha256'],'input arrays changed')
    checks={};error_arrays={}
    for seed in SEEDS:
        model=model_for(seed,p);state={k:v.clone() for k,v in model.state_dict().items()}
        actual=frozen_prediction(seed,p,batch)
        actual_endpoint=validate_prediction(actual,batch,seed,'Actual')
        small=subset(batch,128)
        replay=predict_history_trajectory_model(model,small,use_history=True,batch_size=128,device=DEVICE)
        parity={}
        for key,values in vars(replay).items():
            expected=getattr(actual,key)[:128]
            np.testing.assert_allclose(values,expected,rtol=RTOL,atol=ATOL)
            parity[key]=float(np.max(np.abs(values-expected)))
        leakage=causality_check(model,subset(held,4),DEVICE)
        # Synthetic constant-command contract: repeated hold is idempotent, same history.
        constant=subset(held,4)
        a=predict_history_trajectory_model(model,constant,use_history=True,batch_size=128,device=DEVICE)
        b=predict_history_trajectory_model(model,hold_controls(constant),use_history=True,batch_size=128,device=DEVICE)
        for key,value in vars(a).items():np.testing.assert_array_equal(value,getattr(b,key))
        hold=predict_history_trajectory_model(model,held,use_history=True,batch_size=128,device=DEVICE)
        np.savez_compressed(ART/f'Hold_seed{seed}_predictions.npz',**vars(hold),window_ids=batch.trajectory.window_ids.astype(str))
        hold_endpoint=validate_prediction(hold,batch,seed,'Hold')
        for key,value in vars(hold).items():np.testing.assert_allclose(value[:,:2],getattr(actual,key)[:,:2],rtol=RTOL,atol=ATOL)
        constant_mask=np.all(batch.trajectory.controls==batch.trajectory.controls[:,:1],axis=(1,2))
        for key,value in vars(hold).items():
            np.testing.assert_allclose(value[constant_mask],getattr(actual,key)[constant_mask],rtol=RTOL,atol=ATOL)
        for key,value in model.state_dict().items():require(torch.equal(value,state[key]),'model buffer/weights mutated')
        _,summary,_=aggregate_flights(actual_endpoint)
        expected=pd.read_csv(S3/'per_seed_summary.csv').query('seed==@seed').sort_values(['cohort','horizon_s'])
        np.testing.assert_allclose(summary[list(METRICS)],expected[list(METRICS)],rtol=1e-12,atol=1e-12)
        for condition,pred,endpoint in [('Actual',actual,actual_endpoint),('Hold',hold,hold_endpoint)]:
            errors=squared_errors(pred,batch.trajectory.truth)
            # New full-curve helper must agree with the frozen endpoint evaluator.
            for horizon,k in HORIZONS.items():
                e=endpoint[endpoint.horizon_s==horizon]
                sourcecols=['position_error_m','velocity_error_m_s','attitude_error_deg','body_rate_error_rad_s']
                np.testing.assert_allclose(np.sqrt(errors[:,k-1]),e[sourcecols],rtol=1e-10,atol=1e-9)
            error_arrays[f'{condition}_{seed}']=errors
        checks[str(seed)]=dict(actual_reuse_max_abs_error=parity,actual_frozen_summary_parity=True,
            first_step_same=True,naturally_constant_origins=int(constant_mask.sum()),synthetic_constant_bitwise=True,
            **leakage,model_weights_buffers_bitwise_unchanged=True,normalization_unchanged=True,
            predictions_finite=True,quaternions_unit=True,origin_count=2582,states=51)
        print(f'Seed{seed}: Actual reuse verified; Hold inference and sanity passed.',flush=True)
    require(array_digest(batch)==p['validation_array_sha256'],'original batch modified')
    np.savez_compressed(ART/'per_origin_squared_errors.npz',**error_arrays,window_ids=batch.trajectory.window_ids.astype(str))
    write_json(OUT/'sanity_checks.json',dict(status='passed',seeds=checks,
        history_labels_dt_unchanged=True,thresholds_frozen_before_hold=True,all_origins_retained=True,
        sealed_test_opened_this_run=False,reserved_evaluation_opened_this_run=False,training_performed=False))
    summarize(p,batch,error_arrays)
    from report_paper_dynamics_input_diagnostic import generate
    generate(OUT,ART,p,batch)
    verify(p['frozen_input_sha256'])
    (OUT/'git_status_at_completion.txt').write_text(subprocess.check_output(['git','status','--short'],text=True))
    write_json(OUT/'completion.json',dict(status='complete',completed_at_unix=time.time(),
        artifact_sha256={rel(f):file_hash(f) for directory in (OUT,ART) for f in directory.iterdir() if f.is_file() and f.name not in ('completion.json','status.json','run.log')},
        tests=read(OUT/'tests.json'),engineering_checks_passed=True,scientific_pass_threshold=None,
        sealed_test_opened_this_run=False,reserved_evaluation_opened_this_run=False,training_performed=False))


def summarize(p,batch,error_arrays):
    activity=pd.read_csv(ART/'origin_control_activity.csv').query('partition=="validation"')
    logids=batch.trajectory.log_ids;cohorts=np.array([cohort(x) for x in logids]);n=len(logids)
    labels={k:activity[activity.step==k].set_index('window_id').loc[batch.trajectory.window_ids,'group'].to_numpy() for k in HORIZONS.values()}
    rows=[];curves=[]
    for seed in SEEDS:
        for condition in ('Actual','Hold'):
            errors=error_arrays[f'{condition}_{seed}']
            for horizon,k in HORIZONS.items():
                for group in ('ALL',*GROUPS):
                    mask=np.ones(n,bool) if group=='ALL' else labels[k]==group
                    for kind,mse in [('endpoint',errors[:,k-1]),('interval',interval_mse(errors,batch.trajectory.dt_s,k))]:
                        rows.extend(flight_metrics(mse,logids,cohorts,mask,kind=kind,condition=condition,seed=seed,
                            group=group,group_rule='horizon_specific_train_quantiles',step=k,horizon_s=horizon))
            for k in range(1,51):
                for group in ('ALL',*GROUPS):
                    mask=np.ones(n,bool) if group=='ALL' else labels[25]==group
                    curves.extend(flight_metrics(errors[:,k-1],logids,cohorts,mask,kind='evolution',condition=condition,seed=seed,
                        group=group,group_rule='fixed_500ms_train_quantiles',step=k,horizon_s=k*.02))
    per=pd.DataFrame(rows);curve=pd.DataFrame(curves)
    per.to_csv(OUT/'per_flight.csv',index=False);curve.to_csv(ART/'per_flight_error_evolution.csv',index=False)
    seed,multi=aggregate(per);cs,cm=aggregate(curve)
    seed.to_csv(OUT/'per_seed_summary.csv',index=False);multi.to_csv(OUT/'multiseed_summary.csv',index=False)
    multi.query('kind=="interval"').to_csv(OUT/'trajectory_interval_errors.csv',index=False)
    cs.to_csv(ART/'per_seed_error_evolution.csv',index=False)
    cm=cm.merge(pd.read_csv(OUT/'native_elapsed_time.csv'),on=['cohort','group','step','n_origins'],validate='many_to_one')
    cm.to_csv(OUT/'error_evolution.csv',index=False)
    paired_comparisons(seed,per).to_csv(OUT/'paired_differences.csv',index=False)
    # Endpoint native steps equal corresponding ALL evolution rows.
    keys=['condition','seed','cohort','step','metric']
    a=seed.query('kind=="endpoint" and group=="ALL"').sort_values(keys)
    b=cs[(cs.group=='ALL')&cs.step.isin(HORIZONS.values())].sort_values(keys)
    np.testing.assert_allclose(a.value,b.value,rtol=0,atol=0)


def cli():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=['audit','run'],required=True)
    args=parser.parse_args();torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    if args.phase=='audit':audit();return
    require(not (OUT/'completion.json').exists(),'completed/failed analysis exists; no implicit rerun')
    require(read(OUT/'tests.json')['returncode']==0,'tests required')
    try:run_inference()
    except BaseException:
        write_json(OUT/'failure.json',dict(status='failed',error=traceback.format_exc(),automatic_retry=False))
        raise


if __name__=='__main__':cli()
