"""Paper Step4: frozen H26 suffix ablation, nine new runs, no test access.

Reuses Step3 model factory, training stages, objectives, RNG and evaluator.
Only the length of the encoder's input arrays changes; rollout remains50steps.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import torch
import run_paper_standard_gru_multiseed as main
from run_paper_baseline_comparison import write_json, causality_check, _FrequencyLossAdapter
from system_identification.data.trajectory_dataset import CONTROL_COLUMNS
from system_identification.evaluation.paper_baselines import HORIZONS,METRIC_MAP,endpoint_metrics,aggregate_flights,cohort
from system_identification.models.trajectory import TrajectoryPrediction
from system_identification.training.trajectory_main_v1 import assemble_history_trajectory_windows,predict_history_trajectory_model
from system_identification.training.main_v2_increment import train_increment_stage

OUT=ROOT/'docs/analysis/results/paper_history_length_ablation_v1'
ART=ROOT/'artifacts/paper_history_length_ablation_v1'
SCRIPT=Path(__file__).resolve()
HS=(1,5,13,26)
SEEDS=(17,23,42)
JOBS=[(h,s) for h in HS[:-1] for s in SEEDS]
DEVICE=main.DEVICE
require=main.require
sha=main.file_hash
read_json=main.read_json
rel=main.rel


def truncate_history(batch,h):
    """Use exactly the last H legitimate tokens, retaining the SAME trajectory."""
    require(h in HS,'undeclared history length')
    require(batch.history_state_features.shape[1:]==(26,12),'expected frozen H26 source')
    require(batch.history_controls.shape[1:]==(26,4),'history control shape')
    require(batch.history_mask.shape==batch.history_state_features.shape[:2] and batch.history_mask.all(),'missing H26 history')
    return replace(batch,history_state_features=batch.history_state_features[:,-h:],
        history_controls=batch.history_controls[:,-h:],history_mask=batch.history_mask[:,-h:])


def folder(h,s):
    return ART/f'H{h}'/f'seed{s}'


def prediction_path26(seed):
    return main.OLD/f'{main.MODEL}_predictions.npz' if seed==17 else main.ART/f'seed{seed}'/'predictions.npz'


def rows_path26(seed):
    return main.ART/'seed17_per_origin.csv' if seed==17 else main.ART/f'seed{seed}'/'per_origin.csv'


def schedule(p,seed):
    t=p['training_schedule']
    stage_seeds=t['sampling_seeds'][str(seed)]
    require(stage_seeds==[seed,seed+12],'seed contract changed')
    result=[('base',t['base_epochs'],stage_seeds[0],t['learning_rates'][0],False),
            ('continuation',t['continuation_epochs'],stage_seeds[1],t['learning_rates'][1],True)]
    require(result==main.stages(seed),'Step3 schedule/helper mismatch')
    return result


def load():
    p=read_json(OUT/'protocol.json')
    main.verify_pins(p['frozen_input_sha256'])
    source,batches,stats=main.load_inputs()
    require(sha(main.OUT/'protocol.json')==p['source_protocol_sha256'],'source protocol changed')
    return p,source,batches,stats


def read_prediction(path,batch):
    with np.load(path,allow_pickle=False) as f:
        np.testing.assert_array_equal(f['window_ids'],batch.trajectory.window_ids)
        return TrajectoryPrediction(**{k:f[k] for k in vars(batch.trajectory.truth)})


def compare_summary(actual,expected):
    keys=['seed','cohort','horizon_s']
    a=actual.sort_values(keys).reset_index(drop=True)
    b=expected.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[keys],b[keys])
    np.testing.assert_allclose(a[list(METRIC_MAP.values())],b[list(METRIC_MAP.values())],rtol=1e-12,atol=1e-12)


def duration_stats(values):
    values=np.asarray(values,dtype=float)
    require(np.isfinite(values).all() and (values>=0).all(),'invalid native history duration')
    return dict(mean_s=float(values.mean()),median_s=float(np.median(values)),std_s=float(values.std(ddof=1)),
        p05_s=float(np.quantile(values,.05)),p95_s=float(np.quantile(values,.95)),
        min_s=float(values.min()),max_s=float(values.max()),n_origins=len(values))


def audit():
    require(not (OUT/'protocol.json').exists(),'do not overwrite frozen audit')
    OUT.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
    source,batches,stats=main.load_inputs()
    complete=read_json(main.OUT/'completion.json')
    require(complete['status']=='complete' and complete['robustness']['screen_passed'],'source experiment incomplete')
    main.verify_pins(complete['artifact_sha256'])
    pins={**source['frozen_input_sha256'],**complete['artifact_sha256']}
    for path in [main.OUT/'completion.json',main.OUT/'protocol.json',main.ART/'prepared.pt',
                 ROOT/'scripts/run_paper_standard_gru_multiseed.py',ROOT/'scripts/report_paper_standard_gru_multiseed.py']:
        pins[rel(path)]=sha(path)
    for seed in SEEDS:
        schedule(source,seed)
        for path in [main.checkpoint_path(seed),prediction_path26(seed),rows_path26(seed)]:pins[rel(path)]=sha(path)
        cp=torch.load(main.checkpoint_path(seed),map_location='cpu',weights_only=False)
        model=main.build_model(seed,stats)
        model.load_state_dict(cp['state_dict'],strict=True)
        require(sum(v.numel() for v in model.parameters())==21383,'parameter count changed')
        require(main.normalization_hash({k:cp['state_dict'][k].numpy() for k in main.STATS})==source['normalization_sha256'],'normalization changed')
        pred=read_prediction(prediction_path26(seed),batches['validation'])
        rows=endpoint_metrics(pred,batches['validation'].trajectory,model=main.MODEL,seed=seed)
        saved=pd.read_csv(rows_path26(seed))
        main.validate_rows(saved,batches['validation'],seed)
        order=['window_id','horizon_s']
        np.testing.assert_allclose(rows.sort_values(order)[list(METRIC_MAP)],saved.sort_values(order)[list(METRIC_MAP)],rtol=1e-11,atol=1e-10)
        _,summary,_=aggregate_flights(saved)
        expected=pd.read_csv(main.OUT/'per_seed_summary.csv').query('seed == @seed')
        compare_summary(summary,expected)
    duration_rows=[];detailed=[];checks={}
    dataset=ROOT/source['manifest_path'];manifest=read_json(dataset)
    forbidden=set(manifest['split_contract']['assignments']['sealed_test'])|set(manifest['split_contract']['assignments']['reserved_evaluation'])
    for part,count in [('train',28293),('validation',2582)]:
        batch=batches[part]
        require(len(batch.trajectory.window_ids)==count,'origin count changed')
        origins=pd.read_csv(main.BASE/f'{part}_origins.csv')
        np.testing.assert_array_equal(batch.trajectory.window_ids,origins.window_id)
        require(not set(origins.log_id)&forbidden,'forbidden flight')
        require(set(origins.log_id)==set(source[f'{part}_flights']),'flight split changed')
        # Explicit allowlisted partition only; no dataset glob, raw logs or test results.
        sample_path=dataset.parent/f'samples_{part}.parquet'
        require(sha(sample_path)==source['artifact_sha256'][sample_path.name],'sample bytes changed')
        samples=pd.read_parquet(sample_path)
        require(set(samples.split)=={part} and set(samples.log_id)==set(origins.log_id),'sample split mismatch')
        lookup=samples[samples.valid_core].set_index(['log_id','segment_id','sample_in_segment'])
        require(not lookup.index.duplicated().any(),'duplicate sample identity')
        def at(offset,cols):
            index=pd.MultiIndex.from_arrays([origins.log_id,origins.segment_id,origins.start_sample_in_segment+offset])
            return lookup.loc[index,cols].to_numpy()
        timestamps=at(0,'timestamp_us')
        np.testing.assert_array_equal(timestamps,origins.origin_timestamp_us)
        # Every history control token shares the exact original logged sample index.
        for j in range(26):
            np.testing.assert_array_equal(batch.history_controls[:,j],at(j-25,list(CONTROL_COLUMNS)))
        np.testing.assert_array_equal(batch.history_controls[:,-1],batch.trajectory.controls[:,0])
        selected=np.array([0,count//2,count-1])
        windows=pd.read_parquet(dataset.parent/f'windows_{part}.parquet').iloc[selected]
        for h in HS:
            cut=truncate_history(batch,h)
            require(cut.trajectory is batch.trajectory,'future trajectory changed')
            for key in ['history_state_features','history_controls','history_mask']:
                np.testing.assert_array_equal(getattr(cut,key),getattr(batch,key)[:,-h:])
            # Independent original-loader check catches phase-anchor/index/shape errors.
            rebuilt=assemble_history_trajectory_windows(samples,windows,history_steps=h)
            for key in ['history_state_features','history_controls','history_mask']:
                np.testing.assert_array_equal(getattr(cut,key)[selected],getattr(rebuilt,key))
            span=(timestamps-at(1-h,'timestamp_us'))*1e-6
            if h==26:
                np.testing.assert_allclose(span,origins.history_span_s,rtol=0,atol=1e-12)
            frame=pd.DataFrame(dict(partition=part,history_steps=h,window_id=origins.window_id,
                log_id=origins.log_id,cohort=origins.log_id.map(cohort) if part=='validation' else 'train',duration_s=span))
            detailed.append(frame)
            groups=[('ALL',frame)] if part=='train' else [('ALL',frame),*list(frame.groupby('cohort'))]
            for name,g in groups:
                duration_rows.append(dict(history_steps=h,partition=part,cohort=name,**duration_stats(g.duration_s)))
        checks[part]=dict(origins=count,flights=len(set(origins.log_id)),all_H_exact_suffix=True,
            trajectory_object_unchanged=True,controls_match_all26_source_timestamps=True,
            origin_control_equals_first_future_control=True,original_loader_parity_fixed3origins_all_H=True)
    pd.DataFrame(duration_rows).to_csv(OUT/'history_duration_summary.csv',index=False)
    pd.concat(detailed,ignore_index=True).to_csv(ART/'history_duration_per_origin.csv',index=False)
    for path in [SCRIPT,ROOT/'scripts/report_paper_history_length_ablation.py',ROOT/'tests/test_paper_history_length_ablation.py']:
        pins[rel(path)]=sha(path)
    selection=read_json(main.BASE/'representative_selection.json')
    protocol=dict(experiment='paper_history_length_ablation_v1',source_main_model_commit='19546ada11c72f640ef31fc557664e6d02734c6f',
        source_protocol=rel(main.OUT/'protocol.json'),source_protocol_sha256=sha(main.OUT/'protocol.json'),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),branch=source['branch'],
        history_values=list(HS),seeds=list(SEEDS),new_training_runs=[dict(history_steps=h,seed=s) for h,s in JOBS],
        reused_H26=complete['training_runs'],source_frozen_contract=source,
        history_semantics=dict(state_shape='[N,H,12]',control_shape='[N,H,4]',mask_shape='[N,H]',
            indices='origin-(H-1) ... origin inclusive; all tokens legal; no padding in fixed origin set',
            last_token='current origin t; state and control from same sample',
            H1='single-sample context at t; no earlier state/control token; future GRU transition still recurrent',
            encoding='hidden initialized to zero per window, shared GRUCell consumes H tokens through t; first derivative head receives this hidden plus current state/control again, unchanged inherited definition',
            phase_anchor='unchanged relative encoder phase referenced to origin; truncation does not re-anchor'),
        duration_definition='(timestamp_us(origin)-timestamp_us(origin-H+1))*1e-6; H1=0; native dt, no 50Hz resampling',
        duration_statistics='across fixed origins; sample SD ddof=1; NumPy linear p05/p95; train ALL and validation ALL/Sep7/Sep17',
        architecture=source['architecture'],parameter_count_by_H={str(h):21383 for h in HS},
        training_schedule=source['training_schedule'],normalization_sha256=source['normalization_sha256'],
        train_window_sha256=source['train_window_sha256'],validation_origin_sha256=source['validation_origin_sha256'],
        manifest_sha256=source['manifest_sha256'],train_origins=28293,validation_origins=2582,
        horizons_s=source['horizons_s'],primary_horizon_s=.5,metrics=source['metrics'],
        checkpoint_selection=source['checkpoint_selection'],randomness_contract=source['stage_seed_rule'],
        execution=dict(device=DEVICE,concurrent_workers=2,job_order=JOBS,no_automatic_retry=True),
        qualitative_case=dict(**selection,seed=17,histories=[1,13,26],selection='frozen Step1 representative origin, independent of all ablation errors'),
        interpretation='Report all metrics/cohorts/seeds and paired per-flight directions; no model switch. Descriptive classification at ALL500ms dynamics: B if all H13/H26 mean differences within2%; A if H26 best on all3 with >2% H13 gains and all3seeds same direction; C if H5 or H13 beats H26 by >2% on at least2metrics with no >2% regression on third; otherwise Mixed. This is not a significance or equivalence test.',
        sealed_test_opened_this_run=False,reserved_evaluation_opened_this_run=False,
        sealed_status='Only frozen train/validation data and existing open validation predictions; excluded names metadata only.',
        frozen_input_sha256=pins)
    write_json(OUT/'protocol.json',protocol)
    write_json(OUT/'audit.json',dict(status='passed',data=checks,H26_prediction_metric_parity=True,
        normalization_refit=False,parameter_count_all_H=21383,sealed_test_opened_this_run=False,reserved_evaluation_opened_this_run=False))
    (OUT/'git_status_at_launch.txt').write_text(subprocess.check_output(['git','status','--short'],text=True))
    print(pd.DataFrame(duration_rows).to_string(index=False),flush=True)


def check_model(model,source):
    require(sum(p.numel() for p in model.parameters())==21383,'parameter count mismatch; stop')
    require(main.normalization_hash({k:model.state_dict()[k].cpu().numpy() for k in main.STATS})==source['normalization_sha256'],'normalization changed')
    require(all(bool(torch.isfinite(v).all()) for v in model.state_dict().values()),'nonfinite checkpoint')


def preflight():
    p,source,batches,stats=load()
    require(torch.cuda.is_available(),'CUDA unavailable; no CPU fallback')
    checks={}
    for h in HS:
        model=main.build_model(17,stats);check_model(model,source)
        small=truncate_history(main.subset(batches['train'],4),h)
        if h==26:
            model.load_state_dict(torch.load(main.checkpoint_path(17),map_location='cpu',weights_only=False)['state_dict'],strict=True)
        for name,_,seed,lr,frequency in (schedule(source,17) if h!=26 else []):
            main.seed_all(seed)
            wrapped=_FrequencyLossAdapter(model) if frequency else model
            trained,hist=train_increment_stage(wrapped,small,scales=source['increment_scales'],weights=source['increment_weights'],
                device=DEVICE,epochs=1,seed=seed,learning_rate=lr,actuator=frequency,
                steps=source['training_schedule']['rollout_steps'],batch_size=source['batch_size'])
            model=trained.model if frequency else trained
            require(np.isfinite(hist.select_dtypes('number').to_numpy()).all(),'preflight loss nonfinite')
        check_model(model,source)
        val=truncate_history(main.subset(batches['validation'],4),h)
        leakage=causality_check(model,val,DEVICE)
        pred=predict_history_trajectory_model(model,val,use_history=True,batch_size=128,device=DEVICE)
        metrics=endpoint_metrics(pred,val.trajectory,model=main.MODEL,seed=17)
        loss=main.val_loss(model,val,source)
        checks[str(h)]=dict(**leakage,finite_metric_rows=len(metrics),final_objective_smoke=loss,parameter_count=21383)
    write_json(OUT/'preflight.json',dict(status='passed',checks=checks,runtime=main.runtime(17),
        disposable_models_only=True,no_H26_experiment_training=True,protocol_sha256=sha(OUT/'protocol.json')))
    print('GPU preflight passed: all H, both stages, evaluation, label poisoning and prefix parity.',flush=True)


def worker(h,seed):
    require((h,seed) in JOBS,'only nine declared new runs allowed; never train H26')
    dest=folder(h,seed);dest.mkdir(parents=True,exist_ok=False)
    write_json(dest/'status.json',dict(status='starting',pid=os.getpid(),history_steps=h,seed=seed))
    try:
        p,source,batches,stats=load()
        batch=truncate_history(batches['train'],h)
        write_json(dest/'runtime.json',main.runtime(seed))
        model=main.build_model(seed,stats);check_model(model,source)
        for stage,epochs,stage_seed,lr,frequency in schedule(source,seed):
            main.seed_all(stage_seed)
            wrapped=_FrequencyLossAdapter(model) if frequency else model
            def callback(_,history):
                history.to_csv(dest/f'{stage}_history.csv',index=False)
                write_json(dest/'status.json',dict(status='training',pid=os.getpid(),history_steps=h,seed=seed,
                    stage=stage,epoch=int(history.epoch.iloc[-1]),loss=float(history.loss.iloc[-1])))
            trained,history=train_increment_stage(wrapped,batch,scales=source['increment_scales'],weights=source['increment_weights'],
                device=DEVICE,epochs=epochs,seed=stage_seed,learning_rate=lr,actuator=frequency,
                steps=source['training_schedule']['rollout_steps'],batch_size=source['batch_size'],callback=callback)
            model=trained.model if frequency else trained
            main.validate_history(history,epochs);check_model(model,source)
            torch.save(dict(state_dict=model.state_dict(),history_steps=h,seed=seed,completed_stage=stage,
                protocol_sha256=sha(OUT/'protocol.json'),normalization_sha256=source['normalization_sha256']),dest/f'{stage}.pt')
        torch.save(dict(state_dict=model.state_dict(),history_steps=h,seed=seed,final_epoch=65,
            protocol_sha256=sha(OUT/'protocol.json'),normalization_sha256=source['normalization_sha256']),dest/'model.pt')
        write_json(dest/'status.json',dict(status='training_complete',history_steps=h,seed=seed,final_epoch=65,sha256=sha(dest/'model.pt')))
    except BaseException:
        write_json(dest/'status.json',dict(status='failed',history_steps=h,seed=seed,error=traceback.format_exc(),automatic_retry=False))
        raise


def evaluate():
    p,source,batches,stats=load()
    summaries=[];flights=[];runs=[];checks={};examples={}
    selected=p['qualitative_case']['source_index']
    require(str(batches['validation'].trajectory.window_ids[selected])==p['qualitative_case']['window_id'],'representative identity changed')
    for h in HS:
        val=truncate_history(batches['validation'],h)
        for seed in SEEDS:
            if h==26:
                saved=pd.read_csv(rows_path26(seed))
                main.validate_rows(saved,val,seed)
                per,summary,_=aggregate_flights(saved)
                compare_summary(summary,pd.read_csv(main.OUT/'per_seed_summary.csv').query('seed == @seed'))
                run=pd.read_csv(main.OUT/'training_runs.csv').query('seed == @seed').iloc[0].to_dict()
                run.update(history_steps=26,status='reused',sha256=run['checkpoint_sha256'])
                checks[f'H{h}/seed{seed}']=dict(reused=True,metric_parity=True,
                    source_sanity_checks=rel(main.OUT/'sanity_checks.json'))
                if seed==17:pred=read_prediction(prediction_path26(seed),val)
            else:
                dest=folder(h,seed);cp=torch.load(dest/'model.pt',map_location='cpu',weights_only=False)
                require(cp['protocol_sha256']==sha(OUT/'protocol.json') and cp['final_epoch']==65,'checkpoint provenance')
                require(cp['history_steps']==h and cp['seed']==seed,'checkpoint identity')
                model=main.build_model(seed,stats);model.load_state_dict(cp['state_dict'],strict=True);check_model(model,source)
                check=causality_check(model,main.subset(val,3),DEVICE)
                pred=predict_history_trajectory_model(model,val,use_history=True,batch_size=128,device=DEVICE)
                np.savez_compressed(dest/'predictions.npz',**vars(pred),window_ids=val.trajectory.window_ids.astype(str))
                rows=endpoint_metrics(pred,val.trajectory,model=main.MODEL,seed=seed)
                main.validate_rows(rows,val,seed);rows.to_csv(dest/'per_origin.csv',index=False)
                per,summary,_=aggregate_flights(rows)
                histories=[]
                for stage,epochs,*_ in schedule(source,seed):
                    hist=pd.read_csv(dest/f'{stage}_history.csv');main.validate_history(hist,epochs);histories.append(hist)
                run=dict(history_steps=h,seed=seed,checkpoint_path=rel(dest/'model.pt'),sha256=sha(dest/'model.pt'),
                    status='complete',final_epoch=65,best_epoch=None,final_train_loss=float(histories[-1].loss.iloc[-1]),
                    final_val_loss=main.val_loss(model,val,source),training_time_s=sum(float(x.wall_time_s.iloc[-1]) for x in histories))
                checks[f'H{h}/seed{seed}']=dict(**check,parameter_count=21383,normalization_identical=True,
                    n_origins=2582,metric_rows=len(rows),quaternion_max_norm_error=float(np.max(np.abs(np.linalg.norm(pred.quaternion_nb,axis=-1)-1))),
                    finite_losses=True,native_horizon_timing=True)
            summaries.append(summary.assign(history_steps=h));flights.append(per.assign(history_steps=h));runs.append(run)
            if seed==17 and h in (1,13,26):
                for key,value in vars(pred).items():examples[f'H{h}_{key}']=value[selected,:26]
            pd.DataFrame(runs).to_csv(OUT/'training_runs.csv',index=False)
    full=pd.concat(summaries,ignore_index=True);per=pd.concat(flights,ignore_index=True)
    require(len(full)==144 and len(per)==816,'summary sample count mismatch')
    full.to_csv(OUT/'per_seed_summary.csv',index=False)
    per.to_csv(OUT/'per_flight_wide.csv',index=False)
    per.rename(columns={'log_id':'flight_id'}).melt(id_vars=['history_steps','seed','flight_id','cohort','horizon_s','n_windows'],
        value_vars=list(METRIC_MAP.values()),var_name='metric',value_name='value').to_csv(OUT/'per_flight.csv',index=False)
    for key,value in vars(batches['validation'].trajectory.truth).items():examples[f'truth_{key}']=value[selected,:26]
    examples['time_s']=np.r_[0,np.cumsum(batches['validation'].trajectory.dt_s[selected,:25])]
    np.savez_compressed(OUT/'representative_prediction.npz',**examples)
    write_json(OUT/'sanity_checks.json',dict(status='passed',runs=checks,all_H_exact_suffix=True,
        same_train_origins=True,same_validation_origins=True,same_future_targets_controls_dt=True,
        normalization_refit=False,H26_reused=True,sealed_test_opened_this_run=False,reserved_evaluation_opened_this_run=False))
    from report_paper_history_length_ablation import generate
    generate(OUT,p)
    main.verify_pins(p['frozen_input_sha256'])
    (OUT/'git_status_at_completion.txt').write_text(subprocess.check_output(['git','status','--short'],text=True))
    hashes={rel(f):sha(f) for f in OUT.iterdir() if f.is_file() and f.name!='completion.json'}
    for h,s in JOBS:
        hashes.update({rel(f):sha(f) for f in folder(h,s).iterdir() if f.is_file()})
    write_json(OUT/'completion.json',dict(status='complete',completed_runs=12,new_training_runs=9,reused_H26_seeds=list(SEEDS),
        artifact_sha256=hashes,training_runs='training_runs.csv',tests=read_json(OUT/'tests.json'),
        sealed_test_opened_this_run=False,reserved_evaluation_opened_this_run=False,main_model_changed=False,
        next_ablation_started=False,completed_at_unix=time.time()))


def run():
    write_json(ART/'status.json',dict(status='running',pid=os.getpid(),started_at_unix=time.time(),jobs=JOBS))
    try:
        def execute(job):
            h,s=job
            with (ART/f'H{h}_seed{s}.log').open('w') as log:
                child=subprocess.Popen([sys.executable,str(SCRIPT),'--phase','worker','--history',str(h),'--seed',str(s)],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                write_json(ART/f'H{h}_seed{s}_launch.json',dict(history_steps=h,seed=s,pid=child.pid,launched_at_unix=time.time()))
                return dict(history_steps=h,seed=s,returncode=child.wait())
        # Fixed queue; completion wait only, no monitoring or metric-based intervention.
        with ThreadPoolExecutor(max_workers=2) as pool:exits=list(pool.map(execute,JOBS))
        write_json(ART/'worker_exit_codes.json',exits)
        require(all(r['returncode']==0 for r in exits),'one or more runs failed; no automatic retry')
        evaluate()
        write_json(ART/'status.json',dict(status='complete',completion=rel(OUT/'completion.json')))
    except BaseException:
        failure=dict(status='failed',error=traceback.format_exc(),automatic_retry=False,
            sealed_test_opened_this_run=False,reserved_evaluation_opened_this_run=False)
        write_json(ART/'status.json',failure);write_json(OUT/'completion.json',failure)
        raise


def launch():
    require(not (ART/'launch.json').exists(),'already launched; do not rerun implicitly')
    require(read_json(OUT/'tests.json')['returncode']==0,'tests not passed')
    preflight()
    with (ART/'run.log').open('w') as log:
        child=subprocess.Popen([sys.executable,str(SCRIPT),'--phase','run'],cwd=ROOT,stdout=log,
                               stderr=subprocess.STDOUT,start_new_session=True)
    write_json(ART/'launch.json',dict(pid=child.pid,launched_at_unix=time.time(),jobs=JOBS,
        concurrent_workers=2,device=DEVICE,automatic_evaluation_and_report=True,epoch_monitoring=False,
        protocol_sha256=sha(OUT/'protocol.json')))
    print(json.dumps(read_json(ART/'launch.json')),flush=True)


def cli():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['audit','preflight','launch','run','worker','evaluate'],required=True)
    parser.add_argument('--history',type=int,choices=[1,5,13])
    parser.add_argument('--seed',type=int,choices=list(SEEDS))
    args=parser.parse_args();main.configure()
    if args.phase=='worker':worker(args.history,args.seed)
    else:globals()[args.phase]()


if __name__=='__main__':cli()
