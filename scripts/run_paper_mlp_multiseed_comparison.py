"""Paper Step6: two fresh MLP training runs; reuse all frozen comparators."""
from __future__ import annotations
import argparse
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
import run_paper_standard_gru_multiseed as common
from run_paper_baseline_comparison import write_json,build_model as old_factory,_FrequencyLossAdapter,causality_check,STATS
from system_identification.models.paper_baselines import MemorylessTrajectoryModel
from system_identification.models.trajectory import TrajectoryPrediction
from system_identification.evaluation.paper_baselines import endpoint_metrics,aggregate_flights,METRIC_MAP
from system_identification.training.main_v2_increment import train_increment_stage
from system_identification.training.trajectory_main_v1 import predict_history_trajectory_model

OUT=ROOT/'docs/analysis/results/paper_mlp_multiseed_comparison_v1'
ART=ROOT/'artifacts/paper_mlp_multiseed_comparison_v1'
BASE=common.BASE;OLD=common.OLD;S3=common.OUT;A3=common.ART
SCRIPT=Path(__file__).resolve();SEEDS=(17,23,42);DEVICE='cuda:1';MODEL='B1_MLP'
read=common.read_json;sha=common.file_hash;rel=common.rel;require=common.require
SOURCES=['scripts/run_paper_baseline_comparison.py','src/system_identification/models/paper_baselines.py',
 'src/system_identification/models/neural.py','src/system_identification/models/trajectory_main_v1.py',
 'src/system_identification/models/trajectory.py','src/system_identification/training/main_v2_increment.py',
 'src/system_identification/training/trajectory_main_v1.py','src/system_identification/evaluation/paper_baselines.py',
 'src/system_identification/evaluation/trajectory.py']


def build_mlp(seed,stats):
    """Do not call the historical factory: it resets every initialization to17."""
    common.seed_all(seed)
    return MemorylessTrajectoryModel(**stats)


def parameter_hash(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def check_model(model,p):
    require(sum(x.numel() for x in model.parameters())==5703,'MLP architecture mismatch')
    require(all(x.requires_grad for x in model.parameters()),'unexpected frozen parameter')
    require(all(bool(torch.isfinite(x).all()) for x in model.state_dict().values()),'nonfinite weights/buffers')
    require(common.normalization_hash({k:model.state_dict()[k].cpu().numpy() for k in STATS})==p['normalization_sha256'],'normalization changed')


def stages(p,seed):
    schedule=p['training_schedule']
    require(schedule['sampling_seeds'][str(seed)]==[seed,seed+12],'stage seeds changed')
    result=[('base',schedule['base_epochs'],seed,schedule['learning_rates'][0],False),
            ('continuation',schedule['continuation_epochs'],seed+12,schedule['learning_rates'][1],True)]
    require(result==common.stages(seed),'stage protocol differs from frozen implementation')
    return result


def audit():
    require(not (OUT/'protocol.json').exists(),'protocol already frozen')
    OUT.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
    base=read(BASE/'protocol.json');c1=read(BASE/'completion.json');s3=read(S3/'protocol.json');c3=read(S3/'completion.json')
    require(c1['status']=='complete' and c3['status']=='complete','source experiment incomplete')
    pins={}
    def pin(path,expected=None):
        value=sha(path);require(expected is None or value==expected,f'required artifact mismatch: {path}')
        pins[rel(path)]=value
    for name in ['protocol.json','summary.csv','per_flight.csv']:
        pin(BASE/name,c1['artifact_sha256'][name])
    for name in ['protocol.json','per_seed_summary.csv','per_flight_wide.csv','training_runs.csv']:
        pin(S3/name,c3['artifact_sha256'][rel(S3/name)])
    for source in SOURCES:pin(ROOT/source,base['source_sha256'][source])
    pin(ROOT/'scripts/run_paper_standard_gru_multiseed.py')
    manifest=ROOT/base['manifest_path'];pin(manifest,base['manifest_sha256'])
    assignments=read(manifest)['split_contract']['assignments']
    require(not read(manifest)['split_contract']['sealed_test_opened'],'sealed partition marked open')
    excluded=set(assignments['sealed_test'])|set(assignments['reserved_evaluation'])
    require(not set(assignments['train'])&set(assignments['validation']),'split overlap')
    require(not (set(assignments['train'])|set(assignments['validation']))&excluded,'sealed/reserved flight overlap')
    for part in ('train','validation'):
        require(assignments[part]==base[f'{part}_flights']==s3[f'{part}_flights'],'split mismatch')
        for kind in ('samples','windows'):
            name=f'{kind}_{part}.parquet';pin(manifest.parent/name,base['artifact_sha256'][name])
        pin(BASE/f'{part}_origins.csv',s3['train_window_sha256'] if part=='train' else s3['validation_origin_sha256'])
    pin(A3/'prepared.pt',s3['prepared_sha256'])
    batches=torch.load(A3/'prepared.pt',map_location='cpu',weights_only=False)
    for part,n in [('train',28293),('validation',2582)]:
        b=batches[part];origins=pd.read_csv(BASE/f'{part}_origins.csv')
        require(len(b.trajectory.window_ids)==n,'origin count mismatch')
        np.testing.assert_array_equal(b.trajectory.window_ids,origins.window_id)
        np.testing.assert_array_equal(b.trajectory.log_ids,origins.log_id)
        require(b.history_mask.all(),'incomplete common origin history')
    cp_path=OLD/'B1_MLP.pt';pin(cp_path,base['checkpoint_sha256'][MODEL])
    cp=torch.load(cp_path,map_location='cpu',weights_only=False)
    require(cp['schedule']=='matched-budget' and cp['identity']['manifest_sha256']==base['manifest_sha256'],'B1 seed17 identity/schedule mismatch')
    for source in SOURCES:require(cp['source_sha256'][source]==base['source_sha256'][source],'B1 source definition changed')
    stats={k:cp['state_dict'][k].numpy() for k in STATS}
    for k in STATS:np.testing.assert_array_equal(stats[k],np.asarray(base['normalization']['values'][k],dtype=np.float32))
    normhash=common.normalization_hash(stats);require(normhash==s3['normalization_sha256'],'B1/H26 normalization difference')
    for field,value in base['training_schedule'].items():
        if field!='sampling_seeds':require(value==s3['training_schedule'][field],f'B1/H26 schedule mismatch {field}')
    require(base['actual_training_seeds'][MODEL]=={'initialization':17,'stage_sampling':[17,29]},'historical seeds differ')
    a=build_mlp(17,stats);b=old_factory(MODEL,stats)
    for k,v in a.state_dict().items():require(torch.equal(v,b.state_dict()[k]),'seed17 factory parity failed')
    a.load_state_dict(cp['state_dict'],strict=True)
    initial_hashes={str(seed):parameter_hash(build_mlp(seed,stats)) for seed in SEEDS}
    require(len(set(initial_hashes.values()))==3,'random initializations unexpectedly identical')
    require(parameter_hash(build_mlp(23,stats))==initial_hashes['23'],'same seed not reproducible')
    for stage,n in [('base',40),('continuation',25)]:
        path=OLD/f'B1_MLP_{stage}_history.csv';pin(path);common.validate_history(pd.read_csv(path),n)
    pin(OLD/'B1_MLP_continuation.pt')
    final_stage=torch.load(OLD/'B1_MLP_continuation.pt',map_location='cpu',weights_only=False)
    for k,v in cp['state_dict'].items():require(torch.equal(v,final_stage['state_dict'][k]),'seed17 is not last continuation checkpoint')
    # Reuse original prediction/metrics, rather than running historical training again.
    pin(OLD/'B1_MLP_predictions.npz');pin(OLD/'per_origin.csv')
    old_rows=pd.read_csv(OLD/'per_origin.csv')
    val=batches['validation']
    mlp_rows=old_rows[old_rows.model==MODEL].reset_index(drop=True)
    with np.load(OLD/'B1_MLP_predictions.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(f['window_ids'],val.trajectory.window_ids)
        pred=TrajectoryPrediction(**{k:f[k] for k in vars(val.trajectory.truth)})
    recomputed=endpoint_metrics(pred,val.trajectory,model=MODEL,seed=17)
    keys=['window_id','horizon_s']
    np.testing.assert_allclose(recomputed.sort_values(keys)[list(METRIC_MAP)],mlp_rows.sort_values(keys)[list(METRIC_MAP)],rtol=1e-11,atol=1e-10)
    for name in ['B0_ConstantVelocity',MODEL]:
        rows=old_rows[old_rows.model==name]
        require(len(rows)==2582*4 and not rows.duplicated(keys).any(),'historical metric identity/count')
        for _,g in rows.groupby('horizon_s'):require(set(g.window_id)==set(val.trajectory.window_ids),'historical origin mismatch')
        _,summary,_=aggregate_flights(rows)
        expected=pd.read_csv(BASE/'summary.csv').query('model==@name').sort_values(['cohort','horizon_s'])
        np.testing.assert_allclose(summary[list(METRIC_MAP.values())],expected[list(METRIC_MAP.values())],rtol=1e-12,atol=1e-12)
    gru_runs=pd.read_csv(S3/'training_runs.csv')
    for row in gru_runs.itertuples():pin(ROOT/row.checkpoint_path,row.checkpoint_sha256)
    for path in [SCRIPT,ROOT/'tests/test_paper_mlp_multiseed_comparison.py']:pin(path)
    protocol=dict(experiment='paper_mlp_multiseed_comparison_v1',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        source_baseline_protocol=rel(BASE/'protocol.json'),source_gru_protocol=rel(S3/'protocol.json'),
        mlp_class='MemorylessTrajectoryModel; original16->64ReLU->64ReLU->7; dropout0; final linear weight/bias zero; inherited original constructor random draw order',
        mlp_parameter_count=5703,gru_parameter_count=21383,all_mlp_parameters_trainable=True,
        model_names={'B0':'Constant velocity/body-rate extrapolation (deterministic)','B1':'MLP','B2':'Standard GRU / H26 (main model)'},
        trained_seeds=[23,42],reused_mlp_seed17=rel(cp_path),reused_gru_runs=gru_runs.to_dict(orient='records'),
        reused_B0='original deterministic per-flight results; no seed replication or seedSD',seeds=list(SEEDS),
        initialization='fresh random MLP for each new seed; never load learned seed17 weights; load only six frozen normalization buffers',
        initial_parameter_sha256=initial_hashes,seed17_factory_bitwise_parity=True,
        random_seed_rule='initialization/base=s, continuation=s+12;17/29,23/35,42/54; CPU torch.Generator(stage_seed) randperm, no DataLoader',
        training_schedule=s3['training_schedule'],increment_scales=base['ours_training_protocol']['scales'],
        increment_weights=base['ours_training_protocol']['weights'],batch_size=256,epochs=65,updates_per_epoch=111,total_updates=7215,
        optimizer=s3['optimizer'],checkpoint_rule='last epoch65, both stages all parameters trained; no validation selection, early stop or retry',
        objective=base['training_schedule']['objective'],normalization_sha256=normhash,normalization_source=rel(cp_path),normalization_refit=False,
        manifest_sha256=base['manifest_sha256'],manifest_path=base['manifest_path'],
        train_flights=base['train_flights'],validation_flights=base['validation_flights'],excluded_flights=base['excluded_sealed_flights'],
        train_origins=28293,validation_origins=2582,train_window_sha256=s3['train_window_sha256'],validation_origin_sha256=s3['validation_origin_sha256'],
        prepared_sha256=s3['prepared_sha256'],future_inputs='Actual commands only; future physical states/phase/frequency labels only',
        frames=base['frames'],phase_contract=base['model_phase_source_detail'],history='sameH26-eligible origins;MLP ignores history;GRU unchangedH26',
        horizons_s=[.1,.2,.5,1.],native_steps=[5,10,25,50],primary_horizon_s=.5,timing_contract=base['timing_contract'],
        metrics=base['metrics'],aggregation='per-flight RMSE, equal-flight macro mean,3seed mean/sampleSD ddof1;B0 deterministic no seedSD; no ensemble',
        gains='MLP-GRU; relative%=100*(MLP-GRU)/MLP on macro errors, zero denominator undefined; matched seed labels not same initial weights/random trajectories',
        scientific_threshold=None,family_comparison_not_history_causal_ablation=True,
        execution=dict(device=DEVICE,max_concurrent_new_runs=2,no_automatic_retry=True,deterministic_algorithms=True,cudnn_benchmark=False,threads=4),
        runtime_reference=common.runtime(23),frozen_input_sha256=pins,
        historical_registry_difference='Not audited or repaired; explicit required manifest/cache and origin identities checked.',
        old_history_classification_unchanged='Mixed',future_control_conclusion_unchanged=True,
        sealed_test_opened_this_run=False,reserved_evaluation_opened_this_run=False)
    # Historical best_epoch is absent, not a NaN JSON number.
    for run in protocol['reused_gru_runs']:
        run['best_epoch']=None
    check_model(a,protocol)
    write_json(OUT/'protocol.json',protocol)
    write_json(OUT/'audit.json',dict(status='passed',seed17_reusable=True,seed17_last_checkpoint_verified=True,
        seed17_factory_bitwise_parity=True,new_seed_initializations_distinct=True,parameter_count=5703,
        no_normalization_fit=True,train_origins=28293,validation_origins=2582,full_cached_data_identity_verified=True))
    (OUT/'git_status_at_launch.txt').write_text(subprocess.check_output(['git','status','--short'],text=True))
    print('Audit passed: B1seed17 reusable; correct initialization seeds; frozen matched-budget/data contract.',flush=True)


def load():
    p=read(OUT/'protocol.json');common.verify_pins(p['frozen_input_sha256'])
    batches=torch.load(A3/'prepared.pt',map_location='cpu',weights_only=False)
    cp=torch.load(OLD/'B1_MLP.pt',map_location='cpu',weights_only=False)
    return p,batches,{k:cp['state_dict'][k].numpy() for k in STATS}


def preflight():
    p,batches,stats=load();require(torch.cuda.is_available(),'CUDA required')
    model=build_mlp(23,stats);check_model(model,p)
    small=common.subset(batches['train'],4)
    checks=[]
    for stage,_,seed,lr,frequency in stages(p,23):
        common.seed_all(seed);wrapped=_FrequencyLossAdapter(model) if frequency else model
        trained,history=train_increment_stage(wrapped,small,scales=p['increment_scales'],weights=p['increment_weights'],
            device=DEVICE,epochs=1,seed=seed,learning_rate=lr,actuator=frequency,steps=50,batch_size=256)
        model=trained.model if frequency else trained;check_model(model,p)
        require(np.isfinite(history.select_dtypes('number').to_numpy()).all(),'preflight nonfinite')
        checks.append(dict(stage=stage,loss=float(history.loss.iloc[-1])))
    val=common.subset(batches['validation'],4)
    leak=causality_check(model,val,DEVICE)
    pred=predict_history_trajectory_model(model,val,use_history=True,batch_size=128,device=DEVICE)
    endpoint_metrics(pred,val.trajectory,model=MODEL,seed=23)
    common.val_loss(model,val,p)
    # Source seed17 fixed small-batch prediction parity, not a full re-evaluation.
    original=build_mlp(17,stats);original.load_state_dict(torch.load(OLD/'B1_MLP.pt',map_location='cpu',weights_only=False)['state_dict'])
    replay=predict_history_trajectory_model(original,common.subset(batches['validation'],128),use_history=True,batch_size=128,device=DEVICE)
    with np.load(OLD/'B1_MLP_predictions.npz') as f:
        for k,v in vars(replay).items():np.testing.assert_allclose(v,f[k][:128],rtol=1e-6,atol=2e-5)
    write_json(OUT/'preflight.json',dict(status='passed',stages=checks,leakage=leak,seed17_fixed128_prediction_parity=True,
        disposable_model_only=True,runtime=common.runtime(23)))
    print('GPU preflight passed; both stages/evaluator and historical seed17 parity.',flush=True)


def worker(seed):
    require(seed in (23,42),'only new MLP seeds23/42 permitted')
    dest=ART/f'seed{seed}';dest.mkdir(exist_ok=False)
    try:
        p,batches,stats=load();model=build_mlp(seed,stats);check_model(model,p)
        require(parameter_hash(model)==p['initial_parameter_sha256'][str(seed)],'initialization mismatch')
        write_json(dest/'runtime.json',common.runtime(seed))
        write_json(dest/'config.json',dict(model=MODEL,seed=seed,initialization='fresh random, no warm-start',
            initial_parameter_sha256=parameter_hash(model),normalization_sha256=p['normalization_sha256'],
            protocol_sha256=sha(OUT/'protocol.json'),parameter_count=5703))
        for stage,epochs,stage_seed,lr,frequency in stages(p,seed):
            common.seed_all(stage_seed);wrapped=_FrequencyLossAdapter(model) if frequency else model
            def callback(_,history):
                history.to_csv(dest/f'{stage}_history.csv',index=False)
                write_json(dest/'status.json',dict(status='training',seed=seed,pid=os.getpid(),stage=stage,
                    epoch=int(history.epoch.iloc[-1]),loss=float(history.loss.iloc[-1]),wall_time_s=float(history.wall_time_s.iloc[-1])))
            trained,history=train_increment_stage(wrapped,batches['train'],scales=p['increment_scales'],weights=p['increment_weights'],
                device=DEVICE,epochs=epochs,seed=stage_seed,learning_rate=lr,actuator=frequency,steps=50,batch_size=256,callback=callback)
            model=trained.model if frequency else trained;common.validate_history(history,epochs);check_model(model,p)
            torch.save(dict(state_dict=model.state_dict(),seed=seed,completed_stage=stage,model=MODEL,
                protocol_sha256=sha(OUT/'protocol.json')),dest/f'{stage}.pt')
        torch.save(dict(state_dict=model.state_dict(),seed=seed,model=MODEL,final_epoch=65,
            protocol_sha256=sha(OUT/'protocol.json'),normalization_sha256=p['normalization_sha256']),dest/'model.pt')
        write_json(dest/'status.json',dict(status='training_complete',seed=seed,final_epoch=65,checkpoint_sha256=sha(dest/'model.pt')))
    except BaseException:
        write_json(dest/'status.json',dict(status='failed',seed=seed,error=traceback.format_exc(),automatic_retry=False));raise


def train():
    write_json(ART/'status.json',dict(status='training',pid=os.getpid(),seeds=[23,42]))
    jobs=[]
    for seed in (23,42):
        with (ART/f'seed{seed}.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,str(SCRIPT),'--phase','worker','--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
        jobs.append((seed,proc))
    write_json(ART/'workers.json',{str(seed):proc.pid for seed,proc in jobs})
    exits={str(seed):proc.wait() for seed,proc in jobs}
    write_json(ART/'training_completion.json',dict(status='complete' if all(x==0 for x in exits.values()) else 'failed',exit_codes=exits))
    require(all(x==0 for x in exits.values()),'worker failure; no automatic retry')
    write_json(ART/'status.json',dict(status='training_complete_evaluation_pending'))


def launch():
    require(not (ART/'launch.json').exists(),'already launched; no retry')
    require(read(OUT/'tests.json')['returncode']==0,'tests required')
    preflight()
    with (ART/'training.log').open('w') as log:
        proc=subprocess.Popen([sys.executable,str(SCRIPT),'--phase','train'],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,start_new_session=True)
    write_json(ART/'launch.json',dict(pid=proc.pid,seeds=[23,42],launched_at_unix=time.time(),
        protocol_sha256=sha(OUT/'protocol.json'),evaluation='assistant will continue to evaluation/report after training; not a final task completion'))
    print(json.dumps(read(ART/'launch.json')),flush=True)


def evaluate():
    require(read(ART/'training_completion.json')['status']=='complete','two trainings not complete')
    require(not (OUT/'completion.json').exists(),'completed experiment must not be overwritten')
    p,batches,stats=load();val=batches['validation'];rows=[];runs=[];checks={}
    old=pd.read_csv(OLD/'per_origin.csv')
    mlp17=old[old.model==MODEL];per0,summary0,_=aggregate_flights(old[old.model=='B0_ConstantVelocity'])
    per0['seed']=None;summary0['seed']=None
    rows.append(mlp17)
    historical=[]
    for stage,epochs,*_ in stages(p,17):
        d=pd.read_csv(OLD/f'B1_MLP_{stage}_history.csv');common.validate_history(d,epochs);historical.append(d)
    runs.append(dict(model=MODEL,seed=17,status='reused',checkpoint_path=rel(OLD/'B1_MLP.pt'),checkpoint_sha256=sha(OLD/'B1_MLP.pt'),
        final_epoch=65,final_train_loss=float(historical[-1].loss.iloc[-1]),final_val_loss=None,
        final_val_loss_note='not historically recorded; no new full seed17 inference',training_time_s=sum(float(d.wall_time_s.iloc[-1]) for d in historical)))
    for seed in (23,42):
        dest=ART/f'seed{seed}';cp=torch.load(dest/'model.pt',map_location='cpu',weights_only=False)
        require(cp['seed']==seed and cp['final_epoch']==65 and cp['protocol_sha256']==sha(OUT/'protocol.json'),'checkpoint protocol mismatch')
        model=build_mlp(seed,stats);model.load_state_dict(cp['state_dict'],strict=True);check_model(model,p)
        state={k:v.clone() for k,v in model.state_dict().items()}
        check=causality_check(model,common.subset(val,4),DEVICE)
        pred=predict_history_trajectory_model(model,val,use_history=True,batch_size=128,device=DEVICE)
        np.savez_compressed(dest/'predictions.npz',**vars(pred),window_ids=val.trajectory.window_ids.astype(str))
        frame=endpoint_metrics(pred,val.trajectory,model=MODEL,seed=seed)
        require(len(frame)==10328 and not frame.duplicated(['window_id','horizon_s']).any(),'origin mismatch')
        for _,g in frame.groupby('horizon_s'):require(set(g.window_id)==set(val.trajectory.window_ids),'missing evaluation origin')
        frame.to_csv(dest/'per_origin.csv',index=False);rows.append(frame)
        histories=[]
        for stage,epochs,*_ in stages(p,seed):
            d=pd.read_csv(dest/f'{stage}_history.csv');common.validate_history(d,epochs);histories.append(d)
        runs.append(dict(model=MODEL,seed=seed,status='complete',checkpoint_path=rel(dest/'model.pt'),checkpoint_sha256=sha(dest/'model.pt'),
            final_epoch=65,final_train_loss=float(histories[-1].loss.iloc[-1]),final_val_loss=common.val_loss(model,val,p),
            final_val_loss_note='post-training report only; never selected checkpoint',training_time_s=sum(float(d.wall_time_s.iloc[-1]) for d in histories)))
        for k,v in model.state_dict().items():require(torch.equal(v,state[k]),'evaluation modified model')
        checks[str(seed)]=dict(**check,finite_losses_gradients_parameters_predictions=True,normalization_unchanged=True,
            all_parameters_trainable=True,parameter_count=5703,origins=2582,metric_rows=len(frame),native_timing=True,
            max_quaternion_norm_error=float(np.max(abs(np.linalg.norm(pred.quaternion_nb,axis=-1)-1))),last_epoch=65,updates=7215)
    per1,sum1,_=aggregate_flights(pd.concat(rows,ignore_index=True))
    per2=pd.read_csv(S3/'per_flight_wide.csv');sum2=pd.read_csv(S3/'per_seed_summary.csv')
    per=pd.concat([per0,per1,per2],ignore_index=True);summary=pd.concat([summary0,sum1,sum2],ignore_index=True)
    require(len(per)==476 and len(summary)==84,'comparison sample count mismatch')
    per.rename(columns={'log_id':'flight_id'}).to_csv(OUT/'per_flight_wide.csv',index=False)
    per.rename(columns={'log_id':'flight_id'}).melt(id_vars=['model','seed','flight_id','cohort','horizon_s','n_windows'],
        value_vars=list(METRIC_MAP.values()),var_name='metric',value_name='value').to_csv(OUT/'per_flight.csv',index=False)
    summary.to_csv(OUT/'per_seed_summary.csv',index=False)
    for row in pd.read_csv(S3/'training_runs.csv').to_dict(orient='records'):
        row.update(model='B2_StandardGRU',status='reused');runs.append(row)
    pd.DataFrame(runs).to_csv(OUT/'training_runs.csv',index=False)
    write_json(OUT/'sanity_checks.json',dict(status='passed',new_seeds=checks,seed17_factory_parity=True,
        seed17_reused_untouched=True,H26_reused_untouched=True,B0_deterministic_no_seedSD=True,
        common_origins_controls_truth_dt=True,history_classification_unchanged='Mixed',future_control_conclusion_unchanged=True,
        sealed_test_opened_this_run=False,reserved_evaluation_opened_this_run=False))
    from report_paper_mlp_multiseed_comparison import generate
    generate(OUT,p)
    common.verify_pins(p['frozen_input_sha256'])
    (OUT/'git_status_at_completion.txt').write_text(subprocess.check_output(['git','status','--short'],text=True))
    outputs={rel(f):sha(f) for f in OUT.iterdir() if f.is_file() and f.name!='completion.json'}
    for seed in (23,42):outputs.update({rel(f):sha(f) for f in (ART/f'seed{seed}').iterdir() if f.is_file()})
    write_json(OUT/'completion.json',dict(status='complete',new_training_runs=2,mlp_seed17_reused=True,gru_all_seeds_reused=True,
        B0_reused=True,tests=read(OUT/'tests.json'),artifact_sha256=outputs,
        report_script_sha256=sha(ROOT/'scripts/report_paper_mlp_multiseed_comparison.py'),
        sealed_test_opened_this_run=False,reserved_evaluation_opened_this_run=False,no_other_experiment_started=True))
    write_json(ART/'status.json',dict(status='complete',completion=rel(OUT/'completion.json')))


def cli():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['audit','preflight','launch','train','worker','evaluate'],required=True)
    parser.add_argument('--seed',type=int,choices=[23,42]);args=parser.parse_args();common.configure()
    if args.phase=='worker':worker(args.seed)
    else:globals()[args.phase]()


if __name__=='__main__':cli()
