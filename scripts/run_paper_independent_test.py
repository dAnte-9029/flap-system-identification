"""Register before opening authorized Sep8/Sep19; inference only, no training."""
from __future__ import annotations
import argparse,copy,json,os,sys,subprocess,traceback
from pathlib import Path
from datetime import datetime,timezone
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import torch,yaml
from pyulog import ULog
from run_paper_standard_gru_multiseed import subset,normalization_hash,configure,runtime
from run_paper_baseline_comparison import STATS,causality_check
from run_paper_history_length_ablation import truncate_history
from run_paper_dynamics_input_diagnostic import array_digest,stats_distribution
from build_expanded_september import windows
from system_identification.data.september_trajectory import file_hash,observed_phase,hall_diagnostics,CRITICAL_TOPICS
from system_identification.data.trajectory_dataset import extract_trajectory_samples,_duration_s
from system_identification.data.ulg_audit import _dataset
from system_identification.models.trajectory import TrajectoryPrediction,ConstantTwistPredictor
from system_identification.models.trajectory_main_v1 import CausalHistoryTrajectoryModel
from system_identification.models.paper_baselines import MemorylessTrajectoryModel
from system_identification.training.trajectory_main_v1 import assemble_history_trajectory_windows,predict_history_trajectory_model
from system_identification.evaluation.future_control_diagnostic import hold_controls,excitation,assign_groups,GROUPS
from paper_independent_test_core import *
OUT=ROOT/'docs/analysis/results/paper_independent_test_v1';ART=ROOT/'artifacts/paper_independent_test_v1'
RESULTS=ROOT/'docs/analysis/results';DEVICE='cuda:1';SCRIPT=Path(__file__).resolve()
SOURCE_NAMES=['paper_mlp_multiseed_comparison_v1','paper_standard_gru_multiseed_v1','paper_history_length_ablation_v1','paper_dynamics_input_diagnostic_v1']
DATES={'Sep8':'sealed_test','Sep19':'reserved_evaluation'}
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def rel(p):return str(Path(p).relative_to(ROOT))
def require(ok,msg):
    if not ok:raise ValueError(msg)
def verify(pins):
    for path,h in pins.items():require(file_hash(ROOT/path)==h,'frozen input changed: '+path)
def model_load(row):
    cp=torch.load(ROOT/row['checkpoint_path'],map_location='cpu',weights_only=False)
    stats={k:cp['state_dict'][k].numpy() for k in STATS}
    model=MemorylessTrajectoryModel(**stats) if row['model']=='MLP' else CausalHistoryTrajectoryModel(**stats,hidden_size=64,use_controls=True)
    model.load_state_dict(cp['state_dict'],strict=True);model.eval()
    require(sum(p.numel() for p in model.parameters())==row['parameter_count'],'architecture mismatch')
    require(normalization_hash(stats)==row['normalization_sha256'],'normalization mismatch')
    return model


def register():
    require(not (OUT/'protocol.json').exists(),'registration already exists')
    OUT.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
    pins={};sources={}
    def pin(p,h=None):
        v=file_hash(p);require(h is None or v==h,'source mismatch '+str(p));pins[rel(p)]=v
    for name in SOURCE_NAMES:
        d=RESULTS/name;sources[name]=read(d/'protocol.json')
        for f in ['protocol.json','report.md','review_report.md','completion.json']:
            if (d/f).exists():pin(d/f)
    mlp=sources[SOURCE_NAMES[0]];main=sources[SOURCE_NAMES[1]];hist=sources[SOURCE_NAMES[2]];control=sources[SOURCE_NAMES[3]]
    manifest=ROOT/main['manifest_path'];pin(manifest,main['manifest_sha256']);m=read(manifest)
    configpath=ROOT/'configs/data/trajectory_september_v2.yaml';pin(configpath);cfg=yaml.safe_load(configpath.read_text())
    parent=ROOT/'dataset/trajectory_v2_september_phase_observed/manifest.json';pin(parent,m['parent_manifest_sha256']);parentmeta=read(parent)
    for name,h in parentmeta['provenance']['builder_source_sha256'].items():pin(ROOT/name,h)
    pin(ROOT/'scripts/build_expanded_september.py',m['builder_sha256'])
    for name in ['src/system_identification/evaluation/future_control_diagnostic.py','scripts/run_paper_history_length_ablation.py',
                 'scripts/run_paper_dynamics_input_diagnostic.py','scripts/run_paper_standard_gru_multiseed.py','scripts/run_paper_baseline_comparison.py',
                 'scripts/paper_independent_test_core.py','scripts/run_paper_independent_test.py','tests/test_paper_independent_test.py']:
        pin(ROOT/name)
    for name,h in read(RESULTS/'paper_baseline_comparison_v1/protocol.json')['source_sha256'].items():
        if name.startswith('src/'):pin(ROOT/name,h)
    rows=[]
    for source,history in [(RESULTS/SOURCE_NAMES[0]/'training_runs.csv',False),(RESULTS/SOURCE_NAMES[2]/'training_runs.csv',True)]:
        pin(source)
        for row in pd.read_csv(source).to_dict('records'):
            if not history and row['model']!='B1_MLP':continue
            h=int(row['history_steps']) if history else 1
            sha=row.get('sha256') if history else row['checkpoint_sha256']
            if not isinstance(sha,str):sha=row['checkpoint_sha256']
            path=ROOT/row['checkpoint_path'];pin(path,sha)
            item=dict(model=f'H{h}' if history else 'MLP',history_steps=h,seed=int(row['seed']),checkpoint_path=rel(path),checkpoint_sha256=sha,
                      parameter_count=21383 if history else 5703,normalization_sha256=main['normalization_sha256'])
            model_load(item);rows.append(item)
    require(len(rows)==15 and len({r['checkpoint_path'] for r in rows})==15,'expected15 checkpoints')
    thresholds_path=RESULTS/SOURCE_NAMES[3]/'control_group_thresholds.csv';pin(thresholds_path)
    thresholds=pd.read_csv(thresholds_path).to_dict('records')
    require(file_hash(thresholds_path)==read(RESULTS/SOURCE_NAMES[3]/'completion.json')['artifact_sha256'][rel(thresholds_path)],'threshold hash changed')
    cache=ROOT/'artifacts/paper_standard_gru_multiseed_v1/prepared.pt';pin(cache,main['prepared_sha256'])
    validation=torch.load(cache,map_location='cpu',weights_only=False)['validation'];small=subset(validation,4)
    # Only old validation data: interface, immutable inputs, all15 models load/poison/prefix.
    checks={}
    for row in rows:
        model=model_load(row);b=truncate_history(small,row['history_steps'])
        checks[f'{row["model"]}_{row["seed"]}']=causality_check(model,b,DEVICE)
    model=model_load(next(r for r in rows if r['model']=='H26' and r['seed']==17))
    predpath=ROOT/control['actual_predictions']['17'];pin(predpath,control['frozen_input_sha256'][rel(predpath)])
    with np.load(predpath) as f:pred=TrajectoryPrediction(**{k:f[k][:128] for k in vars(validation.trajectory.truth)})
    b=subset(validation,128)
    from system_identification.evaluation.paper_baselines import endpoint_metrics
    old=endpoint_metrics(pred,b.trajectory,model='H26',seed=17)
    new,_=endpoint_adapter(pred,b,model='H26',seed=17,date='validation',allowlist=main['validation_flights'])
    np.testing.assert_array_equal(old[list(ERROR_COLUMNS)],new[list(ERROR_COLUMNS)])
    fresh=predict_history_trajectory_model(model,b,use_history=True,batch_size=128,device=DEVICE)
    for k,v in vars(fresh).items():np.testing.assert_allclose(v,getattr(pred,k),rtol=1e-6,atol=2e-5)
    const=hold_controls(small);a=predict_history_trajectory_model(model,const,use_history=True,batch_size=128,device=DEVICE)
    z=predict_history_trajectory_model(model,hold_controls(const),use_history=True,batch_size=128,device=DEVICE)
    for k,v in vars(a).items():np.testing.assert_array_equal(v,getattr(z,k))
    inventory=ROOT/'docs/analysis/results/raw_log_inventory_20260920/logs.csv'
    identity=pd.read_csv(inventory,usecols=['relative_path','sha256','bytes']).set_index('relative_path')
    parts={}
    for date,part in DATES.items():
        ids=m['split_contract']['assignments'][part]
        require(not set(ids)&(set(main['train_flights'])|set(main['validation_flights'])),'split overlap')
        paths=[]
        for flight in ids:
            path=Path(cfg['source_root'] if date=='Sep8' else '/home/zn/数据')/flight
            require(path.is_file(),'required raw file missing '+str(path))  # metadata only, no read/hash
            paths.append(dict(flight_id=flight,path=str(path),bytes=path.stat().st_size,
                source_sha256=identity.loc[flight,'sha256'] if flight in identity.index else None,session=flight.split('/')[0]))
        parts[date]=dict(role='primary independent test' if date=='Sep8' else 'supplementary held-out date',original_partition=part,flights=paths)
    result=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_paper_independent_test.py','tests/test_paper_baselines.py','tests/test_future_control_diagnostic.py'],capture_output=True,text=True)
    write(OUT/'tests.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,stage='before_test_access'))
    require(result.returncode==0,'pre-registration tests failed')
    write(OUT/'preflight.json',dict(status='passed',fifteen_checkpoints=checks,validation_adapter_bitwise_parity=True,
        validation128_prediction_parity=True,constant_control_bitwise=True,validation_only=True))
    protocol=dict(experiment='paper_independent_test_v1',frozen_at_utc=now(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),runtime=runtime(17),partitions=parts,models=rows,
        B0='original ConstantTwistPredictor constant NED velocity/body rate, quaternion exponential; deterministic',
        train_flights=main['train_flights'],validation_flights=main['validation_flights'],source_manifest=rel(manifest),source_manifest_sha256=file_hash(manifest),
        preprocessing=dict(config_path=rel(configpath),sampling='native publication timestamps; past-only asof hold; no resampling; original valid_core/segments',
            Sep8_gate='inherit September v2: firmware and all structural/output parameters exact; no changed parameters; critical topics present and strictly increasing publication timestamps; core nonempty; original Hall ratio diagnostic gate',
            Sep19_gate='inherit v3 expansion: no changed parameters; output_mapping and structural prefixes CA_SV/PWM_MAIN/FLAP_RATIO/SYS_AUTOSTART exact; critical topics; valid_duration>=30s and at least10 full-history50future windows at stride10; no firmware equality gate',
            common_origins='original v3 windows: for each valid contiguous segment start in range(25,n-50,50),51 states; all H are suffixes of these H26 windows; no extra windows',
            quality_failure='fixed gate violation excludes flight with recorded rule; unexpected parse/runtime or identity failure invalidates affected date rather than silently dropping flight',
            no_fit=True,normalization_sha256=main['normalization_sha256'],history_semantics=control['history_semantics'],phase=main['model_phase_source_detail'],frames=main['frames']),
        prior_exposure=dict(Sep8=parentmeta['split_contract'].get('test_prior_exposure'),Sep19='previous raw-log inventory includes file identity/magic metadata; no frozen15-model test evaluation located in known experiment manifests or named output directories',
            claim='first evaluation of this frozen matrix based on available provenance, not first-ever raw-log access; no exhaustive claim about unrecorded outside use'),
        conditions=control['conditions'],control_std=control['control_std'],grouping=control['grouping'],thresholds=thresholds,
        horizons=control['horizons'],primary_horizon_s=.5,metrics=list(METRICS),aggregation=control['aggregation']+'; each date separately; B0 single deterministic estimate/no seedSD; no pooled-date ALL',
        interval_formula=control['interval_formula'],gains='reference-comparison; denominator reference; MLP->H26,H1->H26,H13->H26,Hold->Actual; zero undefined',
        tolerances=control['tolerances'],plotting_time=control['plotting_time'],plots=['each-date model versus horizon','each-date history500ms','Actual/Hold evolution ALL and fixed500ms groups','paired-flight500ms'],
        qualitative_cases='not generated in this quantitative evaluation; no error-selected cases',
        failure_policy='save complete invalid predictions and failed origin IDs/context; mark whole affected model/date/condition invalid, never aggregate surviving subset; continue other models/dates, no retries/substitutions',
        claims='held-out known-input predictive accuracy and descriptive paired directions; history not physical time constant; Hold lacks real counterfactual truth; no arbitrary control causality/OOD/wind/body generalization or closed-loop/long-rollout claim',
        locked_choices='all15 checkpoints/seeds/H,allowlists/date roles,quality/window rules,normalization,metrics,thresholds,main500ms,conditions; engineering fixes append amendment and exposure record; never optimize test outcomes',
        no_training=True,no_parameter_tuning=True,history_Mixed_unchanged=True,H26_main_unchanged=True,frozen_input_sha256=pins)
    registration=('# Frozen held-out evaluation registration\n\nFrozen UTC: '+protocol['frozen_at_utc']+'\n\n'+
        'Sep8 is the primary independent test; Sep19 is the supplementary held-out date. Both must be executed and reported separately irrespective of errors. '+
        'This is not a claim of never-read raw logs: Sep8 previously received a descriptive quality audit. Prior access metadata is retained in protocol.json.\n\n'+
        'The exact15 checkpoints, seeds17/23/42, native-time quality gates, H26 shared origins, Actual/Hold conditions, original train control thresholds and statistical rules are registered in protocol.json. '+
        'H26 Actual is inferred once per seed/date and reused across comparisons. No training, fitting, seed selection, ensemble, or validation/test-dependent filtering. '+
        'Each date: within-flight vector/geodesic RMS, equal-flight mean, three-seed sample SD. B0 has no seed SD. Primary25steps, auxiliary5/10/50. '+
        'H13 remains an ablation, H26 remains main; original validation Mixed classification unchanged.\n\n'+
        'Admission follows the original data-family gates: Sep8 inherits v2 Sep7 gate, Sep19 inherits v3 Sep17 gate; common native stride50 windows require25 prior samples and50future transitions. '+
        'See exact source hashes and rule details in protocol.json. No test-sequence read occurs before this file and protocol are frozen.\n\n'+
        'Frozen train thresholds: '+json.dumps(thresholds)+'\n\n'+
        'No qualitative case is generated; all quantitative metrics/dates are shown. Finite but poor predictions are retained. Numerical/runtime failures remain explicit missing full results, not filtered samples. '+
        'Engineering fixes require append-only amendment with results exposure; no scientific threshold or tuning. These tests cannot establish arbitrary-action causality, closed-loop benefits, or long-horizon stability.\n')
    (OUT/'evaluation_registration.md').write_text(registration)
    write(OUT/'protocol.json',protocol)
    write(OUT/'registration_lock.json',dict(frozen_at_utc=protocol['frozen_at_utc'],protocol_sha256=file_hash(OUT/'protocol.json'),registration_sha256=file_hash(OUT/'evaluation_registration.md')))
    (OUT/'git_status_at_registration.txt').write_text(subprocess.check_output(['git','status','--short'],text=True))
    print('REGISTERED',protocol['frozen_at_utc'],'No test sequence read.',flush=True)


def load_registration():
    p=read(OUT/'protocol.json');lock=read(OUT/'registration_lock.json')
    require(file_hash(OUT/'protocol.json')==lock['protocol_sha256'],'registration modified')
    require(file_hash(OUT/'evaluation_registration.md')==lock['registration_sha256'],'registration modified')
    verify(p['frozen_input_sha256']);return p


def access(date,flight,path,event,**details):
    with (OUT/'test_access_log.jsonl').open('a') as f:
        f.write(json.dumps(dict(at_utc=now(),date=date,flight_id=flight,path=str(path),event=event,
            entrypoint=rel(SCRIPT),protocol_sha256=file_hash(OUT/'protocol.json'),**details),ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())


class GateFailure(ValueError):pass

def gate(ok,reason):
    if not ok:raise GateFailure(reason)


def prepare():
    p=load_registration();require(not (ART/'prepared.pt').exists(),'already prepared; no overwrite')
    cfg=yaml.safe_load((ROOT/p['preprocessing']['config_path']).read_text())
    allcoverage=[];batches={};allorigins=[];durations=[];elapsed=[];failures=[]
    for date,part in p['partitions'].items():
        frames=[];fatal=False
        for entry in part['flights']:
            flight=entry['flight_id'];path=Path(entry['path']);row=dict(date=date,session=entry['session'],flight_id=flight,status='pending',raw_samples=None,valid_samples=None,raw_possible_origins=None,final_origins=0)
            access(date,flight,path,'first_read_this_run_before_hash_and_ULog')
            try:
                sha=file_hash(path);row['sha256']=sha
                require(entry['source_sha256'] is None or sha==entry['source_sha256'],'raw identity mismatch')
                require(sha not in read(ROOT/p['source_manifest'])['source']['ulog_sha256'].values(),'raw hash overlaps train/validation')
                ul=ULog(str(path));row['raw_samples']=len(_dataset(ul,'vehicle_local_position').data['timestamp']) if _dataset(ul,'vehicle_local_position') else 0
                row['firmware']=ul.msg_info_dict.get('ver_sw');gate(not ul.changed_parameters,'parameters changed during flight')
                params=cfg['structural_parameters'] if date=='Sep8' else {k:v for k,v in cfg['structural_parameters'].items() if k.startswith(('CA_SV','PWM_MAIN','FLAP_RATIO','SYS_AUTOSTART'))}
                required={**params,**cfg['output_mapping']};mismatch={k:dict(expected=v,actual=ul.initial_parameters.get(k)) for k,v in required.items() if ul.initial_parameters.get(k)!=v}
                gate(not mismatch,'structural/output parameter mismatch: '+json.dumps(mismatch))
                if date=='Sep8':gate(row['firmware']==cfg['firmware'],'firmware mismatch')
                gate(all(_dataset(ul,n) is not None for n in CRITICAL_TOPICS),'missing critical topic')
                if date=='Sep8':
                    for name in (*CRITICAL_TOPICS,'hall_event'):
                        topic=_dataset(ul,name)
                        if topic is not None:
                            ts=np.asarray(topic.data['timestamp']);gate(len(ts)>0 and np.all(np.diff(ts)>0),'invalid publication timestamps: '+name)
                frame=extract_trajectory_samples(path,log_id=flight,split=date,transmission_ratio=cfg['transmission_ratio'],timestamp_basis='publication')
                wing=_dataset(ul,'wing_phase');hall=_dataset(ul,'hall_event')
                phase=observed_phase(frame.timestamp_us.to_numpy(),wing.data,hall.data if hall else None)
                for c in phase:frame[c]=phase[c].to_numpy()
                row.update(raw_samples=len(frame),valid_samples=int(frame.valid_core.sum()),valid_duration_s=_duration_s(frame),
                    raw_possible_origins=max(0,len(range(25,len(frame)-50,50))),
                    exclusion_counts=json.dumps(frame.loc[~frame.valid_core,'exclusion_reason'].value_counts().to_dict()))
                candidate=windows(frame,date,50);row['quality_window_origins']=len(candidate)
                if date=='Sep8':
                    gate(row['valid_samples']>0,'no valid core samples')
                    try:diag=hall_diagnostics(_dataset(ul,'encoder_count').data,hall.data if hall else None,cfg['transmission_ratio'])
                    except ValueError as e:raise GateFailure(str(e))
                    row['hall_diagnostic']=json.dumps(diag)
                else:gate(row['valid_duration_s']>=30 and len(windows(frame,date,10))>=10,'insufficient contiguous airborne data: duration<30 or stride10 windows<10')
                gate(len(candidate)>0,'no H26/50future origin')
                require(file_hash(path)==sha,'raw file changed during extraction')
                frames.append(frame);row.update(status='admitted',final_origins=len(candidate),reason='')
            except GateFailure as e:row.update(status='excluded_by_frozen_gate',reason=str(e))
            except Exception as e:
                row.update(status='engineering_failure',reason=str(e));fatal=True
                failures.append(dict(date=date,flight_id=flight,error=traceback.format_exc()))
            allcoverage.append(row);access(date,flight,path,'read_complete',status=row['status'])
            pd.DataFrame(allcoverage).to_csv(OUT/'test_flight_coverage.csv',index=False)
            print(date,flight,row['status'],row['final_origins'],flush=True)
        if fatal or not frames:
            failures.append(dict(date=date,error='date unavailable: engineering failure or no admitted flights'));continue
        samples=pd.concat(frames,ignore_index=True);w=windows(samples,date,50)
        b=assemble_history_trajectory_windows(samples,w,history_steps=26);require(b.history_mask.all(),'incomplete history')
        require(set(b.trajectory.log_ids)<=set(x['flight_id'] for x in part['flights']),'test allowlist mismatch')
        groups={(str(log),int(seg)):g.sort_values('sample_in_segment') for (log,seg),g in samples.loc[samples.valid_core].groupby(['log_id','segment_id'])}
        times=[];spans={h:[] for h in (1,5,13,26)}
        for r in w.itertuples():
            ts=groups[(r.log_id,int(r.segment_id))].timestamp_us.to_numpy();i=int(r.start_sample_in_segment);times.append(int(ts[i]))
            for h in spans:spans[h].append((ts[i]-ts[i-h+1])*1e-6)
        for h,v in spans.items():
            short=truncate_history(b,h);direct=assemble_history_trajectory_windows(samples,w,history_steps=h)
            for field in ['history_state_features','history_controls','history_mask']:np.testing.assert_array_equal(getattr(short,field),getattr(direct,field))
            require(short.trajectory is b.trajectory,'future tape changed')
            for session in ['ALL',*sorted(samples.log_id.str.split('/').str[0].unique())]:
                mask=np.ones(len(w),bool) if session=='ALL' else w.log_id.str.startswith(session+'/').to_numpy()
                durations.append(dict(date=date,session=session,history_steps=h,n_origins=int(mask.sum()),**stats_distribution(np.asarray(v)[mask])))
        w['date']=date;w['session']=w.log_id.str.split('/').str[0];w['origin_timestamp_us']=times;allorigins.append(w)
        for k in range(1,51):elapsed.append(dict(date=date,step=k,n_origins=len(w),**stats_distribution(b.trajectory.dt_s[:,:k].sum(1))))
        samples.to_parquet(ART/f'samples_{date}.parquet',index=False);w.to_csv(ART/f'origins_{date}.csv',index=False);batches[date]=b
    torch.save(batches,ART/'prepared.pt')
    pd.concat(allorigins,ignore_index=True).to_csv(OUT/'test_origins.csv',index=False) if allorigins else pd.DataFrame().to_csv(OUT/'test_origins.csv',index=False)
    pd.DataFrame(durations).to_csv(OUT/'history_duration_summary.csv',index=False);pd.DataFrame(elapsed).to_csv(OUT/'native_elapsed_time.csv',index=False)
    write(OUT/'preparation_status.json',dict(available_dates=list(batches),failures=failures,completed_at=now(),no_training=True,
        origin_identity={d:array_digest(b) for d,b in batches.items()},prepared_sha256=file_hash(ART/'prepared.pt')))
    coverage=pd.DataFrame(allcoverage)
    (OUT/'dataset_quality_report.md').write_text('# Frozen test data preparation\n\n'+'```csv\n'+coverage.to_csv(index=False)+'```'+'\n\nRules are frozen in protocol.json. Raw possible origins count ignores core gaps; quality_window_origins applies valid contiguous segments; final_origins additionally applies flight gate. Unknown counters remain blank for a flight rejected before extraction, never fabricated as zero.\n\nSep8 inherits v2 flight gates; Sep19 inherits v3 expansion gates. Native publication alignment and common stride50 H26 windows are identical to the current validation construction. Prior Sep8 descriptive quality exposure is acknowledged. Both dates have now been accessed; old historical unopened statements are not rewritten.\n')


def infer():
    p=load_registration();prep=read(OUT/'preparation_status.json');require(file_hash(ART/'prepared.pt')==prep['prepared_sha256'],'prepared changed')
    batches=torch.load(ART/'prepared.pt',map_location='cpu',weights_only=False)
    require(not (ART/'inference_status.json').exists(),'inference already attempted, inspect failures before any retry')
    write(ART/'inference_status.json',dict(status='running',started=now()))
    outcomes=[];checks=[];coverage=[];activity=[];elapsed=[]
    for date,b in batches.items():
        require(array_digest(b)==prep['origin_identity'][date],'batch changed');base_digest=array_digest(b);n=len(b.trajectory.window_ids)
        labels={}
        for tr in p['thresholds']:
            k=int(tr['step']);e=excitation(b.trajectory.controls,p['control_std'],k);labels[k]=assign_groups(e,tr['train_q25'],tr['train_q75'])
            for i in range(n):activity.append(dict(date=date,flight_id=b.trajectory.log_ids[i],window_id=b.trajectory.window_ids[i],step=k,E=e[i],group=labels[k][i]))
            for group in ['ALL',*GROUPS]:
                mask=np.ones(n,bool) if group=='ALL' else labels[k]==group
                for flight in ['ALL',*sorted(set(b.trajectory.log_ids))]:
                    selected=mask if flight=='ALL' else mask&(b.trajectory.log_ids==flight)
                    coverage.append(dict(date=date,step=k,horizon_s=k*.02,group=group,flight_id=flight,
                        n_origins=int(selected.sum()),n_flights=len(set(b.trajectory.log_ids[selected])),
                        E_median=float(np.median(e[selected])) if selected.any() else None))
        for group in ['ALL',*GROUPS]:
            mask=np.ones(n,bool) if group=='ALL' else labels[25]==group
            for k in range(1,51):elapsed.append(dict(date=date,group=group,step=k,n_origins=int(mask.sum()),
                **stats_distribution(b.trajectory.dt_s[mask,:k].sum(1))) if mask.any() else dict(date=date,group=group,step=k,n_origins=0))
        pd.DataFrame(coverage).to_csv(OUT/'control_group_coverage.csv',index=False)
        jobs=[dict(model='B0',seed=None,history_steps=0)]+p['models']
        for job in jobs:
            model_id=job['model'];seed=job['seed'];dest=ART/date/f'{model_id}_seed{seed}';dest.mkdir(parents=True,exist_ok=True)
            actual=None;model=None
            for condition in (['Actual','Hold'] if model_id=='H26' else ['Actual']):
                try:
                    batch=b if model_id=='B0' else truncate_history(b,job['history_steps'])
                    if condition=='Hold':batch=hold_controls(batch)
                    before=array_digest(batch)
                    if model_id=='B0':pred=ConstantTwistPredictor().rollout(batch.trajectory.initial_state(),batch.trajectory.controls,batch.trajectory.dt_s)
                    else:
                        if model is None:model=model_load(job)
                        state={k:v.clone() for k,v in model.state_dict().items()}
                        leak=causality_check(model,subset(batch,4),DEVICE)
                        pred=predict_history_trajectory_model(model,batch,use_history=True,batch_size=128,device=DEVICE)
                        for key,v in model.state_dict().items():require(torch.equal(v,state[key]),'model changed')
                    np.savez_compressed(dest/f'{condition}_predictions.npz',**vars(pred),window_ids=b.trajectory.window_ids.astype(str))
                    bad=np.zeros(n,bool)
                    for v in vars(pred).values():bad|=~np.isfinite(v).reshape(n,-1).all(1)
                    bad|=(abs(np.linalg.norm(pred.quaternion_nb,axis=-1)-1)>1e-5).any(1)
                    if bad.any():
                        pd.DataFrame(dict(window_id=b.trajectory.window_ids[bad],flight_id=b.trajectory.log_ids[bad])).to_csv(dest/f'{condition}_failed_origins.csv',index=False)
                        np.savez_compressed(dest/f'{condition}_failed_context.npz',controls=b.trajectory.controls[bad],dt=b.trajectory.dt_s[bad])
                        raise ValueError(f'{bad.sum()} invalid predictions; whole condition invalid, no filtered result')
                    frame,errors=endpoint_adapter(pred,batch,model=model_id,seed=seed,date=date,allowlist=[x['flight_id'] for x in p['partitions'][date]['flights']])
                    if condition=='Actual':actual=pred
                    if condition=='Hold':
                        require(actual is not None,'Actual unavailable for first-step check')
                        for key,v in vars(pred).items():np.testing.assert_allclose(v[:,:2],getattr(actual,key)[:,:2],rtol=1e-6,atol=2e-5)
                        constant=np.all(b.trajectory.controls==b.trajectory.controls[:,:1],axis=(1,2))
                        for key,v in vars(pred).items():np.testing.assert_allclose(v[constant],getattr(actual,key)[constant],rtol=1e-6,atol=2e-5)
                    frame.to_csv(dest/f'{condition}_endpoints.csv',index=False);np.save(dest/f'{condition}_squared_errors.npy',errors)
                    require(array_digest(batch)==before and array_digest(b)==base_digest,'batch changed')
                    outcomes.append(dict(date=date,model=model_id,history_steps=job['history_steps'],seed=seed,condition=condition,status='valid',n_origins=n,directory=rel(dest)))
                    checks.append(dict(date=date,model=model_id,seed=seed,condition=condition,finite=True,origin_count=n,normalization_unchanged=True,
                        max_quaternion_norm_error=float(np.max(abs(np.linalg.norm(pred.quaternion_nb,axis=-1)-1))),
                        future_label_poisoning=True if model_id!='B0' else 'not applicable',prefix=True,hold_first_step=True if condition=='Hold' else None))
                    print(date,model_id,seed,condition,'valid',n,flush=True)
                except Exception:
                    error=traceback.format_exc();write(dest/f'{condition}_failure.json',dict(error=error,at_utc=now(),all_origins_invalid_for_complete_aggregation=True))
                    outcomes.append(dict(date=date,model=model_id,history_steps=job['history_steps'],seed=seed,condition=condition,status='invalid',n_origins=n,directory=rel(dest),error=error))
                    print(date,model_id,seed,condition,'FAILED',error,flush=True)
            write(ART/'inference_status.json',dict(status='running',outcomes=outcomes))
    pd.DataFrame(activity).to_csv(ART/'origin_control_activity.csv',index=False);pd.DataFrame(elapsed).to_csv(OUT/'group_elapsed_time.csv',index=False)
    write(OUT/'sanity_checks.json',dict(status='passed' if all(r['status']=='valid' for r in outcomes) else 'has_failures',runs=checks,
        preparation=prep,short_history_suffix_exact=True,H26_Actual_once_per_seed_date=True,thresholds_reused=True,training_performed=False,
        dates_accessed=list(p['partitions']),old_sources_unchanged=True))
    write(ART/'inference_status.json',dict(status='complete',outcomes=outcomes,completed_at=now()))
    verify(p['frozen_input_sha256'])


def summarize():
    p=load_registration();batches=torch.load(ART/'prepared.pt',map_location='cpu',weights_only=False)
    outcomes=read(ART/'inference_status.json')['outcomes'];activity=pd.read_csv(ART/'origin_control_activity.csv');rows=[];curves=[]
    for run in outcomes:
        if run['status']!='valid':continue
        date=run['date'];b=batches[date];n=len(b.trajectory.window_ids);log=b.trajectory.log_ids;dates=np.full(n,date)
        errors=np.load(ROOT/run['directory']/f'{run["condition"]}_squared_errors.npy')
        labels={k:activity[(activity.date==date)&(activity.step==k)].set_index('window_id').loc[b.trajectory.window_ids,'group'].to_numpy() for k in HORIZONS.values()}
        meta={k:run[k] for k in ('model','history_steps','condition','seed')}
        for h,k in HORIZONS.items():
            for group in ['ALL',*GROUPS] if run['model']=='H26' else ['ALL']:
                mask=np.ones(n,bool) if group=='ALL' else labels[k]==group
                for kind,mse in [('endpoint',errors[:,k-1])]+([('interval',interval_mse(errors,b.trajectory.dt_s,k))] if run['model']=='H26' else []):
                    rows.extend(flight_metrics(mse,log,dates,mask,**meta,date=date,kind=kind,group=group,step=k,horizon_s=h))
        if run['model']=='H26':
            for k in range(1,51):
                for group in ['ALL',*GROUPS]:
                    mask=np.ones(n,bool) if group=='ALL' else labels[25]==group
                    curves.extend(flight_metrics(errors[:,k-1],log,dates,mask,**meta,date=date,kind='evolution',group=group,step=k,horizon_s=k*.02))
    per=pd.DataFrame(rows);curve=pd.DataFrame(curves);per.to_csv(OUT/'per_flight.csv',index=False);curve.to_csv(ART/'per_flight_evolution.csv',index=False)
    seed,multi=aggregate(per);cs,cm=aggregate(curve)
    # Empty/failed experiment cells are explicit, never represented as surviving-seed means.
    expected=[]
    for date in p['partitions']:
        for model,h in [('B0',0),('MLP',1),('H1',1),('H5',5),('H13',13),('H26',26)]:
            for condition in ['Actual','Hold'] if model=='H26' else ['Actual']:
                for horizon,k in HORIZONS.items():
                    for group in ['ALL',*GROUPS] if model=='H26' else ['ALL']:
                        for kind in ['endpoint','interval'] if model=='H26' else ['endpoint']:
                            for metric in METRICS:expected.append(dict(date=date,model=model,history_steps=h,condition=condition,kind=kind,group=group,step=k,horizon_s=horizon,metric=metric))
    multi=pd.DataFrame(expected).merge(multi,on=KEYS,how='left',validate='one_to_one');multi['status']=multi.status.fillna('empty_or_unavailable')
    seed.to_csv(OUT/'per_seed_summary.csv',index=False);multi.to_csv(OUT/'multiseed_summary.csv',index=False)
    ps,pf,g=paired(seed,per)
    ps.to_csv(OUT/'paired_seed_differences.csv',index=False);pf.to_csv(OUT/'paired_flight_differences.csv',index=False);g.to_csv(OUT/'relative_improvement.csv',index=False)
    cm=cm.merge(pd.read_csv(OUT/'group_elapsed_time.csv'),on=['date','group','step','n_origins'],how='left',validate='many_to_one')
    cm.to_csv(OUT/'error_evolution.csv',index=False);multi[multi.kind=='interval'].to_csv(OUT/'trajectory_interval_errors.csv',index=False)
    from report_paper_independent_test import generate
    generate(OUT,p)
    verify(p['frozen_input_sha256'])
    write(OUT/'completion.json',dict(status='complete' if all(r['status']=='valid' for r in outcomes) and len(batches)==2 else 'complete_with_recorded_failures',
        completed_at_utc=now(),protocol_sha256=file_hash(OUT/'protocol.json'),training_performed=False,parameter_tuning=False,main_model='Standard GRU/H26',
        test_dates_accessed=list(p['partitions']),previous_sealed_Sep8_used=True,previous_reserved_Sep19_used=True,
        outcomes=outcomes,tests=read(OUT/'tests.json'),artifact_sha256={rel(f):file_hash(f) for f in OUT.iterdir() if f.is_file() and f.name!='completion.json'},
        large_artifact_sha256={rel(f):file_hash(f) for f in ART.rglob('*') if f.is_file()},report_source_sha256=file_hash(ROOT/'scripts/report_paper_independent_test.py')))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--phase',required=True,choices=['register','prepare','infer','summarize']);args=a.parse_args();configure();globals()[args.phase]()
