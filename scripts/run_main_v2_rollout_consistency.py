#!/usr/bin/env python3
"""Reproduce the approved L50 + lambda * short-prefix ablation, train/val only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys, json, argparse, subprocess, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from train_main_v2_rollout_consistency import MATRIX
from train_main_v2_increment import inputs
from run_main_v2_free_running import sha


def prepare(results):
    protocol_path=results/'protocol.json'
    if protocol_path.exists():
        existing=json.loads(protocol_path.read_text())
        if existing['objective_contract']=='legacy50_plus_short_prefix':return existing
        if existing['objective_contract']!='pending_user_clarification':raise ValueError('incompatible existing protocol')
    batch,_,_,_=inputs()
    def stats(x):return dict(min=float(x.min()),mean=float(x.mean()),max=float(x.max()),p1=float(np.quantile(x,.01)),p99=float(np.quantile(x,.99)))
    durations={str(h):stats(batch.trajectory.dt_s[:,:h].sum(1)) for h in [5,10,20,50]}
    old=ROOT/'docs/analysis/results'
    historical={}
    for directory in ['main_v2_free_running_5s','main_v2_training_objective_ablation','main_v2_dynamics_observability_phase','main_v2_increment_supervision','main_v2_phase_reference','main_v2_recurrent_representation']:
        for p in (old/directory).rglob('*'):
            if p.is_file() and not p.is_symlink():historical[str(p.relative_to(ROOT))]=sha(p)
    for p in (ROOT/'docs/audits').glob('*main_v2*.md'):historical[str(p.relative_to(ROOT))]=sha(p)
    for directory in ['trajectory_main_v2','main_v2_training_objective_ablation','main_v2_increment_supervision','main_v2_phase_reference','main_v2_recurrent_representation']:
        for p in (ROOT/'artifacts'/directory).rglob('*.pt'):historical[str(p.relative_to(ROOT))]=sha(p)
    frozen=json.loads((old/'main_v2_free_running_5s/summary.json').read_text())['source_hashes']
    for p,h in frozen.items():
        if sha(ROOT/p)!=h:raise ValueError('frozen source changed: '+p)
    data=ROOT/'dataset/trajectory_v1_august_f5_c4'
    data_hashes={str((data/p).relative_to(ROOT)):sha(data/p) for p in ['manifest.json','samples_train.parquet','samples_validation.parquet','windows_train.parquet']}
    experiments=[dict(name=n,added_horizon=h,lambda_roll=w,short_group='velocity+attitude+body_rate',priority_prefix_step_weight=.02+(w/h if h else 0.),priority_late_step_weight=.02,priority_total_weight=1+w,**{f'duration_{k}_s':v for k,v in durations[str(h or 50)].items()}) for n,(h,w) in MATRIX.items()]
    protocol=dict(objective_contract='legacy50_plus_short_prefix',approved_by_user=True,
        scientific_question='short-prefix reweighting of existing 50-step free-running objective; not first introduction of closed-loop training',
        experiments=experiments,short_weights='uniform 1/H; original channel normalization retained',
        state_groups='A translation/B rotation/C all-state are component analysis only, not additional training',
        training_steps=50,training_windows=len(batch.trajectory.window_ids),history=26,
        seeds=dict(base=17,actuator=29),epochs=dict(base=40,actuator=25),optimizer='AdamW',batch_size=256,
        learning_rates=dict(base=.0003,actuator=.0005),weight_decay=1e-5,gradient_clip=5.,checkpoint_selection='fixed final epoch',
        duration_s=durations,dt_s=stats(batch.trajectory.dt_s),dataset='explicit historical trajectory_v1_august_f5_c4 reproduction',
        partitions=['train','validation'],data_hashes=data_hashes,frozen_source_hashes=frozen,historical_hashes=historical,
        frozen_validation_windows_sha256=sha(old/'main_v2_free_running_5s/windows.csv'),
        gates=dict(teacher_max_degradation_pct=10,short_max_degradation_pct=10,frequency_max_degradation_pct=10,
            mid='0.5/1/2s velocity and attitude all improve, at least one paired-flight CI below zero',
            long='one of velocity/attitude/rate improves at both 3/5s with at least one CI below zero, no other degradation over 10%',
            variation='5s prefix and last1s median ratio at least 95% of S0',support='no increase in support/numerical/clipping failures'),
        ranking='rank gate-eligible candidates by geometric mean of 3/5s velocity/attitude RMSE ratio; if none eligible, descriptive best only',
        uncertainty='exact five-flight cluster bootstrap, one paired seed policy, no across-seed inference',
        perturbation='fixed train 10 origins per log and validation 10 per log, linspace indices, all 16 signed directions; fixed commands',
        jacobian='14D scaled physical tangent chart, hidden/proxies fixed; epsilon .001 and .002; partial not full recurrent Jacobian',
        architecture_changed=False,phase_changed=False,simulator_changed=False,actuator_changed=False,sealed_test_opened=False)
    results.mkdir(parents=True,exist_ok=True)
    protocol_path.write_text(json.dumps(protocol,indent=2))
    pd.DataFrame(experiments).to_csv(results/'experiment_manifest.csv',index=False)
    return protocol


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--artifacts',type=Path,required=True)
    ap.add_argument('--device',default='cuda:1');ap.add_argument('--contract',choices=['legacy50_plus_short_prefix'],required=True)
    ap.add_argument('--prepare-only',action='store_true');a=ap.parse_args()
    a.results=a.results.resolve();a.artifacts=a.artifacts.resolve();p=prepare(a.results)
    if a.prepare_only:return
    started=time.time();status=a.results/'run_status.json'
    def run(script,extra,log):
        command=[sys.executable,str(ROOT/'scripts'/script),'--results',str(a.results),'--artifacts',str(a.artifacts),*extra]
        with log.open('w') as f:subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    try:
        for name in MATRIX:
            status.write_text(json.dumps(dict(status='running',experiment=name,started_unix=started),indent=2))
            checkpoint=a.artifacts/name/'training_summary.json'
            if not checkpoint.exists():run('train_main_v2_rollout_consistency.py',['--experiment',name,'--device',a.device],a.results/(name+'_training.log'))
            else:
                summary=json.loads(checkpoint.read_text())
                if summary['smoke'] or summary['checkpoint_sha256']!=sha(a.artifacts/name/'model.pt'):raise ValueError('invalid resume checkpoint')
            run('evaluate_main_v2_rollout_consistency.py',['--experiment',name,'--device',a.device],a.results/(name+'_evaluation.log'))
            if name=='S0':
                reproduction=json.loads((a.results/'baseline_reproduction.json').read_text())
                if not reproduction['all_state_tensors_equal'] or reproduction['validation_benchmark']!='722 common origins passed':raise ValueError('baseline gate failed')
        run('evaluate_main_v2_rollout_consistency.py',['--experiment','Step5_S1','--device',a.device],a.results/'Step5_S1_evaluation.log')
        pd.DataFrame([json.loads((a.artifacts/n/'training_summary.json').read_text()) for n in MATRIX]).to_csv(a.results/'training_summary.csv',index=False)
        run('analyze_main_v2_prefix_components.py',['--device',a.device],a.results/'loss_component_analysis.log')
        run('report_main_v2_rollout_consistency.py',[],a.results/'report_generation.log')
        changed=[path for path,h in {**p['historical_hashes'],**p['frozen_source_hashes'],**p['data_hashes']}.items() if sha(ROOT/path)!=h]
        if changed:raise ValueError('protected inputs changed: '+str(changed))
        testfiles=['test_trajectory_main_v2.py','test_rollout_consistency.py','test_main_v2_increment.py','test_recurrent_representation.py','test_main_v2_step4.py','test_main_v2_simulator.py','test_main_v2_free_running.py','test_main_v2_objectives.py','test_main_v2_phase_reference.py']
        with (a.results/'pytest.log').open('w') as f:subprocess.run([sys.executable,'-m','pytest','-q',*['tests/'+s for s in testfiles]],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
        verification=dict(protected_files_checked=len({**p['historical_hashes'],**p['frozen_source_hashes'],**p['data_hashes']}),protected_files_changed=changed,pytest_passed=True,git_diff_check_passed=True,sealed_test_opened=False)
        verification['pytest_scope']='nine affected Main V2 modules'
        if (a.results/'full_suite_result.json').exists():verification['full_suite']=json.loads((a.results/'full_suite_result.json').read_text())
        source_paths=['scripts/run_main_v2_rollout_consistency.py','scripts/train_main_v2_rollout_consistency.py','scripts/evaluate_main_v2_rollout_consistency.py','scripts/report_main_v2_rollout_consistency.py','scripts/analyze_main_v2_prefix_components.py','src/system_identification/training/rollout_consistency.py','src/system_identification/evaluation/transition_stability.py','tests/test_rollout_consistency.py']
        verification['step8_source_hashes']={path:sha(ROOT/path) for path in source_paths}
        import torch
        verification['environment']=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(a.device))
        verification['git_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        (a.results/'verification.json').write_text(json.dumps(verification,indent=2))
        status.write_text(json.dumps(dict(status='completed',wall_time_s=time.time()-started),indent=2))
    except Exception as e:
        status.write_text(json.dumps(dict(status='failed',error=str(e),wall_time_s=time.time()-started),indent=2));raise

if __name__=='__main__':main()
