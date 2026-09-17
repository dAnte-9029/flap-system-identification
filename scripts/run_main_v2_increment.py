#!/usr/bin/env python3
"""One-command fixed-architecture increment experiment and full validation."""
import argparse,json,sys,time,subprocess,hashlib
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
PROMOTION_POLICY = {
    'declared_before_S1_validation': True,
    'gate1': 'both teacher increment paired-flight CI upper below zero',
    'gate2': 'velocity or attitude improved at 2/3/5s; at least two CIs below zero',
    'gate3': 'prefix5 and last1s median variation closer to1',
    'gate4': '0.2/0.5s velocity attitude body-rate degradation under10percent',
    'gate5': 'frequency degradation under10percent',
    'gate6': 'numeric clipping support failure counts do not increase',
    'noise': 'omega spectral L1, free incremental omega RMSE, angular acceleration RMSE do not worsen',
    'uncertainty': 'five-flight exact empirical cluster bootstrap; conditional single-seed evidence',
}


def decide_s2(results,artifacts):
    protocol=json.loads((results/'protocol.json').read_text())
    g0=pd.read_csv(artifacts/'S0/final_gradient_probes.csv');g1=pd.read_csv(artifacts/'S1/final_gradient_probes.csv')
    ratios=(g1[['increment_v','increment_omega']].mean()/g0[['increment_v','increment_omega']].mean()).to_numpy()
    grad=float(g1.gradient_ratio.median());improvement=float(1-ratios.mean())
    trigger=bool(improvement<.05 and grad<.1)
    decision=dict(train_only=True,train_increment_mean_relative_improvement=improvement,final_median_gradient_ratio=grad,
        s2_triggered=trigger,rule=protocol['s2_rule'],s3_trained=False,s3_reason='optional follow-up omitted from bounded primary experiment')
    if trigger and not any(c['name']=='S2' for c in protocol['experiments']):
        protocol['experiments'].append(dict(name='S2',weights=[4*w for w in protocol['experiments'][1]['weights']]))
        (results/'protocol.json').write_text(json.dumps(protocol,indent=2));pd.DataFrame(protocol['experiments']).to_csv(results/'experiment_manifest.csv',index=False)
    (results/'conditional_experiments.json').write_text(json.dumps(decision,indent=2))
    return trigger


def verify(results,artifacts=None):
    protocol=json.loads((results/'protocol.json').read_text())
    for name,h in {**protocol['historical_hashes'],**protocol['baseline_hashes']}.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    training=pd.read_csv(results/'training_summary.csv')
    assert training.loc[training.experiment=='S0','old_checkpoint_max_parameter_difference'].iloc[0]==0
    reference=pd.read_csv(ROOT/'docs/analysis/results/main_v2_training_objective_ablation/A0_baseline_retrain/per_horizon.csv').set_index('horizon_s')
    baseline=pd.read_csv(results/'S0/per_horizon.csv').set_index('horizon_s')
    metrics=[c for c in reference if c.endswith('_equal_log_rmse')]
    np.testing.assert_allclose(baseline[metrics],reference[metrics],rtol=1e-7,atol=1e-8)
    if artifacts is not None:
        expected=pd.read_csv(ROOT/'docs/analysis/results/main_v2_free_running_5s/windows.csv').window_id.to_numpy(str)
        for experiment in protocol['experiments']:
            with np.load(artifacts/experiment['name']/'validation_rollout.npz') as trace:
                np.testing.assert_array_equal(trace['window_ids'],expected)
                for key in trace:
                    array=trace[key]
                    if array.dtype.kind in 'fiu':assert np.isfinite(array).all(),(experiment['name'],key)
    tests=sorted(str(p.relative_to(ROOT)) for pattern in ['test_main_v2*.py','test_trajectory*.py','test_september_trajectory.py'] for p in (ROOT/'tests').glob(pattern))
    test=subprocess.run([sys.executable,'-m','pytest','-q',*tests],cwd=ROOT,capture_output=True,text=True)
    (results/'pytest.txt').write_text(test.stdout+test.stderr)
    diff=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True)
    sources=list((ROOT/'scripts').glob('*main_v2_increment.py'))+[ROOT/'src/system_identification/training/main_v2_increment.py',ROOT/'tests/test_main_v2_increment.py']
    whitespace=[subprocess.run(['git','diff','--no-index','--check','/dev/null',str(p)],cwd=ROOT,capture_output=True,text=True) for p in sources]
    # --no-index may return 1 for a differing file even with no --check errors.
    whitespace_clean=all(x.returncode in (0,1) and not x.stdout and not x.stderr for x in whitespace)
    (results/'verification.json').write_text(json.dumps(dict(pytest_returncode=test.returncode,git_diff_check_returncode=diff.returncode,
        new_source_whitespace_clean=whitespace_clean,baseline_parameters_exact=True,baseline_benchmark_reproduced=True,
        common_origins_and_finite_traces_checked=artifacts is not None,historical_hashes_unchanged=True,sealed_test_opened=False,sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),indent=2))
    assert test.returncode==0,test.stdout+test.stderr
    assert diff.returncode==0,diff.stdout+diff.stderr
    assert whitespace_clean,''.join(x.stdout+x.stderr for x in whitespace)


def finish(r,a,device):
    """Finish the evaluation of completed training without restarting workers."""
    def run(script,*arguments):subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,arguments)],cwd=ROOT,check=True)
    if decide_s2(r,a):run('train_main_v2_increment.py','--results',r,'--artifacts',a,'--device',device,'--experiment','S2')
    names=[c['name'] for c in json.loads((r/'protocol.json').read_text())['experiments']]
    pd.DataFrame([json.loads((a/n/'training_summary.json').read_text()) for n in names]).to_csv(r/'training_summary.csv',index=False)
    run('evaluate_main_v2_objectives.py','--results',r,'--artifacts',a,'--device',device)
    run('evaluate_main_v2_increment.py','--results',r,'--artifacts',a,'--device',device)
    run('report_main_v2_increment.py','--results',r,'--artifacts',a)
    verify(r,a)
    (r/'status.json').write_text(json.dumps(dict(status='completed',experiments=names),indent=2))


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--results',type=Path,required=True);pa.add_argument('--artifacts',type=Path,required=True);pa.add_argument('--device',default='cuda:1');args=pa.parse_args();r=args.results.resolve();a=args.artifacts.resolve()
    def run(script,*arguments):subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,arguments)],cwd=ROOT,check=True)
    if a.exists() and any(a.iterdir()):raise FileExistsError(a)
    run('train_main_v2_increment.py','--prepare','--results',r,'--artifacts',a,'--device',args.device)
    (r/'promotion_policy.json').write_text(json.dumps(PROMOTION_POLICY,indent=2))
    a.mkdir(parents=True,exist_ok=True);jobs=[]
    for name in ['S0','S1']:
        log=(a/(name+'.log')).open('w')
        proc=subprocess.Popen([sys.executable,str(ROOT/'scripts/train_main_v2_increment.py'),'--results',str(r),'--artifacts',str(a),'--device',args.device,'--experiment',name],cwd=ROOT,stdout=log,stderr=log);jobs.append((name,proc,log))
    for name,proc,log in jobs:
        code=proc.wait();log.close()
        if code:raise RuntimeError(f'{name} failed; inspect {a/(name+".log")}')
    finish(r,a,args.device)

if __name__=='__main__':main()
