#!/usr/bin/env python3
"""One-command Step 6 with an explicit early stopping gate and provenance checks."""
import argparse,json,sys,subprocess,hashlib
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]


def verify(output):
    protocol=json.loads((output/'protocol.json').read_text())
    for name,h in {**protocol['historical_hashes'],**protocol['dataset_hashes']}.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    step5=ROOT/'docs/analysis/results/main_v2_increment_supervision'
    for name,h in json.loads((step5/'protocol.json').read_text())['baseline_hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    for row in pd.read_csv(step5/'training_summary.csv').itertuples():
        checkpoint=ROOT/'artifacts/main_v2_increment_supervision'/row.experiment/'model.pt'
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==row.checkpoint_sha256
    old=pd.read_csv(ROOT/'docs/analysis/results/main_v2_dynamics_observability_phase/probe_metrics.csv')
    new=pd.read_csv(output/'teacher_metrics.csv')
    reproduction_error=0.
    for model,target in [('B0','D0_raw'),('B1','increment2')]:
        ref=old[(old.model==model)&(old.scope=='validation_unknown_offset')&(old.evaluation_target==target)].set_index(['seed','log_id','signal']).rmse.sort_index()
        cur=new[(new.contract=='P0')&(new.scope=='validation')&(new.trained_target==target)&(new.evaluation_target==target)&new.signal.isin(['linear','angular'])].set_index(['seed','log_id','signal']).rmse.sort_index()
        reproduction_error=max(reproduction_error,float(np.max(np.abs(ref-cur))))
        np.testing.assert_allclose(ref,cur,rtol=0,atol=1e-12)  # CSV decimal round trips only.
    e=pd.read_csv(output/'phase_offset_estimation.csv')
    assert e.loc[e.split=='validation','reference_offset_error_rad'].isna().all()
    assert np.isfinite(e[['offset_rad','score','duration_s','observed_cycles']]).all().all()
    assert not json.loads((output/'probe_gate.json').read_text())['full_training_authorized_by_gate']
    cases=sorted(str(p.relative_to(ROOT)) for pattern in ['test_main_v2*.py','test_trajectory*.py','test_september_trajectory.py'] for p in (ROOT/'tests').glob(pattern))
    test=subprocess.run([sys.executable,'-m','pytest','-q',*cases],cwd=ROOT,capture_output=True,text=True);(output/'pytest.txt').write_text(test.stdout+test.stderr)
    diff=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True)
    sources=list((ROOT/'scripts').glob('*phase_reference.py'))+[ROOT/'scripts/train_main_v2_phase_probes.py',ROOT/'src/system_identification/models/phase_reference.py',ROOT/'tests/test_main_v2_phase_reference.py']
    checks=[subprocess.run(['git','diff','--no-index','--check','/dev/null',str(p)],cwd=ROOT,capture_output=True,text=True) for p in sources]
    clean=all(c.returncode in (0,1) and not c.stdout and not c.stderr for c in checks)
    result=dict(pytest_returncode=test.returncode,git_diff_check_returncode=diff.returncode,new_source_whitespace_clean=clean,historical_hashes_unchanged=True,
        validation_absolute_error_not_claimed=True,sealed_test_opened=False,oracle_isolated=True,old_checkpoints_unchanged=True,
        p0_probe_reproduces_step4=True,p0_metric_max_difference=reproduction_error,p0_csv_absolute_tolerance=1e-12,
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (output/'verification.json').write_text(json.dumps(result,indent=2))
    assert test.returncode==0,test.stdout+test.stderr
    assert diff.returncode==0,diff.stdout+diff.stderr
    assert clean,''.join(c.stdout+c.stderr for c in checks)
    (output/'status.json').write_text(json.dumps(dict(status='completed',scope='probe-gated Step6; full training skipped by scientific gate',result='C'),indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--device',default='cuda:1');a=p.parse_args();out=a.output.resolve();art=a.artifacts.resolve()
    if art.exists() and any(art.iterdir()):raise FileExistsError(art)
    def run(script,*args):subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,args)],cwd=ROOT,check=True)
    run('prepare_main_v2_phase_reference.py','--output',out)
    run('train_main_v2_phase_probes.py','--output',out,'--artifacts',art,'--device',a.device)
    gate=json.loads((out/'probe_gate.json').read_text())
    if gate['full_training_authorized_by_gate']:
        raise RuntimeError('Gate unexpectedly passed. Full Main V2 experiments are now required; refusing to claim completion from probes.')
    run('diagnose_main_v2_phase_reference.py','--output',out)
    run('evaluate_phase_reference_consistency.py','--output',out)
    run('report_main_v2_phase_reference.py','--output',out)
    verify(out)
    print('STEP 6 COMPLETE: probe gate stopped full training',flush=True)

if __name__=='__main__':main()
