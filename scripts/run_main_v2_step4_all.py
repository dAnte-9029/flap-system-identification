#!/usr/bin/env python3
"""One-command Step 4, with strict analysis-before-probes sequencing."""
import argparse,subprocess,sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def verify(out):
    manifest=json.loads((out/'manifest.json').read_text())
    for p,h in {**manifest['baseline_hashes'],**manifest['step3_hashes']}.items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    tests=['tests/test_main_v2_step4.py','tests/test_main_v2_objectives.py','tests/test_main_v2_simulator.py','tests/test_main_v2_free_running.py']+[str(p.relative_to(ROOT)) for p in sorted((ROOT/'tests').glob('test_trajectory*.py'))]+['tests/test_september_trajectory.py']
    result=subprocess.run([sys.executable,'-m','pytest','-q',*tests],cwd=ROOT,capture_output=True,text=True)
    (out/'pytest.txt').write_text(result.stdout+result.stderr)
    diff=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True)
    source=list((ROOT/'scripts').glob('*main_v2_step4*.py'))+[ROOT/'tests/test_main_v2_step4.py',ROOT/'docs/audits/2026-09-15_main_v2_phase_observability_audit.md']
    (out/'verification.json').write_text(json.dumps(dict(pytest_returncode=result.returncode,pytest_output=result.stdout,git_diff_check_returncode=diff.returncode,baseline_and_step3_hashes_unchanged=True,sealed_test_opened=False,sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source}),indent=2))
    assert result.returncode==0,result.stdout+result.stderr
    assert diff.returncode==0,diff.stdout+diff.stderr


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);pa.add_argument('--artifacts',type=Path,required=True);pa.add_argument('--device',default='cuda:1');a=pa.parse_args();out=a.output.resolve()
    def run(name,*args):subprocess.run([sys.executable,str(ROOT/'scripts'/name),*map(str,args)],cwd=ROOT,check=True)
    run('run_main_v2_step4.py','--output',out,'--device',a.device)
    run('extend_main_v2_step4.py','--output',out,'--device',a.device)
    run('train_main_v2_step4_probes.py','--results',out,'--artifacts',a.artifacts.resolve(),'--device',a.device)
    run('report_main_v2_step4.py','--output',out)
    verify(out)

if __name__=='__main__':main()
