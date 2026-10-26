"""Finalize provenance; do not fit, infer, or read any test artifact."""
from common import *
import platform,subprocess
import torch,scipy,pyarrow,matplotlib,pytest
def main():
    receipt=json.loads((OUT/'source_receipt.json').read_text())
    for path,expected in receipt['files'].items():assert sha(path)==expected,path
    assert sha(HERE/'config.json')==receipt['config_sha256']
    runtime=json.loads((HERE/'runtime.json').read_text())
    for name,expected in runtime['training_source_sha256'].items():assert sha(HERE/name)==expected,name
    old=json.loads((ABLATION/'results/inference_checks.json').read_text())
    for r in old['checkpoints']:assert sha(ROOT/r['path'])==r['sha256'],r['path']
    for name in ['M0','M1','M2']:
        s=json.loads((HERE/'training'/name/'summary.json').read_text())
        for key in ['best','last']:assert sha(HERE/'training'/name/(key+'.pt'))==s[key+'_sha256']
        assert s['epochs']==6 and s['updates']==636 and s['seed']==17
    for name in ['cases','pairs','cohorts','windows']:assert sha(OUT/f'{name}.csv')==sha(SWAP/'results'/f'{name}.csv')
    assert '12 passed' in (HERE/'tests.log').read_text()
    verification=json.loads((OUT/'independent_verification.json').read_text());assert verification['passed']
    base='2c99a7fcf8495cfb1d21535e928716537aa7eb73';branch='exp/structured-control-effectiveness-v1'
    assert subprocess.check_output(['git','-C',str(ROOT),'branch','--show-current'],text=True).strip()==branch
    commands=[f'{sys.executable} experiments/structured_control_effectiveness_v1/{name}.py' for name in ['preflight','train','holdout_plan','evaluate','verify','plot','describe']]
    save(HERE/'execution.json',dict(base_sha=base,branch=branch,worktree=str(ROOT),python=sys.version,interpreter=sys.executable,platform=platform.platform(),versions={k:v.__version__ for k,v in [('numpy',np),('pandas',pd),('torch',torch),('scipy',scipy),('pyarrow',pyarrow),('matplotlib',matplotlib),('pytest',pytest)]},commands=commands,tests_command=f'{sys.executable} -m pytest -q experiments/structured_control_effectiveness_v1/test_contract.py',actual_training_runs=3,seeds=[17],epochs_each=6,updates_each=636,discarded_smoke_updates=9,all_best_epochs=6,parameter_search=False,loss_changed=False,time_contract_changed=False,cache_rebuilt=False,dependency_changes=False,test_read=False,old_checkpoints_verified_unchanged=6,old_source_and_artifacts_verified_unchanged=len(receipt['files']),next_stage_training_executed=False,source_notes=['preflight index copying and descriptive historical-reference export added to reproduction after training; training/model/common source hashes unchanged','Training and evaluation passed without failed optimization runs or hyperparameter changes','Source files and all added outputs reside in this experiment; dirty root workspace untouched'],publication='Containing Git commit and verified remote ref determine publication.'))
    files=[]
    for path in sorted(HERE.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or '.pytest_cache' in path.parts or path.name=='archive_manifest.json':continue
        files.append(dict(path=str(path.relative_to(HERE)),bytes=path.stat().st_size,sha256=sha(path)))
    save(HERE/'archive_manifest.json',dict(base_sha=base,branch=branch,file_count=len(files),test_read=False,files=files))
    print('Archived',len(files),'files; verified old sources and six checkpoints unchanged; largest',max(r['bytes'] for r in files))
if __name__=='__main__':main()
