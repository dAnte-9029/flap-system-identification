"""Seal this experiment only; exclude the self-referential manifest and caches."""
from common import *
import sys,platform,subprocess
import torch,scipy,pyarrow,matplotlib,pytest

def main():
 source=['src/system_identification/models/velocity_rate_gru.py','src/system_identification/models/control_events.py','src/system_identification/data/velocity_rate_dataset.py','scripts/prepare_velocity_rate_learning.py','experiments/control_response_eval_v1/evaluate.py','experiments/control_identifiability_audit_v1/audit.py','experiments/control_event_canonicalization_v1/run.py','experiments/local_control_response_rollout_v1/run.py','experiments/local_control_response_rollout_v1/analyze.py']
 execution=dict(base_sha='9c9cb02098a29186922d9dcb061d3776f690605e',branch='exp/local-control-response-swap-sensitivity-v1',worktree=str(ROOT),environment_interpreter=sys.executable,python=sys.version,versions={k:v.__version__ for k,v in [('numpy',np),('pandas',pd),('torch',torch),('scipy',scipy),('pyarrow',pyarrow),('matplotlib',matplotlib),('pytest',pytest)]},gpu='NVIDIA GeForce RTX 4090',platform=platform.platform(),source_code_sha256={s:sha(ROOT/s) for s in source},commands=[f'{sys.executable} experiments/local_control_response_swap_sensitivity_v1/{s}.py' for s in ['run','support','analyze','verify','plot','describe']]+[f'{sys.executable} -m pytest -q experiments/local_control_response_swap_sensitivity_v1/test_contract.py'],upstream_unchanged=True,test_read=False,weights_updated=False,dependency_changes=False,training=False,development_corrections=['Scope manifest lists five unavailable auxiliary files; all used scope assets were present and hash-verified.','Clock extraction retained the sample_in_flight index column.','Zero-duration padding events excluded from command range checks; independent regression test added.'],result_contract='All prediction and analysis files belong only to this new experiment. Reports are human-authored from reproducible report_facts and summary files.')
 verification=json.loads((OUT/'independent_verification.json').read_text());assert verification['passed'] and verification['max_differential_mean_error']<1e-12;assert '9 passed' in (HERE/'tests.log').read_text()
 save(HERE/'execution.json',execution)
 files=[]
 for p in sorted(HERE.rglob('*')):
  if not p.is_file() or '__pycache__' in p.parts or '.pytest_cache' in p.parts or p.name=='archive_manifest.json':continue
  files.append(dict(path=str(p.relative_to(HERE)),bytes=p.stat().st_size,sha256=sha(p)))
 save(HERE/'archive_manifest.json',dict(base_sha=execution['base_sha'],branch=execution['branch'],file_count=len(files),test_read=False,publication='Containing Git commit and remote SHA determine publication; manifest excludes itself.',files=files))
 print('Sealed',len(files),'files')
if __name__=='__main__':main()
