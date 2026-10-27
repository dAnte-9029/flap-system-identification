"""Refresh portable artifact manifest after tests/report; no data/model operations."""
from pathlib import Path
import hashlib,json,platform,sys,subprocess
from datetime import datetime,timezone
import numpy,pandas,scipy,torch,pytest
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 env=dict(python=sys.executable,version=sys.version,platform=platform.platform(),packages={k:m.__version__ for k,m in [('numpy',numpy),('pandas',pandas),('scipy',scipy),('torch',torch),('pytest',pytest)]},training_runs=0,optimizer_updates=0,gpu_used=False,dependencies_modified=False,recorded_utc=datetime.now(timezone.utc).isoformat())
 (OUT/'environment.json').write_text(json.dumps(env,indent=2))
 assignments=pandas.read_csv(ROOT/'experiments/structured_control_effectiveness_v1/results/holdout_pair_assignments.csv');assert len(assignments)==48 and assignments.pair_uid.nunique()==48 and assignments.component_key.nunique()==11
 assert assignments.groupby('fold').size().tolist()==[25,12,11]
 source=['scripts/train_velocity_rate_learning.py','scripts/train_velocity_rate_horizons.py','scripts/prepare_velocity_rate_learning.py','src/system_identification/models/velocity_rate_gru.py','experiments/control_identifiability_audit_v1/audit.py','experiments/local_control_response_scope_v1/analyze.py','experiments/local_control_response_scope_v1/config.json','experiments/structured_control_effectiveness_v1/model.py','experiments/structured_control_effectiveness_group_holdout_v1/report_zh.md','experiments/closed_loop_control_informativity_v1/report_zh.md']
 manifest=dict(base_sha='4fa56ea3893be814d226d1bfd3ad1ef966a15d67',branch='exp/control-conditioned-training-weights-v1',frozen_main_pairs=48,frozen_components=11,external_code_sha256={p:sha(ROOT/p) for p in source},files={})
 for p in sorted(HERE.rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and p.name!='artifact_manifest.json':manifest['files'][str(p.relative_to(HERE))]=dict(bytes=p.stat().st_size,sha256=sha(p))
 (OUT/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2));print('manifest files:',len(manifest['files']))
if __name__=='__main__':main()
