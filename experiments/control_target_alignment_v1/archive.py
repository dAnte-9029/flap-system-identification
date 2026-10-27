"""Seal experiment-only artifacts; verify upstream source bytes remain unchanged."""
from common import *
import platform,subprocess,importlib.metadata,time
receipt=json.loads((OUT/'source_receipt.json').read_text())
for path,digest in receipt['files'].items():assert sha(path)==digest,path
for name in ['truth_regression','model_regression','independent_fixed_verification','independent_full_verification']:
 assert json.loads((OUT/f'{name}.json').read_text())['passed']
assert '14 passed' in (OUT/'tests.log').read_text()
record=dict(base_sha='d8181c294bb4f5af628822ff21d6b6d1ed005c2b',branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),python=sys.executable,python_version=platform.python_version(),platform=platform.platform(),packages={k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','matplotlib','torch','pytest','pyarrow']},training_runs=0,gpu_inference_runs=0,checkpoint_loads=0,test_data_read=False,raw_ulog_reads=0,model_or_loss_modified=False,upstream_source_hashes_rechecked=len(receipt['files']),full_train_windows=107920,full_validation_windows=36332,full_train_flights=94,full_validation_flights=27,fixed_windows=225,fixed_pairs=113,main_train_pairs=48,dependence_components=11,tests=14,independent_full_windows=363,execution=['prepare.py','oracles.py','full_audit.py','compare_models.py','describe.py','extra.py','verify.py','pytest test_audit.py','plot.py','report.py','archive.py'],implementation_corrections=['full centered-filter support explicitly bounded to registered native support','independent endpoint check includes frozen common four-combination control support; does not expand reference masks'],stop='offline audit only; no auxiliary loss training, lambda selection, phase-shift scan or MPC')
save(OUT/'execution_record.json',record)
files=[]
for path in sorted(HERE.rglob('*')):
 if not path.is_file() or '__pycache__' in path.parts or '.pytest_cache' in path.parts or path.name=='archive_manifest.json':continue
 files.append(dict(path=str(path.relative_to(HERE)),bytes=path.stat().st_size,sha256=sha(path)))
save(HERE/'archive_manifest.json',dict(base_sha=record['base_sha'],branch=record['branch'],files=files))
for r in files:assert sha(HERE/r['path'])==r['sha256']
print('Archived',len(files),'files',sum(r['bytes'] for r in files),'bytes; source and output hashes verified')
