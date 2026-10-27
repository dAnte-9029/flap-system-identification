from common import *
import platform,subprocess,time,importlib.metadata
receipt=json.loads((OUT/'receipt.json').read_text())
assert '9 passed' in (OUT/'tests.log').read_text()
for p,h in receipt['files'].items():assert sha(p)==h
save(OUT/'execution_record.json',dict(base_sha='3ee850af770631f6be3d5a9326987aa443307394',branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),python=sys.executable,python_version=platform.python_version(),packages={k:importlib.metadata.version(k) for k in ['numpy','pandas','torch','matplotlib','pytest','pyarrow']},completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),training_runs=0,GRU_instances=0,checkpoint_loads=0,GPU_inference=0,test_read=False,raw_ULog_reads=0,windows=144252,flights=121,history_rows=863457,target_contracts=['T0','T1'],tests=9,source_receipt_reused=str(PREV/'results/source_receipt.json'),decision='C; do not start 6epoch training with this C1',implementation_note='Unsupported kinematic outputs are NaN; both contracts stop on identical first missing B target. No missing target imputation.'))
files=[dict(path=str(p.relative_to(HERE)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and p.name!='archive_manifest.json']
save(HERE/'archive_manifest.json',dict(base_sha='3ee850af770631f6be3d5a9326987aa443307394',files=files))
print('Archived',len(files),'files',sum(r['bytes'] for r in files),'bytes')
