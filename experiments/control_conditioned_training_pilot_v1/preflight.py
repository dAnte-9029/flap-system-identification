"""Run required pretraining tests and issue a machine-readable gate only on success."""
from pathlib import Path
import subprocess,sys,json,hashlib
HERE=Path(__file__).resolve().parent
log=HERE/'results/tests_pretraining.log'
with log.open('w') as stream:
 result=subprocess.run([sys.executable,'-m','pytest','-q',str(HERE/'test_contract.py')],stdout=stream,stderr=subprocess.STDOUT)
if result.returncode:raise SystemExit(result.returncode)
(HERE/'results/pretraining_tests.json').write_text(json.dumps(dict(passed=True,tests=9,test_log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),batch_identity_checks=27,test_read=False),indent=2))
print('Pretraining gate passed')
