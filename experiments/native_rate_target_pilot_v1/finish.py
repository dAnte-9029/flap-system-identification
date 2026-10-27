"""Run after the fixed six training runs complete. No additional training."""
from common import *
import subprocess
for script,log in [('verify_training.py','training_verification.log'),('evaluate.py','evaluation.log'),('analyze.py','analysis.log'),('describe.py','description.log'),('verify_results.py','result_verification.log'),('plot.py','plot.log')]:
 with (OUT/log).open('w') as stream:
  result=subprocess.run([sys.executable,str(HERE/script)],stdout=stream,stderr=subprocess.STDOUT)
 if result.returncode:raise RuntimeError(f'{script} failed; inspect {log}')
 print(script,'completed',flush=True)
