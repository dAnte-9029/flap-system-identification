"""Finish the prespecified gated study after both small-cohort runs complete."""
import json
import subprocess
import sys
import time
import train_control_windows_v2 as base
from reference_capacity_v6 import OUT

def main():
    for arm in ['integrated','direct']:
        path=OUT/'tiny'/f'{arm}_17'/'status.json'
        while True:
            try:s=json.loads(path.read_text())
            except (FileNotFoundError,json.JSONDecodeError):time.sleep(1);continue
            if s['status']=='complete':break
            if s['status']=='failed':raise RuntimeError(s)
            time.sleep(5)
    sequence=['evaluate_reference_capacity_v6_tiny.py','run_reference_capacity_v6_formal.py','evaluate_reference_capacity_v6_formal.py','report_reference_capacity_v6.py']
    try:
        for script in sequence:
            base.write(OUT/'pipeline_status.json',dict(status='running',stage=script));print('Running',script,flush=True)
            subprocess.run([sys.executable,str(base.ROOT/'scripts'/script)],check=True)
        base.write(OUT/'pipeline_status.json',dict(status='complete'))
    except Exception as exc:
        base.write(OUT/'pipeline_status.json',dict(status='failed',error=repr(exc)));raise
if __name__=='__main__':main()
