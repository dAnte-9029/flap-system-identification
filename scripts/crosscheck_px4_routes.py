"""Independent aggregate and exact-source checks; opens no raw flight data."""
import json, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
R=Path('/home/zn/flap-system-identification/.worktrees');out=Path(__file__).resolve().parents[1]/'experiments/px4_identifiability_v1'
A=R/'px4-identifiability-a-v1/experiments/px4_identifiability_a_v1';B=R/'px4-identifiability-b-v1/experiments/px4_b_v1';C=R/'px4-identifiability-c-v1/experiments/px4_partial_identifiability_c_v1'
a=json.loads((A/'summary.json').read_text());b=json.loads((B/'summary.json').read_text());c=json.loads((C/'summary.json').read_text());info=json.loads((B/'information.json').read_text())
assert a['windows']==b['cohort_windows']==info['n']==11221
assert not a['response_estimation_performed']
f=pd.read_csv(B/'flight_metrics.csv')
for m in b['metrics']:
 g=f[f.metric==m['metric']];n=g.n.sum();assert n==m['n'];np.testing.assert_allclose(np.sqrt(g.sse.sum()/n),m['rmse'],atol=1e-12)
 if m['r2'] is not None:np.testing.assert_allclose(1-g.sse.sum()/(g.sy2.sum()-g.sy.sum()**2/n),m['r2'],atol=1e-10)
assert max(info['compensation_max_errors'])<2e-14
assert c['profile']['rank']==4 and c['profile']['data_only_kernel_set']=='R^4'
assert c['raw_control_parity_max_error']<1e-12
count=0
for r in json.loads((A/'source_evidence.json').read_text()):
 assert r['available'];lines=subprocess.check_output(['git','-C','/home/zn/PX4-Autopilot','show',r['firmware']+':'+r['path']],text=True).splitlines()
 for e in r['evidence']:
  assert lines[e['line']-1].strip()==e['text'].strip();count+=1
result=dict(cohort_consistent=True,b_metrics_recomputed_from_sums=True,a_source_excerpts_verified=count,c_native_control_parity_verified=True,unrestricted_innovation_is_assumption_class_not_universal_impossibility=True)
(out/'crosscheck.json').write_text(json.dumps(result,indent=2));print(result)
