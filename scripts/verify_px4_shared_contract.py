import json,subprocess
from pathlib import Path
import pandas as pd
from pyulog import ULog
root=Path('/home/zn/flap-system-identification'); ds=root/'dataset/velocity_rate_multistep_v1'
p=pd.read_pickle(root/'.worktrees/pacld-control-identifiability-v1/experiments/phase_anchored_coupled_latent_v1/control_identifiability/features.pkl')
meta=[m for m in json.loads((ds/'flight_provenance.json').read_text()) if m['split']=='train' and m['flight_id'] in set(p.flight)]
rows=[]
for path in sorted(set(m['log_path'] for m in meta)):
 m=next(m for m in meta if m['log_path']==path);u=ULog('/home/zn/QgcLogs/'+path,message_name_filter_list=[])
 sha=u.msg_info_dict.get('ver_sw');sha=sha.decode() if isinstance(sha,bytes) else sha
 ok=subprocess.run(['git','-C','/home/zn/PX4-Autopilot','cat-file','-e',str(sha)+'^{commit}'],capture_output=True).returncode==0
 rows.append(dict(log_path=path,declared=m['firmware'],logged=sha,match=sha==m['firmware'],local_git_object=ok))
out=root/'.worktrees/px4-identifiability-contract-v1/experiments/px4_identifiability_v1'
(out/'firmware_train.json').write_text(json.dumps(rows,indent=2));(out/'train_flights.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(dict(logs=len(rows),flights=len(meta),windows=len(p),dates=p.date.nunique(),firmwares=sorted(set(str(r['logged']) for r in rows)),mismatches=[r for r in rows if not r['match']],missing_git=sum(not r['local_git_object'] for r in rows)),indent=2))
