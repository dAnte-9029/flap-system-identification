"""Supplement stage-one audit with development-only signal and model identity checks."""
import csv
import json
from pathlib import Path
import numpy as np
from pyulog import ULog
from audit_control_readiness_stage1 import ROOT, OUT, LOGS, digest, save, table

models=[]
for name in ['S0','S1']:
    p=ROOT/f'artifacts/main_v2_increment_supervision/{name}/model.pt'
    expected=json.loads((ROOT/f'docs/analysis/results/main_v2_increment_supervision/{name}/summary.json').read_text())['checkpoint_sha256']
    actual=digest(p); assert actual==expected
    models.append(dict(name=name,path=str(p),sha256=actual))
save('model_references.json',dict(models=models,historical_dataset='dataset/trajectory_v1_august_f5_c4',scope='identity_only_no_inference',historical_protocol_sha256=digest(ROOT/'docs/analysis/results/main_v2_increment_supervision/protocol.json')))
rows=[]
for p in sorted((LOGS/'9.18数据').glob('*.ulg')):
    if p.name.startswith('._'): continue
    u=ULog(str(p),message_name_filter_list=['flap_b2b_adaptive','wing_phase','vehicle_attitude_setpoint','vehicle_status'])
    for d in u.data_list:
        if d.name in ['flap_b2b_adaptive','wing_phase']:
            for field in ['enabled','gate','transferred_i_raw','phase_valid']:
                if field not in d.data: continue
                v=d.data[field]; valid=np.isfinite(v); x=v[valid].astype(float)
                rows.append(dict(log=p.name,topic=d.name,field=field,n=len(v),finite_fraction=float(valid.mean()),min=float(x.min()) if len(x) else '',max=float(x.max()) if len(x) else '',nonzero_sample_fraction=float((abs(x)>1e-8).mean()) if len(x) else ''))
        if d.name=='vehicle_attitude_setpoint':
            w,x,y,z=[d.data[f'q_d[{i}]'] for i in range(4)]
            pitch=np.rad2deg(np.arcsin(np.clip(2*(w*y-z*x),-1,1)))
            rows.append(dict(log=p.name,topic=d.name,field='pitch_degrees_all_setpoint_samples',n=len(pitch),min=float(np.nanmin(pitch)),max=float(np.nanmax(pitch))))
table('development_signal_summary.csv',rows)
print(json.dumps(models,indent=2)); print(rows)
