"""Check shared train cohort by keys; no validation/test data or models opened."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path('/home/zn/flap-system-identification'); HERE=Path(__file__).resolve().parents[1]
BASE='experiments/phase_anchored_coupled_latent_v1'
src=ROOT/'.worktrees/pacld-control-identifiability-v1'/BASE/'control_identifiability'
a2=ROOT/'.worktrees/pacld-control-effectiveness-v2'/BASE/'control_effectiveness_a2_v2'
crm=ROOT/'.worktrees/pacld-control-residual-moment-v1'/BASE/'control_residual_moment_v1'
c=json.loads((a2/'contract.json').read_text())
for file,key in [('features.pkl','source_feature_sha256'),('context.npy','source_context_sha256')]:
 assert hashlib.sha256((src/file).read_bytes()).hexdigest()==c[key]
f=pd.read_pickle(src/'features.pkl'); w=pd.read_csv(crm/'windows.csv'); identity=pd.read_csv(a2/'identity.csv')
assert not f.duplicated(['flight','t']).any()
assert set(zip(f.flight,f.t))==set(zip(w.flight,w.t))==set(zip(identity.flight,identity.t))
x=f.merge(w,on=['flight','t','date','mode'],validate='one_to_one',suffixes=('_source','_crm'))
np.testing.assert_allclose(np.stack(x.Y_source)[:,0],x.Y_crm,rtol=0,atol=1e-12)
np.testing.assert_allclose(np.stack(x.U)[:,5],x.U_normal,rtol=0,atol=1e-12)
np.testing.assert_allclose(.2+np.stack(x.offset)[:,0]*1e-6,x.query_s,rtol=0,atol=1e-12)
m=json.loads((HERE/'experiments/px4_identifiability_v1/train_flights.json').read_text()); assert {v['flight_id'] for v in m}==set(f.flight) and all(v['split']=='train' for v in m)
fw=json.loads((HERE/'experiments/px4_identifiability_v1/firmware_train.json').read_text());assert all(v['match'] and v['local_git_object'] for v in fw)
s=dict(windows=len(f),flights=f.flight.nunique(),dates=f.date.nunique(),logs=len(fw),firmwares=len({v['logged'] for v in fw}),keyed_identity_verified=True,feature_and_context_identity_verified=True,native_target_horizon_s=[float(x.query_s.min()),float(x.query_s.max())],split='train',other_splits_opened=False)
(HERE/'experiments/px4_identifiability_v1/cohort_check.json').write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2))
import subprocess
paths=['src/modules/fw_att_control/FixedwingAttitudeControl.cpp','src/modules/fw_rate_control/FixedwingRateControl.cpp','src/lib/rate_control/rate_control.cpp','src/modules/control_allocator/VehicleActuatorEffectiveness/ActuatorEffectivenessControlSurfaces.cpp']
rows=[]
for sha in sorted({r['logged'] for r in fw}):
 for path in paths:
  a=subprocess.run(['git','-C','/home/zn/PX4-Autopilot','rev-parse',sha+':'+path],capture_output=True,text=True)
  rows.append(dict(firmware=sha,path=path,exists=a.returncode==0,blob=a.stdout.strip() if a.returncode==0 else None))
assert all(r['exists'] for r in rows)
(HERE/'experiments/px4_identifiability_v1/source_coverage.json').write_text(json.dumps(rows,indent=2))
