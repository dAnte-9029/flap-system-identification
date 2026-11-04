"""Pack the original GRU contract on the Hall fixed-cohort intersection, validation only."""
import ast,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
out=Path(__file__).parent;parent=out.parent
old=Path('/home/zn/flap-system-identification/.worktrees/velocity-rate-dataset-v1');data=Path('/home/zn/flap-system-identification/dataset/velocity_rate_multistep_v1')
tree=ast.parse((old/'src/system_identification/data/trajectory_dataset.py').read_text());names={}
for node in tree.body:
 if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ['STATE_COLUMNS','CONTROL_COLUMNS']:names[node.targets[0].id]=list(ast.literal_eval(node.value))
STATE=names['STATE_COLUMNS'];CONTROL=names['CONTROL_COLUMNS'];PHASE=['relative_flap_phase_sin','relative_flap_phase_cos','flap_frequency_hz'];TARGET=['target_'+c for c in STATE]
tree=ast.parse((old/'scripts/prepare_velocity_rate_learning.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='interval_events');exec(compile(ast.Module(body=[node],type_ignores=[]),'frozen_interval_events','exec'))
idx=json.loads((parent/'direction-A_seed17/index.json').read_text());selected=[r for r in idx if r['sampling_group']=='main_50hz'];assert len(selected)==108
w=pd.read_parquet(data/'windows_validation.parquet').set_index('window_id');rows=w.loc[[r['window_id'] for r in selected]].reset_index()
ss=pd.read_parquet(data/'samples_validation.parquet');sg=dict(tuple(ss.groupby('flight_id',sort=False)));native={n:dict(tuple(pd.read_parquet(data/f'native_actuator_{n}_validation.parquet').groupby('flight_id',sort=False))) for n in ['motors','servos']}
cache=np.load(old/'artifacts/velocity_rate_gru64_round1/validation.npz');cm=pd.read_parquet(old/'experiments/velocity_rate_gru64_round1/validation_windows.parquet');ci={v:i for i,v in enumerate(cm.window_id)};checks=[];metadata=[]
# Check feature/control packing against the archived original packer outputs, no model retraining.
orig_cache={k:cache[k] for k in ['history','history_mask','history_dt','history_length','initial','dt','events','event_length']}
for i,(r,item) in enumerate(zip(rows.itertuples(),selected)):
 s=sg[r.flight_id].reset_index(drop=True);a,o,e=int(r.start_sample),int(r.origin_sample),int(r.end_sample);H=o-a+1;K=e-o;hist=s.iloc[a:o+1];future=s.iloc[o+1:e+1]
 values=hist[STATE+CONTROL+PHASE].to_numpy(float);mask=np.isfinite(values);mask[:,13:17]&=hist.valid_control.to_numpy()[:,None];mask[:,17:19]&=hist.valid_encoder.to_numpy()[:,None];mask[:,19]&=hist.valid_frequency.to_numpy()
 initial=s.iloc[o][STATE].to_numpy(float);position_origin=initial[:3].copy();history=np.where(mask,values,0).astype(np.float32);history[:,:3]-=position_origin;ini=initial.astype(np.float32);ini[:3]=0
 # Match archived packer's float32 subtraction order, including position rounding.
 history[:,:3]=np.where(mask,values,0).astype(np.float32)[:,:3]-position_origin
 target=future[TARGET].to_numpy(float);target[:,:3]-=position_origin
 grid=s.state_measurement_us.to_numpy(np.int64);starts=grid[o:e];ends=grid[o+1:e+1];events,length=interval_events(native['motors'][r.flight_id],native['servos'][r.flight_id],starts,ends)
 packed=dict(history=history[None],history_mask=mask[None],history_dt=np.r_[0,np.diff(grid[a:o+1])*1e-6].astype('float32')[None],history_length=np.array([H],np.int64),initial=ini[None],dt=((ends-starts)*1e-6).astype('float32')[None],events=events[None],event_length=length[None])
 if r.horizon_us==200000:
  j=ci[r.window_id]
  for k,v in packed.items():
   ref=orig_cache[k][j:j+1]
   if k in ['history','history_mask','history_dt']:ref=ref[:,:H]
   if k in ['dt','events','event_length']:ref=ref[:,:K]
   if k=='events':ref=ref[:,:,:events.shape[1]]
   np.testing.assert_array_equal(v,ref,err_msg=f'{k} {r.window_id}')
  checks.append(r.window_id)
 np.savez_compressed(out/f'inputs_{i:03d}.npz',**packed,target=target,origin_rate=initial[10:13],position_origin=position_origin,times=(ends-int(r.decision_us))*1e-6,rate_times=(future.target_rate_measurement_us.to_numpy(np.int64)-int(r.decision_us))*1e-6,attitude_times=(future.target_attitude_measurement_us.to_numpy(np.int64)-int(r.decision_us))*1e-6)
 metadata.append(dict(i=i,original_i=item['i'],window_id=r.window_id,flight_id=r.flight_id,horizon_us=int(r.horizon_us),decision_us=int(r.decision_us),history_count=H,future_count=K,origin_state_time_us=int(grid[o]),origin_rate_time_us=int(s.iloc[o].rate_measurement_us)))
assert len(checks)==27 and len({r['flight_id'] for r in metadata})==27
(out/'index.json').write_text(json.dumps(metadata,indent=2))
records=[]
for seed in [17,29]:
 p=old/f'experiments/velocity_rate_gru64_horizon_training_v1/train_1000ms/GRU-Control/seed_{seed}/best.pt';summary=json.loads((p.parent/'summary.json').read_text());digest=hashlib.sha256(p.read_bytes()).hexdigest();assert digest==summary['checkpoint_sha256'];records.append(dict(seed=seed,checkpoint=str(p),sha256=digest,summary=summary))
(out/'preparation.json').write_text(json.dumps(dict(windows=108,flights=27,packing_exact_checks=len(checks),baseline='frozen H1000-GRU-Control',checkpoints=records,source_model=str(old/'src/system_identification/models/velocity_rate_gru.py'),source_model_sha256=hashlib.sha256((old/'src/system_identification/models/velocity_rate_gru.py').read_bytes()).hexdigest(),test_access=False),indent=2))
print('Prepared 108 windows / 27 flights; 27 archived packing checks exactly equal; 2 frozen checkpoint identities verified.')
