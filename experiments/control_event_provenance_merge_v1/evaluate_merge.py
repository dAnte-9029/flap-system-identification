"""Full frozen validation: exact redundant control merge, unchanged model and native dt."""
import time,json
import numpy as np
import pandas as pd
import torch
from common import *
from system_identification.models.velocity_rate_gru import VelocityRateGRU,errors
from train_velocity_rate_horizons import crop_horizon
KEYS=['history','history_mask','history_dt','history_length','initial','dt','events','event_length']
METRICS=['position_m','velocity_m_s','attitude_deg','rate_rad_s']

def redundant(events,length,mask):
 return (np.all(events[:,:,1:,:4]==events[:,:,:-1,:4],axis=-1)
         &(np.arange(1,events.shape[2])[None,None,:]<length[:,:,None])&mask[:,:,None])

def merged_events(data):
 ev=data['events'];length=data['event_length'];same=redundant(ev,length,data['future_mask']);changed=same.any(2)
 out=ev.copy();le=length.copy()
 for i,k in zip(*np.where(changed)):
  seq=[]
  for row in ev[i,k,:length[i,k]]:
   duration=int(round(float(row[4])*1e6))
   if seq and np.array_equal(seq[-1][0],row[:4]):seq[-1]=(seq[-1][0],seq[-1][1]+duration)
   else:seq.append((row[:4].copy(),duration))
  out[i,k]=0;le[i,k]=len(seq)
  for j,(u,d) in enumerate(seq):out[i,k,j,:4]=u;out[i,k,j,4]=d*1e-6
  assert sum(int(round(float(r[4])*1e6)) for r in ev[i,k,:length[i,k]])==sum(d for u,d in seq)
 return out,le,changed

def main():
 tick=time.time();OUT.mkdir(exist_ok=True);digest=cache_hash('validation')
 z=np.load(CACHE/'validation.npz');meta=pd.read_parquet(TRAIN/'validation_windows.parquet');assert len(meta)==36332 and meta.flight_id.nunique()==27
 assert meta.split.eq('validation').all() and meta.sampling_group.eq('main_50hz').all()
 data=crop_horizon({k:z[k] for k in z.files},meta.count_1000ms.to_numpy());merged,mlength,changed=merged_events(data)
 np.testing.assert_array_equal(data['events'][~changed],merged[~changed]);np.testing.assert_array_equal(data['event_length'][~changed],mlength[~changed])
 meta=meta.assign(changed_steps=changed.sum(1));meta.to_parquet(OUT/'validation_windows.parquet',index=False)
 np.savez_compressed(OUT/'merge_changes.npz',i=np.where(changed)[0],k=np.where(changed)[1],before=data['events'][changed],after=merged[changed],before_length=data['event_length'][changed],after_length=mlength[changed])
 device=torch.device('cuda:0');assert torch.cuda.is_available();torch.set_num_threads(4)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 rows=[];checks=[];checkpoints={};archive={}
 # All windows use the same batch order and padded tensor shapes in both representations.
 for seed in [17,29,43]:
  path=TRAIN/'train_1000ms/GRU-Control'/f'seed_{seed}/best.pt';expected=json.loads((path.parent/'summary.json').read_text())['checkpoint_sha256'];assert sha(path)==expected;checkpoints[str(path.relative_to(ROOT))]=expected
  ck=torch.load(path,map_location=device,weights_only=False);model=VelocityRateGRU(ck['normalization'],control=True).to(device);model.load_state_dict(ck['model']);model.eval()
  predictions={}
  with torch.no_grad():
   for variant in ['native','merged']:
    out=[]
    for start in range(0,len(meta),512):
     end=min(start+512,len(meta));inputs={k:torch.as_tensor(data[k][start:end],device=device) for k in KEYS}
     if variant=='merged':inputs['events']=torch.as_tensor(merged[start:end],device=device);inputs['event_length']=torch.as_tensor(mlength[start:end],device=device)
     out.append(model(inputs).cpu().numpy())
    predictions[variant]=np.concatenate(out)
    print('inferred',seed,variant,flush=True)
  assert all(np.isfinite(x).all() for x in predictions.values())
  untouched=~changed.any(1);maxunchanged=float(np.abs(predictions['native'][untouched]-predictions['merged'][untouched]).max());assert maxunchanged==0
  checks.append(dict(seed=seed,unchanged_prediction_max_difference=maxunchanged))
  for h,countcol in [(500,'count_500ms'),(1000,'count_1000ms')]:
   idx=meta[countcol].to_numpy(int)-1;truth=data['target'][np.arange(len(meta)),idx]
   effect=np.array([changed[i,:k+1].any() for i,k in enumerate(idx)])
   for variant,pred in predictions.items():
    selected=pred[np.arange(len(meta)),idx]
    e=errors(torch.tensor(selected,dtype=torch.float64),torch.tensor(truth,dtype=torch.float64)).numpy();e[:,2]*=180/np.pi
    f=meta[['window_id','flight_id']].copy();f['seed']=seed;f['horizon_ms']=h;f['variant']=variant;f['affected']=effect
    f['actual_time_s']=np.sum(data['dt']*(np.arange(data['dt'].shape[1])[None,:]<=idx[:,None]),axis=1)
    for j,metric in enumerate(METRICS):f[metric]=e[:,j]
    rows.append(f)
    archive[f'{seed}_{h}_{variant}']=selected
   archive[f'{h}_target']=truth;archive[f'{h}_affected']=effect
  archive[f'{seed}_native_changed']=predictions['native'][changed.any(1)];archive[f'{seed}_merged_changed']=predictions['merged'][changed.any(1)]
  # A second forward with exactly the same first batch checks deterministic inference.
  with torch.no_grad():
   inputs={k:torch.as_tensor(data[k][:512],device=device) for k in KEYS};again=model(inputs).cpu().numpy()
  np.testing.assert_array_equal(again,predictions['native'][:512])
 pd.concat(rows).to_parquet(OUT/'endpoint_errors.parquet',index=False)
 archive['changed_window_indices']=np.flatnonzero(changed.any(1));np.savez_compressed(OUT/'predictions.npz',**archive)
 save(OUT/'evaluation_manifest.json',dict(windows=len(meta),flights=meta.flight_id.nunique(),changed_windows=int(changed.any(1).sum()),changed_steps=int(changed.sum()),
   cache_sha256=digest,checkpoints=checkpoints,checks=checks,device=torch.cuda.get_device_name(device),seconds=time.time()-tick,
   model_code_sha256=sha(ROOT/'src/system_identification/models/velocity_rate_gru.py'),code_sha256=sha(Path(__file__)),test_opened=False,retrained=False,
   rule='exact equality of all four float32 command values within each native state interval; merge durations on integer microsecond grid; preserve shape, all genuine discontinuities and native dt'))
 print('COMPLETE',time.time()-tick,flush=True)
if __name__=='__main__':main()
