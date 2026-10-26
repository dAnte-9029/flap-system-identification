"""Frozen four-cell factorial replay; only future event inputs vary within receiver."""
from common import *
import time,torch
replay=module(OLD/'run.py','frozen_replay');audit=replay.response.audit

def require_support(t,age,start,end,max_age=160000):
 assert end>start and start>=t[0] and end<=t[-1],(start,end,t[0],t[-1])
 knots=np.unique(np.r_[start,t[(t>start)&(t<end)],end]);idx=np.searchsorted(t,knots,side='right')-1
 assert np.all(age[idx]+knots-t[idx]<=max_age),'stale command'
 return int(np.max(age[idx]+knots-t[idx]))

def main():
 if OUT.exists() and any(OUT.iterdir()):raise FileExistsError(OUT)
 OUT.mkdir(parents=True,exist_ok=True);tick=time.time();cfg=json.loads((HERE/'config.json').read_text());source={}
 for exp in [OLD,TIMING]:
  for f in json.loads((exp/'archive_manifest.json').read_text())['files']:
   path=exp/f['path'];assert sha(path)==f['sha256'],path;source[str(path.relative_to(ROOT))]=f['sha256']
 for name in ['report_zh.md','README.md','config.json','results/frozen_selection.json','results/description/asset_receipt.json','results/description/selected_pair_index.csv','results/description/background_index.csv']:
  source[str((SCOPE/name).relative_to(ROOT))]=sha(SCOPE/name)
 z=dict(np.load(OLD/'results/native_rollouts.npz'));w=pd.read_csv(OLD/'results/windows.csv');pairs=pd.read_csv(OLD/'results/pairs.csv');mem=pd.read_csv(OLD/'results/cohorts.csv')
 assert set(w.split)=={'train','validation'} and len(mem.query("cohort=='main' and split=='train'"))==48 and len(mem.query("cohort=='main' and split=='validation'"))==1
 # Pure historical input payload: no truth or target arrays passed to models.
 original={k:z[k].copy() for k in replay.KEYS};checkpoints=replay.checkpoint_manifest();data_manifest=json.loads((DATA/'dataset_manifest.json').read_text())
 assert sha(DATA/'dataset_manifest.json')==json.loads((OLD/'results/input_receipt.json').read_text())['data']['manifest_sha256']
 natives={};ages={};datahash={}
 for sp in ['train','validation']:
  frames=[]
  for stem in ['native_actuator_motors','native_actuator_servos']:
   name=f'{stem}_{sp}.parquet';datahash[name]=sha(DATA/name);assert datahash[name]==data_manifest['artifact_sha256'][name]
   frame=pd.read_parquet(DATA/name);frames.append({f:g for f,g in frame[frame.flight_id.isin(w[w.split==sp].flight_id)].groupby('flight_id')})
  for fid in frames[0]:
   t,u,age=audit.tape(frames[0][fid],frames[1][fid]);natives[fid]=(t,u);ages[fid]=age
 bounds=z['native_time_us']+w.initial_measurement_us.to_numpy()[:,None]
 counts=np.array([np.count_nonzero(bounds[i,1:int(n)+1]<=w.decision_us.iloc[i]+500000) for i,n in enumerate(z['future_count'])])
 # Before any inference, audit own native events against the saved canonical encoder.
 parts=[];window_checks=[];step_rows=[]
 for x in w.itertuples():
  i=x.rollout_index;t,u=natives[x.flight_id];own=replay.response.event_pack(t,u,bounds[i]);parts.append(own)
  age=require_support(t,ages[x.flight_id],int(x.decision_us),int(x.command_end_us))
  h=int(z['history_length'][i])-1;phase=float(np.arctan2(z['history'][i,h,17],z['history'][i,h,18]));freq=float(z['history'][i,h,19]);end=float(z['native_time_us'][i,counts[i]])*1e-6
  window_checks.append(dict(window_id=x.window_id,split=x.split,flight_id=x.flight_id,position_t0_us=int(x.initial_measurement_us),decision_us=int(x.decision_us),decision_latency_us=int(x.decision_us-x.initial_measurement_us),original_steps=int(z['future_count'][i]),supported_steps=int(counts[i]),supported_end_s=end,truncated=bool(counts[i]!=z['future_count'][i]),maximum_control_age_us=age,last_phase=phase,last_frequency=freq))
  for k in range(int(counts[i])):
   step_rows.append(dict(window_id=x.window_id,step=k,start_us=int(bounds[i,k]),end_us=int(bounds[i,k+1]),dt_us=int(bounds[i,k+1]-bounds[i,k]),phase_at_start=phase+2*np.pi*freq*z['native_time_us'][i,k]*1e-6,phase_at_end=phase+2*np.pi*freq*z['native_time_us'][i,k+1]*1e-6))
 own=replay.response.pad_events(parts,z['dt'].shape[1]);own['events'],own['event_length']=replay.canonicalize_control_events_numpy(own['events'],own['event_length'],z['dt'])
 np.testing.assert_array_equal(own['events'],z['events']);np.testing.assert_array_equal(own['event_length'],z['event_length'])
 cases=[];seq=[];validcounts=[]
 for pair in pairs.itertuples():
  for cell,receiver,donor in [('AA',pair.a,pair.a),('AB',pair.a,pair.b),('BA',pair.b,pair.a),('BB',pair.b,pair.b)]:
   r=w.iloc[int(receiver)];d=w.iloc[int(donor)];assert r.split==d.split==pair.split
   tape=replay.response.splice_tape(natives[r.flight_id],natives[d.flight_id],int(r.decision_us),int(d.decision_us),500000)
   n=int(counts[int(receiver)]);bb=bounds[int(receiver)].copy();bb[n+1:]=bb[n]
   # Full supported steps only; preserve own prefix before receiver decision.
   seq.append(replay.response.event_pack(*tape,bb));validcounts.append(n)
   cases.append(dict(case_id=len(cases),pair_uid=pair.pair_uid,split=pair.split,cell=cell,receiver=int(receiver),donor=int(donor),receiver_window=r.window_id,donor_window=d.window_id,receiver_flight=r.flight_id,donor_flight=d.flight_id,shift_us=int(r.decision_us-d.decision_us),valid_steps=n,support_end_s=(bb[n]-bb[0])*1e-6))
 cases=pd.DataFrame(cases);ix=cases.receiver.to_numpy();data={k:original[k][ix].copy() for k in replay.KEYS if k not in ['events','event_length']}
 for i,n in enumerate(validcounts):data['dt'][i,n:]=0
 data.update(replay.response.pad_events(seq,data['dt'].shape[1]));data['events'],data['event_length']=replay.canonicalize_control_events_numpy(data['events'],data['event_length'],data['dt'])
 for j in range(0,len(cases),4):
  for a,b in [(j,j+1),(j+2,j+3)]:
   for k in replay.KEYS:
    if k not in ['events','event_length']:np.testing.assert_array_equal(data[k][a],data[k][b])
 # Save inputs and membership before reading any new prediction result.
 np.savez_compressed(OUT/'inputs.npz',**data,valid_steps=np.array(validcounts),receiver_index=ix,native_time_us=z['native_time_us'][ix])
 cases.to_csv(OUT/'cases.csv',index=False);w.to_csv(OUT/'windows.csv',index=False);pairs.to_csv(OUT/'pairs.csv',index=False);mem.to_csv(OUT/'cohorts.csv',index=False)
 pd.DataFrame(window_checks).to_csv(OUT/'time_contract.csv',index=False);pd.DataFrame(step_rows).to_csv(OUT/'step_contract.csv',index=False)
 save(OUT/'source_receipt.json',dict(base_sha=cfg['base_sha'],prior_sha256=source,data_sha256=datahash,dataset_manifest_sha256=sha(DATA/'dataset_manifest.json'),checkpoints=checkpoints,original_rollouts_sha256=sha(OLD/'results/native_rollouts.npz'),config_sha256=sha(HERE/'config.json'),test_read=False))
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 device=torch.device(cfg['device']);checks=[];models=[];preds={}
 with torch.inference_mode():
  # Gate all six full factual replays BEFORE any exchanged prediction.
  for record in checkpoints:
   model,name,seed,norm=replay.model_from(record,device);pred=replay.predict(model,original,device);err=float(np.abs(pred-z[f'{name}_{seed}']).max());assert err<cfg['reference_atol']
   checks.append(dict(check='full factual archive replay',model=name,seed=seed,max_abs_error=err));models.append((model,name,seed,norm))
  print('All six factual gates passed',flush=True)
  for model,name,seed,norm in models:
   hidden=[];phase_inputs=[]
   h1=model.history_gru.register_forward_hook(lambda m,a,o:hidden.append(o[0].detach().cpu().numpy()))
   h2=model.step_gru.register_forward_pre_hook(lambda m,a:phase_inputs.append(a[0][:,12:18].detach().cpu().numpy()))
   pred=replay.predict(model,data,device);h1.remove();h2.remove();offset=0;maxhidden=0.
   for h in hidden:
    for j in range(0,len(h),4):
     for a,b in [(j,j+1),(j+2,j+3)]:
      n=int(data['history_length'][offset+a]);np.testing.assert_array_equal(h[a,n-1],h[b,n-1])
    offset+=len(h)
   for block in phase_inputs:
    np.testing.assert_array_equal(block[0::4],block[1::4]);np.testing.assert_array_equal(block[2::4],block[3::4])
   ownerr=0.
   for j,row in cases.iterrows():
    if row.cell in ['AA','BB']:ownerr=max(ownerr,float(np.abs(pred[j,:row.valid_steps]-z[f'{name}_{seed}'][row.receiver,:row.valid_steps]).max()))
   assert ownerr<cfg['reference_atol']
   if name=='GRU-NoControl':np.testing.assert_array_equal(pred[0::4],pred[1::4]);np.testing.assert_array_equal(pred[2::4],pred[3::4])
   small={k:v[:8] for k,v in data.items()};ev,le=replay.canonical_run.refine(small['events'],small['event_length'],4);refined=dict(small,events=ev,event_length=le)
   np.testing.assert_array_equal(replay.predict(model,small,device),replay.predict(model,refined,device))
   poison=dict(small,truth=np.full((8,1,13),np.nan),future_phase=np.full((8,1),1e9));np.testing.assert_array_equal(replay.predict(model,small,device),replay.predict(model,poison,device))
   checks.append(dict(check='four-cell invariants',model=name,seed=seed,factual_prefix_max_abs_error=ownerr,history_hidden_identical=True,phase_and_dt_identical=True,redundant_events_identical=True,future_labels_excluded=True,nocontrol_identical=name=='GRU-NoControl',normalization_sha256=norm))
   preds[f'{name}_{seed}']=pred
 np.savez_compressed(OUT/'predictions.npz',**preds)
 for record in checkpoints:assert sha(ROOT/record['path'])==record['sha256']
 save(OUT/'inference_checks.json',dict(passed=True,checks=checks,test_read=False,trained=False,weight_updates=False,seconds=time.time()-tick,gpu=torch.cuda.get_device_name(device),torch=torch.__version__,case_count=len(cases),unique_windows=len(w),pair_count=len(pairs),truncated_windows=int(sum(counts!=z['future_count'])),predictions_sha256=sha(OUT/'predictions.npz'),inputs_sha256=sha(OUT/'inputs.npz')))
 print('Done',len(cases),'cases',time.time()-tick,flush=True)
if __name__=='__main__':main()
