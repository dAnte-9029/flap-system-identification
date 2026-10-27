"""Only T01/T10 new full integrations; T00/T11 archival values reused."""
from common import *

def main():
 torch.set_num_threads(4);oldwindows=pd.read_csv(PREV/'results/windows.csv.gz');history=pd.read_parquet(PREV/'results/history_states.parquet',columns=['split','flight_id','sample_index','B_p','B_q','B_r']);hg=dict(tuple(history.groupby('flight_id',sort=False)))
 rows=[];restored=[];reg=[];checks=[];receipt=json.loads((PREV/'results/receipt.json').read_text());total_saved=0;total_missing=0
 for split in ['train','validation']:
  meta=pd.read_parquet(TRAIN/f'{split}_windows.parquet');prior=oldwindows[oldwindows.split==split].set_index('registered_index').loc[meta.index];np.testing.assert_array_equal(meta.window_id,prior.window_id);np.testing.assert_array_equal(meta.flight_id,prior.flight_id)
  z=np.load(CACHE/f'{split}.npz');initial=z['initial'];targets=z['target'];dt=z['dt']
  # Times/native data only needed to recover unarchived tail B nodes, not for a new clock audit.
  timeframes=dict(tuple(pd.read_parquet(DATA/f'samples_{split}.parquet',columns=['flight_id','state_measurement_us']).groupby('flight_id',sort=False)))
  ng=dict(tuple(pd.read_parquet(DATA/f'native_vehicle_angular_velocity_{split}.parquet',columns=['flight_id','timestamp_sample_us','xyz[0]','xyz[1]','xyz[2]']).groupby('flight_id',sort=False)))
  for flight_no,(fid,mm) in enumerate(meta.groupby('flight_id',sort=False)):
   ids=mm.index.to_numpy();previous=prior.loc[ids];o=mm.origin_sample.to_numpy(int);count=mm.count_1000ms.to_numpy(int);k=int(count.max());fi=np.minimum(o[:,None]+np.arange(1,k+1),(o+count)[:,None]);prefix=np.arange(k)[None,:]<previous.common_prefix_steps.to_numpy()[:,None]
   hv=hg[fid].set_index('sample_index');assert not hv.index.duplicated().any();needed=np.unique(fi[prefix]);known=needed[np.isin(needed,hv.index)];missing=needed[~np.isin(needed,hv.index)];total_saved+=len(known);total_missing+=len(missing)
   table=hv[['B_p','B_q','B_r']].copy()
   if len(missing):
    ss=timeframes[fid].reset_index(drop=True);nn=ng[fid];ref=int(ss.state_measurement_us.iloc[0]);nt=(nn.timestamp_sample_us.to_numpy()-ref)*1e-6;ny=nn[['xyz[0]','xyz[1]','xyz[2]']].to_numpy(float);times=(ss.state_measurement_us.to_numpy()[missing]-ref)*1e-6;b,ok,_,_=sample(nt,ny,times)
    assert ok.all() and np.isfinite(b).all(),'previous support cannot be expanded or repaired'
    more=pd.DataFrame(b,index=missing,columns=table.columns);table=pd.concat([table,more])
    for index,value in zip(missing,b):restored.append(dict(split=split,flight_id=fid,sample_index=int(index),B_p=value[0],B_q=value[1],B_r=value[2]))
   b=table.reindex(fi.ravel()).to_numpy().reshape(len(ids),k,3);assert np.isfinite(b[prefix]).all()
   ini0=initial[ids].astype(float);ini1=ini0.copy();ini1[:,10:13]=previous[['C1_p','C1_q','C1_r']].to_numpy().astype(np.float32).astype(float)
   np.testing.assert_array_equal(ini0[:,10:13],previous[['C0_p','C0_q','C0_r']].to_numpy().astype(np.float32));np.testing.assert_array_equal(ini0[:,:10],ini1[:,:10])
   target=targets[ids,:k].astype(float);d=dt[ids,:k].astype(float);velocity=target[:,:,3:6];a=target[:,:,10:13]
   q01,p01=kinematics(ini0,velocity,b,d,target,prefix);q10,p10=kinematics(ini1,velocity,a,d,target,prefix);np.testing.assert_array_equal(p01,p10)
   # Tiny interface regression per flight; these checks are not new full T00/T11 runs.
   sel=np.unique(np.array([0,len(ids)-1]));maxerr=0.
   for name,ini,rate,oldtag in [('T00',ini0,a,'T0'),('T11',ini1,b,'T1')]:
    qq,_=kinematics(ini[sel],velocity[sel],rate[sel],d[sel],target[sel],prefix[sel])
    for j,local in enumerate(sel):
     used=prefix[local];n=int(used.sum());mse=float(np.mean(qq[j,used]**2));ep=float(qq[j,n-1]);maxerr=max(maxerr,abs(mse-previous[f'{oldtag}_attitude_mse'].iloc[local]),abs(ep-previous[f'{oldtag}_endpoint_rad'].iloc[local]))
   assert maxerr<1e-11,maxerr;reg.append(dict(split=split,flight_id=fid,windows_checked=len(sel),max_error=maxerr))
   for local,ix in enumerate(ids):
    r=previous.iloc[local];use=prefix[local];n=int(use.sum());row=dict(split=split,registered_index=int(ix),window_id=r.window_id,flight_id=fid,complete_B=bool(r.complete_B),future_steps=int(r.future_steps),common_prefix_steps=n,common_duration_s=r.common_duration_s,position_mse=r.position_mse)
    for name,mse,ep in [('T00',r.T0_attitude_mse,r.T0_endpoint_rad),('T11',r.T1_attitude_mse,r.T1_endpoint_rad),('T01',float(np.mean(q01[local,use]**2)),q01[local,n-1]),('T10',float(np.mean(q10[local,use]**2)),q10[local,n-1])]:row[f'{name}_path_mse']=mse;row[f'{name}_endpoint_mse']=ep**2
    for name,ini,rate in [('T00',ini0,a),('T01',ini0,b),('T10',ini1,a),('T11',ini1,b)]:
     jump=rate[local,0]-ini[local,10:13];row[f'{name}_first_jump_norm']=np.linalg.norm(jump)
     for j,axis in enumerate(VARS):row[f'{name}_first_jump_{axis}']=jump[j]
    for c in ['C0','C1']:
     for axis in VARS:row[f'{c}_{axis}_error']=r[f'{c}_{axis}_error']
    for quantity in ['path','endpoint']:
     for name,v in contrasts(*[row[f'{tag}_{quantity}_mse'] for tag in COMBOS]).items():row[f'{quantity}_{name}']=v
    rows.append(row)
   checks.append(dict(split=split,flight_id=fid,windows=len(ids),same_initial_position_velocity_attitude=True,same_dt_velocity_target_support=True,prefix_steps=int(prefix.sum()),B_saved_nodes=len(known),B_restored_nodes=len(missing)))
   if flight_no%20==0:print(split,flight_no+1,'flights',flush=True)
  del initial,targets,dt,z,timeframes,ng
 out=pd.DataFrame(rows);out.to_csv(OUT/'factorial_windows.csv.gz',index=False);pd.DataFrame(restored).to_csv(OUT/'restored_B_tail.csv.gz',index=False)
 save(OUT/'run_receipt.json',dict(base_sha='d111ff23bee90464f5946f7143eb94528da0d922',training_runs=0,test_read=False,model_inference=0,full_new_integrations=['T01','T10'],reused_full_results=['T00','T11'],previous_receipt_sha256=sha(PREV/'results/receipt.json'),previous_windows_sha256=sha(PREV/'results/windows.csv.gz'),previous_history_sha256=sha(PREV/'results/history_states.parquet'),integrator_sha256=sha(ROOT/'src/system_identification/models/velocity_rate_gru.py'),prior_data_hashes_reused=receipt['files'],no_raw_ULog=True,B_unique_saved_nodes=total_saved,B_unique_restored_nodes=total_missing,tail_recovery_reason='Previous artifacts save B at historical rows, not the full future tape; missing supported tail nodes restored with unchanged native interpolation, never C1 re-estimation.',regression=reg,checks=checks))
 print('Done',len(out),'windows; saved/restored B nodes',total_saved,total_missing,flush=True)
if __name__=='__main__':main()
