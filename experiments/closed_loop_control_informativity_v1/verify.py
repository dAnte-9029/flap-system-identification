"""Independent numerical checks use explicit interval quadrature and augmented least squares."""
from extract import *
from scipy.linalg import lstsq

def main():
 m=pd.read_csv(RES/'windows.csv.gz');f=pd.read_parquet(RES/'causal_features.parquet');arr=dict(np.load(RES/'diagnostic_arrays.npz'));yy=arr['target'];preds=dict(np.load(RES/'crossfit_predictions.npz'))
 receipt=json.loads((RES/'source_receipt.json').read_text())
 assert not receipt['test_read'] and receipt['partitions']==['train','validation']
 for p,h in receipt['artifacts'].items():assert old.sha(p)==h
 for p,h in json.loads((RES/'upstream_hashes.json').read_text()).items():assert old.sha(ROOT/p)==h
 raw=json.loads((RES/'raw_access.json').read_text());fl=pd.read_csv(Path(CFG['data_root'])/'flights.csv');allowed=fl[fl.split.isin(CFG['partitions'])&(fl.sampling_group=='main_50hz')].set_index('flight_id')
 for r in raw:
  assert r['flight_id'] in allowed.index and r['split']==allowed.loc[r['flight_id'],'split'];assert r['sha256']==allowed.loc[r['flight_id'],'sha256']
 # Repeat source hash verification, not merely trusting the extraction receipt.
 for p,h in {r['path']:r['sha256'] for r in raw}.items():assert old.sha(p)==h
 # No overlap of full source support within each log.
 for (_,sha),g in m[m.nonoverlap].groupby(['split','log_sha']):
  g=g.sort_values('decision_us');assert np.all(g.history_start_us.to_numpy()[1:]>g.pair_support_end_us.to_numpy()[:-1])
 audits=pd.read_csv(RES/'causality_audit.csv');assert (audits.latest_history_future_us<=0).all();assert (audits.min_command_support_margin_us>=0).all()
 checks=[];maxfit=0
 # Independently rebuild preprocessing and fit using an augmented weighted least-squares system.
 fits=json.loads((RES/'crossfit_receipt.json').read_text())
 for rec in fits:
  fold=5 if rec['fold']=='validation' else rec['fold'];name=rec['variant'];ff=np.load(RES/'auxiliary_fits'/f'{name}_fold{fold}.npz');tr=ff['train_rows'];te=ff['eval_rows'];assert set(m.iloc[tr].log_sha).isdisjoint(m.iloc[te].log_sha);assert (m.iloc[tr].split=='train').all();assert m.iloc[tr].nonoverlap.all()
  assert set(m.iloc[tr].log_sha)==set(rec['train_logs'])
  if name not in ['SU_reference500','U_500'] or fold not in [0,5]:continue
  X=np.c_[f[rec['features']].to_numpy(float),np.column_stack([(m.controller_id==c).to_numpy(float) for c in rec['categories']])]
  train=X[tr];med=np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0 for col in train.T]);clean=np.c_[np.where(np.isfinite(train),train,med),~np.isfinite(train)]
  counts=m.iloc[tr].flight_id.value_counts().to_dict();w=np.array([1/counts[k] for k in m.iloc[tr].flight_id]);w/=sum(w)
  mu=np.sum(clean*w[:,None],axis=0);sd=np.sqrt(np.sum((clean-mu)**2*w[:,None],axis=0));sd[sd<1e-8]=1;Z=(clean-mu)/sd;y0=np.sum(yy[tr]*w[:,None],axis=0)
  A=np.vstack([Z*np.sqrt(w[:,None]),np.sqrt(rec['lambda_value'])*np.eye(Z.shape[1])]);B=np.vstack([(yy[tr]-y0)*np.sqrt(w[:,None]),np.zeros((Z.shape[1],12))]);beta=lstsq(A,B,lapack_driver='gelsd')[0]
  test=np.c_[np.where(np.isfinite(X[te]),X[te],med),~np.isfinite(X[te])];p=(test-mu)/sd@beta+y0;err=np.max(abs(p-preds[name][te]));maxfit=max(maxfit,float(err));assert err<1e-10
 # Every published conditional summary: independently aggregate flight mean squared residuals.
 stats=pd.read_csv(RES/'conditional_summary.csv');err_metric=0;direction_error=0
 for r in stats.itertuples():
  sel=(m.split==r.split)&(m.nonoverlap if r.subset=='nonoverlap' else True);col=CFG['horizons_ms'].index(r.horizon_ms)*4+COORDS.index(r.coordinate)
  tmp=pd.DataFrame({'fid':m.loc[sel,'flight_id'],'y2':yy[sel,col]**2,'e2':(yy[sel,col]-preds[r.variant][sel,col])**2})
  rawr=np.sqrt(tmp.groupby('fid').y2.mean().mean());rr=np.sqrt(tmp.groupby('fid').e2.mean().mean());err_metric=max(err_metric,abs(rr-r.residual_rms),abs(rawr-r.raw_rms));assert abs(rr-r.residual_rms)<1e-12
  ids=np.flatnonzero(sel);cnt=m.iloc[ids].flight_id.value_counts();ww=np.array([1/cnt[z] for z in m.iloc[ids].flight_id]);ww/=ww.sum();act=abs(yy[ids,col])>=CFG['direction_threshold']
  if act.any():
   direct=np.sum(ww[act]*(yy[ids[act],col]*preds[r.variant][ids[act],col]>0))/np.sum(ww[act]);direction_error=max(direction_error,abs(direct-r.direction_accuracy));assert abs(direct-r.direction_accuracy)<1e-12
 # Independently verify lagged Gram eigenvalues with weighted SVD on data, not eig(Gram).
 err_svd=0
 for file in RES.glob('gram_*.npz'):
  _,split,kind,order,unit=file.stem.split('_');g=np.load(file);ix=g['rows'];lr=g['lag_rows'];X=(yy if kind=='raw' else yy-preds['SU_reference500'])[lr,:4].reshape(len(ix),-1);cnt=m.iloc[ix].flight_id.value_counts();w=np.array([1/cnt[z] for z in m.iloc[ix].flight_id]);w/=w.sum();X=X-np.sum(w[:,None]*X,axis=0)
  if unit=='standardized':X=X/g['scale']
  sv=np.linalg.svd(X*np.sqrt(w[:,None]),compute_uv=False);se=np.sqrt(np.maximum(np.linalg.eigvalsh(g['gram']),0))[::-1];err_svd=max(err_svd,float(np.max(abs(sv-se))));assert np.max(abs(sv-se))<1e-10
 # Exact target integration on a deterministic 200-window sample in each split, using union of raw native timestamps.
 zoh_err=0;feature_err=0;integrals=0;data=Path(CFG['data_root'])
 for split in CFG['partitions']:
  motor=old.load_parquet(data,'native_actuator_motors',split);servo=old.load_parquet(data,'native_actuator_servos',split)
  ids=np.flatnonzero(m.split==split);ids=ids[np.linspace(0,len(ids)-1,200,dtype=int)]
  for fid,g in m.loc[ids].groupby('flight_id'):
   mm=motor[motor.flight_id==fid];ss=servo[servo.flight_id==fid];mt=mm.timestamp_us.to_numpy();st=ss.timestamp_us.to_numpy();mv=mm['control[0]'].to_numpy(dtype=float);sv=ss[['control[0]','control[1]','control[2]']].to_numpy(dtype=float)
   def u(t):
    raw=np.c_[mv[np.searchsorted(mt,t,side='right')-1],sv[np.searchsorted(st,t,side='right')-1]]
    return np.c_[raw[:,0],(raw[:,1]+raw[:,2])/2,(raw[:,2]-raw[:,1])/2,raw[:,3]]
   for row in g.itertuples():
    t=int(row.decision_us);u0=u(np.array([t]))[0]
    for k,H in enumerate(CFG['horizons_ms']):
     e=t+H*1000;knots=np.unique(np.r_[t,e,mt[(mt>t)&(mt<e)],st[(st>t)&(st<e)]]);uu=u(knots[:-1]);value=np.sum((uu-u0)*np.diff(knots)[:,None],axis=0)/(e-t);zoh_err=max(zoh_err,float(np.max(abs(value-yy[row.Index,k*4:k*4+4]))));integrals+=1
    for lag in CFG['history_lags_ms']:
     expected=u(np.array([t-lag*1000]))[0];got=f.loc[row.Index,[f'U{lag}_{c}' for c in COORDS]].to_numpy(float);feature_err=max(feature_err,float(np.max(abs(expected-got))))
 assert zoh_err<1e-12 and feature_err<1e-12
 # Negative-control convention checked from saved physical-time arrays.
 temporal=pd.read_csv(RES/'temporal_per_flight.csv');terr=0
 for r in temporal[(temporal.axis=='p')&(temporal.command=='raw_future300')&(temporal.horizon_ms==300)].itertuples():
  ix=np.flatnonzero((m.flight_id==r.flight_id)&m.nonoverlap);mtn=arr['motion'];z=mtn[ix,40,0]-mtn[ix,25,0] if r.motion_period=='future' else mtn[ix,25,0]-mtn[ix,10,0];x=yy[ix,6];good=np.isfinite(z)&np.isfinite(x)
  expected=np.corrcoef(x[good],z[good])[0,1];terr=max(terr,abs(expected-r.corr));assert abs(expected-r.corr)<1e-12
 # Published fixed index coverage and formal model results remain traceable.
 pair=pd.read_csv(RES/'pair_information.csv');coh=pd.read_csv(RES/'frozen_cohorts.csv');assert len(coh[(coh.split=='train')&(coh.cohort=='main')])==48
 assert set(pair.pair_uid)==set(pd.read_csv(HERE.parent/'structured_control_effectiveness_group_holdout_v1/results/pairs.csv').pair_uid)
 write_json(RES/'independent_verification.json',dict(passed=True,checked_raw_flights=len(raw),test_opened=False,summary_rows=len(stats),ridge_augmented_lstsq_max_error=maxfit,summary_rms_max_error=err_metric,direction_max_error=direction_error,gram_svd_max_error=err_svd,zoh_integrals=integrals,zoh_max_error=zoh_err,history_command_max_error=feature_err,negative_control_corr_max_error=terr,full_frozen_pairs=pair.pair_uid.nunique(),main_train_pairs=48,method='independent augmented SVD solve; independent per-flight reductions; direct native interval quadrature; no dynamics training'))
 print((RES/'independent_verification.json').read_text().strip())
if __name__=='__main__':main()
