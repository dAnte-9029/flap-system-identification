"""Cross-fitted command diagnostics only: does not fit any motion or IV model."""
from extract import *
from scipy.signal import welch
from scipy.stats import spearmanr
import warnings

def weights(meta):
 n=meta.groupby('flight_id').flight_id.transform('size').to_numpy();w=1/n;return w/w.sum()
def ridge(x,y,groups,lam):
 # Fit all preprocessing exclusively on training rows. Equal total weight per flight.
 with warnings.catch_warnings():
  warnings.simplefilter('ignore');med=np.nanmedian(x,axis=0)
 med=np.nan_to_num(med);xx=np.where(np.isfinite(x),x,med)
 xx=np.c_[xx,~np.isfinite(x)]
 _,inv,cnt=np.unique(groups,return_inverse=True,return_counts=True);w=1/cnt[inv];w/=w.sum()
 mu=w@xx;sd=np.sqrt(w@((xx-mu)**2));sd[sd<1e-8]=1
 z=(xx-mu)/sd;ym=w@y
 b=np.linalg.solve(z.T@(w[:,None]*z)+lam*np.eye(z.shape[1]),z.T@(w[:,None]*(y-ym)))
 return dict(median=med,mean=mu,scale=sd,beta=b,intercept=ym)
def predict(f,x):
 xx=np.c_[np.where(np.isfinite(x),x,f['median']),~np.isfinite(x)]
 return (xx-f['mean'])/f['scale']@f['beta']+f['intercept']
def metrics(y,p,w):
 r=y-p;raw=np.sqrt(w@(y*y));res=np.sqrt(w@(r*r));active=np.abs(y)>=CFG['direction_threshold']
 return dict(raw_rms=raw,residual_rms=res,residual_rms_ratio=res/raw if raw else np.nan,unexplained_mse_ratio=(res/raw)**2 if raw else np.nan,mean_residual=w@r,positive_fraction=w@(y>CFG['direction_threshold']),negative_fraction=w@(y<-CFG['direction_threshold']),near_zero_fraction=w@(~active),direction_accuracy=(w[active]@(y[active]*p[active]>0))/w[active].sum() if active.any() else np.nan,remaining_above_0p02=w@(np.abs(r)>=.02))
def feature_columns(f,variant):
 hist=200 if variant.endswith('200') else 500
 prefixes=['U'] if variant.startswith('U_') else ['U','S']
 if 'mission' in variant:prefixes+=['M']
 if 'reference' in variant or 'internal' in variant:prefixes+=['M','R']
 if 'internal' in variant:prefixes+=['I']
 cols=[c for c in f if c[0] in prefixes and c.split('_')[0][1:].isdigit() and int(c.split('_')[0][1:])<=hist]
 if 'M' in prefixes:cols+=['mission_changed_500']
 return cols

def main():
 meta=pd.read_csv(RES/'windows.csv.gz');f=pd.read_parquet(RES/'causal_features.parquet');a=np.load(RES/'diagnostic_arrays.npz');y=a['target'];N=len(meta)
 foldmap=pd.read_csv(RES/'crossfit_groups.csv').set_index('log_sha').fold.to_dict();meta['fold']=meta.log_sha.map(foldmap).fillna(-1).astype(int)
 # Categories known as metadata only; fitted one-hot vocabulary and coefficients exclude holdout.
 variants=['U_200','U_500','SU_200','SU_500','SU_mission500','SU_reference500','SU_internal500']
 runs=[(v,CFG['ridge_lambda']) for v in variants]+[('SU_reference500',l) for l in CFG['ridge_sensitivity']]
 preds={};summary=[];perflight=[];records=[];coeff=[]
 if '--resume' in sys.argv:
  preds=dict(np.load(RES/'crossfit_predictions.npz'));runs=[]
 for variant,lam in runs:
  name=variant if lam==CFG['ridge_lambda'] else f'{variant}_lambda{lam}'
  cols=feature_columns(f,variant);x=f[cols].to_numpy(float);out=np.full_like(y,np.nan)
  for fold in range(CFG['crossfit_folds']+1):
   val=fold==CFG['crossfit_folds'];tr=(meta.split=='train')&meta.nonoverlap
   te=(meta.split=='validation') if val else (meta.split=='train')&(meta.fold==fold)
   if not val:tr&=meta.fold!=fold
   trainlogs=set(meta.loc[tr,'log_sha']);evallogs=set(meta.loc[te,'log_sha']);assert trainlogs.isdisjoint(evallogs)
   cats=sorted(meta.loc[tr,'controller_id'].unique())
   xx=np.c_[x,np.column_stack([(meta.controller_id==c).to_numpy(float) for c in cats])]
   model=ridge(xx[tr],y[tr],meta.loc[tr,'flight_id'].to_numpy(),lam)
   out[te]=predict(model,xx[te])
   # Numerical reproducibility of fit and exact membership are stored, no neural weights.
   dest=RES/'auxiliary_fits';dest.mkdir(exist_ok=True)
   np.savez_compressed(dest/f'{name}_fold{fold}.npz',**model,train_rows=np.flatnonzero(tr),eval_rows=np.flatnonzero(te))
   records.append(dict(variant=name,fold='validation' if val else fold,train_windows=int(tr.sum()),eval_windows=int(te.sum()),train_logs=sorted(trainlogs),eval_logs=sorted(evallogs),features=cols,categories=cats,lambda_value=lam))
  assert np.isfinite(out).all();preds[name]=out
  for split in CFG['partitions']:
   for subset in ['all','nonoverlap']:
    mask=(meta.split==split)&(meta.nonoverlap if subset=='nonoverlap' else True)
    ids=np.flatnonzero(mask);w=weights(meta.loc[mask])
    for h,H in enumerate(CFG['horizons_ms']):
     for j,ch in enumerate(COORDS):summary.append(dict(variant=name,split=split,subset=subset,horizon_ms=H,coordinate=ch,windows=len(ids),flights=meta.loc[mask,'flight_id'].nunique(),**metrics(y[ids,h*4+j],out[ids,h*4+j],w)))
   for fid,g in meta[(meta.split==split)&meta.nonoverlap].groupby('flight_id'):
    ids=g.index.to_numpy()
    for h,H in enumerate(CFG['horizons_ms']):
     for j,ch in enumerate(COORDS):perflight.append(dict(variant=name,split=split,flight_id=fid,date=g.date.iloc[0],firmware=g.firmware.iloc[0],controller_id=g.controller_id.iloc[0],horizon_ms=H,coordinate=ch,windows=len(g),**metrics(y[ids,h*4+j],out[ids,h*4+j],np.ones(len(g))/len(g))))
  print('fit',name,flush=True)
 if records:
  np.savez_compressed(RES/'crossfit_predictions.npz',**preds)
  write_json(RES/'crossfit_receipt.json',records)
  pd.DataFrame(summary).to_csv(RES/'conditional_summary.csv',index=False);pd.DataFrame(perflight).to_csv(RES/'conditional_per_flight.csv',index=False)
 base=preds['SU_reference500'];res=y-base
 # Correlations and low-order lagged matrices, rows selected independent of movement.
 coupling=[];info=[];weak=[];leave=[]
 for split in CFG['partitions']:
  mask=(meta.split==split)&meta.nonoverlap;ids=np.flatnonzero(mask);w=weights(meta.loc[mask])
  for label,xx in [('raw',y),('conditional',res)]:
   xx=xx[ids,4:8];mu=w@xx;cov=(xx-mu).T@(w[:,None]*(xx-mu));sd=np.sqrt(np.diag(cov));cc=cov/np.outer(sd,sd)
   for i,ch in enumerate(COORDS):
    for j,cj in enumerate(COORDS):coupling.append(dict(split=split,kind=label,a=ch,b=cj,correlation=cc[i,j]))
  for order in CFG['lag_orders']:
   matrix_rows=[];rowids=[]
   for fid,g in meta[meta.split==split].groupby('flight_id'):
    tt=g.decision_us.to_numpy(np.int64);ii=g.index.to_numpy(); origins=np.flatnonzero(g.nonoverlap)
    for oi in origins:
     idx=oi-np.arange(order)
     if (idx<0).any() or (order>1 and np.max(np.diff(tt[idx[::-1]]))>150000):continue
     if len(set(g.segment_id.to_numpy()[idx]))!=1:continue
     matrix_rows.append(ii[idx]);rowids.append(ii[oi])
   matrix_rows=np.array(matrix_rows,dtype=int);rowids=np.array(rowids,dtype=int);ww=weights(meta.loc[rowids])
   for label,xx in [('raw',y),('conditional',res)]:
    X=xx[matrix_rows,:4].reshape(len(rowids),-1);mu=ww@X;X=X-mu;scale=np.sqrt(ww@(X*X));scale[scale<1e-10]=1
    for units,Z in [('command',X),('standardized',X/scale)]:
     gram=Z.T@(ww[:,None]*Z);eig,V=np.linalg.eigh(gram);sv=np.sqrt(np.maximum(eig,0))[::-1];energy=np.maximum(eig,0)/max(eig.sum(),1e-30)
     info.append(dict(split=split,kind=label,order=order,units=units,rows=len(X),flights=meta.loc[rowids,'flight_id'].nunique(),dimension=X.shape[1],smax=sv[0],smin=sv[-1],condition=sv[0]/max(sv[-1],1e-30),relative_rank=int((sv>sv[0]*.01).sum()),entropy_effective_rank=float(np.exp(-np.sum(energy*np.log(energy+1e-30))))))
     for k,val in enumerate(V[:,0]):weak.append(dict(split=split,kind=label,order=order,units=units,lag_index=k//4,coordinate=COORDS[k%4],coefficient=val))
     np.savez_compressed(RES/f'gram_{split}_{label}_{order}_{units}.npz',gram=gram,rows=rowids,lag_rows=matrix_rows,lag_age_us=meta.decision_us.to_numpy()[rowids,None]-meta.decision_us.to_numpy()[matrix_rows],mean=mu,scale=scale)
     if units=='standardized' and order==6:
      for fid in meta.loc[rowids,'flight_id'].unique():
       keep=meta.loc[rowids,'flight_id'].to_numpy()!=fid;w2=weights(meta.loc[rowids[keep]]);z=Z[keep];z=z-w2@z;e=np.linalg.eigvalsh(z.T@(w2[:,None]*z));leave.append(dict(split=split,kind=label,omitted_flight=fid,condition=np.sqrt(max(e[-1],0)/max(e[0],1e-30))))
 pd.DataFrame(coupling).to_csv(RES/'coupling.csv',index=False);pd.DataFrame(info).to_csv(RES/'information_matrices.csv',index=False);pd.DataFrame(weak).to_csv(RES/'weak_directions.csv',index=False);pd.DataFrame(leave).to_csv(RES/'information_leave_flight_out.csv',index=False)
 # Native control spectrum: 20ms backward ZOH, individual contiguous AUTO intervals >=4s.
 native=pd.read_csv(RES/'native_auto_controls.csv.gz');ps=[]
 for (split,fid,segment),g in native.groupby(['split','flight_id','interval']):
  t=g.time_us.to_numpy(np.int64);end=int(t[-1]+g.dt_us.iloc[-1]);tt=np.arange(t[0],end,20000)
  if len(tt)<200:continue
  xx=old.zoh(t,g[COORDS].to_numpy(),tt);freq,pow=welch(xx,fs=50,nperseg=200,noverlap=100,axis=0,detrend='constant')
  for k,fr in enumerate(freq):
   for j,ch in enumerate(COORDS):ps.append(dict(split=split,flight_id=fid,interval=segment,coordinate=ch,hz=fr,power=pow[k,j],seconds=len(tt)/50))
 pd.DataFrame(ps).to_csv(RES/'native_psd.csv.gz',index=False)
 # Residual spectrum at 10Hz applies to 100ms-average innovations, not native actuator commands.
 ps=[]
 for fid,g in meta.groupby('flight_id'):
  tt=g.decision_us.to_numpy();ii=g.index.to_numpy();breaks=np.r_[0,np.where(np.diff(tt)>150000)[0]+1,len(g)]
  for lo,hi in zip(breaks[:-1],breaks[1:]):
   if hi-lo<40:continue
   ts=np.arange(tt[lo],tt[hi-1]+1,100000);idx=np.searchsorted(tt[lo:hi],ts,side='right')-1
   if len(ts)<40:continue
   for name,xx in [('raw100mean',y),('residual100mean',res)]:
    vv=xx[ii[lo:hi][idx],:4];freq,power=welch(vv,fs=10,nperseg=40,noverlap=20,axis=0)
    for k,fr in enumerate(freq):
     for j,ch in enumerate(COORDS):ps.append(dict(split=g.split.iloc[0],flight_id=fid,kind=name,coordinate=ch,hz=fr,power=power[k,j],seconds=len(ts)/10))
 pd.DataFrame(ps).to_csv(RES/'conditional_psd.csv.gz',index=False)
 temporal(meta,a,y,res,f)
 pairs(meta,y,preds,f)
 coverage_report(meta,native)


def assoc(x,y):
 ok=np.isfinite(x)&np.isfinite(y);x=x[ok];y=y[ok]
 if len(x)<4 or np.std(x)<1e-10 or np.std(y)<1e-10:return dict(n=len(x),corr=np.nan,slope=np.nan)
 return dict(n=len(x),corr=np.corrcoef(x,y)[0,1],slope=np.dot(x-x.mean(),y-y.mean())/np.sum((x-x.mean())**2))

def temporal(meta,a,y,res,f):
 motion=a['motion'];ctrl=a['control'];ref=a['diagnostic_reference'];rr=[];event=[];relevance=[]
 for fid,g in meta[meta.nonoverlap].groupby('flight_id'):
  ids=g.index.to_numpy();common=dict(split=g.split.iloc[0],flight_id=fid,date=g.date.iloc[0],controller_id=g.controller_id.iloc[0],firmware=g.firmware.iloc[0])
  for H in [100,200,300,400,500]:
   k=H//20
   past=motion[ids,25]-motion[ids,25-k];future=motion[ids,25+k]-motion[ids,25]
   # Same fixed future300 control variable vs already completed past response (negative control).
   for source,x in [('raw_future300',y[ids,6]),('conditional_future300',res[ids,6])]:
    for sign,yy in [('past',past),('future',future)]:
     for j,ch in enumerate(['p','q','r']):rr.append(dict(**common,horizon_ms=H,command=source,motion_period=sign,axis=ch,**assoc(x,yy[:,j])))
   qualifying=(g.differential_dominant.to_numpy())
   for j,ch in enumerate(['p','q','r']):
    for direction,yy in [('past',past[:,j]),('future',future[:,j])]:
     for n in np.where(qualifying)[0]:
      event.append(dict(**common,window_id=g.window_id.iloc[n],horizon_ms=H,axis=ch,period=direction,control_mean=y[ids[n],10],motion_change=yy[n],signed_motion=np.sign(y[ids[n],10])*yy[n]))
  for lag in [-500,-300,-100,0,100,300,500]:
   k=25+lag//20
   for j,name in [(0,'roll_rate_setpoint'),(1,'pitch_rate_setpoint'),(2,'yaw_rate_setpoint'),(6,'roll_torque')]:
    relevance.append(dict(**common,reference=name,lag_ms=lag,**assoc(y[ids,6],ref[ids,k,j]-ref[ids,0,j])))
  for name,col in [('mission_north','M0_north_m'),('mission_east','M0_east_m'),('mission_alt','M0_alt_error'),('roll_rate_error',None)]:
   x=(f.loc[ids,'R0_vehicle_rates_setpoint_roll'].to_numpy()-f.loc[ids,'S0_p'].to_numpy()) if col is None else f.loc[ids,col].to_numpy()
   relevance.append(dict(**common,reference=name,lag_ms=0,**assoc(y[ids,6],x)))
 pd.DataFrame(rr).to_csv(RES/'temporal_per_flight.csv',index=False);pd.DataFrame(event).to_csv(RES/'event_motion.csv.gz',index=False);pd.DataFrame(relevance).to_csv(RES/'reference_relevance.csv',index=False)
 # Native-time curve cases are all qualified events, no motion-based selection.
 np.savez_compressed(RES/'temporal_curves.npz',grid_us=a['grid_us'],rows=np.flatnonzero(meta.nonoverlap&meta.differential_dominant),motion=motion[meta.nonoverlap&meta.differential_dominant],control=ctrl[meta.nonoverlap&meta.differential_dominant])


def pairs(meta,y,preds,f):
 prev=HERE.parent/'structured_control_effectiveness_group_holdout_v1/results'
 pp=pd.read_csv(prev/'pairs.csv');coh=pd.read_csv(prev/'cohorts.csv');comp=pd.read_csv(prev/'component_assignments.csv')
 idx=dict(zip(meta.window_id,meta.index));out=[]
 for r in pp.itertuples():
  assert r.window_a in idx and r.window_b in idx
  i,j=idx[r.window_a],idx[r.window_b]
  for variant,p in preds.items():
   rec=dict(pair_uid=r.pair_uid,split=r.split,variant=variant,flight_a=r.flight_a,flight_b=r.flight_b,same_firmware=r.same_firmware,same_controller=r.same_controller,strict=r.strict,history_rate_rms=r.rate_rms,historical_control_contrast=r.difference_differential_rms,command_difference_300=y[i,6]-y[j,6],explained_difference_300=p[i,6]-p[j,6],residual_difference_300=(y[i,6]-p[i,6])-(y[j,6]-p[j,6]),origin_rate_gap_a=meta.rate_initial_gap.iloc[i],origin_rate_gap_b=meta.rate_initial_gap.iloc[j],mission_changed_a=f.mission_changed_500.iloc[i],mission_changed_b=f.mission_changed_500.iloc[j],rate_reference_available_a=meta.reference_complete.iloc[i],rate_reference_available_b=meta.reference_complete.iloc[j],dwell_a_ms=meta.differential_longest_us.iloc[i]/1000,dwell_b_ms=meta.differential_longest_us.iloc[j]/1000)
   for c,ch in enumerate(COORDS):rec['delta_'+ch]=y[i,4+c]-y[j,4+c]
   out.append(rec)
 p=pd.DataFrame(out).merge(comp[['pair_uid','component_id','fold']],how='left',on='pair_uid');p.to_csv(RES/'pair_information.csv',index=False)
 # Attach saved predictions, not re-run GRU. Main pairs use their frozen heldout fold.
 factual=pd.read_csv(prev/'factual_points.csv.gz')
 print('factual columns',factual.columns.tolist(),flush=True)
 factual=factual[(factual.time_label.astype(str)=='300')&(factual.variable=='p')&(factual.method.isin(['B','D']))]
 merged=factual.merge(p[p.variant=='SU_reference500'],on='pair_uid',suffixes=('','_information'))
 merged.to_csv(RES/'pair_failure_links.csv',index=False)
 out=[]
 for cohort,g in coh.groupby('cohort'):
  for split in CFG['partitions']:
   ids=g.loc[g.split==split,'pair_uid'];gg=p[(p.split==split)&p.pair_uid.isin(ids)]
   for variant,h in gg.groupby('variant'):
    out.append(dict(cohort=cohort,split=split,variant=variant,pairs=len(h),raw_contrast_rms=np.sqrt(np.mean(h.command_difference_300**2)),residual_contrast_rms=np.sqrt(np.mean(h.residual_difference_300**2)),fraction_residual_below_0p02=np.mean(abs(h.residual_difference_300)<.02),mean_history_rate_rms=h.history_rate_rms.mean()))
 pd.DataFrame(out).to_csv(RES/'pair_cohort_summary.csv',index=False)
 coh.to_csv(RES/'frozen_cohorts.csv',index=False)


def coverage_report(meta,native):
 rows=[]
 for (split,fid),g in meta.groupby(['split','flight_id']):
  n=native[(native.split==split)&(native.flight_id==fid)];w=n.dt_us/n.dt_us.sum()
  rec=dict(split=split,flight_id=fid,windows=len(g),nonoverlap=int(g.nonoverlap.sum()),union_seconds=n.dt_us.sum()/1e6,native_intervals=n.interval.nunique(),date=g.date.iloc[0],firmware=g.firmware.iloc[0],controller_id=g.controller_id.iloc[0],differential_dominant_nonoverlap=int((g.differential_dominant&g.nonoverlap).sum()),quiet_nonoverlap=int((g.quiet&g.nonoverlap).sum()),rate_gap_rms=np.sqrt(np.mean(g.rate_initial_gap**2)))
  for ch in COORDS:rec[ch+'_std']=np.sqrt(w@((n[ch]-w@n[ch])**2))
  rows.append(rec)
 pd.DataFrame(rows).to_csv(RES/'coverage_per_flight.csv',index=False)
if __name__=='__main__':main()
