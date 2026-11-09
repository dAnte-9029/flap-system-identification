"""Train-only grouped A2 diagnostic; no dynamics model or held-out dataset access."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
class StandardScaler:
 def fit(self,x):
  self.mean_=np.mean(x,axis=0);self.scale_=np.std(x,axis=0);self.scale_[self.scale_<1e-12]=1;return self
 def transform(self,x):return (x-self.mean_)/self.scale_
class Ridge:
 def __init__(self,alpha=1,fit_intercept=True):self.alpha=alpha;self.intercept=fit_intercept
 def fit(self,x,y,sample_weight=None):
  w=np.ones(len(x)) if sample_weight is None else sample_weight;self.xm=np.average(x,axis=0,weights=w) if self.intercept else np.zeros(x.shape[1]);self.ym=np.average(y,axis=0,weights=w) if self.intercept else np.zeros(y.shape[1]);a=(x-self.xm)*np.sqrt(w[:,None]);b=(y-self.ym)*np.sqrt(w[:,None]);self.coef_=np.linalg.solve(a.T@a+self.alpha*np.eye(x.shape[1]),a.T@b).T;return self
 def predict(self,x):return (x-self.xm)@self.coef_.T+self.ym
class GroupKFold:
 def __init__(self,n):self.n=n
 def split(self,x,y,g):
  ids,counts=np.unique(g,return_counts=True);bins=[[] for _ in range(self.n)];sizes=np.zeros(self.n)
  for j in np.argsort(-counts,kind='stable'):
   k=np.argmin(sizes);bins[k].append(ids[j]);sizes[k]+=counts[j]
  for b in bins:
   test=np.isin(g,b);yield np.flatnonzero(~test),np.flatnonzero(test)
class NearestNeighbors:
 def __init__(self,n_neighbors):self.k=n_neighbors
 def fit(self,x):self.tree=cKDTree(x);return self
 def kneighbors(self,x):return self.tree.query(x,k=self.k)
class ScaledRidge:
 def __init__(self,alpha):self.alpha=alpha
 def fit(self,x,y):self.sc=StandardScaler().fit(x);self.m=Ridge(self.alpha).fit(self.sc.transform(x),y);return self
 def predict(self,x):return self.m.predict(self.sc.transform(x))
from system_identification.data.pacld import open_windows
OUT=Path('experiments/phase_anchored_coupled_latent_v1/control_identifiability'); OUT.mkdir(exist_ok=True)
C=['motor','differential','symmetric','rudder']
def transform(a):
 return np.stack([a[...,0],(a[...,2]-a[...,1])/2,(a[...,1]+a[...,2])/2,a[...,3]],axis=-1)
def fit(x,y,g):
 scores=[]
 for alpha in [1.,10.,100.,1000.]:
  errors=[]
  for tr,te in GroupKFold(3).split(x,y,g):
   err=(ScaledRidge(alpha).fit(x[tr],y[tr]).predict(x[te])-y[te])**2
   errors.extend([err[g[te]==f].mean() for f in np.unique(g[te])])
  scores.append(np.mean(errors))
 return ScaledRidge([1.,10.,100.,1000.][np.argmin(scores)]).fit(x,y)

def extract():
 w=open_windows(json.load(open('configs/pacld/default.json')),'train');d=w.dataset
 idx=w.index; audit={'raw_hall_windows':len(idx),'unique_decisions':len(idx.drop_duplicates(['flight_id','decision_us'])),'flights':idx.flight_id.nunique()}
 rows=[];coverage=[]
 for fid,gg in idx.groupby('flight_id',sort=True):
  d._load(fid);meta=d.provenance[fid]; assert all(meta['parameters'].get(k)==v for k,v in d.contract['required_mapping_parameters'].items())
  status=pd.read_parquet(d.root/'native_vehicle_status_train.parquet',filters=[('flight_id','==',fid)]).sort_values('timestamp_us')
  g=gg[(gg.horizon_us==500000)&~gg.mode_change].drop_duplicates('decision_us').sort_values('decision_us');last=-1e20;n0=len(rows)
  def before(name,t,age,fields):
   tab=d.tables[name];av=tab.get('availability_us',tab.timestamp_us).to_numpy();j=np.searchsorted(av,t,side='right')-1
   if j<0:raise ValueError('missing')
   r=tab.iloc[j];mt=r.timestamp_sample_us if r.timestamp_sample_us>0 else r.timestamp_us
   if name in ['wing_phase','flap_frequency']:mt=r.timestamp_us
   if t-mt>age or not r.in_reviewed_flight:raise ValueError('stale')
   if name=='wing_phase' and not r.phase_usable:raise ValueError('phase')
   if name=='vehicle_local_position' and not (r.v_xy_valid and r.v_z_valid):raise ValueError('velocity')
   return r[fields].to_numpy(float)
  for r in g.itertuples():
   t=int(r.decision_us)
   if t-last<1000000:continue
   try:
    v=before('vehicle_local_position',t,160000,['vx','vy','vz']);q=before('vehicle_attitude',t,100000,[f'q[{i}]' for i in range(4)]);rate=before('vehicle_angular_velocity',t,50000,[f'xyz[{i}]' for i in range(3)])
    rot=Rotation.from_quat(q[[1,2,3,0]]);freq=before('flap_frequency',t,120000,['frequency_hz']);phase=before('wing_phase',t,70000,['phase_rad'])[0]
    tape=d._controls(t-500000,t+500000,t,int(r.support_start_us));ts=np.array(tape.times) if hasattr(tape,'times') else np.array(tape.times_s)
    a=tape.values.numpy(); base=a[np.searchsorted(ts,0,side='left')-1]
    def avg(lo,hi):
     edges=np.r_[lo,ts[(ts>lo)&(ts<hi)],hi];j=np.searchsorted(ts,edges[:-1],side='right')-1
     return np.sum(a[j]*np.diff(edges)[:,None],axis=0)/(hi-lo)
    hist=np.r_[transform(base),transform(avg(-.5,-.25)),transform(avg(-.25,0))]
    X=np.r_[rot.inv().apply(v),rate,rot.inv().apply([0,0,1]),freq,np.sin(phase),np.cos(phase),hist]
    st=status[status.timestamp_us<=t]
    if st.empty:raise ValueError('mode')
    mode=int(st.iloc[-1].nav_state)
    fut=status[(status.timestamp_us>t)&(status.timestamp_us<=t+500000)]
    if (fut.nav_state!=mode).any():raise ValueError('mode change')
    tab=d.tables['vehicle_angular_velocity'];mt=tab.timestamp_sample_us.where(tab.timestamp_sample_us>0,tab.timestamp_us).to_numpy();ys=[];offset=[]
    for h in [200000,500000]:
     j=np.argmin(abs(mt-(t+h)));err=int(mt[j]-t-h)
     if abs(err)>d.contract['groups'][r.sampling_group]['horizon_tolerance_us'] or not tab.iloc[j].in_reviewed_flight:raise ValueError('target')
     ys.extend(tab.iloc[j][[f'xyz[{i}]' for i in range(3)]].to_numpy(float)-rate);offset.append(err)
    U=np.r_[transform(avg(0,.05)-base),transform(avg(0,.1)-base)]
    if not np.isfinite(np.r_[X,U,ys]).all():raise ValueError('finite')
    rows.append(dict(flight=fid,date=meta['flight_date'],mode=mode,t=t,X=X.tolist(),U=U.tolist(),Y=ys,offset=offset,saturation=(abs(a)>=.999).mean(axis=0).tolist()));last=t
   except ValueError:continue
  coverage.append(dict(flight=fid,date=meta['flight_date'],raw=len(gg),unique=gg.decision_us.nunique(),eligible=len(g),spaced=len(rows)-n0));print(len(coverage),len(rows),flush=True)
 pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False);(OUT/'audit.json').write_text(json.dumps(audit,indent=2));pd.DataFrame(rows).to_pickle(OUT/'features.pkl')
def analyze():
 df=pd.read_pickle(OUT/'features.pkl');X=np.stack(df.X);U=np.stack(df.U);Y=np.stack(df.Y);groups=df.flight.to_numpy();
 from a2_raw_context import attach
 context=attach(df)
 X=np.c_[X,context,pd.get_dummies(df['mode']).to_numpy()]
 plain=np.c_[np.stack(df.X),pd.get_dummies(df['mode']).to_numpy()];plain0=np.zeros_like(Y);plain1=plain0.copy();pred=np.zeros_like(Y);p0=pred.copy();res=np.zeros_like(U);drop=np.zeros((len(df),4,6));folds=[]
 for fold,(tr,te) in enumerate(GroupKFold(5).split(X,Y,groups)):
  print('fold',fold,flush=True);plain0[te]=fit(plain[tr],Y[tr],groups[tr]).predict(plain[te]);plain1[te]=fit(np.c_[plain[tr],U[tr]],Y[tr],groups[tr]).predict(np.c_[plain[te],U[te]]);p0[te]=fit(X[tr],Y[tr],groups[tr]).predict(X[te]);pred[te]=fit(np.c_[X[tr],U[tr]],Y[tr],groups[tr]).predict(np.c_[X[te],U[te]]);res[te]=U[te]-fit(X[tr],U[tr],groups[tr]).predict(X[te])
  for c in range(4):
   keep=[j for j in range(8) if j%4!=c];drop[te,c]=fit(np.c_[X[tr],U[tr][:,keep]],Y[tr],groups[tr]).predict(np.c_[X[te],U[te][:,keep]])
  folds.append(dict(fold=fold,train_flights=len(set(groups[tr])),evaluation_flights=len(set(groups[te]))))
 records=[]
 for f in np.unique(groups):
  s=groups==f
  for j in range(6):
   records.append(dict(flight=f,date=df[s].date.iloc[0],horizon=[.2,.5][j//3],axis='pqr'[j%3],plain0=np.mean((Y[s,j]-plain0[s,j])**2),plain1=np.mean((Y[s,j]-plain1[s,j])**2),m0=np.mean((Y[s,j]-p0[s,j])**2),m1=np.mean((Y[s,j]-pred[s,j])**2),**{C[c]:np.mean((Y[s,j]-drop[s,c,j])**2) for c in range(4)}))
 pd.DataFrame(records).to_csv(OUT/'flight_errors.csv',index=False)
 ex=[]
 for width in range(2):
  z=res[:,width*4:width*4+4];cov=np.cov(z.T);ev=np.linalg.eigvalsh(cov);np.savetxt(OUT/f'residual_cov_{width}.csv',cov,delimiter=',');np.savetxt(OUT/f'residual_corr_{width}.csv',np.corrcoef(z.T),delimiter=',')
  for c in range(4):ex.append(dict(window_ms=[50,100][width],channel=C[c],raw_rms=np.sqrt(np.mean(U[:,width*4+c]**2)),residual_rms=np.sqrt(np.mean(z[:,c]**2)),effective_rank=np.exp(-np.sum((ev/ev.sum())*np.log(np.maximum(ev/ev.sum(),1e-30)))),condition=np.linalg.cond(cov)))
 pd.DataFrame(ex).to_csv(OUT/'excitation.csv',index=False)
 local=[]
 # Each anchor flight is excluded from scaling and local fitting; pairs cross flights, same mode.
 for f in np.unique(groups):
  tr=np.flatnonzero(groups!=f);te=np.flatnonzero(groups==f);sc=StandardScaler().fit(X[tr]);Z=sc.transform(X);su=StandardScaler().fit(U[tr,4:]);V=su.transform(U[:,4:]);
  for mode in df.iloc[te]['mode'].unique():
   a=te[df.iloc[te]['mode'].to_numpy()==mode];b=tr[df.iloc[tr]['mode'].to_numpy()==mode]
   if len(b)<30:continue
   dist,nn=NearestNeighbors(n_neighbors=min(24,len(b))).fit(Z[b]).kneighbors(Z[a]);ii=np.repeat(a,nn.shape[1]);jj=b[nn.ravel()];ds=dist.ravel();DX=Z[jj]-Z[ii];DU=V[jj]-V[ii];DY=Y[jj]-Y[ii];bw=max(np.median(ds),1e-6)
   for scale in [.5,1.,2.]:
    weights=np.exp(-.5*(ds/(bw*scale))**2);design=np.c_[DX,DU];model=Ridge(alpha=10,fit_intercept=False).fit(design,DY,sample_weight=weights);gain=model.coef_[:,-4:]/su.scale_[None,:];ur=DU-Ridge(10,False).fit(DX,DU,sample_weight=weights).predict(DX);cov=(ur*np.sqrt(weights[:,None])).T@(ur*np.sqrt(weights[:,None]));cond=np.linalg.cond(cov)
    for j in range(6):
     for c in range(4):local.append(dict(flight=f,mode=int(mode),width=scale,horizon=[.2,.5][j//3],axis='pqr'[j%3],channel=C[c],G=gain[j,c],condition=cond,residual_control_rms=float(np.sqrt(np.average(ur[:,c]**2,weights=weights))),pairs=len(ii)))
 pd.DataFrame(local).to_csv(OUT/'local_g.csv',index=False)
 np.savez_compressed(OUT/'predictions.npz',Y=Y,M0=p0,M1=pred,U=U,residual=res,flight=groups.astype(str));(OUT/'folds.json').write_text(json.dumps(folds,indent=2))
if __name__=='__main__':
 import sys
 if '--analyze' not in sys.argv:extract()
 analyze()
