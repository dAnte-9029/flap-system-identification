"""Independent antiderivative integration and direct group reductions."""
from pathlib import Path
import hashlib,json,os
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'local_control_response_rollout_v1';OUT=Path(os.environ.get('LOCAL_DIAG_OUTPUT',str(HERE/'results')))

def sample(t,y,at):
 at=np.asarray(at);v=np.column_stack([np.interp(at,t,y[:,j]) for j in range(3)])
 v[(at<t[0]-1e-10)|(at>t[-1]+1e-10)]=np.nan
 return v

def primitive(t,y,at):
 # Closed-form antiderivative within each linear segment, independent of trapezoid-at-query implementation.
 dt=np.diff(t);slope=np.diff(y,axis=0)/dt[:,None];cum=np.vstack([np.zeros(3),np.cumsum(dt[:,None]*(y[:-1]+y[1:])/2,axis=0)])
 j=np.clip(np.searchsorted(t,at,side='right')-1,0,len(t)-2);h=np.asarray(at)-t[j]
 return cum[j]+y[j]*h[:,None]+slope[j]*h[:,None]**2/2

def mean(t,y,at,width):
 lo=at-width/2;hi=at+width/2;out=(primitive(t,y,hi)-primitive(t,y,lo))/width
 out[(lo<t[0]-1e-10)|(hi>t[-1]+1e-10)]=np.nan;return out

def components(p):
 adj={}
 for row in p.itertuples():adj.setdefault(row.flight_a,set()).add(row.flight_b);adj.setdefault(row.flight_b,set()).add(row.flight_a)
 remaining=set(adj);out=[]
 while remaining:
  todo=[min(remaining)];seen=set()
  while todo:
   v=todo.pop()
   if v in seen:continue
   seen.add(v);todo+=list(adj[v]-seen)
  remaining-=seen;out.append(set(p.loc[p.flight_a.isin(seen),'pair_uid']))
 return out

def main():
 z=dict(np.load(OLD/'results/native_rollouts.npz'));nat=dict(np.load(OUT/'native_diagnostic_inputs.npz'));pairs=pd.read_csv(OLD/'results/pairs.csv').set_index('pair_uid',drop=False)
 d=pd.read_csv(OUT/'responses.csv.gz');maxerr=0.;checks=0
 def tape(i,kind,model,seed):
  n=int(z['future_count'][i]);t=z['native_time_us'][i,:n+1]*1e-6
  if kind=='pred':y=np.vstack([z['initial'][i,10:],z[f'{model}_{seed}'][i,:n,10:]]).astype(float);return t,y-y[0]
  if kind=='A':y=np.vstack([z['truth_initial'][i,10:],z['truth'][i,:n,10:]]);return t,y-y[0]
  if kind=='B_sparse':tt=nat[f'sparse_t_{i}'];yy=nat[f'sparse_y_{i}'];return tt,yy-sample(tt,yy,[0])[0]
  tt=nat[f'rate_t_{i}'];yy=nat[f'rate_y_{i}'];newt=np.r_[0,tt[(tt>0)&(tt<min(.5,t[-1]))],min(.5,t[-1])]
  return newt,sample(tt,yy,newt)-sample(tt,yy,[0])[0]
 for (uid,model,seed,width,method),g in d.groupby(['pair_uid','model','seed','width_s','method'],sort=False):
  pair=pairs.loc[uid];at=g.loc[g.variable=='p','time_s'].to_numpy();a,b=int(pair.a),int(pair.b)
  source='A' if method in ['A','C','C_cycle','fast_A'] else 'B_sparse' if method=='B_sparse' else 'B'
  r1=tape(a,source,model,seed);r2=tape(b,source,model,seed);p1=tape(a,'pred',model,seed);p2=tape(b,'pred',model,seed)
  rawr=sample(*r1,at)-sample(*r2,at);rawp=sample(*p1,at)-sample(*p2,at)
  if method in ['A','B','B_sparse']:real,pred=rawr,rawp
  else:
   real=mean(*r1,at,width)-mean(*r2,at,width);pred=mean(*p1,at,width)-mean(*p2,at,width)
   if method.endswith('_cycle'):real-=real[0];pred-=pred[0]
   if method.startswith('fast_'):real=rawr-real;pred=rawp-pred
  for col,expected in [('real',real),('pred',pred),('error',pred-real)]:
   saved=g[col].to_numpy().reshape(3,-1).T;np.testing.assert_allclose(saved,expected,atol=2e-12,rtol=1e-9,equal_nan=True,err_msg=str((uid,model,seed,width,method,col)))
   maxerr=max(maxerr,float(np.nanmax(abs(saved-expected))));checks+=int(np.isfinite(saved).sum())
 print('Independent curves passed',checks,'max',maxerr,flush=True)
 summary=pd.read_csv(OUT/'summary.csv');points=pd.read_csv(OUT/'points.csv.gz');members=pd.read_csv(OLD/'results/cohorts.csv')
 groups={k:g for k,g in points.groupby(['domain','width_s','method','point','variable','model'],sort=False)}
 cohorts={(sp,co):set(g.pair_uid) for (sp,co),g in members.groupby(['split','cohort'])};agmax=0.
 for row in summary.itertuples():
  g=groups[(row.domain,row.width_s,row.method,row.point,row.variable,row.model)];g=g[g.pair_uid.isin(cohorts[row.split,row.cohort])]
  if row.seed!='pooled':g=g[g.seed==int(row.seed)]
  g=g[np.isfinite(g.real)&np.isfinite(g.pred)];pp=pairs.loc[g.pair_uid.unique()]
  er=g.pred-g.real;active=g.real.abs()>.1;correct=(np.sign(g.real)==np.sign(g.pred))&(g.pred.abs()>1e-8)&active
  vals=pd.DataFrame(dict(pair_uid=g.pair_uid,real2=g.real**2,pred2=g.pred**2,cross=g.pred*g.real,error2=er**2,bias=er,mae=abs(er),active=active,correct=correct)).groupby('pair_uid').mean()
  if row.weighting=='pair':avg=vals.mean()
  else:
   sets=([set(pp.loc[(pp.flight_a==f)|(pp.flight_b==f),'pair_uid']) for f in sorted(set(pp.flight_a)|set(pp.flight_b))] if row.weighting=='flight' else components(pp))
   index={uid:i for i,uid in enumerate(vals.index)};matrix=vals.to_numpy()
   avg=pd.Series(np.mean([matrix[[index[u] for u in sorted(ids)]].mean(axis=0) for ids in sets],axis=0),index=vals.columns)
  expected=dict(rmse=np.sqrt(avg.error2),bias=avg.bias,mae=avg.mae,scale_ratio=np.sqrt(avg.pred2/avg.real2) if avg.real2>1e-18 else np.nan,
    signed_gain=avg.cross/avg.real2 if avg.real2>1e-18 else np.nan,near_zero_fraction=1-avg.active,direction_accuracy=avg.correct/avg.active if avg.active>0 else np.nan)
  assert len(pp)==row.pairs
  for key,val in expected.items():
   got=getattr(row,key);assert np.isclose(val,got,atol=2e-12,rtol=1e-8,equal_nan=True),(row.Index,key,val,got)
   if np.isfinite(val):agmax=max(agmax,abs(val-got))
 result=dict(passed=True,response_scalar_checks=checks,response_max_abs_difference=maxerr,summary_rows=len(summary),aggregate_max_abs_difference=agmax,model_inference=False,test_read=False,verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (OUT/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
