"""Outcome-blind CPU audit. Only explicit train parquet columns and NPZ clock/mask."""
from pathlib import Path
import hashlib,json,sys,importlib.util,time
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from weighting import compute_weights,ess
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
old=module(ROOT/'experiments/control_identifiability_audit_v1/audit.py','old')
def load():
 cfg=json.loads((HERE/'config.json').read_text());data=Path(cfg['data_root']);cache=Path(cfg['cache_root'])
 manifest=json.loads((data/'dataset_manifest.json').read_text());receipt={}
 assert sha(data/'dataset_manifest.json')=='8afb8623bf7174e4da5b2044f644331b8b251433fed3576f6dcdab470935bcb5'
 names=['flights.csv','flight_provenance.json','split_manifest.json','samples_train.parquet','windows_train.parquet','native_actuator_motors_train.parquet','native_actuator_servos_train.parquet']
 for n in names:
  h=sha(data/n);assert h==manifest['artifact_sha256'][n],n;receipt[str(data/n)]=h
 wp=ROOT/'experiments/velocity_rate_gru64_horizon_training_v1/train_windows.parquet'
 assert sha(wp)=='2b51e72201ff6f99c1be03e43951a6e75dc8a419a2716c996cd67a7377225862'
 cp=cache/'train.npz';assert sha(cp)=='5702d548d4bc8486ee2dca95ced79b1f58d0c78776a3866a161cee99931ae451'
 receipt[str(wp)]=sha(wp);receipt[str(cp)]=sha(cp)
 w=pd.read_parquet(wp);assert len(w)==107920 and w.flight_id.nunique()==94 and w.split.eq('train').all()
 with np.load(cp) as a:mask=a['future_mask'].astype(bool);dt=a['dt']
 cols=list(dict.fromkeys(old.META+old.V+old.Q+old.W+old.U))
 assert not any(x.startswith('target') for x in cols)
 s=pd.read_parquet(data/'samples_train.parquet',columns=cols);s=s[s.flight_id.isin(w.flight_id)]
 f=pd.read_csv(data/'flights.csv');f=f[f.flight_id.isin(w.flight_id)].set_index('flight_id');assert f.split.eq('train').all()
 prov={x['flight_id']:x for x in json.loads((data/'flight_provenance.json').read_text())}
 motors=pd.read_parquet(data/'native_actuator_motors_train.parquet');servos=pd.read_parquet(data/'native_actuator_servos_train.parquet')
 mg=dict(tuple(motors.groupby('flight_id')));sg=dict(tuple(servos.groupby('flight_id')))
 groups={k:g.reset_index(drop=True) for k,g in s.groupby('flight_id')}
 tapes={fid:old.tape(mg[fid],sg[fid]) for fid in groups}
 registered=pd.read_parquet(data/'windows_train.parquet');r=registered[(registered.history_us==500000)&(registered.horizon_us==500000)].set_index(['flight_id','origin_sample'])
 keys=pd.MultiIndex.from_frame(w[['flight_id','origin_sample']]);assert keys.is_unique and keys.isin(r.index).all()
 rr=r.loc[keys]
 for col in ['start_sample','initial_measurement_us','decision_us','segment_id']:
  assert np.array_equal(w[col],rr[col]),col
 hp=ROOT/'experiments/structured_control_effectiveness_v1/results';protocol=json.loads((hp/'holdout_protocol.json').read_text());indices=dict(np.load(hp/'holdout_window_indices.npz'))
 for n in ['holdout_protocol.json','holdout_window_indices.npz','holdout_pair_assignments.csv']:receipt[str(hp/n)]=sha(hp/n)
 sp=ROOT/'experiments/local_control_response_scope_v1';controllers=json.loads((sp/'results/train/controller_headers.json').read_text());cid={x['flight_id']:x['controller_id'] for x in controllers}
 for p in [sp/'config.json',sp/'results/train/controller_headers.json',sp/'results/train/candidate_index.csv.gz']:receipt[str(p)]=sha(p)
 frozen=pd.read_csv(sp/'results/train/candidate_index.csv.gz');scope_cfg=json.loads((sp/'config.json').read_text())
 dump(OUT/'source_receipt.json',dict(sha256=receipt,partitions_read=['train'],npz_members_read=['dt','future_mask'],sample_columns_read=cols,test_opened=False,future_targets_read=False,training_runs=0,gpu_used=False))
 return cfg,w,mask,dt,groups,tapes,f,prov,cid,frozen,protocol,indices,scope_cfg

def histories(w,groups,f,prov,cid,cfg):
 m=w[['window_id','flight_id','origin_sample','start_sample','end_sample','initial_measurement_us','decision_us','future_count','segment_id']].copy();m['registered_index']=np.arange(len(w))
 for fid,g in m.groupby('flight_id',sort=False):
  s=groups[fid];o=g.origin_sample.to_numpy();a=g.start_sample.to_numpy();t=s.state_measurement_us.to_numpy();pub=s.state_publication_us.to_numpy()
  assert np.array_equal(t[o],g.initial_measurement_us) and np.array_equal(pub[o],g.decision_us)
  ix=np.searchsorted(t,t[o,None]+np.linspace(-500000,0,26).astype(np.int64),side='right')-1
  assert (ix>=a[:,None]).all() and (ix<=o[:,None]).all() and (pub[ix]<=pub[o,None]).all()
  vals=s[old.V+old.Q+old.W+['flap_frequency_hz']+old.U].to_numpy()[ix];vals[:,:,3:7]/=np.linalg.norm(vals[:,:,3:7],axis=2,keepdims=True)
  # Match historical scope's float32 history storage before feature extraction.
  vals=vals.astype(np.float32);rot=Rotation.from_quat(vals[:,:,3:7].reshape(-1,4)[:,[1,2,3,0]]);eu=rot.as_euler('xyz',degrees=True).reshape(-1,26,3)
  vb=rot.inv().apply(vals[:,:,:3].reshape(-1,3)).reshape(-1,26,3);speed=np.linalg.norm(vals[:,:,:3],axis=2);rate=np.linalg.norm(vals[:,:,7:10],axis=2)
  near=(vals[:,:,11]>=.98)|(vals[:,:,11]<=.02)|(abs(vals[:,:,12:15])>=.98).any(2)
  feats=dict(roll_max=abs(eu[:,:,0]).max(1),pitch_max=abs(eu[:,:,1]).max(1),rate_rms=np.sqrt((rate**2).mean(1)),rate_initial=rate[:,-1],speed_min=speed.min(1),speed_max=speed.max(1),body_forward_min=vb[:,:,0].min(1),vertical_max=abs(vals[:,:,2]).max(1),frequency_min=vals[:,:,10].min(1),frequency_max=vals[:,:,10].max(1),history_near_limit_fraction=near.mean(1),history_speed=speed.mean(1))
  good=(s.valid_frequency&s.valid_status&~s.failsafe.astype(bool)).to_numpy();bad=np.r_[0,np.cumsum(~good)];mode=s.nav_state.to_numpy();changes=np.r_[0,np.cumsum(np.r_[False,np.diff(mode)!=0])]
  past=(bad[o+1]==bad[a])&(changes[o+1]==changes[a+1]);b=cfg['common_history_bounds'];c=cfg['regimes']['AUTO_stable']
  stable=past&(mode[o]==c['mode'])&(feats['speed_min']>=b['ground_speed_min'])&(feats['speed_max']<=b['ground_speed_max'])&(feats['body_forward_min']>=b['body_forward_min'])&(feats['frequency_min']>=b['frequency_min'])&(feats['frequency_max']<=b['frequency_max'])&(feats['history_near_limit_fraction']<=b['near_limit_fraction_max'])&(feats['roll_max']<=c['roll_max_deg'])&(feats['pitch_max']<=c['pitch_max_deg'])&(feats['rate_rms']<=c['rate_rms_max'])&(feats['rate_initial']<=c['rate_rms_max'])&(feats['vertical_max']<=c['vertical_max'])
  for k,v in feats.items():m.loc[g.index,k]=v
  m.loc[g.index,'auto_stable']=stable;m.loc[g.index,'past_valid']=past
  for k,v in dict(log_sha=f.loc[fid,'sha256'],date=prov[fid]['flight_date'],firmware=prov[fid]['firmware'],controller_id=cid.get(fid,'unknown')).items():m.loc[g.index,k]=v
  m.loc[g.index,'historical_d0']=(s[old.U[2]].to_numpy()[o]-s[old.U[1]].to_numpy()[o])/2
 return m

def cut(native,start,end,segment_end):
 t,u,age=native;i=np.searchsorted(t,start,side='right')-1;j=np.searchsorted(t,end,side='left')
 if i<0:return np.array([0,max(1,end-start)]),np.full((1,4),np.nan),0
 tt=np.r_[start,t[i+1:j],end];uu=u[i:j];expiry=t[i:j]+160000-age[i:j]
 # Expiry is earliest motor/servo freshness expiry in each held interval.
 bad=np.flatnonzero(expiry<tt[1:]);support=min(end,segment_end)
 if len(bad):support=min(support,int(max(tt[bad[0]],expiry[bad[0]])))
 return tt-start,uu,max(0,support-start)

def calculate(m,mask,dt,groups,tapes,indices):
 n=len(indices);k=mask.shape[1];a=dict(w0=np.ones(n),w1=np.ones(n),w2=np.ones((n,k)),future_mask=mask[indices],output_us=np.zeros((n,k),np.int64),exposure_us=np.zeros((n,k)),w1_supported=np.zeros(n,bool),w2_supported=np.zeros((n,k),bool),registered_index=np.asarray(indices),support_until_us=np.zeros(n,np.int64),delta_d=np.full(n,np.nan),mean_change=np.full((n,4),np.nan),rms_change=np.full((n,4),np.nan),duration300_us=np.full(n,np.nan))
 segends={fid:s.groupby('segment_id').state_publication_us.max().to_dict() for fid,s in groups.items()}
 for j,idx in enumerate(indices):
  row=m.iloc[idx];s=groups[row.flight_id];o=int(row.origin_sample);end=int(row.end_sample);out=s.state_measurement_us.to_numpy()[o+1:end+1];count=int(mask[idx].sum());assert len(out)==count
  assert np.array_equal(mask[idx],np.arange(k)<count)
  elapsed=out-int(row.initial_measurement_us);assert np.max(abs(elapsed-np.cumsum(dt[idx,:count],dtype=np.float64)*1e6))<1
  a['output_us'][j,:count]=out-int(row.decision_us)
  stop=max(int(out[-1]),int(row.decision_us)+300000);t,u,support=cut(tapes[row.flight_id],int(row.decision_us),stop,int(segends[row.flight_id][row.segment_id]))
  result=compute_weights(t,u,a['output_us'][j],mask[idx],auto_stable=bool(row.auto_stable),supported_until=support)
  a['support_until_us'][j]=support
  for key in result:
   if key in a:a[key][j]=result[key]
  if j%20000==0:print('computed',j,'/',n,flush=True)
 a['window_id']=m.iloc[indices].window_id.to_numpy().astype(str);a['flight_id']=m.iloc[indices].flight_id.to_numpy().astype(str)
 return a

def audit(name,m,a):
 folder=OUT/name;folder.mkdir(exist_ok=True);np.savez_compressed(folder/'weights.npz',**a)
 n=len(m);mask=a['future_mask'];count=mask.sum(1);w1=a['w1'];w2=a['w2'];tw=mask*w2;te=ess(tw,axis=1)/count
 m=m.copy();m['w1']=w1;m['w2_time_ess_ratio']=te;m['delta_d']=a['delta_d'];m['duration_ms']=a['duration300_us']/1000;m['w1_supported']=a['w1_supported'];m['w2_supported_steps']=a['w2_supported'].sum(1);m['w2_enhanced_steps']=(mask&(w2>1)).sum(1)
 m['reason_w1']=np.select([~m.auto_stable.astype(bool),~m.w1_supported,abs(m.delta_d)<=.02],['history_not_AUTO_stable','control_support_missing','change_below_threshold'],default='enhanced')
 m['reason_w2']=np.select([~m.auto_stable.astype(bool),m.w2_supported_steps==0,m.w2_enhanced_steps==0],['history_not_AUTO_stable','control_support_missing','no_supported_exposure'],default='enhanced')
 m.to_csv(folder/'window_audit.csv.gz',index=False)
 base=np.ones(n)/n;p1=w1/w1.sum();coeff0=mask/count[:,None]/n;coeff1=mask/count[:,None]*p1[:,None];coeff2=tw/tw.sum(1)[:,None]/n
 dist=[]
 for scheme,x in [('W0_window',np.ones(n)),('W1_window',w1),('W2_valid_step',w2[mask])]:
  for label,sel in [('1',x==1),('(1,1.25]',(x>1)&(x<=1.25)),('(1.25,1.5]',(x>1.25)&(x<=1.5)),('(1.5,2]',x>1.5)]:dist.append(dict(scheme=scheme,bin=label,count=int(sel.sum()),fraction=float(sel.mean())))
 pd.DataFrame(dist).to_csv(folder/'weight_distribution.csv',index=False)
 cover=[];flightmax=0
 for group in ['flight_id','log_sha','date','firmware','controller_id','auto_stable']:
  for value,ix in m.groupby(group,dropna=False).indices.items():
   ix=np.asarray(ix);r=dict(group=group,value=str(value),windows=len(ix),W0_share=float(base[ix].sum()),W1_share=float(p1[ix].sum()),W2_share=float(base[ix].sum()),W1_added=float((w1[ix]-1).sum()),W2_added=float(((w2[ix]-1)*mask[ix]).sum()),positive_enhanced=int(((w1[ix]>1)&(a['delta_d'][ix]>0)).sum()),negative_enhanced=int(((w1[ix]>1)&(a['delta_d'][ix]<0)).sum()))
   r['W1_share_ratio']=r['W1_share']/r['W0_share'];cover.append(r)
   if group=='flight_id':flightmax=max(flightmax,r['W1_share_ratio'])
 pd.DataFrame(cover).to_csv(folder/'group_contributions.csv',index=False)
 for key,values,bins in [('speed',m.history_speed,[0,4,6,8,12,np.inf]),('amplitude',abs(m.delta_d),[0,.02,.05,.10,np.inf]),('duration_ms',m.duration_ms,[-1e-9,100,250,np.inf])]:
  labels=pd.cut(values,bins,include_lowest=True,right=False).astype(str)
  for value in sorted(set(labels)):
   ix=np.flatnonzero(labels==value);cover.append(dict(group=key,value=value,windows=len(ix),W0_share=float(base[ix].sum()),W1_share=float(p1[ix].sum()),W2_share=float(base[ix].sum()),W1_added=float((w1[ix]-1).sum()),W2_added=float(((w2[ix]-1)*mask[ix]).sum())))
 pd.DataFrame(cover).to_csv(folder/'coverage.csv',index=False)
 time=[]
 for label,coeff in [('W0',coeff0),('W1',coeff1),('W2',coeff2)]:
  for k in range(mask.shape[1]):time.append(dict(scheme=label,step=k+1,contribution=float(coeff[:,k].sum()),supported_windows=int(mask[:,k].sum())))
 pd.DataFrame(time).to_csv(folder/'step_contribution.csv',index=False)
 physical=[]
 for label,coeff in [('W0',coeff0),('W1',coeff1),('W2',coeff2)]:
  for lo,hi in [(0,100000),(100000,200000),(200000,300000),(300000,500000),(500000,800000),(800000,1100000)]:physical.append(dict(scheme=label,start_us=lo,end_us=hi,contribution=float(coeff[(a['output_us']>=lo)&(a['output_us']<hi)].sum())))
 pd.DataFrame(physical).to_csv(folder/'time_contribution.csv',index=False)
 corr=[];valid=np.isfinite(a['mean_change']).all(1)
 for label,ww in [('W0',np.ones(n)),('W1',w1)]:
  x=a['mean_change'][valid];weights=ww[valid];mu=np.average(x,axis=0,weights=weights);cov=((x-mu)*weights[:,None]).T@(x-mu)/weights.sum();c=cov/np.sqrt(np.outer(np.diag(cov),np.diag(cov)))
  for i,axis in enumerate(['motor','common','differential','rudder']):
   for j,other in enumerate(['motor','common','differential','rudder']):corr.append(dict(scheme=label,axis=axis,other=other,correlation=float(c[i,j])))
 pd.DataFrame(corr).to_csv(folder/'control_coupling.csv',index=False)
 summary=dict(windows=n,flights=int(m.flight_id.nunique()),auto_stable=int(m.auto_stable.sum()),w1_enhanced=int((w1>1).sum()),w2_enhanced_windows=int((m.w2_enhanced_steps>0).sum()),w1_ess=float(ess(w1)),w1_ess_ratio=float(ess(w1)/n),w2_window_ess=n,w2_time_ess_ratio_min=float(te.min()),w2_time_ess_ratio_median=float(np.median(te)),w1_min=float(w1.min()),w1_max=float(w1.max()),w1_mean=float(w1.mean()),w2_max=float(w2.max()),w2_valid_step_mean=float(w2[mask].mean()),max_flight_share_ratio_W1=flightmax,max_flight_share_ratio_W2=1.,positive_enhanced_flights=int(m.loc[(w1>1)&(m.delta_d>0),'flight_id'].nunique()),negative_enhanced_flights=int(m.loc[(w1>1)&(m.delta_d<0),'flight_id'].nunique()),positive_enhanced_windows=int(((w1>1)&(m.delta_d>0)).sum()),negative_enhanced_windows=int(((w1>1)&(m.delta_d<0)).sum()))
 fg=pd.DataFrame(cover);fg=fg[fg.group=='flight_id'];added=fg.W1_added.sum();summary['top1_added_share_W1']=float(fg.W1_added.max()/added);summary['top5_added_share_W1']=float(fg.W1_added.nlargest(5).sum()/added)
 summary['W1_numeric_gate_pass']=bool(ess(w1)/n>=.85 and w1.max()<=2 and flightmax<=1.5);summary['W2_numeric_gate_pass']=bool(te.min()>=.85 and w2.max()<=2)
 dump(folder/'summary.json',summary);return summary

def main():
 started=time.time();cfg,w,mask,dt,groups,tapes,f,prov,cid,frozen,protocol,indices,scfg=load();m=histories(w,groups,f,prov,cid,scfg)
 match=m.merge(frozen[['flight_id','origin_sample','AUTO_stable']],on=['flight_id','origin_sample'],how='left',validate='one_to_one');known=match.AUTO_stable.notna();assert np.array_equal(match.loc[known,'auto_stable'].astype(bool),match.loc[known,'AUTO_stable'].astype(bool))
 dump(OUT/'identity_check.json',dict(exact_H500_registered_matches=len(w),existing_scope_candidate_matches=int(known.sum()),scope_flag_mismatches=0,history_only_stable_not_in_candidate=int((~known&match.auto_stable.astype(bool)).sum())))
 arrays=calculate(m,mask,dt,groups,tapes,np.arange(len(w)));summaries={'full_train':audit('full_train',m,arrays)}
 for fold,expected in zip(protocol['folds'],[87239,95627,87750]):
  k=fold['fold'];ix=indices[f'fold{k}_train_indices'];assert len(ix)==expected;fit=set(fold['train_flights']);held=set(fold['held_flights']);val=set(fold['validation_flights']);assert not fit&held and not fit&val and not {x.split(':')[0] for x in fit}&{x.split(':')[0] for x in held|val};assert set(m.iloc[ix].flight_id)==fit
  a=calculate(m,mask,dt,groups,tapes,ix)
  for key in ['w1','w2','output_us','w1_supported','w2_supported']:assert np.array_equal(a[key],arrays[key][ix]),key
  summaries[f'fold{k}']=audit(f'fold{k}',m.iloc[ix].reset_index(drop=True),a)
  dump(OUT/f'fold{k}'/'isolation.json',dict(train_flights=sorted(fit),held_flights=sorted(held),validation_flights=sorted(val),windows=len(ix),fitted_statistics=False,thresholds_fixed_before_analysis=True,full_array_equality=True,whole_log_disjoint=True))
 dump(OUT/'summary.json',summaries);dump(OUT/'execution.json',dict(seconds=time.time()-started,training_runs=0,optimizer_steps=0,gpu_used=False,partitions=['train'],validation_outcomes_used=False,test_read=False));print(json.dumps(summaries,indent=2),flush=True)
if __name__=='__main__':main()
