from analyze import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':9})
FIG=HERE/'figures'
def save(fig,name):
 fig.tight_layout();fig.savefig(FIG/name,dpi=160);plt.close(fig)
def main():
 m=pd.read_csv(RES/'windows.csv.gz');f=pd.read_parquet(RES/'causal_features.parquet');a=dict(np.load(RES/'diagnostic_arrays.npz'));y=a['target'];pred=np.load(RES/'crossfit_predictions.npz')['SU_reference500'];res=y-pred
 # Per configuration/date and predeclared physical bins; summary is descriptive, not a rule search.
 m['speed_bin']=pd.cut(m.history_speed,[0,6,8,12,np.inf]).astype(str)
 m['change_bin']=pd.cut(abs(y[:,6]),[0,.02,.05,.1,np.inf],include_lowest=True).astype(str)
 m['dwell_bin']=pd.cut(m.differential_longest_us/1000,[-1,100,250,500]).astype(str)
 grouped=[]
 for split in CFG['partitions']:
  for group in ['date','firmware','controller_id','speed_bin','change_bin','dwell_bin']:
   for label,g in m[(m.split==split)&m.nonoverlap].groupby(group):
    ids=g.index.to_numpy();w=weights(g)
    grouped.append(dict(split=split,grouping=group,label=label,windows=len(g),flights=g.flight_id.nunique(),**metrics(y[ids,6],pred[ids,6],w),frequency_median=g.history_frequency.median(),rate_history_median=g.history_rate.median(),origin_rate_gap_rms=np.sqrt(w@(g.rate_initial_gap**2))))
 pd.DataFrame(grouped).to_csv(RES/'support_strata.csv',index=False)
 chain=[];missions=[];res_sign=[]
 for fid,g in m[m.nonoverlap].groupby('flight_id'):
  ii=g.index.to_numpy();d=a['control'][ii,25,2];rate=f.loc[ii,'S0_p'].to_numpy();sp=f.loc[ii,'R0_vehicle_rates_setpoint_roll'].to_numpy()
  for name,x in [('roll_rate',rate),('roll_rate_error',sp-rate),('roll_rate_sp',sp),('torque',a['diagnostic_reference'][ii,25,6]),('p_term',f.loc[ii,'I0_rate_ctrl_terms_p_term[0]'].to_numpy()),('i_term',f.loc[ii,'I0_rate_ctrl_terms_i_term[0]'].to_numpy()),('d_term',f.loc[ii,'I0_rate_ctrl_terms_d_term[0]'].to_numpy()),('ff_term',f.loc[ii,'I0_rate_ctrl_terms_ff_term[0]'].to_numpy())]:
   chain.append(dict(split=g.split.iloc[0],flight_id=fid,signal=name,**assoc(x,d)))
  mc=f.loc[ii,'mission_changed_500'].to_numpy();missions.append(dict(split=g.split.iloc[0],flight_id=fid,windows=len(ii),change_windows=int(np.nansum(mc)),finite=int(np.isfinite(mc).sum()),change_raw_rms=float(np.sqrt(np.mean(y[ii[mc==1],6]**2))) if (mc==1).any() else None,nochange_raw_rms=float(np.sqrt(np.mean(y[ii[mc==0],6]**2))) if (mc==0).any() else None))
 pd.DataFrame(chain).to_csv(RES/'chain_per_flight.csv',index=False);pd.DataFrame(missions).to_csv(RES/'mission_changes.csv',index=False)
 # Grouped association with historical GRU failures; no refitting or selecting pairs.
 p=pd.read_csv(RES/'pair_failure_links.csv');coh=pd.read_csv(RES/'frozen_cohorts.csv');main=set(coh[(coh.cohort=='main')&(coh.split=='train')].pair_uid)
 p=p[(p.role=='heldout')&p.pair_uid.isin(main)].copy();p['absolute_error']=abs(p.pred-p.real);p['wrong']=(p.pred*p.real<0)&(abs(p.real)>.1);p['active']=abs(p.real)>.1
 associations=[]
 for (model,method,seed),g in p.groupby(['model','method','seed']):
  for feature in ['history_rate_rms','historical_control_contrast','residual_difference_300','origin_rate_gap_a','origin_rate_gap_b']:
   xx=abs(g[feature].to_numpy())
   associations.append(dict(model=model,method=method,seed=seed,feature=feature,pairs=len(g),spearman_abs_error=float(spearmanr(xx,g.absolute_error).statistic),mean_feature_wrong=float(np.mean(xx[g.wrong])) if g.wrong.any() else None,mean_feature_correct=float(np.mean(xx[g.active&~g.wrong])) if (g.active&~g.wrong).any() else None))
 pd.DataFrame(associations).to_csv(RES/'failure_associations.csv',index=False)
 dep=[]
 for (model,method,seed,component),g in p.groupby(['model','method','seed','component_id']):
  dep.append(dict(model=model,method=method,seed=seed,component=component,pairs=len(g),active=int(g.active.sum()),wrong=int(g.wrong.sum()),absolute_error=g.absolute_error.mean(),residual_rms=np.sqrt(np.mean(g.residual_difference_300**2)),history_rate_rms=g.history_rate_rms.mean()))
 pd.DataFrame(dep).to_csv(RES/'component_failure_summary.csv',index=False)
 # Pair/flight/component-equal descriptive diagnostics using the original memberships.
 pairs=pd.read_csv(RES/'pair_information.csv');ps=[]
 for (split,cohort),cg in coh.groupby(['split','cohort']):
  q=pairs[(pairs.split==split)&pairs.pair_uid.isin(cg.pair_uid)&(pairs.variant=='SU_reference500')].copy()
  for weighting in ['pair','flight','component']:
   if weighting=='pair':w=np.ones(len(q))
   elif weighting=='flight':
    counts=pd.concat([q.flight_a,q.flight_b]).value_counts();w=np.array([1/counts[x]+1/counts[z] for x,z in zip(q.flight_a,q.flight_b)])
   else:
    # Only primary dependencies are frozen; derive the identical shared-flight graph for each other cohort.
    parent={}
    def root(x):
     parent.setdefault(x,x)
     while parent[x]!=x:x=parent[x]
     return x
    for x,z in zip(q.flight_a,q.flight_b):parent[root(x)]=root(z)
    cc=[root(x) for x in q.flight_a];cnt=pd.Series(cc).value_counts();w=np.array([1/cnt[x] for x in cc])
   w=w/w.sum();ps.append(dict(split=split,cohort=cohort,weighting=weighting,pairs=len(q),raw_rms=np.sqrt(w@(q.command_difference_300**2)),residual_rms=np.sqrt(w@(q.residual_difference_300**2)),near_zero_fraction=w@(abs(q.residual_difference_300)<.02)))
 pd.DataFrame(ps).to_csv(RES/'pair_dependence_summary.csv',index=False)
 # Current command at t is not a pure feedforward input. These correlations quantify that distinction.
 chain=pd.DataFrame(chain)
 cond=pd.read_csv(RES/'conditional_summary.csv');fig,ax=plt.subplots(figsize=(10,4))
 for sp,color in [('train','C0'),('validation','C1')]:
  g=cond[(cond.split==sp)&(cond.subset=='nonoverlap')&(cond.coordinate=='differential')&(cond.horizon_ms==300)&~cond.variant.str.contains('lambda')]
  ax.plot(g.variant,g.residual_rms_ratio,'o-',label=sp,color=color)
 ax.set(ylabel='OOF residual RMS / original change RMS',title='Future 300ms mean differential change; equal flight weights');ax.tick_params(axis='x',rotation=25);ax.legend();save(fig,'conditional_control.png')
 fig,axs=plt.subplots(1,2,figsize=(10,4));cc=pd.read_csv(RES/'coupling.csv')
 for ax,kind in zip(axs,['raw','conditional']):
  q=cc[(cc.split=='train')&(cc.kind==kind)].pivot(index='a',columns='b',values='correlation').reindex(index=COORDS,columns=COORDS);im=ax.imshow(q,vmin=-1,vmax=1,cmap='coolwarm');ax.set_xticks(range(4),COORDS,rotation=30);ax.set_yticks(range(4),COORDS);ax.set_title(kind)
  for i in range(4):
   for j in range(4):ax.text(j,i,f'{q.iloc[i,j]:.2f}',ha='center')
 save(fig,'coupling.png')
 fig,axs=plt.subplots(1,2,figsize=(10,4))
 for label in ['raw','conditional']:
  agram=np.load(RES/f'gram_train_{label}_6_standardized.npz')['gram'];axs[0].semilogy(np.arange(1,25),np.sqrt(np.linalg.eigvalsh(agram))[::-1],'o-',label=label)
  q=pd.read_csv(RES/'information_leave_flight_out.csv');q=q[(q.split=='train')&(q.kind==label)];axs[1].plot(np.arange(len(q)),q.condition,'.',label=label)
 axs[0].set(xlabel='Singular-value index',ylabel='Normalized singular value');axs[1].set(xlabel='Omitted flight',ylabel='Standardized condition number');axs[0].legend();axs[1].legend();save(fig,'information_spectrum.png')
 t=pd.read_csv(RES/'temporal_per_flight.csv');fig,axs=plt.subplots(1,3,figsize=(12,4))
 for ax,axis in zip(axs,['p','q','r']):
  for cmd in ['raw_future300','conditional_future300']:
   for side in ['past','future']:
    g=t[(t.split=='train')&(t.axis==axis)&(t.command==cmd)&(t.motion_period==side)];v=g.groupby('horizon_ms')['corr'].median();ax.plot(v.index*(1 if side=='future' else -1),v.values,'o-',label=cmd+' '+side)
  ax.axhline(0,color='k',lw=.5);ax.axvline(0,color='k',lw=.5);ax.set(title=axis,xlabel='Motion interval from/to t (ms)',ylabel='Median within-flight correlation')
 axs[2].legend(fontsize=6);save(fig,'temporal_negative_control.png')
 fig,ax=plt.subplots(figsize=(8,4))
 pp=pd.read_csv(RES/'native_psd.csv.gz');q=pp[(pp.split=='train')&(pp.coordinate=='differential')].groupby(['flight_id','hz']).power.mean().groupby('hz').mean();ax.semilogy(q.index,q);ax.axvspan(2.5,5.5,alpha=.15,color='orange');ax.set(xlim=(0,15),xlabel='Hz',ylabel='Mean flight PSD',title='Native differential, 4s Welch; shaded historical flapping range');save(fig,'native_spectrum.png')
 ids=['train:124','train:336','train:203','train:99','train:412','validation:44'];pp=pd.read_csv(HERE.parent/'structured_control_effectiveness_group_holdout_v1/results/pairs.csv');mi=dict(zip(m.window_id,m.index));fig,axs=plt.subplots(6,2,figsize=(11,16))
 for k,uid in enumerate(ids):
  rr=pp[pp.pair_uid==uid].iloc[0];ii=[mi[rr.window_a],mi[rr.window_b]]
  for j,label in zip(ii,['A','B']):
   axs[k,0].plot(a['grid_us']/1000,a['control'][j,:,2],label=label+' d')
   axs[k,1].plot(a['grid_us']/1000,a['motion'][j,:,0]-a['motion'][j,25,0],label=label+' delta p')
  for ax in axs[k]:ax.axvline(0,color='k',lw=.7);ax.set_title(uid);ax.legend()
 axs[-1,0].set(xlabel='ms from publication decision',ylabel='normalized command');axs[-1,1].set(xlabel='ms from publication decision',ylabel='rad/s; native-sample offline evaluation');save(fig,'fixed_cases.png')
 fig,axs=plt.subplots(1,2,figsize=(11,4));g=chain[chain.split=='train'];signals=g.signal.unique();axs[0].boxplot([g[g.signal==s]['corr'].dropna() for s in signals],tick_labels=signals);axs[0].tick_params(axis='x',rotation=45);axs[0].set(ylabel='Within-flight command correlation')
 s=pd.DataFrame(dep);g=s[(s.method=='D')&(s.model=='M1')].groupby('component').agg(residual=('residual_rms','mean'),error=('absolute_error','mean'),pairs=('pairs','first'));axs[1].scatter(g.residual,g.error,s=g.pairs*15)
 for idx,r in g.iterrows():axs[1].annotate(idx,(r.residual,r.error))
 axs[1].set(xlabel='Residual differential contrast RMS',ylabel='M1 D absolute response error',title='All 11 dependency components; descriptive');save(fig,'chain_and_failures.png')
 # Output compact fixed-window arrays needed to inspect published cases without loading raw logs.
 fixed=sorted(set(pp.window_a)|set(pp.window_b));ix=np.array([mi[x] for x in fixed]);np.savez_compressed(RES/'frozen_window_diagnostics.npz',window_id=np.array(fixed),target=y[ix],prediction=pred[ix],motion=a['motion'][ix],control=a['control'][ix],grid_us=a['grid_us'],reference=a['diagnostic_reference'][ix])
 write_json(RES/'report_numbers.json',dict(coverage=pd.read_csv(RES/'coverage_per_flight.csv').groupby('split').agg(flights=('flight_id','size'),windows=('windows','sum'),nonoverlap=('nonoverlap','sum'),seconds=('union_seconds','sum')).reset_index().to_dict('records'),conditional300=cond[(cond.subset=='nonoverlap')&(cond.coordinate=='differential')&(cond.horizon_ms==300)].to_dict('records'),temporal300=t[(t.split=='train')&(t.axis=='p')&(t.horizon_ms==300)].groupby(['command','motion_period'])['corr'].agg(['median','mean']).reset_index().to_dict('records'),mission_change=pd.DataFrame(missions).groupby('split').agg(windows=('windows','sum'),changed=('change_windows','sum'),flights_with_change=('change_windows',lambda x:int((x>0).sum()))).reset_index().to_dict('records')))
if __name__=='__main__':main()
