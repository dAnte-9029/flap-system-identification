"""Read archived predictions only: independent budget-separated regression."""
from common import *

def main():
 oracle=pd.read_csv(OUT/'oracle_points.csv.gz');o0=oracle[oracle.model=='O0'].drop(columns='model');z,n,w,p,coh=read_assets();x=np.load(SWAP/'results/inputs.npz');swp=pd.read_csv(SWAP/'results/pairs.csv');np.testing.assert_array_equal(p.pair_uid,swp.pair_uid)
 rows=[];singles=[];regr=[];summ=[];associations=[]
 assignment=pd.read_csv(OUT/'component_assignments.csv');mainids=set(assignment.pair_uid);pairloc={u:i for i,u in enumerate(p.pair_uid)}
 for budget,exp,models,seeds in [(6,PILOT,['M0W0','M1W0'],[17]),(24,FORMAL,['M0','M1'],[17,29,43])]:
  facts=pd.read_csv(exp/'results/factual_points.csv.gz');facts=facts[facts.model.isin(models)&facts.variable.isin(VARS)]
  preds={f:dict(np.load(exp/f'results/predictions_fold{f}.npz')) for f in range(3)};lookup={f:{ci:i for i,ci in enumerate(preds[f]['case_ids'])} for f in preds}
  err=0.;num=0
  for r in facts[facts.supported].itertuples():
   pi=pairloc[r.pair_uid];arr=preds[r.fold][f'{r.model}_{r.seed}'];a=[]
   for ci in [pi*4,pi*4+3]:
    count=int(x['valid_steps'][ci]);tt=x['native_time_us'][ci,:count+1]*1e-6;yy=np.vstack([x['initial'][ci,10:13],arr[lookup[r.fold][ci],:count,10:13]]).astype(float);yy-=yy[0].copy()
    a.append(values((tt,yy),[r.time_ms/1000],r.method=='D')[0,VARS.index(r.variable)])
   delta=a[0]-a[1];err=max(err,abs(delta-r.pred));num+=1
  assert err<1e-11,err;regr.append(dict(budget=budget,rows=num,max_prediction_difference=err))
  joined=facts.merge(o0[['pair_uid','variable','method','time_label','real','pred','supported']],on=['pair_uid','variable','method','time_label'],suffixes=('','_O0'),validate='many_to_one')
  np.testing.assert_allclose(joined.real,joined.real_O0,atol=1e-10,rtol=0,equal_nan=True);np.testing.assert_array_equal(joined.supported,joined.supported_O0)
  joined['budget_epochs']=budget;joined['label_error']=joined.pred_O0-joined.real;joined['learning_error']=joined.pred-joined.pred_O0;joined['total_error']=joined.pred-joined.real
  np.testing.assert_allclose(joined.total_error,joined.learning_error+joined.label_error,atol=1e-15,rtol=0,equal_nan=True)
  joined['direction_active']=abs(joined.real)>.1;joined['model_correct']=(joined.real*joined.pred>0)&(abs(joined.pred)>1e-8);joined['oracle_correct']=(joined.real*joined.pred_O0>0)&(abs(joined.pred_O0)>1e-8)
  rows.append(joined)
  # Native registered future-node error to original A labels, without pair cancellation.
  for fold,pp in preds.items():
   for model in models:
    for seed in seeds:
     arr=pp[f'{model}_{seed}']
     for ci in pp['case_ids']:
      if ci%4 not in [0,3]:continue
      wi=int(x['receiver_index'][ci]);count=int(x['valid_steps'][ci]);a=z['truth'][wi,:count,10:13].astype(float);pred=arr[lookup[fold][ci],:count,10:13].astype(float)
      for j,var in enumerate(VARS):singles.append(dict(budget_epochs=budget,fold=fold,model=model,seed=seed,pair_uid=p.pair_uid.iloc[ci//4],window_id=w.window_id.iloc[wi],flight_id=w.flight_id.iloc[wi],split=w.split.iloc[wi],variable=var,nodes=count,absolute_A_mse=np.mean((pred[:,j]-a[:,j])**2),absolute_A_bias=np.mean(pred[:,j]-a[:,j])))
  # Reproduce main frozen summaries per seed, using original reference/weights.
  frozen=pd.read_csv(exp/'results/factual_summary.csv')
  for key,g in joined[(joined.time_label=='300')&joined.pair_uid.isin(mainids)].groupby(['model','seed','variable','method']):
   meta=dict(zip(['model','seed','variable','method'],key));cp=p[p.pair_uid.isin(g.pair_uid)]
   for weighting in ['pair','flight','component']:
    weights=g.pair_uid.map(stats.weights(cp,weighting)).to_numpy();met=metric(g,weights);r=dict(budget_epochs=budget,**meta,weighting=weighting,**met);weights/=weights.sum()
    for field in ['label_error','learning_error','total_error']:r[field+'_rms']=np.sqrt(np.dot(weights,g[field]**2))
    r['error_cross_moment']=np.dot(weights,g.label_error*g.learning_error);r['oracle_correct_model_wrong']=int((g.direction_active&g.oracle_correct&~g.model_correct).sum());r['oracle_wrong_model_correct']=int((g.direction_active&~g.oracle_correct&g.model_correct).sum());summ.append(r)
    base=frozen[(frozen.scope=='all_heldout')&(frozen.cohort=='main')&(frozen.model==key[0])&(frozen.seed.astype(str)==str(key[1]))&(frozen.variable==key[2])&(frozen.method==key[3])&(frozen.time_label=='300')&(frozen.weighting==weighting)]
    assert len(base)==1
    for field in ['direction_accuracy','scale_ratio','signed_gain','rmse']:assert abs(met[field]-base[field].iloc[0])<1e-12
   for grouping,parts in [('pair',[('all',g)]),('component',[(v,g[g.pair_uid.isin(a.pair_uid)]) for v,a in assignment.groupby('component_id')])]:
    for name,gg in parts:
     if len(gg)<2:continue
     corr=lambda aa,bb:float(np.corrcoef(aa,bb)[0,1]) if np.std(aa)>1e-12 and np.std(bb)>1e-12 else np.nan
     associations.append(dict(budget_epochs=budget,**meta,grouping=grouping,group=name,pairs=len(gg),corr_signed_label_model_error=corr(gg.label_error,gg.total_error),corr_abs_label_abs_model_error=corr(abs(gg.label_error),abs(gg.total_error)),direction_errors=int((gg.direction_active&~gg.model_correct).sum()),label_rms=np.sqrt(np.mean(gg.label_error**2))))
 pd.concat(rows).to_csv(OUT/'model_error_decomposition.csv.gz',index=False);pd.DataFrame(singles).to_csv(OUT/'model_single_window_A_error.csv.gz',index=False);pd.DataFrame(summ).to_csv(OUT/'model_comparison_300.csv',index=False);pd.DataFrame(associations).to_csv(OUT/'model_label_associations.csv',index=False)
 save(OUT/'model_regression.json',dict(passed=True,checks=regr,checkpoint_loaded=False,inference=False,training_runs=0))
 print('Archived model checks passed',regr,flush=True)
if __name__=='__main__':main()
