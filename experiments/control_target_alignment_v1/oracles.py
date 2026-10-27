"""Four offline label oracles and exact additive error decomposition."""
from common import *

def build():
 z,n,w,p,coh=read_assets();templates=pd.read_csv(ABL/'results/factual_curves.csv.gz').query("variant=='V0' and model=='GRU-Control' and seed==17 and variable in ['p','q','r']")
 rows=[];single=[];decomp=[];maxtruth=0.
 for pair in p.itertuples():
  template=templates[templates.pair_uid==pair.pair_uid];tt=template[['time_ms','time_kind','time_label']].drop_duplicates();at=tt.time_ms.to_numpy()/1000
  vals=[];parts=[]
  for i in [pair.a,pair.b]:
   tape,initial,ti,b0=tapes(z,n,i);v={};part={};end=tape['A'][0][-1]
   nt=n[f'rate_t_{i}'];ny=n[f'rate_y_{i}'];native=(nt,ny)
   for method in ['A','B','D']:
    slow=method=='D';supported=(at<=min(.5,end)+1e-10)
    if slow:supported&=(at>=.135-1e-10)&(at+.135<=min(.5,end)+1e-10)
    true=values(tape['A'],at) if method=='A' else (tc.boxmean(nt,ny-b0,at,.27,.02) if slow else tc.interp(nt,ny-b0,at,.02))
    a=values(tape['label'],at,slow);b=(tc.boxmean(nt,ny,at,.27,.02) if slow else tc.interp(nt,ny,at,.02))
    ef=a-b;eo=initial-b0;ideal=values(tape['O0'],at,slow);bridge=ideal-(a-initial)
    pred={k:values(tape[k],at,slow) for k in ['O0','O1','O2','O3']}
    # O3 is exact native reference self-check, even off the registered grid.
    pred['O3']=b-b0
    # O2 changes only the increment anchor. At t0 it is not a causal-state rollout;
    # this intentional diagnostic is distinct from O0's continuous registered interface.
    pred['O2']=b-initial
    v[method]=dict(real=np.where(supported[:,None],true,np.nan),**{k:np.where(supported[:,None],a_,np.nan) for k,a_ in pred.items()})
    part[method]=dict(future=ef,origin=np.broadcast_to(eo,ef.shape),bridge=bridge,increment=ideal-(b-b0))
    if method in ['B','D']:
     np.testing.assert_allclose(part[method]['increment'],ef-eo+bridge,atol=2e-15,rtol=0,equal_nan=True)
     for k,t in enumerate(tt.itertuples()):
      for j,var in enumerate(VARS):single.append(dict(window_id=w.window_id.iloc[i],rollout_index=i,flight_id=w.flight_id.iloc[i],split=pair.split,method=method,variable=var,time_label=t.time_label,time_ms=t.time_ms,supported=bool(supported[k] and np.isfinite(ideal[k,j]) and np.isfinite(b[k,j])),E_future=ef[k,j] if supported[k] else np.nan,E_origin=eo[j],bridge=bridge[k,j] if supported[k] else np.nan,E_increment=part[method]['increment'][k,j] if supported[k] else np.nan,initial_rate=initial[j],truth_initial_rate=ti[j],B_initial_rate=b0[j],target_A=a[k,j] if supported[k] else np.nan,native_B=b[k,j] if supported[k] else np.nan))
   vals.append(v);parts.append(part)
  ridx={(r.time_label,r.method,r.variable):r for r in template.itertuples()}
  for method in ['A','B','D']:
   real=vals[0][method]['real']-vals[1][method]['real']
   for k,t in enumerate(tt.itertuples()):
    for j,var in enumerate(VARS):
     frozen=ridx[(t.time_label,method,var)];r=real[k,j]
     if frozen.supported:
      assert np.isfinite(r);maxtruth=max(maxtruth,abs(r-frozen.real))
     for oracle in ['O0','O1','O2','O3']:
      pred=(vals[0][method][oracle]-vals[1][method][oracle])[k,j];supported=bool(frozen.supported and np.isfinite(pred))
      rows.append(dict(pair_uid=pair.pair_uid,split=pair.split,model=oracle,method=method,variable=var,time_ms=t.time_ms,time_kind=t.time_kind,time_label=t.time_label,supported=supported,real=r if supported else np.nan,pred=pred if supported else np.nan,error=pred-r if supported else np.nan))
     if method in ['B','D']:
      terms={key:(parts[0][method][key]-parts[1][method][key])[k,j] for key in parts[0][method]}
      decomp.append(dict(pair_uid=pair.pair_uid,split=pair.split,method=method,variable=var,time_label=t.time_label,time_ms=t.time_ms,supported=frozen.supported,**{key:(v if frozen.supported else np.nan) for key,v in terms.items()}))
 assert maxtruth<1e-10,maxtruth
 d=pd.DataFrame(rows);d.to_csv(OUT/'oracle_curves.csv.gz',index=False)
 points=d[(d.time_kind!='grid')|d.time_label.isin(['100','200','300','400','500'])];points.to_csv(OUT/'oracle_points.csv.gz',index=False)
 pd.DataFrame(single).drop_duplicates(['window_id','method','variable','time_label']).to_csv(OUT/'window_decomposition.csv.gz',index=False)
 pd.DataFrame(decomp).to_csv(OUT/'pair_decomposition.csv.gz',index=False)
 save(OUT/'truth_regression.json',dict(passed=True,max_abs_difference=maxtruth,reference_rows=len(templates),training_runs=0))
 return points,p,coh

def aggregate(d,p,coh):
 rows=[];group=[]
 for (split,cohort),members in coh.groupby(['split','cohort']):
  pp=p[p.pair_uid.isin(members.pair_uid)];dd=d[d.pair_uid.isin(members.pair_uid)&d.supported]
  for key,g in dd.groupby(['model','method','variable','time_label']):
   meta=dict(zip(['model','method','variable','time_label'],key));ep=pp[pp.pair_uid.isin(g.pair_uid)]
   for weighting in (['pair','flight','component'] if key[-1]=='300' else ['flight']):
    wm=stats.weights(ep,weighting);rows.append(dict(split=split,cohort=cohort,**meta,weighting=weighting,requested_pairs=len(pp),missing_pairs=len(pp)-len(ep),**metric(g,g.pair_uid.map(wm))))
   if cohort=='main' and key[-1]=='300':
    a=pd.read_csv(OUT/'component_assignments.csv');groups=[('component',v,set(x.pair_uid)) for v,x in a.groupby('component_id')]+[('flight',v,set(ep[(ep.flight_a==v)|(ep.flight_b==v)].pair_uid)) for v in sorted(set(ep.flight_a)|set(ep.flight_b))]
    for kind,value,ids in groups:
     x=g[g.pair_uid.isin(ids)]
     if len(x):group.append(dict(split=split,**meta,grouping=kind,group=value,**metric(x,np.ones(len(x)))))
 pd.DataFrame(rows).to_csv(OUT/'oracle_summary.csv',index=False);pd.DataFrame(group).to_csv(OUT/'oracle_per_group.csv',index=False)
 main=pd.DataFrame(rows).query("split=='train' and cohort=='main' and variable=='p' and time_label=='300' and weighting=='flight'");main.to_csv(OUT/'main_p_300.csv',index=False);print(main[['model','method','direction_accuracy','scale_ratio','signed_gain','rmse']].to_string(index=False),flush=True)
if __name__=='__main__':aggregate(*build())
