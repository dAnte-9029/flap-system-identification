"""Companion fast-residual summaries, using the same fixed memberships and weights."""
from common import *
from analyze import metric
OUT=RESULTS;d=pd.read_csv(OUT/'responses.csv.gz');p=pd.read_csv(OLD/'results/pairs.csv');m=pd.read_csv(OLD/'results/cohorts.csv');rows=[]
q=d[d.method.isin(['fast_A','fast_B'])&(d.point=='300')]
for (width,method,var,model),g in q.groupby(['width_s','method','variable','model']):
 for (split,cohort),mem in m.groupby(['split','cohort']):
  sub=g[g.pair_uid.isin(mem.pair_uid)];pp=p[p.pair_uid.isin(mem.pair_uid)]
  for seed in sorted(sub.seed.unique().tolist())+['pooled']:
   a=sub if seed=='pooled' else sub[sub.seed==seed]
   for weight in ['pair','flight','component']:rows.append(dict(split=split,cohort=cohort,width_s=width,method=method,variable=var,model=model,seed=seed,weighting=weight,point='300',**metric(a,a.pair_uid.map(weights(pp,weight)).to_numpy())))
pd.DataFrame(rows).to_csv(OUT/'fast_summary.csv',index=False)
print('Fast residual summary rows',len(rows))
