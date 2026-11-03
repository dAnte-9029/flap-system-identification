#!/usr/bin/env python3
"""Compare execution engines only, not predictive model quality."""
import argparse,json,statistics
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);a=p.parse_args()
ref=json.loads((a.reference/'timings.json').read_text());cand=json.loads((a.candidate/'timings.json').read_text())
def compare_tree(x,y):
 if isinstance(x,torch.Tensor):
  torch.testing.assert_close(x,y,atol=1e-8,rtol=1e-8)
  return float((x-y).abs().max()) if x.numel() else 0.
 if isinstance(x,dict):
  assert x.keys()==y.keys();return max((compare_tree(x[k],y[k]) for k in x),default=0.)
 if isinstance(x,(tuple,list)):
  assert len(x)==len(y);return max((compare_tree(u,v) for u,v in zip(x,y)),default=0.)
 assert x==y;return 0.
rows=[]
for c in cand:
 r=next(t for t in ref if t['horizon']==c['horizon'] and t['mode']==c['mode'])
 name=f"{c['mode']}_{c['horizon']}.pt";x=torch.load(a.reference/name,weights_only=True);y=torch.load(a.candidate/name,weights_only=True)
 assert x['window_id']==y['window_id'];assert x['schedule']==y['schedule']
 errors={k:compare_tree(x[k],y[k]) for k in ('pred','loss','grads')}
 for b in (x,y):
  force=sum(float(v.abs().sum()) for k,v in b['grads'].items() if 'mixer.forcing' in k)
  control=sum(float(v.abs().sum()) for k,v in b['grads'].items() if 'mixer.control' in k)
  assert force>0 and (control>0 if c['mode']=='A' else control==0)
 elapsed=lambda row:statistics.mean(sum(v['wall'] for v in s.values()) for s in row['samples'])
 rows.append(dict(mode=c['mode'],horizon=c['horizon'],steps=len(x['schedule']),schedule_exact=True,reference_seconds=elapsed(r),candidate_seconds=elapsed(c),speedup=elapsed(r)/elapsed(c),max_errors=errors,
                  warmed_graphs_before=c['counters_before'].get('stats',{}).get('unique_graphs',0),warmed_graphs_after=c['counters_after'].get('stats',{}).get('unique_graphs',0)))
result=dict(rows=rows,tolerance=dict(atol=1e-8,rtol=1e-8))
if (a.candidate/'updates.pt').exists():
 x=torch.load(a.reference/'updates.pt',map_location='cpu',weights_only=True);y=torch.load(a.candidate/'updates.pt',map_location='cpu',weights_only=True)
 result['update_max_errors']={k:compare_tree(x[k],y[k]) for k in ('model','optimizer')}
 assert x['sampler']==y['sampler']
 old=json.loads((a.reference/'updates.json').read_text());new=json.loads((a.candidate/'updates.json').read_text())
 assert [r['window_id'] for r in old['draws']]==[r['window_id'] for r in new['draws']]
 result['loss_trajectory_max_error']=max(abs(u['total']-v['total']) for u,v in zip(old['draws'],new['draws']))
 for u,v in zip(old['draws'],new['draws']):torch.testing.assert_close(torch.tensor(u['total'],dtype=torch.float64),torch.tensor(v['total'],dtype=torch.float64),atol=1e-8,rtol=1e-8)
 result['training']=dict(reference_seconds=old['wall_seconds'],candidate_seconds=new['wall_seconds'],steps=len(old['draws']),speedup=old['wall_seconds']/new['wall_seconds'])
(a.candidate/'comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
