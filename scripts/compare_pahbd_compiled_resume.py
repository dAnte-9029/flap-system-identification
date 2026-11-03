#!/usr/bin/env python3
"""Compare bounded checkpoint continuation and split/resume diagnostics."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);a=p.parse_args()
def compare(x,y,exact=False):
 if isinstance(x,torch.Tensor):
  torch.testing.assert_close(x,y,atol=0 if exact else 1e-8,rtol=0 if exact else 1e-8)
  return float((x.double()-y.double()).abs().max()) if x.numel() else 0.
 if isinstance(x,np.ndarray):
  np.testing.assert_array_equal(x,y);return 0.
 if isinstance(x,dict):
  assert x.keys()==y.keys();return max((compare(x[k],y[k],exact) for k in x),default=0.)
 if isinstance(x,(list,tuple)):
  assert len(x)==len(y);return max((compare(u,v,exact) for u,v in zip(x,y)),default=0.)
 assert x==y;return 0.
x=torch.load(a.reference/'final.pt',map_location='cpu',weights_only=False);y=torch.load(a.candidate/'final.pt',map_location='cpu',weights_only=False)
errors={k:compare(x[k],y[k],k in ('sampler','schedule','rng')) for k in x}
r=json.loads((a.reference/'result.json').read_text());c=json.loads((a.candidate/'result.json').read_text())
assert r['start_step']==c['start_step'] and r['end_step']==c['end_step']
assert [d['window_id'] for d in r['draws']]==[d['window_id'] for d in c['draws']]
loss_errors=[]
for u,v in zip(r['draws'],c['draws']):
 torch.testing.assert_close(torch.tensor(u['total'],dtype=torch.float64),torch.tensor(v['total'],dtype=torch.float64),atol=1e-8,rtol=1e-8);loss_errors.append(abs(u['total']-v['total']))
torch.testing.assert_close(torch.tensor(r['validation']['score'],dtype=torch.float64),torch.tensor(c['validation']['score'],dtype=torch.float64),atol=1e-8,rtol=1e-8)
result=dict(start=r['start_step'],end=r['end_step'],max_errors=errors,loss_max_error=max(loss_errors),validation_score_error=abs(r['validation']['score']-c['validation']['score']),sampler_schedule_rng_exact=True)
(a.candidate/'comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
