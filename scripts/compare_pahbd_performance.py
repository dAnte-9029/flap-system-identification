"""Compare fixed-input GPU snapshots and measured throughput; no model ranking."""
import argparse,json,statistics
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);a=p.parse_args()
old=json.loads((a.baseline/'timings.json').read_text());new=json.loads((a.candidate/'timings.json').read_text());rows=[]
for n in new:
    o=next(r for r in old if (r['mode'],r['horizon'])==(n['mode'],n['horizon']))
    name=f"{n['mode']}_{n['horizon']}.pt";x=torch.load(a.baseline/name,weights_only=True);y=torch.load(a.candidate/name,weights_only=True)
    assert x['window_id']==y['window_id'];errors={}
    for group in ('pred','loss','grads'):
        assert x[group].keys()==y[group].keys()
        errors[group]={}
        for k in x[group]:
            u,v=x[group][k],y[group][k];torch.testing.assert_close(u,v,atol=1e-8,rtol=1e-8)
            errors[group][k]=float((u-v).abs().max())
    for bundle in (x,y):
        forcing=sum(float(v.abs().sum()) for k,v in bundle['grads'].items() if 'mixer.forcing' in k)
        control=sum(float(v.abs().sum()) for k,v in bundle['grads'].items() if 'mixer.control' in k)
        assert forcing>0 and (control>0 if n['mode']=='A' else control==0)
    def elapsed(r):return statistics.mean(sum(t['wall'] for t in sample.values()) for sample in r['samples'])+r['load']['wall']+r.get('optimizer',{}).get('wall',0.)
    rows.append(dict(mode=n['mode'],horizon=n['horizon'],old_seconds=elapsed(o),new_seconds=elapsed(n),speedup=elapsed(o)/elapsed(n),errors=errors))
result=dict(tolerance=dict(atol=1e-8,rtol=1e-8),rows=rows)
if len(rows)==8:
    average=statistics.mean(r['new_seconds'] for r in rows);validation=statistics.mean(r['validation']['wall'] for r in new)
    result.update(mean_step_seconds=average,validation_window_seconds=validation,queue_eta_seconds=6*(10944*average+12*156*validation))
(a.candidate/'comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps({**result,'rows':[{k:v for k,v in r.items() if k!='errors'} for r in rows]},indent=2))
