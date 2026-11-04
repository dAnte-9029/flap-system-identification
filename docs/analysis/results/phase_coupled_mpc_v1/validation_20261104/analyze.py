"""Read saved validation outputs and checkpoints; never opens flight datasets."""
import csv,json,math,sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
repo=Path('/home/zn/flap-system-identification/.worktrees/pahbd-compile-v4');sys.path.insert(0,str(repo/'src'))
from system_identification.training.pahbd import load_checkpoint
root=Path('/home/zn/pahbd-runs/ab-compiled-dual-20261103-1812');out=Path(__file__).parent
q=json.loads((root/'queue.json').read_text());assert q['state']=='paused'
selected={};last={};histories={};draws={};identities=[];summaries=[];pauses={}
def history(directory,limit,meta):
    samples=[];validations=[]
    if meta.get('resumed_from'):
        ancestor=Path(meta['resumed_from']);_,_,b=load_checkpoint(ancestor)
        samples,validations=history(ancestor.parent,b['step'],b['metadata'])
    p=directory/'training_draws.jsonl'
    if p.exists():samples += [d for line in p.read_text().splitlines() if (d:=json.loads(line))['step']<=limit]
    p=directory/'progress.json'
    if p.exists():validations += [r for r in json.loads(p.read_text())['records'] if r['step']<=limit]
    return samples,validations
for task in q['tasks']:
    name=task['name'];directory=root/name
    ptr=json.loads((directory/'latest.json').read_text());_,opt,final=load_checkpoint(ptr['checkpoint'],optimizer=True)
    assert final['step']==final['training_state']['sampler']['draws']==ptr['step']
    assert len(opt.state)==19
    if task['seed']==43:
        assert task['state']=='paused' and task['exit_code']==75
        pauses[name]=dict(step=final['step'],checkpoint=ptr['checkpoint'],adam_states=len(opt.state));continue
    assert task['state']=='completed' and task['exit_code']==0 and final['step']==10944
    cfg=json.loads((root/'configs'/f'{name}.json').read_text());assert cfg==final['metadata']['config']
    best=json.loads((directory/'best.json').read_text());_,_,b=load_checkpoint(best['checkpoint'])
    ds,hs=history(directory,10944,final['metadata']);draws[name]=ds;histories[name]=hs
    assert [d['step'] for d in ds]==list(range(1,10945))
    best_record=next(r for r in hs if r['step']==b['step']);last_record=next(r for r in hs if r['step']==10944)
    assert abs(best_record['validation']['score']-best['score'])<1e-10
    assert abs(min(r['validation']['score'] for r in hs)-best['score'])<1e-10
    selected[name]=best_record['validation'];last[name]=last_record['validation']
    meta=b['metadata'];identities.append((meta['validation_window_ids'],meta['data']['manifest_sha256'],meta['phase_contract']))
    assert len(meta['validation_window_ids'])==156 and len(selected[name]['per_flight'])==39
    first=hs[0]
    summaries.append(dict(task=name,best_step=b['step'],best_score=best['score'],last_score=last[name]['score'],first_step=first['step'],first_score=first['validation']['score'],best_improvement_vs_first_pct=100*(1-best['score']/first['validation']['score']),checkpoint=best['checkpoint']))
assert all(x==identities[0] for x in identities)
for seed in [17,29]:
    a,b=(f'{m}_seed{seed}' for m in ['A','B'])
    assert [r['window_id'] for r in draws[a]]==[r['window_id'] for r in draws[b]]
    ca=json.loads((root/'configs'/f'{a}.json').read_text());cb=json.loads((root/'configs'/f'{b}.json').read_text());ca['model'].pop('mode');cb['model'].pop('mode');assert ca==cb
paired=[];native=[];flights=[];horizon_losses=[]
for seed in [17,29]:
    a,b=(selected[f'{m}_seed{seed}'] for m in ['A','B']);keys=sorted(a['per_flight']);assert keys==sorted(b['per_flight'])
    delta=[a['per_flight'][f]-b['per_flight'][f] for f in keys]
    paired.append(dict(seed=seed,A=a['score'],B=b['score'],A_improvement_pct=100*(1-a['score']/b['score']),A_wins=sum(d<0 for d in delta),flights=len(keys),mean_A_minus_B=float(np.mean(delta)),median_A_minus_B=float(np.median(delta)),last_A_improvement_pct=100*(1-last[f'A_seed{seed}']['score']/last[f'B_seed{seed}']['score'])))
    for f in keys:flights.append(dict(seed=seed,flight_id=f,A=a['per_flight'][f],B=b['per_flight'][f],delta_A_minus_B=a['per_flight'][f]-b['per_flight'][f]))
    am={(r['component'],r['horizon_s']):r for r in a['native_flight_equal']};bm={(r['component'],r['horizon_s']):r for r in b['native_flight_equal']};assert am.keys()==bm.keys()
    for k in sorted(am):
        assert am[k]['flights']==bm[k]['flights']
        af={f for f,rows in a['native_per_flight'].items() if any((r['component'],r['horizon_s'])==k for r in rows)}
        bf={f for f,rows in b['native_per_flight'].items() if any((r['component'],r['horizon_s'])==k for r in rows)}
        assert af==bf and len(af)==am[k]['flights']
        native.append(dict(seed=seed,component=k[0],horizon_s=k[1],flights=am[k]['flights'],A=am[k]['rmse'],B=bm[k]['rmse'],A_improvement_pct=100*(1-am[k]['rmse']/bm[k]['rmse'])))
    for h in [200000,500000,1000000,2000000]:
        va=[r['loss'] for r in a['per_flight_horizon'] if r['horizon_us']==h];vb=[r['loss'] for r in b['per_flight_horizon'] if r['horizon_us']==h];assert len(va)==len(vb)==39
        horizon_losses.append(dict(seed=seed,horizon_us=h,A=float(np.mean(va)),B=float(np.mean(vb))))
for name,rows in [('models',summaries),('paired_scores',paired),('native_metrics',native),('paired_flights',flights),('horizon_loss',horizon_losses)]:
    with (out/f'{name}.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
pooled=[]
for component in ['p','v','R','omega']:
    for horizon in [.2,.5,1.,2.]:
        rs=[r for r in native if r['component']==component and r['horizon_s']==horizon]
        a=math.sqrt(np.mean([r['A']**2 for r in rs]));b=math.sqrt(np.mean([r['B']**2 for r in rs]));pooled.append(dict(component=component,horizon_s=horizon,flights=rs[0]['flights'],A=a,B=b,A_improvement_pct=100*(1-a/b)))
summary=dict(run=str(root),source_revision=q['code_revision'],cohort_windows=156,cohort_flights=39,validation_cohort_equal=True,paired_training_order_exact=True,paused=pauses,models=summaries,paired_scores=paired,pooled_native=pooled,method='Existing fixed-cohort validation outputs; best selected independently by original score; two seeds equal weight in MSE; no fresh inference or test access')
(out/'summary.json').write_text(json.dumps(summary,indent=2))
fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for ax,seed in zip(axes,[17,29]):
    for mode in ['A','B']:
        hs=histories[f'{mode}_seed{seed}'];ax.plot([r['step'] for r in hs],[r['validation']['score'] for r in hs],marker='o',markersize=3,label=mode)
    ax.set(title=f'Seed {seed}: fixed validation cohort',xlabel='Optimizer updates',ylabel='Flight-equal validation objective');ax.legend();ax.grid(alpha=.2)
fig.savefig(out/'validation_curves.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for ax,c,unit in zip(axes,['v','omega'],['m/s','rad/s']):
    rs=[r for r in pooled if r['component']==c]
    for mode in ['A','B']:ax.plot([r['horizon_s'] for r in rs],[r[mode] for r in rs],marker='o',label=mode)
    ax.set(xlabel='Horizon (s)',ylabel=f'{c} vector RMSE ({unit})',title='Best checkpoints; equal flight/seed MSE');ax.legend();ax.grid(alpha=.2)
fig.savefig(out/'velocity_rate_rmse.png',dpi=180);plt.close(fig)
print(json.dumps(summary,indent=2))
