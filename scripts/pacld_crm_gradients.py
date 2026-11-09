"""Frozen GPU autograd diagnostics. No optimizer; parameters asserted unchanged."""
import argparse
import hashlib
import json
import pickle
import time
import numpy as np
import pandas as pd
import torch
from pacld_crm_v1 import OUT, SOURCE, CHECKPOINT, SHA, save
from evaluate_pacld_a2_v2_gpu import prepare_batch
from system_identification.models.pacld import Config
from system_identification.models.pacld.batch import BatchedPACLD, FIELDS
from system_identification.data.pacld_batch import compile_windows, steps, pad_steps

def module(name):
    for k in ['coupling','equilibrium','relaxation','slow','decoder']:
        if name.startswith('flow.dynamics.'+k+'.'):return k
    return 'belief' if name.startswith('belief.') else 'other'

def cosine(a,b):
    denom=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/denom) if denom>0 else None

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--microbatch',type=int,default=4);args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(17)
    assert torch.cuda.is_available()
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()==SHA
    bundle=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    model=BatchedPACLD(Config(**bundle['config']['model']),bundle['precision'],checkpoint_steps=True).to('cuda:0').eval()
    model.load_state_dict(bundle['model']);names,params=zip(*model.named_parameters())
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
    groups=np.concatenate([np.repeat(module(n),p.numel()) for n,p in zip(names,params)])
    df=pd.read_csv(OUT/'windows.csv');cohort=pd.read_csv(OUT/'gradient_cohort.csv')
    identity=dict(checkpoint_sha256=SHA,cohort_sha256=hashlib.sha256((OUT/'gradient_cohort.csv').read_bytes()).hexdigest(),window_values_sha256=hashlib.sha256(df[['flight','t','Y','R_normal','Y_hat']].to_csv(index=False).encode()).hexdigest())
    identity_path=OUT/'gradient_input_identity.json'
    if identity_path.exists():
        assert json.loads(identity_path.read_text())==identity, 'gradient cache identity changed'
    else:
        assert not list(OUT.glob('gradient_vectors_batch*.npz')), 'unidentified gradient cache'
        save('gradient_input_identity.json',identity)
    frame=pd.read_pickle(SOURCE/'features.pkl')
    scales=json.loads((OUT/'moments.json').read_text())['scales'];su=scales['s_u'][0];sp=scales['s_p']
    vectors={};records=[];checks=[];start=time.monotonic()
    def gradient(loss,retain=False):
        gs=torch.autograd.grad(loss,params,allow_unused=True,retain_graph=retain)
        return np.concatenate([(torch.zeros_like(p) if g is None else g).detach().cpu().double().numpy().ravel() for g,p in zip(gs,params)])
    for batch_id,g in cohort.groupby('batch'):
        path=OUT/f'gradient_vectors_batch{batch_id}.npz'
        if path.exists():
            old=np.load(path);vectors[int(batch_id)]={k:old[k] for k in old.files};continue
        accum={k:np.zeros(sum(p.numel() for p in params)) for k in ['FULL','M','M_stop']};mv=0.;fullval=0.;parity=0.;chain=0.
        for lo in range(0,len(g),args.microbatch):
            keys=g.iloc[lo:lo+args.microbatch][['flight','t']]
            src=keys.merge(frame,on=['flight','t'],validate='one_to_one')
            obs=keys.merge(df,on=['flight','t'],validate='one_to_one')
            _,ws,queries=pickle.loads(prepare_batch((bundle['config'],'train',src)))
            batch=compile_windows(ws,model.config).to('cuda:0',torch.float64)
            result=model(batch,0.)
            belief=result['belief']; dt,u=pad_steps([steps(0,q[0],w.controls.canonical(),model.config.max_step) for w,q in zip(ws,queries)])
            plan=(dt.to(belief.x),u.to(belief.x))
            future=model.propagate_plan(belief,plan)
            yhat=future.x[:,10]-belief.x[:,10]
            y=torch.tensor(obs.Y.to_numpy(),device='cuda:0');r=torch.tensor(obs.R_normal.to_numpy()/su,device='cuda:0')
            m=(r*(y-yhat)/sp).mean();full=result['full'].mean()
            frac=len(ws)/len(g);mv+=float(m.detach())*frac;fullval+=float(full.detach())*frac
            parity=max(parity,float(np.max(abs(yhat.detach().cpu().numpy()-obs.Y_hat.to_numpy()))))
            gm=gradient(m,True);gm2=gradient(m.square(),True)
            chain=max(chain,float(np.max(abs(gm2-2*float(m.detach())*gm))))
            gf=gradient(full)
            accum['FULL']+=frac*gf;accum['M']+=frac*gm
            detached=belief.with_(**{k:getattr(belief,k).detach() for k in FIELDS})
            stop_future=model.propagate_plan(detached,plan)
            ms=(r*(y-(stop_future.x[:,10]-detached.x[:,10]))/sp).mean()
            accum['M_stop']+=frac*gradient(ms)
            del result,belief,future,m,full,stop_future,ms,batch
            print('batch',batch_id,'windows',lo+len(ws),'/',len(g),'elapsed',round(time.monotonic()-start,1),flush=True)
        accum['M2']=2*mv*accum['M'];accum['M2_stop']=2*mv*accum['M_stop']
        assert all(np.isfinite(v).all() for v in accum.values())
        accum['moment_value']=np.array(mv);accum['full_value']=np.array(fullval)
        accum['parity_max']=np.array(parity);accum['chain_max']=np.array(chain)
        np.savez_compressed(path,**accum);vectors[int(batch_id)]=accum
    for b,v in vectors.items():
        for kind in ['FULL','M','M2','M_stop','M2_stop']:
            for group in sorted(set(groups)):
                ix=groups==group;a=v[kind][ix];f=v['FULL'][ix]
                records.append(dict(batch=b,objective=kind,module=group,parameters=int(ix.sum()),l2=float(np.linalg.norm(a)),rms=float(np.sqrt(np.mean(a*a))),max_abs=float(np.max(abs(a))),cosine_FULL=cosine(a,f),finite=bool(np.isfinite(a).all())))
        checks.append(dict(batch=b,M=float(v['moment_value']),FULL=float(v['full_value']),factual_parity_max=float(v['parity_max']),autograd_M2_chain_max=float(v['chain_max'])))
    stable=[]
    for i in vectors:
        for j in vectors:
            if i>=j:continue
            for k in ['FULL','M','M2','M_stop','M2_stop']:
                for group in sorted(set(groups)):
                    ix=groups==group;stable.append(dict(batch_a=i,batch_b=j,objective=k,module=group,cosine=cosine(vectors[i][k][ix],vectors[j][k][ix])))
    assert all(torch.equal(initial[n],p.detach().cpu()) for n,p in model.named_parameters())
    pd.DataFrame(records).to_csv(OUT/'gradient_statistics.csv',index=False)
    pd.DataFrame(stable).to_csv(OUT/'gradient_stability.csv',index=False)
    save('gradient_checks.json',dict(device=torch.cuda.get_device_name(),precision=bundle['precision'],batches=checks,parameters_unchanged=True,optimizer_created=False,microbatch=args.microbatch,cohort_reduction='equal windows within each fixed diagnostic batch; exact accumulated gradients; M2=2*M*grad(M), additionally checked by autograd per microbatch',full='original native physical FULL loss on same 0.5s windows; Macro untouched, not included in FULL comparator',moment='exact native .2s target; logged ZOH; no future assimilation',stop='detach every initial belief state field on Moment branch only',elapsed_s=time.monotonic()-start))

if __name__=='__main__':main()
