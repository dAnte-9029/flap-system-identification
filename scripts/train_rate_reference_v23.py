"""One fixed, exactly paired rate-reference training run."""
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from rate_reference_v23 import OUT,V8,load,ReferenceGRUPhaseSchedule
from reference_groupcv_v8 import predict
from reference_capacity_v6 import metric_frames


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,required=True);ap.add_argument('--seed',type=int,required=True);a=ap.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    protocol=json.loads((OUT/'protocol.json').read_text())
    for name,digest in protocol['source_pins'].items():assert base.sha(base.ROOT/name)==digest,name
    x,y=load();w=pd.read_csv(OUT/'windows.csv');f=OUT/f'fold{a.fold}';info=json.loads((f/'manifest.json').read_text())
    assert base.sha(f/'normalization.npz')==info['normalization_sha256']
    with np.load(f/'normalization.npz') as z:stats={k:z[k] for k in z.files}
    old=V8/f'fold{a.fold}/hold_{a.seed}';oldstatus=json.loads((old/'status.json').read_text())
    assert base.sha(old/'sample_draws.npz')==oldstatus['draws_sha256']
    with np.load(old/'sample_draws.npz') as z:draws=z['array_index']
    assert draws.shape==(1600,32) and set(draws.ravel())<=set(info['fit_indices'])
    out=f/f'seed{a.seed}';out.mkdir(exist_ok=False)
    torch.manual_seed(a.seed);model=ReferenceGRUPhaseSchedule(stats)
    fingerprint=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    assert fingerprint==json.loads((old/'run_protocol.json').read_text())['initial_parameter_sha256']
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);curve=[];start=time.monotonic()
    base.write(out/'run_protocol.json',dict(fold=a.fold,seed=a.seed,initial_parameter_sha256=fingerprint,
         draws_sha256=base.sha(old/'sample_draws.npz'),protocol_sha256=base.sha(OUT/'protocol.json')))
    for step,ids in enumerate(draws,1):
        opt.zero_grad(set_to_none=True);pred=model(**{k:v[ids] for k,v in x.items()})
        loss=base.loss_fn(pred,{k:v[ids] for k,v in y.items()},model);assert torch.isfinite(loss)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        if step==1 or step%200==0:
            row=dict(step=step,loss=float(loss.detach()),gradient_norm=float(norm),elapsed_s=time.monotonic()-start);curve.append(row)
            base.write(out/'status.json',dict(status='running',**row));pd.DataFrame(curve).to_csv(out/'training_curve.csv',index=False)
    checkpoint=out/'final.pt';torch.save(dict(state_dict=model.state_dict(),steps=1600,fold=a.fold,seed=a.seed,
         reference_contract='rate3_thrust_reset_v23',protocol_sha256=base.sha(OUT/'protocol.json')),checkpoint)
    metrics=[];waves=[]
    for name,ids in [('fit',info['fit_indices']),('oof',info['query_indices'])]:
        pred=predict(model.eval(),x,ids);np.savez_compressed(out/f'{name}_predictions.npz',array_index=ids,**pred)
        f1,w1=metric_frames(pred,{k:v[ids].numpy() for k,v in y.items()},w.iloc[ids],'rate',a.seed)
        f1['cohort']=name;w1['cohort']=name;metrics.append(f1);waves.append(w1)
    held=dict(x);held['future_reference']=x['future_reference'].clone()
    held['future_reference'][:,:,:4]=x['future_reference'][:,:1,:4]
    pred=predict(model.eval(),held,info['query_indices']);np.savez_compressed(out/'oof_held.npz',array_index=info['query_indices'],**pred)
    reload=ReferenceGRUPhaseSchedule(stats);reload.load_state_dict(torch.load(checkpoint,weights_only=False)['state_dict']);reload.eval()
    ids=info['query_indices'][:2];a1=predict(model,x,ids);a2=predict(reload,x,ids)
    for k in a1:np.testing.assert_array_equal(a1[k],a2[k])
    altered=dict(x);altered['future_reference']=x['future_reference'].clone();altered['future_reference'][:,10:,:3]+=.1
    b=predict(reload,altered,ids)
    for k in a1:np.testing.assert_array_equal(a1[k][:,:10],b[k][:,:10])
    pd.concat(metrics).to_csv(out/'per_flight.csv',index=False);pd.concat(waves).to_csv(out/'waveforms.csv',index=False)
    base.write(out/'status.json',dict(status='complete',step=1600,elapsed_s=time.monotonic()-start,
         checkpoint_sha256=base.sha(checkpoint),exact_init_and_draws=True,reload_and_prefix_causality=True))


if __name__=='__main__':main()
