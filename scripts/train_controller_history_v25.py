"""One of 30 same-budget paired runs; no future controller observations."""
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from controller_history_training_v25 import OUT,V23,V8,ARMS,model,load,fingerprint
from reference_groupcv_v8 import predict


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--arm',choices=ARMS,required=True);a=ap.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    p=json.loads((OUT/'protocol.json').read_text())
    for f,h in p['code_pins'].items():assert base.sha(base.ROOT/f)==h,f
    x,y=load();info=json.loads((V23/f'fold{a.fold}/manifest.json').read_text())
    old=V8/f'fold{a.fold}/hold_{a.seed}';st=json.loads((old/'status.json').read_text())
    assert base.sha(old/'sample_draws.npz')==st['draws_sha256']
    with np.load(old/'sample_draws.npz') as z:draws=z['array_index']
    assert draws.shape==(1600,32) and set(draws.ravel())<=set(info['fit_indices'])
    torch.manual_seed(a.seed);m=model(a.fold,a.arm)
    assert fingerprint(m,True)==json.loads((old/'run_protocol.json').read_text())['initial_parameter_sha256']
    assert not m.encoder.weight_ih_l0[:,25:].detach().count_nonzero()
    dest=OUT/f'fold{a.fold}/{a.arm}/seed{a.seed}';dest.mkdir(parents=True,exist_ok=False)
    base.write(dest/'run_protocol.json',dict(fold=a.fold,seed=a.seed,arm=a.arm,initial_parameters_sha256=fingerprint(m),
        common_initial_parameters_sha256=fingerprint(m,True),draws_sha256=base.sha(old/'sample_draws.npz'),
        parent_protocol_sha256=base.sha(OUT/'protocol.json')))
    opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);curve=[];start=time.monotonic()
    for step,ids in enumerate(draws,1):
        opt.zero_grad(set_to_none=True);pred=m(**{k:v[ids] for k,v in x.items()})
        loss=base.loss_fn(pred,{k:v[ids] for k,v in y.items()},m);assert torch.isfinite(loss)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
        if step==1 or step%200==0:
            row=dict(step=step,loss=float(loss.detach()),gradient_norm=float(norm),elapsed_s=time.monotonic()-start);curve.append(row)
            base.write(dest/'status.json',dict(status='running',**row));pd.DataFrame(curve).to_csv(dest/'training_curve.csv',index=False)
    if a.arm=='zero_history_control':assert not m.encoder.weight_ih_l0[:,25:].detach().count_nonzero()
    path=dest/'final.pt';torch.save(dict(state_dict=m.state_dict(),fold=a.fold,seed=a.seed,arm=a.arm,steps=1600,
        protocol_sha256=base.sha(OUT/'protocol.json'),feature_contract='rate_reference_past_controller_v25'),path)
    for cohort,ids in [('fit',info['fit_indices']),('oof',info['query_indices'])]:
        pred=predict(m.eval(),x,ids);np.savez_compressed(dest/f'{cohort}_predictions.npz',array_index=ids,**pred)
    held=dict(x);held['future_reference']=x['future_reference'].clone();held['future_reference'][:,:,:4]=x['future_reference'][:,:1,:4]
    pred=predict(m.eval(),held,info['query_indices']);np.savez_compressed(dest/'oof_held.npz',array_index=info['query_indices'],**pred)
    reload=model(a.fold,a.arm);reload.load_state_dict(torch.load(path,weights_only=False)['state_dict']);reload.eval()
    ids=info['query_indices'][:2];original=predict(m,x,ids);restored=predict(reload,x,ids)
    for k in original:np.testing.assert_array_equal(original[k],restored[k])
    altered=dict(x);altered['future_reference']=x['future_reference'].clone();altered['future_reference'][:,10:,:3]+=.1
    suffix=predict(reload,altered,ids)
    for k in original:np.testing.assert_array_equal(original[k][:,:10],suffix[k][:,:10])
    base.write(dest/'status.json',dict(status='complete',step=1600,elapsed_s=time.monotonic()-start,
       checkpoint_sha256=base.sha(path),initialization_and_draws_checked=True,reload_and_prefix_causality=True,
       extra_encoder_weight_norm=float(torch.linalg.vector_norm(m.encoder.weight_ih_l0[:,25:]).detach())))


if __name__=='__main__':main()
