"""Fixed initialization and minibatch tape, first20ms supervision only."""
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from reference_groupcv_v8 import load_train,predict,ReferenceGRUPhaseSchedule
from reference_single_step_v17 import OUT,V8,first_step_inputs,first_step_loss
from run_reference_generalization_v7 import verify_manifest


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,required=True);ap.add_argument('--seed',type=int,required=True);args=ap.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    verify_manifest(OUT/'protocol.json',['source_pins']);x,y,w=load_train()
    src=V8/f'fold{args.fold}';parent=src/f'hold_{args.seed}';info=json.loads((src/'manifest.json').read_text())
    fit=np.array(info['fit_indices']);query=np.array(info['query_indices'])
    assert set(w.iloc[fit].log_id).isdisjoint(w.iloc[query].log_id)
    assert base.sha(src/'normalization.npz')==info['normalization_sha256']
    with np.load(src/'normalization.npz') as z:stats={k:z[k] for k in z.files}
    with np.load(parent/'sample_draws.npz') as z:draws=z['array_index']
    assert draws.shape==(1600,32) and np.isin(draws,fit).all()
    dest=OUT/f'fold{args.fold}'/f'seed{args.seed}';dest.mkdir(exist_ok=False)
    torch.manual_seed(args.seed);model=ReferenceGRUPhaseSchedule(stats)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    fingerprint=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    assert fingerprint==json.loads((parent/'run_protocol.json').read_text())['initial_parameter_sha256']
    assert sum(p.numel() for p in model.parameters())==40070
    batch={k:v[draws[0]] for k,v in x.items()};target={k:v[draws[0]] for k,v in y.items()}
    full=first_step_loss(model(**batch),target,model)
    fg=torch.autograd.grad(full,tuple(model.parameters()))
    short=first_step_loss(model(**first_step_inputs(batch)),target,model)
    sg=torch.autograd.grad(short,tuple(model.parameters()))
    grad_diff=max(float((a-b).abs().max()) for a,b in zip(fg,sg));assert grad_diff<1e-6
    assert abs(float(full.detach()-short.detach()))<1e-7
    base.write(dest/'run_protocol.json',dict(fold=args.fold,seed=args.seed,initial_parameter_sha256=fingerprint,
        source_draws_sha256=base.sha(parent/'sample_draws.npz'),normalization_sha256=base.sha(src/'normalization.npz'),
        source_fold_manifest_sha256=base.sha(src/'manifest.json'),fit_indices=fit.tolist(),query_indices=query.tolist(),
        protocol_sha256=base.sha(OUT/'protocol.json'),initial_full_short_gradient_max_abs=grad_diff))
    start=time.monotonic();curve=[]
    for step,ix in enumerate(draws,1):
        opt.zero_grad(set_to_none=True)
        pred=model(**first_step_inputs({k:v[ix] for k,v in x.items()}))
        loss=first_step_loss(pred,{k:v[ix] for k,v in y.items()},model)
        assert torch.isfinite(loss);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        if step==1 or step%200==0:
            row=dict(step=step,loss=float(loss.detach()),gradient_norm=float(norm),elapsed_s=time.monotonic()-start)
            curve.append(row);pd.DataFrame(curve).to_csv(dest/'training_curve.csv',index=False)
            base.write(dest/'status.json',dict(status='running',**row));print(json.dumps(row),flush=True)
    torch.save(dict(state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),torch_rng_state=torch.get_rng_state(),
        fold=args.fold,seed=args.seed,steps=1600,feature_contract='unit_gravity_v1',train_unroll_steps=1,eval_rollout_steps=50,
        protocol_sha256=base.sha(dest/'run_protocol.json')),dest/'final.pt')
    np.savez_compressed(dest/'sample_draws.npz',array_index=draws)
    for cohort,ids in [('fit',fit),('oof',query)]:
        pred=predict(model.eval(),x,ids)
        np.savez_compressed(dest/f'{cohort}_predictions.npz',array_index=ids,**pred)
    base.write(dest/'status.json',dict(status='complete',step=1600,elapsed_s=time.monotonic()-start,
        checkpoint_sha256=base.sha(dest/'final.pt'),all_predictions_finite=True))


if __name__=='__main__':main()
