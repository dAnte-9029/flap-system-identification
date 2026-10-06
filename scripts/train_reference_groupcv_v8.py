"""Paired fixed-budget fresh training for one v8 flight fold."""
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from reference_groupcv_v8 import OUT,load_train,predict,ReferenceGRUPhaseSchedule
from reference_capacity_v6 import metric_frames


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,required=True);ap.add_argument('--arm',choices=['hold','phase'],required=True);ap.add_argument('--seed',type=int,required=True);args=ap.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    protocol=json.loads((OUT/'protocol.json').read_text())
    for name,digest in protocol['source_pins'].items():assert base.sha(base.ROOT/name)==digest,name
    x,y,w=load_train();source=OUT/f'fold{args.fold}';fold=json.loads((source/'manifest.json').read_text())
    assert base.sha(source/'normalization.npz')==fold['normalization_sha256']
    assert base.sha(source/'phase_schedules.npz')==fold['phase_schedules_sha256']
    fit=np.array(fold['fit_indices']);query=np.array(fold['query_indices'])
    with np.load(source/'normalization.npz') as z:stats={k:z[k] for k in z.files}
    with np.load(source/'phase_schedules.npz') as z:schedule=torch.tensor(z['phase_start']) if args.arm=='phase' else None
    out=source/f'{args.arm}_{args.seed}';out.mkdir(exist_ok=False)
    torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed);model=ReferenceGRUPhaseSchedule(stats)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    fingerprint=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    logs=w.iloc[fit].log_id.to_numpy();_,codes,counts=np.unique(logs,return_inverse=True,return_counts=True);prob=1/counts[codes];prob/=prob.sum()
    base.write(out/'run_protocol.json',dict(fold=args.fold,arm=args.arm,seed=args.seed,initial_parameter_sha256=fingerprint,
        parent_protocol_sha256=base.sha(OUT/'protocol.json'),fold_manifest_sha256=base.sha(source/'manifest.json'),fit_indices=fit.tolist(),query_indices=query.tolist(),probabilities=prob.tolist()))
    curve=[];draws=[];start=time.monotonic();base.write(out/'status.json',dict(status='running',step=0))
    for step in range(1,protocol['steps']+1):
        ix=fit[rng.choice(len(fit),protocol['batch'],replace=True,p=prob)];draws.append(ix)
        opt.zero_grad(set_to_none=True);batch={k:v[ix] for k,v in x.items()}
        if schedule is not None:batch['phase_start']=schedule[ix]
        pred=model(**batch);loss=base.loss_fn(pred,{k:v[ix] for k,v in y.items()},model)
        assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        if step==1 or step%200==0:
            row=dict(step=step,loss=float(loss.detach()),gradient_norm=float(norm),elapsed_s=time.monotonic()-start);curve.append(row);print(json.dumps(row),flush=True)
            pd.DataFrame(curve).to_csv(out/'training_curve.csv',index=False);base.write(out/'status.json',dict(status='running',**row))
    checkpoint=out/'final.pt';torch.save(dict(state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),numpy_rng_state=rng.bit_generator.state,
        torch_rng_state=torch.get_rng_state(),fold=args.fold,arm=args.arm,seed=args.seed,steps=protocol['steps'],feature_contract='unit_gravity_v1',protocol_sha256=base.sha(out/'run_protocol.json')),checkpoint)
    np.savez_compressed(out/'sample_draws.npz',array_index=np.array(draws))
    frames=[];waves=[]
    for name,indices in [('fit',fit),('oof',query)]:
        pred=predict(model.eval(),x,indices,schedule);np.savez_compressed(out/f'{name}_predictions.npz',array_index=indices,**pred)
        f,wa=metric_frames(pred,{k:v[indices].numpy() for k,v in y.items()},w.iloc[indices],args.arm,args.seed)
        f['cohort']=name;wa['cohort']=name;f['fold']=args.fold;wa['fold']=args.fold;frames.append(f);waves.append(wa)
    pd.concat(frames).to_csv(out/'per_flight.csv',index=False);pd.concat(waves).to_csv(out/'waveforms.csv',index=False)
    base.write(out/'status.json',dict(status='complete',step=protocol['steps'],elapsed_s=time.monotonic()-start,checkpoint_sha256=base.sha(checkpoint),draws_sha256=base.sha(out/'sample_draws.npz')))

if __name__=='__main__':main()
