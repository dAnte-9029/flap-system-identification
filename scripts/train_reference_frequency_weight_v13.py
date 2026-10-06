"""One paired fixed-budget q-weighting run or a unit-weight reproduction sentinel."""
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from reference_groupcv_v8 import load_train,predict,ReferenceGRUPhaseSchedule
from reference_capacity_v6 import metric_frames
from reference_frequency_weight_v13 import weighted_q_loss
from prepare_reference_frequency_weight_v13 import OUT,V8
from run_reference_generalization_v7 import verify_manifest


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--identity',action='store_true');args=ap.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    verify_manifest(OUT/'protocol.json',['source_pins']);x,y,w=load_train();source=V8/f'fold{args.fold}';parent=source/f'hold_{args.seed}'
    info=json.loads((OUT/f'fold{args.fold}/manifest.json').read_text());fit=np.array(info['fit_indices']);query=np.array(info['query_indices'])
    assert base.sha(OUT/f'fold{args.fold}/weights.npz')==info['weights_sha256']
    source_info=json.loads((source/'manifest.json').read_text());assert base.sha(source/'normalization.npz')==source_info['normalization_sha256']
    with np.load(source/'normalization.npz') as z:stats={k:z[k] for k in z.files}
    with np.load(parent/'sample_draws.npz') as z:draws=z['array_index']
    assert draws.shape==(1600,32) and np.isin(draws,fit).all()
    with np.load(OUT/f'fold{args.fold}/weights.npz') as z:weight=torch.ones(len(w));weight[fit]=torch.tensor(z['weight'],dtype=torch.float32)
    if args.identity:weight.fill_(1.)
    dest=OUT/f'fold{args.fold}'/('identity' if args.identity else f'seed{args.seed}');dest.mkdir(exist_ok=False)
    torch.manual_seed(args.seed);model=ReferenceGRUPhaseSchedule(stats);opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    fingerprint=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    original=json.loads((parent/'run_protocol.json').read_text());assert fingerprint==original['initial_parameter_sha256']
    base.write(dest/'run_protocol.json',dict(fold=args.fold,seed=args.seed,identity=args.identity,initial_parameter_sha256=fingerprint,source_draws_sha256=base.sha(parent/'sample_draws.npz'),protocol_sha256=base.sha(OUT/'protocol.json'),weights_sha256=info['weights_sha256']))
    start=time.monotonic();curve=[]
    for step,ix in enumerate(draws,1):
        opt.zero_grad(set_to_none=True);pred=model(**{k:v[ix] for k,v in x.items()});loss=weighted_q_loss(pred,{k:v[ix] for k,v in y.items()},model,weight[ix]);assert torch.isfinite(loss)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        if step==1 or step%200==0:
            row=dict(step=step,loss=float(loss.detach()),gradient_norm=float(norm),elapsed_s=time.monotonic()-start);curve.append(row);print(json.dumps(row),flush=True)
            pd.DataFrame(curve).to_csv(dest/'training_curve.csv',index=False);base.write(dest/'status.json',dict(status='running',**row))
    torch.save(dict(state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),fold=args.fold,seed=args.seed,steps=1600,identity=args.identity,feature_contract='unit_gravity_v1',protocol_sha256=base.sha(dest/'run_protocol.json')),dest/'final.pt')
    np.savez_compressed(dest/'sample_draws.npz',array_index=draws)
    frames=[];waves=[]
    for cohort,ids in [('fit',fit),('oof',query)]:
        pred=predict(model.eval(),x,ids);np.savez_compressed(dest/f'{cohort}_predictions.npz',array_index=ids,**pred)
        f,wa=metric_frames(pred,{k:v[ids].numpy() for k,v in y.items()},w.iloc[ids],'weighted',args.seed)
        for df in [f,wa]:df['cohort']=cohort;df['fold']=args.fold
        frames.append(f);waves.append(wa)
    pd.concat(frames).to_csv(dest/'per_flight.csv',index=False);pd.concat(waves).to_csv(dest/'waveforms.csv',index=False)
    extra={}
    if args.identity:
        old=torch.load(parent/'final.pt',map_location='cpu',weights_only=False)['state_dict'];err=max(float((model.state_dict()[k]-v).abs().max()) for k,v in old.items());extra['identity_parameter_max_abs']=err
        assert err<1e-6,err
    base.write(dest/'status.json',dict(status='complete',step=1600,elapsed_s=time.monotonic()-start,checkpoint_sha256=base.sha(dest/'final.pt'),**extra))

if __name__=='__main__':main()
