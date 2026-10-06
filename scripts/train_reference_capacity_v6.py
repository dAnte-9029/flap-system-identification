"""Fixed trajectory training; preserve optimizer through budget extension."""
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from reference_capacity_v6 import OUT,CLASSES,metric_frames
from control_windows_v3 import predict
from train_reference_unit_gravity import ART


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['integrated','direct'],required=True);ap.add_argument('--seed',type=int,default=17);ap.add_argument('--formal',action='store_true');args=ap.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    protocol=json.loads((OUT/'protocol.json').read_text())
    for p,digest in protocol['source_pins'].items():assert base.sha(base.ROOT/p)==digest,p
    x,y,stats=base.load();w=pd.read_csv(base.DATA/'windows_v2.csv');scope='formal' if args.formal else 'tiny'
    if args.formal:
        assert args.arm in json.loads((OUT/'tiny_decision.json').read_text())['formal_arms']
        cohort=pd.read_csv(base.ROOT/'docs/analysis/results/control_windows_v3/unit_gravity/arm_A.csv');prob=cohort.sampling_probability.to_numpy();prob/=prob.sum()
    else:
        assert args.seed==17;cohort=w.set_index('window_id').loc[protocol['tiny_cohort']].reset_index();prob=None
    assert (cohort.partition=='train').all();ids=cohort.array_index.to_numpy()
    out=OUT/scope/f'{args.arm}_{args.seed}';out.mkdir(parents=True,exist_ok=False)
    base.write(out/'run_protocol.json',dict(arm=args.arm,seed=args.seed,scope=scope,parent_protocol_sha256=base.sha(OUT/'protocol.json'),script_sha256=base.sha(__file__),
        model_sha256=base.sha(base.ROOT/('src/system_identification/models/reference_gru_direct_rate.py' if args.arm=='direct' else 'src/system_identification/models/reference_gru_unit_gravity.py')),
        window_ids=cohort.window_id.tolist(),probability=prob.tolist() if prob is not None else 'all8 every update'))
    torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed);model=CLASSES[args.arm](stats);opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    curve=[];completed=[];started=time.monotonic()
    base.write(out/'status.json',dict(status='running',completed=completed))
    for step in range(1,6401):
        ix=ids[rng.choice(len(ids),32,replace=True,p=prob)] if args.formal else ids
        model.train();opt.zero_grad(set_to_none=True);loss=base.loss_fn(model(**{k:v[ix] for k,v in x.items()}),{k:v[ix] for k,v in y.items()},model)
        assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        if step==1 or step%200==0:
            row=dict(step=step,loss=float(loss.detach()),gradient_norm=float(norm),elapsed_s=time.monotonic()-started);curve.append(row);print(json.dumps(row),flush=True)
        if step in [1600,6400]:
            checkpoint=out/f'step_{step}.pt';torch.save(dict(state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),numpy_rng_state=rng.bit_generator.state,
                torch_rng_state=torch.get_rng_state(),arm=args.arm,scope=scope,seed=args.seed,steps=step,feature_contract='unit_gravity_v1',protocol_sha256=base.sha(out/'run_protocol.json')),checkpoint)
            reproduced=None
            if args.arm=='integrated' and step==1600:
                old=ART/f'A_long_{args.seed}/final.pt' if args.formal else base.ROOT/'docs/analysis/results/control_windows_v4/tiny_fit.pt'
                ck=torch.load(old,map_location='cpu',weights_only=True);reproduced=max(float((v-ck['state_dict'][k]).abs().max()) for k,v in model.state_dict().items());assert reproduced<1e-6,reproduced
            frames=[];waves=[];subsets=[('train',w[w.partition=='train']),('validation',w[w.partition=='validation'])] if args.formal else [('fit8',cohort)]
            for subset,meta in subsets:
                ev=meta.array_index.to_numpy();pred=predict(model.eval(),x,ev);np.savez_compressed(out/f'{step}_{subset}_predictions.npz',array_index=ev,**pred)
                truth={k:v[ev].numpy() for k,v in y.items()};f,wa=metric_frames(pred,truth,meta,args.arm,args.seed);f['cohort']=subset;wa['cohort']=subset;f['steps']=step;wa['steps']=step;frames.append(f);waves.append(wa)
            pd.concat(frames).to_csv(out/f'{step}_per_flight.csv',index=False);pd.concat(waves).to_csv(out/f'{step}_waveform.csv',index=False)
            completed.append(dict(step=step,checkpoint_sha256=base.sha(checkpoint),reproduction_max_abs=reproduced));pd.DataFrame(curve).to_csv(out/'training_curve.csv',index=False)
            base.write(out/'status.json',dict(status='running',completed=completed,elapsed_s=time.monotonic()-started))
    base.write(out/'status.json',dict(status='complete',completed=completed,elapsed_s=time.monotonic()-started))
if __name__=='__main__':main()
