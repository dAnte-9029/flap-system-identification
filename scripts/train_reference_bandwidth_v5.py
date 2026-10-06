"""Fixed-budget small-cohort or gated full-cohort bandwidth experiments."""
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from reference_bandwidth_v5 import OUT,load,forecast_loss,as_50hz,causal_expand_50hz
from train_reference_unit_gravity import model_from,ART
from control_windows_v3 import predict
from system_identification.models.reference_gru_unit_gravity import ReferenceGRUUnitGravity
from evaluate_control_windows_v2 import error_rows,summarize


def metrics(pred,truth,history,meta,arm,seed):
    frames=[];wave=[]
    pp={k:torch.from_numpy(v) for k,v in pred.items()}
    for readout in ['raw50','aa50']:
        p,t=as_50hz(pp,truth,history,readout);p={k:v.numpy() for k,v in p.items()};t={k:v.numpy() for k,v in t.items()}
        f=summarize(error_rows(p,t,meta,arm,seed));f['readout']=readout;frames.append(f)
        rates=np.sqrt(np.mean((p['rates_b']-t['rates_b'])**2,axis=1))
        velocity=np.sqrt(np.mean((p['velocity_n']-t['velocity_n'])**2,axis=(1,2)))
        dot=abs(np.sum(p['quaternion']*t['quaternion'],axis=-1)).clip(0,1);attitude=np.sqrt(np.mean((2*np.arccos(dot))**2,axis=1))
        z=meta[['array_index','window_id','log_id','partition']].copy();z['arm']=arm;z['seed']=seed;z['readout']=readout
        for j,axis in enumerate(['p','q','r']):z[axis]=rates[:,j]
        z['velocity']=velocity;z['attitude']=attitude;wave.append(z)
    return pd.concat(frames),pd.concat(wave)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['raw100','aa50'],required=True);ap.add_argument('--seed',type=int,default=17);ap.add_argument('--formal',action='store_true');args=ap.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    x,y,stats,w=load();protocol=json.loads((OUT/'protocol.json').read_text());scope='formal' if args.formal else 'tiny'
    if args.formal:
        gate=json.loads((OUT/'tiny_decision.json').read_text());assert args.arm in gate['formal_arms'],'not authorized by frozen training-only gate'
        cohort=pd.read_csv(base.ROOT/'docs/analysis/results/control_windows_v3/unit_gravity/arm_A.csv');assert (cohort.partition=='train').all()
        indices=cohort.array_index.to_numpy();prob=cohort.sampling_probability.to_numpy();prob/=prob.sum();batch=32
    else:
        assert args.seed==17;cohort=w.set_index('window_id').loc[protocol['tiny_cohort']].reset_index();assert (cohort.partition=='train').all()
        indices=cohort.array_index.to_numpy();prob=None;batch=8
    out=OUT/scope/f'{args.arm}_{args.seed}';out.mkdir(parents=True,exist_ok=False)
    base.write(out/'run_protocol.json',dict(parent_protocol_sha256=base.sha(OUT/'protocol.json'),dataset_manifest_sha256=base.sha(OUT/'dataset_manifest.json'),
        script_sha256=base.sha(__file__),shared_sha256=base.sha(base.ROOT/'scripts/reference_bandwidth_v5.py'),
        model_sha256=base.sha(base.ROOT/'src/system_identification/models/reference_gru_unit_gravity.py'),
        filter_sha256=base.sha(base.ROOT/'src/system_identification/models/reference_antialias_readout.py'),
        arm=args.arm,scope=scope,seed=args.seed,steps=1600,batch=batch,train_window_ids=cohort.window_id.tolist(),normalization_source='originalAtrainonly',sealed_test_accessed=False))
    torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed);m=ReferenceGRUUnitGravity(stats);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4)
    curve=[];started=time.monotonic();base.write(out/'status.json',dict(status='running',arm=args.arm,scope=scope,seed=args.seed))
    for step in range(1,1601):
        ix=indices[rng.choice(len(indices),batch,replace=True,p=prob)] if args.formal else indices
        opt.zero_grad(set_to_none=True);loss=forecast_loss(m(**{k:v[ix] for k,v in x.items()}),{k:v[ix] for k,v in y.items()},m,x['history_state'][ix,:,3:6],args.arm)
        assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
        if step==1 or step%100==0:
            row=dict(step=step,loss=float(loss.detach()),gradient_norm=float(norm),elapsed_s=time.monotonic()-started);curve.append(row);print(json.dumps(row),flush=True)
    checkpoint=out/'final.pt';torch.save(dict(state_dict=m.state_dict(),feature_contract='unit_gravity_v1',arm=args.arm,scope=scope,seed=args.seed,steps=1600,
        inference_grid_hz=100,history_steps=51,horizon_steps=100,output_readout='raw100' if args.arm=='raw100' else 'causal_fir21_downsample50_delay100ms',protocol_sha256=base.sha(out/'run_protocol.json')),checkpoint)
    pd.DataFrame(curve).to_csv(out/'training_curve.csv',index=False);frames=[];waves=[]
    subsets=[('train',w[w.partition=='train']),('validation',w[w.partition=='validation'])] if args.formal else [('fit8',cohort)]
    for subset,meta in subsets:
        ix=meta.array_index.to_numpy();pp=predict(m.eval(),x,ix);np.savez_compressed(out/f'{subset}_predictions.npz',array_index=ix,**pp)
        f,wa=metrics(pp,{k:v[ix] for k,v in y.items()},x['history_state'][ix,:,3:6],meta,args.arm,args.seed);f['cohort']=subset;wa['cohort']=subset;frames.append(f);waves.append(wa)
    pd.concat(frames).to_csv(out/'per_flight.csv',index=False);pd.concat(waves).to_csv(out/'waveform_errors.csv',index=False)
    base.write(out/'status.json',dict(status='complete',arm=args.arm,scope=scope,seed=args.seed,elapsed_s=time.monotonic()-started,
        checkpoint_sha256=base.sha(checkpoint),updates=1600,sealed_test_accessed=False))

if __name__=='__main__':main()
