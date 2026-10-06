"""Conditional full-data phase candidate; entered only after the frozen OOF gate."""
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from reference_groupcv_v8 import OUT,PHASE,ReferenceGRUPhaseSchedule,crossfit_phase,phase_start,predict
from reference_capacity_v6 import metric_frames


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);args=ap.parse_args()
    decision=json.loads((OUT/'cv_decision.json').read_text());assert decision['cv_gate']
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    protocol=json.loads((OUT/'protocol.json').read_text());x,y,stats=base.load();w=pd.read_csv(base.DATA/'windows_v2.csv')
    fit=w[w.partition=='train'].array_index.to_numpy();query=w[w.partition=='validation'].array_index.to_numpy()
    with np.load(PHASE) as z:phase=z['phase']
    a,b,prov,parameters=crossfit_phase(x['history_state'].numpy(),phase,w.log_id.to_numpy(),fit,query)
    schedule=np.zeros((len(w),50),np.float32);schedule[fit]=phase_start(a);schedule[query]=phase_start(b);schedule=torch.tensor(schedule)
    out=OUT/'full'/f'phase_{args.seed}';out.mkdir(parents=True,exist_ok=False)
    base.write(out/'run_protocol.json',dict(seed=args.seed,decision_sha256=base.sha(OUT/'cv_decision.json'),script_sha256=base.sha(__file__),parent_protocol_sha256=base.sha(OUT/'protocol.json'),phase_provenance=prov))
    np.savez_compressed(out/'phase_schedules.npz',phase_start=schedule.numpy(),fit_indices=fit,query_indices=query)
    for n,pars in enumerate(parameters):np.savez_compressed(out/f'phase_predictor{n}.npz',**prov[n],**pars)
    torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed);model=ReferenceGRUPhaseSchedule(stats);opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    logs=w.iloc[fit].log_id.to_numpy();_,codes,counts=np.unique(logs,return_inverse=True,return_counts=True);prob=1/counts[codes];prob/=prob.sum()
    start=time.monotonic();curve=[];base.write(out/'status.json',dict(status='running',step=0))
    for step in range(1,protocol['steps']+1):
        ix=fit[rng.choice(len(fit),protocol['batch'],replace=True,p=prob)];batch={k:v[ix] for k,v in x.items()};batch['phase_start']=schedule[ix]
        opt.zero_grad(set_to_none=True);loss=base.loss_fn(model(**batch),{k:v[ix] for k,v in y.items()},model)
        assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        if step==1 or step%200==0:
            row=dict(step=step,loss=float(loss.detach()),gradient_norm=float(norm),elapsed_s=time.monotonic()-start);curve.append(row);print(json.dumps(row),flush=True)
            pd.DataFrame(curve).to_csv(out/'training_curve.csv',index=False);base.write(out/'status.json',dict(status='running',**row))
    ck=out/'final.pt';torch.save(dict(state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),arm='phase',seed=args.seed,steps=protocol['steps'],feature_contract='unit_gravity_v1'),ck)
    frames=[]
    for name,ids in [('fit',fit),('development',query)]:
        p=predict(model.eval(),x,ids,schedule);np.savez_compressed(out/f'{name}_predictions.npz',array_index=ids,**p)
        f,_=metric_frames(p,{k:v[ids].numpy() for k,v in y.items()},w.iloc[ids],'phase',args.seed);f['cohort']=name;frames.append(f)
    pd.concat(frames).to_csv(out/'per_flight.csv',index=False);base.write(out/'status.json',dict(status='complete',checkpoint_sha256=base.sha(ck),elapsed_s=time.monotonic()-start))

if __name__=='__main__':main()
