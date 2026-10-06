"""Frozen v42 tilt-supervision recipe; zero control here, candidates in v43."""
import argparse
import time
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from train_joint_increment_v34 import load_inputs,read,DATA,PREP as V34,ActuatorCommandGRU,fingerprint
from joint_increment_v34 import select_local
from attitude_supervision_v42 import objective
from reference_groupcv_v8 import predict
from audit_controller_support_v26 import verify_parent

PREP=base.ROOT/'docs/analysis/results/control_windows_v42'
OUT=PREP.parent/'control_windows_v43'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fold',type=int,choices=range(5),required=True)
    parser.add_argument('--seed',type=int,choices=[17,23,42],required=True);parser.add_argument('--zero-control',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    protocol=read(PREP/'protocol.json')
    for name,digest in protocol['code'].items():assert base.sha(base.ROOT/name)==digest
    run=next(r for r in read(PREP/'training_plan.json')['runs'] if (r['fold'],r['seed'])==(args.fold,args.seed))
    if args.zero_control:
        assert (args.fold,args.seed)==(0,17);dest=PREP/'zero_control'
        assert read(PREP/'kinematics_verification.json')['preparation_gate']
    else:
        verify_parent(PREP)
        assert read(PREP/'zero_weight_reproduction.json')['status']=='passed'
        assert read(OUT/'protocol.json')['preparation_delivery_sha256']==base.sha(PREP/'delivery_manifest.json')
        dest=OUT/f'fold{args.fold}/seed{args.seed}'
    dest.mkdir(parents=True,exist_ok=False)
    weight=0. if args.zero_control else run['tilt_weight']
    assert base.sha(base.ROOT/run['sample_plan'])==run['sample_plan_sha256']
    assert base.sha(DATA/f'fold{args.fold}/normalization.npz')==run['normalization_sha256']
    assert base.sha(base.ROOT/run['baseline_checkpoint'])==run['baseline_checkpoint_sha256']
    with np.load(base.ROOT/run['sample_plan']) as z:draws=z['array_index'];offsets=z['offset']
    with np.load(DATA/f'fold{args.fold}/normalization.npz') as z:stats={k:z[k] for k in z.files}
    info=read(DATA/f'fold{args.fold}/manifest.json');fit=info['fit_indices'];assert set(draws.ravel())<=set(fit)
    x,y,lx,ly=load_inputs();torch.manual_seed(args.seed);model=ActuatorCommandGRU(stats)
    assert fingerprint(model)==run['initial_parameter_sha256']
    assert sum(p.numel() for p in model.parameters())==40070
    base.write(dest/'run_protocol.json',dict(**run,executed_weight=weight,preparation_protocol_sha256=base.sha(PREP/'protocol.json')))
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);rows=[];start=time.monotonic()
    for step,(ids,offs) in enumerate(zip(draws,offsets),1):
        bx={k:v[ids] for k,v in x.items()};by={k:v[ids] for k,v in y.items()}
        local_x,local_y=select_local(lx,ly,ids,offs,fit)
        optimizer.zero_grad(set_to_none=True)
        loss,legacy,tilt=objective(model,bx,by,local_x,local_y,weight)
        assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if step==1 or step%200==0:
            row=dict(step=step,loss=float(loss.detach()),v35_loss=float(legacy.detach()),tilt=None if tilt is None else float(tilt.detach()),gradient_norm=float(norm),elapsed_s=time.monotonic()-start)
            rows.append(row);pd.DataFrame(rows).to_csv(dest/'training_curve.csv',index=False)
            base.write(dest/'status.json',dict(status='running',**row));print(row,flush=True)
    ck=dest/'final.pt';torch.save(dict(state_dict=model.state_dict(),fold=args.fold,seed=args.seed,steps=1600,
        input_contract='post_allocation_motor0_servo012_v29',tilt_weight=weight,local_weight=1.,protocol_sha256=base.sha(PREP/'protocol.json')),ck)
    if args.zero_control:
        reference=torch.load(base.ROOT/run['baseline_checkpoint'],map_location='cpu',weights_only=False)['state_dict']
        for key,value in model.state_dict().items():torch.testing.assert_close(value,reference[key],rtol=0,atol=0)
        base.write(PREP/'zero_weight_reproduction.json',dict(status='passed',steps=1600,
            all_parameters_buffers_exact=True,baseline_checkpoint=run['baseline_checkpoint'],
            baseline_sha256=run['baseline_checkpoint_sha256'],reproduction_sha256=base.sha(ck),candidate_training_started=False))
    else:
        for cohort,ids in [('fit',fit),('oof',info['query_indices'])]:
            np.savez_compressed(dest/f'{cohort}_predictions.npz',array_index=ids,**predict(model.eval(),x,ids))
    base.write(dest/'status.json',dict(status='complete',step=1600,checkpoint_sha256=base.sha(ck),elapsed_s=time.monotonic()-start,tilt_weight=weight))


if __name__=='__main__':main()
