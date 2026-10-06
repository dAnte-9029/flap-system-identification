"""Executable frozen recipe; v34 prepares, v35 holds subsequent trained outputs."""
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from audit_controller_support_v26 import verify_parent
from actuator_training_v30 import ActuatorCommandGRU, model_from
from prepare_actuator_inputs_v29 import fingerprint
from reference_groupcv_v8 import predict
from joint_increment_v34 import objective,select_local,LOCAL_WEIGHT

PREP=base.ROOT/'docs/analysis/results/control_windows_v34'
OUT=PREP.parent/'control_windows_v35'
DATA=PREP.parent/'control_windows_v29';LOCAL=PREP.parent/'control_windows_v33';CONTROL=PREP.parent/'control_windows_v30'


def read(p):return json.loads(p.read_text())


def verify_frozen():
    verify_parent(PREP)
    for name,digest in read(PREP/'protocol.json')['parents'].items():
        root=base.ROOT/name
        assert base.sha(root/'delivery_manifest.json')==digest
        verify_parent(root)


def load_inputs():
    def load(root,filename):
        assert base.sha(root/filename)==read(root/'dataset_manifest.json')['arrays_sha256']
        with np.load(root/filename) as z:
            x={k[6:]:torch.tensor(z[k]) for k in z.files if k.startswith('input_')}
            y={k[6:]:torch.tensor(z[k]) for k in z.files if k.startswith('label_')}
        return x,y
    x,y=load(DATA,'windows.npz');lx,ly=load(LOCAL,'rolling_inputs.npz')
    return x,y,lx,ly


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fold',type=int,choices=range(5),required=True)
    parser.add_argument('--seed',type=int,choices=[17,23,42],required=True)
    parser.add_argument('--zero-control',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    verify_frozen()
    execution=read(OUT/'protocol.json')
    assert execution['preparation_delivery_sha256']==base.sha(PREP/'delivery_manifest.json')
    run=next(r for r in read(PREP/'training_plan.json')['runs'] if (r['fold'],r['seed'])==(args.fold,args.seed))
    if args.zero_control:assert (args.fold,args.seed)==(0,17)
    else:assert read(OUT/'zero_weight_reproduction.json')['status']=='passed'
    info=read(DATA/f'fold{args.fold}/manifest.json')
    assert base.sha(base.ROOT/run['control_checkpoint'])==run['control_checkpoint_sha256']
    assert base.sha(DATA/f'fold{args.fold}/normalization.npz')==run['normalization_sha256']
    assert base.sha(PREP/run['sample_plan'])==run['sample_plan_sha256']
    with np.load(DATA/f'fold{args.fold}/normalization.npz') as z:stats={k:z[k] for k in z.files}
    with np.load(PREP/run['sample_plan']) as z:draws=z['array_index'];offsets=z['offset']
    assert draws.shape==offsets.shape==(1600,32) and set(draws.ravel())<=set(info['fit_indices'])
    x,y,lx,ly=load_inputs();torch.manual_seed(args.seed);model=ActuatorCommandGRU(stats)
    assert fingerprint(model)==run['initial_parameter_sha256']
    weight=0. if args.zero_control else LOCAL_WEIGHT
    arm='zero_weight_control' if args.zero_control else 'joint_increment'
    dest=OUT/f'fold{args.fold}/{arm}/seed{args.seed}';dest.mkdir(parents=True,exist_ok=False)
    base.write(dest/'run_protocol.json',dict(**run,weight=weight,execution_protocol_sha256=base.sha(OUT/'protocol.json'),
        preparation_protocol_sha256=base.sha(PREP/'protocol.json')))
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    rows=[];start=time.monotonic()
    for step,(ids,offs) in enumerate(zip(draws,offsets),1):
        full_x={k:v[ids] for k,v in x.items()};full_y={k:v[ids] for k,v in y.items()}
        local_x,local_y=(None,None) if weight==0 else select_local(lx,ly,ids,offs,info['fit_indices'])
        optimizer.zero_grad(set_to_none=True)
        loss,full_loss,local_loss=objective(model,full_x,full_y,local_x,local_y,weight=weight)
        assert torch.isfinite(loss);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if step==1 or step%200==0:
            row=dict(step=step,loss=float(loss.detach()),full_loss=float(full_loss.detach()),
                local_loss=None if local_loss is None else float(local_loss.detach()),gradient_norm=float(norm),elapsed_s=time.monotonic()-start)
            rows.append(row);pd.DataFrame(rows).to_csv(dest/'training_curve.csv',index=False)
            base.write(dest/'status.json',dict(status='running',**row));print(row,flush=True)
    checkpoint=dest/'final.pt'
    torch.save(dict(state_dict=model.state_dict(),steps=1600,fold=args.fold,seed=args.seed,
        input_contract='post_allocation_motor0_servo012_v29',local_weight=weight,protocol_sha256=base.sha(PREP/'protocol.json')),checkpoint)
    if args.zero_control:
        reference=torch.load(CONTROL/'fold0/seed17/final.pt',weights_only=False,map_location='cpu')['state_dict']
        for k,v in model.state_dict().items():torch.testing.assert_close(v,reference[k],rtol=0,atol=0)
        base.write(OUT/'zero_weight_reproduction.json',dict(status='passed',steps=1600,all_parameters_and_buffers_exact=True,
            checkpoint_sha256=base.sha(checkpoint),reference_sha256=base.sha(CONTROL/'fold0/seed17/final.pt')))
    else:
        for cohort,ids in [('fit',info['fit_indices']),('oof',info['query_indices'])]:
            pred=predict(model.eval(),x,ids);np.savez_compressed(dest/f'{cohort}_predictions.npz',array_index=ids,**pred)
        held=dict(x);held['future_command']=x['future_command'][:,:1].expand(-1,50,-1)
        np.savez_compressed(dest/'oof_held.npz',array_index=info['query_indices'],**predict(model,held,info['query_indices']))
        restored=model_from(checkpoint);ids=info['query_indices'][:2]
        a=predict(model,x,ids);b=predict(restored,x,ids)
        for k in a:np.testing.assert_array_equal(a[k],b[k])
    base.write(dest/'status.json',dict(status='complete',step=1600,elapsed_s=time.monotonic()-start,
        checkpoint_sha256=base.sha(checkpoint),local_weight=weight,paired_init_and_parent_draws=True))


if __name__=='__main__':main()
