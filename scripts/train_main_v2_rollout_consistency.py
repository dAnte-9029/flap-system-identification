#!/usr/bin/env python3
"""Step 8 fixed-budget training. S0 is the exact legacy objective/schedule."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys,json,time,argparse,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from train_main_v2_increment import inputs
from run_main_v2_free_running import sha
from system_identification.models.trajectory_main_v1 import CausalHistoryTrajectoryModel
from system_identification.models.trajectory_main_v2 import ActuatorAwareTrajectoryModel
from system_identification.training.main_v2_increment import train_increment_stage

MATRIX={'S0':(0,0.),'R1':(5,.1),'R2':(10,.1),'R3':(20,.1),'R3-high':(20,.5),'R3-max':(20,1.)}


def state_hash(state,keys=None):
    digest=hashlib.sha256()
    for name in sorted(state if keys is None else keys):
        value=state[name].detach().cpu().contiguous()
        digest.update(name.encode());digest.update(str(value.dtype).encode());digest.update(str(tuple(value.shape)).encode());digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--artifacts',type=Path,required=True)
    ap.add_argument('--experiment',choices=list(MATRIX),required=True);ap.add_argument('--device',default='cuda:1');ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    if not args.device.startswith('cuda') or not torch.cuda.is_available():raise RuntimeError('GPU required; no CPU fallback')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    folder=args.artifacts/args.experiment
    if folder.exists():raise FileExistsError(folder)
    folder.mkdir(parents=True);args.results.mkdir(parents=True,exist_ok=True)
    batch,stats,tail,old=inputs();steps,weight=MATRIX[args.experiment]
    if args.experiment!='S0':
        protocol=json.loads((args.results/'protocol.json').read_text())
        if protocol['objective_contract']!='legacy50_plus_short_prefix':raise ValueError('explicit objective contract required')
        from system_identification.training.rollout_consistency import train_consistency_stage
    stage='base';started=time.monotonic()
    def callback(model,h):
        row=dict(stage=stage,status='running',experiment=args.experiment,**h.iloc[-1].to_dict())
        (folder/'status.json').write_text(json.dumps(row,indent=2))
        if len(h)%5==0:print(json.dumps(row),flush=True)
    def train(model,actuator=False):
        kw=dict(device=args.device,epochs=(1 if args.smoke else (25 if actuator else 40)),seed=29 if actuator else 17,
                learning_rate=.0005 if actuator else .0003,actuator=actuator,callback=callback)
        if steps==0:return train_increment_stage(model,batch,scales=[1.,1.],weights=[0.,0.],**kw)
        return train_consistency_stage(model,batch,horizon=steps,rollout_weight=weight,**kw)
    torch.manual_seed(17);torch.cuda.manual_seed_all(17)
    model=CausalHistoryTrajectoryModel(hidden_size=64,use_controls=False,**vars(stats))
    # Foreground smoke uses same real rows and transition. No data or split change.
    model,hb=train(model);torch.save(dict(state_dict=model.state_dict()),folder/'base.pt');hb.to_csv(folder/'base_history.csv',index=False)
    stage='actuator';torch.manual_seed(29);torch.cuda.manual_seed_all(29)
    model=ActuatorAwareTrajectoryModel(base_model=model,use_drive=True,use_tail=True,gated_tail=True,tail_mean=tail.tail_mean,tail_std=tail.tail_std)
    model,ha=train(model,True);ha.to_csv(folder/'actuator_history.csv',index=False)
    state=model.state_dict();parameter_names=list(dict(model.named_parameters()))
    config=dict(experiment=args.experiment,added_horizon=steps,lambda_roll=weight,base_epochs=1 if args.smoke else 40,
                actuator_epochs=1 if args.smoke else 25,base_seed=17,actuator_seed=29,training_steps=50,history=26,
                optimizer='AdamW',lr_base=.0003,lr_actuator=.0005,weight_decay=1e-5,gradient_clip=5.,checkpoint_selection='fixed final epoch',
                checkpoint_contract='unchanged original model',smoke=args.smoke)
    torch.save(dict(state_dict=state,config=config,tail_mean=tail.tail_mean,tail_std=tail.tail_std),folder/'model.pt')
    summary=dict(**config,wall_time_s=time.monotonic()-started,base_optimizer_steps=int(hb.iloc[-1].optimizer_steps),
                 actuator_optimizer_steps=int(ha.iloc[-1].optimizer_steps),base_samples_per_s=float(hb.iloc[-1].windows_per_second),
                 actuator_samples_per_s=float(ha.iloc[-1].windows_per_second),peak_gpu_bytes=int(max(hb.gpu_peak_allocated_bytes.max(),ha.gpu_peak_allocated_bytes.max())),
                 checkpoint_sha256=sha(folder/'model.pt'),state_hash=state_hash(state),parameter_hash=state_hash(state,parameter_names),device=args.device)
    if args.experiment=='S0' and not args.smoke:
        comparison=dict(old_state_hash=state_hash(old['state_dict']),new_state_hash=state_hash(state),
                        old_parameter_hash=state_hash(old['state_dict'],parameter_names),new_parameter_hash=state_hash(state,parameter_names),
                        all_state_tensors_equal=all(torch.equal(state[k],old['state_dict'][k]) for k in state),
                        max_parameter_difference=max(float((state[k]-old['state_dict'][k]).abs().max()) for k in parameter_names),
                        validation_benchmark='pending separate 722-origin evaluation',checkpoint_file_hash_note='serialization metadata differs; compare canonical tensor hashes')
        (args.results/'baseline_reproduction.json').write_text(json.dumps(comparison,indent=2))
        if not comparison['all_state_tensors_equal']:raise RuntimeError('S0 failed exact baseline reproduction; stop candidates')
    (folder/'training_summary.json').write_text(json.dumps(summary,indent=2));(folder/'status.json').write_text(json.dumps(dict(status='training_completed',**summary),indent=2))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
