"""Optional short-prefix weighting of the existing autonomous rollout objective.

The original Main V2 is already trained with 50 free-running transitions.
This module adds temporal emphasis, not a new teacher-forcing-free mechanism.
"""
import time
import numpy as np
import pandas as pd
import torch
from .main_v2_objectives import loss_components
from .trajectory_main_v1 import _model_call,trajectory_rollout_loss

GROUPS={'A':('position','velocity'),'B':('attitude','body_rate'),
        'C':('position','velocity','attitude','body_rate','frequency'),
        'priority':('velocity','attitude','body_rate')}


def short_prefix_loss(prediction,truth,horizon,group='priority'):
    if horizon not in (5,10,20) or group not in GROUPS:raise ValueError('invalid declared prefix/group')
    if prediction.position_n.shape[1]<=horizon or truth.position_n.shape[1]<=horizon:raise ValueError('insufficient rollout')
    components=loss_components(prediction,truth,horizon)
    return sum(components[key] for key in GROUPS[group])


def consistency_objective(prediction,truth,horizon=0,weight=0.):
    original=trajectory_rollout_loss(prediction,truth,objective_steps=50)
    if horizon==0 and weight==0:return original
    if weight not in (.1,.5,1.):raise ValueError('only predeclared lambda values allowed')
    return original+weight*short_prefix_loss(prediction,truth,horizon)


def train_consistency_stage(model,batch,*,horizon,rollout_weight,device,epochs,seed,learning_rate,
                            actuator=False,batch_size=256,callback=None):
    if horizon not in (5,10,20) or rollout_weight not in (.1,.5,1.):raise ValueError('invalid experiment')
    model=model.to(device);parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=learning_rate,weight_decay=1e-5)
    generator=torch.Generator().manual_seed(seed);count=len(batch.trajectory.window_ids)
    history=[];started=time.monotonic();torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(epochs):
        permutation=torch.randperm(count,generator=generator).numpy();totals={};maxnorm=0.;model.train()
        for start in range(0,count,batch_size):
            ids=permutation[start:start+batch_size];optimizer.zero_grad(set_to_none=True)
            p,t=_model_call(model,batch,ids,use_history=True,rollout_steps=50,device=torch.device(device))
            original=trajectory_rollout_loss(p,t,objective_steps=50)
            if actuator:
                original=original+.2*(p.flap_frequency_hz[:,1:51]-t.flap_frequency_hz[:,1:51]).square().mean()+model.control_regularization_loss(residual_l2=1e-3,tail_gate_l1=1e-2)
            components=loss_components(p,t,horizon)
            short=sum(components[key] for key in GROUPS['priority']);loss=original+rollout_weight*short
            if not torch.isfinite(loss):raise ValueError('nonfinite objective')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,5.)
            if not torch.isfinite(norm):raise ValueError('nonfinite gradients')
            optimizer.step();maxnorm=max(maxnorm,float(norm))
            values=dict(loss=loss,original_loss=original,short_prefix_loss=short,**{'short_'+k:v for k,v in components.items()})
            for k,v in values.items():totals[k]=totals.get(k,0.)+float(v.detach())*len(ids)
        elapsed=time.monotonic()-started
        row=dict(epoch=epoch+1,**{k:v/count for k,v in totals.items()},gradient_norm_max=maxnorm,wall_time_s=elapsed,
                 windows_per_second=(epoch+1)*count/elapsed,optimizer_steps=(epoch+1)*int(np.ceil(count/batch_size)),gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        history.append(row)
        if callback:callback(model,pd.DataFrame(history))
    return model.cpu().eval(),pd.DataFrame(history)
