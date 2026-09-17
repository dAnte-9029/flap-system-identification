"""Additional objectives only; the frozen architecture and integration are unchanged."""
import torch
from system_identification.training.trajectory_main_v1 import trajectory_rollout_loss


def loss_components(prediction, truth, steps):
    s=slice(1,steps+1)
    pq=prediction.quaternion_nb[:,s];tq=truth.quaternion_nb[:,s]
    pq=pq/torch.linalg.vector_norm(pq,dim=-1,keepdim=True).clamp_min(1e-8)
    tq=tq/torch.linalg.vector_norm(tq,dim=-1,keepdim=True).clamp_min(1e-8)
    return dict(position=(prediction.position_n[:,s]-truth.position_n[:,s]).square().sum(-1).mean(),
        velocity=((prediction.velocity_n[:,s]-truth.velocity_n[:,s])/2).square().sum(-1).mean(),
        attitude=(4*(1-(pq*tq).sum(-1).square())/(.35**2)).mean(),
        body_rate=((prediction.angular_velocity_b[:,s]-truth.angular_velocity_b[:,s])/2).square().sum(-1).mean(),
        phase=.1*(2-2*torch.cos(prediction.relative_phase_rad[:,s]-truth.relative_phase_rad[:,s])).mean(),
        frequency=.1*((prediction.flap_frequency_hz[:,s]-truth.flap_frequency_hz[:,s])/3).square().mean())


def objective(prediction,truth,*,steps,prefix_weights=(),delta_lag=1,delta_weight=0.):
    if delta_lag<1 or delta_lag>steps or delta_weight<0:
        raise ValueError('invalid delta loss parameters')
    if prefix_weights:
        if max(k for k,w in prefix_weights)!=steps or any(k<1 or w<=0 for k,w in prefix_weights) or abs(sum(w for k,w in prefix_weights)-1)>1e-6:
            raise ValueError('invalid prefix weights')
        loss=sum(w*trajectory_rollout_loss(prediction,truth,objective_steps=k) for k,w in prefix_weights)
    else:
        loss=trajectory_rollout_loss(prediction,truth,objective_steps=steps)
    # Aligned vector increments, not amplitude reward: noise/phase errors are penalized.
    if delta_weight:
        p=prediction.angular_velocity_b[:,:steps+1];t=truth.angular_velocity_b[:,:steps+1]
        d=(p[:,delta_lag:]-p[:,:-delta_lag])-(t[:,delta_lag:]-t[:,:-delta_lag])
        loss=loss+delta_weight*(d/2).square().sum(-1).mean()
    return loss


def train_stage(model,batch,*,steps,epochs,seed,learning_rate,device,
                prefix_weights=(),delta_lag=1,delta_weight=0.,actuator=False,
                batch_size=256,callback=None):
    """Same AdamW/order/budget as legacy fitting; only trajectory objective varies."""
    import time
    import numpy as np
    import pandas as pd
    from system_identification.training.trajectory_main_v1 import _model_call
    model=model.to(device)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=learning_rate,weight_decay=1e-5)
    generator=torch.Generator().manual_seed(seed)
    count=len(batch.trajectory.window_ids);history=[];started=time.monotonic()
    if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(epochs):
        indices=torch.randperm(count,generator=generator).numpy();totals=0.;maxnorm=0.
        model.train()
        for start in range(0,count,batch_size):
            selected=indices[start:start+batch_size];optimizer.zero_grad(set_to_none=True)
            p,t=_model_call(model,batch,selected,use_history=True,rollout_steps=steps,device=torch.device(device))
            loss=objective(p,t,steps=steps,prefix_weights=prefix_weights,delta_lag=delta_lag,delta_weight=delta_weight)
            if actuator:
                drive=(p.flap_frequency_hz[:,1:steps+1]-t.flap_frequency_hz[:,1:steps+1]).square().mean()
                loss=loss+.2*drive+model.control_regularization_loss(residual_l2=1e-3,tail_gate_l1=1e-2)
            if not torch.isfinite(loss):raise ValueError('nonfinite objective')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,5.)
            if not torch.isfinite(norm):raise ValueError('nonfinite gradient')
            optimizer.step();totals+=float(loss.detach())*len(selected);maxnorm=max(maxnorm,float(norm))
        elapsed=time.monotonic()-started
        row=dict(epoch=epoch+1,loss=totals/count,gradient_norm_max=maxnorm,wall_time_s=elapsed,
            windows_per_second=(epoch+1)*count/elapsed,optimizer_steps=(epoch+1)*int(np.ceil(count/batch_size)),
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else 0)
        history.append(row)
        if callback:callback(model,pd.DataFrame(history))
    return model.cpu().eval(),pd.DataFrame(history)
