"""Passive derivative-head capture for frozen local angular diagnostics."""
import numpy as np
import torch


@torch.no_grad()
def capture_rollout(model, inputs):
    """Capture unclipped normalized outputs without changing inference."""
    raw=[]
    def capture(module, args, output):
        raw.append(output.detach().clone())
    handle=model.derivative_head.register_forward_hook(capture)
    try:
        prediction=model(**inputs)
    finally:
        handle.remove()
    outputs=torch.stack(raw,dim=1)
    if outputs.shape[1]!=inputs['dt_s'].shape[1]:raise ValueError('Derivative capture count mismatch')
    return prediction,outputs


def angular_metrics(error):
    """Vector RMSE and signed axis bias; retain nonfinite failure as infinity."""
    error=np.asarray(error)
    finite=np.isfinite(error).all(axis=1)
    safe=np.where(np.isfinite(error),error,np.inf)
    result={'rate_vector_rmse_rad_s':float(np.sqrt(np.mean(np.sum(safe*safe,axis=1)))),
            'nonfinite_fraction':float(1-finite.mean())}
    for j,axis in enumerate('xyz'):
        result[f'{axis}_rmse_rad_s']=float(np.sqrt(np.mean(safe[:,j]**2)))
        result[f'{axis}_bias_rad_s']=float(np.mean(error[:,j])) if finite.all() else np.nan
    return result
