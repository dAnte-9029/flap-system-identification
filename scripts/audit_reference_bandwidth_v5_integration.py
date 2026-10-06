"""Oracle logged-rate replay: compare integration grids, not deployable models."""
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from reference_bandwidth_v5 import OUT,load
from system_identification.models.trajectory_main_v1 import _normalize_quaternion,_delta_quaternion,_quaternion_multiply

def integrate(initial_quaternion,initial_rate,rates,dt):
    q=initial_quaternion;prev=initial_rate;result=[]
    for k in range(rates.shape[1]):
        q=_normalize_quaternion(_quaternion_multiply(q,_delta_quaternion(.5*(prev+rates[:,k])*dt)))
        prev=rates[:,k];result.append(q)
    return torch.stack(result,1)

def main():
    torch.set_num_threads(1);x,y,_,w=load();rows=[]
    for label,stride,dt in [('oracle_logged100',1,.01),('oracle_logged50',2,.02)]:
        pred=integrate(x['initial_quaternion'],x['initial_rates_b'],y['rates_b'][:,stride-1::stride],dt)
        truth=y['quaternion'][:,stride-1::stride]
        dot=(pred*truth).sum(-1).abs().clamp(0,1);error=(2*torch.acos(dot))**2
        for h in [.5,1.]:
            f=w[['array_index','log_id','partition']].copy();f['model']=label;f['horizon_s']=h;f['attitude_squared_rad']=error[:,round(h/dt)-1].numpy();rows.append(f)
    a=pd.concat(rows);a.to_csv(OUT/'oracle_rate_integration.csv',index=False)
    np.sqrt(a.groupby(['model','partition','log_id','horizon_s']).attitude_squared_rad.mean()).groupby(['model','partition','horizon_s']).mean().rename('attitude_rmse_rad').to_csv(OUT/'oracle_rate_integration_summary.csv')
    base.write(OUT/'oracle_integration_scope.json',dict(script_sha256=base.sha(__file__),future_true_rates_used=True,deployable=False,
        interpretation='same grid/published-as-of pose targets; attitude/rate sample times differ, so replay mismatch is not pure physical inconsistency or a strict model accuracy lower bound',
        training_or_gate_changed=False,sealed_test_accessed=False))
if __name__=='__main__':main()
