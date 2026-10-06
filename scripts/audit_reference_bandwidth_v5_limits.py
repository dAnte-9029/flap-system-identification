"""Descriptive capacity bound for the unchanged +/-120rad/s2 rate derivative head."""
import numpy as np
import pandas as pd
import train_control_windows_v2 as base
from reference_bandwidth_v5 import OUT,DATA

def main():
    w=pd.read_csv(OUT/'windows.csv');rows=[]
    with np.load(DATA/'windows_100hz.npz') as z:
        rates=z['label_rates_b'];initial=z['input_initial_rates_b']
        for grid,stride,dt in [('raw100',1,.01),('raw50',2,.02)]:
            target=rates[:,stride-1::stride];delta=np.diff(np.concatenate([initial[:,None],target],axis=1),axis=1);excess=np.maximum(abs(delta)-120*dt,0)
            for i,row in enumerate(w.itertuples()):
                for ax,name in enumerate(['p','q','r']):
                    rows.append(dict(array_index=i,partition=row.partition,log_id=row.log_id,grid=grid,axis=name,
                        target_step_above_model_limit_fraction=float(np.mean(abs(delta[i,:,ax])>120*dt)),
                        sequence_rmse_lower_bound=float(np.sqrt(np.mean(excess[i,:,ax]**2)/4))))
    a=pd.DataFrame(rows);a.to_csv(OUT/'rate_increment_limits.csv',index=False)
    a.groupby(['partition','grid','axis'])[['target_step_above_model_limit_fraction','sequence_rmse_lower_bound']].mean().to_csv(OUT/'rate_increment_limits_summary.csv')
    base.write(OUT/'rate_increment_limits_scope.json',dict(script_sha256=base.sha(__file__),bound_rad_s2=120,
        derivation='sum excess_i^2 <= sum(e_i-e_(i-1))^2 <=4 sum(e_i^2), e0=0; weak sequence-RMSE lower bound, not endpoint bound',
        scope='logged sample increments, not measured physical angular acceleration; no cap adjustment or gate change',sealed_test_accessed=False))
if __name__=='__main__':main()
