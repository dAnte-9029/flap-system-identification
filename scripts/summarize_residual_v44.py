"""Flight-balanced summaries and exact paired coupling increments."""
import numpy as np
import pandas as pd
import train_control_windows_v2 as base
from run_residual_diagnostic_v44 import OUT,read


def balanced(frame,keys,values):
    # First average observations within a source flight for each model.
    f=frame.groupby(keys+['fold','seed','source_fold'])[values].mean().reset_index()
    a=f.groupby(keys+['fold','seed'])[values].mean().groupby(keys).mean().reset_index()
    return f,a


def main():
    assert read(OUT/'verification.json')['status']=='passed'
    r=pd.read_csv(OUT/'rate_parts_per_window.csv')
    keys=['cohort','variant','arm','period','metric'];values=['mse','bias','bias_squared','centered_mse']
    f,a=balanced(r,keys,values)
    f.to_csv(OUT/'rate_parts_per_flight.csv',index=False);a.to_csv(OUT/'rate_summary.csv',index=False)
    _,mode=balanced(r,keys+['mode'],values);mode.to_csv(OUT/'rate_by_mode.csv',index=False)
    r.groupby(keys+['fold','seed','source_fold','mode']).size().rename('windows').to_csv(OUT/'mode_counts.csv')
    flight=f.groupby(keys+['source_fold'])[values].mean().reset_index();flight.to_csv(OUT/'rate_by_source_flight.csv',index=False)
    c=pd.read_csv(OUT/'state_curves_per_flight.csv');c['rmse']=np.sqrt(c.mse)
    k=['cohort','variant','arm','mode','metric','step','time_s']
    c.groupby(k+['fold','seed']).rmse.mean().groupby(k).mean().to_csv(OUT/'curves_summary.csv')
    # Aggregate time before square root for waveform RMSE, preserving v43.
    waves=[]
    for period,start,end in [('full',1,50),('first',1,25),('last',26,50)]:
        cc=c[c.step.between(start,end)]
        g=cc.groupby(['cohort','variant','arm','mode','metric','fold','seed','source_fold']).mse.mean().reset_index()
        g['rmse']=np.sqrt(g.mse);g['period']=period;waves.append(g)
    w=pd.concat(waves);w.to_csv(OUT/'state_wave_per_flight.csv',index=False)
    k=['cohort','variant','arm','mode','metric','period']
    w.groupby(k+['fold','seed'])[['mse','rmse']].mean().groupby(k).mean().to_csv(OUT/'state_wave_summary.csv')
    w.groupby(k+['source_fold'])[['mse','rmse']].mean().to_csv(OUT/'state_wave_by_source_flight.csv')
    at=pd.read_csv(OUT/'kinematic_attribution_per_window.csv');av=['baseline_mse','candidate_mse','delta_mse','p','q','r']
    _,s=balanced(at,['cohort','variant','metric'],av);s.to_csv(OUT/'attribution_summary.csv',index=False)
    _,s=balanced(at,['cohort','variant','metric','mode'],av);s.to_csv(OUT/'attribution_by_mode.csv',index=False)
    f,_=balanced(at,['cohort','variant','metric'],av)
    f.groupby(['cohort','variant','metric','source_fold'])[av].mean().to_csv(OUT/'attribution_by_source_flight.csv')
    oracle=pd.read_csv(OUT/'kinematic_oracles_per_window.csv')
    f,_=balanced(oracle,['cohort','variant','arm','metric'],['native_mse','debiased_mse','true_rate_mse'])
    for n in ['native','debiased','true_rate']:f[n+'_rmse']=np.sqrt(f[n+'_mse'])
    k=['cohort','variant','arm','metric'];vs=['native_rmse','debiased_rmse','true_rate_rmse']
    f.groupby(k+['fold','seed'])[vs].mean().groupby(k).mean().to_csv(OUT/'oracle_summary.csv')
    actions=pd.read_csv(OUT/'action_parts_per_window.csv')
    _,a=balanced(actions,['variant','arm','axis'],values);a.to_csv(OUT/'action_summary.csv',index=False)
    # Same source, same starting history, different future command boundary.
    joined=w[(w.cohort=='oof')&(w['mode']=='all')].pivot(index=['variant','period','fold','seed','source_fold','metric'],columns='arm',values='mse').reset_index()
    joined['delta_mse']=joined.candidate-joined.baseline
    conditioned=joined[joined.variant=='conditional'].drop(columns=['variant']).rename(columns={'baseline':'conditional_baseline','candidate':'conditional_candidate','delta_mse':'conditional_delta'})
    comp=joined[joined.variant!='conditional'].merge(conditioned,on=['period','fold','seed','source_fold','metric'],validate='many_to_one')
    comp['additional_coupled_delta']=comp.delta_mse-comp.conditional_delta
    np.testing.assert_allclose(comp.delta_mse,comp.conditional_delta+comp.additional_coupled_delta,rtol=1e-12,atol=1e-12)
    comp.to_csv(OUT/'coupling_changes_per_flight.csv',index=False)
    comp.groupby(['variant','period','metric'])[['delta_mse','conditional_delta','additional_coupled_delta']].mean().to_csv(OUT/'coupling_changes_summary.csv')
    base.write(OUT/'summary_status.json',dict(status='complete',window_selection_changed=False))


if __name__=='__main__':main()
