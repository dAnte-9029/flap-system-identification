"""Frozen paired primary/protection gates and secondary v23 comparison."""
import json
import numpy as np
import pandas as pd
import train_control_windows_v2 as base
from controller_no_age_v27 import OUT,V23,V25,load
ARMS=['zero_history_control','no_age_history_control']
from evaluate_rate_reference_v23 import errors


def main():
    assert json.loads((OUT/'launch/status.json').read_text())['status']=='complete'
    _,y=load();w=pd.read_csv(V23/'windows.csv');rows=[];response=[];checks=[]
    for fold in range(5):
        info=json.loads((V23/f'fold{fold}/manifest.json').read_text())
        for seed in [17,23,42]:
            init=[];draw=[]
            for arm in ARMS:
                dest=(V25 if arm=='zero_history_control' else OUT)/f'fold{fold}/{arm}/seed{seed}';status=json.loads((dest/'status.json').read_text())
                assert status['step']==1600 and base.sha(dest/'final.pt')==status['checkpoint_sha256']
                protocol=json.loads((dest/'run_protocol.json').read_text());init.append(protocol['initial_parameters_sha256']);draw.append(protocol['draws_sha256'])
                for cohort in ['fit','oof']:
                    with np.load(dest/f'{cohort}_predictions.npz') as z:
                        ids=z['array_index'];pred={k:z[k].astype(float) for k in y}
                    np.testing.assert_array_equal(ids,info['fit_indices' if cohort=='fit' else 'query_indices'])
                    true={k:v[ids].numpy().astype(float) for k,v in y.items()}
                    meta=w.iloc[ids][['array_index','log_id','mode','active_reference']].copy().assign(fold=fold,seed=seed,arm=arm,cohort=cohort)
                    for h in [1,5,25,50]:
                        wave,end=errors(pred,true,h)
                        for kind,data in [('wave',wave),('endpoint',end)]:
                            for metric,mse in data.items():rows.append(meta.assign(horizon=h,kind=kind,metric=metric,mse=mse))
                    if cohort=='oof':
                        with np.load(dest/'oof_held.npz') as z:held=z['rates_b'].astype(float)
                        for j,axis in enumerate(['p','q','r']):response.append(meta.assign(metric=axis,
                            factual_mse=np.mean((pred['rates_b'][:,:,j]-true['rates_b'][:,:,j])**2,axis=1),
                            held_mse=np.mean((held[:,:,j]-true['rates_b'][:,:,j])**2,axis=1),
                            effect_mse=np.mean((held[:,:,j]-pred['rates_b'][:,:,j])**2,axis=1)))
                checks.append(dict(fold=fold,seed=seed,arm=arm,**status))
            assert len(set(init))==len(set(draw))==1
    a=pd.concat(rows,ignore_index=True);a.to_csv(OUT/'metrics_per_window.csv',index=False)
    keys=['seed','cohort','arm','horizon','kind','metric','log_id']
    f=a.groupby(keys).mse.mean().pow(.5).reset_index(name='rmse');f.to_csv(OUT/'metrics_per_flight.csv',index=False)
    av=f.groupby(keys[:-1]).rmse.mean().reset_index()
    pair=av.pivot(index=['seed','cohort','horizon','kind','metric'],columns='arm',values='rmse').reset_index()
    pair['improvement_percent']=100*(1-pair.no_age_history_control/pair.zero_history_control)
    pair.to_csv(OUT/'comparison.csv',index=False)
    full=pd.read_csv(V25/'comparison.csv')
    full=full[['seed','cohort','horizon','kind','metric','observed_history_control']]
    full=pair.merge(full,on=['seed','cohort','horizon','kind','metric'],validate='one_to_one')
    full['no_age_vs_full_percent']=100*(1-full.no_age_history_control/full.observed_history_control)
    full.to_csv(OUT/'secondary_full_history.csv',index=False)
    mode=a.groupby(keys+['mode']).mse.mean().pow(.5).groupby(keys[:-1]+['mode']).mean().reset_index(name='rmse')
    mode.to_csv(OUT/'mode_comparison.csv',index=False)
    old=pd.read_csv(V23/'comparison.csv').rename(columns={'rate':'v23'})
    secondary=pair.merge(old[['seed','cohort','horizon','kind','metric','v23']],on=['seed','cohort','horizon','kind','metric'],validate='one_to_one')
    secondary['zero_vs_v23_percent']=100*(1-secondary.zero_history_control/secondary.v23)
    secondary.to_csv(OUT/'secondary_v23.csv',index=False)
    q=pair[(pair.cohort=='oof')&(pair.kind=='wave')&(pair.horizon==50)&(pair.metric=='q')]
    qflight=f[(f.cohort=='oof')&(f.kind=='wave')&(f.horizon==50)&(f.metric=='q')].groupby(['log_id','arm']).rmse.mean().unstack()
    qgain=100*(1-q.no_age_history_control.mean()/q.zero_history_control.mean())
    positive=int((qflight.no_age_history_control<qflight.zero_history_control).sum())
    protection=pair[(pair.cohort=='oof')&(((pair.kind=='wave')&(pair.horizon==50)&pair.metric.isin(['p','r']))|
                    ((pair.kind=='endpoint')&pair.horizon.isin([25,50])&pair.metric.isin(['velocity','attitude'])))]
    active=a[(a.cohort=='oof')&(a.kind=='endpoint')&(a.horizon==25)&a.active_reference]
    active=active.groupby(['seed','arm','metric','log_id']).mse.mean().pow(.5).groupby(['seed','arm','metric']).mean().reset_index(name='rmse')
    active['limit']=active.metric.map({'p':.15,'q':.15,'attitude':np.deg2rad(3),'velocity':.5})
    active=active[active.limit.notna()].copy();active['passed']=active.rmse<=active.limit
    active.to_csv(OUT/'absolute_active_gate.csv',index=False)
    ac=active[active.arm=='no_age_history_control']
    pd.concat(response,ignore_index=True).to_csv(OUT/'reference_hold_diagnostic.csv',index=False)
    base.write(OUT/'training_checks.json',checks)
    base.write(OUT/'evaluation.json',dict(stage='v27',completed_trainings=15,steps_each=1600,
        primary_q_improvement_percent=float(qgain),all_seeds_positive=bool((q.improvement_percent>0).all()),positive_flights=positive,
        protection_passed=int((protection.improvement_percent>=-5).sum()),protection_total=len(protection),
        numerical_gate=bool(qgain>=5 and (q.improvement_percent>0).all() and positive>=4 and (protection.improvement_percent>=-5).all()),
        absolute_active_passed=int(ac.passed.sum()),absolute_active_total=len(ac),promoted=False,control_ready=False,
        baseline_retained='A1600',limitation='Paired historical observation block under recorded feedback-generated rate-reference conditioning. No causal effect or independent closed-loop certification.',
        validation_accessed=False,sealed_test_accessed=False))


if __name__=='__main__':main()
