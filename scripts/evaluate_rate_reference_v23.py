"""Same-label, explicitly conditional comparison against immutable v8 predictions."""
import json
import numpy as np
import pandas as pd
import train_control_windows_v2 as base
from rate_reference_v23 import OUT,V8,load


def errors(pred,true,horizon):
    dot=np.abs(np.sum(pred['quaternion']*true['quaternion'],axis=-1)).clip(0,1)
    angle=2*np.arccos(dot)
    result={axis:np.mean((pred['rates_b'][:,:horizon,j]-true['rates_b'][:,:horizon,j])**2,axis=1)
            for j,axis in enumerate(['p','q','r'])}
    result['velocity']=np.mean((pred['velocity_n'][:,:horizon]-true['velocity_n'][:,:horizon])**2,axis=(1,2))
    result['attitude']=np.mean(angle[:,:horizon]**2,axis=1)
    endpoint={axis:(pred['rates_b'][:,horizon-1,j]-true['rates_b'][:,horizon-1,j])**2
              for j,axis in enumerate(['p','q','r'])}
    endpoint['velocity']=np.mean((pred['velocity_n'][:,horizon-1]-true['velocity_n'][:,horizon-1])**2,axis=1)
    endpoint['attitude']=angle[:,horizon-1]**2
    return result,endpoint


def main():
    assert json.loads((OUT/'launch/status.json').read_text())['status']=='complete'
    _,y=load();w=pd.read_csv(OUT/'windows.csv');rows=[];response=[];checks=[]
    for fold in range(5):
        info=json.loads((OUT/f'fold{fold}/manifest.json').read_text())
        for seed in [17,23,42]:
            dest=OUT/f'fold{fold}/seed{seed}';status=json.loads((dest/'status.json').read_text())
            assert status['step']==1600 and base.sha(dest/'final.pt')==status['checkpoint_sha256']
            for cohort in ['fit','oof']:
                for arm,root in [('baseline',V8/f'fold{fold}/hold_{seed}'),('rate',dest)]:
                    with np.load(root/f'{cohort}_predictions.npz') as z:
                        ids=z['array_index'];pred={k:z[k].astype(float) for k in y}
                    expected=info['fit_indices' if cohort=='fit' else 'query_indices']
                    np.testing.assert_array_equal(ids,expected)
                    true={k:v[ids].numpy().astype(float) for k,v in y.items()}
                    for h in [1,5,25,50]:
                        wave,end=errors(pred,true,h)
                        for typ,data in [('wave',wave),('endpoint',end)]:
                            for metric,v in data.items():
                                for idx,mse in zip(ids,v):rows.append(dict(fold=fold,seed=seed,cohort=cohort,arm=arm,
                                    array_index=int(idx),log_id=w.iloc[idx].log_id,active_reference=bool(w.iloc[idx].active_reference),
                                    horizon=h,kind=typ,metric=metric,mse=float(mse)))
                if cohort=='oof':
                    with np.load(dest/'oof_held.npz') as z:held={k:z[k].astype(float) for k in y}
                    for j,axis in enumerate(['p','q','r']):
                        for i,idx in enumerate(ids):response.append(dict(fold=fold,seed=seed,array_index=int(idx),
                            log_id=w.iloc[idx].log_id,metric=axis,factual_mse=float(np.mean((pred['rates_b'][i,:,j]-true['rates_b'][i,:,j])**2)),
                            held_mse=float(np.mean((held['rates_b'][i,:,j]-true['rates_b'][i,:,j])**2)),
                            effect_mse=float(np.mean((pred['rates_b'][i,:,j]-held['rates_b'][i,:,j])**2))))
            checks.append(dict(fold=fold,seed=seed,**status))
    allrows=pd.DataFrame(rows);allrows.to_csv(OUT/'metrics_per_window.csv',index=False)
    keys=['seed','cohort','arm','horizon','kind','metric','log_id']
    flight=allrows.groupby(keys).mse.mean().pow(.5).reset_index(name='rmse');flight.to_csv(OUT/'metrics_per_flight.csv',index=False)
    average=flight.groupby(keys[:-1]).rmse.mean().reset_index()
    paired=average.pivot(index=['seed','cohort','horizon','kind','metric'],columns='arm',values='rmse').reset_index()
    paired['improvement_percent']=100*(1-paired.rate/paired.baseline)
    paired.to_csv(OUT/'comparison.csv',index=False)
    main=paired[(paired.cohort=='oof')&(paired.horizon==50)&(paired.kind=='wave')&(paired.metric=='q')]
    perflight=flight[(flight.cohort=='oof')&(flight.horizon==50)&(flight.kind=='wave')&(flight.metric=='q')]
    perflight=perflight.groupby(['log_id','arm']).rmse.mean().unstack()
    positive=int((perflight.rate<perflight.baseline).sum())
    primary=100*(1-main.rate.mean()/main.baseline.mean())
    protection=paired[(paired.cohort=='oof')&(((paired.kind=='wave')&(paired.horizon==50)&paired.metric.isin(['p','r']))|
                      ((paired.kind=='endpoint')&paired.horizon.isin([25,50])&paired.metric.isin(['velocity','attitude'])))]
    active=allrows[(allrows.cohort=='oof')&(allrows.arm=='rate')&(allrows.kind=='endpoint')&(allrows.horizon==25)&allrows.active_reference]
    active=active.groupby(['seed','metric','log_id']).mse.mean().pow(.5).groupby(['seed','metric']).mean().reset_index(name='rmse')
    active['limit']=active.metric.map({'p':.15,'q':.15,'attitude':np.deg2rad(3),'velocity':.5})
    active=active[active.limit.notna()].copy();active['passed']=active.rmse<=active.limit
    active.to_csv(OUT/'absolute_active_gate.csv',index=False)
    pd.DataFrame(response).to_csv(OUT/'reference_hold_diagnostic.csv',index=False)
    base.write(OUT/'evaluation.json',dict(completed_trainings=15,steps_each=1600,
        primary_q_wave_improvement_percent=float(primary),all_seeds_q_positive=bool((main.improvement_percent>0).all()),
        positive_flights=positive,protection_passed=int((protection.improvement_percent>=-5).sum()),protection_total=len(protection),
        descriptive_gate=bool(primary>=5 and (main.improvement_percent>0).all() and positive>=4 and (protection.improvement_percent>=-5).all()),
        absolute_active_passed=int(active.passed.sum()),absolute_active_total=len(active),
        promoted=False,control_ready=False,limit='Conditional prediction under feedback-generated recorded rate references; different information boundary from attitude baseline. Independent response/closed-loop evidence absent.',
        validation_accessed=False,sealed_test_accessed=False))
    base.write(OUT/'training_checks.json',checks)


if __name__=='__main__':main()
