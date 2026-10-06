"""Fixed common-data q waveform, state and reference-usefulness gates."""
import json
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from reference_groupcv_v8 import load_train,STAT_KEYS,ReferenceGRUPhaseSchedule,predict
from reference_condition_diagnostics_v12 import waveform,aggregate,bins
from run_reference_axis_diagnostics_v10 import read_prediction
from prepare_reference_frequency_weight_v13 import OUT,V8,V12
from evaluate_control_windows_v2 import METRICS
from run_reference_generalization_v7 import verify_manifest


def model_from(path):
    ck=torch.load(path,map_location='cpu',weights_only=False);sd=ck['state_dict'];m=ReferenceGRUPhaseSchedule({k:sd[k] for k in STAT_KEYS});m.load_state_dict(sd);return m.eval()


def main():
    torch.set_num_threads(1);verify_manifest(OUT/'protocol.json',['source_pins'])
    assert json.loads((OUT/'launch/status.json').read_text())['status']=='complete'
    assert not (OUT/'decision.json').exists()
    x,y,w=load_train();old=pd.read_csv(V12/'waveform_per_window.csv');rows=[];states=[];members=[]
    for fold in range(5):
        info=json.loads((OUT/f'fold{fold}/manifest.json').read_text());fit=np.array(info['fit_indices']);query=np.array(info['query_indices'])
        with np.load(OUT/f'fold{fold}/weights.npz') as z:cuts=z['cuts']
        for cohort,ids in [('fit',fit),('oof',query)]:
            mm=w.iloc[ids][['array_index','log_id']].copy();mm['fold']=fold;mm['cohort']=cohort;mm['stratum']=bins(x['history_state'][ids,:,11].numpy().mean(1,dtype=float),cuts);members.append(mm)
        for seed in [17,23,42]:
            run=OUT/f'fold{fold}/seed{seed}';model=model_from(run/'final.pt')
            a=pd.read_csv(run/'per_flight.csv');states.append(a)
            b=pd.read_csv(V8/f'fold{fold}/hold_{seed}/per_flight.csv');b['arm']='baseline';states.append(b)
            for cohort,ids in [('fit',fit),('oof',query)]:
                factual=read_prediction(run/f'{cohort}_predictions.npz',ids)
                heldx={k:v for k,v in x.items()};heldx['future_reference']=x['history_reference'][:,-1:,:].expand_as(x['future_reference'])
                held=predict(model,heldx,ids);np.savez_compressed(run/f'{cohort}_held.npz',array_index=ids,rates_b=held['rates_b'])
                for axis,j in [('p',0),('q',1),('r',2)]:
                    for steps in [25,50]:
                        d=waveform(factual['rates_b'][:,:steps,j].astype(float),y['rates_b'][ids,:steps,j].numpy().astype(float),held['rates_b'][:,:steps,j].astype(float))
                        df=w.iloc[ids][['array_index','log_id']].copy();df['fold']=fold;df['seed']=seed;df['cohort']=cohort;df['axis']=axis;df['horizon_s']=steps*.02;df['arm']='weighted'
                        for k,v in d.items():df[k]=v
                        rows.append(df)
                # Old r was not part of v12; compute its error-only diagnostics, not reference gates.
                baseline=read_prediction(V8/f'fold{fold}/hold_{seed}/{cohort}_predictions.npz',ids)
                for steps in [25,50]:
                    p=baseline['rates_b'][:,:steps,2].astype(float);d=waveform(p,y['rates_b'][ids,:steps,2].numpy().astype(float),p)
                    df=w.iloc[ids][['array_index','log_id']].copy();df['fold']=fold;df['seed']=seed;df['cohort']=cohort;df['axis']='r';df['horizon_s']=steps*.02;df['arm']='baseline'
                    for k,v in d.items():df[k]=v
                    df['held_mse']=np.nan;df['reference_effect_mse']=np.nan;rows.append(df)
    old['arm']='baseline';rows=pd.concat([old]+rows);rows.to_csv(OUT/'waveform_per_window.csv',index=False)
    sf=pd.concat(states);sf.to_csv(OUT/'state_per_flight.csv',index=False)
    members=pd.concat(members);members.to_csv(OUT/'frequency_membership.csv',index=False)
    common=['arm','seed','fold','log_id','cohort','axis','horizon_s']
    f=rows.groupby(common).apply(aggregate,include_groups=False).reset_index();f.to_csv(OUT/'waveform_per_flight.csv',index=False)
    f.groupby(['arm','cohort','axis','horizon_s']).mean(numeric_only=True).to_csv(OUT/'waveform_summary.csv')
    strata=rows.merge(members,on=['fold','log_id','cohort','array_index'],validate='many_to_one')
    g=strata.groupby(common+['stratum']).apply(aggregate,include_groups=False).reset_index();g.to_csv(OUT/'strata_per_flight.csv',index=False)
    g.groupby(['arm','cohort','axis','horizon_s','stratum']).mean(numeric_only=True).to_csv(OUT/'strata_summary.csv')
    q=f[(f.cohort=='oof')&(f.axis=='q')].groupby(['arm','seed','log_id']).rmse.mean()
    cube=np.stack([q.loc[arm].unstack('log_id').loc[[17,23,42]].to_numpy() for arm in ['baseline','weighted']])
    gains=1-cube[1].mean(1)/cube[0].mean(1);fgain=1-cube[1].mean(0)/cube[0].mean(0);primary=float(1-cube[1].mean()/cube[0].mean())
    state=sf[(sf.cohort=='oof')&(sf.subset=='all')&sf.horizon_s.isin([.5,1.])].groupby(['arm','seed','horizon_s'])[METRICS].mean()
    reg=state.loc['weighted']/state.loc['baseline']-1;reg.to_csv(OUT/'state_regressions.csv')
    wave=f[(f.cohort=='oof')&f.axis.isin(['p','r'])].groupby(['arm','seed','axis','horizon_s']).rmse.mean();pr=wave.loc['weighted']/wave.loc['baseline']-1
    cond=g[(g.cohort=='oof')&(g.axis=='q')].groupby(['arm','seed','horizon_s','stratum']).rmse.mean();cr=cond.loc['weighted']/cond.loc['baseline']-1
    qr=f[(f.cohort=='oof')&(f.axis=='q')&(f.horizon_s==1)].groupby(['arm','seed'])[['reference_effect_rms','reference_utility']].mean()
    ratio=qr.loc['weighted'].reference_effect_rms/qr.loc['baseline'].reference_effect_rms;delta=qr.loc['weighted'].reference_utility-qr.loc['baseline'].reference_utility
    rng=np.random.default_rng(913);boot=[]
    for _ in range(2000):
        ix=rng.integers(5,size=5);v=cube[:,:,ix].mean(axis=(1,2));boot.append(1-v[1]/v[0])
    absolute=[]
    for seed in [17,23,42]:
        s=state.loc[('weighted',seed,.5)];absolute.append(dict(seed=seed,p=float(s.p),q=float(s.q),r=float(s.r),attitude_deg=float(np.rad2deg(s.attitude)),velocity=float(s.velocity),passed=bool((s[['p','q','r']]<=.15).all() and s.attitude<=np.deg2rad(3) and s.velocity<=.5)))
    passed=bool(primary>=.05 and (gains>0).all() and (fgain>0).sum()>=4 and (reg<=.05).all().all() and (pr<=.05).all() and (cr<=.05).all() and ratio.between(.9,1.1).all() and (delta>=-.02).all())
    decision=dict(primary_gain=primary,seed_gains=gains.tolist(),flight_gains=fgain.tolist(),flight_order=q.loc['baseline'].unstack('log_id').columns.tolist(),bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
        max_state_regression=float(reg.to_numpy().max()),max_pr_waveform_regression=float(pr.max()),max_q_stratum_regression=float(cr.max()),reference_effect_ratios=ratio.to_dict(),reference_utility_deltas=delta.to_dict(),
        relative_gate=passed,absolute=absolute,control_validated=False,development_evaluated=False,sealed_test_accessed=False)
    base.write(OUT/'decision.json',decision);print(json.dumps(decision,indent=2),flush=True)

if __name__=='__main__':main()
