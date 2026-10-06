"""Paired whole-flight first-step and autonomous-rollout evaluation of v18."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from reference_q_aux_v18 import OUT,V8,METRICS,errors
from reference_groupcv_v8 import load_train,model_from,predict,SEEDS,STAT_KEYS
from reference_increment_v16 import safe_corr
from run_reference_generalization_v7 import verify_manifest


def aggregate(frame,keys,columns):
    # Equal source-flight/seed weights, first average repeated fit models.
    bysource=frame.groupby(keys+['log_id','seed'])[columns].mean().reset_index()
    return bysource,bysource.groupby(keys)[columns].mean().reset_index()


def main():
    torch.set_num_threads(1);verify_manifest(OUT/'protocol.json',['source_pins'])
    launch=json.loads((OUT/'launch/status.json').read_text());assert launch['status']=='complete' and len(launch['completed'])==16
    x,y,w=load_train();rows=[];profiles=[];increments=[];checks=[];saturation=[];references=[]
    for fold in range(5):
        src=V8/f'fold{fold}';info=json.loads((src/'manifest.json').read_text())
        with np.load(src/'normalization.npz') as z:stats={k:z[k] for k in z.files}
        for seed in SEEDS:
            dest=OUT/f'fold{fold}/seed{seed}';protocol=json.loads((dest/'run_protocol.json').read_text())
            parent=src/f'hold_{seed}'
            assert protocol['initial_parameter_sha256']==json.loads((parent/'run_protocol.json').read_text())['initial_parameter_sha256']
            with np.load(parent/'sample_draws.npz') as a,np.load(dest/'sample_draws.npz') as b:np.testing.assert_array_equal(a['array_index'],b['array_index'])
            assert protocol['normalization_sha256']==base.sha(src/'normalization.npz')
            model=model_from(dest/'final.pt')
            for k in STAT_KEYS:np.testing.assert_array_equal(model.state_dict()[k].numpy(),np.asarray(stats[k],dtype=np.float32))
            for cohort,ids in [('fit',np.array(info['fit_indices'])),('oof',np.array(info['query_indices']))]:
                truth={k:v[ids].numpy() for k,v in y.items()}
                preds={}
                for arm,folder in [('baseline',parent),('aux',dest)]:
                    with np.load(folder/f'{cohort}_predictions.npz') as z:
                        np.testing.assert_array_equal(z['array_index'],ids)
                        preds[arm]={k:z[k] for k in ['rates_b','velocity_n','quaternion']}
                preds['persistence']={k:np.repeat(x[source][ids].numpy()[:,None],50,axis=1) for k,source in [('rates_b','initial_rates_b'),('velocity_n','initial_velocity_n'),('quaternion','initial_quaternion')]}
                replay=predict(model,x,ids)
                discrepancy=max(float(np.max(abs(replay[k]-preds['aux'][k]))) for k in replay)
                assert discrepancy<1e-6
                checks.append(dict(fold=fold,seed=seed,cohort=cohort,reload_max_abs=discrepancy))
                for arm,pred in preds.items():
                    e=errors(pred,truth);assert np.isfinite(e).all()
                    for log in w.iloc[ids].log_id.unique():
                        choose=w.iloc[ids].log_id.eq(log).to_numpy();mean=e[choose].mean(axis=0)
                        common=dict(fold=fold,seed=seed,cohort=cohort,log_id=log,arm=arm,windows=int(choose.sum()))
                        for n in [1,5,10,25,50]:
                            vals=np.sqrt(mean[:n].mean(axis=0))
                            rows.append(dict(**common,kind='prefix',horizon_s=n*.02,**dict(zip(METRICS,vals))))
                        for n in [25,50]:rows.append(dict(**common,kind='endpoint',horizon_s=n*.02,**dict(zip(METRICS,np.sqrt(mean[n-1])))))
                        for n in range(50):profiles.append(dict(**common,age_s=(n+1)*.02,**dict(zip(METRICS,np.sqrt(mean[n])))))
                        for j,axis in enumerate('pqr'):
                            t=truth['rates_b'][choose,0,j]-x['initial_rates_b'][ids[choose],j].numpy()
                            p=pred['rates_b'][choose,0,j]-x['initial_rates_b'][ids[choose],j].numpy()
                            eligible=abs(t)>.01
                            increments.append(dict(**common,axis=axis,truth_std=float(t.std()),pred_std=float(p.std()),
                                std_ratio=float(p.std()/t.std()),correlation=safe_corr(p,t),
                                sign_agreement=float(((np.sign(t[eligible])==np.sign(p[eligible]))&(abs(p[eligible])>.01)).mean()),
                                sign_eligible=int(eligible.sum())))
                if cohort=='oof':
                    held_x={k:v.clone() for k,v in x.items()}
                    held_x['future_reference']=x['future_reference'][:,:1].expand_as(x['future_reference']).clone()
                    for arm,local_model in [('baseline',model_from(parent/'final.pt')),('aux',model)]:
                        held=predict(local_model,held_x,ids)
                        if arm=='aux':np.savez_compressed(dest/'oof_held_reference_predictions.npz',array_index=ids,**held)
                        for log in w.iloc[ids].log_id.unique():
                            mask=w.iloc[ids].log_id.eq(log).to_numpy()
                            factual=preds[arm]['rates_b'][mask,:,1].astype(float);h=held['rates_b'][mask,:,1].astype(float);t=truth['rates_b'][mask,:,1]
                            fr=float(np.sqrt(np.mean((factual-t)**2)));hr=float(np.sqrt(np.mean((h-t)**2)))
                            references.append(dict(fold=fold,seed=seed,log_id=log,arm=arm,factual_rmse=fr,held_rmse=hr,utility=1-fr/hr,effect_rms=float(np.sqrt(np.mean((factual-h)**2)))))
                # Inspect output saturation without changing the checkpoint.
                heads=[]
                hook=model.head.register_forward_hook(lambda module,args,out:heads.append(out.detach().numpy().copy()))
                try:
                    with torch.no_grad():model(**{k:v[ids] for k,v in x.items()})
                finally:hook.remove()
                head=np.stack(heads,axis=1)
                for j,axis in enumerate('pqr'):
                    saturation.append(dict(fold=fold,seed=seed,cohort=cohort,axis=axis,
                        first_fraction=float(np.mean(abs(head[:,0,j+3])>=6)),rollout_fraction=float(np.mean(abs(head[:,:,j+3])>=6)),
                        maximum_abs=float(np.max(abs(head[:,:,j+3])))))
            query=np.array(info['query_indices'])[:3];batch={k:v[query].clone() for k,v in x.items()}
            with torch.no_grad():a=model(**batch);batch['future_reference'][:,1:]+=10;b=model(**batch)
            assert all(torch.equal(a[k][:,:1],b[k][:,:1]) for k in a)
    pf=pd.DataFrame(rows);pf.to_csv(OUT/'per_flight.csv',index=False)
    source,summary=aggregate(pf,['cohort','arm','kind','horizon_s'],METRICS)
    source.to_csv(OUT/'source_flight.csv',index=False);summary.to_csv(OUT/'summary.csv',index=False)
    profile=pd.DataFrame(profiles);profile.to_csv(OUT/'age_per_flight.csv',index=False)
    _,ps=aggregate(profile,['cohort','arm','age_s'],METRICS);ps.to_csv(OUT/'age_summary.csv',index=False)
    inc=pd.DataFrame(increments);inc.to_csv(OUT/'increment_per_flight.csv',index=False)
    _,incsummary=aggregate(inc,['cohort','arm','axis'],['truth_std','pred_std','std_ratio','correlation','sign_agreement'])
    incsummary.to_csv(OUT/'increment_summary.csv',index=False)
    pd.DataFrame(saturation).to_csv(OUT/'saturation.csv',index=False)
    gains=[];diagnostic={}
    first=source[(source.kind=='prefix')&(source.horizon_s==.02)&source.arm.isin(['baseline','aux'])].copy()
    first['primary']=first['q']
    for cohort in ['fit','oof']:
        frame=first[first.cohort==cohort]
        overall=frame.groupby('arm').primary.mean();gain=1-overall['aux']/overall['baseline']
        byseed=frame.groupby(['arm','seed']).primary.mean().unstack('arm');sg=1-byseed.aux/byseed.baseline
        byflight=frame.groupby(['arm','log_id']).primary.mean().unstack('arm');fg=1-byflight.aux/byflight.baseline
        diagnostic[cohort]=dict(primary_gain=float(gain),seed_gains=sg.to_dict(),flight_gains=fg.to_dict(),
            positive_seeds=int((sg>0).sum()),positive_flights=int((fg>0).sum()),
            passes=bool(gain>=.05 and (sg>0).all() and (fg>0).sum()>=4))
    # Every metric/horizon/seed must retain the prespecified full-rollout protection.
    scores=source[(source.cohort=='oof')&source.arm.isin(['baseline','aux'])].groupby(['arm','seed','kind','horizon_s'])[METRICS].mean()
    for (seed,kind,horizon),a in scores.loc['baseline'].iterrows():
        b=scores.loc[('aux',seed,kind,horizon)]
        for metric in METRICS:gains.append(dict(seed=seed,kind=kind,horizon_s=horizon,metric=metric,baseline=a[metric],aux=b[metric],gain=1-b[metric]/a[metric]))
    gainframe=pd.DataFrame(gains);gainframe.to_csv(OUT/'oof_seed_gains.csv',index=False)
    protect=gainframe[(gainframe.kind=='prefix')&gainframe.horizon_s.isin([.5,1.])]
    early=gainframe[(gainframe.kind=='prefix')&(gainframe.horizon_s==.02)&gainframe.metric.isin(['p','r','velocity','attitude'])]
    protect=pd.concat([protect,early]);failed=protect[protect.gain < -.05]
    limits=dict(p=.15,q=.15,r=.15,velocity=.5,attitude=np.deg2rad(3))
    absolute=gainframe[(gainframe.kind=='endpoint')&(gainframe.horizon_s==.5)].copy()
    absolute['limit']=absolute.metric.map(limits);absolute['passes']=absolute.aux<=absolute.limit
    absolute.to_csv(OUT/'absolute_checks.csv',index=False)
    ref=pd.DataFrame(references);ref.to_csv(OUT/'reference_per_flight.csv',index=False)
    rs=ref.groupby(['seed','arm'])[['utility','effect_rms']].mean().unstack('arm');rc=[]
    for seed,row in rs.iterrows():
        drop=row[('utility','aux')]-row[('utility','baseline')];ratio=row[('effect_rms','aux')]/row[('effect_rms','baseline')]
        rc.append(dict(seed=seed,utility_change=float(drop),effect_ratio=float(ratio),passes=bool(drop>=-.02 and .9<=ratio<=1.1)))
    base.write(OUT/'reference_checks.json',rc)
    identity=json.loads((OUT/'fold0/identity/status.json').read_text());assert identity['status']=='complete' and identity['identity_parameter_max_abs']<1e-6
    decision=dict(stage_complete=True,diagnostic=diagnostic,reference_protection_passes=all(r['passes'] for r in rc),rollout_protection_passes=bool(failed.empty),
        failed_protection_checks=failed.to_dict('records'),absolute_passes=bool(absolute.passes.all()),
        model_promoted=False,control_validated=False,development_evaluated=False,sealed_test_accessed=False)
    base.write(OUT/'decision.json',decision)
    base.write(OUT/'verification.json',dict(models=15,identity_parameter_max_abs=identity['identity_parameter_max_abs'],updates_each=1600,original_windows=407,reloads=checks,
        identical_initialization_and_draws=True,normalization_unchanged=True,all15_reference_firststep_causal=True,
        all50step_predictions_finite=True,sealed_test_accessed=False))
    print(json.dumps(decision,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
