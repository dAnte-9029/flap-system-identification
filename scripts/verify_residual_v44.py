"""Independent permutation attribution, source errors and v43 metric parity."""
import itertools
import numpy as np
import pandas as pd
import train_control_windows_v2 as base
from run_residual_diagnostic_v44 import OUT,PARENTS,DATA,V43,read,iter_sources,METRICS
from audit_controller_support_v26 import verify_parent


def main():
    assert read(OUT/'summary_status.json')['status']=='complete'
    with np.load(DATA/'windows.npz') as z:truth=z['label_rates_b'].astype(float)
    rows=pd.read_csv(OUT/'rate_parts_per_window.csv').set_index(['fold','seed','cohort','variant','arm','period','metric','array_index']).sort_index()
    attribution=pd.read_csv(OUT/'kinematic_attribution_per_window.csv').set_index(['fold','seed','cohort','variant','metric','array_index']).sort_index()
    curves=pd.read_csv(OUT/'state_curves_per_flight.csv',dtype={'mode':str}).set_index(['fold','seed','cohort','variant','arm','source_fold','mode','metric','step']).sort_index()
    windows=pd.read_csv(DATA/'windows.csv');source_fold={}
    for f in range(5):
        for i in read(DATA/f'fold{f}/manifest.json')['query_indices']:source_fold[i]=f
    total=0;checks=0;curve_checks=0;seen={}
    for fold,seed,cohort,variant,ids,paths in iter_sources():
        with np.load(OUT/f'fold{fold}/{cohort}_{variant}_seed{seed}.npz') as z:a={k:z[k] for k in z.files}
        np.testing.assert_array_equal(a['array_index'],ids)
        for j,(label,path) in enumerate(zip(['baseline','candidate'],paths)):
            with np.load(path) as z:error=z['rates_b'].astype(float)-truth[ids]
            np.testing.assert_array_equal(error,a['rate_error'][:,j])
            for period,sl in [('full',slice(None)),('first',slice(0,25)),('last',slice(25,50))]:
                e=error[:,sl];bias=e.sum(1)/e.shape[1];mse=(e*e).sum(1)/e.shape[1]
                centered=np.var(e,axis=1,ddof=0)
                for axis,metric in enumerate('pqr'):
                    actual=rows.loc[(fold,seed,cohort,variant,label,period,metric)].loc[ids]
                    np.testing.assert_allclose(actual[['mse','bias','bias_squared','centered_mse']],
                        np.column_stack([mse[:,axis],bias[:,axis],bias[:,axis]**2,centered[:,axis]]),rtol=1e-10,atol=1e-12)
                    checks+=len(ids)*4
            for sf in sorted(set(source_fold[i] for i in ids)):
                flight=np.asarray([source_fold[i]==sf for i in ids]);modes=windows.iloc[ids]['mode'].to_numpy()
                for mode in ['all',*map(str,sorted(set(modes[flight])))]:
                    take=flight if mode=='all' else flight&(modes==int(mode))
                    for k,metric in enumerate(METRICS):
                        recorded=curves.loc[(fold,seed,cohort,variant,label,sf,mode,metric)].loc[np.arange(1,51)]
                        np.testing.assert_array_equal(recorded.windows,np.full(50,take.sum()))
                        np.testing.assert_allclose(recorded.mse,a['state_squared'][take,j,:,k].mean(0),rtol=1e-10,atol=1e-12);curve_checks+=50
        # Independent average over all six complete substitution orders.
        v=a['factorial_mse'];sh=np.zeros_like(a['shapley_mse'])
        for order in itertools.permutations(range(3)):
            mask=0
            for axis in order:
                next_mask=mask|(1<<axis);sh[:,axis]+=(v[:,next_mask]-v[:,mask])/6;mask=next_mask
        np.testing.assert_allclose(sh,a['shapley_mse'],rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(sh.sum(1),v[:,7]-v[:,0],rtol=1e-10,atol=1e-12)
        for axis,metric in enumerate(['roll','pitch','tilt']):
            recorded=attribution.loc[(fold,seed,cohort,variant,metric)].loc[ids]
            np.testing.assert_allclose(recorded[['p','q','r']],sh[:,:,axis],rtol=1e-10,atol=1e-12)
            native=a['state_squared'][:,:,:,METRICS.index(metric)].mean(2)
            np.testing.assert_allclose(v[:,[0,7],axis],native,rtol=1e-10,atol=1e-12)
        total+=len(ids)
        if cohort=='oof':seen.setdefault((seed,variant),[]).extend(ids)
    for ids in seen.values():np.testing.assert_array_equal(sorted(ids),np.arange(407))
    w=pd.read_csv(OUT/'state_wave_per_flight.csv',dtype={'mode':str});w=w[(w['mode']=='all')&(w.period=='full')]
    # Reproduce every original full1s OOF metric per fold/seed, not just means.
    for folder,variant in [('conditional','conditional'),('coupled',None)]:
        parent=pd.read_csv(V43/f'{folder}/state_per_flight.csv');parent=parent[(parent.horizon==50)&(parent.kind=='wave')]
        if variant is not None:parent['variant']=variant
        candidate=w[(w.cohort=='oof')&(w.variant=='conditional' if variant else w.variant!='conditional')]
        keys=['variant','fold','seed','arm','metric']
        parent=parent.set_index(keys);candidate=candidate.set_index(keys)
        assert len(parent)==len(candidate)
        np.testing.assert_allclose(candidate.rmse,parent.loc[candidate.index,'rmse'],rtol=1e-10,atol=1e-12)
    old=pd.read_csv(V43/'conditional/fit_wave_metrics.csv').set_index(['fold','seed','arm','source_fold','metric'])
    fit=w[(w.cohort=='fit')&(w.variant=='conditional')].set_index(['fold','seed','arm','source_fold','metric'])
    np.testing.assert_allclose(fit.rmse,old.loc[fit.index,'rmse'],rtol=1e-10,atol=1e-12)
    for p in PARENTS:verify_parent(p)
    for n,d in read(OUT/'protocol.json')['code'].items():assert base.sha(base.ROOT/n)==d
    base.write(OUT/'independent_verification.json',dict(status='passed',paired_trajectories=total,rate_decomposition_scalars_recomputed=checks,curve_cells_recomputed=curve_checks,
        all_six_permutations_match_shapley=True,all_original_v43_full1s_metrics_reproduced=True,all407_oof_per_seed_variant=True,
        fit_predictions_are_repeated_model_views_not_independent_samples=True,parent_hashes_unchanged=True))
    print('Independent residual decomposition, permutation attribution and v43 parity passed',flush=True)


if __name__=='__main__':main()
