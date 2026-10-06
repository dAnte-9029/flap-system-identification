"""Equal-flight summaries and verification for frozen v19 diagnostics."""
import json
import numpy as np
import pandas as pd
import train_control_windows_v2 as base
from run_reference_localization_v19 import OUT,V18
from run_reference_generalization_v7 import verify_manifest
from reference_groupcv_v8 import load_train
from reference_condition_diagnostics_v12 import weighted_terciles,bins
from reference_localization_v19 import reference_activity,BLOCKS


def summary(df,keys,values):
    src=df.groupby(keys+['seed','log_id'])[values].mean().reset_index()
    return src,src.groupby(keys)[values].mean().reset_index()


def main():
    verify_manifest(OUT/'protocol.json',['source_pins'])
    q=pd.read_csv(OUT/'q_decomposition_per_window.csv');ref=pd.read_csv(OUT/'reference_per_window.csv');att=pd.read_csv(OUT/'attitude_attribution_per_window.csv')
    keys=['fold','seed','cohort','log_id','arm','block']
    components=['total_mse','bias_mse','amplitude_mse','shape_mse','signed_bias']
    qf=q.groupby(keys)[components].mean().reset_index();qf['rmse']=np.sqrt(qf.total_mse)
    np.testing.assert_allclose(qf.total_mse,qf.bias_mse+qf.amplitude_mse+qf.shape_mse,atol=1e-12)
    qf.to_csv(OUT/'q_decomposition_per_flight.csv',index=False)
    _,qs=summary(qf,['cohort','arm','block'],components+['rmse']);qs.to_csv(OUT/'q_decomposition_summary.csv',index=False)
    rf=[]
    for stratum,g in [('all',ref)]+list(ref.groupby('stratum')):
        f=g.groupby(keys).agg(windows=('array_index','size'),factual_mse=('factual_mse','mean'),held_mse=('held_mse','mean'),effect_mse=('effect_mse','mean'),activity=('reference_activity','mean')).reset_index()
        f['stratum']=stratum;f['factual_rmse']=np.sqrt(f.factual_mse);f['held_rmse']=np.sqrt(f.held_mse);f['effect_rms']=np.sqrt(f.effect_mse)
        f['utility']=1-f.factual_rmse/f.held_rmse;rf.append(f)
    rf=pd.concat(rf,ignore_index=True);rf.to_csv(OUT/'reference_per_flight.csv',index=False)
    _,rs=summary(rf,['cohort','arm','block','stratum'],['factual_rmse','held_rmse','effect_rms','utility','effect_mse','activity'])
    rs.to_csv(OUT/'reference_summary.csv',index=False)
    pairs=rf[rf.cohort=='oof'].pivot(index=['seed','log_id','block','stratum'],columns='arm',values=['windows','effect_rms','effect_mse','utility','activity','factual_rmse'])
    ratios=pairs['effect_rms']['aux']/pairs['effect_rms']['baseline']
    result=pd.DataFrame(dict(effect_ratio=ratios,baseline_effect=pairs['effect_rms']['baseline'],aux_effect=pairs['effect_rms']['aux'],
        delta_effect_mse=pairs['effect_mse']['aux']-pairs['effect_mse']['baseline'],utility_change=pairs['utility']['aux']-pairs['utility']['baseline'],
        baseline_q_rmse=pairs['factual_rmse']['baseline'],aux_q_rmse=pairs['factual_rmse']['aux'],windows=pairs['windows']['aux'],activity=pairs['activity']['aux'])).reset_index()
    result.to_csv(OUT/'reference_oof_pairs.csv',index=False)
    af=[];asm=[]
    for kind,columns in [('axis',['p','q','r']),('time',['early_20_100ms','late_120_1000ms'])]:
        part=att[att.kind==kind]
        values=['baseline_mse','aux_mse','recorded_rate_closure_mse']+columns
        f=part.groupby(['fold','seed','cohort','log_id','kind','horizon_s'])[values].mean().reset_index()
        f['delta_mse']=f.aux_mse-f.baseline_mse;np.testing.assert_allclose(f[columns].sum(axis=1),f.delta_mse,atol=1e-12)
        for name in ['baseline','aux','recorded_rate_closure']:f[name+'_rmse']=np.sqrt(f[name+'_mse'])
        f['rmse_gain']=1-f.aux_rmse/f.baseline_rmse;af.append(f)
        _,s=summary(f,['cohort','kind','horizon_s'],values+['delta_mse','baseline_rmse','aux_rmse','recorded_rate_closure_rmse']);asm.append(s)
    pd.concat(af,ignore_index=True).to_csv(OUT/'attitude_attribution_per_flight.csv',index=False)
    pd.concat(asm,ignore_index=True).to_csv(OUT/'attitude_attribution_summary.csv',index=False)
    # Exact coverage: every OOF parent appears once per seed/arm/block.
    oof=q[q.cohort=='oof'];assert not oof.duplicated(['seed','arm','block','array_index']).any()
    assert oof.groupby(['seed','arm','block']).array_index.nunique().eq(407).all()
    # Decomposition blocks partition the original time support (the full block is redundant).
    for (seed,arm),g in oof.groupby(['seed','arm']):
        full=g[g.block=='full1s'].set_index('array_index').total_mse.sort_index()
        total=np.zeros(407)
        for block,(start,end) in BLOCKS.items():
            if block=='full1s':continue
            total+=g[g.block==block].set_index('array_index').total_mse.sort_index().to_numpy()*(end-start)/50
        np.testing.assert_allclose(total,full,atol=1e-12)
    # Verify frozen input-only cuts against the original data and fold normalization.
    x,y,w=load_train()
    for cut in json.loads((OUT/'reference_cuts.json').read_text()):
        with np.load(OUT.parent/f"control_windows_v8/fold{cut['fold']}/normalization.npz") as z:activity=reference_activity(x['future_reference'].numpy(),z['reference_scale'])
        ids=np.array(cut['fit_indices']);expected=weighted_terciles(activity[ids],w.iloc[ids].log_id.to_numpy())
        np.testing.assert_array_equal(expected,cut['cuts'])
        g=ref[ref.fold==cut['fold']];np.testing.assert_array_equal(bins(activity[g.array_index],expected),g.stratum)
    # Reproduce the original reference-sensitivity scores, no new aggregation basis.
    prior=pd.read_csv(V18/'reference_per_flight.csv')
    current=rf[(rf.cohort=='oof')&(rf.block=='full1s')&(rf.stratum=='all')]
    merged=current.merge(prior,on=['fold','seed','log_id','arm'],suffixes=('_new','_old'),validate='one_to_one')
    discrepancy=max(float(abs(merged[k+'_new']-merged[k+'_old']).max()) for k in ['factual_rmse','held_rmse','utility','effect_rms']);assert discrepancy<1e-10
    for path in OUT.glob('fold*/*_attitude_seed*.npz'):
        with np.load(path) as z:
            np.testing.assert_allclose(z['axis_contributions'].sum(-1),z['aux_squared_angle']-z['baseline_squared_angle'],atol=1e-12)
            np.testing.assert_allclose(z['time_contributions'][:,:5,1],0,atol=1e-12)
    coverage=rf.groupby(['cohort','stratum','block']).windows.agg(['min','max','count']).reset_index();coverage.to_csv(OUT/'reference_coverage.csv',index=False)
    v=json.loads((OUT/'verification.json').read_text());v.update(original_reference_metrics_max_abs=discrepancy,
        oof_all407_unique=True,block_mse_identity=True,shapley_efficiency=True,late_tape_no_effect_on_first100ms=True,input_only_reference_cuts_verified=True)
    base.write(OUT/'verification.json',v)
    print('v19 summaries and independent consistency checks complete')


if __name__=='__main__':main()
