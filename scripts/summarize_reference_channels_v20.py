"""Summarize all channel/support cases and verify frozen predecessors."""
import json
import numpy as np
import pandas as pd
import train_control_windows_v2 as base
from reference_channels_v20 import MASKS,CHANNELS
from reference_increment_v16 import safe_corr
from run_reference_channels_v20 import OUT,V19
from run_reference_generalization_v7 import verify_manifest


def main():
    verify_manifest(OUT/'protocol.json',['source_pins'])
    d=pd.read_csv(OUT/'response_per_window.csv');support=pd.read_csv(OUT/'support_per_window.csv');f=pd.read_csv(OUT/'input_features.csv')
    vals=['signed_bias','bias_mse','mse','versus_held_mse','versus_factual_mse','nonadditive_mse']
    keys=['fold','seed','cohort','log_id','arm','mask','block'];records=[]
    for stratum,g in [('all',d)]+list(d.groupby('stratum')):
        a=g.groupby(keys)[vals].mean().reset_index();a['windows']=g.groupby(keys).size().to_numpy();a['stratum']=stratum
        for name in ['bias','', 'versus_held','versus_factual','nonadditive']:
            mse=(name+'_' if name else '')+'mse';a[(name+'_' if name else '')+'rmse']=np.sqrt(a[mse])
        records.append(a)
    pf=pd.concat(records,ignore_index=True);pf.to_csv(OUT/'response_per_flight.csv',index=False)
    metrics=vals+['bias_rmse','rmse','versus_held_rmse','versus_factual_rmse','nonadditive_rmse']
    common=['cohort','arm','mask','block','stratum'];source=pf.groupby(common+['log_id','seed'])[metrics].mean().reset_index()
    source.groupby(common)[metrics].mean().reset_index().to_csv(OUT/'response_summary.csv',index=False)
    pairs=pf[(pf.cohort=='oof')].pivot(index=['seed','log_id','mask','block','stratum'],columns='arm',values=['versus_held_rmse','versus_factual_rmse','signed_bias','bias_rmse','rmse','nonadditive_rmse','windows'])
    o=pd.DataFrame({f'{metric}_{arm}':pairs[metric][arm] for metric in ['versus_held_rmse','versus_factual_rmse','signed_bias','bias_rmse','rmse','nonadditive_rmse','windows'] for arm in ['baseline','aux']}).reset_index()
    o.to_csv(OUT/'oof_channel_pairs.csv',index=False)
    meta=d[(d.cohort=='oof')&(d.seed==17)&(d.arm=='baseline')&(d["mask"]=='factual')&(d.block=='full1s')][['fold','array_index','stratum']]
    support=support.merge(meta,on=['fold','array_index'],validate='many_to_one');sr=[]
    for st,g in [('all',support)]+list(support.groupby('stratum')):
        a=g.groupby(['fold','log_id','mask','group']).agg(windows=('array_index','size'),outside_fraction=('outside','mean'),median_distance=('distance','median'),max_distance=('distance','max'),threshold=('threshold','first')).reset_index();a['stratum']=st;sr.append(a)
    pd.concat(sr,ignore_index=True).to_csv(OUT/'support_per_flight.csv',index=False)
    bounds=pd.read_csv(OUT/'feature_bounds.csv');bounds=bounds.merge(meta,on=['fold','array_index'],validate='many_to_one');br=[]
    for st,g in [('all',bounds)]+list(bounds.groupby('stratum')):
        a=g.groupby(['fold','log_id','feature']).agg(windows=('array_index','size'),query_min=('value','min'),query_max=('value','max'),query_mean=('value','mean'),fit_min=('fit_min','first'),fit_max=('fit_max','first'),outside_minmax=('outside_minmax','mean'),outside_01_99=('outside_01_99','mean')).reset_index();a['stratum']=st;br.append(a)
    pd.concat(br,ignore_index=True).to_csv(OUT/'bounds_per_flight.csv',index=False)
    factual=d[(d.cohort=='oof')&(d["mask"]=='factual')&(d.block=='full1s')];merged=factual[factual.arm=='baseline'].merge(factual[factual.arm=='aux'],on=['fold','seed','array_index','log_id'],suffixes=('_base','_aux'),validate='one_to_one')
    corr=[]
    for (fold,seed,log),g in merged.groupby(['fold','seed','log_id']):
        ids=g.array_index.to_numpy()
        for feature in f:
            ranks=f.iloc[ids][feature].rank()
            corr.append(dict(fold=fold,seed=seed,log_id=log,feature=feature,n=len(g),signed_bias_change_spearman=safe_corr(ranks,(g.signed_bias_aux-g.signed_bias_base).rank()),bias_square_change_spearman=safe_corr(ranks,(g.bias_mse_aux-g.bias_mse_base).rank())))
    pd.DataFrame(corr).to_csv(OUT/'bias_change_conditions.csv',index=False)
    oof=d[d.cohort=='oof'];assert not oof.duplicated(['seed','arm','mask','block','array_index']).any()
    assert oof.groupby(['seed','arm','mask','block']).array_index.nunique().eq(407).all()
    prior=pd.read_csv(V19/'reference_per_flight.csv');prior=prior[(prior.cohort=='oof')&(prior.block=='full1s')&(prior.stratum=='all')]
    current=pf[(pf.cohort=='oof')&(pf["mask"]=='factual')&(pf.block=='full1s')&(pf.stratum=='all')]
    audit=current.merge(prior,on=['fold','seed','log_id','arm'],validate='one_to_one')
    discrepancy=float(abs(audit.versus_held_rmse-audit.effect_rms).max());assert discrepancy<1e-10
    # New input-only feature and support cases cannot alter original strata/parents.
    fit_map={r['fold']:set(r['fit_indices']) for r in json.loads((OUT/'support_calibration.json').read_text())}
    for fold,g in support.groupby('fold'):
        assert set(g.nearest_fit_array_index).issubset(fit_map[fold]);assert set(g.array_index).isdisjoint(fit_map[fold])
    # Preserve complete implicated flights, not only the high-error windows.
    review=d[(d.cohort=='oof')&((d.log_id.str.contains('log_18_')&(d.seed==23))|(d.log_id.str.contains('log_28_')&d.seed.isin([23,42])))&(d.block=='full1s')]
    review.to_csv(OUT/'review_windows.csv',index=False)
    v=json.loads((OUT/'verification.json').read_text());v.update(oof_unique407=True,predecessor_response_max_abs=discrepancy,nearest_neighbors_exclude_query_flight=True)
    base.write(OUT/'verification.json',v)
    print('v20 channel/support summaries and checks complete')


if __name__=='__main__':main()
