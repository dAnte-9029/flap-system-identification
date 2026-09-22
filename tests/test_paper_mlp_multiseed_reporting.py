"""Statistical contracts: no pooled windows, fabricated B0 seeds, or ensembles."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from report_paper_mlp_multiseed_comparison import (
    METRICS,B0,MLP,GRU,aggregate_seeds,paired_seeds,paired_flights,gains_table,relative_gain,generate)


def fixture():
    rows=[]
    # Unequal origin counts make pooled and equal-flight results very different.
    for model in [B0,MLP,GRU]:
        for seed in ([None] if model==B0 else [17,23,42]):
            for horizon in [.1,.2,.5,1.]:
                for flight,cohort,n,base in [('a','Sep7',1,2.),('b','Sep17',99,8.)]:
                    error=base+(0 if seed is None else {17:0,23:1,42:2}[seed])
                    if model==GRU:error+=-1 if flight=='a' else 2
                    rows.append(dict(model=model,seed=seed,flight_id=flight,cohort=cohort,horizon_s=horizon,n_windows=n,
                                     **{m:error for m in METRICS}))
    per=pd.DataFrame(rows);summaries=[]
    for (model,seed,horizon),g in per.groupby(['model','seed','horizon_s'],dropna=False):
        for cohort in ['ALL','Sep7','Sep17']:
            f=g if cohort=='ALL' else g[g.cohort==cohort]
            summaries.append(dict(model=model,seed=seed,cohort=cohort,horizon_s=horizon,n_flights=len(f),n_windows=f.n_windows.sum(),
                                  **{m:f[m].mean() for m in METRICS}))
    return per,pd.DataFrame(summaries)


def test_macro_seed_sd_and_deterministic_reference():
    per,summary=fixture();a=aggregate_seeds(summary)
    row=a[(a.model==MLP)&(a.cohort=='ALL')].iloc[0]
    assert row['mean']==6 and row['std']==1 and row.n_seeds==3
    assert row['mean']!=np.sqrt((3**2+99*9**2)/100)
    assert a[a.model==B0]['std'].isna().all()
    assert (a[a.model==B0].n_seeds==0).all()
    with pytest.raises(AssertionError):aggregate_seeds(pd.concat([summary,summary[summary.model==B0]]))
    with pytest.raises(AssertionError):aggregate_seeds(summary[~((summary.model==MLP)&(summary.seed==42))])


def test_paired_directions_zero_denominator_and_flight_means():
    per,summary=fixture();s=paired_seeds(summary);f=paired_flights(per)
    assert set(f[f.flight_id=='a'].absolute_gain)=={1.}
    assert set(f[f.flight_id=='b'].absolute_gain)=={-2.}
    assert set(s[s.cohort=='ALL'].absolute_gain)=={-.5}
    gains=gains_table(aggregate_seeds(summary),s,f)
    g=gains[gains.cohort=='ALL'].iloc[0]
    assert g.seeds_gru_better==0 and g.seeds_mlp_better==3
    assert g.flights_gru_better==g.flights_mlp_better==1
    assert g.relative_gain_pct==relative_gain(6,6.5)
    assert g.relative_gain_pct!=s[s.cohort=='ALL'].relative_gain_pct.mean()
    assert np.isnan(relative_gain(0,1)) and relative_gain(2,1)==50
    with pytest.raises(AssertionError):paired_flights(per.drop(per[(per.model==GRU)&(per.seed==42)].index))


def test_full_report_smoke_and_pooled_summary_rejected(tmp_path):
    per,summary=fixture()
    per.to_csv(tmp_path/'per_flight_wide.csv',index=False);summary.to_csv(tmp_path/'per_seed_summary.csv',index=False)
    pd.DataFrame([dict(model=MLP,seed=23,status='complete',final_epoch=65,final_train_loss=1.,training_time_s=1.,checkpoint_sha256='synthetic')]).to_csv(tmp_path/'training_runs.csv',index=False)
    p=dict(git_commit='synthetic',branch='synthetic',train_flights=['train'],validation_flights=['a','b'])
    generate(tmp_path,p)
    assert len(list(tmp_path.glob('*.pdf')))==5
    assert len(pd.read_csv(tmp_path/'paired_seed_differences.csv'))==144
    assert pd.read_csv(tmp_path/'multiseed_summary.csv').query('model==@B0')['std'].isna().all()
    summary.loc[(summary.model==MLP)&(summary.cohort=='ALL'),'velocity_rmse_m_s']=99
    summary.to_csv(tmp_path/'per_seed_summary.csv',index=False)
    with pytest.raises(AssertionError):generate(tmp_path,p)
