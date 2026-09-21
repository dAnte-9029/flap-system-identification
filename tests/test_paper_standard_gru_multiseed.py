"""Reject sample pooling, missing seeds and changed RNG/normalization contracts."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from report_paper_standard_gru_multiseed import across_seeds, METRICS
from run_paper_standard_gru_multiseed import stages, normalization_hash, STATS, same_tree
from system_identification.evaluation.paper_baselines import aggregate_flights, METRIC_MAP


def grid():
    rows=[]
    for seed,value in [(17,1.),(23,2.),(42,3.)]:
        for cohort in ['ALL','Sep7','Sep17']:
            for h in [.1,.2,.5,1.]:
                rows.append(dict(seed=seed,cohort=cohort,horizon_s=h,**{m:value for m in METRICS}))
    return pd.DataFrame(rows)


def test_three_seed_sample_sd_and_missing_duplicate_rejection():
    frame=grid(); result=across_seeds(frame)
    np.testing.assert_allclose(result['mean'],2)
    np.testing.assert_allclose(result['std'],1)
    assert (result.n_seeds==3).all()
    with pytest.raises(ValueError):across_seeds(frame.iloc[1:])
    with pytest.raises(ValueError):across_seeds(pd.concat([frame,frame.iloc[:1]]))
    frame.loc[0,'seed']=99
    with pytest.raises(ValueError):across_seeds(frame)


def test_each_seed_uses_equal_flight_means_before_seed_sd():
    rows=[]
    for seed,offset in [(17,0),(23,1),(42,2)]:
        for cohort in ['Sep7','Sep17']:
            for h in [.1,.2,.5,1.]:
                for log,n,value in [('many',99,1+offset),('few',1,3+offset)]:
                    rows.extend(dict(model='B2',seed=seed,cohort=cohort,log_id=cohort+log,horizon_s=h,
                        **{m:value for m in METRIC_MAP}) for _ in range(n))
    _,summary,_=aggregate_flights(pd.DataFrame(rows))
    aggregate=across_seeds(summary)
    np.testing.assert_allclose(aggregate['mean'],3.)
    np.testing.assert_allclose(aggregate['std'],1.)


def test_seed_offset_and_fixed_normalization_hash():
    assert [s[2] for s in stages(17)]==[17,29]
    assert [s[2] for s in stages(23)]==[23,35]
    assert [s[2] for s in stages(42)]==[42,54]
    with pytest.raises(ValueError):stages(1)
    stats={k:np.array([1,2],dtype=np.float32) for k in STATS}
    digest=normalization_hash(stats)
    assert normalization_hash(dict(reversed(list(stats.items()))))==digest
    stats['feature_mean'][0]+=1
    assert normalization_hash(stats)!=digest
    with pytest.raises(ValueError):same_tree(np.array([1.]),np.array([2.]))


def test_report_pipeline_smoke(tmp_path):
    from report_paper_standard_gru_multiseed import generate
    summary=grid()
    # Synthetic fixture only, never written to scientific result directories.
    for metric in METRICS:
        summary[metric]*=summary.horizon_s
    summary.to_csv(tmp_path/'per_seed_summary.csv',index=False)
    pd.DataFrame([dict(seed=s,checkpoint_sha256='synthetic',final_epoch=65) for s in [17,23,42]]).to_csv(tmp_path/'training_runs.csv',index=False)
    protocol=dict(architecture='synthetic test',manifest_sha256='test',train_window_sha256='test',
        validation_origin_sha256='test',normalization_sha256='test',seed17_rng_provenance='fixture',
        robustness_screen=dict(max_primary_cv=.2,max_1s_cv=.2))
    generate(tmp_path,protocol)
    assert len(pd.read_csv(tmp_path/'multiseed_summary.csv'))==48
    assert (tmp_path/'report.md').stat().st_size>1000
    assert len(list(tmp_path.glob('*.png')))==8
    assert len(list(tmp_path.glob('*.pdf')))==8
