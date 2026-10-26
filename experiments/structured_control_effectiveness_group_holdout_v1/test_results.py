"""Post-run acceptance tests, separate from the pretraining gate."""
from common import *
import pytest

def test_completed_budget_and_independent_verification():
    d=json.loads((OUT/'training_completion.json').read_text());assert d['passed'] and d['completed']==27 and d['epochs_each']==24 and d['optimizer_updates']==57456
    d=json.loads((OUT/'independent_verification.json').read_text());assert d['passed'] and d['main_pairs_once_per_model_seed']==48 and not d['test_read']
    assert json.loads((OUT/'inference_checks.json').read_text())['models_checked']==27

def test_train_validation_log_disjoint():
    ids=[]
    for split in ['train','validation']:
        m=pd.read_parquet(TRAIN/f'{split}_windows.parquet');ids.append({f.split(':')[0] for f in m.flight_id})
    assert ids[0].isdisjoint(ids[1])

def test_old_sources_unchanged():
    receipt=json.loads((OUT/'source_receipt.json').read_text())
    for p,h in receipt['files'].items():assert sha(p)==h,p

def test_exact_membership_and_support():
    d=pd.read_csv(OUT/'factual_points.csv.gz');a=pd.read_csv(OUT/'holdout_pair_assignments.csv');p=d[(d.pair_uid.isin(a.pair_uid))&(d.time_label=='300')&(d.variable=='p')]
    assert p.groupby(['model','seed','method']).pair_uid.nunique().eq(48).all() and p.supported.all()
    assert not p.duplicated(['pair_uid','model','seed','method']).any()
    support=pd.read_csv(OUT/'evaluated_pair_support.csv');assert support[support.role=='heldout'].both_flights_unseen.all()
    unavailable=pd.read_csv(OUT/'unevaluable_background_pairs.csv');pairs=pd.read_csv(OUT/'pairs.csv');plan=json.loads((OUT/'holdout_protocol.json').read_text())
    eligible={u for f in plan['folds'] for u in f['eligible_all_frozen_pairs']};expected=set(pairs[pairs.split=='train'].pair_uid)-eligible
    assert set(unavailable.pair_uid)==expected and len(expected)==5 and expected.isdisjoint(a.pair_uid)
    other_validation=pd.read_csv(OUT/'excluded_validation_background.csv')
    assert set(other_validation.pair_uid)==set(pairs[pairs.split=='validation'].pair_uid)-{'validation:44'} and len(other_validation)==2

def test_fold_support_does_not_filter():
    d=pd.read_csv(OUT/'fold_control_support.csv');p=pd.read_csv(OUT/'evaluated_pair_support.csv')
    assert d.groupby(['fold','pair_uid']).size().eq(4).all()
    assert set(zip(d.fold,d.pair_uid))==set(zip(p.fold,p.pair_uid))
    assert json.loads((OUT/'fold_support_reference.json').read_text())['excluded_by_support']==0

def test_all_saved_predictions_preserve_integrator():
    x=np.load(SWAP/'results/inputs.npz')
    for fold in [0,1,2]:
        z=np.load(OUT/f'predictions_fold{fold}.npz');ids=z['case_ids'];dt=x['dt'][ids].astype(float)
        for key in z.files:
            if key=='case_ids':continue
            y=z[key].astype(float);assert y.shape==(len(ids),dt.shape[1],13) and np.isfinite(y).all()
            previous=np.concatenate([x['initial'][ids,None].astype(float),y[:,:-1]],axis=1)
            position=previous[...,:3]+.5*(previous[...,3:6]+y[...,3:6])*dt[...,None]
            np.testing.assert_allclose(y[...,:3],position,rtol=0,atol=3e-5)
            np.testing.assert_allclose(np.linalg.norm(y[...,6:10],axis=-1),1.,rtol=0,atol=2e-6)

def test_native_clock_and_sealed_partition_guard():
    with pytest.raises(AssertionError,match='test remains sealed'):load_numpy('test')
    x=np.load(SWAP/'results/inputs.npz');cases=pd.read_csv(OUT/'cases.csv')
    for c in cases.itertuples():
        n=int(x['valid_steps'][c.case_id]);time=x['native_time_us'][c.case_id,:n+1]
        assert time[0]==0 and (np.diff(time)>0).all()
        np.testing.assert_array_equal(np.rint(x['dt'][c.case_id,:n].astype(float)*1e6).astype(np.int64),np.diff(time))
        assert time[-1]*1e-6<=c.support_end_s+1e-9
