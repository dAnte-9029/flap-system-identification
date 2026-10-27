from common import *
import pytest

def test_complete_fixed_training():
 audit=json.loads((OUT/'training_completion.json').read_text());assert audit['passed'] and audit['completed']==12 and audit['epoch_records']==72 and audit['optimizer_updates']==6384
 h=pd.read_csv(OUT/'training_history.csv');assert h.groupby(['fold','model']).size().eq(6).all();assert h.groupby(['fold','epoch']).batch_order_sha256.nunique().eq(1).all();assert h.groupby(['fold','epoch']).updates.nunique().eq(1).all()

@pytest.mark.parametrize('method,count',[('B',40),('D',36)])
def test_frozen_primary_support(method,count):
 f=pd.read_csv(OUT/'factual_points.csv.gz');ids=set(pd.read_csv(OUT/'holdout_pair_assignments.csv').pair_uid);f=f[f.pair_uid.isin(ids)&(f.time_label=='300')&(f.variable=='p')&(f.method==method)]
 for name,g in f.groupby('model'):
  assert len(g)==48 and g.pair_uid.nunique()==48 and g.supported.all() and (abs(g.real)>.1).sum()==count
  assert g.seed.eq(17).all()
 assert f.groupby('pair_uid').real.nunique().eq(1).all()

def test_pairwise_structure_kept_fixed():
 d=pd.read_csv(OUT/'paired_changes.csv.gz');assert set(d.model)=={'M0W1','M1W1'}
 assert (d.model.str[:2]==d.model_W0.str[:2]).all() and d.model_W0.str.endswith('W0').all()
 np.testing.assert_allclose(d.real,d.real_W0,rtol=0,atol=0,equal_nan=True)
 np.testing.assert_array_equal(d.supported,d.supported_W0)

def test_independent_audit_and_no_test():
 a=json.loads((OUT/'independent_verification.json').read_text());assert a['passed'] and not a['test_read'] and a['components']==11
 assert a['main_pairs_once_per_model_seed']==48 and a['trajectory_checked_windows']==768
 assert a['closure_max_error']<1e-12 and a['summary_max_error']<1e-11
