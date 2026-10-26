import numpy as np
import pandas as pd
import pytest
from extract import asof,interp,scope,old,RES,CFG,HERE
from analyze import ridge,predict,metrics

def test_coordinates_sign_and_inverse():
 u=np.array([[.7,-.2,.4,.05]])
 x=scope.coords(u);assert np.allclose(x,[.7,.1,.3,.05])
 assert np.allclose(np.c_[x[:,0],x[:,1]-x[:,2],x[:,1]+x[:,2],x[:,3]],u)
def test_causal_asof_future_poison():
 t=np.array([10,20,30]);x=np.arange(3.)[:,None];a=np.array([9,10,19,20,25])
 y,age=asof(t,x,a);z,_=asof(np.r_[t,40],np.r_[x,[[999]]],a)
 assert np.allclose(y,z,equal_nan=True);assert np.isnan(y[0]);assert y[-1]==1
 assert (age[1:]>=0).all()
def test_reference_expiry():
 y,_=asof(np.array([10]),np.array([[2.]]),np.array([10,21]),10)
 assert y[0]==2 and np.isnan(y[1])
def test_no_extrapolation_and_gap():
 t=np.array([0,20000,100000]);x=np.array([[0.],[2.],[10.]])
 z=interp(t,x,np.array([-1,0,10000,30000,100001]));assert np.isnan(z[[0,3,4]]).all();assert z[2]==1
 # Conservative gap rule also masks the right sample of a wide bracket.
 assert np.isnan(interp(t,x,np.array([100000]))[0])
 assert interp(np.array([0,20000]),np.array([[1.],[1.]]),np.array([0,10000,20000])).tolist()==[[1.],[1.],[1.]]

def test_zoh_redundant_segmentation():
 t=np.array([0,100,200,300]);u=np.array([[0.,0,0,0],[0.,0,0,0],[1.,2,3,4]])
 a=scope.integral_stats(t,u,[.1]*4)
 b=scope.integral_stats(np.array([0,200,300]),u[[0,2]],[.1]*4)
 for x,y in zip(a,b):assert np.array_equal(x,y)
def test_split_guard():
 with pytest.raises(ValueError):old.permitted('test')
def test_ridge_group_reproducibility_and_finite():
 x=np.arange(24.).reshape(12,2);x[0,0]=np.nan;y=np.arange(12.)[:,None];g=np.repeat(['a','b','c'],4)
 model=ridge(x,y,g,.01);p=predict(model,x);assert np.isfinite(p).all();assert p.shape==y.shape
 assert np.mean((p-y)**2)<.1

def test_residual_not_exogeneity_certificate():
 # An omitted nonlinear deterministic feedback can leave large residuals.
 x=np.linspace(-1,1,101)[:,None];y=x*x
 p=predict(ridge(x,y,np.repeat('a',101),.01),x)
 assert np.std(y-p)>.2

def test_information_svd_gram():
 rng=np.random.default_rng(17);x=rng.normal(size=(100,4));x[:,3]=x[:,0]+x[:,1];x-=x.mean(0)
 s=np.linalg.svd(x/10,compute_uv=False);e=np.sqrt(np.maximum(np.linalg.eigvalsh(x.T@x/100),0))[::-1]
 assert np.allclose(s*s,e*e,atol=1e-12);assert (s>s[0]*.01).sum()==3

def test_negative_control_sign():
 time=np.array([-.3,0,.3]);p=2*time
 past=p[1]-p[0];future=p[2]-p[1]
 assert past==pytest.approx(.6) and future==pytest.approx(.6)
 # Negative lag means completed past interval, not negating its change.
 assert past==future

def test_frozen_indices_and_all_folds():
 m=pd.read_csv(RES/'windows.csv.gz');assert m.groupby('split').size().to_dict()=={'train':56080,'validation':14269}
 assert m.groupby('split').nonoverlap.sum().to_dict()=={'train':6838,'validation':1763}
 a=pd.read_csv(RES/'crossfit_groups.csv');assert set(a.fold)==set(range(5));assert a.log_sha.is_unique
 coh=pd.read_csv(RES/'frozen_cohorts.csv');assert len(coh[(coh.split=='train')&(coh.cohort=='main')])==48
 assert len(coh[(coh.split=='train')&(coh.cohort=='flight_disjoint')])==17

def test_independent_verification_passed():
 import json
 d=json.loads((RES/'independent_verification.json').read_text());assert d['passed'];assert d['zoh_integrals']==1200

def test_only_declared_causal_feature_families():
 from analyze import feature_columns
 f=pd.DataFrame(columns=['U0_differential','S0_p','R0_vehicle_rates_setpoint_roll','mission_changed_500','future_p','target_p','future_phase'])
 cols=feature_columns(f,'SU_reference500')
 assert 'future_p' not in cols and 'target_p' not in cols and 'future_phase' not in cols
 assert set(cols)=={'U0_differential','S0_p','R0_vehicle_rates_setpoint_roll','mission_changed_500'}
