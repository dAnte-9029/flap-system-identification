"""Directed contracts, not statistical evidence about the real aircraft."""
import unittest
import numpy as np
from px4_clid_extract import zoh_integral
from px4_clid_models import *
class Contracts(unittest.TestCase):
 def test_zoh(self):
  t=np.array([0,10000,30000,60000]);v=np.array([[0.],[2.],[4.],[0.]])
  a,ok=zoh_integral(t,v,np.array([20000,40000]));np.testing.assert_allclose(a[:,0],[1,3]);assert ok.all()
 def test_nan_before_valid_interval(self):
  a,ok=zoh_integral(np.array([0,20000,40000,60000]),np.array([[np.nan],[1.],[2.],[3.]]),np.array([20000,40000,60000]));assert not ok[0];assert ok[1:].all();np.testing.assert_allclose(a[1:,0],[1,2])
 def test_delay_sign_and_no_other_channel(self):
  for d in [0,1,2]:
   t=np.zeros((1+9+20,3));t[1+9+1,0]=2
   m=dict(theta=t,c=np.ones(1),cfg=(3,5,d,1e-4))
   p=pulse(m);assert np.flatnonzero(p[:,0])[0]==d;assert np.all(p[:,1:]==0);assert p[d,0]==2
 def test_no_future_state(self):
  rng=np.random.default_rng(1);s=dict(y=rng.normal(size=(100,3)),u=rng.normal(size=(100,4)))
  t=rng.normal(size=(30,3))*.01;m=dict(theta=t,c=np.array([1,.4]),cfg=(3,5,0,1e-4))
  origins,a=rollout(m,s);origin=origins[0];ss=dict(s,y=s['y'].copy());ss['y'][origin+1:]+=123
  _,b=rollout(m,ss);np.testing.assert_allclose(a[0],b[0]);np.testing.assert_array_equal(a,rollout(m,s)[1])
 def test_causal_innovation(self):
  s=dict(y=np.ones((100,3)),u=np.zeros((100,4)));m=dict(theta=np.zeros((30,3)),c=np.array([1,.5]),cfg=(3,5,0,1e-4))
  e,_=innovations(m,s);np.testing.assert_allclose(e[:3,0],[1,.5,.75])
 def test_invertible(self):
  for z in [[-.95],[.95],[.95,-.95],[-.95,.95]]:assert max(abs(np.roots(noise(z))))<1
 def test_gap_break(self):
  a,ok=zoh_integral(np.array([0,10000,100000]),np.ones((3,1)),np.array([80000,100000]));assert not ok.any()
 def test_sign_profile_refits(self):
  rng=np.random.default_rng(42);s=dict(y=rng.normal(size=(600,3)),u=rng.normal(size=(600,4)))
  cfg=(3,5,0,1e-3);b=blocks_for([s],cfg);m=fit(b,cfg);v=abs(pulse(m)[-1,0])
  for sign in [-1,1]:
   q=sign_refit(m,b,sign);assert abs(pulse(q)[-1,0]-sign*v)<1e-5;assert q['rho']<1
 def test_synthetic_recovery(self):
  rng=np.random.default_rng(19);u=rng.normal(size=(5000,4));y=np.zeros((5000,3));eps=rng.normal(size=y.shape)*.02
  for k in range(1,len(y)):y[k]=.6*y[k-1]+u[k-1,:3]*[.3,-.2,.1]+eps[k]+.4*eps[k-1]
  m=fit(blocks_for([dict(y=y,u=u)],(3,5,1,1e-5)),(3,5,1,1e-5),1)
  assert abs(m['c'][1]-.4)<.1;assert m['rho']<1;assert abs(m['theta'][10,0]-.3)<.01
if __name__=='__main__':unittest.main()
