import ast
import numpy as np
import pandas as pd
import pytest
from common import *

def test_interpolation_no_extrapolation():
 t=np.array([0,.01,.02]);y=np.c_[t,t*2,t*0]
 got=interp(t,y,[-.001,0,.005,.02,.021]);assert np.isnan(got[[0,4]]).all();np.testing.assert_allclose(got[2],[.005,.01,0])

def test_monotonic_and_gap():
 with pytest.raises(AssertionError):interp(np.array([0,.01,.01]),np.ones((3,3)),[0])
 assert np.isnan(interp(np.array([0,.1]),np.ones((2,3)),[.05])).all()
 assert np.isnan(boxmean(np.array([0,.1]),np.ones((2,3)),[.05],.1)).all()

def test_constant_filter_boundary():
 t=np.arange(101)/100;y=np.ones((101,3))*np.array([2,-3,0])
 out=boxmean(t,y,[0,.135,.5,.865,1],.27)
 assert np.isnan(out[[0,4]]).all();np.testing.assert_allclose(out[1:4],np.tile([2,-3,0],(3,1)),atol=1e-14)

def test_synthetic_period_and_linear_signal():
 t=np.linspace(0,1,1001);period=.25;y=np.c_[np.sin(2*np.pi*t/period),np.cos(2*np.pi*t/period),2*t+1]
 at=np.array([.25,.5,.75]);out=boxmean(t,y,at,period)
 np.testing.assert_allclose(out[:,:2],0,atol=1e-14);np.testing.assert_allclose(out[:,2],2*at+1,atol=1e-14)

def test_additive_decomposition_and_rebased_origin():
 t=np.linspace(0,.5,51);y=np.c_[np.sin(20*t)+2,t*3,t**2];at=np.array([.135,.2,.3,.365])
 raw=interp(t,y,at)-y[0];slow=boxmean(t,y,at,.27)-y[0];fast=raw-slow
 np.testing.assert_allclose(slow+fast,raw);cycle=slow-slow[0];np.testing.assert_array_equal(cycle[0],np.zeros(3))
 assert np.any(abs(slow[0])>0) # Preserving an instantaneous anchor does not force filtered response to zero.

def test_physical_origin():
 t=np.array([-.01,.01,.03]);y=np.c_[3*t+4,t*0,t*0];at=np.array([0,.01,.02]);v=interp(t,y,at)-interp(t,y,[0])[0]
 np.testing.assert_array_equal(v[0],np.zeros(3));np.testing.assert_allclose(v[:,0],3*at,atol=1e-14)

def test_train_only_design_and_split_isolation():
 cfg=json.loads((HERE/'config.json').read_text());w=pd.read_csv(OLD/'results/windows.csv');p=pd.read_csv(OLD/'results/pairs.csv');m=pd.read_csv(OLD/'results/cohorts.csv')
 ids=m.query("cohort=='main' and split=='train'").pair_uid;pp=p[p.pair_uid.isin(ids)];ix=np.unique(pp[['a','b']].to_numpy());f=w.iloc[ix].history_frequency
 assert cfg['main_width_s']==round(1/f.median(),2) and cfg['sensitivity_width_s']==round(1/f.min(),2)
 assert not set(w[w.split=='train'].flight_id)&set(w[w.split=='validation'].flight_id)
 assert len(pp)==48 and len(m.query("cohort=='main' and split=='validation'"))==1

def test_no_inference_and_immutable_inputs():
 receipt=json.loads((RESULTS/'source_receipt.json').read_text());z=np.load(OLD/'results/native_rollouts.npz')
 for k,h in receipt['input_keys_hashes'].items():assert hashlib.sha256(z[k].tobytes()).hexdigest()==h
 for name in ['analyze.py','prepare.py','baseline.py','common.py']:
  source=(HERE/name).read_text();tree=ast.parse(source)
  for node in ast.walk(tree):
   if isinstance(node,ast.Import):assert all(x.name.split('.')[0]!='torch' for x in node.names)
   if isinstance(node,ast.ImportFrom):assert not (node.module or '').startswith('torch')
   if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):assert node.func.attr not in ['forward','backward','load_state_dict']
 assert not receipt['test_read']


def test_roundoff_at_supported_endpoint():
 t=np.array([0,.01,.02]);y=np.c_[t,t,t]
 np.testing.assert_allclose(interp(t,y,[.02+1e-17])[0],[.02]*3)
 assert np.isnan(interp(t,y,[.02001])).all()

def test_unknown_encoder_measurement_time_not_fabricated():
 phase=pd.read_csv(RESULTS/'phase_forecast_audit.csv')
 assert not phase.encoder_measurement_time_known.any()
 assert not phase.frequency_measurement_time_known.any()
 assert phase.encoder_source_age_at_origin_us.abs().max()<200000
 assert phase.frequency_source_age_at_origin_us.abs().max()<200000
