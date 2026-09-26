import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from run_local_rate_error_decomposition import decompose

def test_exact_decomposition_native_dt():
 rng=np.random.default_rng(17);p=rng.normal(size=(7,5,3));t=rng.normal(size=p.shape);dt=rng.uniform(.01,.03,size=(7,5))
 a=decompose(p,t,dt);np.testing.assert_allclose(a['mse'],a['bias_mse']+a['amplitude_mse']+a['shape_mse'],atol=1e-12)
def test_constant_bias_only():
 t=np.arange(5.)[None,:,None];a=decompose(t+2,t,np.ones((1,5)))
 np.testing.assert_allclose(a['bias_mse'],4);np.testing.assert_allclose(a['amplitude_mse']+a['shape_mse'],0,atol=1e-12)
def test_amplitude_only_centered():
 t=np.array([-2.,-1,0,1,2])[None,:,None];a=decompose(t*.5,t,np.ones((1,5)))
 np.testing.assert_allclose(a['mse'],a['amplitude_mse'],atol=1e-12)
def test_constant_correlation_undefined():
 a=decompose(np.ones((1,5,1)),np.ones((1,5,1)),np.ones((1,5)));assert np.isnan(a['corr']).all()
