import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from run_response_timescale_diagnostic import lowpass,channel_scores,isolated_masks

def test_constant_and_native_step():
    dt=np.array([[.01,.04]]);x=np.ones((1,3,1))*3
    np.testing.assert_array_equal(lowpass(x,dt,.1),x)
    x[:,0]=0;y=lowpass(x,dt,.1)
    np.testing.assert_allclose(y[0,-1,0],3*(1-np.exp(-.05/.1)))
def test_causal_prefix():
    x=np.arange(9.).reshape(1,9,1);dt=np.ones((1,8))*.02
    np.testing.assert_array_equal(lowpass(x,dt,.1)[:,:5],lowpass(x[:,:5],dt[:,:4],.1))
def test_error_identity():
    rng=np.random.default_rng(17);a=rng.normal(size=(3,51,3));b=rng.normal(size=a.shape);dt=np.full((3,50),.02)
    e=a-b;s=lowpass(a,dt,.1)-lowpass(b,dt,.1);f=e-s
    np.testing.assert_allclose(e**2,s**2+f**2+2*s*f,atol=1e-12)
def test_isolation_not_coactivation():
    x=np.array([[2.,0,0,0],[2,2,0,0],[0,0,0,0]])
    m=isolated_masks(x,np.full(4,.5),np.ones(4))
    np.testing.assert_array_equal(m['drive']['isolated'],[True,False,False])
def test_activity_has_no_state_dependency():
    u=np.zeros((2,50,4));u[1,1:25,0]=1
    a,mean,p=channel_scores(u,np.ones(4))
    assert a[1,0]>0 and p[1,0]>.8 and np.isnan(p[0]).all()
