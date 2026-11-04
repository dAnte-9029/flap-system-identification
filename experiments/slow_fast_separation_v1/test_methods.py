import numpy as np
from methods import cell,fit,causal_template,weights,GRID

def fixture():
    cs=[]
    for i in range(81):
        t=i*.25+np.linspace(.006,.244,16);p=(t-i*.25)*8*np.pi
        cs.append(cell(t,p,1+.5*t+.1*np.cos(p),i*.25,(i+1)*.25))
    return cs

def test_grid_missing():
    c=cell(np.array([.1,.15,.2]),np.array([2.,3.,4.]),np.ones(3),0,.25)
    assert np.isnan(c['g'][0]) and np.isnan(c['g'][-1])

def test_linear():
    cs=fixture()
    for i,c in enumerate(cs):cs[i]=cell(c['t'],c['p'],1+.5*c['t'],c['lo'],c['hi'])
    assert np.max(abs(fit(cs,40)[4]))<1e-12
    for c in cs:assert abs(c['mu']-(1+.5*c['tau']))<1e-12

def test_holdout():
    cs=fixture();before=fit(cs,42,True)
    for i,c in enumerate(cs):
        if i%5==2:
            c['y']=c['y']+1e6;c['g']=c['g']+1e6;c['mu']+=1e6
    after=fit(cs,42,True)
    for i in [0,1,2,3,4]:np.testing.assert_allclose(before[i],after[i],atol=1e-12)

def test_mean_and_causal():
    cs=fixture();b=fit(cs,40)[4];assert abs(cs[40]['w']@b)<1e-12
    a=causal_template(cs[:51])[0]
    for c in cs[51:]:c['y']*=1e9;c['mu']*=1e9;c['g']*=1e9
    np.testing.assert_array_equal(a,causal_template(cs[:51])[0])

def test_causal_lead_and_weighted_time():
    cs=fixture();d1=causal_template(cs[:51],1)[0];d2=causal_template(cs[:51],2)[0]
    assert np.isfinite(d1).all() and np.isfinite(d2).all()
    t=np.array([.02,.08,.21]);w=weights(t,0,.25)
    assert abs(w@(3+2*t)-(3+2*(w@t)))<1e-14
