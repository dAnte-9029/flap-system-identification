import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pandas as pd
from report_model_flight_response_diagnostic import alignment,macro

def test_alignment_zero_and_known_lag():
    rng=np.random.default_rng(17);t=rng.normal(size=(2,50,2));dt=np.full((2,50),.02)
    _,_,lag,seconds,valid=alignment(t,t,dt)
    assert valid.all() and (lag==0).all()
    p=np.concatenate([np.zeros((2,2,2)),t[:,:-2]],axis=1)
    _,_,lag,seconds,valid=alignment(p,t,dt)
    assert (lag==2).all();np.testing.assert_allclose(seconds,.04)

def test_flat_alignment_undefined():
    _,best,_,seconds,valid=alignment(np.zeros((1,50,2)),np.zeros((1,50,2)),np.ones((1,50)))
    assert not valid.any() and np.isnan(best).all() and np.isnan(seconds).all()

def test_macro_equal_flight_sd():
    f=pd.DataFrame([dict(group='ALL',cohort='Sep7',seed=s,flight_id=flight,value=v+s) for s in (0,1,2) for flight,v in [('a',1),('b',3)]])
    per,multi=macro(f,['group'],['value']);g=multi[multi.cohort=='ALL'].iloc[0]
    assert g.value_mean==3 and g.value_std==1
