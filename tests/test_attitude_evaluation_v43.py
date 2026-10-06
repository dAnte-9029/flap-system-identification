import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_attitude_v43 import metrics,protections,response_gate
from verify_attitude_v43 import independent_squares


def test_tilt_ignores_yaw_but_full_attitude_keeps_it():
    q=np.zeros((2,50,4));q[...,0]=1
    true=dict(quaternion=q,rates_b=np.zeros((2,50,3)),velocity_n=np.zeros((2,50,3)))
    pred={k:v.copy() for k,v in true.items()};a=.4
    pred['quaternion'][0,:,0]=np.cos(a/2);pred['quaternion'][0,:,3]=np.sin(a/2)
    pred['quaternion'][1,:,0]=-np.cos(a/2);pred['quaternion'][1,:,1]=-np.sin(a/2)
    wave,end=metrics(pred,true,50);direct=independent_squares(pred,true)
    np.testing.assert_allclose(wave['tilt'],[0,a*a],atol=1e-14)
    np.testing.assert_allclose(wave['attitude'],a*a,atol=1e-14)
    for key in wave:np.testing.assert_allclose(wave[key],direct[key].mean(1),atol=1e-14)


def test_gate_requires_cross_seed_and_flight_consistency():
    rows=[]
    for seed in [17,23,42]:
        for fold in range(5):
            for arm,value in [('baseline',1.),('candidate',.8 if fold<3 else 1.01)]:
                rows.append(dict(seed=seed,fold=fold,arm=arm,rmse=value,metric='tilt',horizon=50,kind='wave'))
    f=pd.DataFrame(rows);g=response_gate(f,'tilt')
    assert g['improvement_percent']>5 and g['improving_seeds']==3
    assert g['improving_flights']==3 and not g['passed']
    f.loc[(f.arm=='candidate')&(f.fold==3),'rmse']=.8
    assert response_gate(f,'tilt')['passed']
    f.loc[(f.arm=='candidate')&(f.seed==42),'rmse']=1.01
    assert not response_gate(f,'tilt')['passed']


def test_protections_include_all_twenty_seven_frozen_cells():
    rows=[dict(seed=s,horizon=h,kind=k,metric=m) for s in [17,23,42] for h in [25,50]
          for k in ['wave','endpoint'] for m in ['p','q','r','roll','pitch','tilt','velocity','attitude']]
    p=protections(pd.DataFrame(rows))
    assert len(p)==27 and (p.metric=='q').sum()==3 and (p.metric=='roll').sum()==3
    assert not (p.metric=='tilt').any()
