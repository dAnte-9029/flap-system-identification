import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from reference_groupcv_v8 import crossfit_phase,fit_normalization
from reference_generalization_v7 import held_start_schedule
from test_reference_generalization_v7 import synthetic_history
from test_reference_capacity_v6 import models
from system_identification.models.reference_gru_phase_schedule import ReferenceGRUPhaseSchedule


def test_schedule_training_matches_baseline_with_held_phase_and_has_gradients():
    old,_,x=models();torch.nn.init.normal_(old.head[-1].weight,std=.01)
    keys=['state_mean','state_scale','reference_mean','reference_scale','actuator_mean','actuator_scale','rate_scale','velocity_scale']
    model=ReferenceGRUPhaseSchedule({k:old.state_dict()[k] for k in keys});model.load_state_dict(old.state_dict())
    schedule=held_start_schedule(x);p=model(**x,phase_start=schedule);q=old(**x)
    for k in p:torch.testing.assert_close(p[k],q[k],rtol=0,atol=0)
    loss=sum(v.square().mean() for v in p.values());loss.backward()
    assert model.head[-1].weight.grad.abs().sum()>0
    assert all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters())
    changed={k:v.clone() for k,v in x.items()};changed['future_reference'][:,5:]+=10;schedule[:,5:]+=1
    future=model(**changed,phase_start=schedule)
    for k in p:torch.testing.assert_close(p[k][:,:5],future[k][:,:5],rtol=0,atol=0)


def test_crossfit_phase_cannot_read_own_or_outer_flight_labels():
    h=synthetic_history(16);logs=np.repeat(['a','b','c','d'],4);phi=2*np.pi*h[:,-1,11,None]*np.arange(1,51)*.02
    fit=np.arange(12);query=np.arange(12,16)
    a,b,prov,_=crossfit_phase(h,phi,logs,fit,query)
    changed=phi.copy();changed[12:]+=np.arange(50)*.01
    c,d,_,_=crossfit_phase(h,changed,logs,fit,query)
    np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d)
    changed=phi.copy();changed[:4]+=np.arange(50)*.01
    c,_,_,_=crossfit_phase(h,changed,logs,fit,query)
    np.testing.assert_array_equal(a[:4],c[:4])
    for p in prov:assert not set(logs[p['fit_indices']])&set(logs[p['query_indices']])


def test_normalization_excludes_unselected_flight_and_labels():
    rng=np.random.default_rng(3)
    def sample():return {k:rng.normal(size=(110,n)).astype(np.float32) for k,n in [('state_features',14),('references',5),('actuator_commands',4),('rates_b',3)]}
    arrays={'a':sample(),'b':sample()};w=pd.DataFrame(dict(log_id=['a','b'],partition=['train','train'],origin_index=[30,30],array_index=[0,1]));target=rng.normal(size=(2,50,3)).astype(np.float32)
    first=fit_normalization(arrays,w.iloc[:1],target)
    for value in arrays['b'].values():value[:]=1e6
    target[1]=1e6;second=fit_normalization(arrays,w.iloc[:1],target)
    for k in first:np.testing.assert_array_equal(first[k],second[k])
    assert np.all(first['state_mean'][9:11]==0) and np.all(first['state_scale'][9:11]==1)
