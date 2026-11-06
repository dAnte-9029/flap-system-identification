"""Focused training/loss/resume contracts; synthetic CPU data only."""
import copy
from dataclasses import replace
import json
import random
import numpy as np
import pandas as pd
import pytest
import torch
from system_identification.models.pacld import PACLD, Config, ControlTape
from system_identification.models.pahbd import Config as SyntheticConfig
from system_identification.models.pahbd import geometry as so3
from system_identification.data.pahbd_synthetic import controlled_periodic_window
from system_identification.data.pacld import PACLDWindows, open_windows
from system_identification.training.pacld import (window_objective,valid_pairs,gradient_norms,state_loss,
    transition_residual,update,save_checkpoint,load_checkpoint,seed_all,FlightBalancedSampler)
from system_identification.training.pahbd import trajectory_loss
from system_identification.evaluation.pacld import hierarchical, evaluate

torch.set_num_threads(1)


def fixture():
    seed_all(17)
    m=PACLD(Config(width=16)).double()
    w=controlled_periodic_window(SyntheticConfig(),.45)[0]
    w.metadata.update(horizon_us=500000,sampling_group='main_50hz')
    return m,w


def test_full_macro_separate_gradients_reach_shared_dynamics():
    m,w=fixture();r=window_objective(m,w,.2)
    assert r['macro'] is not None and torch.isfinite(r['total'])
    norms=gradient_norms(m,r)
    for name in ('full','macro'):
        for module in ('belief.shared','belief.slow_head','belief.fast_head',
                       'flow.dynamics.decoder','flow.dynamics.slow','flow.dynamics.equilibrium',
                       'flow.dynamics.relaxation','flow.dynamics.coupling'):
            assert np.isfinite(norms[name]['modules'][module]) and norms[name]['modules'][module]>0
    # Exact pure state term agrees with old state component, not its total objective.
    _,old=trajectory_loss(r['predictions']['FULL'],w.targets,r['times'])
    assert torch.equal(r['full'],old['state'])
    changed=[s.with_(variance=s.variance*1e12) for s in r['predictions']['FULL']]
    assert torch.equal(r['full'],state_loss(changed,w.targets,r['times']))


def test_no_macro_is_missing_not_zero_full_still_updates():
    m,w=fixture();w.metadata['cycle_pairs']=[]
    r=window_objective(m,w,.3)
    assert r['macro'] is None and r['total'] is r['full']
    opt=torch.optim.Adam(m.parameters(),lr=1e-4)
    before=m.flow.dynamics.decoder[-1].weight.detach().clone()
    record=update(m,w,opt,.3)
    assert record['macro'] is None and record['gradient_norms']['macro'] is None
    assert not torch.equal(before,m.flow.dynamics.decoder[-1].weight)


@pytest.mark.parametrize('bad',['source','mask','time','availability','phase'])
def test_bad_pairs_are_excluded(bad):
    m,w=fixture();i,j=w.metadata['cycle_pairs'][0];w.metadata['cycle_pairs']=[(i,j)]
    if bad=='source': w.targets[j]=replace(w.targets[j],source='invalid')
    if bad=='mask': w.targets[j]=replace(w.targets[j],mask=w.targets[j].mask*0)
    if bad=='time': w.targets[j]=replace(w.targets[j],measurement=w.targets[i].measurement)
    if bad=='availability': w.targets[j]=replace(w.targets[j],availability=-1.)
    if bad=='phase': w.metadata['target_phase_reference']=[0.]*len(w.targets)
    assert valid_pairs(w)==([],1)


def test_relative_so3_transition_geometry():
    _,w=fixture();a=w.initial.x.clone();b=a.clone()
    a[6:10]=so3.exp(a.new_tensor([.2,-.3,.1]));b[6:10]=so3.exp(a.new_tensor([-.2,.4,.1]))
    pa=a.clone();pb=b.clone();rotation=so3.exp(a.new_tensor([.4,.2,-.1]))
    pa[6:10]=so3.multiply(rotation,a[6:10]);pb[6:10]=so3.multiply(rotation,b[6:10])
    assert transition_residual(pa,pb,a,b)[6:9].abs().max()<1e-12
    pb[6:10]*=-1
    assert transition_residual(pa,pb,a,b)[6:9].abs().max()<1e-12


def test_one_encode_same_belief_future_targets_and_zoh_representation():
    m,w=fixture();calls=[];ids=[]
    handle=m.belief.register_forward_hook(lambda *args:calls.append(1))
    original=m.flow.rollout
    def rollout(s,*args,**kwargs):
        ids.append(id(s));return original(s,*args,**kwargs)
    m.flow.rollout=rollout
    a=window_objective(m,w,.2)
    assert len(calls)==1 and len(ids)==2 and ids[0]==ids[1]
    handle.remove()
    poison=copy.deepcopy(w)
    poison.targets=[replace(e,value=e.value+100) for e in poison.targets]
    b=window_objective(m,poison,.2)
    assert torch.equal(a['belief'].x,b['belief'].x)
    for mode in ('FULL','BASE'):
        assert torch.equal(a['predictions'][mode][-1].x,b['predictions'][mode][-1].x)
    times=[];values=[]
    for i,t in enumerate(w.controls.times):
        times.append(t);values.append(w.controls.values[i])
        end=w.controls.times[i+1] if i+1<len(w.controls.times) else .45
        times.append((t+end)/2);values.append(w.controls.values[i])
    split=replace(w,controls=ControlTape(times,torch.stack(values)))
    c=window_objective(m,split,.2)
    assert torch.equal(a['total'],c['total'])


def test_checkpoint_optimizer_sampler_rng_and_next_update(tmp_path):
    m,w=fixture();config=json.load(open('configs/pacld/default.json'))
    config['model']['width']=16
    opt=torch.optim.Adam(m.parameters(),**config['optimizer']);update(m,w,opt,.2)
    index=pd.DataFrame([dict(split='train',flight_id='s',window_id='w',horizon_us=500000)])
    sampler=FlightBalancedSampler(index,17);sampler.draw()
    identity={'train':'synthetic','validation':'synthetic_fixed'}
    save_checkpoint(tmp_path/'model.pt',m,opt,config,identity,sampler,dict(step=1,best_score=1.),.2)
    expected=(torch.rand(2),random.random(),np.random.rand())
    restored,other,b=load_checkpoint(tmp_path/'model.pt',config,identity)
    actual=(torch.rand(2),random.random(),np.random.rand())
    assert torch.equal(actual[0],expected[0]) and actual[1:]==expected[1:]
    other_sampler=FlightBalancedSampler(index,17);other_sampler.load_state_dict(b['sampler'])
    assert other_sampler.draw()==sampler.draw()
    assert b['step']==1 and b['lambda_macro']==.2
    update(m,w,opt,.2);update(restored,w,other,.2)
    for key,value in m.state_dict().items(): assert torch.equal(value,restored.state_dict()[key]),key
    for left,right in zip(opt.state.values(),other.state.values()):
        for key in left: assert torch.equal(left[key],right[key])
    with pytest.raises(ValueError):load_checkpoint(tmp_path/'model.pt',config,{'wrong':'identity'})
    bad=copy.deepcopy(config);bad['optimizer']['lr']=.01
    with pytest.raises(ValueError):load_checkpoint(tmp_path/'model.pt',bad,identity)
    torch.save(dict(schema='pahbd_v3_hall'),tmp_path/'wrong.pt')
    with pytest.raises(ValueError):load_checkpoint(tmp_path/'wrong.pt',config,identity)


def test_read_only_wrapper_no_latent_and_sealed_rejection():
    m,w=fixture()
    class Dataset:
        split='train'
        windows=pd.DataFrame([dict(split='train',shared_periodic_eligible=True)])
        def __getitem__(self,index):return w
    w.metadata.update(shared_periodic_eligible=True,periodic_enabled=True)
    old=w.initial.h.clone();view=PACLDWindows(Dataset())
    assert view[0] is w and torch.equal(w.initial.h,old) and not hasattr(m.config,'latent')
    for split in ('test','sealed_test','TEST'):
        with pytest.raises(ValueError):open_windows({},split)
    Dataset.split='test'
    with pytest.raises(ValueError):PACLDWindows(Dataset())


def test_validation_full_only_selection_and_missing_macro():
    m,w=fixture();w.metadata['split']='validation';w.metadata['cycle_pairs']=[]
    config=json.load(open('configs/pacld/default.json'))
    result=evaluate(m,[w],config,diagnostics=True)
    assert result['selection_score']==result['windows'][0]['full']
    assert result['loss_reduction']['macro']['score'] is None
    assert result['coverage']['paired_windows']==0
    rows=[dict(flight_id='a',horizon_us=1,full=1.),dict(flight_id='a',horizon_us=1,full=3.),
          dict(flight_id='a',horizon_us=2,full=8.),dict(flight_id='b',horizon_us=1,full=1.)]
    assert hierarchical(rows,'full')['score']==3.
    with pytest.raises(ValueError):update(m,w,torch.optim.Adam(m.parameters()),.2)


def test_formal_schedule_checkpoint_and_resume_on_synthetic_windows(tmp_path):
    from system_identification.training.pacld import fit
    m,w=fixture();config=json.load(open('configs/pacld/default.json'));config['model']['width']=16
    class Windows:
        def __init__(self,split):
            self.w=copy.deepcopy(w);self.w.metadata['split']=split
            self.index=pd.DataFrame([dict(split=split,flight_id=w.metadata['flight_id'],window_id='synthetic',horizon_us=500000)])
        def __len__(self):return 1
        def __getitem__(self,i):
            if i!=0:raise IndexError(i)
            return self.w
    train,val=Windows('train'),Windows('validation');identity={'scope':'synthetic_schedule_test'}
    opt=torch.optim.Adam(m.parameters(),**config['optimizer'])
    first=fit(m,train,val,opt,config,identity,.2,1,tmp_path/'first')
    restored,other,b=load_checkpoint(tmp_path/'first/step_00000001.pt',config,identity)
    second=fit(restored,train,val,other,config,identity,.2,2,tmp_path/'second',b)
    assert first['step']==1 and second['step']==2
    assert (tmp_path/'second/step_00000002.pt').exists()
    assert json.loads((tmp_path/'second/training.jsonl').read_text())['step']==2
    assert json.loads((tmp_path/'second/validation_00000002.json').read_text())['selection_score'] is not None


def test_accelerated_census_matches_original_pair_search():
    import math
    import runpy
    count=runpy.run_path('scripts/audit_pacld_pairs.py')['count_pairs']
    rng=np.random.default_rng(17)
    for size in (2,13,87):
        phase=np.cumsum(rng.uniform(.02,.8,size));mask=rng.random((size,3))>.2
        valid=matched=0
        for i in range(size-1):
            j,error=min([(j,abs(phase[j]-phase[i]-2*math.pi)) for j in range(i+1,size)],key=lambda x:x[1])
            tolerance=min(abs(phase[k]-phase[j]) for k in range(size) if k!=j)/2
            if error<=tolerance:
                matched+=1;valid+=bool((mask[i]&mask[j]).any())
        assert count(phase,mask)==(valid,matched,size-1)
