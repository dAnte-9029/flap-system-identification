"""Directed observational-gradient, clipping and reviewed-resume checks."""
import copy
import json
import numpy as np
import pandas as pd
import pytest
import torch
from system_identification.models.pacld import PACLD, Config
from system_identification.models.pahbd import Config as SyntheticConfig
from system_identification.data.pahbd_synthetic import controlled_periodic_window
from system_identification.training.pacld import (update,seed_all,diagnostics_due,save_checkpoint,load_checkpoint,FlightBalancedSampler)
from system_identification.training.pacld_compatibility import current_contract,check_source_contract

torch.set_num_threads(1)


def setup():
    seed_all(17);m=PACLD(Config(width=16)).double()
    w=controlled_periodic_window(SyntheticConfig(),.45)[0]
    cfg=json.load(open('configs/pacld/default.json'));cfg['model']['width']=16
    return m,w,cfg


def assert_same(a,b):
    if torch.is_tensor(a):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:assert_same(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_same(x,y)
    else:assert a==b


def test_diagnostic_is_observational_including_gradients_adam_and_parameters():
    m,w,cfg=setup();o=torch.optim.Adam(m.parameters(),**cfg['optimizer'])
    update(m,w,o,1.,diagnostics=False,max_grad_norm=None)  # nonempty identical Adam state
    other=copy.deepcopy(m);oo=torch.optim.Adam(other.parameters(),**cfg['optimizer']);oo.load_state_dict(copy.deepcopy(o.state_dict()))
    seed_all(17);a=update(m,w,o,1.,diagnostics=True,max_grad_norm=None)
    seed_all(17);b=update(other,w,oo,1.,diagnostics=False,max_grad_norm=None)
    assert a['total']==b['total'] and a['gradient_norms'] is not None and b['gradient_norms'] is None
    for p,q in zip(m.parameters(),other.parameters()):
        assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:assert torch.equal(p.grad,q.grad)
    assert_same(m.state_dict(),other.state_dict());assert_same(o.state_dict(),oo.state_dict())


@pytest.mark.parametrize('threshold',[100000.,.001])
def test_clipping_below_and_above_threshold(threshold):
    m,w,cfg=setup();other=copy.deepcopy(m)
    o=torch.optim.Adam(m.parameters(),**cfg['optimizer']);oo=torch.optim.Adam(other.parameters(),**cfg['optimizer'])
    a=update(m,w,o,1.,diagnostics=False,max_grad_norm=threshold)
    b=update(other,w,oo,1.,diagnostics=False,max_grad_norm=None)
    norm=torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in m.parameters() if p.grad is not None]))
    assert torch.isfinite(norm) and norm<=threshold*(1+1e-12)
    assert a['pre_clip_grad_norm']==b['pre_clip_grad_norm']
    if threshold==100000.:
        assert not a['gradient_clipped'];assert_same(m.state_dict(),other.state_dict());assert_same(o.state_dict(),oo.state_dict())
    else:
        assert a['gradient_clipped'] and 0<a['clip_coefficient']<1
        assert any(not torch.equal(p.grad,q.grad) for p,q in zip(m.parameters(),other.parameters()) if p.grad is not None)


def test_schedule_global_steps_and_null_records():
    assert [i+1 for i in range(205) if diagnostics_due(i,100)]==[1,101,201]
    assert diagnostics_due(100,100) and not diagnostics_due(99,100)
    with pytest.raises(ValueError):diagnostics_due(0,0)


def test_clipped_resume_next_draw_update_and_operational_compatibility(tmp_path):
    m,w,cfg=setup();o=torch.optim.Adam(m.parameters(),**cfg['optimizer'])
    frame=pd.DataFrame([dict(split='train',window_id=f'w{i}',flight_id=f'f{i}',horizon_us=500000) for i in range(4)])
    sampler=FlightBalancedSampler(frame,17);first=sampler.draw()
    update(m,w,o,1.,diagnostics=True,max_grad_norm=10.)
    save_checkpoint(tmp_path/'model.pt',m,o,cfg,{'train':'synthetic'},sampler,{'step':1,'best_score':1.},1.)
    changed=copy.deepcopy(cfg);changed['data_cache']['max_flights']=2
    changed['training']['gradient_diagnostics_interval']=200;changed['training']['terminal_log_interval']=20
    restored,oo,b=load_checkpoint(tmp_path/'model.pt',changed,{'train':'synthetic'},sampler_index=frame)
    resumed=FlightBalancedSampler(frame,17);resumed.load_state_dict(b['sampler'])
    second=sampler.draw();assert resumed.draw()==second and second!=first
    a=update(m,w,o,1.,diagnostics=False,max_grad_norm=10.)
    c=update(restored,w,oo,1.,diagnostics=False,max_grad_norm=10.)
    assert a['total']==c['total'];assert_same(m.state_dict(),restored.state_dict());assert_same(o.state_dict(),oo.state_dict())
    bad=copy.deepcopy(cfg);bad['training']['max_grad_norm']=None
    with pytest.raises(ValueError,match='clipping'):load_checkpoint(tmp_path/'model.pt',bad,{'train':'synthetic'})
    with pytest.raises(ValueError,match='identity'):load_checkpoint(tmp_path/'model.pt',cfg,{'train':'different'})
    bad_frame=frame.copy();bad_frame.loc[0,'window_id']='changed'
    with pytest.raises(ValueError):load_checkpoint(tmp_path/'model.pt',cfg,{'train':'synthetic'},sampler_index=bad_frame)
    bundle=torch.load(tmp_path/'model.pt',weights_only=True);bundle['git_revision']='different-log-only-commit'
    torch.save(bundle,tmp_path/'git.pt')
    load_checkpoint(tmp_path/'git.pt',cfg,{'train':'synthetic'})  # Git itself is not a semantic version
    bundle['source_contract']['sources']['loss']='unknown'
    torch.save(bundle,tmp_path/'source.pt')
    with pytest.raises(ValueError,match='source'):load_checkpoint(tmp_path/'source.pt',cfg,{'train':'synthetic'})


def test_unknown_current_source_and_old_schema_rejected(monkeypatch,tmp_path):
    from system_identification.training import pacld_compatibility as compat
    contract=current_contract();changed=copy.deepcopy(contract);changed['versions']['model']='other'
    with pytest.raises(ValueError,match='version'):check_source_contract(changed)
    digests=compat.source_digests();digests['model']='unreviewed'
    monkeypatch.setattr(compat,'source_digests',lambda:digests)
    with pytest.raises(ValueError,match='unreviewed'):current_contract()
    torch.save({'schema':'pacld_training_v1'},tmp_path/'old.pt')
    _,_,cfg=setup()
    with pytest.raises(ValueError,match='legacy'):load_checkpoint(tmp_path/'old.pt',cfg,{})


def test_nonfinite_state_stops_before_optimizer():
    m,w,cfg=setup();o=torch.optim.Adam(m.parameters(),**cfg['optimizer']);before=copy.deepcopy(m.state_dict())
    original=m.encode
    def poisoned(window):
        b=original(window);return b.with_(c=b.c*float('nan'))
    m.encode=poisoned
    with pytest.raises(FloatingPointError):update(m,w,o,1.,diagnostics=False)
    assert_same(before,m.state_dict());assert not o.state


def test_lru_bounds_eviction_and_split_keys(monkeypatch):
    from system_identification.data.pacld import CachedNativeWindows
    from system_identification.data.pahbd import NativeWindows
    loads=[]
    def init(self,split='train',**kwargs):
        self.split=split;self._cached_id=None
    def load(self,fid):
        loads.append((self.split,fid))
        self.tables={'values':pd.DataFrame({'v':[self.split+fid]})}
        self.samples=pd.DataFrame({'sample':[1]});self._cached_id=fid
    def encoder(self,fid):return pd.DataFrame({'count':[len(fid)]})
    monkeypatch.setattr(NativeWindows,'__init__',init)
    monkeypatch.setattr(NativeWindows,'_load',load)
    monkeypatch.setattr(NativeWindows,'_encoder_counts',encoder)
    reader=CachedNativeWindows(split='train',cache_flights=2,cache_bytes=10000)
    reader._load('a');original=reader.tables;reader._load('b');reader._load('a')
    assert reader.tables is original and loads==[('train','a'),('train','b')]
    reader._load('c');reader._load('b')
    assert loads[-2:]==[('train','c'),('train','b')] and reader.cache_stats()['evictions']==2
    reader.split='validation';reader._load('b')
    assert loads[-1]==('validation','b') and reader.tables['values'].iloc[0,0]=='validationb'
    assert ('validation','b') in reader._flights and len(reader._flights)<=2
    assert reader.cache_stats()['retained_bytes']<=10000
    tiny=CachedNativeWindows(split='train',cache_flights=2,cache_bytes=1)
    tiny._load('a');tiny._load('a')
    assert tiny.cache_stats()['entries']==0 and tiny.cache_stats()['misses']==2


def test_embedded_checkpoint_model_and_optimizer_consistency(tmp_path):
    m,w,cfg=setup();o=torch.optim.Adam(m.parameters(),**cfg['optimizer'])
    frame=pd.DataFrame([dict(window_id='w',flight_id='f',horizon_us=500000,split='train')])
    sampler=FlightBalancedSampler(frame,17)
    update(m,w,o,1.,diagnostics=False)
    save_checkpoint(tmp_path/'ok.pt',m,o,cfg,{},sampler,{'step':1},1.)
    bundle=torch.load(tmp_path/'ok.pt',weights_only=True)
    for name in ('model','dtype','Adam'):
        broken=copy.deepcopy(bundle)
        if name=='model':broken['model_config']['width']=32
        elif name=='dtype':broken['dtype']='float32'
        else:broken['optimizer']['param_groups'][0]['lr']=.1
        torch.save(broken,tmp_path/'bad.pt')
        with pytest.raises(ValueError,match='embedded'):load_checkpoint(tmp_path/'bad.pt',cfg,{})
