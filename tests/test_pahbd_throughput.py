import copy
import pytest
import pandas as pd
import torch
from test_pahbd_readiness import model_window
from system_identification.data.pahbd import ConfiguredWindows
from system_identification.data.pahbd_transfer import window_to_device
from system_identification.training.pahbd import FlightBalancedSampler,train_steps
from system_identification.training.pahbd_prefetch import WindowStream


class Dataset:
    device='cpu'
    def __init__(self,w):
        self.w=w
        self.windows=pd.DataFrame([dict(w.metadata,window_id=f'w{i}',flight_id=f'f{i}') for i in range(3)])
    def __getitem__(self,i):
        w=copy.deepcopy(self.w);w.metadata=dict(self.windows.iloc[i]);return w


def test_prefetch_consumption_and_exact_updates():
    m,w=model_window(.02);data=Dataset(w);windows=ConfiguredWindows(data,range(3),m.config)
    sampler=FlightBalancedSampler(windows.index)
    stream=WindowStream(windows,sampler,5,True)
    assert sampler.draws==0
    index,got,_=stream.next();assert sampler.draws==1
    saved=sampler.state_dict();stream.close()
    restored=FlightBalancedSampler(windows.index);restored.load_state_dict(saved)
    assert [restored.draw() for _ in range(5)]==[sampler.draw() for _ in range(5)]
    other=copy.deepcopy(m);o=torch.optim.Adam(m.parameters());oo=torch.optim.Adam(other.parameters())
    a=train_steps(m,windows,o,3)
    b=train_steps(other,windows,oo,3,prefetch=True)
    assert [r['window_id'] for r in a]==[r['window_id'] for r in b]
    assert all(torch.equal(x,other.state_dict()[k]) for k,x in m.state_dict().items())
    for key,state in o.state_dict()['state'].items():
        for name,val in state.items():assert torch.equal(val,oo.state_dict()['state'][key][name])


def test_packed_transfer_preserves_events_and_control():
    _,w=model_window(.03);moved=window_to_device(w,'cpu')
    assert moved.controls.times==w.controls.times
    assert torch.equal(moved.controls.values,w.controls.values)
    for a,b in zip(moved.history+moved.targets,w.history+w.targets):
        assert (a.measurement,a.availability,a.source)==(b.measurement,b.availability,b.source)
        for key in ('value','mask','variance','quality'):assert torch.equal(getattr(a,key),getattr(b,key))
    assert torch.equal(moved.initial.x,w.initial.x)


@pytest.mark.parametrize('action',['file','ctrl_c','failure'])
def test_two_workers_overlap_and_global_pause(tmp_path,action):
    import json,os,shutil,subprocess,sys,time
    from pathlib import Path
    repo=Path(__file__).resolve().parents[1];checkout=tmp_path/'checkout';checkout.mkdir()
    shutil.copy2(repo/'scripts/queue_pahbd_runs.py',checkout/'queue_pahbd_runs.py')
    (checkout/'run_pahbd.py').write_text('''import sys,os,json,time
from pathlib import Path
out=Path(sys.argv[sys.argv.index('--output')+1]);out.mkdir()
(out/'seen.json').write_text(json.dumps(dict(gpu=os.environ['CUDA_VISIBLE_DEVICES'],affinity=sorted(os.sched_getaffinity(0)),start=time.time())))
if out.name=='t1' and FAIL:
    time.sleep(.3);sys.exit(7)
while not (out/'PAUSE_REQUEST').exists():time.sleep(.01)
sys.exit(75)
'''.replace('FAIL',str(action=='failure')))
    def git(*args):return subprocess.check_output(['git',*args],cwd=checkout,text=True).strip()
    git('init','-q');git('add','.');git('-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','fixture')
    root=tmp_path/'run';root.mkdir()
    q=dict(state='prepared',code_revision=git('rev-parse','HEAD'),device='cuda:0',workers=[dict(id='a',physical_gpu=0),dict(id='b',physical_gpu=1)],tasks=[dict(name=f't{i}',steps=2,state='pending') for i in range(4)])
    (root/'queue.json').write_text(json.dumps(q))
    proc=subprocess.Popen([sys.executable,str(checkout/'queue_pahbd_runs.py'),'--run-root',str(root)],cwd=checkout,env=dict(os.environ,PYTHONPATH=str(repo/'src')))
    try:
        deadline=time.monotonic()+20
        while not all((root/f't{i}/seen.json').exists() for i in (0,1)):
            assert time.monotonic()<deadline;time.sleep(.05)
        a=json.loads((root/'t0/seen.json').read_text());b=json.loads((root/'t1/seen.json').read_text())
        assert a['gpu']=='0' and b['gpu']=='1' and abs(a['start']-b['start'])<2
        if action=='file':(root/'PAUSE_REQUEST').touch()
        elif action=='ctrl_c':
            import signal
            proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=15)==(1 if action=='failure' else 0)
        q=json.loads((root/'queue.json').read_text())
        assert q['state']==('failed' if action=='failure' else 'paused')
        assert [t['state'] for t in q['tasks']]==['paused','failed' if action=='failure' else 'paused','pending','pending']
    finally:
        if proc.poll() is None:
            (root/'PAUSE_REQUEST').touch();proc.wait(timeout=15)


def test_phase_factor_cache_gradients_and_invalidation():
    import math
    from system_identification.models.pahbd.dynamics import PhasePropagator
    phase=PhasePropagator((3.,20.,-5.))
    for step in range(3):
        if step==2:phase.coefficients.add_(.1)
        f=torch.tensor([5.],dtype=torch.float64,requires_grad=True)
        phi=torch.tensor([.4],dtype=torch.float64,requires_grad=True)
        motor=torch.tensor([.7],dtype=torch.float64,requires_grad=True)
        dt=.003;b0,bu,bf=phase.coefficients.unbind();steady=-(b0+bu*motor)/bf
        expected=(steady+(f-steady)*torch.exp(bf*dt),phi+2*math.pi*(steady*dt+(f-steady)*torch.expm1(bf*dt)/bf))
        actual=phase(f,phi,motor,dt)
        for a,b in zip(actual,expected):assert torch.equal(a,b)
        ga=torch.autograd.grad(sum(actual).sum(),(f,phi,motor),retain_graph=True)
        gb=torch.autograd.grad(sum(expected).sum(),(f,phi,motor))
        for a,b in zip(ga,gb):torch.testing.assert_close(a,b,atol=1e-14,rtol=1e-14)


def test_gauge_cache_invalidation_and_gradient():
    from system_identification.models.pahbd.dynamics import Config,ControlMixer
    mixer=ControlMixer(Config()).double()
    for offset in (0.,.3):
        mixer.gauge_offset.fill_(offset)
        value=torch.randn(3,2,6,dtype=torch.float64,requires_grad=True)
        a=mixer._rotate(value);b=mixer.rotate_coefficients(value,mixer.gauge_offset)
        assert torch.equal(a,b)
        assert torch.equal(torch.autograd.grad(a.sum(),value,retain_graph=True)[0],torch.autograd.grad(b.sum(),value)[0])


def test_prefetch_pause_resume_keeps_committed_sampler(tmp_path):
    import json
    from system_identification.training.pahbd import formal_fit,load_checkpoint
    m,w=model_window(.02);other=copy.deepcopy(m)
    windows=ConfiguredWindows(Dataset(w),range(3),m.config)
    v=copy.deepcopy(w);v.metadata['split']='validation'
    meta={'config':{'purpose':'prefetch_pause'}}
    kwargs=dict(steps=3,validation_interval=2,recovery_interval=1,fixed_budget=True,metadata=meta)
    o=torch.optim.Adam(m.parameters());oo=torch.optim.Adam(other.parameters())
    full=formal_fit(m,windows,[v],o,output=tmp_path/'full',**kwargs)
    def pause():return (tmp_path/'part/training_draws.jsonl').exists()
    formal_fit(other,windows,[v],oo,output=tmp_path/'part',pause_requested=pause,prefetch=True,**kwargs)
    latest=json.loads((tmp_path/'part/latest.json').read_text())
    restored,ro,bundle=load_checkpoint(latest['checkpoint'],optimizer=True)
    assert bundle['step']==bundle['training_state']['sampler']['draws']==1
    result=formal_fit(restored,windows,[v],ro,output=tmp_path/'resume',resume=bundle,prefetch=True,**kwargs)
    assert full['coverage']==result['coverage']
    assert all(torch.equal(x,restored.state_dict()[k]) for k,x in m.state_dict().items())
