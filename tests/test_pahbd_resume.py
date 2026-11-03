"""Interruption at update and validation boundaries must preserve the trajectory."""
import copy
import json
import pytest
import torch
from test_pahbd_readiness import model_window
from system_identification.training.pahbd import formal_fit,load_checkpoint,save_checkpoint


def equal_tree(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:equal_tree(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal_tree(x,y)
    else:assert a==b


@pytest.mark.parametrize('stop_step',[1,2,3])
@pytest.mark.parametrize('device',['cpu','cuda:0'] if torch.cuda.is_available() else ['cpu'])
def test_pause_resume_exact(tmp_path,stop_step,device):
    torch.manual_seed(17);base,w=model_window(.02)
    if device!='cpu':
        from dataclasses import fields,replace
        def move(obj):
            return replace(obj,**{f.name:getattr(obj,f.name).to(device) for f in fields(obj)
                                  if isinstance(getattr(obj,f.name),torch.Tensor)})
        base=base.to(device)
        w=replace(w,initial=move(w.initial),controls=move(w.controls),
                  history=[move(e) for e in w.history],targets=[move(e) for e in w.targets])
    v=copy.deepcopy(w);v.metadata['split']='validation'
    full=copy.deepcopy(base);part=copy.deepcopy(base)
    fo=torch.optim.Adam(full.parameters(),lr=1e-4);po=torch.optim.Adam(part.parameters(),lr=1e-4)
    meta={'config':{'purpose':'resume_test'}}
    kwargs=dict(steps=3,metadata=meta,validation_interval=2,recovery_interval=1,fixed_budget=True)
    formal_fit(full,[w],[v],fo,output=tmp_path/'full',**kwargs)
    def pause():
        log=tmp_path/'part/training_draws.jsonl'
        return log.exists() and len(log.read_text().splitlines())>=stop_step
    result=formal_fit(part,[w],[v],po,output=tmp_path/'part',pause_requested=pause,**kwargs)
    assert result['state']=='paused' and result['schedule']['step']==stop_step
    latest=json.loads((tmp_path/'part/latest.json').read_text())
    restored,ro,bundle=load_checkpoint(latest['checkpoint'],device=device,optimizer=True)
    result=formal_fit(restored,[w],[v],ro,output=tmp_path/'resume',resume=bundle,**kwargs)
    assert result['state']=='completed' and result['schedule']['step']==3
    equal_tree(full.state_dict(),restored.state_dict());equal_tree(fo.state_dict(),ro.state_dict())
    assert result['schedule']['best_score']==json.loads((tmp_path/'full/best.json').read_text())['score']
    logs=lambda folder:[json.loads(x)['window_id'] for x in (folder/'training_draws.jsonl').read_text().splitlines()] if (folder/'training_draws.jsonl').exists() else []
    assert logs(tmp_path/'full')==logs(tmp_path/'part')+logs(tmp_path/'resume')


def test_atomic_checkpoint_failed_write_and_no_overwrite(tmp_path,monkeypatch):
    m,w=model_window(.02);path=tmp_path/'save.pt'
    def fail(*args,**kwargs):raise OSError('disk failure')
    original=torch.save
    monkeypatch.setattr(torch,'save',fail)
    with pytest.raises(OSError):save_checkpoint(path,m,None,{},0)
    assert not path.exists() and not list(tmp_path.iterdir())
    monkeypatch.setattr(torch,'save',original);save_checkpoint(path,m,None,{},0)
    with pytest.raises(FileExistsError):save_checkpoint(path,m,None,{},0)
    load_checkpoint(path)


@pytest.mark.parametrize('exit_code,state',[(0,'completed'),(75,'paused'),(7,'failed')])
def test_queue_distinguishes_pause_failure_completion(tmp_path,exit_code,state):
    import os
    from pathlib import Path
    import shutil
    import subprocess
    import sys
    repo=Path(__file__).resolve().parents[1]
    checkout=tmp_path/'checkout';checkout.mkdir()
    shutil.copy2(repo/'scripts/queue_pahbd_runs.py',checkout/'queue_pahbd_runs.py')
    (checkout/'run_pahbd.py').write_text(f'import sys\nsys.exit({exit_code})\n')
    def git(*args):return subprocess.check_output(['git',*args],cwd=checkout,text=True).strip()
    git('init','-q');git('add','.');git('-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','fixture')
    root=tmp_path/'run';root.mkdir()
    (root/'queue.json').write_text(json.dumps(dict(state='prepared',code_revision=git('rev-parse','HEAD'),device='cpu',tasks=[dict(name='test',steps=1,state='pending')])))
    result=subprocess.run([sys.executable,str(checkout/'queue_pahbd_runs.py'),'--run-root',str(root)],cwd=checkout,
                          env=dict(os.environ,PYTHONPATH=str(repo/'src')),capture_output=True,text=True)
    q=json.loads((root/'queue.json').read_text())
    assert q['state']==state and q['tasks'][0]['exit_code']==exit_code
    assert (root/'FAILED.json').exists()==(state=='failed')
    assert (root/'COMPLETED.json').exists()==(state=='completed')
    assert (root/'PAUSED.json').exists()==(state=='paused')
    assert result.returncode==(1 if state=='failed' else 0)
