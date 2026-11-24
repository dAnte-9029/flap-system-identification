"""Fixed training and native-time flight-equal evaluation for the command-reference contrast."""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
RELATIVE = Path('experiments/autopilot_command_reference')
torch.set_num_threads(2)


def load(part):
    meta=json.loads((HERE/'data.json').read_text())
    assert meta['dataset']==os.environ['RA_DATA_VERSION']
    rec=meta['split'][part]
    path=Path(rec['file'])
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==rec['sha256']
    with np.load(path,allow_pickle=False) as f:
        batch={k:torch.from_numpy(f[k]) for k in ('hist','pooled','u0','time','y','mask')}
        flights=f['flight']
    assert len(flights)==rec['windows'] and len(np.unique(flights))==rec['flights']
    return batch,flights


def predict(model,b,ix):
    return model(*(b[k][ix] for k in ('hist','pooled','u0','time')))


def train():
    from model import FlightPredictor
    seed=int(os.environ['RA_SEED'])
    torch.manual_seed(seed)
    steps=int(os.environ['RA_STEPS'])
    b,_=load('train')
    model=FlightPredictor(b['hist'].mean(0),b['hist'].std(0).clamp_min(.05))
    y,mask=b['y'].reshape(-1,3,2,3),b['mask'].reshape(-1,3,2,1)
    scale=((y.square()*mask).sum((0,1))/mask.sum((0,1))).sqrt().clamp_min(.05)
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
    generator=torch.Generator().manual_seed(seed)
    losses=[]
    for step in range(steps):
        ix=torch.randint(len(y),(384,),generator=generator)
        pred=predict(model,b,ix).reshape(-1,3,2,3)
        sq=((pred-y[ix])/scale).square()*mask[ix]
        loss=sq.sum()/(3*mask[ix].sum())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        opt.step()
        losses.append(float(loss.detach()))
    torch.save(model.state_dict(),os.environ['RA_MODEL'])
    info=dict(steps=steps,samples=steps*384,train_loss=np.mean(losses[-50:]),first50=np.mean(losses[:50]),
              parameters=sum(p.numel() for p in model.parameters()))
    Path(os.environ['RA_TRAIN_INFO']).write_text(json.dumps(info))
    print(info)


def evaluate():
    b,flights=load('validation')
    src=Path(os.environ['RA_MODEL_CODE'])/RELATIVE/'model.py'
    spec=importlib.util.spec_from_file_location('fixed_candidate',src)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    params=torch.load(os.environ['RA_MODEL'],weights_only=True,map_location='cpu')
    model=mod.FlightPredictor(params['mean'],params['std'])
    model.load_state_dict(params)
    model.eval()
    with torch.inference_mode():
        pred=torch.cat([predict(model,b,slice(i,i+768)) for i in range(0,len(flights),768)])
    sq=(pred-b['y']).square().sum(-1).numpy()
    valid=b['mask'].numpy().astype(bool)
    names=('v05','w05','v1','w1','v2','w2')
    metrics={}
    perflight={}
    for i,name in enumerate(names):
        group={f:float(sq[(flights==f)&valid[:,i],i].mean()) for f in np.unique(flights[valid[:,i]])}
        assert len(group)==39
        perflight[name]=group
        metrics[name]=float(np.sqrt(np.mean(list(group.values()))))
    Path(os.environ['RA_METRICS']).write_text(json.dumps(metrics))
    (Path(os.environ['RA_OUTPUT'])/'flight_mse.json').write_text(json.dumps(perflight))
    print(metrics)


def preflight():
    from prepare import pool_commands
    assert os.environ['RA_SPLIT']=='train'
    b,_=load('train')
    # A future pulse must not affect a query inside the preceding native segment.
    tape=np.array([[[[.2,.1,.3,.4,13000],[.7,-.4,.1,.3,7000]]]],dtype=np.float64)
    time=np.array([[.009,.018]])
    original=pool_commands(tape,time)
    tape[0,0,1,:4]=-.5
    changed=pool_commands(tape,time)
    assert np.array_equal(original[:,0],changed[:,0])
    assert np.max(abs(original[:,1]-changed[:,1]))>1e-4
    assert b['mask'].sum()>0
    print('native command prefix and fixed training archive verified')


if __name__=='__main__':
    phase=sys.argv[1]
    assert os.environ['RA_SPLIT']==('validation' if phase=='evaluate' else 'train')
    {'preflight':preflight,'train':train,'evaluate':evaluate}[phase]()
