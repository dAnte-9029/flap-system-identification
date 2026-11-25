"""Fit all three output geometries against the identical frozen network and native targets."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from model import FlightPredictor
from native_path import query

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
BASE=ROOT/'outputs/autopilot_flight_dynamics_20261124'
AUG=ROOT/'outputs/autopilot_periodic_motion_20261125'
NAMES=("v05","w05","v1","w1","v2","w2")
CASES=("matched","held","corrected")
RIDGE=1e-5


def get_base(key):
    if key=="incumbent17":
        return ROOT/'research/rounds/history-identification-20261125/extended/runs/low/final/model.pt'
    if key.startswith('terminal'):
        return HERE/'backbones'/('incumbent'+key.removeprefix('terminal'))/'final/model.pt'
    if key.startswith('incumbent'):
        return HERE/'backbones'/key/'step49152/model.pt'
    sid=json.loads((HERE.parent/'studies.json').read_text())[key]
    s=ROOT/'research/experiments'/sid
    b=json.loads((s/'state.json').read_text())['baseline']
    return s/b['attempts'][-1]['directory']/f'seed-{key}'/'output/step49152.pt'


def load(part,dev):
    with np.load(BASE/f'{part}.npz') as z:
        b={k:torch.as_tensor(z[k],device=dev) for k in ('time','pooled','u0','y','hist','mask')}
        ids=z['flight']
    x=np.load(AUG/f'{part}_features.npy')
    b['hist']=torch.cat((b['hist'],torch.as_tensor(x,device=dev)),dim=1)
    if part=='train':
        idx=np.random.default_rng(17).integers(0,6,len(x))
        for k in ('time','pooled','y','mask'):
            a=np.load(AUG/f'train_{k}.npy',mmap_mode='r')
            b[k+'_native']=torch.as_tensor(a[np.arange(len(x)),idx],device=dev)
    return b,ids


def evaluate(model,b,flights,arm):
    if arm in CASES:
        model.clock_choice.zero_();model.clock_choice[CASES.index(arm)]=1
    metrics=[]; max_equal=0.
    for start in range(0,len(flights),768):
        sl=slice(start,start+768); h=b['hist'][sl]; args=[b[k][sl] for k in ('pooled','u0','time')]
        pred=model.core(h[:,:model.mean.numel()],*args) if arm=='strong' else model(h,*args)
        sq=(pred-b['y'][sl]).square().sum(-1)[:,[1,3,7]].reshape(-1,6)
        metrics.append(sq.cpu().numpy())
    a=np.concatenate(metrics)
    valid=b['mask'][:,[1,3,7]].cpu().numpy().reshape(-1,6)
    return {name:{f:float(a[(flights==f)&valid[:,j],j].mean()) for f in np.unique(flights)} for j,name in enumerate(NAMES)}


@torch.inference_mode()
def fit(key,device,schedule):
    out=HERE/{'jittered':'runs','canonical':'canonical_runs','original':'original_runs'}[schedule]/key
    out.mkdir(parents=True,exist_ok=True)
    anchor=get_base(key)
    state=torch.load(anchor,map_location='cpu',weights_only=True)
    model=FlightPredictor(state['mean'],state['std']).to(device)
    model.core.load_state_dict(state);model.eval()
    coeff=json.loads((HERE.parent/'clock/clock_fit.json').read_text())['coefficients']
    model.clock_coeff.copy_(torch.as_tensor(coeff,device=device))
    b,ids=load('train',device)
    h=b['hist']
    raw=torch.cat([model.raw_context(c) for c in h.split(1024)])
    model.center.copy_(raw.mean(0));model.scale.copy_(raw.std(0).clamp_min(.025))
    x=torch.cat((torch.ones_like(raw[:,:1]),(raw-model.center)/model.scale),1)
    n=x.shape[1]*3
    normal=torch.zeros((3,2,n,n),device=device,dtype=torch.float64)
    rhs=torch.zeros((3,2,n,3),device=device,dtype=torch.float64)
    count=[0,0]
    for start in range(0,len(h),1024):
        sl=slice(start,start+1024)
        for suffix in (('',) if schedule=='original' else ('','_native')):
            t=b['time'+suffix][sl];u=b['u0'][sl];pool=b['pooled'+suffix][sl];mask=b['mask'+suffix][sl]
            hh=h[sl,:model.mean.numel()]
            if schedule=='canonical' and suffix:
                core=query(model.core,hh,u,b['time'][sl],b['pooled'][sl],t,pool)
            else:
                core=model.core(hh,pool,u,t)
            y=(b['y'+suffix][sl]-core).double()
            angle=model.clocks(h[sl],pool,u,t)
            r=-torch.expm1(-t/.35);e=torch.exp(-t/2)
            for stream in (0,1):
                good=mask[:,:,stream].reshape(-1)
                count[stream]+=int(good.sum())
                for arm in range(3):
                    theta=angle[:,:,stream,arm]
                    basis=torch.stack((r[:,:,stream],e[:,:,stream]*theta.cos()-1,e[:,:,stream]*theta.sin()),-1)
                    z=(basis[...,None].double()*x[sl,None,None,:].double()).reshape(-1,n)
                    # Mask both sides via zeroing each invalid design row.
                    z=z*good[:,None]
                    normal[arm,stream].addmm_(z.T,z)
                    rhs[arm,stream].addmm_(z.T,y[:,:,stream,:].reshape(-1,3))
    regularized=normal+RIDGE*torch.diag_embed(torch.diagonal(normal,dim1=-2,dim2=-1))
    # A constant training flag becomes exactly zero after centering. Assign the
    # minimum-norm zero coefficient on its null space; the empirical loss is unchanged.
    inactive=torch.diagonal(normal,dim1=-2,dim2=-1)==0
    solvable=regularized+torch.diag_embed(inactive.double())
    weights=torch.linalg.solve(solvable,rhs)
    residual=((regularized@weights-rhs).abs().amax(dim=(-1,-2))/rhs.abs().amax(dim=(-1,-2))).cpu().numpy()
    assert np.max(residual)<1e-9
    dev,flights=load('validation',device)
    assert len(set(flights))==39 and not set(ids)&set(flights)
    errors={'strong':evaluate(model,dev,flights,'strong')}
    for i,arm in enumerate(CASES):
        model.weights.copy_(weights[i].reshape(2,3,47,3))
        errors[arm]=evaluate(model,dev,flights,arm)
        torch.save(model.state_dict(),out/(arm+'.pt'))
    (out/'flight_mse.json').write_text(json.dumps(errors)+'\n')
    scores={m:{k:float(np.sqrt(np.mean(list(v.values())))) for k,v in row.items()} for m,row in errors.items()}
    (out/'fit.json').write_text(json.dumps(dict(anchor=str(anchor),samples=count,stationarity=residual.tolist(),metrics=scores,
              zero_design_columns=inactive.sum(-1).cpu().tolist(),schedule=schedule,parameters=846,ridge=RIDGE,context_dim=x.shape[1]),indent=2)+'\n')
    print(key,{m:[v['v2'],v['w2']] for m,v in scores.items()},flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('key');ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--schedule',choices=['jittered','canonical','original'],default='jittered')
    args=ap.parse_args();torch.set_num_threads(2);fit(args.key,args.device,args.schedule)
