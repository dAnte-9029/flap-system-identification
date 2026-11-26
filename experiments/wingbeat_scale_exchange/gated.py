"""Test a forecast-conditioned fast-node readout on an already corrected ODE."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run import FROZEN, PROJECT, ControlledODE, prepare, score, METRICS, paired
from module import features


def import_file(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def old_forecast(root,split,device,data,base):
    folder = PROJECT/'research/rounds/periodic-motion-20261125/readout'
    clock_class = import_file('periodic_readout',folder/'model.py').FlightPredictor
    state = torch.load(folder/'original_runs/incumbent17/corrected.pt',map_location=device,weights_only=True)
    clock = clock_class(state['mean'],state['std']).to(device)
    clock.load_state_dict(state)
    transport = import_file('harmonic_module',FROZEN/'waveform/coherence/inference.py')
    head = transport.HarmonicTransport.from_files(17, directory=FROZEN/'waveform/coherence',device=device)
    with np.load(root/f'outputs/autopilot_flight_dynamics_20261124/{split}.npz') as src:
        hist,pool,u0 = [torch.as_tensor(src[k],device=device) for k in ('hist','pooled','u0')]
        np.testing.assert_array_equal(src['time'],data['time'].cpu().numpy())
    pred = []
    for i in range(0,len(base),512):
        s=slice(i,i+512)
        t=data['time'][s]
        angle=clock.clocks(hist[s],pool[s],u0[s],t)[...,-1]
        cond=transport.causal_conditions(hist[s],pool[s],u0[s],t)
        coeff=data['coeff'][s,...,4:].unflatten(-1,(2,2))
        pred.append(base[s]+head(coeff,t,angle,cond))
    return torch.cat(pred)


def amplitude(coeff):
    return coeff[...,2:].square().sum(-1).sqrt()


def design(data,base,units,arm):
    t=data['time']
    phi=features(**data,mean_frequency=None,arm='clock_deleted')
    coarse_gate=(base/units[0]).tanh()
    own_gate=(amplitude(data['coeff'])/units[1]).tanh()[:,None].expand_as(coarse_gate)
    # All arms see both sources additively, including the time-varying ODE prediction.
    envelope=-torch.expm1(-t/.35)
    common_gate=torch.cat((coarse_gate,own_gate),-1)*envelope[...,None]
    common_gate=common_gate.unsqueeze(-2).expand(*phi.shape[:-1],6)
    x=torch.cat((phi,common_gate),-1)
    if arm!='deleted':
        gate=coarse_gate if arm=='cross' else own_gate
        fast=phi[..., -18:]
        interaction=(fast[...,None]*gate[:,:,:,None,None,:]).flatten(-2)
        x=torch.cat((x,interaction),-1)
    return x


class GatedReadout(torch.nn.Module):
    """Add the learned correction to the frozen waveform forecast at native times."""
    def __init__(self, weights, units, arm):
        super().__init__()
        self.register_buffer("weights", weights)
        self.register_buffer("units", units)
        self.arm = arm

    def forward(self, ode_prediction, waveform_prediction, coeff, valid, frequency, motor, time):
        inputs = dict(coeff=coeff, valid=valid, frequency=frequency, motor=motor, time=time)
        x = design(inputs, ode_prediction, self.units, self.arm)
        return waveform_prediction + (x*self.weights).sum(-1)


@torch.inference_mode()
def main(root,out,device):
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    out.mkdir(parents=True,exist_ok=False)
    state=torch.load(FROZEN/'ode_baseline/best.pt',map_location=device,weights_only=True)
    net=ControlledODE(state['mean'],state['scale']).to(device).eval()
    net.load_state_dict(state)
    train,b,y,ft,_=prepare(root,'train',device,net)
    dev,bv,yv,fv,_=prepare(root,'validation',device,net)
    assert set(ft).isdisjoint(fv)
    prior=old_forecast(root,'train',device,train,b)
    priorv=old_forecast(root,'validation',device,dev,bv)
    tables={'reference':score(priorv,yv,fv)}
    old=json.loads((FROZEN/'transfer/ode_result.json').read_text())['flight_mse']['second_round']
    delta=(tables['reference']-pd.DataFrame(old).loc[tables['reference'].index,METRICS]).abs().max().max()
    assert delta<1e-5,delta
    units=torch.stack((b.square().mean((0,1)).sqrt(),
                       amplitude(train['coeff']).square().mean(0).sqrt())).clamp_min(.05)
    residual={}
    for arm in ('cross','self','deleted'):
        n=39 if arm=='deleted' else 93
        matrix=torch.zeros(2,3,n,n,dtype=torch.float64,device=device)
        rhs=torch.zeros_like(matrix[...,0])
        for i in range(0,len(b),512):
            s=slice(i,i+512)
            x=design({k:v[s] for k,v in train.items()},b[s],units,arm).double()
            x=x.permute(2,3,0,1,4).flatten(2,3)
            target=(y[s]-prior[s]).permute(2,3,0,1).flatten(2,3).double()
            matrix+=x.transpose(-1,-2)@x
            rhs+=(x*target[...,None]).sum(-2)
        matrix+=1e-4*torch.diag_embed(matrix.diagonal(dim1=-1,dim2=-2).clamp_min(1e-8))
        w=torch.linalg.solve(matrix,rhs[...,None]).squeeze(-1)
        residual[arm]=float(((matrix@w[...,None]).squeeze(-1)-rhs).abs().max()/rhs.abs().max())
        assert residual[arm]<1e-8
        w=w.float()
        torch.save(dict(weights=w,units=units,arm=arm),out/f'{arm}.pt')
        readout=GatedReadout(w,units,arm)
        pred=[]
        for i in range(0,len(bv),512):
            s=slice(i,i+512)
            pred.append(readout(bv[s],priorv[s],**{k:v[s] for k,v in dev.items()}))
        tables[arm]=score(torch.cat(pred),yv,fv)
    draws=np.random.default_rng(17).integers(0,39,(10000,39))
    comparisons={ref:paired(table,tables['cross'],draws) for ref,table in tables.items() if ref!='cross'}
    result=dict(rmse={k:v.mean().pow(.5).to_dict() for k,v in tables.items()},
                reference_minus_cross=comparisons,prior_mse_replay_max=float(delta),
                normal_equation_relative_residuals=residual)
    pd.concat(tables,names=['arm']).to_csv(out/'flight_mse.csv')
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['rmse']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('/home/zn/flap-system-identification'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    main(a.root,a.out,a.device)
