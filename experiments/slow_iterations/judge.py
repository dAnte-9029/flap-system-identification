"""Shared flight-equal validation for every independent fit and frozen-state intervention."""
import importlib.util
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from shared import HERE,PRIOR,ROOT,PREV,wave,read,load_tuned
from dataset import anchor_input
from fit_coupled import call
from shifted import observations

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

@torch.inference_mode()
def run():
    torch.set_num_threads(3)
    ref=module('fixed_metrics',PRIOR/'evaluate.py')
    code=Path(os.environ.get('RA_MODEL_CODE',HERE.parents[1]))
    method=module('candidate_recipe',code/'experiments/slow_iterations/recipe.py')
    out=Path(os.environ['RA_OUTPUT'])
    packs=torch.load(os.environ['RA_MODEL'],weights_only=True,map_location='cpu')
    orig=load_tuned(17,'cpu').eval()
    d,flight,_=wave.load(ROOT,'validation','cpu')
    with np.load(PREV/'validation.npz') as z:b={k:z[k] for k in ('ok','origin','anchor','window')}
    with np.load(ROOT/'outputs/autopilot_flight_dynamics_20261124/validation.npz') as z:np.testing.assert_array_equal(z['window'],b['window'])
    b['a']=anchor_input(b);b['origin_tensor']=torch.from_numpy(b['origin']).float()
    cache=wave.native_cache(ROOT,'cpu');points=observations(d,flight,b)
    fast=read('validation','cpu');results={}
    for name,pack in packs.items():
        net=method.build(orig,pack['arm']).eval()
        modes=[(name,'state',False)]
        if net.coupled and pack['arm'].get('feedback'):modes.append((name+'_held','state',True))
        modes.append((name+'_selected','selected',False))
        for tag,key,freeze in modes:
            net.load_state_dict(pack[key])
            sub=out/tag;sub.mkdir(exist_ok=True)
            errors=[]
            for ix in torch.arange(len(flight)).split(768):
                p,_=call(net,fast,ix,freeze)
                errors.append(ref.mse(p[:,[1,3,7]].numpy(),fast['y'][ix][:,[1,3,7]].numpy()).reshape(len(ix),6))
            table=pd.DataFrame(np.concatenate(errors),columns=wave.METRICS).assign(flight=flight).groupby('flight').mean()
            table.to_csv(sub/'endpoints.csv')
            ep=table.mean().pow(.5).to_dict()
            results[tag]=dict(endpoint=ep)
            if key=='state':
                results[tag]['beats']=ref.beats(net,orig,d,b,flight,freeze,cache,sub)
                results[tag]['shifted']=ref.shifted(net,orig,d,b,flight,freeze,points,sub)
            print(tag,ep,flush=True)
    (out/'validation.json').write_text(json.dumps(results,allow_nan=False,indent=2)+'\n')
    m=results.get('full',results.get('baseline'))
    metrics={**m['endpoint'],'slowrate':m['beats']['full_all']['rmse'],'offgrid2':m['shifted']['2.0']['minus125']}
    Path(os.environ['RA_METRICS']).write_text(json.dumps(metrics,allow_nan=False)+'\n')

if __name__=='__main__':run()
