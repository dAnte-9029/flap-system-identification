import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from bc_features import ROOT,PREV,CACHE,wave,load_tuned,anchor_input
from bc_fit import partition,endpoint
from bc_model import Predictor
from shifted import observations
import importlib.util
spec=importlib.util.spec_from_file_location('native_scores',Path(__file__).resolve().parents[1]/'coupled_slow/evaluate.py')
ref=importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)

def points(model,clock,d,b,wi,t,p,freeze,slow=False):
    answers=[];trends=[]
    for lo in range(0,len(wi),512):
        ids=wi[lo:lo+512];qt=t[lo:lo+512];qp=p[lo:lo+512]
        ct=d['time'][ids];cp=d['pooled'][ids]
        before=(ct[:,:,1]<qt[:,0,1,None]-1e-7).sum(1) if model.include_a else torch.zeros(len(ids),dtype=torch.long)
        full=torch.empty(len(ids),3);trend=torch.empty_like(full)
        for n in torch.unique(before).tolist():
            ix=torch.where(before==n)[0];rows=ids[ix]
            tt=torch.cat((ct[ix,:n,1],qt[ix,:,1]),1)[:,:,None].expand(-1,-1,2)
            pp=torch.cat((cp[ix,:n,1],qp[ix,:,1]),1)[:,:,None].expand(-1,-1,2,-1)
            orig=b['origin_tensor'][rows]
            raw,base=model.forecast(d['hist'][rows],pp,d['u0'][rows],tt,d['coef'][rows],
                                   b['a'][rows],orig,b['stats'],clock)
            full[ix]=raw[:,-1,1]+orig;trend[ix]=base[:,-1,1]+orig
        answers.append(full.numpy());trends.append(trend.numpy())
    return np.concatenate(answers),np.concatenate(trends) if slow else None

@torch.inference_mode()
def run():
    torch.set_num_threads(3)
    out=Path(os.environ['RA_OUTPUT']);pack=torch.load(os.environ['RA_MODEL'],map_location='cpu',weights_only=True)
    clock=load_tuned();del clock.first_order.core
    model=Predictor(pack['include_a']).eval()
    d,flight,_=wave.load(ROOT,'validation','cpu');v=partition('validation','cpu')
    with np.load(PREV/'validation.npz') as z:b={k:z[k] for k in ('window','origin','ok','anchor')}
    with np.load(ROOT/'outputs/autopilot_flight_dynamics_20261124/validation.npz') as z:np.testing.assert_array_equal(z['window'],b['window'])
    b.update(a=anchor_input(b),origin_tensor=torch.from_numpy(b['origin']).float(),stats=torch.load(CACHE/'stats.pt',weights_only=True))
    cache=wave.native_cache(ROOT,'cpu');shift=observations(d,flight,b)
    results={}
    for tag in ('state','selected'):
        sub=out/tag;sub.mkdir(exist_ok=True)
        model.load_state_dict(pack[tag])
        err=[]
        for ix in torch.arange(len(flight)).split(640):
            full=endpoint(model,v,ix)[0]
            err.append(ref.mse(full[:,[1,3,7]].numpy(),v['y'][ix][:,[1,3,7]].numpy()).reshape(len(ix),6))
        fr=pd.DataFrame(np.concatenate(err),columns=wave.METRICS).assign(flight=flight).groupby('flight').mean()
        fr.to_csv(sub/'endpoints.csv');results[tag]=dict(endpoint=fr.mean().pow(.5).to_dict())
        if tag=='state':
            results[tag]['beats']=ref.beats(model,clock,d,b,flight,False,cache,sub,point_fn=points)
            results[tag]['shifted']=ref.shifted(model,clock,d,b,flight,False,shift,sub,point_fn=points)
    (out/'validation.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    m=results['state'];metrics=dict(**m['endpoint'],slowrate=m['beats']['full_all']['rmse'],offgrid2=m['shifted']['2.0']['minus125'])
    Path(os.environ['RA_METRICS']).write_text(json.dumps(metrics,allow_nan=False)+'\n')
    print(metrics,flush=True)
if __name__=='__main__':run()
