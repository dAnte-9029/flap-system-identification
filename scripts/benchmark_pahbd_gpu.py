#!/usr/bin/env python3
"""Real Hall-window GPU launch check; isolated from formal runs and budgets."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from system_identification.data.pahbd import NativeWindows,ConfiguredWindows
from system_identification.models.pahbd import PAHBD,Config
from system_identification.training.pahbd import window_loss,validation_selection_score,save_checkpoint,load_checkpoint
from run_pahbd import validation_indices


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--device',required=True)
    a=p.parse_args();torch.set_num_threads(1)
    if not a.device.startswith('cuda'):raise ValueError('requires actual GPU measurements')
    root=a.run_root;out=root/'preflight';out.mkdir(exist_ok=False)
    cfg=json.loads((root/'configs/A_seed17.json').read_text());records=[]
    started=time.monotonic();train=NativeWindows(**cfg['data'],split='train',device=a.device)
    val=NativeWindows(**cfg['data'],split='validation',device=a.device)
    index_seconds=time.monotonic()-started
    ids=train.windows[train.windows.shared_periodic_eligible].groupby('horizon_us',sort=True).head(1).index.tolist()
    assert sorted(train.windows.loc[ids].horizon_us)==[200000,500000,1000000,2000000]
    for mode in ('A','B'):
        torch.manual_seed(17);mc=Config(**dict(cfg['model'],mode=mode));model=PAHBD(mc).to(device=a.device,dtype=torch.float64)
        opt=torch.optim.Adam(model.parameters(),**cfg['optimizer']);windows=ConfiguredWindows(train,ids,mc)
        for n,wid in enumerate(ids):
            start=time.monotonic();w=windows[n];torch.cuda.synchronize();loaded=time.monotonic()
            torch.cuda.reset_peak_memory_stats();opt.zero_grad(set_to_none=True)
            loss,*_=window_loss(model,w,**cfg['loss']);loss.backward()
            assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            forcing=sum(float(p.grad.abs().sum()) for p in model.flow.mixer.forcing.parameters() if p.grad is not None)
            control=sum(float(p.grad.abs().sum()) for p in model.flow.mixer.control.parameters() if p.grad is not None)
            assert forcing>0 and (control>0 if mode=='A' else control==0)
            torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step();torch.cuda.synchronize()
            r=dict(mode=mode,horizon_us=int(train.windows.loc[wid].horizon_us),loss=float(loss),forcing_gradient=forcing,
                   control_gradient=control,data_seconds=loaded-start,compute_seconds=time.monotonic()-loaded,
                   total_seconds=time.monotonic()-start,peak_memory_bytes=torch.cuda.max_memory_allocated())
            records.append(r);print(json.dumps(r),flush=True)
        vf=val.windows[val.windows.shared_periodic_eligible];vi=validation_indices(vf,1701,1)
        # Representative one-per-horizon validation; extrapolate full fixed cohort cost explicitly.
        subset=val.windows.loc[vi].groupby('horizon_us',sort=True).head(1).index.tolist()
        vw=ConfiguredWindows(val,subset,mc);torch.cuda.synchronize();start=time.monotonic()
        result=validation_selection_score(model,vw,cfg['loss']);torch.cuda.synchronize()
        val_seconds=time.monotonic()-start
        assert torch.isfinite(torch.tensor(result['score']))
        path=out/f'{mode}.pt';save_checkpoint(path,model,opt,dict(config=cfg),4)
        restored,_,_=load_checkpoint(path,a.device,optimizer=True)
        assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
        records.append(dict(mode=mode,validation_seconds=val_seconds,validation_sample_count=len(subset),full_validation_count=len(vi)))
        del restored,model,opt;torch.cuda.empty_cache()
    costs=[r['total_seconds'] for r in records if 'total_seconds' in r]
    validation_cost=sum(r['validation_seconds']/r['validation_sample_count']*r['full_validation_count'] for r in records if 'validation_seconds' in r)/2
    task_eta=index_seconds+sum(costs)/len(costs)*10944+validation_cost*12
    result=dict(records=records,index_seconds=index_seconds,mean_step_seconds=sum(costs)/len(costs),
                task_eta_seconds=task_eta,queue_eta_seconds=task_eta*6,
                limitation='8 real train updates; 4 validation windows per variant; cohort and long-run throughput extrapolated')
    (root/'benchmark.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
