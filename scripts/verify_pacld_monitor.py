#!/usr/bin/env python
"""Paired no-Macro GPU regression and bounded monitor overhead measurement."""
import argparse,contextlib,copy,importlib.util,io,json,statistics,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from run_pacld import datasets
from system_identification.models.pacld import Config
from system_identification.models.pacld.batch import BatchedPACLD
from system_identification.data.pacld_batch import compile_windows
from system_identification.training.pacld_monitor import Monitor


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(17)
    spec=importlib.util.spec_from_file_location('regression',Path(__file__).resolve().parents[1]/'tests/test_pacld_gpu_batch.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    config=json.loads(Path('configs/pacld/gpu.json').read_text());train,_,_=datasets(config);cfg=Config(**config['model'])
    ids=train.index[train.index.horizon_us==200000].index[:32].tolist()
    batch=compile_windows([train[i] for i in ids],cfg).to('cuda:0')
    assert batch.pairs['batch'].numel()==0,'choose naturally no-pair windows, never erase supervision'
    old=module.LegacyBatchedPACLD(cfg,'mixed',True).cuda();new=BatchedPACLD(cfg,'mixed',True).cuda();new.load_state_dict(old.state_dict())
    initial=copy.deepcopy(old.state_dict());outputs=[];records=[[],[]]
    for iteration in range(5):
        for i,model in enumerate((old,new)):
            model.load_state_dict(initial);optimizer=torch.optim.Adam(model.parameters(),lr=.0003)
            optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
            result=model(batch,1.);result['total'].backward()
            grad=torch.cat([p.grad.detach().flatten().double() for p in model.parameters() if p.grad is not None])
            torch.nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step();torch.cuda.synchronize()
            records[i].append(time.perf_counter()-start)
            if iteration==0:
                outputs.append(dict(pred=result['predictions'][:32].detach().cpu(),loss=result['total'].detach().cpu(),grad=grad.cpu(),params=torch.cat([p.detach().flatten().double() for p in model.parameters()]).cpu(),adam=torch.cat([s[k].detach().flatten().double() for s in optimizer.state.values() for k in ('exp_avg','exp_avg_sq')]).cpu()))
            del result,grad,optimizer
    errors={}
    for key in outputs[0]:
        x,y=outputs[0][key],outputs[1][key];errors[key]=dict(max_abs=float((x-y).abs().max()),relative_l2=float((x-y).norm()/x.norm().clamp_min(1e-15)))
        torch.testing.assert_close(x,y,rtol=1e-4,atol=1e-5)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    schedule=dict(optimizer_steps=20,windows_seen=640,best_score=float('inf'))
    with out.with_suffix('.stdout.log').open('w') as terminal, contextlib.redirect_stdout(terminal):
        monitor=Monitor(out.with_suffix('.progress.log'),17,0,32,342,32,schedule)
        for _ in range(25):monitor.observe(dict(end_to_end_seconds=3.),{})
        begin=time.perf_counter()
        for _ in range(100):monitor.status('TRAINING')
        overhead=(time.perf_counter()-begin)/100;monitor.close()
    means=[statistics.mean(r[1:]) for r in records]
    result=dict(window_ids=[m['window_id'] for m in batch.metadata],macro_pairs=0,precision='mixed',compile=True,errors=errors,seconds=records,steady_mean_seconds=means,speedup=means[0]/means[1],log_block_seconds=overhead,amortized_log_seconds_per_step=overhead/10)
    out.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
