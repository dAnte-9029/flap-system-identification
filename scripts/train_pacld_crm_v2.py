"""Exactly 64 x 512 paired draws per arm, atomic checkpoints and strict resume."""
import argparse
import json
import os
import subprocess
import time
import numpy as np
import pandas as pd
import torch
from pacld_crm_v2 import *

def rng_state():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch']);torch.cuda.set_rng_state_all(s['cuda'])

def run(arm,resume):
    contract=initialize();cal=json.loads((OUT/'loss_calibration.json').read_text());pre=json.loads((OUT/'preflight.json').read_text())
    assert cal['status']=='PASS' and pre['status']=='PASS', 'formal training forbidden by preflight/calibration gate'
    assert sha(CHECKPOINT)==SHA
    # Only new scripts may change. Preserve all original model/data/training implementations.
    assert not subprocess.check_output(['git','diff',BASE_SHA,'--','src'],text=True).strip()
    frozen_code={p:sha(p) for p in ['scripts/pacld_crm_v2.py','scripts/train_pacld_crm_v2.py']}
    contract_id=sha(OUT/'training_contract.json');seed()
    bundle=torch.load(CHECKPOINT,map_location='cpu',weights_only=False);model,opt=model_and_optimizer(bundle);store=Store(bundle)
    sample=pd.read_csv(OUT/'shared_sampling_indices.csv');rows=store.frame.loc[sample.cohort_index]
    assert np.array_equal(rows.flight.to_numpy(),sample.flight_id.to_numpy()) and np.array_equal(rows.t.to_numpy(),sample.decision_us.to_numpy())
    directory=LOCAL/arm;ckpt=directory/'latest.pt';history=[];done=0;seconds=0.;forwards=0
    lam=0. if arm=='B0' else cal['lambdas'][arm]
    if directory.exists() and not resume:raise FileExistsError(f'{directory}: use explicit --resume; never overwrite formal results')
    directory.mkdir(exist_ok=True)
    if ckpt.exists():
        saved=torch.load(ckpt,map_location='cpu',weights_only=False)
        assert saved['contract_id']==contract_id and saved['source_sha256']==frozen_code and saved['group']==arm
        assert saved['lambda_CRM']==lam and saved['sampler_position']==saved['completed_updates']*512
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
        done=saved['completed_updates'];seconds=saved['elapsed_seconds'];history=saved['history'];forwards=saved['forward_counts']
        assert done<=64
    if done==64:
        if not (directory/'model_64.pt').exists():os.link(ckpt,directory/'model_64.pt')
        pd.DataFrame(history).to_csv(directory/'progress.csv',index=False)
        parts=[pd.read_csv(LOCAL/g/'progress.csv') for g in ['B0','B1','B2'] if (LOCAL/g/'progress.csv').exists()]
        pd.concat(parts,ignore_index=True).to_csv(OUT/'training_progress.csv',index=False)
        return
    write_json(directory/'status.json',dict(status='preparing_train_cache',group=arm,completed_updates=done,pid=os.getpid()))
    torch.cuda.reset_peak_memory_stats();model.train()
    for update in range(done+1,65):
        ids=sample.loc[sample['update']==update,'cohort_index'].to_list();assert len(ids)==512
        start=time.monotonic();store.prepare(ids);write_json(directory/'status.json',dict(status='training',group=arm,update=update,completed_updates=update-1,pid=os.getpid()))
        M=first_pass(model,store,ids,contract['microbatch']) if arm!='B0' else np.array([np.nan,np.nan])
        if arm!='B0':assert np.isfinite(M).all()
        base,valid,M_second=accumulated_backward(model,store,ids,contract['microbatch'],arm,lam,M)
        if arm=='B0':M=M_second
        else:np.testing.assert_allclose(M_second,M,rtol=1e-7,atol=1e-9)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),contract['max_grad_norm'],error_if_nonfinite=True)
        opt.step();torch.cuda.synchronize();elapsed=time.monotonic()-start;seconds+=elapsed
        assert all(torch.isfinite(p).all() for p in model.parameters())
        micro_steps=512//contract['microbatch'];forwards+=micro_steps*(3 if arm!='B0' else 2)
        # B0 Moment diagnostic is no-grad and contributes no optimizer gradient.
        loss=float(base[2]+(lam*M[0 if arm=='B1' else 1]**2 if arm!='B0' else 0.))
        record=dict(group=arm,update=update,cumulative_windows=512*update,unique_windows_seen=int(sample.loc[sample['update']<=update,'cohort_index'].nunique()),full=float(base[0]),macro=float(base[1]),loss=loss,M_normal=float(M[0]),M_future100=float(M[1]),lambda_CRM=lam,gradient_norm_preclip=float(norm),clipped=bool(norm>contract['max_grad_norm']),macro_valid_windows=valid,seconds=elapsed,cumulative_seconds=seconds,windows_per_second=512*update/seconds,eta_group_seconds=(64-update)*seconds/update,peak_cuda_bytes=torch.cuda.max_memory_allocated(),forward_counts=forwards,metric_timing='objective and gradient computed at pre-step parameters; recorded after atomic optimizer step')
        history.append(record)
        saved=dict(schema='crm_v2_train',group=arm,model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),contract_id=contract_id,source_sha256=frozen_code,config=bundle['config'],precision=bundle['precision'],data_identity=bundle['data_identity'],lambda_macro=1.,lambda_CRM=lam,completed_updates=update,sampler_position=update*512,sampling_sha256=contract['sampling_sha256'],elapsed_seconds=seconds,forward_counts=forwards,history=history,code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
        temp=ckpt.with_suffix('.tmp');torch.save(saved,temp);temp.replace(ckpt)
        if update==64:
            final=directory/'model_64.pt';assert not final.exists();os.link(ckpt,final)
        pd.DataFrame(history).to_csv(directory/'progress.csv',index=False)
        allrows=[]
        for group in ['B0','B1','B2']:
            p=LOCAL/group/'progress.csv'
            if p.exists():allrows.append(pd.read_csv(p))
        pd.concat(allrows,ignore_index=True).to_csv(OUT/'training_progress.csv',index=False)
        write_json(directory/'status.json',dict(status='complete' if update==64 else 'training',pid=os.getpid(),**record))
        print(json.dumps(record,allow_nan=False),flush=True)
    write_json(directory/'status.json',dict(status='complete',group=arm,completed_updates=64,sampled_windows=32768,elapsed_seconds=seconds,peak_cuda_bytes=torch.cuda.max_memory_allocated(),forward_counts=forwards))

def main():
    p=argparse.ArgumentParser();p.add_argument('--group',choices=['B0','B1','B2'],required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    try:run(a.group,a.resume)
    except Exception as e:
        write_json(LOCAL/a.group/'status.json',dict(status='failed',error=repr(e),pid=os.getpid()));raise
if __name__=='__main__':main()
