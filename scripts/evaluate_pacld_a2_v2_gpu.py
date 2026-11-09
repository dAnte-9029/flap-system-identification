"""Read-only frozen factual PACLD inference for preselected A2 identities."""
import argparse
from pathlib import Path
import hashlib
import json
import time
import pickle
import numpy as np
import pandas as pd
import torch
from system_identification.data.pacld import open_windows
from system_identification.models.pacld import Config
from system_identification.models.pacld.batch import BatchedPACLD
from system_identification.evaluation.pacld_gpu import predict
from diagnose_pacld_a2_v2 import OUT,SOURCE

CHECKPOINT=Path('/home/zn/flap-system-identification/.worktrees/pacld-gpu-batched-v1/runs/pacld_frozen_eval_seed17/model.pt')
EXPECTED='9d82640692c0baec476e276696be4f89fd89a58854361bbcfdf3dbb84cb466c9'

_WORKER_WINDOWS=None
_WORKER_LOOKUP=None

def prepare_batch(payload):
    global _WORKER_WINDOWS, _WORKER_LOOKUP
    cfg,split,cohort=payload
    if _WORKER_WINDOWS is None:
        torch.set_num_threads(1)
        _WORKER_WINDOWS=open_windows(cfg,split)
        selected=_WORKER_WINDOWS.index[_WORKER_WINDOWS.index.horizon_us==500000].drop_duplicates(['flight_id','decision_us'])
        _WORKER_LOOKUP={(r.flight_id,int(r.decision_us)):i for i,r in selected.iterrows()}
    windows=_WORKER_WINDOWS;lookup=_WORKER_LOOKUP;ws=[];queries=[]
    for r in cohort.itertuples():
        w=windows[lookup[(r.flight,int(r.t))]]
        target=min((e for e in w.targets if e.source=='vehicle_angular_velocity'),key=lambda e:abs(e.measurement-.2))
        expected=.2+r.offset[0]*1e-6
        assert abs(target.measurement-expected)<1e-9 and bool(target.mask[9:12].all())
        assert all(e.availability<=0 for e in w.history)
        # Feature target identity; controls come unchanged from the frozen native reader.
        assert np.allclose((target.value[10:13].numpy()-np.array(r.X)[3:6])[[0,2]],np.array(r.Y)[[0,2]],atol=1e-10)
        ws.append(w);queries.append([target.measurement])
    return pickle.dumps((cohort,ws,queries),protocol=5)

def batches(todo,cfg,split,batch_size,workers):
    from concurrent.futures import ProcessPoolExecutor
    from collections import deque
    import multiprocessing
    payloads=iter((cfg,split,todo.iloc[i:i+batch_size]) for i in range(0,len(todo),batch_size))
    if workers==0:
        for p in payloads:yield pickle.loads(prepare_batch(p))
        return
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        queue=deque()
        for _ in range(workers*2):
            p=next(payloads,None)
            if p is not None:queue.append(pool.submit(prepare_batch,p))
        while queue:
            yield pickle.loads(queue.popleft().result())
            p=next(payloads,None)
            if p is not None:queue.append(pool.submit(prepare_batch,p))

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['train','validation'],required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=128);p.add_argument('--workers',type=int,default=4);p.add_argument('--resume',action='store_true');a=p.parse_args()
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()==EXPECTED
    bundle=torch.load(CHECKPOINT,map_location='cpu',weights_only=False);cfg=bundle['config'];torch.set_num_threads(1)
    frozen=json.load(open('configs/pacld/default.json'))
    assert cfg['model']==frozen['model'] and cfg['data']==frozen['data']
    windows=open_windows(cfg,a.split);identity=windows.identity()
    for key in ['manifest_sha256','index_sha256','windows','split']:assert identity[key]==bundle['data_identity'][a.split][key],key
    source=SOURCE if a.split=='train' else OUT/'validation_features'
    frame=pd.read_pickle(source/'features.pkl');assert frame.flight.nunique()==(114 if a.split=='train' else 39)
    selected=windows.index[windows.index.horizon_us==500000].drop_duplicates(['flight_id','decision_us']);lookup={(r.flight_id,int(r.decision_us)):i for i,r in selected.iterrows()}
    model=BatchedPACLD(Config(**cfg['model']),bundle['precision']).to(a.device).eval();model.load_state_dict(bundle['model'])
    destination=OUT/f'pacld_{a.split}.csv';records=[];done=set()
    if destination.exists():
        if not a.resume:raise FileExistsError(destination)
        records=pd.read_csv(destination).to_dict('records');done={(r['flight'],int(r['t'])) for r in records}
    meta=dict(checkpoint_sha256=EXPECTED,config=cfg,data_identity=identity,split=a.split,source=str(source),query='nearest native rate target to .2s; exact measured timestamp',response='factual target rate minus same-window causal belief rate',no_future_observation_assimilation=True)
    (OUT/f'pacld_{a.split}_identity.json').write_text(json.dumps(meta,indent=2))
    start=time.monotonic();todo=frame[~pd.MultiIndex.from_frame(frame[['flight','t']]).isin(done)]
    for cohort,ws,queries in batches(todo,cfg,a.split,a.batch_size,a.workers):
        with torch.inference_mode():result=predict(model,ws,queries,base=False)
        for r,w,rr in zip(cohort.itertuples(),ws,result):
            initial=rr['belief'].x.numpy();future=rr['predictions']['FULL'][0].x.numpy();delta=(future-initial)[[10,12]]
            records.append(dict(flight=r.flight,date=r.date,t=int(r.t),window_id=w.metadata['window_id'],query_s=rr['times'][0],delta_p=float(delta[0]),delta_r=float(delta[1]),belief_p=float(initial[10]),belief_r=float(initial[12])))
        temporary=destination.with_suffix('.tmp.csv');pd.DataFrame(records).to_csv(temporary,index=False);temporary.replace(destination)
        print(a.split,len(records),'/',len(frame),'elapsed_s',round(time.monotonic()-start,1),flush=True)
    assert len(records)==len(frame)
if __name__=='__main__':main()
