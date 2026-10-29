"""Matched true validation-entry timing and loss/covariance component profiling."""
import argparse,sys,json,os,time,inspect
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
os.sched_setaffinity(0,{12});torch.set_num_threads(1);torch.cuda.set_device(0)
sys.path.insert(0,str(Path(a.source)/'src'))
from system_identification.models.pahbd import PAHBD,Config
from system_identification.data.pahbd import NativeWindows,ConfiguredWindows
from system_identification.training.pahbd import validation_selection_score
cfg=json.loads(Path('configs/pahbd/default.json').read_text());d=NativeWindows(**cfg['data'],split='validation',device='cuda:0')
cohort=json.loads(Path('/home/zn/pahbd-runs/ab-20261029-1426/data_identity.json').read_text())['validation_window_ids']
ids=d.windows[d.windows.window_id.isin(cohort)].groupby('horizon_us',sort=True).head(1).index.tolist()
torch.manual_seed(17);m=PAHBD(Config(**cfg['model'])).double().cuda();w=ConfiguredWindows(d,ids,m.config)
validation_selection_score(m,ConfiguredWindows(d,ids[:1],m.config),cfg['loss']);torch.cuda.synchronize();t=time.perf_counter();result=validation_selection_score(m,w,cfg['loss']);torch.cuda.synchronize()
out=dict(seconds=time.perf_counter()-t,windows=len(ids),score=result['score'],ids=d.windows.loc[ids].window_id.tolist())
# Profile exactly the phase covariance method used by integration, from real state.
s=w[0].initial
for _ in range(3):m.flow.propagate_covariance(s,.001)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
    with torch.profiler.record_function('phase_covariance'):
        for _ in range(100):m.flow.propagate_covariance(s,.001)
Path(a.output+'.covariance.txt').write_text(prof.key_averages().table(sort_by='cpu_time_total',row_limit=15))
Path(a.output).write_text(json.dumps(out,indent=2));print(json.dumps(out))
