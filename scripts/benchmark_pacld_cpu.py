import sys,json,time
sys.path.insert(0,'src')
import torch
from system_identification.models.pacld import PACLD,Config
from system_identification.data.pacld import open_windows
from system_identification.training.pacld import window_objective
c=json.load(open('configs/pacld/default.json'));torch.set_num_threads(1);torch.manual_seed(17);d=open_windows(c,'train');records=[]
for h in c['data']['horizon_us']:
 ids=d.index[d.index.horizon_us==h].groupby(['flight_id','sampling_group'],sort=True).head(1).index.tolist()[:8]
 for i in ids:
  t=time.perf_counter();w=d[i];prep=time.perf_counter()-t;m=PACLD(Config(**c['model'])).double();opt=torch.optim.Adam(m.parameters(),**c['optimizer']);t=time.perf_counter();r=window_objective(m,w,.1);f=time.perf_counter()-t;t=time.perf_counter();r['total'].backward();b=time.perf_counter()-t;t=time.perf_counter();torch.nn.utils.clip_grad_norm_(m.parameters(),10);opt.step();o=time.perf_counter()-t
  records.append(dict(horizon_us=h,index=i,window_id=w.metadata['window_id'],prepare_seconds=prep,forward_seconds=f,backward_seconds=b,optimizer_seconds=o,seconds=f+b+o));del r,m,opt
 print(h,sum(r['seconds'] for r in records if r['horizon_us']==h),flush=True)
json.dump(records,open('runs/pacld_gpu_batched/cpu_baseline.json','w'),indent=2)
