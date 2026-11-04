"""Restore archived GRU inference shape/TF32; padding rows are never scored."""
import importlib.util,json,os,time
from pathlib import Path
import numpy as np,torch
out=Path(__file__).parent;os.sched_setaffinity(0,{14});torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=True
prep=json.loads((out/'preparation.json').read_text());idx=json.loads((out/'index.json').read_text())
spec=importlib.util.spec_from_file_location('frozen_gru',prep['source_model']);gru=importlib.util.module_from_spec(spec);spec.loader.exec_module(gru)
batch=2048;H=27;K=100;E=3
x=dict(history=np.zeros((batch,H,20),np.float32),history_mask=np.zeros((batch,H,20),bool),history_dt=np.zeros((batch,H),np.float32),history_length=np.ones(batch,np.int64),initial=np.zeros((batch,13),np.float32),dt=np.zeros((batch,K),np.float32),events=np.zeros((batch,K,E,5),np.float32),event_length=np.ones((batch,K),np.int64));x['initial'][:,6]=1;x['history'][:,:,6]=1
for item in idx:
 i=item['i'];z=np.load(out/f'inputs_{i:03d}.npz');h=item['history_count'];k=item['future_count'];e=z['events'].shape[2]
 for name in ['history','history_mask','history_dt']:x[name][i,:h]=z[name][0]
 for name in ['history_length','initial']:x[name][i]=z[name][0]
 for name in ['dt','event_length']:x[name][i,:k]=z[name][0]
 x['events'][i,:k,:e]=z['events'][0]
x={k:torch.from_numpy(v).cuda() for k,v in x.items()}
for entry in prep['checkpoints']:
 seed=entry['seed'];p=out/f'C_seed{seed}';p.rename(out/f'perwindow_C_seed{seed}');p.mkdir();b=torch.load(entry['checkpoint'],map_location='cpu',weights_only=False);m=gru.VelocityRateGRU(b['normalization'],control=True).cuda();m.load_state_dict(b['model']);m.eval();start=time.perf_counter()
 with torch.no_grad():pred=m(x)[:len(idx)].double().cpu().numpy()
 for item in idx:np.savez_compressed(p/f"window_{item['i']:03d}.npz",pred=pred[item['i'],:item['future_count']])
 (p/'verification.json').write_text(json.dumps(dict(checkpoint=entry['checkpoint'],windows=len(idx),batch_size=2048,precision='float32; TF32 enabled as archived',dummy_rows='padding only, never scored; batch-independent GRU',seconds=time.perf_counter()-start),indent=2));print('C',seed,'complete',flush=True)
