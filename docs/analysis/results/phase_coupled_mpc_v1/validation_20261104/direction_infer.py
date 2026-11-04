"""Fresh inference of four selected PA-HBD checkpoints on the original fixed validation cohort."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--variant',choices=['A','B'],required=True);p.add_argument('--cpu',type=int,required=True);a=p.parse_args()
os.sched_setaffinity(0,{a.cpu});torch.set_num_threads(1);torch.cuda.set_device(0)
repo=Path('/home/zn/flap-system-identification/.worktrees/pahbd-compile-v4');sys.path.insert(0,str(repo/'src'))
from system_identification.training.pahbd import load_checkpoint,window_loss
from system_identification.data.pahbd import NativeWindows,ConfiguredWindows
from system_identification.models.pahbd.tensor_flow import enable_tensor_engine
from system_identification.evaluation.pahbd import native_horizon_errors,aggregate_flights
out=Path(__file__).parent;s=json.loads((out/'summary.json').read_text());root=Path(s['run']);data=None
for seed in [17,29]:
    name=f'{a.variant}_seed{seed}';dest=out/f'direction-{name}';dest.mkdir(exist_ok=False)
    selected=next(r for r in s['models'] if r['task']==name);model,_,bundle=load_checkpoint(selected['checkpoint'],'cuda:0');model.eval();enable_tensor_engine(model,'step')
    meta=bundle['metadata'];cfg=meta['config']
    if data is None:data=NativeWindows(**cfg['data'],split='validation',device='cuda:0')
    lookup={x:i for i,x in enumerate(data.windows.window_id)};indices=[lookup[x] for x in meta['validation_window_ids']];windows=ConfiguredWindows(data,indices,model.config)
    assert data.manifest_sha256==meta['data']['manifest_sha256'] and data.phase_contract==meta['phase_contract']
    losses=[];native=[];index=[];start=time.perf_counter()
    for i in range(len(windows)):
        w=windows[i]
        with torch.no_grad():loss,parts,belief,pred,times=window_loss(model,w,**cfg['loss'])
        native.extend(native_horizon_errors(pred,times,w.targets,w.metadata));losses.append(dict(flight_id=w.metadata['flight_id'],horizon_us=w.metadata['horizon_us'],loss=float(loss)))
        causal=[e for e in w.history if e.source=='vehicle_angular_velocity' and e.measurement<=0 and e.availability<=0 and bool(e.mask[9:12].all())]
        if not causal:raise ValueError('no causal angular-rate anchor')
        anchor=max(causal,key=lambda e:e.measurement)
        assert -anchor.measurement<=data.contract['history_rate_age_us']*1e-6
        gyro=[e for e in w.targets if e.source=='vehicle_angular_velocity' and bool(e.mask[9:12].all())];state=dict(zip(times,pred))
        t=np.array([e.measurement for e in gyro]);truth=torch.stack([e.value[10:13] for e in gyro]).cpu().numpy();forecast=torch.stack([state[e.measurement].x[10:13] for e in gyro]).cpu().numpy()
        assert len(t)>1 and np.all(np.diff(t)>0) and np.isfinite(forecast).all()
        np.savez_compressed(dest/f'window_{i:03d}.npz',times=t,truth=truth,pred=forecast,origin=anchor.value[10:13].cpu().numpy(),belief=belief.x[10:13].cpu().numpy())
        index.append(dict(i=i,window_id=w.metadata['window_id'],flight_id=w.metadata['flight_id'],horizon_us=w.metadata['horizon_us'],sampling_group=w.metadata['sampling_group'],anchor_measurement=anchor.measurement,anchor_availability=anchor.availability,points=len(t)))
        if i<2 or (i+1)%25==0:print(name,i+1,len(windows),'seconds',round(time.perf_counter()-start,1),flush=True)
    score=float(np.mean([r['loss'] for r in losses]));assert len(losses)==156 and len({r['flight_id'] for r in losses})==39
    np.testing.assert_allclose(score,selected['best_score'],rtol=1e-8,atol=1e-8)
    old=json.loads((root/name/'progress.json').read_text())['records'];old=next(r['validation'] for r in old if r['step']==bundle['step'])
    new=aggregate_flights(native);prior={(r['component'],r['horizon_s']):r for r in old['native_flight_equal']}
    for r in new:np.testing.assert_allclose(r['rmse'],prior[r['component'],r['horizon_s']]['rmse'],rtol=1e-8,atol=1e-8)
    (dest/'index.json').write_text(json.dumps(index,indent=2));(dest/'verification.json').write_text(json.dumps(dict(checkpoint=selected['checkpoint'],windows=len(windows),score=score,saved_score=selected['best_score'],score_difference=score-selected['best_score'],native_metrics_match=True,seconds=time.perf_counter()-start),indent=2))
    print(name,'complete; saved score and all native RMSE reproduced',flush=True)
