#!/usr/bin/env python3
"""GPU-only A0/A1/A2/A3, same architecture/two-stage freezing and data contract."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-main-v2-step3')
import sys,json,time,argparse,subprocess
from pathlib import Path
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch,numpy as np,pandas as pd
from run_main_v2_free_running import sha
from system_identification.models.trajectory_main_v1 import CausalHistoryTrajectoryModel
from system_identification.models.trajectory_main_v2 import ActuatorAwareTrajectoryModel
from system_identification.training.trajectory_main_v1 import assemble_history_trajectory_windows,fit_main_v1_stats,_model_call
from system_identification.training.trajectory_main_v2 import fit_main_v2_stats
from system_identification.training.main_v2_objectives import train_stage,objective
from system_identification.evaluation.main_v2_free_running import select_windows,describe


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,default=ROOT/'artifacts/main_v2_training_objective_ablation');pa.add_argument('--results',type=Path,default=ROOT/'docs/analysis/results/main_v2_training_objective_ablation');pa.add_argument('--device',default='cuda:1');pa.add_argument('--preflight-only',action='store_true');pa.add_argument('--experiment');args=pa.parse_args()
    if not args.device.startswith('cuda') or not torch.cuda.is_available():raise RuntimeError('GPU training is required')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    results=args.results;pre=results/'pretraining'
    if not (pre/'summary.json').exists():raise RuntimeError('complete frozen pretraining diagnostics first')
    old=json.loads((ROOT/'docs/analysis/results/main_v2_free_running_5s/summary.json').read_text())
    for n,h in old['source_hashes'].items():
        if sha(ROOT/n)!=h:raise ValueError(f'baseline input changed: {n}')
    dataset=ROOT/'dataset/trajectory_v1_august_f5_c4'
    samples=pd.read_parquet(dataset/'samples_train.parquet');windows=pd.read_parquet(dataset/'windows_train.parquet')
    batch=assemble_history_trajectory_windows(samples,windows,history_steps=26)
    stats=fit_main_v1_stats(samples,batch);tail=fit_main_v2_stats(batch)
    frozen=torch.load(ROOT/'artifacts/trajectory_main_v2/models/main_v2_drive_tail_gated.pt',map_location='cpu',weights_only=False)
    for n,v in vars(stats).items():
        np.testing.assert_array_equal(np.asarray(v,dtype=np.float32),frozen['state_dict']['base_model.'+n].numpy())
    np.testing.assert_array_equal(tail.tail_mean,frozen['tail_mean']);np.testing.assert_array_equal(tail.tail_std,frozen['tail_std'])
    lc=pd.read_csv(pre/'train_loss_components.csv');channels=['position','velocity','attitude','body_rate','phase','frequency']
    means={int(k):{n:float(np.average(g[n],weights=g['count'])) for n in channels} for k,g in lc.groupby('steps')}
    total=np.array([sum(means[k].values()) for k in (10,25,50,100)])
    weights=1/total;weights=weights/weights.sum();prefix=list(zip((10,25,50,100),weights.tolist()))
    # Delta lag two avoids native-dt division and attenuates near-Nyquist
    # differencing noise; pointwise vector alignment penalizes artificial noise.
    lag=2
    w=select_windows(samples)[0];tb=assemble_history_trajectory_windows(samples,w,history_steps=26)
    d=np.load(pre/'train/free_run.npz');p=d['angular_velocity_b'][:,:101];t=tb.trajectory.truth.angular_velocity_b[:,:101]
    delta=((p[:,lag:]-p[:,:-lag])-(t[:,lag:]-t[:,:-lag]))/2
    delta_scale=float(np.mean(np.sum(delta**2,axis=-1)))
    rate_scale=sum(weight*means[k]['body_rate'] for k,weight in prefix)
    delta_weight=rate_scale/delta_scale
    configs=[dict(name='A0_baseline_retrain',steps=50,prefix_weights=[],delta_weight=0.),
             dict(name='A1_longer_rollout',steps=100,prefix_weights=[],delta_weight=0.),
             dict(name='A2_multi_horizon',steps=100,prefix_weights=prefix,delta_weight=0.),
             dict(name='A3_dynamic_delta',steps=100,prefix_weights=prefix,delta_weight=delta_weight)]
    protocol=dict(experiments=configs,delta_lag=lag,train_loss_component_means=means,
        delta_weight_rule='match added delta loss to existing weighted body-rate contribution on frozen train predictions',
        prefix_rule='inverse frozen-train prefix total loss, normalized to sum one; no state weights changed',
        delta_scale=delta_scale,body_rate_scale=rate_scale,train_windows=len(windows),
        train_horizon_seconds={str(k):describe(batch.trajectory.dt_s[:,:k].sum(1)) for k in (50,100)},
        seeds=dict(base=17,actuator=29),epochs=dict(base=40,actuator=25),batch_size=256,
        optimizer='AdamW',learning_rates=dict(base=.0003,actuator=.0005),weight_decay=.00001,gradient_clip_norm=5.,
        checkpoint_criterion='fixed final epoch; no best-epoch tuning',actuator_constants=dict(drive=.1,tail=.04),
        frozen_validation_windows_sha256=sha(ROOT/'docs/analysis/results/main_v2_free_running_5s/windows.csv'),
        architecture_changed=False,simulator_changed=False,sealed_test_opened=False,
        three_second_training='not in primary matrix: fixed original 4214 windows contain 100 steps; avoid a changed cohort confound',
        baseline_hashes=old['source_hashes'])
    def base():
        torch.manual_seed(17);torch.cuda.manual_seed_all(17)
        return CausalHistoryTrajectoryModel(hidden_size=64,use_controls=False,**vars(stats))
    if args.preflight_only:
        m=base().to(args.device)
        for c in configs:
            p,t=_model_call(m,batch,np.arange(8),use_history=True,rollout_steps=c['steps'],device=torch.device(args.device))
            loss=objective(p,t,steps=c['steps'],prefix_weights=c['prefix_weights'],delta_lag=lag,delta_weight=c['delta_weight'])
            m.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),5.)
            assert torch.isfinite(loss) and torch.isfinite(norm)
            print(c['name'],float(loss),float(norm),flush=True)
        print(json.dumps(protocol,indent=2));return
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError(args.output)
    args.output.mkdir(parents=True);results.mkdir(parents=True,exist_ok=True)
    (results/'protocol.json').write_text(json.dumps(protocol,indent=2))
    pd.DataFrame(configs).to_csv(results/'experiment_manifest.csv',index=False)
    allsummary=[]
    for c in configs:
        if args.experiment and c['name'] != args.experiment:
            continue
        name=c['name'];folder=args.output/name;folder.mkdir()
        print('START',name,flush=True);started=time.time()
        def callback(model,history):
            row=history.iloc[-1].to_dict()
            (args.output/'status.json').write_text(json.dumps(dict(experiment=name,stage=stage,status='running',**row),indent=2))
            if int(row['epoch'])%5==0:print(name,stage,row,flush=True)
        shared=dict(steps=c['steps'],prefix_weights=c['prefix_weights'],delta_lag=lag,delta_weight=c['delta_weight'],device=args.device,callback=callback)
        stage='base';m,hb=train_stage(base(),batch,epochs=40,seed=17,learning_rate=.0003,**shared)
        torch.save(dict(state_dict=m.state_dict(),experiment=c,stage='base'),folder/'base.pt');hb.to_csv(folder/'base_history.csv',index=False)
        torch.manual_seed(29);torch.cuda.manual_seed_all(29)
        model=ActuatorAwareTrajectoryModel(base_model=m,use_drive=True,use_tail=True,gated_tail=True,tail_mean=tail.tail_mean,tail_std=tail.tail_std)
        stage='actuator';model,ha=train_stage(model,batch,epochs=25,seed=29,learning_rate=.0005,actuator=True,**shared)
        checkpoint=dict(state_dict=model.state_dict(),experiment=c,protocol=protocol,tail_mean=tail.tail_mean,tail_std=tail.tail_std)
        torch.save(checkpoint,folder/'model.pt');ha.to_csv(folder/'actuator_history.csv',index=False)
        row=dict(experiment=name,steps=c['steps'],base_epochs=40,actuator_epochs=25,wall_time_s=time.time()-started,
            base_windows_per_s=float(hb.iloc[-1].windows_per_second),actuator_windows_per_s=float(ha.iloc[-1].windows_per_second),
            peak_gpu_bytes=max(int(hb.gpu_peak_allocated_bytes.max()),int(ha.gpu_peak_allocated_bytes.max())),
            base_final_loss=float(hb.iloc[-1].loss),actuator_final_loss=float(ha.iloc[-1].loss),checkpoint_sha256=sha(folder/'model.pt'),
            base_optimizer_steps=int(hb.iloc[-1].optimizer_steps),actuator_optimizer_steps=int(ha.iloc[-1].optimizer_steps))
        if name=='A0_baseline_retrain':
            row['legacy_checkpoint_max_parameter_difference']=max(float((v-frozen['state_dict'][k]).abs().max()) for k,v in model.state_dict().items() if v.is_floating_point() and torch.isfinite(v).all())
        allsummary.append(row);pd.DataFrame(allsummary).to_csv(results/'training_summary.csv',index=False)
        print('DONE',row,flush=True)
    (args.output/'status.json').write_text(json.dumps(dict(status='training_completed',experiments=[c['name'] for c in configs]),indent=2))

if __name__=='__main__':main()
