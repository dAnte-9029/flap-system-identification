#!/usr/bin/env python3
"""Prepare train-only scaling or run one fixed-budget S0/S1/S2 experiment."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json,argparse,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd,torch
from run_main_v2_free_running import sha,load_simulator
from system_identification.models.trajectory_main_v1 import CausalHistoryTrajectoryModel
from system_identification.models.trajectory_main_v2 import ActuatorAwareTrajectoryModel
from system_identification.training.trajectory_main_v1 import assemble_history_trajectory_windows,fit_main_v1_stats,_model_call
from system_identification.training.trajectory_main_v2 import fit_main_v2_stats
from system_identification.training.main_v2_objectives import loss_components
from system_identification.training.main_v2_increment import increment_terms,train_increment_stage,gradient_probe
from system_identification.evaluation.main_v2_free_running import describe


def inputs():
    dataset=ROOT/'dataset/trajectory_v1_august_f5_c4'
    samples=pd.read_parquet(dataset/'samples_train.parquet');windows=pd.read_parquet(dataset/'windows_train.parquet')
    batch=assemble_history_trajectory_windows(samples,windows,history_steps=26)
    stats=fit_main_v1_stats(samples,batch);tail=fit_main_v2_stats(batch)
    checkpoint=torch.load(ROOT/'artifacts/trajectory_main_v2/models/main_v2_drive_tail_gated.pt',map_location='cpu',weights_only=False)
    for name,v in vars(stats).items():np.testing.assert_array_equal(np.float32(v),checkpoint['state_dict']['base_model.'+name].numpy())
    np.testing.assert_array_equal(tail.tail_mean,checkpoint['tail_mean']);np.testing.assert_array_equal(tail.tail_std,checkpoint['tail_std'])
    return batch,stats,tail,checkpoint


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--results',type=Path,required=True);pa.add_argument('--artifacts',type=Path,required=True);pa.add_argument('--device',default='cuda:1');pa.add_argument('--prepare',action='store_true');pa.add_argument('--experiment',choices=['S0','S1','S2']);args=pa.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith('cuda'):raise RuntimeError('GPU required')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    r=args.results;batch,stats,tail,old=inputs()
    if args.prepare:
        if r.exists() and any(r.iterdir()):raise FileExistsError(r)
        r.mkdir(parents=True)
        roots=[ROOT/'docs/analysis/results'/n for n in ['main_v2_free_running_5s','main_v2_training_objective_ablation','main_v2_dynamics_observability_phase']]
        hashes={str(p.relative_to(ROOT)):sha(p) for root in roots for p in root.rglob('*') if p.is_file()}
        inventory=[]
        for root in roots[:2]:
            for p in root.rglob('*.csv'):
                f=pd.read_csv(p);inventory.append(dict(path=str(p.relative_to(ROOT)),rows=len(f),columns=len(f.columns)))
        pd.DataFrame(inventory).to_csv(r/'historical_results_inventory.csv',index=False)
        for root in [ROOT/'docs/audits']:
            for p in root.glob('*main_v2*.md'):hashes[str(p.relative_to(ROOT))]=sha(p)
        frozen=json.loads((roots[0]/'summary.json').read_text())['source_hashes']
        for p,h in frozen.items():assert sha(ROOT/p)==h,p
        scales=[];statistics=[]
        for key in ['velocity_n','angular_velocity_b']:
            x=getattr(batch.trajectory.truth,key);d=x[:,2]-x[:,0];rms=float(np.sqrt(np.mean(np.sum(d*d,axis=-1))))
            scales.append(rms);statistics.append(dict(signal=key,train_rms_vector_increment=rms,axis_rms=np.sqrt(np.mean(d*d,axis=0)).tolist()))
        model=load_simulator(ROOT/'artifacts/trajectory_main_v2/models/main_v2_drive_tail_gated.pt',args.device).model
        counts=[];terms=[];components=[]
        with torch.inference_mode():
            for start in range(0,len(batch.trajectory.window_ids),256):
                ix=np.arange(start,min(start+256,len(batch.trajectory.window_ids)))
                p,t=_model_call(model,batch,ix,use_history=True,rollout_steps=50,device=torch.device(args.device))
                terms.append([float(v) for v in increment_terms(p,t,scales)]);counts.append(len(ix));components.append({k:float(v) for k,v in loss_components(p,t,50).items()})
        errors=np.average(terms,axis=0,weights=counts);means={k:float(np.average([d[k] for d in components],weights=counts)) for k in components[0]}
        weights=[means['velocity']/errors[0],means['body_rate']/errors[1]]
        for j,row in enumerate(statistics):row.update(baseline_normalized_error=float(errors[j]),baseline_vector_increment_rmse=float(np.sqrt(errors[j])*scales[j]),lambda_inc=float(weights[j]))
        pd.DataFrame(statistics).to_csv(r/'train_increment_statistics.csv',index=False)
        # Small foreground gradient preflight, no optimizer changes.
        torch.manual_seed(17);m=CausalHistoryTrajectoryModel(hidden_size=64,use_controls=False,**vars(stats))
        pre=gradient_probe(m,batch,np.arange(64),scales=scales,weights=weights,device=args.device)
        protocol=dict(experiments=[dict(name='S0',weights=[0.,0.]),dict(name='S1',weights=weights)],scales=scales,
            normalization='train RMS vector of x[t+2]-x[t], one scale per vector; axes equally weighted',
            lambda_rule='match each normalized increment contribution to its corresponding baseline 50-step state contribution on the original 4214 train windows',
            baseline_components=means,initial_gradient_probe=pre,steps=50,history=26,seeds=dict(base=17,actuator=29),epochs=dict(base=40,actuator=25),batch_size=256,
            optimizer='AdamW',learning_rates=dict(base=.0003,actuator=.0005),weight_decay=1e-5,gradient_clip_norm=5.,checkpoint_criterion='fixed last epoch',
            increment_contract='only true-origin first two own recurrent transitions per original training window; no teacher forcing after t0; no auxiliary head; no jump-step inference',
            raw_derivative_loss_present=False,state_loss_weights_changed=False,
            increment_duration_s=describe(batch.trajectory.dt_s[:,:2].sum(1)),training_duration_s=describe(batch.trajectory.dt_s[:,:50].sum(1)),
            s2_rule='only if S1 final train incremental loss improvement <5% AND median weighted increment/original gradient norm <0.1; S2 uses exactly 4x S1 weights with all original terms retained',
            s3_rule='optional; omitted from this bounded primary test even if S1 improves; no automatic horizon expansion',
            frozen_validation_windows_sha256=sha(roots[0]/'windows.csv'),baseline_hashes=frozen,historical_hashes=hashes,
            architecture_changed=False,phase_changed=False,simulator_changed=False,actuator_time_constants_changed=False,sealed_test_opened=False)
        (r/'protocol.json').write_text(json.dumps(protocol,indent=2));pd.DataFrame(protocol['experiments']).to_csv(r/'experiment_manifest.csv',index=False)
        (r/'pretraining').symlink_to((roots[1]/'pretraining').resolve(),target_is_directory=True)
        print(json.dumps(dict(scales=scales,weights=weights,preflight=pre),indent=2));return
    protocol=json.loads((r/'protocol.json').read_text())
    c=next(c for c in protocol['experiments'] if c['name']==args.experiment)
    folder=args.artifacts/c['name']
    if folder.exists():raise FileExistsError(folder)
    folder.mkdir(parents=True);started=time.monotonic();stage='base'
    def callback(model,h):
        (folder/'status.json').write_text(json.dumps(dict(stage=stage,status='running',**h.iloc[-1].to_dict()),indent=2))
        if len(h)%5==0:print(stage,h.iloc[-1].to_dict(),flush=True)
    common=dict(scales=protocol['scales'],weights=c['weights'],device=args.device,callback=callback)
    torch.manual_seed(17);torch.cuda.manual_seed_all(17)
    model=CausalHistoryTrajectoryModel(hidden_size=64,use_controls=False,**vars(stats))
    model,hb=train_increment_stage(model,batch,epochs=40,seed=17,learning_rate=.0003,**common)
    torch.save(dict(state_dict=model.state_dict()),folder/'base.pt');hb.to_csv(folder/'base_history.csv',index=False)
    stage='actuator';torch.manual_seed(29);torch.cuda.manual_seed_all(29)
    model=ActuatorAwareTrajectoryModel(base_model=model,use_drive=True,use_tail=True,gated_tail=True,tail_mean=tail.tail_mean,tail_std=tail.tail_std)
    model,ha=train_increment_stage(model,batch,epochs=25,seed=29,learning_rate=.0005,actuator=True,**common)
    torch.save(dict(state_dict=model.state_dict(),protocol=protocol,experiment=c,tail_mean=tail.tail_mean,tail_std=tail.tail_std),folder/'model.pt');ha.to_csv(folder/'actuator_history.csv',index=False)
    probes=[]
    for start in np.linspace(0,len(batch.trajectory.window_ids)-128,4,dtype=int):
        probes.append(gradient_probe(model,batch,np.arange(start,start+128),scales=protocol['scales'],weights=protocol['experiments'][1]['weights'],device=args.device,actuator=True))
    pd.DataFrame(probes).to_csv(folder/'final_gradient_probes.csv',index=False)
    summary=dict(experiment=c['name'],base_epochs=40,actuator_epochs=25,base_optimizer_steps=int(hb.iloc[-1].optimizer_steps),actuator_optimizer_steps=int(ha.iloc[-1].optimizer_steps),wall_time_s=time.monotonic()-started,base_windows_per_s=hb.iloc[-1].windows_per_second,actuator_windows_per_s=ha.iloc[-1].windows_per_second,peak_gpu_bytes=int(max(hb.gpu_peak_allocated_bytes.max(),ha.gpu_peak_allocated_bytes.max())),checkpoint_sha256=sha(folder/'model.pt'),device=args.device)
    if c['name']=='S0':summary['old_checkpoint_max_parameter_difference']=max(float((v-old['state_dict'][k]).abs().max()) for k,v in model.state_dict().items() if v.is_floating_point() and torch.isfinite(v).all())
    (folder/'training_summary.json').write_text(json.dumps(summary,indent=2));(folder/'status.json').write_text(json.dumps(dict(status='training_completed',**summary),indent=2));print(summary,flush=True)

if __name__=='__main__':main()
