"""Predeclared from-scratch joint-control comparison on the registered dataset."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import sys,json,time
import numpy as np,pandas as pd,torch,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_main_v2_free_running import sha,make_initial,rollout
from system_identification.models.trajectory_main_v1 import CausalHistoryTrajectoryModel
from system_identification.models.trajectory_main_v2 import ActuatorAwareTrajectoryModel
from system_identification.models.main_v2_simulator import MainV2Simulator
from system_identification.training.trajectory_main_v1 import assemble_history_trajectory_windows,fit_main_v1_stats
from system_identification.training.trajectory_main_v2 import fit_main_v2_stats
from system_identification.training.main_v2_increment import train_increment_stage
from system_identification.evaluation.main_v2_free_running import endpoint_errors

def main():
    reg=yaml.safe_load((ROOT/'configs/data/trajectory_dataset_registry.yaml').read_text());did=reg['default_dataset_id'];entry=reg['datasets'][did];mp=ROOT/entry['manifest_path']
    assert sha(mp)==entry['manifest_sha256'];manifest=json.loads(mp.read_text());root=mp.parent
    names=['samples_train.parquet','windows_train.parquet','samples_validation.parquet','windows_validation.parquet']
    hashes={n:sha(root/n) for n in names};assert all(hashes[n]==manifest['artifact_sha256'][n] for n in names)
    out=ROOT/'artifacts/actuator_only_joint_v1';result=ROOT/'docs/analysis/results/actuator_only_joint_v1';out.mkdir(exist_ok=False);result.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);device='cuda:1'
    if not torch.cuda.is_available() or torch.cuda.device_count()<2:raise RuntimeError('Configured cuda:1 unavailable')
    protocol=dict(dataset_id=did,manifest_sha256=sha(mp),dataset_root=str(root.relative_to(ROOT)),sample_artifact_sha256=hashes,partitions=['train','validation'],phase_contract=manifest['phase_contract'],frequency_contract=manifest['frequency_contract'],
        model='joint GRU64 with command history, actuator-only rollout and signed direct tail effectiveness, trained from scratch',seed=17,epochs=65,batch_size=256,learning_rate=.0003,history_steps=26,rollout_steps=50,
        checkpoint_criterion='last epoch fixed in advance, no validation/closed-loop selection',device=device,
        comparison='65 train passes matches baseline 40+25; joint optimization and control inputs change together, not a single-factor causal ablation',
        objective='same trajectory+increment terms; actuator frequency and residual regularization applied throughout joint training',
        control_contract='historical controls encode initial GRU; rollout backbone control slots fixed to training mean; future controls enter only drive/tail proxies; direct common/differential/rudder angular effectiveness signs +q/-p/+r; tau unchanged and uncalibrated',
        response_audit_manifest_sha256=sha(ROOT/'docs/analysis/results/control_response_lags_v1/manifest.json'),test_opened=False,script_sha256=sha(__file__),source_sha256={n:sha(ROOT/n) for n in ['src/system_identification/models/trajectory_main_v2.py','src/system_identification/models/main_v2_simulator.py','src/system_identification/training/main_v2_increment.py']})
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(result/'protocol.json').write_text(json.dumps(protocol,indent=2));started=time.monotonic()
    s=pd.read_parquet(root/'samples_train.parquet');w=pd.read_parquet(root/'windows_train.parquet');batch=assemble_history_trajectory_windows(s,w,history_steps=26)
    stats=fit_main_v1_stats(s,batch);tail=fit_main_v2_stats(batch)
    scales=[float(np.sqrt(np.mean(np.sum((getattr(batch.trajectory.truth,k)[:,2]-getattr(batch.trajectory.truth,k)[:,0])**2,axis=-1)))) for k in ['velocity_n','angular_velocity_b']]
    weights=[.16458798840801334,.36716770482593264];protocol.update(scales=scales,weights=weights,train_windows=len(w))
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(result/'protocol.json').write_text(json.dumps(protocol,indent=2))
    torch.manual_seed(17);torch.cuda.manual_seed_all(17)
    base=CausalHistoryTrajectoryModel(hidden_size=64,use_controls=True,**vars(stats))
    config=dict(use_drive=True,use_tail=True,gated_tail=True,drive_tau_s=.1,tail_tau_s=.04,initial_tail_gate=.05,joint_control_backbone=True,actuator_only_rollout=True,signed_tail_effectiveness=True)
    model=ActuatorAwareTrajectoryModel(base_model=base,tail_mean=tail.tail_mean,tail_std=tail.tail_std,**config)
    def save(model,path):torch.save(dict(state_dict=model.state_dict(),base_config=dict(hidden_size=64,use_controls=True),config=config,tail_mean=tail.tail_mean,tail_std=tail.tail_std,protocol=protocol),path)
    def callback(model,history):
        history.to_csv(out/'history.csv',index=False);r=history.iloc[-1].to_dict();(out/'status.json').write_text(json.dumps(r));print(json.dumps(r),flush=True)
        if int(r['epoch'])%10==0:save(model,out/'latest_recovery.pt')
    model,h=train_increment_stage(model,batch,scales=scales,weights=weights,device=device,epochs=65,seed=17,learning_rate=.0003,actuator=True,callback=callback)
    save(model,out/'model.pt');del batch,s,w
    v=pd.read_parquet(root/'samples_validation.parquet');vw=pd.read_parquet(root/'windows_validation.parquet');vb=assemble_history_trajectory_windows(v,vw,history_steps=26);sim=MainV2Simulator(model.eval());rows=[]
    from types import SimpleNamespace
    with torch.inference_mode():
        for start in range(0,len(vw),128):
            ix=np.arange(start,min(start+128,len(vw)));initial=make_initial(sim,vb,ix,'warm26','cpu');pred,_=rollout(sim,initial,torch.tensor(vb.trajectory.controls[ix],dtype=torch.float32),torch.tensor(vb.trajectory.dt_s[ix],dtype=torch.float32))
            truth=SimpleNamespace(**{k:getattr(vb.trajectory.truth,k)[ix] for k in ['position_n','velocity_n','quaternion_nb','angular_velocity_b','relative_phase_rad','flap_frequency_hz']})
            for k in [25,50]:
                metrics=endpoint_errors(pred,truth,k)
                for j,i in enumerate(ix):rows.append(dict(model='actuator_only_joint_v1',window_id=vw.iloc[i].window_id,log_id=vw.iloc[i].log_id,cohort='original_sep7' if vw.iloc[i].log_id.startswith('2026.9.7/') else 'new_sep17',horizon_s=k*.02,**{n:float(val[j]) for n,val in metrics.items()}))
    frame=pd.DataFrame(rows);frame.to_csv(result/'per_window.csv',index=False);cols=['position_m','velocity_m_s','attitude_deg','body_rate_rad_s','body_rate_q_rad_s']
    frame.groupby(['cohort','horizon_s'])[cols].agg(lambda x:float(np.sqrt(np.mean(x*x)))).to_csv(result/'summary.csv');frame.groupby(['log_id','horizon_s'])[cols].agg(lambda x:float(np.sqrt(np.mean(x*x)))).to_csv(result/'per_log.csv')
    final=dict(completed=True,wall_time_s=time.monotonic()-started,checkpoint_sha256=sha(out/'model.pt'),protocol=protocol,tail_gates=model.tail_gate_values().detach().tolist(),trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad))
    (result/'manifest.json').write_text(json.dumps(final,indent=2));(out/'status.json').write_text(json.dumps(final,indent=2));print('TRAIN_AND_VALIDATION_COMPLETE',flush=True)
if __name__=='__main__':main()
