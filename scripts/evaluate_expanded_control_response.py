"""Frozen final model tail ablation and local response on explicit validation."""
from pathlib import Path
import sys,json
from types import SimpleNamespace
import numpy as np,pandas as pd,torch,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_main_v2_free_running import sha,make_initial,rollout
from system_identification.models.trajectory_main_v1 import CausalHistoryTrajectoryModel
from system_identification.models.trajectory_main_v2 import ActuatorAwareTrajectoryModel
from system_identification.models.main_v2_simulator import MainV2Simulator
from system_identification.training.trajectory_main_v1 import assemble_history_trajectory_windows
from system_identification.evaluation.main_v2_free_running import endpoint_errors


def load(path):
    cp=torch.load(path,map_location='cpu',weights_only=False);s=cp['state_dict']
    stats={n:s['base_model.'+n].numpy() for n in ['feature_mean','feature_std','control_mean','control_std','derivative_mean','derivative_std']}
    base=CausalHistoryTrajectoryModel(hidden_size=cp['base_config']['hidden_size'],use_controls=cp['base_config'].get('use_controls',False),**stats)
    model=ActuatorAwareTrajectoryModel(base_model=base,tail_mean=cp['tail_mean'],tail_std=cp['tail_std'],**cp['config'])
    model.load_state_dict(s,strict=True)
    return MainV2Simulator(model.eval())


def main():
    root=ROOT/'dataset/trajectory_v3_september_expanded';cp=ROOT/'artifacts/september_expanded_main_v2/model.pt'
    out=ROOT/'docs/analysis/results/september_expanded_main_v2/control_response';out.mkdir(parents=True,exist_ok=False)
    completed=json.loads((out.parent/'manifest.json').read_text());assert sha(cp)==completed['checkpoint_sha256']
    manifest=json.loads((root/'manifest.json').read_text());assert sha(root/'manifest.json')==completed['protocol']['manifest_sha256']
    for name in ['samples_validation.parquet','windows_validation.parquet']:assert sha(root/name)==manifest['artifact_sha256'][name]
    s=pd.read_parquet(root/'samples_validation.parquet');w=pd.read_parquet(root/'windows_validation.parquet');b=assemble_history_trajectory_windows(s,w,history_steps=26)
    torch.set_num_threads(2);sim=load(cp);errors=[];responses=[]
    with torch.inference_mode():
        for start in range(0,len(w),128):
            ix=np.arange(start,min(start+128,len(w)));initial=make_initial(sim,b,ix,'warm26','cpu');u=torch.tensor(b.trajectory.controls[ix],dtype=torch.float32);dt=torch.tensor(b.trajectory.dt_s[ix],dtype=torch.float32)
            truth=SimpleNamespace(**{n:getattr(b.trajectory.truth,n)[ix] for n in ['position_n','velocity_n','quaternion_nb','angular_velocity_b','relative_phase_rad','flap_frequency_hz']})
            for variant in ['logged','disable_tail']:
                sim.model.use_tail=variant=='logged';pred,_=rollout(sim,initial,u,dt)
                for k in [25,50]:
                    m=endpoint_errors(pred,truth,k)
                    for j,i in enumerate(ix):errors.append(dict(window_id=w.iloc[i].window_id,log_id=w.iloc[i].log_id,cohort='original_sep7' if w.iloc[i].log_id.startswith('2026.9.7/') else 'new_sep17',variant=variant,horizon_s=k*.02,**{n:float(m[n][j]) for n in ['position_m','attitude_deg','body_rate_rad_s','body_rate_q_rad_s']}))
            sim.model.use_tail=True
            for name,channels,axis in [('common',[1,2],1),('rudder',[3],2)]:
                valid=(u[:,:,channels].abs()<=.975).all(2).all(1).numpy();plus=u.clone();minus=u.clone();plus[:,:,channels]+=.025;minus[:,:,channels]-=.025
                pp,_=rollout(sim,initial,plus,dt);pm,_=rollout(sim,initial,minus,dt)
                for k in [5,25,50]:
                    value=(pp['angular_velocity_b'][:,k,axis]-pm['angular_velocity_b'][:,k,axis])/.05
                    for j,i in enumerate(ix):
                        if valid[j]:responses.append(dict(window_id=w.iloc[i].window_id,log_id=w.iloc[i].log_id,cohort='original_sep7' if w.iloc[i].log_id.startswith('2026.9.7/') else 'new_sep17',channel=name,horizon_s=k*.02,sensitivity=float(value[j])))
    e=pd.DataFrame(errors);r=pd.DataFrame(responses);e.to_csv(out/'ablation_per_window.csv',index=False);r.to_csv(out/'response_per_window.csv',index=False)
    e.groupby(['cohort','variant','horizon_s'])[['position_m','attitude_deg','body_rate_rad_s','body_rate_q_rad_s']].agg(lambda x:float(np.sqrt(np.mean(x*x)))).to_csv(out/'ablation_summary.csv')
    r.groupby(['cohort','channel','horizon_s']).sensitivity.agg(count='size',median='median',positive_fraction=lambda x:float((x>0).mean())).to_csv(out/'response_summary.csv')
    (out/'manifest.json').write_text(json.dumps(dict(checkpoint_sha256=sha(cp),dataset=completed['protocol'],script_sha256=sha(__file__),test_opened=False,perturbation=.025),indent=2))
if __name__=='__main__':main()
