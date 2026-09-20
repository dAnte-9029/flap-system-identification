"""Separate instantaneous raw-command path from settled actuator residual sensitivity."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from evaluate_expanded_control_response import load
from run_main_v2_free_running import make_initial
from run_isaac_surrogate_straight import sha
from system_identification.training.trajectory_main_v1 import assemble_history_trajectory_windows
from system_identification.models.trajectory_main_v1 import _state_features

def main():
    torch.set_num_threads(2);out=ROOT/'docs/analysis/results/actuator_only_joint_v1';manifest=json.loads((out/'response_manifest.json').read_text());protocol=manifest['dataset'];root=ROOT/protocol['dataset_root']
    assert sha(root/'manifest.json')==protocol['manifest_sha256']
    for n in ['samples_validation.parquet','windows_validation.parquet']:assert sha(root/n)==protocol['sample_artifact_sha256'][n]
    s=pd.read_parquet(root/'samples_validation.parquet');w=pd.read_parquet(root/'windows_validation.parquet');b=assemble_history_trajectory_windows(s,w,history_steps=26)
    indices=[]
    for log,g in w.groupby('log_id',sort=True):indices.extend(g.index[np.linspace(0,len(g)-1,min(16,len(g)),dtype=int)].tolist())
    directions=torch.tensor([[0,1,1,0],[0,1,-1,0],[0,0,0,1]],dtype=torch.float32);rows=[]
    with torch.inference_mode():
      for label,cp,key in [('baseline',ROOT/'artifacts/september_expanded_main_v2/model.pt','baseline_checkpoint_sha256'),('joint',ROOT/'artifacts/actuator_only_joint_v1/model.pt','checkpoint_sha256')]:
        assert sha(cp)==manifest[key];sim=load(cp)
        for start in range(0,len(indices),64):
            ix=np.array(indices[start:start+64]);state=make_initial(sim,b,ix,'warm26','cpu');u=torch.tensor(b.trajectory.controls[ix,0],dtype=torch.float32);dt=torch.tensor(b.trajectory.dt_s[ix,0],dtype=torch.float32)
            f=_state_features(state.velocity_n,state.quaternion_nb,state.angular_velocity_b,state.relative_phase_rad,state.phase_anchor,state.flap_frequency_hz);nf=sim.model.base_model._normalize_features(f)
            for j,(channel,axis,sign) in enumerate([('common',1,1),('differential',0,-1),('rudder',2,1)]):
                plus=u+.001*directions[j];minus=u-.001*directions[j];valid=(plus[:,1:].abs()<=1).all(1)&(minus[:,1:].abs()<=1).all(1)
                plus=torch.where(valid[:,None],plus,u);minus=torch.where(valid[:,None],minus,u)
                _,dp=sim.step(state,plus,dt);_,dm=sim.step(state,minus,dt)
                instantaneous=(dp.angular_acceleration_b[:,axis]-dm.angular_acceleration_b[:,axis])/.002
                settled=sim.model.tail_effectiveness(j,nf)*sim.model.tail_output_mask[j]*sim.model.tail_gate_values()[j]*sim.model.base_model.derivative_std/sim.model.tail_std[j]
                for n,i in enumerate(ix):
                    if valid[n]:rows.append(dict(model=label,log_id=w.iloc[i].log_id,window_id=w.iloc[i].window_id,channel=channel,expected_sign=sign,instantaneous_accel_slope=float(instantaneous[n]),settled_explicit_tail_slope=float(settled[n,3+axis])))
    r=pd.DataFrame(rows);r.to_csv(out/'direct_path_sensitivity.csv',index=False)
    summary=r.groupby(['model','channel']).agg(n=('instantaneous_accel_slope','size'),direct_median=('instantaneous_accel_slope','median'),tail_median=('settled_explicit_tail_slope','median'))
    summary.to_csv(out/'direct_path_summary.csv');print(summary.to_string())
    (out/'direct_path_manifest.json').write_text(json.dumps(dict(dataset=protocol,script_sha256=sha(__file__),response_manifest_sha256=sha(out/'response_manifest.json'),probe='same complete initial state, +/-0.001 input; first-step angular acceleration difference isolates direct raw-command path; tail slope is analytic settled-proxy branch only, not full steady-state sensitivity',test_opened=False),indent=2))
if __name__=='__main__':main()
