"""Frozen validation response comparison and identical-physical-state Isaac cases."""
from pathlib import Path
from types import SimpleNamespace
import sys,json
import numpy as np,pandas as pd,torch,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from evaluate_expanded_control_response import load
from run_main_v2_free_running import make_initial,rollout
from run_isaac_surrogate_straight import sha
from system_identification.training.trajectory_main_v1 import assemble_history_trajectory_windows


def main():
    torch.set_num_threads(4)
    result=ROOT/'docs/analysis/results/actuator_only_joint_v1';trained=json.loads((result/'manifest.json').read_text());cp=ROOT/'artifacts/actuator_only_joint_v1/model.pt';assert sha(cp)==trained['checkpoint_sha256']
    registry=yaml.safe_load((ROOT/'configs/data/trajectory_dataset_registry.yaml').read_text());did=registry['default_dataset_id'];entry=registry['datasets'][did];mp=ROOT/entry['manifest_path'];assert sha(mp)==entry['manifest_sha256']==trained['protocol']['manifest_sha256'];m=json.loads(mp.read_text());root=mp.parent
    for n in ['samples_validation.parquet','windows_validation.parquet']:assert sha(root/n)==m['artifact_sha256'][n]
    s=pd.read_parquet(root/'samples_validation.parquet');w=pd.read_parquet(root/'windows_validation.parquet');b=assemble_history_trajectory_windows(s,w,history_steps=26)
    oldroot=ROOT/'docs/analysis/results/isaac_straight_level_cases_v2';oldmanifest=json.loads((oldroot/'manifest.json').read_text());assert sha(oldroot/'cases.pt')==oldmanifest['cases_sha256']
    cases=torch.load(oldroot/'cases.pt',weights_only=False,map_location='cpu');newcases=[];sim=load(cp);lookup={v:i for i,v in enumerate(w.window_id)}
    with torch.inference_mode():
        for case in cases:
            i=lookup[case['window_id']];state=make_initial(sim,b,np.array([i]),'warm26','cpu')
            for key in ['position_n','velocity_n','quaternion_nb','angular_velocity_b','relative_phase_rad','flap_frequency_hz']:
                torch.testing.assert_close(getattr(state,key),case['state'][key],rtol=0,atol=1e-6)
            newcases.append(dict(log_id=case['log_id'],window_id=case['window_id'],state=state.snapshot(),command=case['command'].clone()))
    caseout=result/'isaac_cases';caseout.mkdir(exist_ok=False);torch.save(newcases,caseout/'cases.pt')
    contract=dict(oldmanifest,checkpoint=str(cp),checkpoint_sha256=sha(cp),cases_sha256=sha(caseout/'cases.pt'),selection='same 17 frozen physical initial states and history windows as flight-chain baseline; GRU re-encoded using joint model',baseline_cases_sha256=oldmanifest['cases_sha256'])
    (caseout/'manifest.json').write_text(json.dumps(contract,indent=2))
    oldcp=Path(oldmanifest['checkpoint']);assert sha(oldcp)==oldmanifest['checkpoint_sha256']
    selection=[]
    for log,g in w.groupby('log_id',sort=True):selection.extend(g.index[np.linspace(0,len(g)-1,min(16,len(g)),dtype=int)].tolist())
    ix=np.array(selection);rows=[];directions=np.array([[0,1,1,0],[0,1,-1,0],[0,0,0,1],[1,0,0,0]],dtype=np.float32)
    with torch.inference_mode():
      for label,model in [('expanded_baseline',load(oldcp)),('actuator_only_joint_v1',sim)]:
       for begin in range(0,len(ix),64):
        inds=ix[begin:begin+64];initial=make_initial(model,b,inds,'warm26','cpu');u=torch.tensor(b.trajectory.controls[inds],dtype=torch.float32);dt=torch.tensor(b.trajectory.dt_s[inds],dtype=torch.float32)
        for j,(channel,axis,sign) in enumerate([('common',1,1),('differential',0,-1),('rudder',2,1),('motor',0,1)]):
            plus=u+.02*torch.tensor(directions[j]);minus=u-.02*torch.tensor(directions[j]);valid=((plus[:,:,1:].abs()<=1).all((1,2))&(minus[:,:,1:].abs()<=1).all((1,2))&(plus[:,:,0]<=1).all(1)&(minus[:,:,0]>=0).all(1))
            # Invalid candidates are replaced with original inputs and excluded from results.
            plus=torch.where(valid[:,None,None],plus,u);minus=torch.where(valid[:,None,None],minus,u)
            pp,_=rollout(model,initial,plus,dt);pm,_=rollout(model,initial,minus,dt)
            for step in [1,2,3,5,10,15,25,40,50]:
                value=(pp['angular_velocity_b'][:,step,axis]-pm['angular_velocity_b'][:,step,axis])/.04 if j<3 else (pp['flap_frequency_hz'][:,step]-pm['flap_frequency_hz'][:,step])/.04
                for n,i in enumerate(inds):
                    rows.append(dict(model=label,log_id=w.iloc[i].log_id,window_id=w.iloc[i].window_id,channel=channel,horizon_s=step*.02,eligible=bool(valid[n]),response=float(value[n]) if valid[n] else None,expected_sign=sign))
        print(label,begin,'/',len(ix),flush=True)
    r=pd.DataFrame(rows);r['sign_correct']=r.response*r.expected_sign>0;r.to_csv(result/'response_per_window.csv',index=False)
    good=r[r.eligible];good.groupby(['model','channel','horizon_s']).agg(n=('response','size'),correct_fraction=('sign_correct','mean'),median=('response','median'),p10=('response',lambda x:x.quantile(.1)),p90=('response',lambda x:x.quantile(.9))).to_csv(result/'response_summary.csv')
    good.groupby(['log_id','model','channel','horizon_s']).agg(n=('response','size'),correct_fraction=('sign_correct','mean'),median=('response','median')).to_csv(result/'response_per_log.csv')
    (result/'response_manifest.json').write_text(json.dumps(dict(dataset=trained['protocol'],checkpoint_sha256=sha(cp),baseline_checkpoint_sha256=sha(oldcp),baseline_cases_sha256=oldmanifest['cases_sha256'],script_sha256=sha(__file__),windows=len(ix),selection='16 evenly spaced validation windows per log, independent of outcome',perturbation=.02,test_opened=False,meaning='model-only response; future logged inputs perturbed, not real causal ground truth'),indent=2))
if __name__=='__main__':main()
