#!/usr/bin/env python3
"""Read-only train loss component analysis for all frozen Step 8 models."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd,torch
from train_main_v2_increment import inputs
from run_main_v2_free_running import load_simulator
from system_identification.training.trajectory_main_v1 import _model_call
from system_identification.training.main_v2_objectives import loss_components
from system_identification.training.rollout_consistency import GROUPS


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--artifacts',type=Path,required=True);ap.add_argument('--device',default='cuda:1');a=ap.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    batch,_,_,_=inputs();protocol=json.loads((a.results/'protocol.json').read_text());rows=[]
    for experiment in protocol['experiments']:
        name=experiment['name'];sim=load_simulator(ROOT/'artifacts/trajectory_main_v2/models/main_v2_drive_tail_gated.pt',a.device)
        sim.model.load_state_dict(torch.load(a.artifacts/name/'model.pt',map_location='cpu',weights_only=False)['state_dict'])
        totals={h:{} for h in [5,10,20,50]};count=len(batch.trajectory.window_ids)
        with torch.no_grad():
            for start in range(0,count,256):
                ids=np.arange(start,min(start+256,count))
                p,t=_model_call(sim.model,batch,ids,use_history=True,rollout_steps=50,device=torch.device(a.device))
                for h in totals:
                    values=loss_components(p,t,h)
                    for key,value in values.items():totals[h][key]=totals[h].get(key,0.)+len(ids)*float(value)
        for h,values in totals.items():
            components={key:value/count for key,value in values.items()}
            row=dict(experiment=name,partition='train',n_windows=count,horizon_steps=h,**components)
            for key,channels in GROUPS.items():row['group_'+key]=sum(components[k] for k in channels)
            row['legacy_state_sum']=sum(components.values())
            row['candidate_added_weighted_loss']=experiment['lambda_roll']*row['group_priority'] if h==experiment['added_horizon'] else 0.
            rows.append(row)
    pd.DataFrame(rows).to_csv(a.results/'loss_component_analysis.csv',index=False)

if __name__=='__main__':main()
