#!/usr/bin/env python3
"""Describe frozen free-run body-axis error; signed integral is not SO(3) error."""
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from run_september_history_lengths import matched_inputs
from run_september_step2 import write_json
from system_identification.data.september_trajectory import file_hash
from system_identification.training.trajectory_main_v1 import assemble_history_trajectory_windows
SOURCE=ROOT/'artifacts/september_oracle_diagnostics_20260911'
OUTPUT=ROOT/'artifacts/september_body_rate_analysis_20260911'

def run():
    manifest,entry,samples,windows,_=matched_inputs();rows=[];hashes={}
    for part in ['train','validation']:
        path=SOURCE/f'{part}_free_run_predictions.npz';hashes[str(path)]=file_hash(path)
        w=windows[part]
        if not w.equals(pd.read_parquet(SOURCE/f'{part}_windows.parquet')):raise ValueError('window mismatch')
        b=assemble_history_trajectory_windows(samples[part],w,history_steps=26)
        pred=np.load(path);np.testing.assert_array_equal(pred['window_ids'],b.trajectory.window_ids.astype(str))
        error=pred['angular_velocity_b']-b.trajectory.truth.angular_velocity_b
        integral=np.cumsum(.5*(error[:,1:]+error[:,:-1])*b.trajectory.dt_s[:,:,None],axis=1)
        for k in range(1,error.shape[1]):
            for log in np.unique(b.trajectory.log_ids):
                m=b.trajectory.log_ids==log
                for j,axis in enumerate(['body_x','body_y','body_z']):
                    e=error[m,k,j];i=integral[m,k-1,j]
                    rows.append(dict(partition=part,log_id=log,axis=axis,horizon_s=k/50,n_windows=int(m.sum()),
                        rate_rmse_rad_s=float(np.sqrt(np.mean(e*e))),signed_rate_bias_rad_s=float(e.mean()),
                        integral_rmse_rad=float(np.sqrt(np.mean(i*i))),signed_integral_bias_rad=float(i.mean())))
    d=pd.DataFrame(rows);d.to_csv(OUTPUT/'per_log_axis_curves.csv',index=False)
    cols=['rate_rmse_rad_s','signed_rate_bias_rad_s','integral_rmse_rad','signed_integral_bias_rad']
    macro=d.groupby(['partition','axis','horizon_s'])[cols].mean().reset_index()
    macro.to_csv(OUTPUT/'equal_log_axis_curves.csv',index=False)
    macro[macro.horizon_s.isin([1,2,3,5])].to_csv(OUTPUT/'axis_endpoints.csv',index=False)
    write_json(OUTPUT/'manifest.json',{'dataset':entry,'dataset_id':manifest['dataset_id'],'source_dataset_manifest':manifest,
        'prediction_hashes':hashes,'script_hash':file_hash(Path(__file__)),'sealed_test_opened':False,
        'interpretation':'body-axis signed rate integral describes persistence; not a net attitude angle or independent causal attribution'})
    write_json(OUTPUT/'summary.json',{'status':'completed','trained':False,'sealed_test_opened':False})
    print(macro[macro.horizon_s.isin([2,5])].round(5).to_string(index=False))

if __name__=='__main__':
    OUTPUT.mkdir(parents=True,exist_ok=False);run()
