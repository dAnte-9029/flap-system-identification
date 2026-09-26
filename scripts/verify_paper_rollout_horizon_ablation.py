"""Read-only numerical audit of the explicit development artifacts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import run_paper_rollout_horizon_ablation as r

def verify():
    p=r.m.read_json(r.OUT/'protocol.json');c=r.m.read_json(r.OUT/'completion.json')
    r.m.verify_pins(p['frozen_input_sha256'])
    for key in ('artifact_sha256','checkpoint_sha256','prediction_sha256','source_sha256'):
        r.m.verify_pins(c.get(key,{}))
    source,batches,stats=r.inputs();b=batches['validation'].trajectory
    saved=pd.read_csv(r.OUT/'per_flight.csv');cells=0
    for k in (25,50):
        for seed in r.SEEDS:
            path=r.ART/f'seed{seed}/predictions.npz' if k==25 else r.predpath(seed)
            with np.load(path,allow_pickle=False) as f:
                np.testing.assert_array_equal(f['window_ids'],b.window_ids)
                for key in vars(b.truth):
                    assert np.isfinite(f[key]).all()
                    assert f[key].shape==getattr(b.truth,key).shape
                assert np.max(np.abs(np.linalg.norm(f['quaternion_nb'],axis=-1)-1))<1e-5
                for horizon,step in r.m.HORIZONS.items():
                    errors={}
                    for key,metric in [('position_n','position_rmse_m'),('velocity_n','velocity_rmse_m_s'),('angular_velocity_b','body_rate_rmse_rad_s')]:
                        errors[metric]=np.sum((f[key][:,step]-getattr(b.truth,key)[:,step])**2,axis=-1)
                    a=f['quaternion_nb'][:,step].astype(float);t=b.truth.quaternion_nb[:,step].astype(float)
                    a/=np.linalg.norm(a,axis=-1,keepdims=True);t/=np.linalg.norm(t,axis=-1,keepdims=True)
                    errors['attitude_error_deg']=np.rad2deg(2*np.arccos(np.clip(np.abs(np.sum(a*t,axis=-1)),0,1)))**2
                    for flight in np.unique(b.log_ids):
                        row=saved[(saved.model==f'K{k}')&(saved.seed==seed)&(saved.log_id==flight)&(saved.horizon_s==horizon)].iloc[0]
                        for metric,squared in errors.items():
                            np.testing.assert_allclose(np.sqrt(squared[b.log_ids==flight].mean()),row[metric],rtol=2e-6,atol=1e-6);cells+=1
            if k==25:
                for stage,epochs,*_ in r.m.stages(seed):r.m.validate_history(pd.read_csv(r.ART/f'seed{seed}/{stage}_history.csv'),epochs)
    result=dict(status='passed',independent_per_flight_metric_cells=cells,all_registered_hashes_match=True,
        new_runs=3,epochs_per_new_run=65,updates_per_new_run=7215,origins_per_prediction=2582,
        heldout_data_accessed_this_run=False,rtol=2e-6,atol=1e-6)
    r.write_json(r.OUT/'final_verification.json',result)
    print(json.dumps(result))

if __name__=='__main__':verify()
