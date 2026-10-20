"""Independent NumPy/SciPy metric recomputation and archived result consistency."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'))
from system_identification.models.velocity_rate_gru import VelocityRateGRU
from system_identification.data.velocity_rate_dataset import file_hash,write_json
from prepare_velocity_rate_learning import EXP,CACHE
from train_velocity_rate_learning import batch,INPUT_KEYS
METRICS=['position_m','velocity_m_s','attitude_deg','rate_rad_s']


def independent_errors(pred,target):
    shape=pred.shape[:-1]
    r1=Rotation.from_quat(pred[...,6:10].reshape(-1,4),scalar_first=True)
    r2=Rotation.from_quat(target[...,6:10].reshape(-1,4),scalar_first=True)
    angles=(r1.inv()*r2).magnitude().reshape(shape)*180/np.pi
    return np.stack((np.linalg.norm(pred[...,:3]-target[...,:3],axis=-1),
        np.linalg.norm(pred[...,3:6]-target[...,3:6],axis=-1),angles,
        np.linalg.norm(pred[...,10:13]-target[...,10:13],axis=-1)),-1)


def main():
    torch.set_num_threads(4); torch.backends.cudnn.deterministic=True; torch.backends.cuda.matmul.allow_tf32=True
    config=json.loads((EXP/'config.json').read_text()); audit=json.loads((EXP/'data_audit.json').read_text())
    root=ROOT/audit['registry']['repository_relative_root']
    flights=pd.read_csv(root/'flights.csv')
    assignments=pd.DataFrame(json.loads((root/'split_manifest.json').read_text())['assignments'])
    assert (assignments.groupby('sha256').split.nunique()==1).all()
    assert (assignments.groupby('flight_date').split.nunique()==1).all()
    for split in ('train','validation'):
        windows=pd.read_parquet(EXP/f'{split}_windows.parquet')
        known=flights.set_index('flight_id').loc[windows.flight_id.unique()]
        assert (known.split==split).all() and (known.sampling_group=='main_50hz').all()
        mapped=assignments.set_index('flight_id').loc[known.index]
        assert (mapped.split==split).all()
    for split in ('train','validation'):
        assert file_hash(CACHE/f'{split}.npz')==audit['partitions'][split]['cache_sha256']
    packed=np.load(CACHE/'validation.npz'); meta=pd.read_parquet(EXP/'validation_windows.parquet')
    target=packed['target'].astype(float); initial=packed['initial'].astype(float)
    time=np.cumsum(packed['dt'].astype(float),axis=1); n,k=time.shape
    # Baseline in float64 using independent closed-form rotation composition, no learned code.
    pred=np.repeat(initial[:,None,:],k,axis=1)
    pred[...,:3]=initial[:,None,:3]+initial[:,None,3:6]*time[...,None]
    ri=Rotation.from_quat(np.repeat(initial[:,None,6:10],k,axis=1).reshape(-1,4),scalar_first=True)
    dr=Rotation.from_rotvec((initial[:,None,10:13]*time[...,None]).reshape(-1,3))
    pred[...,6:10]=(ri*dr).as_quat(scalar_first=True).reshape(n,k,4)
    err=independent_errors(pred,target); mask=packed['future_mask']; i,j=np.where(mask)
    stored=pd.read_parquet(EXP/'metrics/Kinematic_Baseline_seed_-1_native_steps.parquet')
    np.testing.assert_array_equal(stored.window_id.to_numpy(),meta.window_id.to_numpy()[i])
    delta=np.max(abs(stored[METRICS].to_numpy()-err[i,j]),axis=0)
    assert np.all(delta<3e-4), delta
    # Repeat inference from every archived checkpoint for a fixed independent cross-flight sample.
    selected=np.unique(np.r_[np.linspace(0,n-1,128,dtype=int),np.flatnonzero(mask.sum(1)==11)])
    full_inputs={key:torch.from_numpy(packed[key]).cuda() for key in INPUT_KEYS}
    sample={key:value[selected] for key,value in full_inputs.items()}
    max_checkpoint_difference=0.
    batch_shape_difference=np.zeros(4)
    for model in ('GRU-NoControl','GRU-Control'):
        for seed in config['seeds']:
            folder=EXP/model/f'seed_{seed}'
            summary=json.loads((folder/'summary.json').read_text())
            assert file_hash(folder/'best.pt')==summary['checkpoint_sha256']
            c=torch.load(folder/'best.pt',map_location='cuda',weights_only=False)
            m=VelocityRateGRU(c['normalization'],control=model=='GRU-Control').cuda().eval(); m.load_state_dict(c['model'])
            with torch.no_grad():
                small=m(sample).cpu().numpy().astype(float)
                all_pred=torch.cat([m(batch(full_inputs,ix)) for ix in
                    torch.arange(n,device='cuda').split(config['evaluation_batch_size'])])
                p=all_pred[selected].cpu().numpy().astype(float)
            e=independent_errors(p,target[selected])
            batch_shape_difference=np.maximum(batch_shape_difference,
                np.max(abs(e-independent_errors(small,target[selected])),axis=(0,1)))
            f=pd.read_parquet(EXP/f'metrics/{model}_seed_{seed}_endpoints.parquet')
            for h in (100,200):
                rows=f[f.horizon_ms==h].set_index('window_id').loc[meta.window_id.iloc[selected]]
                ix=rows.native_step.to_numpy()-1
                difference=np.max(abs(rows[METRICS].to_numpy()-e[np.arange(len(selected)),ix]))
                max_checkpoint_difference=max(max_checkpoint_difference,float(difference))
                assert difference<5e-4, (model,seed,h,difference)
    # Recompute flight aggregation from archived per-window endpoints, independently of summary routine.
    reported=pd.read_csv(EXP/'summary.csv'); flight_delta=0.; window_delta=0.
    for model in ('Kinematic Baseline','GRU-NoControl','GRU-Control'):
        frames=[]
        for seed in ([-1] if model=='Kinematic Baseline' else config['seeds']):
            f=pd.read_parquet(EXP/f'metrics/{model.replace(" ","_")}_seed_{seed}_endpoints.parquet'); frames.append(f)
        frames=pd.concat(frames,ignore_index=True)
        for h in (100,200):
            f=frames[frames.horizon_ms==h]
            fm=f.groupby(['seed','flight_id'])[METRICS].mean().groupby('flight_id').mean().mean()
            wm=f.groupby('window_id')[METRICS].mean().mean()
            s=reported[(reported.model==model)&(reported.horizon_ms==h)].set_index('metric')
            flight_delta=max(flight_delta,float(np.max(abs(fm-s.loc[METRICS,'flight_mean']))))
            window_delta=max(window_delta,float(np.max(abs(wm-s.loc[METRICS,'window_mean']))))
    assert flight_delta<1e-5 and window_delta<1e-5
    selections=pd.read_csv(EXP/'recomputed_selection.csv'); selection_delta=0.
    for row in selections.itertuples():
        if row.model=='Kinematic Baseline': continue
        s=json.loads((EXP/row.model/f'seed_{row.seed}/summary.json').read_text())
        selection_delta=max(selection_delta,abs(row.validation_flight_loss-s['best_validation_flight_loss']))
    assert selection_delta<1e-5
    result=dict(status='passed',test_partition_opened=False,
        baseline_full_validation_scipy_max_error_by_metric=delta.tolist(),
        checkpoint_replay_windows_per_model=len(selected),checkpoint_scipy_metric_max_difference=max_checkpoint_difference,
        flight_aggregation_max_difference=flight_delta,window_aggregation_max_difference=window_delta,
        checkpoint_selection_max_difference=selection_delta,caches_and_checkpoints_hash_verified=True)
    result['tf32_different_batch_shape_metric_sensitivity']=batch_shape_difference.tolist()
    result['whole_log_date_split_metadata_verified']=True
    write_json(EXP/'verification.json',result); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
