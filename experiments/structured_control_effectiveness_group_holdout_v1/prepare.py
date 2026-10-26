"""Exact frozen folds; fit normalization on train-only sliced tensors; independent audit."""
from common import *
import torch,shutil,subprocess
from scipy.spatial.transform import Rotation

def independent_normalization(d,cfg):
    mean=[];scale=[]
    for c in [3,4,5,10,11,12,19,13,14,15,16]:
        v=d['history'][...,c][d['history_mask'][...,c]].astype(float)
        mean.append(v.mean());scale.append(max(v.std(ddof=1),.05))
    previous=np.concatenate((d['initial'][:,None],d['target'][:,:-1]),1)
    delta=(d['target']-previous)[...,[3,4,5,10,11,12]][d['future_mask']].astype(float)
    output=np.maximum(delta.std(axis=0,ddof=1),.03)
    total=np.zeros(4);count=0
    for start in range(0,len(d['initial']),1024):
        s=d['initial'][start:start+1024].astype(float);target=d['target'][start:start+1024].astype(float);mask=d['future_mask'][start:start+1024];elapsed=d['dt'][start:start+1024].astype(float).cumsum(1)
        position=s[:,None,:3]+s[:,None,3:6]*elapsed[...,None]
        # Closed form constant-rate baseline, independent of the recursive torch integrator.
        origin=np.broadcast_to(s[:,None,[7,8,9,6]],(*elapsed.shape,4)).reshape(-1,4)
        rot=Rotation.from_quat(origin)*Rotation.from_rotvec((s[:,None,10:13]*elapsed[...,None]).reshape(-1,3))
        angle=(rot.inv()*Rotation.from_quat(target[...,[7,8,9,6]].reshape(-1,4))).magnitude().reshape(elapsed.shape)
        squared=np.stack((((position-target[...,:3])**2).sum(-1),((s[:,None,3:6]-target[...,3:6])**2).sum(-1),angle**2,((s[:,None,10:13]-target[...,10:13])**2).sum(-1)),-1)
        total+=(squared*mask[...,None]).sum((0,1));count+=mask.sum()
    return dict(mean=mean,scale=scale,output_scale=output,loss_scales=np.maximum(np.sqrt(total/count),cfg['loss_scale_floors']))

def main():
    cfg=json.loads((HERE/'config.json').read_text());records={}
    for entry in json.loads((PILOT/'archive_manifest.json').read_text())['files']:
        p=PILOT/entry['path'];assert sha(p)==entry['sha256'];records[str(p)]=entry['sha256']
    audit=json.loads((TRAIN/'data_audit.json').read_text());meta={}
    for split in ['train','validation']:
        p=CACHE/f'{split}.npz';assert sha(p)==audit['partitions'][split]['cache_sha256'];records[str(p)]=sha(p)
        p=TRAIN/f'{split}_windows.parquet';meta[split]=pd.read_parquet(p);records[str(p)]=sha(p)
    assert len(meta['train'])==107920 and len(meta['validation'])==36332
    plan=json.loads((PILOT/'results/holdout_protocol.json').read_text());indices=np.load(PILOT/'results/holdout_window_indices.npz');assigned=[]
    for f in plan['folds']:
        i=f['fold'];fit=indices[f'fold{i}_train_indices'];held=indices[f'fold{i}_held_indices']
        np.testing.assert_array_equal(fit,np.flatnonzero(meta['train'].flight_id.isin(f['train_flights'])))
        np.testing.assert_array_equal(held,np.flatnonzero(meta['train'].flight_id.isin(f['held_flights'])))
        np.testing.assert_array_equal(np.sort(np.r_[fit,held]),np.arange(107920))
        assert set(f['train_flights']).isdisjoint(f['held_flights']) and set(f['train_flights']).isdisjoint(meta['validation'].flight_id)
        assert {x.split(':')[0] for x in f['train_flights']}.isdisjoint(x.split(':')[0] for x in f['held_flights'])
        assert set(f['validation_flights'])==set(meta['validation'].flight_id)
        assert len(f['main_pairs'])==[25,12,11][i];assigned.extend(f['main_pairs'])
    assert len(assigned)==len(set(assigned))==48
    for name in ['holdout_protocol.json','holdout_window_indices.npz','holdout_pair_assignments.csv','holdout_coverage.csv']:shutil.copyfile(PILOT/'results'/name,OUT/name)
    for name in ['pairs','cases','windows','cohorts','swap_support','time_support','control_differences']:
        p=SWAP/'results'/f'{name}.csv';shutil.copyfile(p,OUT/p.name);records[str(p)]=sha(p)
    for p in [SWAP/'results/inputs.npz',ABLATION/'results/factual_curves.csv.gz',ABLATION/'results/internal_curves.csv.gz',PILOT/'model.py',ROOT/'scripts/train_velocity_rate_learning.py',ROOT/'scripts/train_velocity_rate_horizons.py',ROOT/'src/system_identification/models/velocity_rate_gru.py',ROOT/'src/system_identification/models/control_events.py']:
        records[str(p)]=sha(p)
    save(OUT/'gpu_before.json',dict(query=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv'],text=True),processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv'],text=True)))
    device=setup('cuda:1');raw=load_numpy('train');norms=[]
    for f in plan['folds']:
        i=f['fold'];folder=HERE/'folds'/f'fold{i}';folder.mkdir(parents=True,exist_ok=False)
        ids=indices[f'fold{i}_train_indices'];d={k:v[ids] for k,v in raw.items()};td=to_device(d,device)
        norm=training.fit_normalization(td,cfg);del td;torch.cuda.empty_cache()
        independent=independent_normalization(d,cfg);differences={k:float(np.max(abs(np.array(norm[k])-independent[k]))) for k in independent}
        assert max(differences.values())<2e-5,differences
        save(folder/'normalization.json',norm);save(folder/'normalization_receipt.json',dict(fold=i,train_windows=len(ids),train_flights=f['train_flights'],excluded_flights=f['held_flights'],validation_used=False,test_read=False,train_indices_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),cache_sha256=audit['partitions']['train']['cache_sha256'],normalization_sha256=sha(folder/'normalization.json'),fit_function_sha256=sha(ROOT/'scripts/train_velocity_rate_learning.py'),independent_float64_max_errors=differences,passed=True))
        norms.append(dict(fold=i,**differences));print('Fold',i,'normalization independently verified',differences,flush=True)
    save(OUT/'source_receipt.json',dict(base_sha='f1324dfaaa807297d83b1d246cf21e362d5140bf',config_sha256=sha(HERE/'config.json'),files=records,dataset_id='velocity_rate_multistep_v1',dataset_manifest_sha256=audit['manifest_sha256'],folds_fixed=True,train_validation_only=True,test_read=False,normalization_recomputed=True,independent_normalization=norms))
if __name__=='__main__':main()
