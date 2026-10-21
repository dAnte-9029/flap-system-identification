"""Independent metric/timing/source verification; never opens test artifacts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import torch
import yaml

from run import HERE, ROOT, TRAIN, CACHE, METRICS, KEYS, SEEDS, sha, save, refine
from system_identification.models.control_events import canonicalize_control_events as canonical


def verify_metrics(folder):
    z=np.load(folder/'predictions.npz');d=pd.read_parquet(folder/'endpoint_errors.parquet')
    maximum=0.;count=0
    for seed in SEEDS:
        for h in (500,1000):
            t=z[f'target_{h}'].astype(float)
            for variant in ('native','canonical'):
                p=z[f'{seed}_{h}_{variant}'].astype(float)
                actual=np.column_stack([np.linalg.norm(p[:,:3]-t[:,:3],axis=1),
                    np.linalg.norm(p[:,3:6]-t[:,3:6],axis=1),
                    np.rad2deg((Rotation.from_quat(p[:,[7,8,9,6]]).inv()*Rotation.from_quat(t[:,[7,8,9,6]])).magnitude()),
                    np.linalg.norm(p[:,10:13]-t[:,10:13],axis=1)])
                selected=d[(d.seed==seed)&(d.horizon_ms==h)&(d.variant==variant)]
                error=float(np.abs(actual-selected[list(METRICS)].to_numpy()).max())
                assert error<1e-10;maximum=max(maximum,error);count+=actual.size
    s=pd.read_csv(folder/'summary.csv')
    for r in s.itertuples():
        g=d[(d.horizon_ms==r.horizon_ms)&(d.variant==r.variant)]
        if r.scope=='affected':g=g[g.affected]
        if r.seed!='all':g=g[g.seed==int(r.seed)]
        per=[np.mean(x[r.metric].to_numpy()**2) for _,x in g.groupby('flight_id')]
        assert abs(np.sqrt(np.mean(per))-r.flight_rmse)<1e-12
    # Same paired cohort and exact endpoint trajectories must survive integration.
    return dict(scalar_metrics_recomputed=count,max_absolute_difference=maximum,summary_rows=len(s))


def function_equal(a,b):
    a=a[np.rint(a[:,4].astype(float)*1e6)>0];b=b[np.rint(b[:,4].astype(float)*1e6)>0]
    ea=np.cumsum(np.rint(a[:,4].astype(float)*1e6).astype(np.int64))
    eb=np.cumsum(np.rint(b[:,4].astype(float)*1e6).astype(np.int64))
    if len(ea)==0:assert len(eb)==0;return
    assert ea[-1]==eb[-1]
    starts=np.unique(np.r_[0,ea[:-1],eb[:-1]])
    np.testing.assert_array_equal(a[np.searchsorted(ea,starts,side='right'),:4],b[np.searchsorted(eb,starts,side='right'),:4])


def main():
    pre=HERE/'results/preflight';post=HERE/'results/integrated'
    assert json.loads((pre/'manifest.json').read_text())['passed']
    assert json.loads((post/'manifest.json').read_text())['passed']
    result={stage:verify_metrics(HERE/'results'/stage) for stage in ['preflight','integrated']}
    a=np.load(pre/'predictions.npz');b=np.load(post/'predictions.npz')
    for name in a.files:np.testing.assert_array_equal(a[name],b[name])
    result['preflight_and_integrated_endpoint_predictions_identical']=True
    prior=np.load(ROOT/'experiments/control_event_provenance_merge_v1/results/predictions.npz')
    for seed in SEEDS:
        for h in (500,1000):
            np.testing.assert_array_equal(prior[f'{seed}_{h}_native'],b[f'{seed}_{h}_native'])
            np.testing.assert_array_equal(prior[f'{seed}_{h}_merged'],b[f'{seed}_{h}_canonical'])
    result['prior_native_and_merged_endpoints_reproduced_exactly']=True
    weights={}
    for name in ['GRU-Control','GRU-NoControl']:
        for seed in SEEDS:
            path=TRAIN/f'train_1000ms/{name}/seed_{seed}/best.pt'
            expected=json.loads((path.parent/'summary.json').read_text())['checkpoint_sha256']
            assert sha(path)==expected;weights[str(path.relative_to(ROOT))]=expected
    result['all_six_checkpoint_hashes']=weights
    registry=yaml.safe_load((ROOT/'configs/data/velocity_rate_dataset_registry.yaml').read_text())
    entry=registry['datasets']['velocity_rate_multistep_v1']
    data_root=Path('/home/zn/flap-system-identification/dataset/velocity_rate_multistep_v1')
    assert sha(data_root/'dataset_manifest.json')==entry['manifest_sha256']
    old=json.loads((ROOT/'experiments/control_event_provenance_merge_v1/results/provenance_manifest.json').read_text())
    checked={}
    for name,digest in old['checked_artifacts'].items():
        assert name.endswith(('_train.parquet','_validation.parquet'))
        assert sha(data_root/name)==digest;checked[name]=digest
    provenance={r['flight_id']:r for r in json.loads((data_root/'flight_provenance.json').read_text()) if r['split'] in ('train','validation')}
    windows=pd.read_parquet(post/'windows.parquet')
    train_windows=pd.read_parquet(TRAIN/'train_windows.parquet')
    assert set(windows.flight_id).isdisjoint(train_windows.flight_id)
    for r in windows.itertuples():
        p=provenance[r.flight_id]
        assert p['split']=='validation'
        assert p['start_us']<=r.support_start_us<=r.decision_us<r.support_end_us<=p['end_us']
        for start,end in p['exclude_intervals_us']:
            assert r.support_end_us<start or r.support_start_us>end
    result['source_checks']=dict(manifest_sha256=entry['manifest_sha256'],native_artifacts=checked,
        validation_windows=len(windows),validation_flights=windows.flight_id.nunique(),train_validation_disjoint=True)
    # Every previously merged interval, checked with an independent boundary union.
    changes=np.load(ROOT/'experiments/control_event_provenance_merge_v1/results/merge_changes.npz')
    x=torch.from_numpy(changes['before']);le=torch.from_numpy(changes['before_length'])
    y,n=canonical(x,le)
    for i in range(len(x)):
        function_equal(x[i,:le[i]].numpy(),y[i,:n[i]].numpy())
        function_equal(changes['after'][i,:changes['after_length'][i]],y[i,:n[i]].numpy())
    result['independent_whole_function_checks']=len(x)
    # Repack selected real intervals from registered native records. This replaces
    # old tests that require caches to exist locally inside this new worktree.
    from prepare_velocity_rate_learning import interval_events
    source_repack=[]
    dataset_manifest=json.loads((data_root/'dataset_manifest.json').read_text())
    for split in ['train','validation']:
        meta=pd.read_parquet(TRAIN/f'{split}_windows.parquet')
        z0=np.load(CACHE/f'{split}.npz')
        cached_events=z0['events'];cached_lengths=z0['event_length']
        sample_path=data_root/f'samples_{split}.parquet'
        assert sha(sample_path)==dataset_manifest['artifact_sha256'][sample_path.name]
        samples=pd.read_parquet(sample_path,columns=['flight_id','state_measurement_us','segment_id','valid_core'])
        natives={kind:pd.read_parquet(data_root/f'native_actuator_{kind}_{split}.parquet') for kind in ['motors','servos']}
        choices=np.unique(np.linspace(0,len(meta)-1,12,dtype=int))
        intervals=0
        for index in choices:
            row=meta.iloc[index];sample=samples[samples.flight_id==row.flight_id].reset_index(drop=True)
            n=int(row.future_count);o=int(row.origin_sample)
            t=sample.state_measurement_us.to_numpy()[o:o+n+1]
            assert sample.valid_core.iloc[int(row.start_sample):int(row.end_sample)+1].all()
            assert sample.segment_id.iloc[int(row.start_sample):int(row.end_sample)+1].eq(row.segment_id).all()
            raw=[natives[k][natives[k].flight_id==row.flight_id] for k in ['motors','servos']]
            fresh,fl=interval_events(*raw,t[:-1],t[1:])
            old= torch.from_numpy(cached_events[index,:n]);ol=torch.from_numpy(cached_lengths[index,:n])
            packed,pl=canonical(old,ol,torch.from_numpy(np.diff(t)*1e-6))
            np.testing.assert_array_equal(fresh,packed.numpy());np.testing.assert_array_equal(fl,pl.numpy())
            for j in range(n):function_equal(old[j,:ol[j]].numpy(),fresh[j,:fl[j]])
            intervals+=n
        source_repack.append(dict(split=split,windows=len(choices),intervals=intervals,sample_sha256=sha(sample_path)))
        del cached_events,cached_lengths,samples,natives
    result['native_repack_checks']=source_repack
    # Gradients only on authorized training inputs. No optimizer or parameter update.
    from system_identification.models.velocity_rate_gru import VelocityRateGRU
    audit=json.loads((TRAIN/'data_audit.json').read_text())
    digest=audit['partitions']['train']['cache_sha256'];assert sha(CACHE/'train.npz')==digest
    z=np.load(CACHE/'train.npz');device=torch.device('cuda:0');torch.set_num_threads(4)
    inp={k:torch.as_tensor(z[k][:32],device=device) for k in KEYS};target=torch.as_tensor(z['target'][:32],device=device)
    mask=torch.as_tensor(z['future_mask'][:32],device=device)
    gradients=[]
    for control in [True,False]:
        name='GRU-Control' if control else 'GRU-NoControl'
        path=TRAIN/f'train_1000ms/{name}/seed_17/best.pt'
        expected=json.loads((path.parent/'summary.json').read_text())['checkpoint_sha256'];assert sha(path)==expected
        ck=torch.load(path,map_location=device,weights_only=False)
        model=VelocityRateGRU(ck['normalization'],control=control).to(device);model.load_state_dict(ck['model'],strict=True)
        original={k:v.detach().clone() for k,v in model.state_dict().items()}
        pred=model(inp)
        from train_velocity_rate_learning import window_loss
        loss=window_loss(pred,target,mask,torch.tensor(ck['normalization']['loss_scales'],device=device),torch.ones(4,device=device)).mean()
        loss.backward();parameters=[p for p in model.parameters() if p.grad is not None]
        assert parameters and all(torch.isfinite(p.grad).all() for p in parameters)
        assert all(torch.equal(original[k],v) for k,v in model.state_dict().items())
        gradients.append(dict(model=name,loss=float(loss.detach()),parameters_with_grad=sum(p.numel() for p in parameters),weights_unchanged=True))
    result['gpu_backward']=dict(train_cache_sha256=digest,windows=32,checks=gradients,optimizer_steps=0)
    result['test_opened']=False
    save(HERE/'results/verification.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
