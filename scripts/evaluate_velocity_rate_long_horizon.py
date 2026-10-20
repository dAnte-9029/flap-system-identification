"""Frozen 200ms-trained checkpoints, autonomous validation out to 2s. Never opens test."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'))
import prepare_velocity_rate_learning as preparation
from system_identification.models.velocity_rate_gru import VelocityRateGRU,baseline,errors
from system_identification.data.velocity_rate_dataset import file_hash,write_json,VelocityRateDataset
from train_velocity_rate_learning import INPUT_KEYS,batch
from evaluate_velocity_rate_learning import oracle,METRICS
from verify_velocity_rate_learning import independent_errors
SOURCE=ROOT/'experiments/velocity_rate_gru64_round1'
OUT=ROOT/'experiments/velocity_rate_gru64_long_horizon_v1'
CACHE=ROOT/'artifacts/velocity_rate_gru64_long_horizon_v1'
HORIZONS=[100,200,500,1000,2000]
MODELS=['Kinematic Baseline','GRU-NoControl','GRU-Control']
SEEDS=[17,29,43]


def prepare():
    OUT.mkdir(exist_ok=False); CACHE.mkdir(exist_ok=False)
    reg=yaml.safe_load((ROOT/'configs/data/velocity_rate_dataset_registry.yaml').read_text())
    assert reg['default_dataset_id']=='velocity_rate_multistep_v1'
    entry=reg['datasets'][reg['default_dataset_id']]; root=ROOT/entry['repository_relative_root']
    assert file_hash(ROOT/entry['manifest_path'])==entry['manifest_sha256']
    manifest=json.loads((root/'dataset_manifest.json').read_text())
    used=['samples_validation.parquet','windows_validation.parquet','native_actuator_motors_validation.parquet',
          'native_actuator_servos_validation.parquet','split_manifest.json','flights.csv']
    hashes={}
    for name in used:
        hashes[name]=file_hash(root/name); assert hashes[name]==manifest['artifact_sha256'][name],name
    old_archive=json.loads((SOURCE/'archive_manifest.json').read_text())
    for name,rec in old_archive['files'].items(): assert file_hash(SOURCE/name)==rec['sha256'],name
    w=pd.read_parquet(root/'windows_validation.parquet')
    w=w[(w.sampling_group=='main_50hz')&(w.history_us==500000)]
    subsets={h:w[w.horizon_us==h*1000].copy() for h in HORIZONS}
    sets=[set(zip(s.flight_id,s.origin_sample)) for s in subsets.values()]
    common=set.intersection(*sets)
    config=json.loads((SOURCE/'config.json').read_text())
    config['horizon_us']=2000000
    config['expected_windows']['validation']=len(subsets[2000])
    config['expected_flights']['validation']=subsets[2000].flight_id.nunique()
    # Reuse the frozen packer but redirect all outputs to this new run.
    preparation.EXP=OUT; preparation.CACHE=CACHE
    packed=preparation.pack(root,'validation',config)
    allmeta=pd.read_parquet(OUT/'validation_windows.parquet')
    keep=np.array([(f,o) in common for f,o in zip(allmeta.flight_id,allmeta.origin_sample)])
    indices=np.flatnonzero(keep); meta=allmeta[keep].reset_index(drop=True)
    np.save(CACHE/'common_indices.npy',indices)
    meta.to_parquet(OUT/'common_windows.parquet',index=False)
    inventory=pd.read_csv(root/'flights.csv').set_index('flight_id').loc[meta.flight_id.unique()]
    assert (inventory.split=='validation').all()
    meta_flights=inventory[['log_path','flight_date','sha256']].reset_index()
    meta_flights.to_csv(OUT/'flight_inventory.csv',index=False)
    for h,subset in subsets.items():
        endpoint=subset.set_index(['flight_id','origin_sample']).loc[list(zip(meta.flight_id,meta.origin_sample))]
        meta[f'index_{h}ms']=(endpoint.end_sample.to_numpy()-meta.origin_sample.to_numpy()-1).astype(int)
    meta.to_parquet(OUT/'common_windows.parquet',index=False)
    settings=dict(dataset_id=reg['default_dataset_id'],partition='validation',history_ms=500,
        trained_horizon_ms=200,evaluation_horizons_ms=HORIZONS,seeds=SEEDS,batch_size=2048,
        retrained=False,reselected_checkpoints=False,normalization_refit=False,test_opened=False,
        cohort='intersection of registered origins at all five horizons; matched flights and origins for every model',
        endpoint='registered end_sample at each nominal horizon; actual dt, tolerance 12ms',
        future_state_feedback=False,future_phase_frequency='hold valid origin frequency; linear phase extrapolation',
        future_controls='recorded event sequence for Control only; no independent intervention',
        source_experiment=str(SOURCE.relative_to(ROOT)),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_code_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in [Path(__file__),ROOT/'src/system_identification/models/velocity_rate_gru.py',ROOT/'scripts/prepare_velocity_rate_learning.py']})
    write_json(OUT/'config.json',settings)
    write_json(OUT/'data_audit.json',dict(registry_entry=entry,verified_artifact_sha256=hashes,
        counts={str(h):dict(windows=len(s),flights=s.flight_id.nunique()) for h,s in subsets.items()},
        common_windows=len(meta),common_flights=meta.flight_id.nunique(),packed=packed,
        source_archive_verified=True,old_archive_sha256=file_hash(SOURCE/'archive_manifest.json')))
    print(f'prepared {len(meta)} common windows, {meta.flight_id.nunique()} flights',flush=True)


def load_data(device):
    archive=np.load(CACHE/'validation.npz'); idx=np.load(CACHE/'common_indices.npy')
    return {key:torch.from_numpy(archive[key][idx]).to(device) for key in archive.files}


def endpoint_frame(err,times,meta,model,seed):
    frames=[]
    for h in HORIZONS:
        ix=meta[f'index_{h}ms'].to_numpy(); rows=np.arange(len(meta))
        actual=times[rows,ix]
        assert np.max(abs(actual-h/1000))<=.012001
        f=meta[['window_id','flight_id','origin_sample','review','high_dynamics','mode_change','control_change']].copy()
        f['model']=model; f['seed']=seed; f['horizon_ms']=h; f['actual_time_s']=actual; f['native_step']=ix+1
        for j,name in enumerate(METRICS): f[name]=err[rows,ix,j].astype(float)
        frames.append(f)
    return pd.concat(frames,ignore_index=True)


def summaries(frames):
    endpoints=pd.concat(frames,ignore_index=True)
    flight=endpoints.groupby(['model','seed','horizon_ms','flight_id'])[METRICS].mean().reset_index()
    flight.to_csv(OUT/'per_flight.csv',index=False)
    seeds=flight.groupby(['model','seed','horizon_ms'])[METRICS].mean().reset_index()
    seeds.to_csv(OUT/'per_seed.csv',index=False)
    result=[]
    for (model,h),g in flight.groupby(['model','horizon_ms']):
        f=g.groupby('flight_id')[METRICS].mean()
        s=seeds[(seeds.model==model)&(seeds.horizon_ms==h)]
        w=endpoints[(endpoints.model==model)&(endpoints.horizon_ms==h)].groupby('window_id')[METRICS].mean()
        for metric in METRICS:
            result.append(dict(model=model,horizon_ms=h,metric=metric,flight_mean=f[metric].mean(),flight_std=f[metric].std(ddof=1),
                seed_std=s[metric].std(ddof=1) if len(s)>1 else None,window_mean=w[metric].mean(),window_std=w[metric].std(ddof=1),
                window_p95=w[metric].quantile(.95),window_p99=w[metric].quantile(.99),n_flights=len(f),n_windows=len(w)))
    pd.DataFrame(result).to_csv(OUT/'summary.csv',index=False)
    rng=np.random.default_rng(20261020); pairs=[]
    for h in HORIZONS:
        f=flight[flight.horizon_ms==h].groupby(['model','flight_id'])[METRICS].mean()
        for ref,can in [(MODELS[0],MODELS[1]),(MODELS[0],MODELS[2]),(MODELS[1],MODELS[2])]:
            for metric in METRICS:
                a=f.loc[ref,metric]; b=f.loc[can,metric].reindex(a.index); d=(b-a).to_numpy()
                ci=np.quantile(d[rng.integers(len(d),size=(5000,len(d)))].mean(1),[.025,.975])
                pairs.append(dict(horizon_ms=h,reference=ref,candidate=can,metric=metric,improvement_pct=100*(a.mean()-b.mean())/a.mean(),
                    delta=d.mean(),better_flights=int((d<0).sum()),bootstrap_low=ci[0],bootstrap_high=ci[1]))
    pd.DataFrame(pairs).to_csv(OUT/'paired_comparisons.csv',index=False)
    groups=[]
    for field in ['review','high_dynamics','mode_change']:
        g=endpoints.groupby(['model','seed','horizon_ms',field,'flight_id'])[METRICS].mean().reset_index()
        g=g.groupby(['model','horizon_ms',field,'flight_id'])[METRICS].mean().reset_index()
        f=g.groupby(['model','horizon_ms',field])[METRICS].agg(['mean','std'])
        f.columns=['_'.join(c) for c in f.columns]; f=f.reset_index().rename(columns={field:'group'})
        f['n_flights']=g.groupby(['model','horizon_ms',field]).size().to_numpy();f['grouping']=field;groups.append(f)
    pd.concat(groups,ignore_index=True).to_csv(OUT/'group_metrics.csv',index=False)


@torch.no_grad()
def evaluate():
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=True
    assert torch.cuda.is_available();device=torch.device('cuda:0')
    data=load_data(device); meta=pd.read_parquet(OUT/'common_windows.parquet'); n=len(meta)
    times=data['dt'].cumsum(1).cpu().numpy(); mask=data['future_mask'].cpu().numpy()
    choices=[]; fids=meta.flight_id.unique()
    for fid in fids[np.linspace(0,len(fids)-1,6,dtype=int)]:
        ix=np.flatnonzero(meta.flight_id.to_numpy()==fid);choices.append(int(ix[len(ix)//2]))
    examples={key:data[key][choices].cpu().numpy() for key in ['initial','target','dt','future_mask']}
    meta.iloc[choices].to_csv(OUT/'example_windows.csv',index=False)
    models=[(MODELS[0],-1,None)]; checkpoints={}
    for name in MODELS[1:]:
        for seed in SEEDS:
            path=SOURCE/name/f'seed_{seed}/best.pt'; expected=json.loads((path.parent/'summary.json').read_text())['checkpoint_sha256']
            assert file_hash(path)==expected;checkpoints[str(path.relative_to(ROOT))]=expected
            c=torch.load(path,map_location=device,weights_only=False)
            m=VelocityRateGRU(c['normalization'],control=name=='GRU-Control').to(device);m.load_state_dict(c['model']);m.eval()
            models.append((name,seed,m))
    folder=OUT/'metrics';folder.mkdir(exist_ok=True)
    frames=[]; native_frames=[];diagnostic=[];runtime=[];checks=[]
    oracle_pred=torch.cat([oracle(data['initial'][ix],data['target'][ix],data['dt'][ix]) for ix in torch.arange(n,device=device).split(2048)])
    er=errors(oracle_pred,data['target']).cpu().numpy();er[...,2]*=180/np.pi
    of=endpoint_frame(er,times,meta,'Observed v/w integration',-1)
    of.to_parquet(folder/'observation_integration.parquet',index=False)
    of.groupby(['horizon_ms','flight_id'])[METRICS].mean().reset_index().to_csv(OUT/'observation_per_flight.csv',index=False)
    for name,seed,model in models:
        tick=time.time();torch.cuda.reset_peak_memory_stats(device);predictions=[]
        for ix in torch.arange(n,device=device).split(2048):
            pred=baseline(data['initial'][ix],data['dt'][ix]) if model is None else model(batch(data,ix))
            assert torch.isfinite(pred).all();predictions.append(pred)
        pred=torch.cat(predictions);er=errors(pred,data['target']).cpu().numpy();er[...,2]*=180/np.pi
        slug=name.replace(' ','_')+f'_seed_{seed}'
        frame=endpoint_frame(er,times,meta,name,seed);frames.append(frame)
        frame.to_parquet(folder/f'{slug}_endpoints.parquet',index=False)
        np.savez_compressed(CACHE/f'{slug}_native_errors.npz',errors=er,mask=mask,actual_time_s=times)
        i,k=np.where(mask)
        native=pd.DataFrame(dict(flight_id=meta.flight_id.to_numpy()[i],native_step=k+1,actual_time_s=times[i,k]))
        for j,metric in enumerate(METRICS):native[metric]=er[i,k,j].astype(float)
        g=native.groupby(['flight_id','native_step'])[[*METRICS,'actual_time_s']].mean().reset_index()
        g['n_windows']=native.groupby(['flight_id','native_step']).size().to_numpy();g['model']=name;g['seed']=seed;native_frames.append(g)
        dyn=errors(pred,oracle_pred).cpu().numpy();dyn[...,2]*=180/np.pi
        df=endpoint_frame(dyn,times,meta,name,seed)
        diagnostic.append(df.groupby(['model','seed','horizon_ms','flight_id'])[['position_m','attitude_deg']].mean().reset_index())
        examples[slug]=pred[choices].cpu().numpy()
        # Independent SciPy metric recomputation, spanning the cohort and all horizons.
        selected=np.linspace(0,n-1,128,dtype=int)
        check=independent_errors(pred[selected].cpu().numpy().astype(float),data['target'][selected].cpu().numpy().astype(float))
        difference=np.max(abs(check-er[selected]),axis=(0,1))
        assert difference.max()<.0002,(name,seed,difference)
        checks.append(dict(model=name,seed=seed,scipy_metric_max_difference=difference.tolist(),
            quaternion_norm_max_error=(torch.linalg.vector_norm(pred[...,6:10],dim=-1)-1).abs().max().item()))
        runtime.append(dict(model=name,seed=seed,seconds=time.time()-tick,peak_gpu_bytes=torch.cuda.max_memory_allocated(device),finite=True))
        print(f'evaluated {name} seed {seed}: {time.time()-tick:.1f}s',flush=True)
    summaries(frames)
    pd.concat(native_frames,ignore_index=True).to_csv(OUT/'native_steps_per_flight.csv',index=False)
    pd.concat(diagnostic,ignore_index=True).to_csv(OUT/'dynamics_vs_observation_integral.csv',index=False)
    np.savez_compressed(OUT/'trajectory_examples.npz',**examples)
    pd.DataFrame(runtime).to_csv(OUT/'runtime.csv',index=False)
    write_json(OUT/'checkpoint_manifest.json',checkpoints)
    write_json(OUT/'numerical_verification.json',dict(status='passed',gpu=torch.cuda.get_device_name(),torch=torch.__version__,tf32=True,batch_size=2048,checks=checks))


def verify():
    # Cross-check the new 2s pack against the authoritative dataset loader (validation only).
    meta=pd.read_parquet(OUT/'common_windows.parquet'); original=pd.read_parquet(OUT/'validation_windows.parquet')
    archive=np.load(CACHE/'validation.npz'); ids=np.load(CACHE/'common_indices.npy')
    ds=VelocityRateDataset(ROOT/'dataset/velocity_rate_multistep_v1',split='validation',horizon_us=2000000,verify_hashes=False)
    for i in [0,len(ids)//2,len(ids)-1]:
        j=int(ids[i]); sample=ds[j];k=len(sample['inputs']['future_dt_s']);h=len(sample['inputs']['history'])
        np.testing.assert_allclose(archive['history'][j,:h,3:],sample['inputs']['history'][:,3:],atol=1e-6,rtol=1e-6)
        np.testing.assert_allclose(archive['dt'][j,:k],sample['inputs']['future_dt_s'],atol=1e-8)
        expected=sample['targets']['state'].copy();expected[:,:3]-=sample['inputs']['initial_state'][:3]
        np.testing.assert_allclose(archive['target'][j,:k],expected,atol=3e-5,rtol=1e-5)
        for step in range(k):
            le=archive['event_length'][j,step];ev=archive['events'][j,step,:le]
            np.testing.assert_allclose(ev[:,4].sum(),archive['dt'][j,step],atol=1e-8)
            np.testing.assert_allclose(ev[0,:4],sample['inputs']['future_control']['step_control'][step],atol=1e-7)
    reported=pd.read_csv(OUT/'summary.csv'); maxdiff=0
    for model in MODELS:
        frames=[pd.read_parquet(OUT/f'metrics/{model.replace(" ","_")}_seed_{seed}_endpoints.parquet') for seed in ([-1] if model==MODELS[0] else SEEDS)]
        allf=pd.concat(frames,ignore_index=True)
        for h in HORIZONS:
            f=allf[allf.horizon_ms==h]
            observed=f.groupby(['seed','flight_id'])[METRICS].mean().groupby('flight_id').mean().mean()
            expected=reported[(reported.model==model)&(reported.horizon_ms==h)].set_index('metric').loc[METRICS,'flight_mean']
            maxdiff=max(maxdiff,float(abs(observed-expected).max()))
    assert maxdiff<1e-12
    old=json.loads((SOURCE/'archive_manifest.json').read_text())
    for name,rec in old['files'].items():assert file_hash(SOURCE/name)==rec['sha256']
    write_json(OUT/'verification.json',dict(status='passed',authoritative_loader_windows_verified=3,
        aggregation_max_difference=maxdiff,prior_experiment_unchanged=True,test_opened=False))
    print('independent verification passed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','evaluate','verify']);args=parser.parse_args()
    globals()[args.stage]()
