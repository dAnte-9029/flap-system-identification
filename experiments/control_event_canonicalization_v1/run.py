"""Frozen full-validation acceptance, followed by integrated-entry verification."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'), str(HERE)]
from legacy_velocity_rate_gru import VelocityRateGRU as LegacyModel, errors
from system_identification.models.control_events import (
    canonicalize_control_events as canonical, CANONICAL_CONTROL_EVENTS)
from train_velocity_rate_horizons import crop_horizon

CACHE = Path('/home/zn/flap-system-identification/.worktrees/velocity-rate-dataset-v1/artifacts/velocity_rate_gru64_horizon_training_v1')
TRAIN = ROOT/'experiments/velocity_rate_gru64_horizon_training_v1'
KEYS = ('history','history_mask','history_dt','history_length','initial','dt','events','event_length')
METRICS = ('position_m','velocity_m_s','attitude_deg','rate_rad_s')
SEEDS = (17,29,43)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(1<<20),b''): h.update(x)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,
                                   default=lambda x:x.item())+'\n')


def refine(events, lengths, parts, rng=None):
    """Integer-clock refinement only; zero-length subpieces have empty support."""
    d=np.rint(events[...,4].astype(float)*1e6).astype(np.int64)
    out=np.repeat(events,parts,axis=-2).copy()
    if rng is None:
        pieces=d[...,None]//parts+(np.arange(parts)<(d[...,None]%parts))
    else:
        cuts=np.sort((rng.random((*d.shape,parts-1))*d[...,None]).astype(np.int64),axis=-1)
        pieces=np.diff(np.concatenate((np.zeros_like(d[...,None]),cuts,d[...,None]),axis=-1),axis=-1)
    out[...,4]=(pieces.reshape(*d.shape[:-1],-1)*1e-6).astype(events.dtype)
    return out, lengths*parts


def forward(model, inputs, integrated):
    if integrated: return model(inputs)
    ev,le=canonical(inputs['events'],inputs['event_length'],inputs['dt'])
    return model(dict(inputs,events=ev,event_length=le))


def check_cases(model,device,integrated,seed):
    """Replay all eight previously diagnosed histories, including real swaps."""
    source=ROOT/'experiments/control_event_invariance_check/results'
    z=np.load(source/'traces.npz')
    cases=pd.read_csv(source/'windows.csv')
    rows=[]
    for r in cases.itertuples():
        def inputs(variant):
            return {k:torch.as_tensor(z[f'{r.window}__{variant}__input_{k}'],device=device) for k in KEYS}
        original=inputs('native')
        a,la=canonical(original['events'],original['event_length'],original['dt'])
        base=forward(model,original,integrated)
        for variant in ['split2','split4','prior_refined','fixed_split4','pad_only','donor']:
            x=inputs(variant)
            b,lb=canonical(x['events'],x['event_length'],x['dt'])
            pred=forward(model,x,integrated)
            equal=variant!='donor'
            if equal:
                assert torch.equal(a,b) and torch.equal(la,lb)
                assert torch.equal(pred,base), (r.window,variant,seed)
            for index,horizon in [(24,500),(49,1000)]:
                e=errors(pred[:,index].double(),base[:,index].double()).cpu().numpy()[0]
                e[2]*=180/np.pi
                rows.append(dict(window_id=r.window_id,flight_id=r.flight_id,seed=seed,
                    variant=variant,horizon_ms=horizon,equivalent=equal,
                    **dict(zip(METRICS,e))))
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['preflight','integrated'],required=True)
    stage=parser.parse_args().stage; integrated=stage=='integrated'
    out=HERE/'results'/stage;out.mkdir(parents=True,exist_ok=True)
    acceptance=json.loads((HERE/'acceptance.json').read_text())
    tick=time.time();torch.set_num_threads(4)
    assert torch.cuda.is_available();device=torch.device('cuda:0')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.reset_peak_memory_stats()
    audit=json.loads((TRAIN/'data_audit.json').read_text())
    expected=audit['partitions']['validation']['cache_sha256']
    assert sha(CACHE/'validation.npz')==expected
    provenance=json.loads((ROOT/'experiments/control_event_provenance_merge_v1/results/provenance_manifest.json').read_text())
    oldmanifest=json.loads((ROOT/'experiments/control_event_provenance_merge_v1/results/evaluation_manifest.json').read_text())
    assert sha(HERE/'legacy_velocity_rate_gru.py')==oldmanifest['model_code_sha256']
    meta=pd.read_parquet(TRAIN/'validation_windows.parquet')
    assert len(meta)==36332 and meta.flight_id.nunique()==27
    assert meta.split.eq('validation').all() and meta.sampling_group.eq('main_50hz').all()
    archive=np.load(CACHE/'validation.npz')
    data=crop_horizon({k:archive[k] for k in archive.files},meta.count_1000ms.to_numpy())
    same=(np.all(data['events'][:,:,1:,:4]==data['events'][:,:,:-1,:4],axis=-1)
          &(np.arange(1,data['events'].shape[2])[None,None,:]<data['event_length'][:,:,None])
          &data['future_mask'][:,:,None])
    changed=same.any(-1)
    meta=meta.assign(changed_steps=changed.sum(1));meta.to_parquet(out/'windows.parquet',index=False)
    selected={h:meta[f'count_{h}ms'].to_numpy()-1 for h in (500,1000)}
    truth={h:data['target'][np.arange(len(meta)),ix] for h,ix in selected.items()}
    payload={f'target_{h}':t for h,t in truth.items()};rows=[];checks=[];cases=[];checkpoints={};throughput=[]
    for seed in SEEDS:
        path=TRAIN/f'train_1000ms/GRU-Control/seed_{seed}/best.pt'
        digest=json.loads((path.parent/'summary.json').read_text())['checkpoint_sha256']
        assert sha(path)==digest;checkpoints[str(path.relative_to(ROOT))]=digest
        ck=torch.load(path,map_location=device,weights_only=False)
        legacy=LegacyModel(ck['normalization'],control=True).to(device).eval();legacy.load_state_dict(ck['model'],strict=True)
        if integrated:
            from system_identification.models.velocity_rate_gru import VelocityRateGRU
            model=VelocityRateGRU(ck['normalization'],control=True).to(device).eval()
            assert model.control_event_policy==CANONICAL_CONTROL_EVENTS
            model.load_state_dict(ck['model'],strict=True)
        else:model=legacy
        endpoints={(variant,h):[] for variant in ['native','canonical'] for h in selected}
        maxdiff={variant:0. for variant in ['split2','split4','random7','repeat','reference']}
        seen=0;start_time=time.time()
        with torch.no_grad():
            for start in range(0,len(meta),512):
                end=min(start+512,len(meta));inp={k:torch.as_tensor(data[k][start:end],device=device) for k in KEYS}
                a,la=canonical(inp['events'],inp['event_length'],inp['dt'])
                aa,lla=canonical(a,la,inp['dt']);assert torch.equal(a,aa) and torch.equal(la,lla)
                native=legacy(inp)
                base=forward(model,inp,integrated)
                if integrated:
                    reference=legacy(dict(inp,events=a,event_length=la))
                    maxdiff['reference']=max(maxdiff['reference'],float((reference-base).abs().max()))
                    assert torch.equal(reference,base)
                for name,parts in [('split2',2),('split4',4),('random7',7)]:
                    ev,le=refine(data['events'][start:end],data['event_length'][start:end],parts,
                                 np.random.default_rng(911+start) if name=='random7' else None)
                    x=dict(inp,events=torch.as_tensor(ev,device=device),event_length=torch.as_tensor(le,device=device))
                    b,lb=canonical(x['events'],x['event_length'],x['dt'])
                    assert torch.equal(a,b) and torch.equal(la,lb), (seed,name,start)
                    pred=forward(model,x,integrated)
                    maxdiff[name]=max(maxdiff[name],float((pred-base).abs().max()))
                    assert torch.equal(pred,base), (seed,name,start)
                if start==0:
                    again=forward(model,inp,integrated)
                    maxdiff['repeat']=float((again-base).abs().max());assert torch.equal(again,base)
                for h,ix in selected.items():
                    j=torch.as_tensor(ix[start:end],device=device);i=torch.arange(end-start,device=device)
                    for variant,pred in [('native',native),('canonical',base)]:endpoints[variant,h].append(pred[i,j].cpu().numpy())
                seen+=end-start
                if start%10240==0: print(stage,'seed',seed,'windows',seen,flush=True)
            cases.extend(check_cases(model,device,integrated,seed))
        elapsed=time.time()-start_time
        throughput.append(dict(seed=seed,seconds=elapsed,full_rollouts=5*len(meta),rollouts_per_second=5*len(meta)/elapsed))
        checks.append(dict(seed=seed,windows=seen,max_abs_prediction_difference=maxdiff))
        for (variant,h),values in endpoints.items():
            pred=np.concatenate(values);payload[f'{seed}_{h}_{variant}']=pred
            e=errors(torch.tensor(pred,dtype=torch.float64),torch.tensor(truth[h],dtype=torch.float64)).numpy();e[:,2]*=180/np.pi
            frame=meta[['window_id','flight_id']].copy();frame['seed']=seed;frame['horizon_ms']=h;frame['variant']=variant
            frame['affected']=(changed&(np.arange(changed.shape[1])[None,:]<=selected[h][:,None])).any(1)
            for j,name in enumerate(METRICS):frame[name]=e[:,j]
            rows.append(frame)
    d=pd.concat(rows,ignore_index=True);d.to_parquet(out/'endpoint_errors.parquet',index=False)
    np.savez_compressed(out/'predictions.npz',**payload)
    pd.DataFrame(cases).to_csv(out/'prior_cases.csv',index=False)
    summaries=[];per_flight=[]
    for scope,subset in [('all',d),('affected',d[d.affected])]:
        for (h,variant),g in subset.groupby(['horizon_ms','variant']):
            for seed in [*SEEDS,'all']:
                x=g if seed=='all' else g[g.seed==seed]
                for metric in METRICS:
                    per=x.assign(e2=x[metric]**2).groupby('flight_id').e2.mean()
                    summaries.append(dict(scope=scope,horizon_ms=h,variant=variant,seed=seed,metric=metric,
                        windows=x.window_id.nunique(),flights=x.flight_id.nunique(),flight_rmse=np.sqrt(per.mean()),window_rmse=np.sqrt(np.mean(x[metric]**2))))
                    per_flight.extend(dict(scope=scope,horizon_ms=h,variant=variant,seed=seed,metric=metric,flight_id=fid,rmse=np.sqrt(val)) for fid,val in per.items())
    s=pd.DataFrame(summaries);s.to_csv(out/'summary.csv',index=False);pd.DataFrame(per_flight).to_csv(out/'per_flight.csv',index=False)
    index=['scope','horizon_ms','seed','metric'];native=s[s.variant=='native'].set_index(index);new=s[s.variant=='canonical'].set_index(index)
    pairs=native[['windows','flights','flight_rmse']].rename(columns={'flight_rmse':'native_rmse'}).copy()
    pairs['canonical_rmse']=new.flight_rmse;pairs['change_pct']=100*(pairs.canonical_rmse/pairs.native_rmse-1)
    pairs=pairs.reset_index();pairs.to_csv(out/'paired_comparisons.csv',index=False)
    gates={scope:bool((pairs[pairs.scope==scope].change_pct<=limit).all()) for scope,limit in acceptance['max_rmse_increase_percent'].items()}
    passed=all(gates.values()) and all(all(v==0 for v in c['max_abs_prediction_difference'].values()) for c in checks)
    manifest=dict(stage=stage,policy=CANONICAL_CONTROL_EVENTS,base_commit='653b60d',windows=len(meta),flights=meta.flight_id.nunique(),
        seeds=SEEDS,changed_windows=int(changed.any(1).sum()),validation_cache_sha256=expected,checkpoints=checkpoints,
        gpu=torch.cuda.get_device_name(device),peak_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=time.time()-tick,
        checks=checks,throughput=throughput,acceptance=acceptance,gates=gates,passed=passed,retrained=False,test_opened=False,
        dataset_provenance=provenance,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),HERE/'legacy_velocity_rate_gru.py',ROOT/'src/system_identification/models/control_events.py',ROOT/'src/system_identification/models/velocity_rate_gru.py']})
    save(out/'manifest.json',manifest)
    print('FINISHED',stage,'passed',passed,'seconds',time.time()-tick,flush=True)
    if not passed: raise RuntimeError('Acceptance failed; do not promote')


if __name__=='__main__': main()
