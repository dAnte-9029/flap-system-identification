"""Frozen-index factual response replay, with original model/packing interfaces."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import time
import numpy as np
import pandas as pd
import torch
import yaml

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SCOPE=ROOT/'experiments/local_control_response_scope_v1'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from system_identification.models.velocity_rate_gru import VelocityRateGRU, baseline, integrate
from system_identification.models.control_events import canonicalize_control_events_numpy, LEGACY_CONTROL_EVENTS
from system_identification.data.velocity_rate_dataset import STATE, CONTROL, PHASE, TARGET
from prepare_velocity_rate_learning import interval_events


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


response=module('response_reference',ROOT/'experiments/control_response_eval_v1/evaluate.py')
canonical_run=module('canonical_reference',ROOT/'experiments/control_event_canonicalization_v1/run.py')
KEYS=response.INPUT_KEYS
# The historical packer checks its raw construction against the original raw
# interval packer. Only that comparison uses legacy encoding; inference is canonical.
response.interval_events=lambda *args,**kwargs:interval_events(*args,**kwargs,event_policy=LEGACY_CONTROL_EVENTS)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,
        default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')


def verify_manifest_files(root,manifest,paths,key='artifacts'):
    entries={x['path']:x['sha256'] for x in json.loads(manifest.read_text())[key]}
    result={}
    for rel in paths:
        actual=sha(root/rel)
        assert actual==entries[rel],rel
        result[str(root/rel)]=actual
    return result


def freeze_selection(out):
    paths=['config.json','results/frozen_selection.json','results/description/selected_pair_index.csv',
           'results/description/background_index.csv','results/description/asset_receipt.json']
    paths += [f'results/{sp}/{name}' for sp in ['train','validation'] for name in
              ['candidate_index.csv.gz','all_history_pairs.csv.gz','selected_pairs.csv','command_events.csv.gz','source_receipt.json']]
    checked=verify_manifest_files(SCOPE,SCOPE/'archive_manifest.json',paths,key='files')
    frozen=json.loads((SCOPE/'results/frozen_selection.json').read_text())
    assert frozen['recommendation']=={'regime':'AUTO_stable','coordinate':'differential'}
    background=pd.read_csv(SCOPE/'results/description/background_index.csv')
    descriptions=pd.read_csv(SCOPE/'results/description/selected_pair_index.csv')
    meta_all=[];pair_all=[];members=[]
    for split in ['train','validation']:
        m=pd.read_csv(SCOPE/f'results/{split}/candidate_index.csv.gz').set_index('candidate_id',drop=False)
        p=pd.read_csv(SCOPE/f'results/{split}/all_history_pairs.csv.gz').set_index('pair_id',drop=False)
        selection=pd.read_csv(SCOPE/f'results/{split}/selected_pairs.csv')
        events=pd.read_csv(SCOPE/f'results/{split}/command_events.csv.gz',usecols=['candidate_id','quiet','differential_mean'])
        assert events.candidate_id.tolist()==m.candidate_id.tolist()
        main=selection[(selection.regime=='AUTO_stable')&(selection.coordinate=='differential')]
        assert len(main)==(48 if split=='train' else 1)
        assert main.flight_disjoint.sum()==(17 if split=='train' else 1)
        chosen={}
        def add(cohort,ids):
            for i in ids:
                members.append(dict(split=split,pair_id=int(i),pair_uid=f'{split}:{int(i)}',cohort=cohort))
                chosen[int(i)]=True
        add('main',main.pair_id)
        add('flight_disjoint',main.loc[main.flight_disjoint,'pair_id'])
        for name in ['same_firmware','same_controller','strict']:
            add(name,main.loc[p.loc[main.pair_id,name].to_numpy(),'pair_id'])
        add('same_firmware_controller',main.loc[(p.loc[main.pair_id,'same_firmware']&p.loc[main.pair_id,'same_controller']).to_numpy(),'pair_id'])
        add('small_pair',selection.loc[(selection.regime=='AUTO_stable')&(selection.coordinate=='background_pair'),'pair_id'])
        for cohort in ['both_future_quiet','small_pair_and_both_quiet','differential_vs_quiet']:
            add(cohort,background.loc[(background.split==split)&(background.cohort==cohort),'pair_id'])
        for pid in sorted(chosen):
            r=p.loc[pid].to_dict();a=m.loc[int(r['i'])];b=m.loc[int(r['j'])]
            assert a.split==b.split==split and a.flight_id!=b.flight_id
            assert a.AUTO_stable and b.AUTO_stable
            r.update(split=split,pair_uid=f'{split}:{pid}',window_a=a.window_id,window_b=b.window_id,
                flight_a=a.flight_id,flight_b=b.flight_id,date_a=a.date,date_b=b.date,
                firmware_a=a.firmware,firmware_b=b.firmware,controller_a=a.controller_id,controller_b=b.controller_id)
            pair_all.append(r)
        ids=np.unique(p.loc[sorted(chosen),['i','j']].to_numpy().ravel()).astype(int)
        meta_all.append(m.loc[ids].copy())
        old=descriptions[(descriptions.split==split)&(descriptions.regime=='AUTO_stable')&(descriptions.coordinate=='differential')].set_index('pair_id')
        for pid in main.pair_id:
            assert old.loc[pid,'window_a']==m.loc[p.loc[pid,'i'],'window_id']
            assert old.loc[pid,'window_b']==m.loc[p.loc[pid,'j'],'window_id']
    windows=pd.concat(meta_all).reset_index(drop=True)
    assert windows.window_id.is_unique
    windows['rollout_index']=np.arange(len(windows))
    pairs=pd.DataFrame(pair_all)
    lookup=windows.set_index('window_id').rollout_index.to_dict()
    pairs['a']=pairs.window_a.map(lookup);pairs['b']=pairs.window_b.map(lookup)
    membership=pd.DataFrame(members)
    for (_,cohort),g in membership.groupby(['split','cohort']):
        assert not g.pair_uid.duplicated().any()
    windows.to_csv(out/'windows.csv',index=False)
    pairs.to_csv(out/'pairs.csv',index=False)
    membership.to_csv(out/'cohorts.csv',index=False)
    save(out/'selection_manifest.json',dict(source_sha256=checked,future_outcomes_used=False,
         windows_sha256=sha(out/'windows.csv'),pairs_sha256=sha(out/'pairs.csv'),cohorts_sha256=sha(out/'cohorts.csv'),
         scope='exact saved memberships and main-cohort flag intersections; no new pairing or repacking of subcohorts'))
    return windows,pairs,membership


def data_receipt(data):
    reg=yaml.safe_load((ROOT/'configs/data/velocity_rate_dataset_registry.yaml').read_text())['datasets']['velocity_rate_multistep_v1']
    assert sha(data/'dataset_manifest.json')==reg['manifest_sha256']
    manifest=json.loads((data/'dataset_manifest.json').read_text())
    checks={}
    for name in ['flights.csv','flight_provenance.json','split_manifest.json']+[f'{stem}_{sp}.parquet' for sp in ['train','validation'] for stem in ['samples','windows','native_actuator_motors','native_actuator_servos']]:
        checks[name]=sha(data/name)
        assert checks[name]==manifest['artifact_sha256'][name],name
    return dict(root=str(data),manifest_sha256=reg['manifest_sha256'],artifact_sha256=checks,
                partitions=['train','validation'],test_opened=False,phase_contract=manifest['config']['phase'])


def pack(meta,groups,natives,raw):
    assert set(meta.split)<= {'train','validation'}
    data,truth,truth0,bounds,checks=response.pack_windows(meta.reset_index(drop=True),groups,natives,raw)
    raw_data={k:v.copy() for k,v in data.items()}
    data['events'],data['event_length']=canonicalize_control_events_numpy(data['events'],data['event_length'],data['dt'])
    assert set(data)==set(KEYS)
    return data,truth,truth0,bounds,checks,raw_data


def checkpoint_manifest():
    prior=ROOT/'experiments/control_response_eval_v1/results/run_manifest.json'
    registered=json.loads(prior.read_text())['checkpoints']
    scope=json.loads((SCOPE/'results/description/asset_receipt.json').read_text())['checkpoints']
    allowed={x['registered_sha256'] for x in scope}
    result=[]
    for path,expected in registered.items():
        actual=sha(ROOT/path);assert actual==expected,path
        if '/GRU-Control/' in path:assert expected in allowed
        result.append(dict(path=path,sha256=actual))
    assert len(result)==6
    return result


def model_from(record,device,policy='exact_zoh_merge_us_v1'):
    ck=torch.load(ROOT/record['path'],map_location=device,weights_only=False)
    control=ck['model_name']=='GRU-Control'
    assert ck['config']['history_us']==500000 and ck['config']['horizon_us']==1000000
    assert ck['config']['dataset_id']=='velocity_rate_multistep_v1'
    model=VelocityRateGRU(ck['normalization'],control=control,control_event_policy=policy).to(device).eval()
    model.load_state_dict(ck['model'],strict=True)
    for p in model.parameters():p.requires_grad_(False)
    np.testing.assert_allclose(model.mean.cpu(),ck['normalization']['mean'])
    np.testing.assert_allclose(model.scale.cpu(),ck['normalization']['scale'])
    digest=hashlib.sha256(json.dumps(ck['normalization'],sort_keys=True).encode()).hexdigest()
    return model,ck['model_name'],int(ck['seed']),digest


@torch.inference_mode()
def predict(model,data,device):
    # Explicit allowlist: targets/truth never enter forward or recurrence.
    return response.predict(model,{k:data[k] for k in KEYS},device,256)


def check_preflight(out,groups,natives,raw,checkpoints,device,cfg):
    checks=[]
    oldroot=ROOT/'experiments/control_response_eval_v1'
    digests=verify_manifest_files(oldroot,oldroot/'results/archive_manifest.json',
            ['results/predictions_500ms.npz','results/windows_500ms.parquet'])
    oldmeta=pd.read_parquet(oldroot/'results/windows_500ms.parquet')
    z=np.load(oldroot/'results/predictions_500ms.npz')
    ids=np.unique(np.linspace(0,len(oldmeta)-1,8).astype(int))
    # Fix dimensions to the archived batch caps for bitwise input comparisons.
    subset=oldmeta.iloc[ids].reset_index(drop=True)
    data,truth,t0,bounds,_,rawdata=pack(subset,groups,natives,raw)
    for k in KEYS:
        a=rawdata[k];b=z[k][ids]
        if k=='history' or k=='history_mask':b=b[:,:a.shape[1]]
        if k=='history_dt':b=b[:,:a.shape[1]]
        if k=='events':b=b[:,:a.shape[1],:a.shape[2]]
        if k in ['dt','event_length']:b=b[:,:a.shape[1]]
        np.testing.assert_array_equal(a,b,err_msg=k)
    checks.append(dict(check='archived raw eight-input packing',windows=len(ids),passed=True))
    for record in checkpoints:
        model,name,seed,_=model_from(record,device,LEGACY_CONTROL_EVENTS)
        pred=predict(model,rawdata,device)
        ix=np.abs(rawdata['dt'].cumsum(1)[:,:,None]-z['point_ms'][None,None,:]/1000).argmin(1)
        got=pred[np.arange(len(ids))[:,None],ix]
        delta=float(np.abs(got-z[f'{name}_{seed}'][ids]).max())
        assert delta<=cfg['preflight_atol'],(name,seed,delta)
        checks.append(dict(check='historical raw-interface prediction',model=name,seed=seed,max_abs_error=delta,passed=True))
    canroot=ROOT/'experiments/control_event_canonicalization_v1'
    cm=json.loads((canroot/'archive_manifest.json').read_text())
    # Archive layout is handled explicitly, not inferred from file modification time.
    key='files' if 'files' in cm else 'artifacts'
    digests.update(verify_manifest_files(ROOT,canroot/'archive_manifest.json',
        ['experiments/control_event_canonicalization_v1/results/integrated/windows.parquet',
         'experiments/control_event_canonicalization_v1/results/integrated/predictions.npz'],key))
    cmeta=pd.read_parquet(canroot/'results/integrated/windows.parquet')
    cids=np.unique(np.linspace(0,len(cmeta)-1,6).astype(int))
    small=cmeta.iloc[cids].reset_index(drop=True)
    cd,_,_,_,_,_=pack(small,groups,natives,raw)
    stored=np.load(canroot/'results/integrated/predictions.npz')
    payload={}
    for record in checkpoints:
        model,name,seed,_=model_from(record,device)
        if name!='GRU-Control':continue
        pred=predict(model,cd,device)
        for horizon in [500,1000]:
            ix=small[f'count_{horizon}ms'].to_numpy()-1
            got=pred[np.arange(len(small)),ix]
            err=float(np.abs(got-stored[f'{seed}_{horizon}_canonical'][cids]).max())
            assert err<=cfg['preflight_atol'],err
            checks.append(dict(check='canonical integrated reference',seed=seed,horizon_ms=horizon,max_abs_error=err,passed=True))
        for parts in [2,4]:
            ev,le=canonical_run.refine(cd['events'],cd['event_length'],parts)
            refined=dict(cd,events=ev,event_length=le)
            aa,al=canonicalize_control_events_numpy(ev,le,cd['dt'])
            bb,bl=canonicalize_control_events_numpy(cd['events'],cd['event_length'],cd['dt'])
            np.testing.assert_array_equal(aa,bb);np.testing.assert_array_equal(al,bl)
            pp=predict(model,refined,device)
            np.testing.assert_array_equal(pred,pp)
            checks.append(dict(check='redundant subdivision exact prediction equality',seed=seed,parts=parts,passed=True))
        poisoned=dict(cd,target=np.full_like(pred,np.nan),future_state=np.full_like(pred,1e9))
        np.testing.assert_array_equal(pred,predict(model,poisoned,device))
        # Truncating future step/command tape must not change earlier recurrence.
        cropped=dict(cd,dt=cd['dt'][:,:10],events=cd['events'][:,:10],event_length=cd['event_length'][:,:10])
        prefix=predict(model,cropped,device)
        err=float(np.abs(prefix-pred[:,:10]).max());assert err<=cfg['preflight_atol']
        checks.append(dict(check='no target input and future-prefix consistency',seed=seed,prefix_max_abs_error=err,passed=True))
        payload[f'seed_{seed}']=pred
    np.savez_compressed(out/'preflight_predictions.npz',**payload)
    small.to_csv(out/'preflight_windows.csv',index=False)
    save(out/'preflight.json',dict(passed=True,checks=checks,reference_hashes=digests,
         drive_stage3='2026-10-23 Google Drive get_file_metadata and fetch returned tool not available; remote Stage3 CPU replay not retrieved',
         reference_boundary='Local archived raw response inputs/predictions plus canonicalization full-validation integrated GPU reference; not a claim of bitwise reproduction of the inaccessible Drive Stage3 CPU run',
         tolerance=cfg['preflight_atol']))


@torch.inference_mode()
def simple_predictions(data,cfg,device):
    initial=torch.as_tensor(data['initial'],device=device)
    dt=torch.as_tensor(data['dt'],device=device)
    hold=baseline(initial,dt).cpu().numpy()
    slope=[]
    for i,h in enumerate(data['history']):
        length=int(data['history_length'][i])
        t=np.cumsum(data['history_dt'][i,:length].astype(float));t-=t[-1]
        ix=t>=-cfg['linear_trend_history_us']*1e-6
        x=t[ix];x=x-x.mean();y=h[:length][ix][:,[3,4,5,10,11,12]].astype(float)
        slope.append((x[:,None]*y).sum(0)/(x*x).sum())
    slopes=torch.as_tensor(np.array(slope),device=device,dtype=initial.dtype)
    s=initial;elapsed=torch.zeros(len(initial),device=device);pred=[]
    for k in range(dt.shape[1]):
        elapsed+=dt[:,k]
        vw=initial[:,[3,4,5,10,11,12]]+slopes*elapsed[:,None]
        s=integrate(s,vw[:,:3],vw[:,3:],dt[:,k]);pred.append(s)
    return hold,torch.stack(pred,1).cpu().numpy(),np.array(slope)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=HERE/'results');args=ap.parse_args()
    out=args.output
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((HERE/'config.json').read_text());tick=time.time()
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device(cfg['device'])
    windows,pairs,membership=freeze_selection(out)
    print('fixed cohorts',membership.groupby(['split','cohort']).size().to_dict(),flush=True)
    receipt=data_receipt(args.data_root);checkpoints=checkpoint_manifest()
    save(out/'input_receipt.json',dict(data=receipt,checkpoints=checkpoints,config_sha256=sha(HERE/'config.json'),
         code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'experiments/control_response_eval_v1/evaluate.py',ROOT/'scripts/prepare_velocity_rate_learning.py',ROOT/'src/system_identification/models/velocity_rate_gru.py',ROOT/'src/system_identification/models/control_events.py']}))
    refmeta=pd.read_parquet(ROOT/'experiments/control_response_eval_v1/results/windows_500ms.parquet')
    refcan=pd.read_parquet(ROOT/'experiments/control_event_canonicalization_v1/results/integrated/windows.parquet')
    ids=set(windows.flight_id)|set(refmeta.iloc[np.unique(np.linspace(0,len(refmeta)-1,8).astype(int))].flight_id)|set(refcan.iloc[np.unique(np.linspace(0,len(refcan)-1,6).astype(int))].flight_id)
    groups,natives,raw=response.load_motion(args.data_root,ids)
    check_preflight(out,groups,natives,raw,checkpoints,device,cfg)
    print('preflight passed',flush=True)
    data,truth,truth0,bounds,packing,_=pack(windows,groups,natives,raw)
    packing.to_csv(out/'packing_checks.csv',index=False)
    # Save source observation timestamp offsets separately from model state clock.
    timestamps=[]
    for w in windows.itertuples():
        # Whitelisted timestamp metadata only, never any test file.
        timestamps.append(dict(rollout_index=w.rollout_index,decision_latency_us=w.decision_us-w.initial_measurement_us,
                               actual_horizon_s=w.actual_horizon_s))
    pd.DataFrame(timestamps).to_csv(out/'timing.csv',index=False)
    payload=dict(data,truth=truth,truth_initial=truth0,native_time_us=np.array(bounds)-np.array(bounds)[:,:1],
                 future_count=windows.future_count.to_numpy(),history_start=windows.start_sample.to_numpy(),origin_sample=windows.origin_sample.to_numpy())
    manifests=[]
    for record in checkpoints:
        model,name,seed,norm_sha=model_from(record,device)
        payload[f'{name}_{seed}']=predict(model,data,device)
        assert all(p.grad is None and not p.requires_grad for p in model.parameters())
        manifests.append(dict(model=name,seed=seed,normalization_sha256=norm_sha,checkpoint_sha256=record['sha256'],eval_mode=not model.training))
    hold,linear,slope=simple_predictions(data,cfg,device)
    payload['Hold_0']=hold;payload['LinearTrend_0']=linear;payload['linear_slope']=slope
    np.savez_compressed(out/'native_rollouts.npz',**payload)
    save(out/'run_manifest.json',dict(models=manifests,windows=len(windows),unique_pairs=len(pairs),
         seconds=time.time()-tick,device=str(device),gpu=torch.cuda.get_device_name(device),torch=torch.__version__,
         python=sys.version,test_opened=False,trained=False,swapped=False,input_keys=list(KEYS),
         future_motion_in_forward=False,policy=cfg['control_event_policy'],offline_future_dt=True,
         selection_sha256=sha(out/'selection_manifest.json'),rollout_sha256=sha(out/'native_rollouts.npz')))
    print('done',len(windows),'windows',len(pairs),'pairs',time.time()-tick,flush=True)


if __name__=='__main__':main()
