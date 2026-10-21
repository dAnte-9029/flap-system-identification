"""Small frozen-model event segmentation diagnosis; production inference is unchanged."""
from pathlib import Path
import sys,json,time
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results'
sys.path.insert(0,str(ROOT/'experiments/control_response_eval_v1'))
from evaluate import INPUT_KEYS,VelocityRateGRU,predict,pad_events,file_hash,write_json

# Canonical integer-microsecond intervals: tuples (command[4], duration_us).
def intervals(events,length):
    return [(row[:4].copy(),int(round(float(row[4])*1e6))) for row in events[:int(length)]]

def merge(seq):
    out=[]
    for u,d in seq:
        assert d>0
        if out and np.array_equal(out[-1][0],u):out[-1]=(u,out[-1][1]+d)
        else:out.append((u.copy(),d))
    return out

def split(seq,n):
    out=[]
    for u,d in seq:
        bounds=np.linspace(0,d,min(n,d)+1).astype(np.int64)
        out.extend((u.copy(),int(x)) for x in np.diff(bounds))
    return out

def at(seq,times):
    ends=np.cumsum([d for u,d in seq]);return np.stack([u for u,d in seq])[np.searchsorted(ends,times,side='right')]

def equal_function(a,b):
    ea=np.cumsum([d for u,d in a]);eb=np.cumsum([d for u,d in b]);assert ea[-1]==eb[-1]
    # Checking each open constant cell of the exact union partition proves equality everywhere on [0,T).
    starts=np.unique(np.r_[0,ea[:-1],eb[:-1]])
    np.testing.assert_array_equal(at(a,starts),at(b,starts));return len(starts)

def fixed_grid(seq,grid_us):
    seq=merge(seq);end=np.cumsum([d for u,d in seq]);total=int(end[-1])
    # Retain every true discontinuity: plain 50 Hz ZOH would change the physical command.
    starts=np.unique(np.r_[0,np.arange(grid_us,total,grid_us),end[:-1]])
    return [(u.copy(),int(d)) for u,d in zip(at(seq,starts),np.diff(np.r_[starts,total]))]

def remove_splice_only_boundary(common,original,decision_offset):
    """Prior refine added donor events strictly after decision; common also inserted the splice."""
    out=[];elapsed=0
    for a,b in zip(common,original):
        starts=np.r_[0,np.cumsum([d for u,d in a])[:-1]]+elapsed
        own=np.r_[0,np.cumsum([d for u,d in b])[:-1]]+elapsed
        seq=[]
        for start,(u,d) in zip(starts,a):
            if start==decision_offset and start not in own and seq:
                assert np.array_equal(seq[-1][0],u);seq[-1]=(u,seq[-1][1]+d)
            else:seq.append((u.copy(),d))
        out.append(seq);elapsed+=sum(d for u,d in a)
    return out

def pack(base,sequences):
    out={k:v.copy() for k,v in base.items() if k not in ['events','event_length']}
    parts=[np.c_[np.stack([u for u,d in seq]),np.array([d for u,d in seq])*1e-6].astype(np.float32) for seq in sequences]
    out.update(pad_events([parts],len(parts)));return out

def state_errors(a,b):
    a=a.astype(float);b=b.astype(float)
    qa=a[...,6:10];qb=b[...,6:10];shape=qa.shape[:-1]
    ra=Rotation.from_quat(qa.reshape(-1,4)[:,[1,2,3,0]]);rb=Rotation.from_quat(qb.reshape(-1,4)[:,[1,2,3,0]])
    return dict(position=np.linalg.norm(a[...,:3]-b[...,:3],axis=-1),velocity=np.linalg.norm(a[...,3:6]-b[...,3:6],axis=-1),
        rate=np.linalg.norm(a[...,10:13]-b[...,10:13],axis=-1),attitude_deg=np.rad2deg((ra.inv()*rb).magnitude()).reshape(shape))

@torch.no_grad()
def traced(model,data,device):
    trace={};step_in=[];step_hidden=[];phase=[];features=[]
    def hist_hook(module,args,out):trace['history_hidden']=out[0][:,int(data['history_length'][0])-1].detach().cpu().numpy()
    def ev_hook(module,args,out):
        lengths=torch.as_tensor(data['event_length'].reshape(-1),device=device)
        trace['event_encoded']=out[0][torch.arange(len(lengths),device=device),lengths-1].detach().cpu().numpy()
        trace['event_gru_input']=args[0].detach().cpu().numpy()
    def step_pre(module,args):step_in.append(args[1].detach().cpu().numpy())
    def step_hook(module,args,out):step_hidden.append(out.detach().cpu().numpy())
    hooks=[model.history_gru.register_forward_hook(hist_hook),model.event_gru.register_forward_hook(ev_hook),
           model.step_gru.register_forward_pre_hook(step_pre),model.step_gru.register_forward_hook(step_hook)]
    original=model.features
    def capture(state,flap,mask,dt):
        out=original(state,flap,mask,dt)
        if state.ndim==2:
            phase.append(torch.cat([flap,mask,dt[:,None]],-1).detach().cpu().numpy());features.append(out.detach().cpu().numpy())
        return out
    model.features=capture
    try:pred=predict(model,data,device,1)
    finally:
        model.features=original
        for hook in hooks:hook.remove()
    trace.update(step_hidden_in=np.concatenate(step_in),step_hidden_out=np.concatenate(step_hidden),phase=np.concatenate(phase),features=np.concatenate(features))
    return pred,trace

def select(source):
    pairs=pd.read_parquet(source/'results/cohorts.parquet').query('horizon_ms==1000 and cohort=="signal" and split=="validation"').copy()
    cases=json.loads((source/'results/cases.json').read_text());ids=[x['pair_uid'] for x in cases if x['horizon_ms']==1000]
    sw=pd.read_parquet(source/'results/swap_metrics.parquet').query('horizon_ms==1000 and point_ms==1000 and cohort=="signal" and split=="validation" and metric=="velocity" and model=="GRU-Control"')
    rank=sw.groupby('pair_uid').timing_refinement_squared_norm.mean().sort_values(ascending=False)
    ids.append(next(x for x in rank.index if x not in ids))
    remaining=pairs[~pairs.pair_uid.isin(ids)].sort_values(['score','pair_index']);ids.append(remaining.iloc[len(remaining)//2].pair_uid)
    chosen=pairs.set_index('pair_uid').loc[ids].reset_index();assert len(chosen)==4
    return chosen

def main():
    tick=time.time();cfg=json.loads((HERE/'config.json').read_text());source=ROOT/cfg['source'];OUT.mkdir(exist_ok=True)
    archive=json.loads((source/'results/archive_manifest.json').read_text());expected={x['path']:x['sha256'] for x in archive['artifacts']}
    names=['results/predictions_1000ms.npz','results/cohorts.parquet','results/windows_1000ms.parquet','results/cases.json','results/swap_metrics.parquet','results/run_manifest.json','evaluate.py']
    for name in names:assert file_hash(source/name)==expected[name],name
    rm=json.loads((source/'results/run_manifest.json').read_text());modelpath=ROOT/'src/system_identification/models/velocity_rate_gru.py'
    assert file_hash(modelpath)==rm['model_code_sha256']
    selected=select(source);selected.to_csv(OUT/'selected_pairs.csv',index=False)
    z=np.load(source/'results/predictions_1000ms.npz');meta=pd.read_parquet(source/'results/windows_1000ms.parquet')
    cfgsource=json.loads((source/'config.json').read_text());device=torch.device(cfg['device']);assert torch.cuda.is_available()
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows=[];layers=[];windows=[];equivalence=[];payload={};checkpoints={};Kpair=len(z['i'])
    for pair in selected.itertuples():
        pi=int(np.flatnonzero(z['pair_index']==pair.pair_index)[0])
        for side,idx,recipient in [('A',int(z['i'][pi]),pi),('B',int(z['j'][pi]),pi+Kpair)]:
            uid=f'{pair.pair_uid}_{side}';m=meta.iloc[idx]
            assert m.split=='validation' and m.sampling_group=='main_50hz'
            K=int(m.future_count);H=int(m.history_count)
            base={k:z[k][idx:idx+1].copy() for k in INPUT_KEYS};base['dt']=base['dt'][:,:K]
            for k in ['history','history_mask','history_dt']:base[k]=base[k][:,:H]
            orig=[intervals(z['events'][idx,k],z['event_length'][idx,k]) for k in range(K)]
            common=[intervals(z['common_events'][recipient,k],z['common_event_length'][recipient,k]) for k in range(K)]
            donor=[intervals(z['swap_events'][recipient,k],z['swap_event_length'][recipient,k]) for k in range(K)]
            sequences={'native':orig,'split2':[split(s,2) for s in orig],'split4':[split(s,4) for s in orig],
                'merged':[merge(s) for s in orig],'prior_common':common,
                'prior_refined':remove_splice_only_boundary(common,orig,int(m.decision_us-m.initial_measurement_us)), 'donor':donor,
                'fixed_native':[fixed_grid(s,cfg['grid_us']) for s in orig],
                'fixed_split4':[fixed_grid(split(s,4),cfg['grid_us']) for s in orig],
                'fixed_donor':[fixed_grid(s,cfg['grid_us']) for s in donor]}
            data={name:pack(base,seq) for name,seq in sequences.items()}
            # Padding-only control: more computed recurrent positions, same valid lengths/tokens.
            data['pad_only']={k:v.copy() for k,v in data['native'].items()}
            width=data['split4']['events'].shape[2]
            data['pad_only']['events']=np.pad(data['native']['events'],((0,0),(0,0),(0,width-data['native']['events'].shape[2]),(0,0)))
            for k in range(K):
                n=int(z['event_length'][idx,k]);np.testing.assert_array_equal(data['native']['events'][0,k,:n],z['events'][idx,k,:n])
            for k in INPUT_KEYS:np.testing.assert_array_equal(data['fixed_native'][k],data['fixed_split4'][k])
            for variant,seq in sequences.items():
                if variant in ['donor','fixed_donor']:continue
                cells=0;maxmean=0.
                for a,b in zip(orig,seq):
                    cells+=equal_function(a,b)
                    ua=sum(u.astype(float)*d for u,d in a)/sum(d for u,d in a)
                    ub=sum(u.astype(float)*d for u,d in b)/sum(d for u,d in b)
                    maxmean=max(maxmean,float(np.abs(ua-ub).max()))
                    np.testing.assert_array_equal(a[0][0],b[0][0]);np.testing.assert_allclose(ua,ub,atol=1e-14,rtol=0)
                for key in INPUT_KEYS:
                    if key not in ['events','event_length']:np.testing.assert_array_equal(data['native'][key],data[variant][key])
                equivalence.append(dict(window=uid,variant=variant,union_cells=cells,exact_zoh_equivalent=True,control_50hz_mean_max_difference=maxmean,
                    valid_event_updates=sum(len(s) for s in seq),event_gru_padded_updates=K*data[variant]['events'].shape[2],state_updates=K,
                    first_changed_event_step=next((j+1 for j,(a,b) in enumerate(zip(orig,seq)) if len(a)!=len(b) or any(da!=db or not np.array_equal(ua,ub) for (ua,da),(ub,db) in zip(a,b))),None)))
            equivalence.append(dict(window=uid,variant='pad_only',union_cells=sum(len(s) for s in orig),exact_zoh_equivalent=True,
                control_50hz_mean_max_difference=0.,valid_event_updates=sum(len(s) for s in orig),event_gru_padded_updates=K*width,state_updates=K,first_changed_event_step=None))
            times=np.cumsum(data['native']['dt'][0]);windows.append(dict(window=uid,window_id=m.window_id,flight_id=m.flight_id,source_index=idx,source_recipient=recipient,pair_uid=pair.pair_uid,side=side,steps=K,history_count=H))
            for variant,d in data.items():
                for key,value in d.items():payload[f'{uid}__{variant}__input_{key}']=value
            for seed in cfg['seeds']:
                path=ROOT/cfgsource['model_root']/'GRU-Control'/f'seed_{seed}/best.pt';key=str(path.relative_to(ROOT))
                assert file_hash(path)==rm['checkpoints'][key];checkpoints[key]=file_hash(path)
                ck=torch.load(path,map_location=device,weights_only=False);model=VelocityRateGRU(ck['normalization'],control=True).to(device);model.load_state_dict(ck['model']);model.eval()
                predictions={};traces={}
                for variant,d in data.items():predictions[variant],traces[variant]=traced(model,d,device)
                predictions['repeat'],traces['repeat']=traced(model,data['native'],device)
                plain=predict(model,data['native'],device,1);np.testing.assert_array_equal(plain,predictions['native'])
                # Same native representation reproduces the previously archived model path at report points.
                ix=np.abs(times[:,None]-z['point_ms'][None]/1000).argmin(0)
                archived=z[f'GRU-Control_{seed}'][idx]
                np.testing.assert_allclose(predictions['native'][0,ix],archived,atol=2e-5,rtol=2e-5)
                for variant,suffix in [('prior_common','common'),('prior_refined','refined'),('donor','swapped')]:
                    np.testing.assert_allclose(predictions[variant][0,ix],z[f'GRU-Control_{seed}_{suffix}'][recipient],atol=2e-5,rtol=2e-5)
                for variant,pred in predictions.items():
                    payload[f'{uid}__{seed}__{variant}__pred']=pred
                    for key,value in traces[variant].items():
                        if key!='event_gru_input':payload[f'{uid}__{seed}__{variant}__{key}']=value
                    reference='fixed_native' if variant in ['fixed_split4','fixed_donor'] else 'native'
                    ref=predictions[reference];err=state_errors(pred,ref);tr=traces[variant];rt=traces[reference]
                    for key in ['history_hidden','phase']:np.testing.assert_array_equal(tr[key],rt[key])
                    for layer in ['event_encoded','step_hidden_in','step_hidden_out','features']:
                        delta=np.max(np.abs(tr[layer]-rt[layer]),axis=-1);changed=np.flatnonzero(delta>1e-7)
                        layers.append(dict(window=uid,seed=seed,variant=variant,reference=reference,layer=layer,max_abs=float(delta.max()),first_changed_step=int(changed[0]+1) if len(changed) else None))
                    for metric,value in err.items():
                        changed=np.flatnonzero(value[0]>cfg['state_difference_thresholds'][metric])
                        for h in cfg['horizons_ms']:
                            k=int(np.argmin(abs(times-h/1000)))
                            rows.append(dict(window=uid,flight_id=m.flight_id,seed=seed,variant=variant,reference=reference,metric=metric,horizon_ms=h,
                                actual_time_s=float(times[k]),difference=float(value[0,k]),max_through_horizon=float(value[0,:k+1].max()),
                                first_changed_step=int(changed[0]+1) if len(changed) else None))
                print('done',uid,seed,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False);pd.DataFrame(layers).to_csv(OUT/'layers.csv',index=False)
    pd.DataFrame(windows).to_csv(OUT/'windows.csv',index=False);pd.DataFrame(equivalence).to_csv(OUT/'equivalence.csv',index=False)
    np.savez_compressed(OUT/'traces.npz',**payload)
    write_json(OUT/'manifest.json',dict(config=cfg,source_hashes={k:expected[k] for k in names},dataset=rm['data'],checkpoints=checkpoints,
        model_code_sha256=file_hash(modelpath),diagnostic_code_sha256=file_hash(Path(__file__)),device=torch.cuda.get_device_name(device),
        torch=torch.__version__,python=sys.executable,elapsed_seconds=time.time()-tick,test_loaded=False,retrained=False,
        data_scope='only 8 selected main_50hz validation histories used for inference; source archived NPZ also contains unused training rows; no target arrays accessed',
        exact_equivalence='integer microsecond union partition; [0,T) half-open semantics; all command values exact; float32 durations unchanged physical microseconds'))
    print('COMPLETE',time.time()-tick,flush=True)

if __name__=='__main__':main()
