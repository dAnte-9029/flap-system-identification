"""Frozen cross-flight response evaluation. Cohorts freeze BEFORE loading motion targets."""
from pathlib import Path
import sys,json,time,subprocess
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results'
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'experiments/control_identifiability_audit_v1'))
import audit
from system_identification.data.velocity_rate_dataset import STATE,CONTROL,PHASE,TARGET,file_hash,write_json
from system_identification.models.velocity_rate_gru import VelocityRateGRU
from prepare_velocity_rate_learning import interval_events
INPUT_KEYS=('history','history_mask','history_dt','history_length','initial','dt','events','event_length')
CHANNELS=audit.CHANNELS

def components(pairs):
    parent={x:x for x in set(pairs.flight_a)|set(pairs.flight_b)}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for a,b in zip(pairs.flight_a,pairs.flight_b):parent[find(a)]=find(b)
    labels={x:i for i,x in enumerate(sorted({find(x) for x in parent}))}
    return np.array([labels[find(a)] for a in pairs.flight_a])

def flight_weights(pairs):
    counts=pd.Series(np.r_[pairs.flight_a,pairs.flight_b]).value_counts()
    weights=np.array([1/counts[a]+1/counts[b] for a,b in zip(pairs.flight_a,pairs.flight_b)])
    return weights/weights.sum()

def select_cohorts(cfg):
    source=Path(cfg['audit_source']);archive=json.loads((source/'results/archive_manifest.json').read_text())
    expected={x['path']:x['sha256'] for x in archive['artifacts']};checks={};records=[];metas={};inventory=[]
    for h in cfg['horizons_ms']:
        for name in [f'results/pairs_{h}ms.parquet',f'results/windows_{h}ms.parquet','config.json','audit.py','results/response_evaluation_support.csv']:
            assert file_hash(source/name)==expected[name],name;checks[name]=expected[name]
        p=pd.read_parquet(source/f'results/pairs_{h}ms.parquet');m=pd.read_parquet(source/f'results/windows_{h}ms.parquet');metas[h]=m
        gaps=p[['future_gap_'+c for c in CHANNELS]].to_numpy()
        contrast=p[['contrast_'+c for c in CHANNELS]].to_numpy()
        ratios=gaps>=2*np.maximum(p[['history_gap_'+c for c in CHANNELS]].to_numpy(),p[['initial_u_gap_'+c for c in CHANNELS]].to_numpy())
        masks=dict(signal=p.primary&contrast.any(1),strengthened=p.primary&(contrast&ratios).any(1),
                   strict=p.strict&contrast.any(1),small_control=p.primary&(gaps<=cfg['small_control_rms_limits']).all(1))
        for cohort,mask in masks.items():
            for group in ['main_50hz','sparse_10hz']:
                for split_pair in ['train-train','train-validation','validation-validation']:
                    sub=p[mask&(p.group==group)&(p.split_pair==split_pair)]
                    ids=audit.disjoint_pairs(sub,m)
                    endpoints=np.unique(p.loc[ids,['i','j']].to_numpy().ravel())
                    inventory.append(dict(horizon_ms=h,cohort=cohort,group=group,split_pair=split_pair,
                        raw_pairs=len(sub),nonoverlap_pairs=len(ids),flights=m.iloc[endpoints].flight_id.nunique(),
                        inference_requested=group=='main_50hz' and split_pair!='train-validation'))
                    if group!='main_50hz' or split_pair=='train-validation':continue
                    flight_only=set(audit.disjoint_pairs(p.loc[ids],m,True))
                    for idx in ids:
                        r=p.loc[idx];a,b=m.iloc[int(r.i)],m.iloc[int(r.j)]
                        rec=r.to_dict();rec.update(pair_index=int(idx),pair_uid=f'{h}:{idx}',horizon_ms=h,cohort=cohort,
                            split=a.split,flight_a=a.flight_id,flight_b=b.flight_id,window_a=a.window_id,window_b=b.window_id,
                            flight_disjoint=idx in flight_only)
                        records.append(rec)
    pairs=pd.DataFrame(records);pairs[['i','j','pair_index','horizon_ms']]=pairs[['i','j','pair_index','horizon_ms']].astype(int)
    for _,g in pairs.groupby(['horizon_ms','cohort','split']):
        pairs.loc[g.index,'component']=components(g);pairs.loc[g.index,'flight_weight']=flight_weights(g)
    pairs.to_parquet(OUT/'cohorts.parquet',index=False);pd.DataFrame(inventory).to_csv(OUT/'source_inventory.csv',index=False)
    # Cross-check the previously archived main-group validation primary counts.
    assert pairs.query('cohort=="signal" and split=="validation"').groupby('horizon_ms').size().to_dict()=={500:21,1000:24}
    write_json(OUT/'selection_manifest.json',dict(audit_files_sha256=checks,cohort_sha256=file_hash(OUT/'cohorts.parquet'),
        selection_inputs='archived historical match errors and command differences only',future_outcomes_used=False,
        rules='original audit calipers and disjoint_pairs reused; no threshold relaxation'))
    return pairs,metas

def splice_tape(own,donor,decision,donor_decision,horizon,mode='swap'):
    """Publication-time shift without time stretching. Self swap preserves every original event."""
    ot,ou=own;dt,du=donor;shift=decision-donor_decision
    donor_times=dt[(dt>donor_decision)&(dt<donor_decision+horizon)]+shift
    if mode=='refine':
        times=np.unique(np.r_[ot,donor_times]);values=audit.zoh(ot,ou,times)
        return times,values
    prefix=ot[ot<decision];start_u=audit.zoh(dt,du,np.array([donor_decision]))[0]
    previous=audit.zoh(ot,ou,np.array([decision-1]))[0]
    # Insert a splice event only if the command changes or an original donor event occurs at zero.
    insert=not np.array_equal(start_u,previous) or np.any(dt==donor_decision)
    times=np.r_[prefix,([decision] if insert else []),donor_times].astype(np.int64)
    values=np.vstack([audit.zoh(ot,ou,prefix),start_u[None] if insert else np.empty((0,4)),audit.zoh(dt,du,donor_times-shift)])
    return times,values

def event_pack(t,u,bounds):
    parts=[]
    for a,b in zip(bounds[:-1],bounds[1:]):
        if b==a:parts.append(np.zeros((1,5),np.float32));continue
        times=np.r_[a,t[(t>a)&(t<b)]]
        values=audit.zoh(t,u,times)
        duration=np.diff(np.r_[times,b])*1e-6
        assert (duration>0).all() and np.isfinite(values).all()
        assert np.all((values[:,0]>=0)&(values[:,0]<=1)) and np.all(np.abs(values[:,1:])<=1)
        parts.append(np.c_[values,duration].astype(np.float32))
    return parts

def load_motion(root,ids):
    groups={};natives={};raw={}
    cols=list(dict.fromkeys(STATE+CONTROL+PHASE+TARGET+['flight_id','split','state_measurement_us','state_publication_us',
         'segment_id','valid_core','valid_control','valid_encoder','valid_frequency','support_start_us','support_end_us']))
    for split in ['train','validation']:
        s=pd.read_parquet(root/f'samples_{split}.parquet',columns=cols);assert set(s.split)=={split}
        for f,g in s[s.flight_id.isin(ids)].groupby('flight_id',sort=False):groups[f]=g.reset_index(drop=True)
        m=pd.read_parquet(root/f'native_actuator_motors_{split}.parquet');v=pd.read_parquet(root/f'native_actuator_servos_{split}.parquet')
        md=dict(tuple(m[m.flight_id.isin(ids)].groupby('flight_id')));vd=dict(tuple(v[v.flight_id.isin(ids)].groupby('flight_id')))
        for f in md:
            t,u,_=audit.tape(md[f],vd[f]);natives[f]=(t,u);raw[f]=(md[f],vd[f])
    return groups,natives,raw

def pack_windows(meta,groups,natives,raw):
    n=len(meta);H=int(meta.history_count.max());K=int(meta.future_count.max())
    data=dict(history=np.zeros((n,H,20),np.float32),history_mask=np.zeros((n,H,20),bool),history_dt=np.zeros((n,H),np.float32),
              history_length=meta.history_count.to_numpy(np.int64),initial=np.zeros((n,13),np.float32),dt=np.zeros((n,K),np.float32))
    truth=np.zeros((n,K,13),np.float64);truth0=np.zeros((n,13));parts=[];bounds=[];checks=[]
    for i,m in meta.iterrows():
        s=groups[m.flight_id];a,o,e=(int(m[k]) for k in ['start_sample','origin_sample','end_sample'])
        segment=s.iloc[a:e+1];assert segment.valid_core.all() and segment.segment_id.nunique()==1
        assert segment.state_publication_us.iloc[o-a]==m.decision_us
        hist=s.iloc[a:o+1];fut=s.iloc[o+1:e+1];lh=len(hist);lf=len(fut)
        values=hist[STATE+CONTROL+PHASE].to_numpy();mask=np.isfinite(values)
        mask[:,13:17] &= hist.valid_control.to_numpy()[:,None]
        mask[:,17:19] &= hist.valid_encoder.to_numpy()[:,None];mask[:,19] &= hist.valid_frequency.to_numpy()
        h=np.where(mask,values,0);initial=h[-1,:13].copy();p0=initial[:3].copy();h[:,:3]-=p0;initial[:3]=0
        data['history'][i,:lh]=h;data['history'][i,lh:]=h[-1];data['history_mask'][i,:lh]=mask
        data['history_dt'][i,1:lh]=np.diff(hist.state_measurement_us)*1e-6;data['initial'][i]=initial
        ts=s.state_measurement_us.to_numpy()[o:e+1];pad=np.pad(ts,(0,K-lf),mode='edge');bounds.append(pad)
        data['dt'][i]=np.diff(pad)*1e-6
        target=fut[TARGET].to_numpy();target[:,:3]-=p0;truth[i,:lf]=target;truth[i,lf:]=target[-1]
        truth0[i]=s.iloc[o][TARGET].to_numpy(float);truth0[i,:3]-=p0
        t,u=natives[m.flight_id];events=event_pack(t,u,pad);parts.append(events)
        # Compare new packer's ordered events with the unchanged original inference packer.
        ref,length=interval_events(*raw[m.flight_id],ts[:-1],ts[1:])
        for k,x in enumerate(events[:lf]):np.testing.assert_allclose(x,ref[k,:length[k]],atol=1e-7)
        checks.append(dict(window_id=m.window_id,initial_rate_alignment_gap=float(np.linalg.norm(truth0[i,10:]-initial[10:])),
            decision_latency_us=int(m.decision_us-ts[0]),actual_horizon_s=(ts[-1]-ts[0])*1e-6))
    data.update(pad_events(parts,K))
    return data,truth,truth0,bounds,pd.DataFrame(checks)

def pad_events(parts,K):
    width=max(len(x) for seq in parts for x in seq)
    events=np.zeros((len(parts),K,width,5),np.float32);length=np.ones((len(parts),K),np.int64)
    for i,seq in enumerate(parts):
        for k,x in enumerate(seq):events[i,k,:len(x)]=x;length[i,k]=len(x)
    return dict(events=events,event_length=length)

def rotation(q):
    return Rotation.from_quat(np.asarray(q)[...,[1,2,3,0]].reshape(-1,4))

def increments(states,initial):
    """Vectors p/v/w plus spatial NED relative rotation matrices; own initial per trajectory."""
    x=np.asarray(states,dtype=float);s=np.asarray(initial,dtype=float)
    n,k=x.shape[:2]
    r=rotation(x[:,:,6:10]).as_matrix().reshape(n,k,3,3)
    r0=rotation(s[:,6:10]).as_matrix()
    return dict(position=x[:,:,:3]-s[:,None,:3],velocity=x[:,:,3:6]-s[:,None,3:6],
                rate=x[:,:,10:13]-s[:,None,10:13],attitude=r@r0[:,None].transpose(0,1,3,2))

def contrast(ia,ib,metric):
    if metric!='attitude':return ia[metric]-ib[metric]
    matrix=ia[metric]@ib[metric].swapaxes(-1,-2)
    return Rotation.from_matrix(matrix.reshape(-1,3,3)).as_rotvec().reshape(matrix.shape[:-2]+(3,))*180/np.pi

def response_error(pred,true,metric):
    if metric!='attitude':return np.linalg.norm(pred-true,axis=-1)
    shape=pred.shape[:-1];r=Rotation.from_rotvec(np.deg2rad(pred).reshape(-1,3));t=Rotation.from_rotvec(np.deg2rad(true).reshape(-1,3))
    return np.rad2deg((r.inv()*t).magnitude()).reshape(shape)

@torch.no_grad()
def predict(model,data,device,batch_size):
    out=[]
    for a in range(0,len(data['initial']),batch_size):
        inputs={k:torch.as_tensor(data[k][a:a+batch_size],device=device) for k in INPUT_KEYS}
        out.append(model(inputs).cpu().numpy())
    p=np.concatenate(out);assert np.isfinite(p).all();return p

def main():
    tick=time.time();cfg=json.loads((HERE/'config.json').read_text());OUT.mkdir(exist_ok=True)
    acfg=json.loads((ROOT/'experiments/control_identifiability_audit_v1/config.json').read_text())
    root,flights,provenance,data_manifest=audit.resolve(acfg)
    pairs,metas=select_cohorts(cfg);print(pairs.groupby(['horizon_ms','cohort','split']).size(),flush=True)
    # Selection is committed to disk before target arrays are opened.
    frozen_selection=file_hash(OUT/'cohorts.parquet')
    ids=set(pairs.flight_a)|set(pairs.flight_b);groups,natives,raw=load_motion(root,ids)
    device=torch.device(cfg['device']);assert torch.cuda.is_available();torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    checkpoints={};runtime=[];timing=[];sanity=[];records=[]
    for horizon in cfg['horizons_ms']:
        pairs_h=pairs[pairs.horizon_ms==horizon].drop_duplicates('pair_index').sort_values('pair_index').reset_index(drop=True)
        m=metas[horizon];endpoints=np.unique(pairs_h[['i','j']].to_numpy().ravel());meta=m.iloc[endpoints].copy().reset_index().rename(columns={'index':'audit_window_index'})
        lookup={i:k for k,i in enumerate(endpoints)};a=np.array([lookup[i] for i in pairs_h.i]);b=np.array([lookup[i] for i in pairs_h.j])
        data,truth,truth0,bounds,check=pack_windows(meta,groups,natives,raw)
        check['horizon_ms']=horizon;timing.append(check);meta.to_parquet(OUT/f'windows_{horizon}ms.parquet',index=False)
        # Two recipient histories per pair, each with donor commands from the other.
        recipient=np.r_[a,b];donor=np.r_[b,a];swap_parts=[];refine_parts=[];common_parts=[];common_swap_parts=[]
        for i,j in zip(recipient,donor):
            mi,mj=meta.iloc[i],meta.iloc[j];own=natives[mi.flight_id];other=natives[mj.flight_id]
            swapped=splice_tape(own,other,int(mi.decision_us),int(mj.decision_us),horizon*1000)
            refined=splice_tape(own,other,int(mi.decision_us),int(mj.decision_us),horizon*1000,mode='refine')
            swap_parts.append(event_pack(*swapped,bounds[i]));refine_parts.append(event_pack(*refined,bounds[i]))
            # Equal continuous controls must not be confused with extra event tokens.
            # Common union subdivision lets both command choices have identical encoder timing.
            common_t=np.unique(np.r_[own[0],swapped[0]])
            common_parts.append(event_pack(common_t,audit.zoh(*own,common_t),bounds[i]))
            common_swap_parts.append(event_pack(common_t,audit.zoh(*swapped,common_t),bounds[i]))
            selftape=splice_tape(own,own,int(mi.decision_us),int(mi.decision_us),horizon*1000)
            # Self swap is exact through H; padding after H is assessed separately below.
            selfevents=event_pack(*selftape,bounds[i]);own_events=event_pack(*own,bounds[i])
            for k,(x,y) in enumerate(zip(selfevents,own_events)):
                if bounds[i][k+1]<=mi.decision_us+horizon*1000:np.testing.assert_array_equal(x,y)
            records.append(dict(horizon_ms=horizon,recipient_window=mi.window_id,donor_window=mj.window_id,
                mapped_donor_end_us=int(mi.decision_us+horizon*1000),recipient_end_us=int(bounds[i][-1]),
                donor_tail_hold_us=max(0,int(bounds[i][-1]-mi.decision_us-horizon*1000))))
        swapped_data={k:v[recipient].copy() for k,v in data.items()};swapped_data.update(pad_events(swap_parts,data['dt'].shape[1]))
        refined_data={k:v[recipient].copy() for k,v in data.items()};refined_data.update(pad_events(refine_parts,data['dt'].shape[1]))
        common_data={k:v[recipient].copy() for k,v in data.items()};common_data.update(pad_events(common_parts,data['dt'].shape[1]))
        common_swap_data={k:v[recipient].copy() for k,v in data.items()};common_swap_data.update(pad_events(common_swap_parts,data['dt'].shape[1]))
        np.testing.assert_array_equal(common_data['event_length'],common_swap_data['event_length'])
        np.testing.assert_array_equal(common_data['events'][...,4],common_swap_data['events'][...,4])
        times=data['dt'].cumsum(1);point=np.array([t for t in cfg['time_points_ms'] if t<=horizon]);ix=np.abs(times[:,:,None]-point[None,None,:]/1000).argmin(1)
        # Padded dt gives repeated final times; argmin selects the first true endpoint.
        target_at=truth[np.arange(len(meta))[:,None],ix];actual_times=times[np.arange(len(meta))[:,None],ix]
        truth_inc=increments(target_at,truth0)
        payload=dict(pair_index=pairs_h.pair_index.to_numpy(),i=a,j=b,point_ms=point,truth=target_at,
                     initial=data['initial'],truth_initial=truth0,actual_time_s=actual_times,recipient=recipient,
                     history=data['history'],history_mask=data['history_mask'],history_dt=data['history_dt'],history_length=data['history_length'],
                     dt=data['dt'],events=data['events'],event_length=data['event_length'],native_truth=truth,
                     swap_events=swapped_data['events'],swap_event_length=swapped_data['event_length'])
        payload.update(common_events=common_data['events'],common_swap_events=common_swap_data['events'],
                       common_event_length=common_data['event_length'])
        for model_name in ['GRU-NoControl','GRU-Control']:
            for seed in cfg['seeds']:
                cp=ROOT/cfg['model_root']/model_name/f'seed_{seed}/best.pt';expected=json.loads((cp.parent/'summary.json').read_text())['checkpoint_sha256']
                assert file_hash(cp)==expected;checkpoints[str(cp.relative_to(ROOT))]=expected
                ck=torch.load(cp,map_location=device,weights_only=False)
                model=VelocityRateGRU(ck['normalization'],control=model_name=='GRU-Control').to(device);model.load_state_dict(ck['model']);model.eval()
                start=time.time();torch.cuda.reset_peak_memory_stats(device)
                pred=predict(model,data,device,cfg['batch_size']);sw=predict(model,swapped_data,device,cfg['batch_size']);ref=predict(model,refined_data,device,cfg['batch_size'])
                cpred=predict(model,common_data,device,cfg['batch_size']);cswap=predict(model,common_swap_data,device,cfg['batch_size'])
                if model_name=='GRU-NoControl':np.testing.assert_allclose(sw,pred[recipient],atol=2e-6,rtol=2e-6)
                # Explicit target poisoning / future-measurement independence: model only sees INPUT_KEYS.
                probe={k:v[:2].copy() for k,v in data.items()};probe['target']=np.full_like(truth[:2],np.nan)
                np.testing.assert_allclose(predict(model,probe,device,2),pred[:2],atol=2e-5,rtol=2e-5)
                maxnorm=float(np.max(np.abs(np.linalg.norm(pred[:,:,6:10],axis=-1)-1)))
                key=f'{model_name}_{seed}';pa=pred[np.arange(len(meta))[:,None],ix]
                sa=sw[np.arange(len(recipient))[:,None],ix[recipient]];ra=ref[np.arange(len(recipient))[:,None],ix[recipient]]
                payload[key]=pa;payload[key+'_swapped']=sa;payload[key+'_refined']=ra
                payload[key+'_common']=cpred[np.arange(len(recipient))[:,None],ix[recipient]]
                payload[key+'_common_swapped']=cswap[np.arange(len(recipient))[:,None],ix[recipient]]
                runtime.append(dict(horizon_ms=horizon,model=model_name,seed=seed,seconds=time.time()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(device),parameters=sum(v.numel() for v in model.parameters())))
                sanity.append(dict(horizon_ms=horizon,model=model_name,seed=seed,finite=True,quaternion_norm_max_error=maxnorm,
                    no_target_access=True,no_control_swap_max=float(np.max(np.abs(sw-pred[recipient]))) if not model.control else None))
                print('evaluated',horizon,key,round(time.time()-start,2),flush=True)
        np.savez_compressed(OUT/f'predictions_{horizon}ms.npz',**payload)
    pd.concat(timing).to_csv(OUT/'window_timing.csv',index=False);pd.DataFrame(records).to_csv(OUT/'swap_timing.csv',index=False)
    pd.DataFrame(runtime).to_csv(OUT/'runtime.csv',index=False)
    assert file_hash(OUT/'cohorts.parquet')==frozen_selection
    write_json(OUT/'run_manifest.json',dict(config=cfg,data=data_manifest,checkpoints=checkpoints,numerical_checks=sanity,
        device=torch.cuda.get_device_name(device),torch_version=torch.__version__,python=sys.executable,
        code_sha256=file_hash(Path(__file__)),model_code_sha256=file_hash(ROOT/'src/system_identification/models/velocity_rate_gru.py'),
        config_sha256=file_hash(HERE/'config.json'),selection_sha256=frozen_selection,elapsed_seconds=time.time()-tick,
        test_opened=False,retrained=False,phase_policy='unchanged last historical valid frequency hold and phase extrapolation',
        target_initial_policy='increments subtract own native aligned target initial; predictions subtract own causal input initial'))
    print('COMPLETE',time.time()-tick,flush=True)

if __name__=='__main__':main()
