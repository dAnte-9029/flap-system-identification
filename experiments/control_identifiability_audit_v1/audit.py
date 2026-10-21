"""Read-only, train/validation-only command coverage and history matching audit.
No target_* column is read. Future motion is neither loaded nor scored.
"""
from pathlib import Path
import sys, json, hashlib, time, subprocess, bisect
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CHANNELS = ['motor', 'left_elevon', 'right_elevon', 'rudder']
V = ['velocity_ned_m_s_'+x for x in 'xyz']
Q = ['attitude_q_'+x for x in 'wxyz']
W = ['angular_velocity_body_rad_s_'+x for x in 'xyz']
U = ['control_flap_motor_normalized','control_left_elevon_normalized',
     'control_right_elevon_normalized','control_rudder_normalized']
META = ['flight_id','split','sampling_group','sample_in_flight','state_measurement_us',
        'state_publication_us','segment_id','valid_core','excluded','review','valid_frequency',
        'valid_encoder','valid_status','nav_state','failsafe','flap_frequency_hz',
        'support_start_us','support_end_us','firmware_verified','provenance_review',
        'attitude_publication_us','rate_publication_us','motor_publication_us',
        'servo_publication_us','frequency_publication_us','encoder_publication_us']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,
        default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')

def qstats(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return dict(zip(['min','p01','p10','p50','p90','p99','max'],np.quantile(x,[0,.01,.1,.5,.9,.99,1]).tolist())) if len(x) else {}

def permitted(split):
    if split not in ('train','validation'): raise ValueError('test is sealed')

def load_parquet(root, stem, split, columns=None):
    permitted(split)
    if columns and any(c.startswith('target_') for c in columns): raise ValueError('future outcome forbidden')
    d=pd.read_parquet(root/f'{stem}_{split}.parquet',columns=columns)
    if 'split' in d: assert set(d.split)=={split}
    return d

def resolve(cfg):
    reg=yaml.safe_load((ROOT/cfg['registry']).read_text())
    e=reg['datasets'][cfg['dataset_id']];root=ROOT/e['repository_relative_root']
    assert sha(ROOT/e['manifest_path'])==e['manifest_sha256']
    m=json.loads((ROOT/e['manifest_path']).read_text());assert m['dataset_id']==cfg['dataset_id']
    names=['flights.csv','flight_provenance.json','split_manifest.json','build_config.yaml']
    names += [f'{stem}_{sp}.parquet' for sp in cfg['partitions'] for stem in
              ['samples','windows','native_actuator_motors','native_actuator_servos']]
    checked={}
    for name in names:
        if name not in m['artifact_sha256']: continue
        h=sha(root/name);assert h==m['artifact_sha256'][name], name;checked[name]=h
    for key,name in m['config']['source_files'].items():
        # Metadata only; never open a test flight log.
        p=Path(m['config']['source_root'])/name
        assert sha(p)==m['config']['source_sha256'][key]
        checked[str(p)]=sha(p)
    flights=pd.read_csv(root/'flights.csv')
    sealed=set(flights.loc[flights.split=='test','flight_id'])
    flights=flights[flights.split.isin(cfg['partitions'])].copy()
    assert len(flights)==161 and flights.split.value_counts().to_dict()=={'train':122,'validation':39}
    assert set(flights.flight_id).isdisjoint(sealed)
    # Shared metadata is permitted; retain only authorized records for all analysis.
    prov={r['flight_id']:r for r in json.loads((root/'flight_provenance.json').read_text()) if r['flight_id'] in set(flights.flight_id)}
    assignments=json.loads((root/'split_manifest.json').read_text())['assignments']
    assigned={r['flight_id']:r['split'] for r in assignments}
    assert all(assigned[r.flight_id]==r.split for r in flights.itertuples())
    return root,flights,prov,dict(manifest_sha256=e['manifest_sha256'],checked_artifact_sha256=checked,
        test_artifacts_opened=False,partitions=cfg['partitions'],phase_contract=m['config']['phase'],
        split_manifest_sha256=sha(root/'split_manifest.json'),sealed_flights=len(sealed))

def tape(m,s):
    t=np.unique(np.r_[m.timestamp_us,s.timestamp_us]).astype(np.int64)
    values=[];ages=[]
    for d,cols in [(m,['control[0]']),(s,['control[0]','control[1]','control[2]'])]:
        st=d.timestamp_us.to_numpy();ix=np.searchsorted(st,t,side='right')-1
        values.append(d[cols].to_numpy()[ix.clip(0)])
        ages.append(np.where(ix>=0,t-st[ix.clip(0)],10**12))
    return t,np.column_stack(values).astype(float),np.max(ages,axis=0)

def zoh(t,u,at):
    ix=np.searchsorted(t,at,side='right')-1
    if np.any(ix<0):raise ValueError('missing start command')
    return u[ix]

def exact_difference(a,b,horizon,thresholds):
    """a,b are relative-time command tapes covering [0,horizon), no interpolation."""
    ta,ua=a;tb,ub=b
    t=np.unique(np.r_[0,ta[(ta>0)&(ta<horizon)],tb[(tb>0)&(tb<horizon)],horizon])
    d=np.abs(zoh(ta,ua,t[:-1])-zoh(tb,ub,t[:-1]));dt=np.diff(t)/horizon
    rms=np.sqrt((d*d*dt[:,None]).sum(0))
    shares=np.array([(dt[:,None]*(d>=threshold)).sum(0) for threshold in thresholds])
    return rms,shares

def event_indices(t,u,threshold,refractory):
    """Cumulative command excursion relative to last accepted level; not independent interventions."""
    anchor=float(u[0]);last=int(t[0]);out=[]
    for i in range(1,len(t)):
        if t[i]-last>=refractory and abs(u[i]-anchor)>=threshold:
            out.append(i);anchor=float(u[i]);last=int(t[i])
    return np.asarray(out,int)

def segment_intervals(s):
    return [(int(g.state_measurement_us.iloc[0]),int(g.state_publication_us.iloc[-1]),int(seg))
            for seg,g in s[s.valid_core].groupby('segment_id',sort=False) if len(g)>1]

def read_all(root,flights):
    groups={};natives={};windows=[]
    for split in ['train','validation']:
        s=load_parquet(root,'samples',split,META+V+Q+W+U)
        assert set(s.flight_id)==set(flights.loc[flights.split==split,'flight_id'])
        groups.update({f:g.reset_index(drop=True) for f,g in s.groupby('flight_id',sort=False)})
        m=load_parquet(root,'native_actuator_motors',split);v=load_parquet(root,'native_actuator_servos',split)
        md=dict(tuple(m.groupby('flight_id',sort=False)));vd=dict(tuple(v.groupby('flight_id',sort=False)))
        for f in md:
            assert f in groups;natives[f]=tape(md[f],vd[f])
        w=load_parquet(root,'windows',split)
        windows.append(w[(w.history_us==500000)&w.horizon_us.isin([500000,1000000])].copy())
    return groups,natives,pd.concat(windows,ignore_index=True)

def coverage(cfg,flights,groups,natives,out):
    records=[];events=[];allrows=[];flightcov=[];increment_stats=[]
    for row in flights.itertuples():
        fid=row.flight_id;s=groups[fid];t,u,age=natives[fid];chunks=[]
        for lo,hi,seg in segment_intervals(s):
            ix=np.flatnonzero((t>=lo)&(t<hi)&(age<=160000))
            if len(ix)<2:continue
            tt=t[ix];uu=u[ix];duration=np.diff(np.r_[tt,hi])*1e-6
            assert np.isfinite(uu).all() and np.all(duration>0)
            # State labels are past-only at command publication.
            si=(np.searchsorted(s.state_publication_us.to_numpy(),tt,side='right')-1).clip(0)
            ss=s.iloc[si];frame=pd.DataFrame(uu,columns=CHANNELS)
            frame['dt']=duration;frame['flight_id']=fid;frame['split']=row.split;frame['sampling_group']=row.sampling_group
            frame['mode']=ss.nav_state.to_numpy();frame['speed']=np.linalg.norm(ss[V],axis=1)
            frame['rate_norm']=np.linalg.norm(ss[W],axis=1);frame['review']=ss.review.to_numpy()
            frame['valid_status']=ss.valid_status.to_numpy();frame['frequency']=ss.flap_frequency_hz.to_numpy()
            frame['high_dynamics']=frame.rate_norm>2
            chunks.append(frame)
            for c,ch in enumerate(CHANNELS):
                delta=np.diff(uu[:,c]);rates=np.abs(delta)/np.diff(tt)*1e6
                if row.split=='train':increment_stats.append(dict(channel=ch,group=row.sampling_group,delta=np.abs(delta),rate=rates))
                for scale in cfg['event_sensitivity']:
                    ids=event_indices(tt,uu[:,c],cfg['event_threshold'][c]*scale,cfg['event_refractory_us'])
                    dwell=np.diff(np.r_[tt[0],tt[ids],hi])*1e-6
                    events.append(dict(flight_id=fid,split=row.split,sampling_group=row.sampling_group,channel=ch,scale=scale,
                        events=len(ids),duration_s=duration.sum(),median_dwell_s=np.median(dwell),max_dwell_s=max(dwell)))
                    if scale==1:
                        for i in ids:records.append((fid,ch,int(tt[i])))
        if chunks:
            d=pd.concat(chunks,ignore_index=True);allrows.append(d)
            for c,ch in enumerate(CHANNELS):
                x=d[ch].to_numpy();weights=d.dt.to_numpy()
                flightcov.append(dict(flight_id=fid,split=row.split,sampling_group=row.sampling_group,channel=ch,
                    duration_s=weights.sum(),min=x.min(),p01=np.quantile(x,.01),median=np.median(x),p99=np.quantile(x,.99),max=x.max(),
                    std=np.std(x),near_limit_fraction=np.average((x>=.98) if c==0 else (np.abs(x)>=.98),weights=weights)))
    d=pd.concat(allrows,ignore_index=True);ec=pd.DataFrame(events);fc=pd.DataFrame(flightcov)
    ec.to_csv(out/'event_counts_per_flight.csv',index=False);fc.to_csv(out/'control_coverage_per_flight.csv',index=False)
    # Event simultaneity: another channel's accepted excursion within +/-50ms.
    sync=[]
    for fid,rs in pd.DataFrame(records,columns=['flight_id','channel','time_us']).groupby('flight_id'):
        times={ch:rs.loc[rs.channel==ch,'time_us'].to_numpy() for ch in CHANNELS}
        for ch in CHANNELS:
            x=times[ch];others=np.sort(np.concatenate([v for k,v in times.items() if k!=ch]))
            j=np.searchsorted(others,x)
            if len(others):dist=np.minimum(np.abs(x-others[(j-1).clip(0,len(others)-1)]),np.abs(x-others[j.clip(0,len(others)-1)]))
            else:dist=np.full(len(x),10**12)
            sync.append(dict(flight_id=fid,channel=ch,events=len(x),isolated_50ms=int((dist>50000).sum())))
    pd.DataFrame(sync).to_csv(out/'event_independence.csv',index=False)
    summary=[]
    for key,g in d.groupby(['split','sampling_group'],sort=False):
        summary.append(dict(split=key[0],sampling_group=key[1],flights=g.flight_id.nunique(),command_rows=len(g),duration_s=g.dt.sum()))
    pd.DataFrame(summary).to_csv(out/'coverage.csv',index=False)
    grouped=[]
    d['speed_bin']=pd.cut(d.speed,[-np.inf,2,4,6,np.inf],labels=['<2','2-4','4-6','>=6']).astype(str)
    for grouping in ['mode','speed_bin','high_dynamics','review']:
        for (group,label),g in d.groupby(['sampling_group',grouping],observed=True):
            for ch in CHANNELS:
                grouped.append(dict(grouping=grouping,sampling_group=group,label=label,channel=ch,flights=g.flight_id.nunique(),
                    duration_s=g.dt.sum(),**qstats(g[ch])))
    pd.DataFrame(grouped).to_csv(out/'coverage_by_condition.csv',index=False)
    inc=[]
    for group in ['main_50hz','sparse_10hz']:
        for ch in CHANNELS:
            for measure in ['delta','rate']:
                x=np.concatenate([r[measure] for r in increment_stats if r['channel']==ch and r['group']==group])
                inc.append(dict(sampling_group=group,channel=ch,measure=measure,**qstats(x)))
    pd.DataFrame(inc).to_csv(out/'train_increment_statistics.csv',index=False)
    corr={};matrix=[]
    for label,g in [('all',d)]+list(d.groupby('sampling_group')):
        x=g[CHANNELS].to_numpy();weight=g.dt.to_numpy();cov=np.cov(x,rowvar=False,aweights=weight)
        sd=np.sqrt(np.diag(cov));cc=cov/np.outer(sd,sd);corr[label]=cc.tolist()
        for i,ch in enumerate(CHANNELS):
            matrix.append(dict(group=label,channel=ch,**dict(zip(CHANNELS,cc[i]))))
    pd.DataFrame(matrix).to_csv(out/'control_correlations.csv',index=False)
    fig,axs=plt.subplots(2,2,figsize=(11,7))
    for ax,ch in zip(axs.flat,CHANNELS):
        for group,g in d.groupby('sampling_group'):
            ax.hist(g[ch],bins=65,weights=g.dt,density=True,alpha=.5,label=group)
        ax.set(title=ch,xlabel='Normalized command',ylabel='Time-weighted density');ax.legend()
    fig.tight_layout();fig.savefig(out/'figures/control_distributions.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,4));im=ax.imshow(corr['all'],vmin=-1,vmax=1,cmap='coolwarm')
    ax.set_xticks(range(4),CHANNELS,rotation=25);ax.set_yticks(range(4),CHANNELS)
    for i in range(4):
        for j in range(4):ax.text(j,i,f'{corr["all"][i][j]:.2f}',ha='center')
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(out/'figures/control_correlation.png',dpi=160);plt.close(fig)
    # Within-flight command-rate correlation avoids conflating differing trim commands.
    return dict(coverage=summary,total_effective_seconds=float(d.dt.sum()),native_event_rows=len(d),
                raw_samples=sum(len(s) for s in groups.values()),valid_samples=sum(int(s.valid_core.sum()) for s in groups.values()))

def build_histories(cfg,flights,prov,groups,natives,windows,horizon):
    meta=[];hist=[];audit=[];features=[]
    ftable=flights.set_index('flight_id');loose=cfg['calipers']['relaxed']
    for fid,ws in windows[windows.horizon_us==horizon].groupby('flight_id',sort=False):
        s=groups[fid];t=s.state_measurement_us.to_numpy();pub=s.state_publication_us.to_numpy()
        a=ws.start_sample.to_numpy();o=ws.origin_sample.to_numpy();e=ws.end_sample.to_numpy()
        assert np.all(s.segment_id.to_numpy()[a]==s.segment_id.to_numpy()[e]) and np.all(s.segment_id.to_numpy()[a]>=0)
        assert np.all(s.support_start_us.to_numpy()[a]>=ftable.loc[fid,'start_us'])
        assert np.all(s.support_end_us.to_numpy()[e]<=ftable.loc[fid,'end_us'])
        ex=np.array(prov[fid]['exclude_intervals_us'],dtype=np.int64).reshape(-1,2)
        for x,y in ex: assert not np.any((ws.support_start_us<=y)&(ws.support_end_us>=x))
        at=t[o,None]+np.linspace(-cfg['history_us'],0,cfg['history_grid_points']).astype(np.int64)
        ix=np.searchsorted(t,at,side='right')-1
        assert (ix>=a[:,None]).all() and (ix<=o[:,None]).all()
        assert np.all(pub[ix]<=pub[o,None])
        # Full native history validity, not just the feature grid.
        good=(s.valid_frequency&s.valid_status&(~s.failsafe.astype(bool))).to_numpy()
        bad=np.r_[0,np.cumsum(~good)];eligible=bad[o+1]==bad[a]
        mode=s.nav_state.to_numpy();changes=np.r_[0,np.cumsum(np.r_[False,np.diff(mode)!=0])]
        eligible &= changes[o+1]==changes[a+1]
        # Future commands start at decision/publication, not at an earlier measurement.
        ends=pub[o]+horizon
        segend=s.groupby('segment_id').state_publication_us.max()
        eligible &= ends<=s.segment_id.map(segend).to_numpy()[o]
        eligible &= ends<=ftable.loc[fid,'end_us']
        for x,y in ex:eligible &= ~((s.support_start_us.to_numpy()[a]<=y)&(ends>=x))
        nt,nu,age=natives[fid]
        # Verify the entire future tape is fresh, including last-hold age at the end.
        ni=np.searchsorted(nt,pub[o],side='right')-1;ne=np.searchsorted(nt,ends,side='left')-1
        nb=np.r_[0,np.cumsum((age>160000)|~np.isfinite(nu).all(1))]
        eligible &= (ni>=0)&(ne>=ni)&(nb[ne+1]==nb[ni.clip(0)])&(ends-nt[ne]+age[ne]<=160000)
        audit.append(dict(flight_id=fid,split=ftable.loc[fid,'split'],sampling_group=ftable.loc[fid,'sampling_group'],
            horizon_us=horizon,registered_windows=len(ws),eligible_windows=int(eligible.sum()),
            missing_frequency_status_or_failsafe=int((bad[o+1]!=bad[a]).sum())))
        if not eligible.any():continue
        ws=ws.loc[eligible].copy();ix=ix[eligible];a=a[eligible];o=o[eligible]
        vals=s[V+Q+W+['flap_frequency_hz']+U].to_numpy()[ix]
        vals[:,:,3:7]/=np.linalg.norm(vals[:,:,3:7],axis=2,keepdims=True)
        # [v3,q4,w3,f1,u4], no position or absolute mechanical phase.
        r=Rotation.from_quat(vals[:,:,3:7].reshape(-1,4)[:,[1,2,3,0]]).as_matrix().reshape(len(vals),-1,9)
        feature=np.c_[vals[:,:,:3].mean(1)/loose['v'],r.mean(1)/np.deg2rad(loose['R_deg']),
            vals[:,:,7:10].mean(1)/loose['w'],vals[:,:,10].mean(1)/loose['f'],vals[:,:,11:15].mean(1)/loose['u']]
        ws['nav_state']=mode[o];ws['log_sha']=ftable.loc[fid,'sha256'];ws['ratio']=ftable.loc[fid,'transmission_ratio']
        ws['hardware_uuid']=prov[fid]['sys_uuid']
        ws['firmware']=prov[fid]['firmware']
        ws['history_speed']=np.linalg.norm(vals[:,:,:3],axis=2).mean(1)
        ws['history_rate']=np.linalg.norm(vals[:,:,7:10],axis=2).mean(1)
        ws['history_frequency']=vals[:,:,10].mean(1)
        for j,ch in enumerate(CHANNELS):ws['history_'+ch]=vals[:,:,11+j].mean(1)
        histreview=np.r_[0,np.cumsum(s.review)]
        ws['history_review']=histreview[o+1]>histreview[a]
        ws['history_phase_valid']=np.all(s.valid_encoder.to_numpy()[ix],axis=1)
        ws['history_start_us']=s.support_start_us.to_numpy()[a]
        ws['command_end_us']=pub[o]+horizon
        ws['pair_support_end_us']=np.maximum(ws.support_end_us,ws.command_end_us)
        meta.append(ws);hist.append(vals.astype(np.float32));features.append(feature)
    return pd.concat(meta,ignore_index=True),np.concatenate(hist),np.concatenate(features),pd.DataFrame(audit)

def history_errors(x,y):
    v=np.sqrt(np.mean(np.sum((x[:,:,:3]-y[:,:,:3])**2,axis=2),axis=1))
    w=np.sqrt(np.mean(np.sum((x[:,:,7:10]-y[:,:,7:10])**2,axis=2),axis=1))
    dot=np.abs(np.sum(x[:,:,3:7]*y[:,:,3:7],axis=2)).clip(0,1)
    angle=2*np.arccos(dot.astype(float));r=np.rad2deg(np.sqrt(np.mean(angle**2,axis=1)))
    f=np.sqrt(np.mean((x[:,:,10]-y[:,:,10])**2,axis=1))
    u=np.sqrt(np.mean((x[:,:,11:]-y[:,:,11:])**2,axis=1))
    initial=np.c_[np.linalg.norm(x[:,-1,:3]-y[:,-1,:3],axis=1),np.rad2deg(angle[:,-1]),
        np.linalg.norm(x[:,-1,7:10]-y[:,-1,7:10],axis=1),np.abs(x[:,-1,10]-y[:,-1,10]),
        np.abs(x[:,-1,11:]-y[:,-1,11:])]
    return np.c_[v,r,w,f,u,initial]

def caliper_mask(errors,c):
    return np.all(errors<=np.tile([c['v'],c['R_deg'],c['w'],c['f']]+[c['u']]*4,2),axis=1)

def disjoint_pairs(pairs,meta,flight_disjoint=False):
    """Greedy reproducible conservative support count, not an ESS estimator or maximum packing."""
    used={};accepted=[]
    logs=meta.log_sha.to_numpy();starts=meta.history_start_us.to_numpy();ends=meta.pair_support_end_us.to_numpy()
    for row in pairs[['score','i','j']].sort_values(['score','i','j'],kind='stable')[['i','j']].itertuples():
        idx=row.Index
        endpoints=[];ok=True
        for k in [int(row.i),int(row.j)]:
            key=logs[k];lo,hi=int(starts[k]),int(ends[k])
            spans=used.get(key,[])
            if flight_disjoint and spans:ok=False;break
            pos=bisect.bisect_left(spans,(lo,hi))
            if (pos and spans[pos-1][1]>=lo) or (pos<len(spans) and spans[pos][0]<=hi):ok=False;break
            endpoints.append((key,lo,hi))
        if ok:
            accepted.append(idx)
            for key,lo,hi in endpoints:bisect.insort(used.setdefault(key,[]),(lo,hi))
    return accepted

def pack_commands(meta,natives,horizon):
    pieces=[];width=0
    for fid,ws in meta.groupby('flight_id',sort=False):
        t,u,_=natives[fid];start=ws.decision_us.to_numpy();end=start+horizon
        a=np.searchsorted(t,start,side='right')-1;b=np.searchsorted(t,end,side='left')
        lengths=b-a;w=int(lengths.max());j=np.arange(w)[None,:]
        ix=(a[:,None]+j).clip(0,len(t)-1)
        times=t[ix]-start[:,None];times[:,0]=0;times=np.where(j<lengths[:,None],times,horizon)
        pieces.append((ws.index.to_numpy(),times,u[ix]));width=max(width,w)
    times=np.full((len(meta),width),horizon,np.int32);values=np.zeros((len(meta),width,4),np.float32)
    for ids,t,u in pieces:times[ids,:t.shape[1]]=t;values[ids,:t.shape[1]]=u
    return times,values

def batch_difference(ta,ua,tb,ub,horizon,thresholds):
    """Exact union of two piecewise-constant tapes; padded end events have zero duration."""
    width=ta.shape[1];times=np.concatenate((ta,tb),axis=1)
    order=np.argsort(times,axis=1,kind='stable');times=np.take_along_axis(times,order,axis=1)
    ia=np.maximum.accumulate(np.where(order<width,order,-1),axis=1).clip(0)
    ib=np.maximum.accumulate(np.where(order>=width,order-width,-1),axis=1).clip(0)
    rows=np.arange(len(ta))[:,None]
    diff=np.abs(ua[rows,ia]-ub[rows,ib]).astype(float)
    dt=np.diff(np.column_stack((times,np.full(len(times),horizon))),axis=1)/horizon
    rms=np.sqrt(np.sum(diff**2*dt[:,:,None],axis=1))
    shares=np.stack([np.sum((diff>=th)*dt[:,:,None],axis=1) for th in thresholds],axis=1)
    return rms,shares

def match(cfg,meta,hist,features,natives,horizon,out):
    pairs=[];candidate_count=0;loose=cfg['calipers']['relaxed']
    # Every pair satisfying full-history loose RMS calipers must satisfy these
    # mean-feature coordinate bounds (Jensen and |Rij(A)-Rij(B)| <= SO3 angle).
    for (group,mode),index in meta.groupby(['sampling_group','nav_state']).groups.items():
        ids=np.array(index);tree=cKDTree(features[ids]);print('radius',horizon,group,mode,len(ids),flush=True)
        for start in range(0,len(ids),256):
            neighbors=tree.query_ball_point(features[ids[start:start+256]],r=1.000002,p=np.inf,workers=4)
            aa=[];bb=[]
            for local,js in enumerate(neighbors):
                i=ids[start+local];j=ids[np.asarray(js,int)];j=j[j>i]
                if len(j):aa.extend([i]*len(j));bb.extend(j.tolist())
            if not aa:continue
            aa=np.array(aa);bb=np.array(bb)
            valid=(meta.log_sha.to_numpy()[aa]!=meta.log_sha.to_numpy()[bb])
            valid &= np.abs(meta.ratio.to_numpy()[aa]-meta.ratio.to_numpy()[bb])<=cfg['max_ratio_difference']
            valid &= meta.hardware_uuid.to_numpy()[aa]==meta.hardware_uuid.to_numpy()[bb]
            aa=aa[valid];bb=bb[valid];candidate_count+=len(aa)
            for st in range(0,len(aa),2048):
                a,b=aa[st:st+2048],bb[st:st+2048];err=history_errors(hist[a],hist[b]);keep=caliper_mask(err,loose)
                if keep.any():pairs.append(np.c_[a[keep],b[keep],err[keep]])
    cols=['i','j','v_rmse','R_rmse_deg','w_rmse','f_rmse']+['history_gap_'+ch for ch in CHANNELS]
    cols+=['initial_v_gap','initial_R_gap_deg','initial_w_gap','initial_f_gap']+['initial_u_gap_'+ch for ch in CHANNELS]
    p=pd.DataFrame(np.concatenate(pairs) if pairs else np.empty((0,len(cols))),columns=cols)
    p[['i','j']]=p[['i','j']].astype(int)
    print('full-history pairs',horizon,len(p),'radius candidates',candidate_count,flush=True)
    diffs=[];fractions=[]
    thresholds=np.array(cfg['future_contrast'])[None,:]*np.array(cfg['contrast_sensitivity'])[:,None]
    times,values=pack_commands(meta,natives,horizon)
    for start in range(0,len(p),4096):
        a=p.i.to_numpy()[start:start+4096];b=p.j.to_numpy()[start:start+4096]
        diff,fraction=batch_difference(times[a],values[a],times[b],values[b],horizon,thresholds)
        diffs.append(diff);fractions.append(fraction)
        if start%409600==0:print('control contrasts',horizon,start,flush=True)
    diffs=np.concatenate(diffs) if diffs else np.empty((0,4))
    fractions=np.concatenate(fractions) if fractions else np.empty((0,3,4))
    del times,values
    for c,ch in enumerate(CHANNELS):p['future_gap_'+ch]=diffs[:,c]
    for lev,c in cfg['calipers'].items():p[lev]=caliper_mask(p[cols[2:]].to_numpy(),c)
    scales=np.tile([cfg['calipers']['primary'][k] for k in ['v','R_deg','w','f']]+[cfg['calipers']['primary']['u']]*4,2)
    p['score']=np.sqrt(np.mean((p[cols[2:]].to_numpy()/scales)**2,axis=1))
    p['group']=meta.sampling_group.to_numpy()[p.i]
    sa=meta.split.to_numpy()[p.i];sb=meta.split.to_numpy()[p.j]
    p['split_pair']=np.where(sa==sb,np.where(sa=='train','train-train','validation-validation'),'train-validation')
    p['any_review']=meta.history_review.to_numpy()[p.i]|meta.history_review.to_numpy()[p.j]
    p['high_dynamics']=(meta.history_rate.to_numpy()[p.i]>2)|(meta.history_rate.to_numpy()[p.j]>2)
    p['same_firmware']=meta.firmware.to_numpy()[p.i]==meta.firmware.to_numpy()[p.j]
    for ix,scale in enumerate(cfg['contrast_sensitivity']):
        for c,ch in enumerate(CHANNELS):p[f'fraction_{scale}_{ch}']=fractions[:,ix,c]
    summary=[];selected=[]
    for level in cfg['calipers']:
        for ix,scale in enumerate(cfg['contrast_sensitivity']):
            print('summarize',horizon,level,scale,flush=True)
            contrast=(diffs>=thresholds[ix])&(fractions[:,ix]>=max(cfg['minimum_contrast_fraction'],cfg['minimum_contrast_duration_s']/(horizon*1e-6)))
            if level=='primary' and scale==1:
                for ch,c in zip(CHANNELS,range(4)):p['contrast_'+ch]=contrast[:,c]
            for channel in ['any']+CHANNELS:
                if channel=='any':mask=contrast.any(1)
                else:
                    c=CHANNELS.index(channel);other=[j for j in range(4) if j!=c]
                    mask=contrast[:,c]&np.all(diffs[:,other]<=np.array(cfg['other_channel_max'])[other],axis=1)
                    if level=='primary' and scale==1:p['isolated_'+channel]=mask
                subset=p[p[level]&mask]
                for group in ['all','main_50hz','sparse_10hz']:
                    sub=subset if group=='all' else subset[subset.group==group]
                    accepted=disjoint_pairs(sub,meta);flightpairs=disjoint_pairs(sub,meta,True)
                    endpoints=np.unique(sub[['i','j']].to_numpy().ravel())
                    chosen=meta.iloc[np.unique(p.loc[accepted,['i','j']].to_numpy().ravel())]
                    summary.append(dict(horizon_ms=horizon//1000,caliper=level,contrast_scale=scale,channel=channel,group=group,
                        raw_pairs=len(sub),matched_windows=len(endpoints),flights=meta.iloc[endpoints].flight_id.nunique(),
                        nonoverlap_pairs=len(accepted),nonoverlap_flights=chosen.flight_id.nunique(),
                        flight_disjoint_pairs=len(flightpairs)))
                    if level=='primary' and scale==1 and group=='all':
                        for idx in accepted:selected.append(dict(pair_index=idx,channel=channel))
    p.to_parquet(out/f'pairs_{horizon//1000}ms.parquet',index=False)
    meta.to_parquet(out/f'windows_{horizon//1000}ms.parquet',index=False)
    pd.DataFrame(selected,columns=['pair_index','channel']).to_csv(out/f'selected_{horizon//1000}ms.csv',index=False)
    return p,pd.DataFrame(summary),dict(horizon_ms=horizon//1000,radius_candidates=candidate_count,loose_history_pairs=len(p),
                                     search='exhaustive radius, no KNN cap')

def source_checks(cfg,root,flights,prov):
    from pyulog import ULog
    checked=[]
    for _,g in flights.groupby(['split','sampling_group']):
        for row in g.sort_values('flight_id').head(cfg['source_spotcheck_per_group_split']).itertuples():
            permitted(row.split);path=Path('/home/zn/QgcLogs')/row.log_path
            assert sha(path)==row.sha256
            log=ULog(str(path),message_name_filter_list=['actuator_motors','actuator_servos'])
            for topic in ['actuator_motors','actuator_servos']:
                raw=next(d.data for d in log.data_list if d.name==topic and d.multi_id==0)
                native=load_parquet(root,'native_'+topic,row.split)
                native=native[native.flight_id==row.flight_id]
                ix=np.searchsorted(raw['timestamp'],native.timestamp_us.to_numpy())
                assert np.array_equal(raw['timestamp'][ix],native.timestamp_us.to_numpy())
                for c in [c for c in native if c.startswith('control[')]:np.testing.assert_array_equal(raw[c][ix],native[c].to_numpy())
            mapping={k:log.initial_parameters.get(k) for k in ['CA_SV_CS0_TYPE','CA_SV_CS1_TYPE','CA_SV_CS2_TYPE','PWM_MAIN_FUNC1','PWM_MAIN_FUNC2','PWM_MAIN_FUNC3','PWM_MAIN_FUNC5']}
            assert list(mapping.values())==[5,6,4,201,202,101,203]
            checked.append(dict(flight_id=row.flight_id,split=row.split,group=row.sampling_group,log_path=row.log_path,
                sha256=row.sha256,native_command_values_and_timestamps_exact=True,mapping=mapping))
    return checked

def main():
    start=time.time();cfg=json.loads((HERE/'config.json').read_text());out=HERE/'results'
    (out/'figures').mkdir(parents=True,exist_ok=True)
    root,flights,prov,manifest=resolve(cfg);print('hashes verified',flush=True)
    groups,natives,windows=read_all(root,flights)
    manifest['coverage']=coverage(cfg,flights,groups,natives,out);print(manifest['coverage'],flush=True)
    summaries=[];elig=[];search=[]
    for horizon in cfg['horizons_us']:
        meta,hist,features,audit=build_histories(cfg,flights,prov,groups,natives,windows,horizon)
        print('eligible',horizon,len(meta),flush=True)
        p,summary,sa=match(cfg,meta,hist,features,natives,horizon,out)
        summaries.append(summary);elig.append(audit);search.append(sa)
    pd.concat(summaries).to_csv(out/'match_sensitivity.csv',index=False)
    pd.concat(elig).to_csv(out/'window_eligibility.csv',index=False)
    manifest.update(source_spotchecks=source_checks(cfg,root,flights,prov),search=search,
        base_commit=cfg['base_commit'],head_at_execution=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        python=sys.executable,config_sha256=sha(HERE/'config.json'),code_sha256=sha(__file__),
        elapsed_seconds=time.time()-start,targets_read=False,neural_models_loaded=False)
    write_json(out/'run_manifest.json',manifest);print('DONE',time.time()-start,flush=True)

if __name__=='__main__':main()
