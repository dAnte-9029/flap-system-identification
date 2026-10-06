"""Input-event mining and nonoverlapping A/B/C/D cohorts for the frozen config."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from system_identification.data.reference_windows import extract_window

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/analysis/results/reference_dynamics_data_v1'
OUT=ROOT/'docs/analysis/results/control_windows_v2'
ART=ROOT/'dataset/control_windows_v2'
THRESH=np.array([np.deg2rad(1),np.deg2rad(1),.05,.03,.03])


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')


def input_events(ref,start,stop,threshold=THRESH):
    """Accumulated change from last trigger; minimum0.2s refractory per channel."""
    events=[]
    for axis in range(ref.shape[1]):
        anchor=float(ref[start,axis]);last=start-10
        for i in range(start+1,stop):
            delta=float(ref[i,axis]-anchor)
            if i-last>=10 and abs(delta)>=threshold[axis]:
                events.append((i,axis,float(delta)));anchor=float(ref[i,axis]);last=i
    return sorted(events)


def choose_origins(start,stop,events):
    """Complete index support [origin-26,origin+50]; forbid even shared endpoints."""
    chosen=[]
    def add(i):
        if i-26<start or i+50>=stop:return
        if any(not(i+50<j-26 or i-26>j+50) for j in chosen):return
        chosen.append(i)
    for i,_,_ in events:
        origin=max(start+26,i-5)
        if origin<=i<origin+50:add(origin)
    for i in range(start+26,stop-50,80):add(i)
    return sorted(chosen)


def cluster_events(events,start,stop):
    clusters=[]
    for i,axis,delta in events:
        left=max(start,i-50);right=min(stop,i+151)
        if not clusters or left>=clusters[-1]['stop']:
            clusters.append(dict(start=left,stop=right,events=[(i,axis,delta)],complete=(i-50>=start and i+151<=stop)))
        else:
            c=clusters[-1];c['stop']=max(c['stop'],right);c['events'].append((i,axis,delta));c['complete'] &= (i-50>=start and i+151<=stop)
    return clusters


def run():
    OUT.mkdir(parents=True,exist_ok=False);ART.mkdir(parents=True,exist_ok=False)
    parent=BASE/'training_contract/manifest.json';m=json.loads(parent.read_text())
    for p,h in m['cache_sha256'].items():assert sha(ROOT/p)==h
    old=pd.read_csv(ROOT/m['windows_path'])
    pooled=pd.read_csv(BASE/'pooled_timing/same_configuration_final_segments.csv')
    write(OUT/'protocol.json',dict(parent_manifest_sha256=sha(parent),script_sha256=sha(__file__),
        plan_sha256=sha(ROOT/'docs/plans/control-oriented-window-selection-v2.md'),partitions=['train','validation'],sealed_test_accessed=False,
        split='Formal comparison retains original5train/9validation flights: no shared flight, no block split or validation reassignment; pooled exploration is descriptive only',
        group=m['group'],dt_s=.02,history_steps=26,forecast_steps=50,block_steps=500,
        event_thresholds=THRESH.tolist(),event_rule='per-channel cumulative change from last event,0.2s refractory; events and sign from references only',
        actuator_events='secondary diagnostic thresholds from train command q75 absolute0.2s changes, floor0.01; not a window-selection criterion',
        window_rule='event-first chronology origin0.1s before event when possible; background1.6s grid; forbid shared support samples; no crossing10s blocks',
        context='1s before and3s after last event; overlapping contexts merged as event clusters, truncated contexts marked',
        quality_classes='Q3 if any supplied servo command abs>=.98 in support; else Q1 if reference range threshold passed; else Q2. No response labels/GRU error used',
        nominal_scope='all originally admitted airborne same-config modes/states; no additional outcome-based cruise exclusion',
        A='all valid, flight-equal',B='same count as C within each log+mode, fixed random seed1309, flight-equal',C='Q1 only, flight-equal',D='all valid, flight-equal then within-flight Q1:Q2:Q3=3:1:1',
        normalization='A train unique rows only, common to all arms; no validation fitting',
        formal_training_started=False))
    rows=[];blocks=[];clusters=[];events_rows=[];arrays={};samples=[];targets=[]
    cachemap={log:g.cache.iloc[0] for log,g in old.groupby('log_id')}
    splitmap={log:g.partition.iloc[0] for log,g in old.groupby('log_id')}
    for log,parts in pooled.groupby('log_id',sort=True):
        path=cachemap[log]
        with np.load(ROOT/path) as z:a={k:z[k] for k in z.files}
        arrays[log]=a;split=splitmap[log]
        for s in parts.itertuples():
            start=int(np.searchsorted(a['timestamp_us'],s.start_timestamp_us));stop=int(np.searchsorted(a['timestamp_us'],s.end_timestamp_us))+1
            ev=input_events(a['references'],start,stop);cl=cluster_events(ev,start,stop)
            for n,c in enumerate(cl):
                cid=f'{s.interval_id}_event{n}'
                clusters.append(dict(event_cluster_id=cid,log_id=log,partition=split,segment_id=s.interval_id,
                    start_index=c['start'],stop_index=c['stop'],duration_s=(c['stop']-c['start'])*.02,
                    complete_context=c['complete'],events=len(c['events']),axes=','.join(map(str,sorted(set(x[1] for x in c['events']))))))
                for i,axis,delta in c['events']:
                    events_rows.append(dict(event_cluster_id=cid,log_id=log,partition=split,index=i,axis=axis,delta=delta,sign=int(np.sign(delta)),timestamp_us=int(a['timestamp_us'][i])))
            for n,left in enumerate(range(start,stop,500)):
                right=min(left+500,stop);bid=f'{s.interval_id}_block{n}'
                local=[e for e in ev if left<=e[0]<right];chosen=choose_origins(left,right,local)
                blocks.append(dict(block_id=bid,log_id=log,partition=split,segment_id=s.interval_id,mode=s.mode,
                    start_index=left,stop_index=right,duration_s=(right-left)*.02,windows=len(chosen),covered_sample_s=len(chosen)*77*.02))
                for i in chosen:
                    x,y=extract_window(a,i);variation=np.ptp(x['future_reference'],axis=0)
                    active=bool((variation>=THRESH).any());hard=bool((np.abs(a['actuator_commands'][i-26:i+51,1:])>=.98).any())
                    quality='Q3' if hard else 'Q1' if active else 'Q2'
                    context_ids=[f'{s.interval_id}_event{k}' for k,c in enumerate(cl) if c['start']<i+51 and c['stop']>i-26]
                    rows.append(dict(window_id=f'{bid}_{i}',array_index=len(samples),log_id=log,partition=split,cache=path,segment_id=s.interval_id,block_id=bid,
                        origin_index=i,mode=int(s.mode),quality=quality,active_reference=active,near_limit=hard,event_clusters=';'.join(context_ids),
                        start_timestamp_us=int(a['timestamp_us'][i-26]),end_timestamp_us=int(a['timestamp_us'][i+50]),
                        roll_range_rad=float(variation[0]),pitch_range_rad=float(variation[1]),manual_yaw_range=float(variation[4])))
                    samples.append(x);targets.append(y)
    w=pd.DataFrame(rows);b=pd.DataFrame(blocks);c=pd.DataFrame(clusters);e=pd.DataFrame(events_rows)
    assert w.window_id.is_unique
    for log,g in w.groupby('log_id'):
        g=g.sort_values('start_timestamp_us');assert np.all(g.start_timestamp_us.to_numpy()[1:]>g.end_timestamp_us.to_numpy()[:-1])
    assert not set(w[w.partition=='train'].log_id)&set(w[w.partition=='validation'].log_id)
    w.to_csv(OUT/'windows_v2.csv',index=False);b.to_csv(OUT/'blocks_v2.csv',index=False);pooled.to_csv(OUT/'segments_v2.csv',index=False)
    c.to_csv(OUT/'event_clusters.csv',index=False);e.to_csv(OUT/'events.csv',index=False)
    inputs={k:np.stack([s[k] for s in samples]) for k in samples[0]};labels={k:np.stack([s[k] for s in targets]) for k in targets[0]}
    np.savez_compressed(ART/'windows.npz',**{'input_'+k:v for k,v in inputs.items()},**{'label_'+k:v for k,v in labels.items()})
    tr=w[w.partition=='train'];cv=tr[tr.quality=='Q1'];assert len(cv)>0
    random=np.random.default_rng(1309);br=[]
    for (log,mode),items in cv.groupby(['log_id','mode']):
        pool=tr[(tr.log_id==log)&(tr['mode']==mode)];br.extend(random.choice(pool.index,len(items),replace=False).tolist())
    armids={'A':tr.index.to_numpy(),'B':np.sort(br),'C':cv.index.to_numpy(),'D':tr.index.to_numpy()}
    for arm,ids in armids.items():
        selected=w.loc[ids].copy();weights=np.ones(len(selected));
        if arm=='D':weights=np.where(selected.quality=='Q1',3.,1.)
        for log in selected.log_id.unique():
            mask=selected.log_id.to_numpy()==log;weights[mask]/=weights[mask].sum()
        weights/=weights.sum();selected['sampling_probability']=weights;selected.to_csv(OUT/f'arm_{arm}.csv',index=False)
    w[w.partition=='validation'].to_csv(OUT/'evaluation_windows.csv',index=False)
    # A-common statistics use unique training rows, not dense overlap multiplicity.
    history=[];reference=[];actuator=[];derivatives=[];rates=[];actdiff=[]
    for log,g in tr.groupby('log_id'):
        a=arrays[log];h=np.unique(np.concatenate([np.arange(i-25,i+1) for i in g.origin_index]));r=np.unique(np.concatenate([np.arange(i-25,i+50) for i in g.origin_index]))
        history.append(a['state_features'][h]);reference.append(a['references'][r]);actuator.append(a['actuator_commands'][h-1])
        for i in g.origin_index:
            rates.append(a['rates_b'][i:i+51]);actdiff.append(a['actuator_commands'][i:i+50]-a['actuator_commands'][i-10:i+40])
    stats={}
    for name,items in [('state',history),('reference',reference),('actuator',actuator)]:
        z=np.concatenate(items).astype(float);stats[name+'_mean']=z.mean(0);stats[name+'_scale']=z.std(0).clip(1e-4)
    stats['state_mean'][9:11]=0;stats['state_scale'][9:11]=1
    # Rollout-loss scales from training current/target physical values, no validation.
    rate_data=np.concatenate(rates);stats['rate_scale']=rate_data.std(0).clip(.1)
    train_indices=tr.array_index.to_numpy();stats['velocity_scale']=labels['velocity_n'][train_indices].reshape(-1,3).std(0).clip(.5)
    np.savez(ART/'normalization.npz',**stats)
    act_threshold=np.maximum(.01,np.quantile(np.abs(np.concatenate(actdiff)),.75,axis=0))
    secondary=[]
    for row in w.itertuples():
        a=arrays[row.log_id];i=row.origin_index;delta=a['actuator_commands'][i:i+50]-a['actuator_commands'][i-10:i+40]
        secondary.append(dict(window_id=row.window_id,actuator_events=int((np.abs(delta)>=act_threshold).any(1).sum())))
    pd.DataFrame(secondary).to_csv(OUT/'actuator_activity.csv',index=False)
    cover=dict(windows=len(w),by_partition=w.groupby('partition').size().to_dict(),by_quality=w.groupby(['partition','quality']).size().reset_index(name='n').to_dict('records'),
        source_logs=w.log_id.nunique(),segments=len(pooled),blocks=len(b),event_clusters=len(c),complete_event_clusters=int(c.complete_context.sum()),
        source_elapsed_s=float(pooled.duration_s.sum()),source_sample_s=float(b.duration_s.sum()),window_elapsed_s=len(w)*1.52,
        covered_sample_s=len(w)*1.54,unused_sample_s=float(b.duration_s.sum()-len(w)*1.54),
        arm_sizes={k:len(v) for k,v in armids.items()},B_C_overlap=int(len(set(armids['B'])&set(armids['C']))),
        actuator_threshold_train_q75=act_threshold.tolist(),no_overlap=True,
        event_count_warning='Individual input threshold crossings within same contextual cluster are not independent trials')
    write(OUT/'coverage_v2.json',cover)
    e.groupby(['partition','axis','sign']).agg(crossings=('index','size'),clusters=('event_cluster_id','nunique'),flights=('log_id','nunique')).reset_index().to_csv(OUT/'event_coverage.csv',index=False)
    write(OUT/'manifest.json',dict(parent_sha256=sha(parent),phase_contract=m['phase_contract'],frequency_contract=m['frequency_contract'],
        tables={p.name:sha(p) for p in OUT.iterdir() if p.is_file()},
        arrays={str(p.relative_to(ROOT)):sha(p) for p in ART.iterdir() if p.is_file()},
        train_flights=sorted(tr.log_id.unique()),validation_flights=sorted(w[w.partition=='validation'].log_id.unique()),
        sealed_test_accessed=False,formal_training_started=False))
    print(json.dumps(cover,indent=2))


if __name__=='__main__':run()
