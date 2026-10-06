"""Pool split labels for a duration audit; merge overlapping window support."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from pyulog import ULog

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/analysis/results/reference_dynamics_data_v1'
OUT=BASE/'pooled_timing'


def merge_intervals(items):
    """items are (start_us,end_us,id,active); closed endpoint elapsed spans."""
    merged=[]
    for start,end,key,active in sorted(items):
        if not merged or start>merged[-1][1]:
            merged.append([start,end,[key],int(active)])
        else:
            merged[-1][1]=max(merged[-1][1],end)
            merged[-1][2].append(key);merged[-1][3]+=int(active)
    return merged


def stats(values):
    a=np.asarray(values,dtype=float)
    if not len(a):return dict(n=0)
    return dict(n=len(a),total_s=float(a.sum()),mean_s=float(a.mean()),
        **{name:float(v) for name,v in zip(['min_s','p10_s','p25_s','p50_s','p75_s','p90_s','p95_s','max_s'],np.quantile(a,[0,.1,.25,.5,.75,.9,.95,1]))})


def run():
    OUT.mkdir(parents=True,exist_ok=False)
    # Simple checks establish merge duration and gaps without project/data assumptions.
    test=merge_intervals([(1,5,'a',True),(3,7,'b',False),(10,15,'c',True)])
    assert [(a,b) for a,b,_,_ in test]==[(1,7),(10,15)]
    assert sum(b-a for a,b,_,_ in test)==11
    assert test[1][0]-test[0][1]==3
    manifest=json.loads((ROOT/'dataset/reference_dynamics_data_v1/manifest.json').read_text())
    tables={'all_screened':BASE/'windows.csv','same_configuration_final':BASE/'training_contract/windows.csv'}
    starts={};result={};pooled=[];allgaps=[];perlog=[]
    for scope,path in tables.items():
        w=pd.read_csv(path);records=[]
        for log,rows in w.groupby('log_id'):
            if log not in starts:
                src=manifest['sources'][log]
                assert hashlib.sha256(Path(src['path']).read_bytes()).hexdigest()==src['sha256']
                starts[log]=ULog(src['path'],message_name_filter_list=['vehicle_status']).start_timestamp
            for sid,part in rows.groupby('segment_id'):
                # Earliest command-history sample = origin-26*20ms; last target = origin+50*20ms.
                spans=[(int(r.origin_timestamp_us)-520000,int(r.origin_timestamp_us)+1000000,r.window_id,r.active_reference) for r in part.itertuples()]
                for j,(a,b,ids,nactive) in enumerate(merge_intervals(spans)):
                    records.append(dict(scope=scope,log_id=log,source_segment_id=sid,interval_id=f'{sid}_union{j}',
                        mode=int(part['mode'].iloc[0]),start_timestamp_us=a,end_timestamp_us=b,
                        start_log_s=(a-starts[log])*1e-6,end_log_s=(b-starts[log])*1e-6,duration_s=(b-a)*1e-6,
                        windows=len(ids),active_reference_windows=nactive))
        f=pd.DataFrame(records).sort_values(['log_id','start_timestamp_us']);gaps=[]
        for log,part in f.groupby('log_id'):
            r=list(part.itertuples());gs=[]
            for prev,nxt in zip(r[:-1],r[1:]):
                gap=(nxt.start_timestamp_us-prev.end_timestamp_us)*1e-6
                assert gap>=0
                gs.append(gap)
                gaps.append(dict(scope=scope,log_id=log,previous_interval=prev.interval_id,next_interval=nxt.interval_id,
                    previous_start_log_s=prev.start_log_s,previous_end_log_s=prev.end_log_s,
                    next_start_log_s=nxt.start_log_s,next_end_log_s=nxt.end_log_s,gap_s=gap,
                    same_mode=prev.mode==nxt.mode,same_original_segment=prev.source_segment_id==nxt.source_segment_id))
            elapsed=float(part.end_log_s.max()-part.start_log_s.min());covered=float(part.duration_s.sum())
            np.testing.assert_allclose(covered+sum(gs),elapsed,atol=1e-6)
            perlog.append(dict(scope=scope,log_id=log,segments=len(part),covered_s=covered,
                first_start_log_s=part.start_log_s.min(),last_end_log_s=part.end_log_s.max(),between_first_last_s=elapsed,
                gaps=len(gs),gap_total_s=sum(gs),gap_mean_s=float(np.mean(gs)) if gs else None))
        d=pd.DataFrame(gaps)
        f.to_csv(OUT/f'{scope}_segments.csv',index=False);d.to_csv(OUT/f'{scope}_gaps.csv',index=False)
        result[scope]=dict(logs=w.log_id.nunique(),windows=len(w),window_elapsed_span_s=1.52,
            naive_window_sum_s=len(w)*1.52,segments=stats(f.duration_s),gaps=stats(d.gap_s),
            short_gaps_le2s=int((d.gap_s<=2).sum()),short_gaps_le5s=int((d.gap_s<=5).sum()),
            same_mode_gaps=int(d.same_mode.sum()),within_original_segment_gaps=int(d.same_original_segment.sum()),
            by_mode={str(mode):stats(items.duration_s) for mode,items in f.groupby('mode')})
        pooled.extend(records);allgaps.extend(gaps)
    pd.DataFrame(perlog).to_csv(OUT/'per_log.csv',index=False)
    receipt=dict(scope='Descriptive pool, ignoring train/validation labels; no training and no split reassignment',
        timing='Elapsed time from earliest observed history/command timestamp to final target timestamp; 1.52s per window, overlapping spans unioned within original quality/mode segment. Endpoints have zero measure, no extra20ms bin added.',
        gap='Next interval start minus previous interval end within same log; excludes pre-first/post-last time and cross-log gaps',
        source_tables={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in tables.values()},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),sealed_test_accessed=False,results=result)
    (OUT/'summary.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':run()
