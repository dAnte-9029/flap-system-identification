"""Read-only cross-checks and descriptive coverage, without changing selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from system_identification.data.reference_windows import extract_window
from train_control_windows_v2 import DATA, ROOT, load, write


def describe(values):
    values=np.asarray(values,dtype=float)
    return dict(n=len(values),mean=float(values.mean()),minimum=float(values.min()),
        maximum=float(values.max()),quantiles=dict(zip(['p05','p25','p50','p75','p95'],
        np.quantile(values,[.05,.25,.5,.75,.95]).tolist()))) if len(values) else dict(n=0)


def main():
    out=DATA/'audit'
    out.mkdir(exist_ok=False)
    x,y,stats=load()
    w=pd.read_csv(DATA/'windows_v2.csv')
    segments=pd.read_csv(DATA/'segments_v2.csv')
    blocks=pd.read_csv(DATA/'blocks_v2.csv')
    clusters=pd.read_csv(DATA/'event_clusters.csv')
    events=pd.read_csv(DATA/'events.csv')
    complete=events[events.event_cluster_id.isin(clusters.loc[clusters.complete_context,'event_cluster_id'])]
    complete.groupby(['partition','axis','sign']).agg(clusters=('event_cluster_id','nunique'),
        flights=('log_id','nunique')).reset_index().to_csv(out/'complete_event_coverage.csv',index=False)
    inputs={k:v.numpy() for k,v in x.items()}
    labels={k:v.numpy() for k,v in y.items()}
    checked=0
    for path,group in w.groupby('cache'):
        with np.load(ROOT/path) as z:
            a={k:z[k] for k in z.files}
        for row in group.itertuples():
            inp,target=extract_window(a,row.origin_index)
            for k,value in inp.items():
                np.testing.assert_allclose(inputs[k][row.array_index],value,rtol=1e-6,atol=1e-6)
            for k,value in target.items():
                np.testing.assert_allclose(labels[k][row.array_index],value,rtol=1e-6,atol=1e-6)
            assert a['timestamp_us'][row.origin_index-26]==row.start_timestamp_us
            assert a['timestamp_us'][row.origin_index+50]==row.end_timestamp_us
            checked+=1
    gaps=[]
    for log,group in w.groupby('log_id'):
        group=group.sort_values('start_timestamp_us')
        values=(group.start_timestamp_us.to_numpy()[1:]-group.end_timestamp_us.to_numpy()[:-1])/1e6
        assert (values>0).all()
        for gap in values:
            gaps.append(dict(log_id=log,gap_s=float(gap)))
    pd.DataFrame(gaps).to_csv(out/'window_gaps.csv',index=False)
    sourcegaps=[]
    for log,group in segments.groupby('log_id'):
        group=group.sort_values('start_timestamp_us')
        sourcegaps.extend(((group.start_timestamp_us.to_numpy()[1:]-group.end_timestamp_us.to_numpy()[:-1])/1e6).tolist())
    conditions=[]
    correlations=[]
    names=['roll_ref','pitch_ref','yaw_rate_ref','thrust_x','manual_yaw']
    for (partition,mode),group in w.groupby(['partition','mode']):
        ix=group.array_index.to_numpy()
        current=inputs['history_state'][ix,-1]
        for name,values in [('speed_body_norm',np.linalg.norm(current[:,:3],axis=1)),
                            ('flap_frequency_hz',current[:,11]),('p',current[:,3]),
                            ('q',current[:,4]),('r',current[:,5])]:
            conditions.append(dict(partition=partition,mode=int(mode),quantity=name,**describe(values)))
        # Within-window increments: report feedback/channel linkage, not raw trim correlations.
        delta=np.diff(inputs['future_reference'][ix],axis=1).reshape(-1,5)
        for i in range(5):
            for j in range(i+1,5):
                estimable=delta[:,i].std()>1e-8 and delta[:,j].std()>1e-8
                correlations.append(dict(partition=partition,mode=int(mode),input_a=names[i],input_b=names[j],
                    correlation=float(np.corrcoef(delta[:,i],delta[:,j])[0,1]) if estimable else None))
    write(out/'conditions.json',conditions)
    write(out/'input_increment_correlations.json',correlations)
    report=dict(checked_windows=checked,all_samples_match_source=True,
        no_shared_support_samples=True,source_segment_duration_s=describe(segments.duration_s),
        management_block_duration_s=describe(blocks.duration_s),
        source_intersegment_gaps_s=describe(sourcegaps),window_support_gaps_s=describe([r['gap_s'] for r in gaps]),
        complete_context_by_partition=clusters.groupby('partition').complete_context.sum().to_dict(),
        mode_counts=w.groupby(['partition','mode']).size().reset_index(name='windows').to_dict('records'),
        quality_scope='Q3 currently denotes servo command near-limit only; not an exhaustive difficult/disturbance classifier',
        statistical_scope='Disjoint samples still share flight and event context; flight is inferential resampling unit',
        sealed_test_accessed=False)
    write(out/'summary.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
