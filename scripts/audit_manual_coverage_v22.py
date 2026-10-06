"""Input-only audit over all original train parents; never selects training rows."""
from pathlib import Path
import json
import subprocess
import numpy as np
import pandas as pd
from pyulog import ULog
import train_control_windows_v2 as base
from audit_manual_chain_v21 import CONTRACT, COMMIT, PX4, rc_normalize, hold

OUT = base.ROOT/'docs/analysis/results/control_windows_v22'
V21 = OUT.parent/'control_windows_v21'
STRATA = OUT.parent/'control_windows_v19/reference_per_window.csv'
TOPICS = ['manual_control_setpoint','input_rc','vehicle_control_mode','vehicle_status',
          'vehicle_attitude_setpoint','vehicle_rates_setpoint','actuator_servos']


def metrics(d, query, value):
    """Message updates are distinct from changes of the held value."""
    ix=hold(d,query)
    age=(query-np.minimum(d['timestamp'][ix],d.get('timestamp_sample',d['timestamp'])[ix]))*.001
    v=np.asarray(value)[ix].astype(float)
    return ix,age,dict(unique_messages=int(len(np.unique(ix))),
                      updates=int(np.count_nonzero(np.diff(ix))),
                      value_changes=int(np.count_nonzero(abs(np.diff(v))>1e-5)),
                      age_median_ms=float(np.median(age)),age_p95_ms=float(np.quantile(age,.95)),
                      age_max_ms=float(age.max()),value_range=float(np.ptp(v)),
                      value_abs_max=float(abs(v).max()))


def activity_bin(amplitude, count_size):
    ratio=amplitude/count_size
    return 'at_most_1_count' if ratio<=1.0001 else '1_to_2_counts' if ratio<=2.0001 else 'over_2_counts'


def rc_pairs(rc, manual, start, end, calibration):
    ix=np.flatnonzero((rc['timestamp']>=start)&(rc['timestamp']<=end))
    mi=hold(manual,rc['timestamp'][ix])
    ch,minimum,trim,maximum,reverse=calibration
    normalized=rc_normalize(rc[f'values[{ch}]'][ix],minimum,trim,maximum,reverse)
    valid=(rc['rc_lost'][ix]==0)&(rc['rc_failsafe'][ix]==0)&(rc['channel_count'][ix]>ch)
    valid &= (manual['valid'][mi]!=0)&(manual['data_source'][mi]==1)
    valid &= rc['timestamp_last_signal'][ix]>=manual['timestamp_sample'][mi]
    return pd.DataFrame(dict(timestamp_us=rc['timestamp'][ix],rc_sample_us=rc['timestamp_last_signal'][ix],
                             normalized_rc_yaw=normalized,manual_yaw=manual['yaw'][mi],
                             manual_sample_us=manual['timestamp_sample'][mi],
                             valid_pair=valid,difference=normalized-manual['yaw'][mi]))


def grouped(w, grids, keys):
    out=[]
    groups=[('all',w)] if not keys else w.groupby(keys,dropna=False)
    for name,g in groups:
        z=grids[grids.array_index.isin(g.array_index)&grids.forecast]
        applicable=g[g.manual_applicable]
        az=z[z.manual_applicable]
        row=dict(group=str(name),windows=len(g),flights=g.log_id.nunique(),
                 support_seconds=float(((g.end_timestamp_us-g.start_timestamp_us)*1e-6).sum()),
                 manual_applicable_windows=len(applicable),
                 manual_over_2_counts=int((applicable.activity_bin=='over_2_counts').sum()),
                 rc_disagreement_over_2_counts_windows=int((applicable.rc_disagreement_count>0).sum()),
                 rc_evidence_windows=int((applicable.rc_valid_pairs>0).sum()))
        for field in ['manual_age_ms','rates_age_ms','servos_age_ms','attitude_reference_age_ms']:
            v=(az if field=='manual_age_ms' else z)[field].to_numpy()
            for n,q in [('p50',.5),('p95',.95),('max',1.)]:
                row[field+'_'+n]=float(np.quantile(v,q)) if len(v) else None
        for field in ['forecast_unique_messages','forecast_value_range','rc_max_abs_difference']:
            v=applicable[field].dropna().to_numpy()
            for n,q in [('p50',.5),('p95',.95),('max',1.)]:
                row[field+'_'+n]=float(np.quantile(v,q)) if len(v) else None
        out.append(row)
    return pd.DataFrame(out)


def main():
    data_manifest=base.DATA/'manifest.json'
    dm=json.loads(data_manifest.read_text())
    windows_path=base.DATA/'windows_v2.csv'
    assert base.sha(windows_path)==dm['tables']['windows_v2.csv']
    w=pd.read_csv(windows_path);w=w[w.partition=='train'].copy()
    assert len(w)==407 and w.array_index.is_unique and w.log_id.nunique()==5
    c=json.loads(CONTRACT.read_text())
    ready=base.ROOT/'docs/analysis/results/reference_dynamics_data_v1/training_ready/manifest.json'
    source_path=base.ROOT/'dataset/reference_dynamics_data_v1/manifest.json'
    srcs=json.loads(source_path.read_text())
    assert base.sha(ready)==c['parent_sha256']
    assert base.sha(source_path)==json.loads(ready.read_text())['parent_sha256']
    assert srcs['parent_manifest_sha256']==c['source_dataset_manifest_sha256']
    assert w.log_id.nunique()==c['train_flights']
    assert set(w.log_id)==set(dm['train_flights'])
    v21=json.loads((V21/'delivery_manifest.json').read_text())
    for f,h in v21['artifacts'].items():assert base.sha(V21/f)==h
    for f,h in v21['code'].items():assert base.sha(base.ROOT/f)==h
    strata=pd.read_csv(STRATA,usecols=['array_index','cohort','seed','stratum','arm','block'])
    strata=strata[(strata.cohort=='oof')&(strata.seed==17)&(strata.arm=='baseline')&(strata.block=='full1s')]
    w=w.merge(strata[['array_index','stratum']],on='array_index',validate='one_to_one')
    assert len(w)==407
    paths=[Path(__file__),base.ROOT/'tests/test_manual_coverage_v22.py',CONTRACT,ready,source_path,
           data_manifest,windows_path,STRATA,V21/'delivery_manifest.json',base.ROOT/'scripts/audit_manual_chain_v21.py',
           base.ROOT/'scripts/reference_timing_v14.py']
    protocol=dict(stage='v22',scope='all 407 original train windows; input-only audit; no training or exclusions',
                  pins={str(p.relative_to(base.ROOT)):base.sha(p) for p in paths},
                  phase_contract=c['phase_contract'],frequency_contract=c['frequency_contract'],
                  support='origin-26..origin+50 inclusive; future reference origin..origin+49 (50 points)',
                  methods='Causal max(publication,sample) eligibility; max age of both. Distinct messages and value changes separately. RC comparisons at native RC publication times within each full support; compare different sample times, never simultaneous measurement errors.',
                  activity='Report continuous range plus <=1,1..2,>2 normalized RC count bins. One count=1/min(trim-min,max-trim), conservative larger normalized step. Numerical boundary tolerance 1e-4 count. Descriptive only, not a quality or exclusion gate.',
                  rc='Require valid RC channel/source/manual, ordered sample times and manual-enabled control at RC publication. Report >2 count differences and full values; no lag fitting or interpolation.',
                  groups=['all','flight','mode','existing v19 OOF planned-reference stratum','flight x mode','manual activity'],
                  summary='Window counts; pooled 50-point future ages (equal per window); manual summaries apply only to mode15. RC evidence uses original full 1.52s support. No physical bandwidth or causality certification.',
                  validation_accessed=False,sealed_test_accessed=False,models_loaded=False)
    OUT.mkdir(exist_ok=False);base.write(OUT/'protocol.json',protocol)
    rows=[];gridrows=[];pairs=[];timings=[];pins=[]
    src_hashes=json.loads((V21/'source_provenance.json').read_text())['files']
    for cache_path,g in w.groupby('cache'):
        log=g.log_id.iloc[0];src=srcs['sources'][log]
        assert src['partition']=='train' and base.sha(src['path'])==src['sha256']
        assert base.sha(base.ROOT/cache_path)==c['cache_sha256'][cache_path]
        with np.load(base.ROOT/cache_path) as z:a={k:z[k] for k in z.files}
        u=ULog(src['path'],message_name_filter_list=TOPICS)
        firmware=u.msg_info_dict['ver_sw'];p=u.initial_parameters
        # Use exact logged firmware semantics, not a current checkout assumption.
        blobs={}
        for file in ['src/modules/rc_update/rc_update.cpp','src/modules/manual_control/ManualControlSelector.cpp',
                     'src/modules/fw_att_control/FixedwingAttitudeControl.cpp']:
            blob=subprocess.check_output(['git','-C',str(PX4),'rev-parse',f'{firmware}:{file}'],text=True).strip()
            assert blob==src_hashes[file]['blob'],(log,file,'semantics differ from v21')
            blobs[file]=blob
        ds={d.name:d.data for d in u.data_list if d.multi_id==0}
        assert set(TOPICS)<=ds.keys()
        ch=int(p['RC_MAP_YAW'])-1
        calibration=(ch,)+tuple(p[f'RC{ch+1}_{k}'] for k in ['MIN','TRIM','MAX','REV'])
        relevant={f'RC{ch+1}_{k}' for k in ['MIN','TRIM','MAX','REV']}|{'RC_MAP_YAW'}
        assert not any(k in relevant for _,k,_ in u.changed_parameters),'time-varying RC calibration needs explicit handling'
        _,minimum,trim,maximum,_=calibration
        count=1/min(trim-minimum,maximum-trim)
        pins.append(dict(log_id=log,source=src,cache=cache_path,cache_sha256=base.sha(base.ROOT/cache_path),
                         firmware=firmware,source_blobs=blobs,calibration=calibration,count_size=count,
                         parameter_changes=len(u.changed_parameters)))
        ordered=g.sort_values('start_timestamp_us')
        assert (ordered.start_timestamp_us.to_numpy()[1:]>ordered.end_timestamp_us.to_numpy()[:-1]).all()
        for r in g.itertuples():
            ids=np.arange(r.origin_index-26,r.origin_index+51)
            q=a['timestamp_us'][ids].astype(np.int64);fq=q[26:76]
            assert (q[0],q[-1])==(r.start_timestamp_us,r.end_timestamp_us)
            mi,age,sm=metrics(ds['manual_control_setpoint'],q,ds['manual_control_setpoint']['yaw'])
            _,_,fm=metrics(ds['manual_control_setpoint'],fq,ds['manual_control_setpoint']['yaw'])
            manual=ds['manual_control_setpoint'];raw=manual['yaw'][mi]
            applicable=a['nav_state'][ids]==15
            reconstructed=np.where(applicable,raw,0)
            np.testing.assert_allclose(reconstructed,a['references'][ids,4],atol=1e-7,rtol=0)
            np.testing.assert_allclose(age*.001,a['manual_reference_age_s'][ids],atol=1e-7,rtol=1e-6)
            assert (a['nav_state'][ids]==r.mode).all()
            cm=ds['vehicle_control_mode'];ci=hold(cm,q)
            mismatch=int(np.sum((cm['flag_control_manual_enabled'][ci]!=0)!=applicable))
            row={k:getattr(r,k) for k in ['array_index','window_id','log_id','mode','quality','stratum','start_timestamp_us','end_timestamp_us']}
            row.update(manual_applicable=bool(r.mode==15),count_size=count,
                       control_mode_mismatch_grid_points=mismatch,
                       manual_invalid_grid_points=int(np.count_nonzero(manual['valid'][mi]==0)),
                       manual_non_rc_grid_points=int(np.count_nonzero(manual['data_source'][mi]!=1)),
                       **{'support_'+k:v for k,v in sm.items()},**{'forecast_'+k:v for k,v in fm.items()})
            row['activity_bin']=activity_bin(fm['value_range'],count) if r.mode==15 else 'manual_not_in_model'
            row['model_manual_range']=float(np.ptp(reconstructed[26:76]))
            pp=rc_pairs(ds['input_rc'],manual,q[0],q[-1],calibration)
            pc=hold(cm,pp.timestamp_us.to_numpy())
            pp['manual_enabled']=cm['flag_control_manual_enabled'][pc]!=0
            pp['usable_for_model_manual']=pp.valid_pair&pp.manual_enabled&(r.mode==15)
            pp['over_2_counts']=abs(pp.difference)>2.0001*count
            pp['array_index']=r.array_index;pp['log_id']=log
            pairs.append(pp)
            valid=pp[pp.usable_for_model_manual]
            row.update(rc_pairs=len(pp),rc_valid_pairs=len(valid),
                       rc_disagreement_count=int(valid.over_2_counts.sum()),
                       rc_max_abs_difference=float(abs(valid.difference).max()) if len(valid) else None)
            grid=pd.DataFrame(dict(array_index=r.array_index,timestamp_us=q,forecast=(np.arange(77)>=26)&(np.arange(77)<76),
                                   manual_applicable=applicable,manual_yaw=raw,model_manual_yaw=reconstructed,manual_age_ms=age))
            for topic,key,label in [('vehicle_rates_setpoint','yaw','rates'),('actuator_servos','control[2]','servos'),
                                    ('vehicle_attitude_setpoint','yaw_sp_move_rate','attitude_reference')]:
                _,ta,_=metrics(ds[topic],q,ds[topic][key]);grid[label+'_age_ms']=ta
            gridrows.append(grid);rows.append(row)
            for topic in TOPICS:
                t=ds[topic]['timestamp'].astype(np.int64);in_window=t[(t>=q[0])&(t<=q[-1])];dt=np.diff(in_window)*.001
                timings.append(dict(array_index=r.array_index,log_id=log,topic=topic,rows=len(in_window),
                                    median_interval_ms=float(np.median(dt)) if len(dt) else None,
                                    max_interval_ms=float(dt.max()) if len(dt) else None))
        print(f'Audited {log}: {len(g)} windows',flush=True)
    table=pd.DataFrame(rows).sort_values('array_index');grids=pd.concat(gridrows,ignore_index=True)
    assert len(table)==407 and table.array_index.is_unique and set(table.array_index)==set(w.array_index)
    table.to_csv(OUT/'per_window.csv',index=False);grids.to_csv(OUT/'grid_evidence.csv',index=False)
    pd.concat(pairs,ignore_index=True).to_csv(OUT/'rc_evidence.csv',index=False)
    pd.DataFrame(timings).to_csv(OUT/'native_timing.csv',index=False)
    base.write(OUT/'source_manifest.json',dict(logs=pins,phase_contract=c['phase_contract'],frequency_contract=c['frequency_contract']))
    for label,keys in [('all',[]),('flight',['log_id']),('mode',['mode']),('stratum',['stratum']),
                       ('flight_mode',['log_id','mode']),('activity',['activity_bin'])]:
        grouped(table,grids,keys).to_csv(OUT/f'summary_{label}.csv',index=False)
    base.write(OUT/'checks.json',dict(windows=407,flights=5,all_original_parents_retained=True,
                nonoverlap_verified=True,source_hashes_and_semantics_verified=True,
                cache_value_and_age_verified_points=len(grids),validation_accessed=False,sealed_test_accessed=False))


if __name__=='__main__':main()
