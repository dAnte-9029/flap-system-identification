"""Train-whitelisted availability audit for a possible closed-loop reference study."""
from pathlib import Path
import numpy as np
import pandas as pd
from pyulog import ULog
import run_paper_rollout_horizon_ablation as source

m=source.m
BASE=m.ROOT/'docs/analysis/results/control_dynamics_realdata_v1'
OUT=BASE/'reference_availability'
TOPICS=('manual_control_setpoint','input_rc','vehicle_attitude_setpoint','vehicle_rates_setpoint')


def run():
    OUT.mkdir(parents=True,exist_ok=False)
    sp=m.read_json(m.OUT/'protocol.json');mp=m.ROOT/sp['manifest_path']
    assert m.file_hash(mp)==sp['manifest_sha256'];manifest=m.read_json(mp)
    logs=sorted(sp['train_flights']);assert set(logs).isdisjoint(sp['validation_flights'])
    m.write_json(OUT/'protocol.json',dict(script_sha256=m.file_hash(Path(__file__)),dataset_id=sp['dataset_id'],
        manifest_sha256=sp['manifest_sha256'],allowed_flights=logs,partitions=['train'],heldout_accessed=False,
        topics=TOPICS,scope='message presence, recorded cadence, finite reference variation and RC parameter mapping only',
        no_claim='reference presence does not establish exogeneity, valid IV, controller-to-plant identification or fresh values at every origin',
        literature='https://arxiv.org/html/2309.05916v1',
        application='audit availability before considering reference-based estimation; nonlinear/hybrid PX4 and pilot feedback may violate paper assumptions'))
    rows=[];parameters=[];pins={}
    for log in logs:
        path=next(p for p in [Path(root)/log for root in manifest['source']['roots']] if p.exists())
        expected=manifest['source']['ulog_sha256'][log]
        assert m.file_hash(path)==expected
        pins[log]=dict(path=str(path),sha256=expected)
        ul=ULog(str(path),message_name_filter_list=list(TOPICS))
        parameters.append(dict(log_id=log,**{k:(float(ul.initial_parameters[k]) if k in ul.initial_parameters else None)
            for k in ['RC_MAP_ROLL','RC_MAP_PITCH','RC_MAP_YAW','RC_MAP_THROTTLE','FW_PR_P','FW_PR_I','FW_PR_D','FW_PR_FF','FLAP_SLOW_EN']}))
        for topic in TOPICS:
            datasets=[d for d in ul.data_list if d.name==topic and d.multi_id==0]
            if not datasets:
                rows.append(dict(log_id=log,topic=topic,rows=0));continue
            d=datasets[0].data;t=d['timestamp'].astype(float);dt=np.diff(t)*1e-6
            row=dict(log_id=log,topic=topic,rows=len(t),median_dt_s=float(np.median(dt)) if len(dt) else None,
                dt_p95_s=float(np.quantile(dt,.95)) if len(dt) else None,nonpositive_dt=int((dt<=0).sum()),
                sample_timestamp='timestamp_sample' in d,fields=','.join(d))
            for key in ['roll','pitch','yaw','throttle']:
                if key in d:
                    a=np.asarray(d[key]);mask=np.isfinite(a);row[key+'_finite_fraction']=float(mask.mean())
                    row[key+'_std']=float(np.std(a[mask])) if mask.any() else None
            if 'valid' in d:row['valid_fraction']=float(np.mean(d['valid']))
            rows.append(row)
        m.write_json(OUT/'status.json',dict(status='running',completed_flights=len(pins),last_flight=log))
        print('audited',len(pins),len(logs),log,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'topics.csv',index=False);pd.DataFrame(parameters).to_csv(OUT/'parameters.csv',index=False)
    m.write_json(OUT/'raw_sources.json',pins)
    m.write_json(OUT/'completion.json',dict(status='complete',heldout_accessed=False,train_flights=len(logs),
        instrument_validated=False,outputs={m.rel(p):m.file_hash(p) for p in OUT.iterdir() if p.is_file() and p.name!='status.json'}))
    m.write_json(OUT/'status.json',dict(status='complete',completed_flights=len(logs)))


if __name__=='__main__':run()
