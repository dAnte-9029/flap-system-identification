"""Independent checks of four explicitly selected non-test raw logs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from pyulog import ULog

ROOT=Path('/home/zn/flap-system-identification/dataset/velocity_rate_multistep_v1')
OUT=Path(__file__).resolve().parent
SELECT=['9.7数据/log_0_2026-9-7-18-00-56.ulg',
        '9.15数据/log_11_2026-9-15-17-51-56.ulg',
        '10.11数据-1/log_2_2026-10-11-08-09-26.ulg',
        '10.11数据-2/log_2_2026-10-11-17-52-20.ulg']
TOPICS=['wing_phase','hall_event','encoder_count','flap_frequency','rpm','pwm_input',
        'actuator_outputs','esc_status','esc_report']

def run():
    meta=json.loads((ROOT/'flight_provenance.json').read_text()); rows=[]
    for m in meta:
        if m['log_path'] not in SELECT: continue
        assert m['split'] in ['train','validation']
        u=ULog(str(Path('/home/zn/QgcLogs')/m['log_path']),message_name_filter_list=TOPICS)
        row=dict(log=m['log_path'],split=m['split'],firmware=m['firmware'],topics={})
        for x in u.data_list:
            row['topics'][f'{x.name}:{x.multi_id}']=dict(rows=len(x.data['timestamp']),fields=list(x.data))
        for topic,fields in [('encoder_count',['total_count']),('wing_phase',['encoder_total_count','phase_rad','phase_valid']),('hall_event',['pulse_count'])]:
            found=[x for x in u.data_list if x.name==topic and x.multi_id==0]
            if not found: continue
            data=found[0].data
            tab=pd.read_parquet(ROOT/f'native_{topic}_{m["split"]}.parquet',filters=[('flight_id','==',m['flight_id'])])
            j=np.searchsorted(data['timestamp'],tab.timestamp_us.to_numpy())
            assert np.array_equal(data['timestamp'][j],tab.timestamp_us)
            errors={f:float(np.max(np.abs(np.asarray(data[f][j],dtype=float)-tab[f].to_numpy()))) for f in fields}
            assert all(v==0 for v in errors.values())
            row[topic+'_check']=dict(rows=len(tab),max_abs_error=errors)
        topics={x.name:x.data for x in u.data_list if x.multi_id==0 and x.name!='rpm'}
        et=topics['encoder_count']['timestamp'].astype(np.int64)
        rpm_candidates=[]
        for x in u.data_list:
            if x.name!='rpm':continue
            # AS5600 also publishes RPM, with encoder timestamps. Do NOT assume
            # rpm instance 0 is Hall: instance numbering differs across boots.
            match=float(np.isin(x.data['timestamp'],et).mean())
            row['topics'][f'rpm:{x.multi_id}']['encoder_timestamp_match_fraction']=match
            if match<.01:rpm_candidates.append(x)
        if len(rpm_candidates)==1:
            topics['rpm']=rpm_candidates[0].data
            row['candidate_hall_rpm_instance']=rpm_candidates[0].multi_id
        if all(k in topics for k in ['hall_event','rpm','encoder_count']):
            ht=topics['hall_event']['timestamp'].astype(np.int64)
            rt=topics['rpm']['timestamp'].astype(np.int64)
            assert np.all(np.diff(ht)>0) and np.all(np.diff(rt)>0)
            # RPM now is stamped BEFORE hall.publish in the same serialized Run.
            # Use second subsequent logged RPM timestamp, not the first.
            j=np.searchsorted(rt,ht,side='right')+1
            k=np.searchsorted(et,ht,side='left')
            good=(j<len(rt))&(k>0)&(k+1<len(et))&(ht>=m['start_us'])&(ht<=m['end_us'])
            jj=np.minimum(j,len(rt)-1);kk=np.clip(k,1,len(et)-2)
            good &= et[kk]-et[kk-1]<=30000
            availability=np.maximum(rt[jj],et[kk+1])
            lag=(availability[good]-ht[good])*1e-6
            row['candidate_hall_availability_bound']=dict(events=int(good.sum()),
                lag_s_quantiles=np.quantile(lag,[.01,.5,.99]).tolist() if len(lag) else [],
                method='max(second RPM timestamp after Hall, next encoder timestamp after right bracket)',
                limitation='requires verified RPMCapture sole publisher and serialized Run; not measured publication time or cross-log physical index calibration')
        rows.append(row);print('verified',m['log_path'],flush=True)
    (OUT/'raw_verification.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))

if __name__=='__main__':run()
