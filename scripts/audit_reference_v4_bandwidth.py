"""Diagnostic logged-rate energy above the 50Hz model grid's Nyquist frequency."""
import json
import numpy as np
import pandas as pd
from pyulog import ULog
import train_control_windows_v2 as base
from audit_control_reference_semantics import past_values
from audit_reference_v4 import OUT

def main():
    pins=json.loads((OUT/'audit.json').read_text())['source_pins'];w=pd.read_csv(base.DATA/'windows_v2.csv');rows=[]
    for log,group in w.groupby('log_id'):
        path=next(p for p in pins if p.endswith(log));assert base.sha(path)==pins[path]
        u=ULog(path,message_name_filter_list=['vehicle_angular_velocity']);d=next(d.data for d in u.data_list if d.name=='vehicle_angular_velocity' and d.multi_id==0)
        for row in group.itertuples():
            # origin is26 samples after beginning of full support; retain fixed1s.
            ts=int(row.start_timestamp_us)+26*20000+np.arange(1,101)*10000
            values,pa,sa=past_values(d,['xyz[0]','xyz[1]','xyz[2]'],ts)
            centered=values-values.mean(0);power=abs(np.fft.rfft(centered,axis=0))**2;power[1:-1]*=2;f=np.fft.rfftfreq(100,.01)
            for j,name in enumerate(['p','q','r']):
                rows.append(dict(array_index=row.array_index,log_id=log,partition=row.partition,axis=name,
                    above25_fraction=float(power[f>25,j].sum()/max(power[:,j].sum(),1e-15)),sample_age_max_s=float(sa.max())))
    a=pd.DataFrame(rows);a.to_csv(OUT/'logged_rate_bandwidth.csv',index=False)
    a.groupby(['partition','axis']).above25_fraction.agg(['median','mean']).to_csv(OUT/'logged_rate_bandwidth_summary.csv')
    base.write(OUT/'bandwidth_scope.json',dict(script_sha256=base.sha(__file__),read_only_source_pins=pins,
        interpretation='100Hz causal held regrid of logged rate, 1Hz bins; above25Hz can alias at50Hz absent antialias filtering, but this is not underlying IMU spectrum or demonstrated alias error',sealed_test_accessed=False))
if __name__=='__main__':main()
