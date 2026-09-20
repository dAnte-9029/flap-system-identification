"""Read only registered train logs; publication/sample ages are not servo latency."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd,yaml
from pyulog import ULog
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_main_v2_free_running import sha

def main():
    reg=yaml.safe_load((ROOT/'configs/data/trajectory_dataset_registry.yaml').read_text());did=reg['default_dataset_id'];e=reg['datasets'][did];mp=ROOT/e['manifest_path'];assert sha(mp)==e['manifest_sha256'];m=json.loads(mp.read_text());sp=mp.parent/'samples_train.parquet';assert sha(sp)==m['artifact_sha256'][sp.name];s=pd.read_parquet(sp)
    out=ROOT/'docs/analysis/results/control_response_lags_v1';topics=['actuator_servos','actuator_motors','vehicle_angular_velocity','vehicle_attitude','vehicle_local_position'];rows=[];sources={}
    for log,g in s[s.valid_core].groupby('log_id'):
        assert log in m['split_contract']['assignments']['train'];paths=[Path(r)/log for r in m['source']['roots']];paths=[p for p in paths if p.exists()];assert paths,log
        p=paths[0];assert sha(p)==m['source']['ulog_sha256'][log];sources[log]=dict(path=str(p),sha256=sha(p));ul=ULog(str(p),message_name_filter_list=topics)
        for topic in topics:
            datasets=[d for d in ul.data_list if d.name==topic and d.multi_id==0];assert len(datasets)==1,(log,topic)
            d=datasets[0].data;t=np.asarray(d['timestamp'],dtype=np.int64);ix=np.searchsorted(t,g.timestamp_us.to_numpy(),side='right')-1;assert (ix>=0).all();age=(g.timestamp_us.to_numpy()-t[ix])*1e-6
            event=np.asarray(d.get('timestamp_sample',np.zeros_like(t)),dtype=np.int64)[ix];ok=(event>0)&(event<=t[ix]);latency=(t[ix][ok]-event[ok])*1e-6
            rows.append(dict(log_id=log,topic=topic,rows=len(ix),hold_age_median_s=np.median(age),hold_age_p95_s=np.quantile(age,.95),sample_time_available=int(ok.sum()),publication_minus_sample_median_s=np.median(latency) if len(latency) else None,publication_minus_sample_p95_s=np.quantile(latency,.95) if len(latency) else None,negative_publication_steps=int((np.diff(t)<0).sum())))
        print(log,flush=True)
    f=pd.DataFrame(rows);f.to_csv(out/'raw_topic_timing.csv',index=False)
    (out/'raw_timing_manifest.json').write_text(json.dumps(dict(dataset_id=did,path=str(mp.parent.relative_to(ROOT)),manifest_sha256=sha(mp),sample_artifact_hashes={sp.name:sha(sp)},partitions=['train'],phase_contract=m['phase_contract'],frequency_contract=m['frequency_contract'],raw_sources=sources,script_sha256=sha(__file__),test_opened=False,limitation='timestamp_sample on actuator topics is upstream sample origin, not measured surface motion; no servo angle feedback used'),indent=2))
    print(f.groupby('topic')[['hold_age_median_s','hold_age_p95_s','publication_minus_sample_median_s']].median())
if __name__=='__main__':main()
