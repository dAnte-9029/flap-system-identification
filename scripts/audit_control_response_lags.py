"""Train-only closed-loop lead/lag diagnostic, not causal delay identification."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_main_v2_free_running import sha
from system_identification.data.trajectory_dataset import CONTROL_COLUMNS
from system_identification.evaluation.trajectory import BODY_RATE_COLUMNS

def main():
    reg=yaml.safe_load((ROOT/'configs/data/trajectory_dataset_registry.yaml').read_text());did=reg['default_dataset_id'];e=reg['datasets'][did];mp=ROOT/e['manifest_path'];assert sha(mp)==e['manifest_sha256'];m=json.loads(mp.read_text());sp=mp.parent/'samples_train.parquet';assert sha(sp)==m['artifact_sha256'][sp.name]
    out=ROOT/'docs/analysis/results/control_response_lags_v1';out.mkdir(parents=True,exist_ok=True)
    s=pd.read_parquet(sp);rows=[];timing=[]
    for log,g in s[s.valid_core].groupby('log_id'):
        delay=((g.timestamp_us-g.state_sample_timestamp_us)*1e-6).dropna()
        timing.append(dict(log_id=log,n=len(g),state_age_count=len(delay),state_age_median_s=delay.median() if len(delay) else None,state_age_p95_s=delay.quantile(.95) if len(delay) else None))
        pairs={}
        for seg,h in g.groupby('segment_id'):
            h=h.sort_values('sample_in_segment');u=h[list(CONTROL_COLUMNS)].to_numpy();u=np.c_[(u[:,1]+u[:,2])/2,(u[:,1]-u[:,2])/2,u[:,3]];rates=h[list(BODY_RATE_COLUMNS)].to_numpy()[:,[1,0,2]]
            # 100 ms increments suppress some measurement noise; overlapping values are not independent.
            for lag in range(-25,26):
                i=np.arange(30,len(h)-30,5);j=i+lag
                if not len(i):continue
                numbers=h.sample_in_segment.to_numpy();ts=h.timestamp_us.to_numpy()
                valid=(numbers[np.maximum(i,j)]-numbers[np.minimum(i,j)-5]==np.maximum(i,j)-np.minimum(i,j)+5)&(np.abs((ts[np.maximum(i,j)]-ts[np.minimum(i,j)-5])*1e-6-(np.maximum(i,j)-np.minimum(i,j)+5)*.02)<.005)
                i=i[valid];j=j[valid]
                pairs.setdefault(lag,[]).append((u[i]-u[i-5],rates[j]-rates[j-5]))
        for lag,blocks in pairs.items():
            a=np.concatenate([p[0] for p in blocks]);b=np.concatenate([p[1] for p in blocks])
            for k,c in enumerate(['common','differential','rudder']):
                corr=np.corrcoef(a[:,k],b[:,k])[0,1] if a[:,k].std()>1e-8 and b[:,k].std()>1e-8 else np.nan
                rows.append(dict(log_id=log,channel=c,lag_s=lag*.02,n=len(a),correlation=corr))
    f=pd.DataFrame(rows);f.to_csv(out/'per_log_lags.csv',index=False);pd.DataFrame(timing).to_csv(out/'publication_state_age.csv',index=False)
    peaks=f.loc[f.groupby(['log_id','channel']).correlation.apply(lambda x:x.abs().idxmax())];peaks.to_csv(out/'peak_lags.csv',index=False)
    contract=dict(dataset_id=did,path=str(mp.parent.relative_to(ROOT)),manifest_sha256=sha(mp),sample_artifact_hashes={sp.name:sha(sp)},partitions=['train'],phase_contract=m['phase_contract'],frequency_contract=m['frequency_contract'])
    (out/'manifest.json').write_text(json.dumps(dict(dataset=contract,script_sha256=sha(__file__),test_opened=False,positive_lag='rate increment follows command increment',method='within-contiguous-segment 100ms command/rate increments, stride100ms, per-log correlations at -0.5..0.5s',limitation='Closed-loop feedback, overlapping increments, phase and unmeasured surface angles confound lag; peaks are not physical actuator delay estimates',decision='retain drive tau0.10s/tail tau0.04s as uncalibrated proxies; no lag selected from validation'),indent=2))
    summary=peaks.groupby('channel').agg(logs=('log_id','size'),median_peak_lag_s=('lag_s','median'),min_peak_lag_s=('lag_s','min'),max_peak_lag_s=('lag_s','max'))
    (out/'report.md').write_text('# 闭环日志响应时序检查\n\n仅使用41个训练日志。控制与状态按发布时间过去值保持；无实测舵角，不能将相关峰当作执行器延迟。正滞后表示角速度变化在指令变化之后。\n\n'+summary.to_string()+'\n\n采用100ms增量、按日志检查±0.5s滞后，既检查控制领先也检查反馈导致的控制落后。保留0.10s驱动/0.04s舵机代理时间常数，不将它们宣称为辨识结果。方向约束来自已确认的舵面与机体方向，不来自相关符号。\n')
    print(summary.to_string())
if __name__=='__main__':main()
