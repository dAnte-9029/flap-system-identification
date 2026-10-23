"""Supplemental source-level future-state poisoning and measurement-clock audit."""
import copy
import argparse
import numpy as np
import pandas as pd
import run as replay

HERE=replay.HERE
ap=argparse.ArgumentParser();ap.add_argument('--results',type=replay.Path,default=HERE/'results');ap.add_argument('--data-root',type=replay.Path);args=ap.parse_args();out=args.results
w=pd.read_csv(out/'windows.csv'); z=np.load(out/'native_rollouts.npz')
root=args.data_root or replay.Path(replay.json.loads((out/'input_receipt.json').read_text())['data']['root'])
ids=np.unique(np.linspace(0,len(w)-1,8).astype(int)); sub=w.iloc[ids].reset_index(drop=True)
g,n,r=replay.response.load_motion(root,set(sub.flight_id))
a,truth,*_=replay.pack(sub,g,n,r)
poison=copy.deepcopy(g)
for x in sub.itertuples():
    # All future motion, including phase and target labels, changes; command/dt tape stays fixed.
    cols=list(dict.fromkeys(replay.STATE+replay.PHASE+replay.TARGET))
    poison[x.flight_id].loc[x.origin_sample+1:x.end_sample,cols]=12345.678
b,other,*_=replay.pack(sub,poison,n,r)
for k in replay.KEYS:np.testing.assert_array_equal(a[k],b[k],err_msg=k)
assert not np.array_equal(truth,other)
checks=replay.checkpoint_manifest()
rows=[]
cols=['flight_id','state_measurement_us','rate_measurement_us','rate_publication_us','target_rate_measurement_us']
for split in ['train','validation']:
    data=pd.read_parquet(root/f'samples_{split}.parquet',columns=cols)
    groups={f:d.reset_index(drop=True) for f,d in data[data.flight_id.isin(w.flight_id)].groupby('flight_id')}
    for x in w[w.split==split].itertuples():
        d=groups[x.flight_id].iloc[x.origin_sample:x.end_sample+1]
        for j,row in enumerate(d.itertuples()):
            rows.append(dict(window_id=x.window_id,flight_id=x.flight_id,split=split,step=j,
                causal_rate_measurement_lag_us=row.state_measurement_us-row.rate_measurement_us,
                target_rate_offset_us=row.target_rate_measurement_us-row.state_measurement_us,
                causal_rate_publication_offset_us=row.rate_publication_us-row.state_measurement_us))
pd.DataFrame(rows).to_csv(out/'rate_clock_audit.csv',index=False)
main=pd.read_csv(out/'cohorts.csv').query("cohort=='main' and split=='train'")
p=pd.read_csv(out/'pairs.csv');m=p[p.pair_uid.isin(main.pair_uid)]
ix=np.unique(m[['a','b']].to_numpy());gap=z['truth_initial'][ix,10]-z['initial'][ix,10]
result=dict(passed=True,poisoned_windows=len(sub),eight_inputs_unchanged=True,future_truth_changed=True,
    checkpoint_hashes_still_match=len(checks),train_main_origin_p_gap_rms=float(np.sqrt(np.mean(gap**2))),
    train_main_origin_p_gap_abs_median=float(np.median(abs(gap))),
    train_main_origin_p_gap_abs_max=float(max(abs(gap))),test_opened=False,
    source_sha256=replay.sha(__file__))
replay.save(out/'input_causality_check.json',result);print(result)
print(pd.DataFrame(rows).drop(columns=['window_id','flight_id','split','step']).describe().to_string())
