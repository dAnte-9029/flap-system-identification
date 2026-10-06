"""Exploratory 2 Hz causal command-screen sensitivity, never a causal plant test.

Primary raw-command results remain unchanged. Thresholds are independently fixed
on filtered TRAIN commands before filtered validation samples are screened.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from audit_control_readiness_stage2 import ROOT, CHANNELS, candidates, sha, dump

OUT=ROOT/'docs/analysis/results/control_readiness_stage2'
CUTOFF=2.0
CONTROL=['control_left_elevon_normalized','control_right_elevon_normalized','control_flap_motor_normalized','control_rudder_normalized']


def filtered(frame):
    f=frame.copy(); tau=1/(2*np.pi*CUTOFF)
    for _,g in f.groupby('log_id',sort=False):
        t=g.timestamp_us.to_numpy()/1e6; x=g[CONTROL].to_numpy(); y=x.copy()
        valid=(g.valid_core&g.valid_logged_phase).to_numpy()
        boundary=(g.mode_boundary|g.reset_boundary|g.gap_boundary).to_numpy()
        for i in range(1,len(g)):
            if not valid[i] or not valid[i-1] or boundary[i] or t[i]-t[i-1]>.05 or not np.isfinite(y[i-1]).all(): continue
            dt=t[i]-t[i-1]; y[i]=y[i-1]+dt/(tau+dt)*(x[i]-y[i-1])
        f.loc[g.index,CONTROL]=y
    return f


def main():
    out=OUT/'bandwidth_sensitivity'; out.mkdir(exist_ok=False)
    provenance=json.loads((OUT/'dataset_provenance.json').read_text()); dataset=Path(provenance['manifest']).parent
    assert sha(dataset/'manifest.json')==provenance['manifest_sha256']
    manifest=json.loads((dataset/'manifest.json').read_text())
    raw_records=[]; results=[]; responses=[]; counts=[]
    protocol=None
    for part in ['train','validation']:
        path=dataset/f'samples_{part}.parquet'; assert sha(path)==provenance['artifacts'][path.name]
        frame=pd.read_parquet(path); rows,curves=candidates(filtered(frame),part,manifest,raw_records)
        if part=='train':
            h=rows.loc[rows.base_valid]
            protocol=dict(cutoff_hz=CUTOFF,time_constant_s=1/(2*np.pi*CUTOFF),causal=True,
                          scope='exploratory sensitivity after raw validation inspected; not primary selection or acceptance',
                          warmup_history_s=.5,raw_response_preserved=True,
                          envelope={k:[float(h[k].quantile(.05)),float(h[k].quantile(.95))] for k in ['speed','pitch_deg','frequency']},
                          excitation={c:float(max(.01,h['delta_'+c].abs().quantile(.75))) for c in CHANNELS},
                          interference={c:float(max(.01,h['excursion_'+c].quantile(.25))) for c in CHANNELS})
            dump(out/'protocol_before_filtered_validation.json',protocol)
        for channel in CHANNELS:
            last={}; kept=[]
            for r in rows.to_dict('records'):
                reasons=[] if r['base_valid'] else [r['base_reasons']]
                if any(not lo<=r[k]<=hi for k,(lo,hi) in protocol['envelope'].items()): reasons.append('outside_train_envelope')
                delta=r['delta_'+channel]
                if not np.isfinite(delta) or abs(delta)<protocol['excitation'][channel]: reasons.append('small_excitation')
                for other in CHANNELS:
                    if other!=channel and (not np.isfinite(r['excursion_'+other]) or r['excursion_'+other]>protocol['interference'][other]): reasons.append('coupled_'+other)
                if not np.isfinite(r['pre_range_'+channel]) or r['pre_range_'+channel]>.5*abs(delta): reasons.append('unstable_pre_command')
                if not np.isfinite(r['hold_'+channel]) or r['hold_'+channel]*np.sign(delta)<.5*abs(delta): reasons.append('not_held')
                if not reasons and r['timestamp_us']-last.get(r['log_id'],-1e12)<1_500_000: reasons.append('overlap')
                if not reasons: last[r['log_id']]=r['timestamp_us']; kept.append(r)
                results.append(dict(**r,channel=channel,accepted=not reasons,reasons='|'.join(reasons)))
            counts.append(dict(partition=part,channel=channel,accepted=len(kept),logs=len({r['log_id'] for r in kept})))
            axis={'symmetric_tail':1,'differential_tail':0,'rudder':2,'motor':3}[channel]
            for r in kept:
                curve=curves[r['id']][:,axis]; base=curve[:10].mean()
                for horizon in [.1,.2,.5,1.]:
                    j=int(round((horizon+.2)/.02)); change=float(curve[j]-base)
                    responses.append(dict(partition=part,channel=channel,log_id=r['log_id'],timestamp_us=r['timestamp_us'],horizon_s=horizon,filtered_command_delta=r['delta_'+channel],raw_response_change=change,apparent_gain=change/r['delta_'+channel]))
    pd.DataFrame(results).to_csv(out/'event_candidates_all.csv',index=False)
    pd.DataFrame(counts).to_csv(out/'event_counts.csv',index=False)
    pd.DataFrame(responses).to_csv(out/'event_responses.csv',index=False)
    dump(out/'manifest.json',dict(script_sha256=sha(Path(__file__)),primary_script_sha256=sha(ROOT/'scripts/audit_control_readiness_stage2.py'),raw_logs=raw_records,completed=True,sealed_test_opened=False,september19_opened=False))
    print(pd.DataFrame(counts).to_string(index=False))

if __name__=='__main__': main()
