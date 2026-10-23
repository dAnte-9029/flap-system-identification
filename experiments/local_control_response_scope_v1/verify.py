"""Independent file/partition/count checks and scalar native-command recomputation.

Does not import the analysis or original audit. Reads only development artifacts.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def pack_disjoint(p,meta,byflight=False):
    spans={};keep=[]
    for idx,r in p.sort_values(['score','i','j'],kind='stable').iterrows():
        pair=[meta.iloc[int(r.i)],meta.iloc[int(r.j)]]
        ok=True
        for w in pair:
            for lo,hi in spans.get(w.log_sha,[]):
                if byflight or not(w.pair_support_end_us<lo or w.history_start_us>hi):ok=False
        if ok:
            keep.append(idx)
            for w in pair:spans.setdefault(w.log_sha,[]).append((w.history_start_us,w.pair_support_end_us))
    return keep


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--results',type=Path,default=HERE/'results')
    a=ap.parse_args()
    cfg=json.loads((HERE/'config.json').read_text())
    frozen=json.loads((a.results/'frozen_selection.json').read_text())
    assert frozen['config_sha256']==sha(HERE/'config.json')
    assert frozen['analysis_sha256']==sha(HERE/'analyze.py')
    f=pd.read_csv(a.data_root/'flights.csv')
    flights={};dates={};boot={};counts={};errors=[];numeric_checks=0
    provenance=json.loads((a.data_root/'flight_provenance.json').read_text())
    for split in ['train','validation']:
        out=a.results/split
        receipt=json.loads((out/'source_receipt.json').read_text())
        assert receipt['partitions']==[split] and not receipt['test_artifacts_opened']
        assert not any(c.startswith('target_') for c in receipt['sample_columns'])
        for name,h in receipt['artifact_sha256'].items():
            assert '_test.' not in name and sha(a.data_root/name)==h
        meta=pd.read_csv(out/'candidate_index.csv.gz')
        p=pd.read_csv(out/'all_history_pairs.csv.gz',index_col='pair_id')
        sel=pd.read_csv(out/'selected_pairs.csv')
        su=pd.read_csv(out/'pair_support.csv')
        ev=pd.read_csv(out/'command_events.csv.gz',index_col='candidate_id')
        assert meta.window_id.is_unique and meta.candidate_id.tolist()==list(range(len(meta)))
        assert set(meta.split)=={split} and set(meta.sampling_group)=={'main_50hz'}
        assert set(meta.flight_id)<=set(f.loc[(f.split==split)&(f.sampling_group=='main_50hz'),'flight_id'])
        assert (meta.command_end_us-meta.decision_us==500000).all()
        assert (meta.history_start_us<meta.decision_us).all()
        flights[split]=set(meta.log_sha)
        dates[split]=set(meta.date)
        boot[split]={(x['sys_uuid'],x['boot_time_utc_us']) for x in provenance if x['flight_id'] in set(meta.flight_id)}
        assert (meta.log_sha.to_numpy()[p.i]!=meta.log_sha.to_numpy()[p.j]).all()
        assert (meta.nav_state.to_numpy()[p.i]==meta.nav_state.to_numpy()[p.j]).all()
        assert (meta.hardware_uuid.to_numpy()[p.i]==meta.hardware_uuid.to_numpy()[p.j]).all()
        assert not p.duplicated(['i','j']).any() and (p.i<p.j).all()
        error_cols=['v_rms','attitude_rms_deg','rate_rms','frequency_rms']+['hist_'+c for c in ['motor','left_elevon','right_elevon','rudder']]
        error_cols+=['v_initial','attitude_initial_deg','rate_initial','frequency_initial']+['initial_'+c for c in ['motor','left_elevon','right_elevon','rudder']]
        assert (p[error_cols].to_numpy()<=np.tile([.75,15,.75,.5]+[.05]*4,2)).all()
        strict=(p[error_cols].to_numpy()<=np.tile([.5,10,.5,.3]+[.03]*4,2)).all(axis=1)
        assert np.array_equal(strict,p.strict.to_numpy())
        for row in su.itertuples():
            inreg=meta[row.regime].to_numpy()
            q=p[inreg[p.i] & inreg[p.j] & p[row.coordinate]]
            ids=pack_disjoint(q,meta)
            assert len(q)==row.raw_pairs and len(ids)==row.nonoverlap_pairs
            assert len(pack_disjoint(q,meta,True))==row.flight_disjoint_pairs
            stored=sel[(sel.regime==row.regime)&(sel.coordinate==row.coordinate)].pair_id.tolist()
            assert ids==stored
        # Supplemental backgrounds are filtered before their own disjoint packing.
        bg=pd.read_csv(a.results/'description/background_comparability.csv')
        bi=pd.read_csv(a.results/'description/background_index.csv')
        for r in bg[bg.split==split].itertuples():
            rr=meta[r.regime].to_numpy()
            q=p[rr[p.i]&rr[p.j]]
            qa=ev.quiet.to_numpy()[q.i];qb=ev.quiet.to_numpy()[q.j]
            masks={'both_future_quiet':qa&qb,'differential_vs_quiet':q.differential&(qa^qb),
                   'motor_vs_quiet':q.motor&(qa^qb),'small_pair_and_both_quiet':q.background_pair&qa&qb}
            q=q[masks[r.cohort]]
            ids=pack_disjoint(q,meta)
            assert len(q)==r.raw_pairs and len(ids)==r.nonoverlap_pairs
            assert len(pack_disjoint(q,meta,True))==r.flight_disjoint_pairs
            assert ids==bi[(bi.split==split)&(bi.cohort==r.cohort)].pair_id.tolist()
        # Rebuild finite-piece contrasts from the two separate source message streams.
        motor=pd.read_parquet(a.data_root/f'native_actuator_motors_{split}.parquet')
        servo=pd.read_parquet(a.data_root/f'native_actuator_servos_{split}.parquet')
        ms=dict(tuple(motor.groupby('flight_id')));ss=dict(tuple(servo.groupby('flight_id')))
        def value(fid,t):
            mm=ms[fid];sv=ss[fid]
            im=np.searchsorted(mm.timestamp_us,t,side='right')-1
            iv=np.searchsorted(sv.timestamp_us,t,side='right')-1
            return np.array([mm['control[0]'].iloc[im],sv['control[0]'].iloc[iv],sv['control[1]'].iloc[iv],sv['control[2]'].iloc[iv]],float)
        def coordinate(u):return np.array([u[0],(u[1]+u[2])*.5,(u[2]-u[1])*.5,u[3]])
        chosen=p.iloc[np.unique(np.linspace(0,len(p)-1,min(24,len(p))).astype(int))]
        for pid,r in chosen.iterrows():
            x,y=meta.iloc[int(r.i)],meta.iloc[int(r.j)]
            times=[0,500000]
            for w in [x,y]:
                for stream in [ms[w.flight_id],ss[w.flight_id]]:
                    ts=stream.timestamp_us.to_numpy()-w.decision_us
                    times.extend(ts[(ts>0)&(ts<500000)].tolist())
            times=sorted(set(times));s1=np.zeros(4);s2=np.zeros(4)
            run=np.zeros((2,4));longest=np.zeros(4)
            for lo,hi in zip(times[:-1],times[1:]):
                d=coordinate(value(x.flight_id,x.decision_us+lo)-value(y.flight_id,y.decision_us+lo))
                dt=hi-lo;s1+=d*dt/500000;s2+=d*d*dt/500000
                for sign_index,sign in enumerate([1,-1]):
                    run[sign_index]=np.where(sign*d>=cfg['change_thresholds'],run[sign_index]+dt,0)
                    longest=np.maximum(longest,run[sign_index])
            for j,ch in enumerate(cfg['coordinates']):
                for suffix,computed in [('mean',s1[j]),('rms',np.sqrt(s2[j])),('longest_us',longest[j])]:
                    saved=r[f'difference_{ch}_{suffix}'];gap=abs(saved-computed)
                    errors.append(gap);numeric_checks+=1
                    assert np.isclose(saved,computed,atol=2e-9,rtol=1e-8),(split,pid,ch,suffix,saved,computed)
        counts[split]=dict(candidates=len(meta),history_pairs=len(p),selected_memberships=len(sel),
                           numerical_pairs=len(chosen),flights=len(flights[split]),dates=len(dates[split]))
    assert flights['train'].isdisjoint(flights['validation'])
    assert dates['train'].isdisjoint(dates['validation'])
    assert boot['train'].isdisjoint(boot['validation'])
    result=dict(passed=True,partitions=['train','validation'],counts=counts,numerical_checks=numeric_checks,
                maximum_scalar_difference=max(errors),test_opened=False,model_loaded=False,
                flight_date_boot_disjoint=True,verification_code_sha256=sha(Path(__file__)))
    (a.results/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
