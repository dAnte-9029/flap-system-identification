#!/usr/bin/env python3
"""Score fixed train templates using past-only reset phase, without validation fitting."""
import sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd
from prepare_main_v2_phase_reference import OLD,DATA
from system_identification.models.phase_reference import harmonic_design


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);a=pa.parse_args();out=a.output
    template=json.loads((out/'canonical_template.json').read_text());selection=json.loads((out/'selected_estimator.json').read_text());c=np.array(template['coefficients'])
    estimates=pd.read_csv(out/'phase_offset_estimation.csv');oracle=pd.read_csv(out/'oracle_offsets_NOT_DEPLOYABLE.csv');rows=[]
    for split in ['train','validation']:
        m=pd.read_csv(OLD/f'{split}_points.csv');data=np.load(OLD/f'{split}_diagnostic_arrays.npz');s=pd.read_parquet(DATA/f'samples_{split}.parquet');ts=s[s.valid_core].set_index(['log_id','segment_id','sample_in_segment']).timestamp_us
        dt=ts.loc[[(r.log_id,r.segment_id,r.sample_in_segment+1) for r in m.itertuples()]].to_numpy()*1e-6-m.timestamp_s.to_numpy()
        truth=data['target_D0_raw'][:,[4,2]]
        selected=estimates[(estimates.split==split)&(estimates.method==selection['method'])&(estimates.requested_history_steps==selection['history_steps'])].set_index('point_index').loc[np.arange(len(m))]
        offset_oracle=m.log_id.map(oracle[oracle.split==split].set_index('log_id').offset_rad).to_numpy()
        for contract,offset in [('P0',-m.phase.to_numpy()),('P1',np.zeros(len(m))),('P2',selected.offset_rad.to_numpy()),('P_oracle',offset_oracle)]:
            # Predicted next phase uses only f(t0), never the observed future phase.
            delta=2*np.pi*m.frequency.to_numpy()*dt;phi=m.phase.to_numpy()+offset+delta/2
            X=harmonic_design(phi)*np.repeat(np.stack([np.sinc(k*delta/(2*np.pi)) for k in [1,2,3]],axis=-1),2,axis=-1)
            pred=X@c
            for log in m.log_id.unique():
                sel=m.log_id.to_numpy()==log
                if split=='train':sel&=m.timestamp_s.to_numpy()>template['cuts'][log]+2
                for j,signal in enumerate(['q_dot','az_n']):
                    e=pred[sel,j]-truth[sel,j];y=truth[sel,j]
                    rows.append(dict(split=split,contract=contract,log_id=log,signal=signal,n=int(sel.sum()),rmse=float(np.sqrt(np.mean(e*e))),r2=float(1-np.sum(e*e)/np.sum((y-y.mean())**2)),
                        validation_fit=False,oracle_not_deployable=contract=='P_oracle'))
    pd.DataFrame(rows).to_csv(out/'phase_conditioned_consistency.csv',index=False)
    # Matched-history cohort prevents unequal eligible points driving comparisons.
    hold=estimates[estimates.train_temporal_holdout&estimates.complete_history]
    ids=set.intersection(*[set(g.point_index) for _,g in hold.groupby(['method','requested_history_steps'])])
    rows=[]
    for keys,g in hold[hold.point_index.isin(ids)].groupby(['method','requested_history_steps']):
        rows.append(dict(method=keys[0],history_steps=keys[1],n=len(g),cycles_median=g.observed_cycles.median(),duration_mean_s=g.duration_s.mean(),
            **{'p'+str(q)+'_offset_error_deg':float(np.rad2deg(g.reference_offset_error_rad.quantile(q/100))) for q in [50,75,90,95]}))
    pd.DataFrame(rows).to_csv(out/'matched_history_ablation.csv',index=False)

if __name__=='__main__':main()
