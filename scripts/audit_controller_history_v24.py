"""Prepare, but do not train, a past-controller-state-only paired experiment."""
from pathlib import Path
import json
import subprocess
import numpy as np
import pandas as pd
from pyulog import ULog
import train_control_windows_v2 as base
from audit_manual_chain_v21 import PX4,COMMIT,hold
from controller_history_v24 import FEATURES,history_block,infer_scaling,fit_history_stats

OUT=base.ROOT/'docs/analysis/results/control_windows_v24'
V23=OUT.parent/'control_windows_v23'
TOPICS=['rate_ctrl_status','rate_ctrl_terms','gain_compression','vehicle_angular_velocity','airspeed_validated']
SOURCES=['src/lib/rate_control/rate_control.cpp','src/lib/rate_control/gain_compression.cpp',
         'src/modules/fw_rate_control/FixedwingRateControl.cpp','src/modules/fw_rate_control/FixedwingRateControl.hpp',
         'src/lib/mathlib/math/filter/AlphaFilter.hpp']


def main():
    delivery=json.loads((V23/'delivery_manifest.json').read_text())
    for name,digest in delivery['artifacts'].items():assert base.sha(V23/name)==digest
    for name,digest in delivery['code'].items():assert base.sha(base.ROOT/name)==digest
    parent=json.loads((V23/'dataset_manifest.json').read_text());w=pd.read_csv(V23/'windows.csv')
    assert len(w)==407 and (w.partition=='train').all()
    paths=[Path(__file__),base.ROOT/'scripts/controller_history_v24.py',base.ROOT/'tests/test_controller_history_v24.py',
           V23/'delivery_manifest.json',V23/'dataset_manifest.json',V23/'windows.csv']
    OUT.mkdir(exist_ok=False)
    base.write(OUT/'protocol.json',dict(stage='v24 controller history audit and materialization',
        pins={str(p.relative_to(base.ROOT)):base.sha(p) for p in paths},
        scope='same407 original train parents; only 26 historical instants per parent. No training, no outcome filtering, no validation/sealed/reserved.',
        source_features=FEATURES,
        integral='rate_ctrl_status post-update integral. Publication timestamp only. Different semantic time from pre-update rate_ctrl_terms.i_term.',
        compression='Published gains after compression.update; not necessarily the gains used by the same cycle torque. Use only as past observation plus publication age.',
        timing='origin-25..origin inclusive. Causal latest publication hold separately for each group. Age is publication age, never physical sampling latency.',
        scaling_diagnostic='Exact timestamp_sample join of rate_ctrl_terms and vehicle_angular_velocity. rate_sp_used=p_term/P+rate. scale=rate_sp_used*FF/ff_term; per-axis |ff_term|>=1e-3, positive finite ratios, largest-|ff_term| axis chosen. No time shifts, target response or rate reference hold. Remains diagnostic, excluded from training features.',
        integral_pairing='First status after terms publication within1ms, before next terms publication, one-to-one; timing consistency only because status has no timestamp_sample.',
        normalization='fit-only unique historical rows;8 mean/std floor1e-4, same normalization for both future comparison arms',
        phase_contract=parent['phase_contract'],frequency_contract=parent['frequency_contract'],sealed_test_accessed=False))
    (OUT/'source').mkdir();source_blobs={}
    for f in SOURCES:
        data=subprocess.check_output(['git','-C',str(PX4),'show',f'{COMMIT}:{f}']);dest=OUT/'source'/Path(f).name;dest.write_bytes(data)
        source_blobs[f]=dict(blob=subprocess.check_output(['git','-C',str(PX4),'rev-parse',f'{COMMIT}:{f}'],text=True).strip(),snapshot=str(dest.relative_to(OUT)),sha256=base.sha(dest))
    base.write(OUT/'source_provenance.json',dict(commit=COMMIT,files=source_blobs))
    features=np.zeros((407,26,8),np.float32);pubs=np.zeros((407,26,2),np.int64);queries=np.zeros((407,26),np.int64)
    grids=[];summary=[];scale_rows=[];integral_rows=[];fit_rows={};pins=[]
    for entry in parent['sources']:
        log=entry['log_id'];src=entry['source'];assert src['partition']=='train' and base.sha(src['path'])==src['sha256']
        assert base.sha(base.ROOT/entry['cache'])==entry['cache_sha256']
        with np.load(base.ROOT/entry['cache']) as z:timestamps=z['timestamp_us']
        g=w[w.log_id==log];idx=np.unique(np.concatenate([np.arange(r.origin_index-25,r.origin_index+1) for r in g.itertuples()]))
        q=timestamps[idx].astype(np.int64)
        u=ULog(src['path'],message_name_filter_list=TOPICS);assert u.msg_info_dict['ver_sw']==COMMIT and not u.changed_parameters
        d={t.name:t.data for t in u.data_list if t.multi_id==0};assert set(TOPICS)<=d.keys()
        status=d['rate_ctrl_status'];gain=d['gain_compression'];terms=d['rate_ctrl_terms'];gyro=d['vehicle_angular_velocity'];air=d['airspeed_validated']
        block,si,gi=history_block(status,gain,q)
        assert (block[:,3:6]>=u.initial_parameters['FW_GC_GAIN_MIN']-1e-6).all() and (block[:,3:6]<=1+1e-6).all()
        fit_rows[log]=(idx,block)
        grid=pd.DataFrame(block,columns=FEATURES);grid.insert(0,'timestamp_us',q);grid.insert(0,'cache_index',idx);grid.insert(0,'log_id',log)
        grid['status_publication_us']=status['timestamp'][si];grid['compression_publication_us']=gain['timestamp'][gi]
        grids.append(grid)
        for r in g.itertuples():
            ii=np.searchsorted(idx,np.arange(r.origin_index-25,r.origin_index+1))
            features[r.array_index]=block[ii];queries[r.array_index]=q[ii]
            pubs[r.array_index,:,0]=status['timestamp'][si[ii]];pubs[r.array_index,:,1]=gain['timestamp'][gi[ii]]
            assert np.all(pubs[r.array_index]<=queries[r.array_index,:,None])
            assert queries[r.array_index,-1]==timestamps[r.origin_index]
        # Audit only the source rows actually held in the histories.
        tx=np.unique(hold(terms,q));tt=terms['timestamp'][tx].astype(np.int64)
        sx=np.searchsorted(status['timestamp'],tt,side='left')
        valid=sx<len(status['timestamp']);sx=np.minimum(sx,len(status['timestamp'])-1)
        gap=status['timestamp'][sx].astype(np.int64)-tt
        nxt=terms['timestamp'][np.minimum(tx+1,len(terms['timestamp'])-1)].astype(np.int64)
        valid &= (gap>=0)&(gap<=1000)&(status['timestamp'][sx]<nxt)
        assert len(np.unique(sx[valid]))==int(valid.sum())
        pre=np.column_stack([terms[f'i_term[{j}]'][tx] for j in range(3)])
        post=np.column_stack([status[f'{ax}speed_integ'][sx] for ax in ['roll','pitch','yaw']])
        for local in np.flatnonzero(valid):
            integral_rows.append(dict(log_id=log,terms_timestamp_us=int(tt[local]),status_timestamp_us=int(status['timestamp'][sx[local]]),
                gap_us=int(gap[local]),**{f'pre_{j}':float(pre[local,j]) for j in range(3)},
                **{f'post_{j}':float(post[local,j]) for j in range(3)}))
        sample=terms['timestamp_sample'][tx].astype(np.int64);gx=np.searchsorted(gyro['timestamp_sample'],sample)
        inrange=gx<len(gyro['timestamp_sample']);gx=np.minimum(gx,len(gyro['timestamp_sample'])-1)
        exact=inrange&(gyro['timestamp_sample'][gx]==sample)
        tr=tx[exact];gr=gx[exact]
        pp=np.column_stack([terms[f'p_term[{j}]'][tr] for j in range(3)])
        ff=np.column_stack([terms[f'ff_term[{j}]'][tr] for j in range(3)])
        rates=np.column_stack([gyro[f'xyz[{j}]'][gr] for j in range(3)])
        pg=[u.initial_parameters[f'FW_{axis}R_P'] for axis in ['R','P','Y']]
        fg=[u.initial_parameters[f'FW_{axis}R_FF'] for axis in ['R','P','Y']]
        sc,ratios,used=infer_scaling(pp,ff,rates,pg,fg)
        available=np.maximum(terms['timestamp'][tr],gyro['timestamp'][gr]).astype(np.int64)
        ai=hold(air,available);airv=air['calibrated_airspeed_m_s'][ai].astype(float)
        raw_proxy=u.initial_parameters['FW_AIRSPD_TRIM']/np.maximum(airv,u.initial_parameters['FW_AIRSPD_STALL'])
        for i in range(len(tr)):
            finite=ratios[i,np.isfinite(ratios[i])]
            scale_rows.append(dict(log_id=log,terms_timestamp_us=int(terms['timestamp'][tr[i]]),sample_us=int(sample[exact][i]),
                available_us=int(available[i]),scale=float(sc[i]),axes_available=len(finite),
                axis_spread=float(np.ptp(finite)) if len(finite)>1 else None,raw_unfiltered_proxy=float(raw_proxy[i]),
                **{f'scale_{j}':float(ratios[i,j]) for j in range(3)}))
        summary.append(dict(log_id=log,windows=len(g),history_points=len(q),integral_source_rows=len(np.unique(si)),compression_source_rows=len(np.unique(gi)),
            integral_age_p50_ms=float(np.median(block[:,6])*1000),integral_age_p95_ms=float(np.quantile(block[:,6],.95)*1000),integral_age_max_ms=float(block[:,6].max()*1000),
            compression_age_p50_ms=float(np.median(block[:,7])*1000),compression_age_p95_ms=float(np.quantile(block[:,7],.95)*1000),compression_age_max_ms=float(block[:,7].max()*1000),
            terms_rows=len(tx),integral_timing_pairs=int(valid.sum()),exact_gyro_sample_pairs=int(exact.sum()),scaling_identifiable_rows=int(np.isfinite(sc).sum()),
            FW_GC_EN=u.initial_parameters['FW_GC_EN'],FW_GC_GAIN_MIN=u.initial_parameters['FW_GC_GAIN_MIN'],
            FW_ARSP_SCALE_EN=u.initial_parameters['FW_ARSP_SCALE_EN'],FW_USE_AIRSPD=u.initial_parameters['FW_USE_AIRSPD']))
        pins.append(entry)
        print(f'Audited controller history {log}',flush=True)
    np.savez_compressed(OUT/'history_controller.npz',array_index=w.array_index.to_numpy(),history_controller=features,publication_us=pubs,query_us=queries)
    pd.concat(grids,ignore_index=True).to_csv(OUT/'history_points.csv',index=False)
    pd.DataFrame(summary).to_csv(OUT/'per_flight.csv',index=False)
    pd.DataFrame(scale_rows).to_csv(OUT/'scaling_diagnostic.csv',index=False)
    pd.DataFrame(integral_rows).to_csv(OUT/'integral_timing_pairs.csv',index=False)
    for fold in range(5):
        info=json.loads((V23/f'fold{fold}/manifest.json').read_text());train=w.iloc[info['fit_indices']];query=w.iloc[info['query_indices']]
        assert set(train.log_id).isdisjoint(query.log_id)
        vals=[]
        for log,g in train.groupby('log_id'):
            ids=np.unique(np.concatenate([np.arange(i-25,i+1) for i in g.origin_index]));index,block=fit_rows[log]
            vals.append(block[np.searchsorted(index,ids)])
        mean,scale=fit_history_stats(np.concatenate(vals));dest=OUT/f'fold{fold}';dest.mkdir()
        np.savez(dest/'controller_normalization.npz',mean=mean,scale=scale)
        base.write(dest/'manifest.json',dict(fit_indices=info['fit_indices'],query_indices=info['query_indices'],
            source_fold_manifest_sha256=base.sha(V23/f'fold{fold}/manifest.json'),normalization_sha256=base.sha(dest/'controller_normalization.npz')))
    base.write(OUT/'dataset_manifest.json',dict(dataset_id='controller_history_train_v24',features=FEATURES,shape=list(features.shape),
        arrays_sha256=base.sha(OUT/'history_controller.npz'),protocol_sha256=base.sha(OUT/'protocol.json'),parent_dataset_manifest_sha256=base.sha(V23/'dataset_manifest.json'),
        sources=pins,phase_contract=parent['phase_contract'],frequency_contract=parent['frequency_contract'],
        use='Only encoder past observations; no future controller state tensor exists. Combine with unchanged v23 train inputs/labels.',sealed_test_accessed=False))


if __name__=='__main__':main()
