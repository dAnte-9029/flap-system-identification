"""Preserve original windows, add causal100Hz native-rate samples and filter state."""
import json
import numpy as np
import pandas as pd
import torch
from scipy.signal import freqz
from pyulog import ULog
import train_control_windows_v2 as base
from audit_control_reference_semantics import past_values
from reference_bandwidth_v5 import OUT,DATA
from system_identification.models.reference_antialias_readout import antialias_taps,antialias_rate_readout


def main():
    OUT.mkdir(exist_ok=False);DATA.mkdir(exist_ok=False)
    oldx,oldy,stats=base.load();w=pd.read_csv(base.DATA/'windows_v2.csv')
    v4=base.ROOT/'docs/analysis/results/control_windows_v4';audit=json.loads((v4/'audit.json').read_text());tiny=json.loads((v4/'protocol.json').read_text())['tiny_cohort']
    taps=antialias_taps(torch.float64).numpy();f,h=freqz(taps,worN=32768,fs=100)
    protocol=dict(dataset='same1146 full-support-disjoint windows,407train739validation,14authorizedlogs',
        parent_manifest_sha256=base.sha(base.DATA/'manifest.json'),source_pins=audit['source_pins'],script_sha256=base.sha(__file__),
        reference_model='PX4 inner-loop plus aircraft; future planned5channel reference only',
        grid='internal100Hz dt.01,51state history samples[-.5,0],100forecast samples[.01,1]; no support expansion',
        other_channels='causal hold of original50Hz cache values; does not create new100Hz measurements; reference held between20ms updates',
        actuator_history='one internal10ms interval before each observed history state; original50Hz commands causally held',
        phase_contract='September logged wing_phase physical zero unconfirmed; original50Hz phase samples held; currentfrequencyheld for prediction',
        frequency_ratio=7.909090995788574,normalization='frozen originalA-train statistics, same for both arms; no fitting to validation',
        filter=dict(taps=taps.tolist(),sample_hz=100,readout_hz=50,cutoff_hz=15,linear_phase_delay_s=.1,
            past_state='last20 observed raw100Hz rates',initialization='no zero padding, no future sample, no outside-window support',
            stopband_max_gain_db=float(20*np.log10(abs(h[f>=25]).max())),passband_0to8_min_gain_db=float(20*np.log10(abs(h[f<=8]).min()))),
        arms=dict(raw100='sameGRU64 physical derivative rollout100Hz, raw100Hz rate loss',aa50='same100Hz physical latent dynamics, filter predicted and measured rates BEFORE50Hz readout/loss; q/v loss unchanged at20ms'),
        tiny_cohort=tiny,tiny_seed=17,tiny_updates=1600,tiny_batch=8,
        tiny_gate='raw100: >=15percent lower train-fit8 mean(p,q) full1s waveform RMSE at common20ms times than frozen tiny50. aa50: >=15percent lower SAME filtered-target score than raw100. Each requires no fit8 pose/velocity waveform metric >10percent worse. No validation-dependent launch.',
        formal_if_triggered='each passing arm:407train,3seeds17/23/42,1600updates,batch32,unchangedAdamW3e-4wd1e-4clip1; fixed final checkpoint',
        formal_comparison='raw100 vs A1600 on common raw50 targets; aa50 vs raw100 through identical FIR (train raw100 comparator if aa50 triggers); no cross-target RMSE win claimed',
        promotion='>=5percent activep/q0.5/1s improvement, everyseed positive, no all-metric >5percent regression; raw/filtered rate target explicitly separated;0.5s activep/q<=.15rad/s attitude<=3deg velocity<=.5m/s eachseed plus independent response evidence',
        budget='same optimizer updates, raw100 costs more recurrent steps than historical50Hz baseline; compute difference reported',
        sealed_test_accessed=False,validation_status='development set already repeatedly viewed')
    base.write(OUT/'protocol.json',protocol);xx={k:[] for k in oldx};yy={k:[] for k in oldy};byindex={};native=[]
    cm=json.loads((base.ROOT/'docs/analysis/results/reference_dynamics_data_v1/training_contract/manifest.json').read_text())
    for cache,group in w.groupby('cache'):
        assert base.sha(base.ROOT/cache)==cm['cache_sha256'][cache]
        with np.load(base.ROOT/cache) as z:a={k:z[k] for k in z.files}
        log=group.log_id.iloc[0];path=next(p for p in audit['source_pins'] if p.endswith(log));assert base.sha(path)==audit['source_pins'][path]
        u=ULog(path,message_name_filter_list=['vehicle_angular_velocity']);d=next(d.data for d in u.data_list if d.name=='vehicle_angular_velocity' and d.multi_id==0)
        for row in group.itertuples():
            i=int(row.origin_index);j=int(row.array_index);offset=np.arange(-50,101);times=a['timestamp_us'][i]+offset*10000;ci=i+np.floor_divide(offset,2)
            rates,pa,sa=past_values(d,['xyz[0]','xyz[1]','xyz[2]'],times)
            assert np.isfinite(rates).all() and (pa>=0).all() and (sa>=0).all() and max(pa.max(),sa.max())<=.06
            np.testing.assert_allclose(rates[::2],a['rates_b'][ci[::2]],atol=1e-6,rtol=1e-6)
            state=a['state_features'][ci[:51]].copy();state[:,3:6]=rates[:51]
            phi=a['phase_rad'][ci[:51]]-a['phase_rad'][i];state[:,9]=np.sin(phi);state[:,10]=np.cos(phi)
            current={k:v[j].numpy().copy() for k,v in oldx.items()}
            current.update(history_state=state,history_reference=a['references'][ci[:51]].copy(),
                history_actuator=a['actuator_commands'][i+np.floor_divide(np.arange(-51,0),2)].copy(),
                future_reference=a['references'][i+np.floor_divide(np.arange(100),2)].copy(),dt_s=np.full(100,.01,np.float32))
            target=dict(rates_b=rates[51:].astype(np.float32),quaternion=a['quaternion'][ci[51:]],velocity_n=a['velocity_n'][ci[51:]])
            for k in target:np.testing.assert_allclose(target[k][1::2],oldy[k][j].numpy(),atol=1e-6,rtol=1e-6)
            np.testing.assert_allclose(current['history_state'][::2],oldx['history_state'][j],atol=1e-6,rtol=1e-6)
            # Original full support starts at-520ms. First actuator lookback is-510ms.
            assert times[0]>=row.start_timestamp_us and times[-1]<=row.end_timestamp_us
            byindex[j]=(current,target)
            native.append(dict(array_index=j,window_id=row.window_id,log_id=log,partition=row.partition,
                sample_age_max_s=float(sa.max()),publication_age_max_s=float(pa.max()),unique_sample_timestamps=len(np.unique(times-np.rint(sa*1e6).astype(np.int64)))))
    for j in range(len(w)):
        current,target=byindex[j]
        for k in xx:xx[k].append(current[k])
        for k in yy:yy[k].append(target[k])
    payload={**{'input_'+k:np.stack(v) for k,v in xx.items()},**{'label_'+k:np.stack(v) for k,v in yy.items()}}
    np.savez_compressed(DATA/'windows_100hz.npz',**payload)
    filtered=antialias_rate_readout(torch.tensor(payload['input_history_state'][:,:,3:6]),torch.tensor(payload['label_rates_b'])).numpy()
    np.savez_compressed(DATA/'antialiased_50hz_labels.npz',array_index=w.array_index.to_numpy(),rates_b=filtered,
        quaternion=payload['label_quaternion'][:,1::2],velocity_n=payload['label_velocity_n'][:,1::2],
        filter_history_raw100=payload['input_history_state'][:,-20:,3:6],taps=taps)
    w.to_csv(OUT/'windows.csv',index=False);pd.DataFrame(native).to_csv(OUT/'native_sampling.csv',index=False)
    pins={str(p.relative_to(base.ROOT)):base.sha(p) for p in [DATA/'windows_100hz.npz',DATA/'antialiased_50hz_labels.npz',OUT/'windows.csv',base.ROOT/'dataset/control_windows_v2/normalization.npz']}
    base.write(OUT/'dataset_manifest.json',dict(status='complete',pins=pins,protocol_sha256=base.sha(OUT/'protocol.json'),source_contract_sha256=base.sha(base.ROOT/'docs/analysis/results/reference_dynamics_data_v1/training_contract/manifest.json'),
        phase_contract=protocol['phase_contract'],frequency_ratio=protocol['frequency_ratio'],partitions=['train','validation'],windows=len(w),same_original_support=True,
        note='aa50 labels are delayed rate observations, not physical rates for direct attitude integration; model inputs retain raw100 history and known filter state',sealed_test_accessed=False))
    print(json.dumps(dict(status='complete',windows=len(w),filter=protocol['filter']),indent=2))
if __name__=='__main__':main()
