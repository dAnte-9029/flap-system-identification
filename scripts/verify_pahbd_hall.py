#!/usr/bin/env python3
"""Frozen train/validation Hall coverage and CPU numerical checks; no optimizer steps."""
import json,sys,subprocess,copy,argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from system_identification.data.pahbd import NativeWindows,configure_latent
from system_identification.models.pahbd import PAHBD,Config
from system_identification.training.pahbd import require_training_admission,window_loss,FlightBalancedSampler


def main():
    torch.set_num_threads(1);torch.manual_seed(17)
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='experiments/phase_coupled_mpc_v1/hall_contract_checks')
    out=Path(parser.parse_args().output);out.mkdir(exist_ok=False)
    cfg=json.loads(Path('configs/pahbd/default.json').read_text());datasets={};coverage=[];alignment=[];anomalies=[];runtime_checks={}
    for split in ('train','validation'):
        print('loading',split,flush=True)
        d=NativeWindows(**cfg['data'],split=split);datasets[split]=d
        f=d.windows.copy()
        f['date']=f.flight_id.map(lambda k:d.provenance[k]['flight_date'])
        f['firmware']=f.flight_id.map(lambda k:d.provenance[k]['firmware'])
        rows=[]
        for (fid,h),g in f.groupby(['flight_id','horizon_us']):
            rows.append(dict(split=split,flight_id=fid,date=g.date.iloc[0],sampling_group=d.provenance[fid]['sampling_group'],firmware=g.firmware.iloc[0],horizon_us=int(h),windows=len(g),shared=int(g.shared_periodic_eligible.sum()),exclusions=json.dumps(g.loc[~g.shared_periodic_eligible,'phase_exclusion'].value_counts().to_dict())))
        pd.DataFrame(rows).to_csv(out/f'{split}_flights.csv',index=False)
        for h,g in f.groupby('horizon_us'):
            good=g[g.shared_periodic_eligible]
            coverage.append(dict(split=split,horizon_us=int(h),flights=int(good.flight_id.nunique()),windows=len(good),average_only=len(g)-len(good),dates=int(good.date.nunique()),firmwares=int(good.firmware.nunique()),exclusions=g.loc[~g.shared_periodic_eligible,'phase_exclusion'].value_counts().to_dict()))
        f.groupby(['date','firmware','horizon_us']).agg(flights=('flight_id','nunique'),windows=('window_id','size'),shared=('shared_periodic_eligible','sum')).reset_index().to_csv(out/f'{split}_dates_firmware.csv',index=False)
        hall=pd.read_parquet(d.root/f'native_hall_event_{split}.parquet')
        for fid in d.provenance:
            t=d.phase_tables[fid];good=t[t.phase_usable];meta=d.provenance[fid]
            ts=t.timestamp_us.to_numpy();p=t.phase_rad.to_numpy();c=t.encoder_total_count.to_numpy(float)
            ht=hall.loc[hall.flight_id==fid,'timestamp_us'].to_numpy();j=np.searchsorted(ts,ht)
            ok=(j>0)&(j<len(ts));j=np.clip(j,1,len(ts)-1)
            ok &= (ts[j]-ts[j-1]<=30000)&t.phase_usable.to_numpy()[j]
            # Reconstruct phase AT Hall from the first following sample and encoder
            # bracket interpolation. Only offline sensor validation, never model input.
            hc=np.interp(ht,ts,c);err=np.angle(np.exp(1j*(p[j]-(c[j]-hc)*2*np.pi/(4096*meta['transmission_ratio']))))[ok]
            alignment.append(dict(split=split,flight_id=fid,date=meta['flight_date'],firmware=meta['firmware'],sampling_group=meta['sampling_group'],rows=len(t),usable=len(good),anomalies=int(t.phase_anomaly.sum()),hall_pairs=len(err),hall_abs_error_p95=float(np.quantile(abs(err),.95)) if len(err) else None,increment_error_p99=float(good.reference_error_rad.abs().quantile(.99)) if len(good) else None))
            for _,r in t[t.phase_anomaly].iterrows():
                anomalies.append(dict(split=split,flight_id=fid,date=meta['flight_date'],timestamp_us=int(r.timestamp_us),reference_error_rad=float(r.reference_error_rad)))
        print(coverage[-4:],flush=True)
        # Materialize one eligible window from every date/horizon: catches
        # metadata/runtime admission disagreement without neural training.
        checked=0
        for _,group in f[f.shared_periodic_eligible].groupby(['date','horizon_us']):
            window=d[int(group.index[0])]
            assert window.metadata['periodic_enabled']
            assert -window.start<=cfg['model']['replay_seconds']
            checked+=1
        runtime_checks[split]=checked
        print('runtime coverage groups checked',split,checked,flush=True)
    pd.DataFrame(alignment).to_csv(out/'sensor_alignment.csv',index=False)
    pd.DataFrame(anomalies).to_csv(out/'anomaly_locations.csv.gz',index=False)
    train=datasets['train'];val=datasets['validation']
    require_training_admission(cfg,Config(**cfg['model']),train.windows,val.windows)
    eligible=train.windows[train.windows.shared_periodic_eligible]
    sampler_a=FlightBalancedSampler(eligible);sampler_b=FlightBalancedSampler(eligible)
    assert [sampler_a.draw() for _ in range(100)]==[sampler_b.draw() for _ in range(100)]
    probes=[]
    # One 200 ms window per distinct retained flight date, at most three dates.
    eligible=eligible[eligible.horizon_us==200000].copy()
    eligible['date']=eligible.flight_id.map(lambda k:train.provenance[k]['flight_date'])
    dates=sorted(eligible.date.unique());selected=[dates[0],'2026-09-15',dates[-1]]
    for date in dict.fromkeys(selected):
        idx=int(eligible[eligible.date==date].index[0]);mc=Config(**cfg['model']);w=configure_latent(train[idx],mc)
        assert w.metadata['periodic_enabled']
        m=PAHBD(mc).double();seen=[];update=m.belief.update
        def inspect(s,e,dt):
            result=update(s,e,dt)
            if e.source=='wing_phase' and bool(e.mask[13]):
                if not bool(s.phase_quality.any()):assert torch.equal(result.phi,e.value[14:15])
                seen.append(float(result.phi))
            return result
        m.belief.update=inspect
        loss,_,belief,*_=window_loss(m,w);loss.backward()
        grads={k:float(getattr(m.flow.mixer,k).weight.grad.abs().sum()) for k in ('forcing','control')}
        assert all(v>0 and np.isfinite(v) for v in grads.values())
        assert all(e.availability<=0 and e.measurement<=e.availability for e in w.history)
        assert len(seen)>1 and min(np.diff(seen))>0
        assert torch.linalg.eigvalsh(belief.phase_covariance).min()>0
        with torch.no_grad():
            baseline=m.encode(w)
            # Poison every not-yet-available sensor reading, including timestamps
            # preceding decision whose publication guard has not expired.
            for name,tab in train.tables.items():
                if name.startswith('actuator'):continue
                av=tab.get('availability_us',tab.timestamp_us)
                fields={'wing_phase':['phase_rad','encoder_total_count'],'flap_frequency':['frequency_hz'],'vehicle_local_position':['x','y','z','vx','vy','vz'],'vehicle_attitude':[f'q[{i}]' for i in range(4)],'vehicle_angular_velocity':[f'xyz[{i}]' for i in range(3)]}[name]
                tab.loc[av>w.metadata['decision_us'],fields]=123456.
            poison=configure_latent(train[idx],mc);same=m.encode(poison)
            assert torch.equal(baseline.x,same.x) and torch.equal(baseline.phi,same.phi)
            # Future phase follows only the exact ZOH propagator and candidate motor.
            future=m.flow.rollout(baseline,[.2],w.controls)[0]
            candidate=copy.deepcopy(w.controls);candidate.values=candidate.values.clone();candidate.values[:,0]+=.1
            alternative=m.flow.rollout(baseline,[.2],candidate)[0]
            assert not torch.equal(future.phi,alternative.phi)
            b=PAHBD(Config(**dict(cfg['model'],mode='B'))).double();b.load_state_dict(m.state_dict())
            sb=b.flow.rollout(baseline,[.2],w.controls)[0]
            assert torch.allclose(future.phi,sb.phi,atol=1e-12)
        probes.append(dict(date=date,flight_id=w.metadata['flight_id'],window_id=w.metadata['window_id'],loss=float(loss.detach()),gradients=grads,hall_updates=len([e for e in w.history if e.source=='wing_phase' and e.measurement>=w.start and bool(e.mask[13])]),future_poison_invariant=True,AB_same_phase=True))
        # Reload after deliberately poisoning cached copies.
        from system_identification.data.pahbd_phase import prepare_phase
        fid=w.metadata['flight_id'];t=pd.read_parquet(train.root/'native_wing_phase_train.parquet',filters=[('flight_id','==',fid)])
        train.phase_tables[fid]=prepare_phase(t,train.provenance[fid]['transmission_ratio'],train.phase_contract)
        train._cached_id=None
        print('probe',probes[-1],flush=True)
    result=dict(code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),seed=17,code_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),runtime_materialized_date_horizon_groups=runtime_checks,manifest_sha256=train.manifest_sha256,phase_contract=train.phase_contract,coverage=coverage,probes=probes,admission='PASS conditional on recorded publication latency and mechanical working assumptions',formal_training_started=False)
    (out/'results.json').write_text(json.dumps(result,indent=2));print('PASS',flush=True)

if __name__=='__main__':main()
