"""Descriptive 1s rate spectra and nominal-horizon response diagnostics."""
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from train_reference_unit_gravity import ART,model_from
from control_windows_v3 import predict
from audit_reference_v4 import OUT

def main():
    torch.set_num_threads(1);x,y,_=base.load();w=pd.read_csv(base.DATA/'windows_v2.csv');rows=[];wave=[]
    model=model_from(ART/'A_long_17/final.pt');pred=predict(model,x,w.array_index.to_numpy())
    freq=np.fft.rfftfreq(50,.02)
    for row in w.itertuples():
        j=row.array_index;truth=y['rates_b'][j].numpy();pp=pred['rates_b'][j];err=pp-truth
        for name,signal in [('truth',truth),('error',err)]:
            # Rectangular 1s record, 1Hz bins; descriptive energy, not system FRF.
            ft=np.fft.rfft(signal-signal.mean(0),axis=0);power=np.abs(ft)**2;power[1:-1]*=2
            for ax,axis in enumerate(['p','q','r']):
                den=max(power[:,ax].sum(),1e-15)
                rows.append(dict(array_index=j,log_id=row.log_id,partition=row.partition,signal=name,axis=axis,
                    low_le2=float(power[(freq>0)&(freq<=2),ax].sum()/den),mid_2to8=float(power[(freq>2)&(freq<=8),ax].sum()/den),
                    high_gt8=float(power[freq>8,ax].sum()/den),dominant_hz=float(freq[np.argmax(power[:,ax])]),
                    current_flap_hz=float(x['history_state'][j,-1,11])))
        for step in [25,50]:
            for ax,axis in enumerate(['p','q','r']):
                initial=float(x['initial_rates_b'][j,ax]);real=truth[:step,ax]-initial;sim=pp[:step,ax]-initial
                past=x['history_state'][j,:,3+ax].numpy();mad=np.median(np.abs(past-np.median(past)))
                amp=np.max(np.abs(real));measurable=amp>=max(3*mad,.05)
                wave.append(dict(array_index=j,log_id=row.log_id,partition=row.partition,axis=axis,horizon_s=step*.02,
                    active=bool(getattr(row,('roll' if ax==0 else 'pitch')+'_range_rad')>=np.deg2rad(1)) if ax<2 else False,
                    measurable=measurable,amplitude_error=abs(amp-np.max(np.abs(sim))),
                    peak_time_error_s=.02*abs(np.argmax(np.abs(real))-np.argmax(np.abs(sim))),waveform_rmse=np.sqrt(np.mean((real-sim)**2))))
    pd.DataFrame(rows).to_csv(OUT/'spectral_energy.csv',index=False)
    pd.DataFrame(rows).groupby(['partition','signal','axis'])[['low_le2','mid_2to8','high_gt8','dominant_hz','current_flap_hz']].median().to_csv(OUT/'spectral_summary.csv')
    pd.DataFrame(wave).to_csv(OUT/'nominal_waveforms_baseline.csv',index=False)
    base.write(OUT/'spectral_scope.json',dict(script_sha256=base.sha(__file__),model_sha256=base.sha(ART/'A_long_17/final.pt'),
        band_edges_hz=[0,2,8,25],resolution_hz=1,scope='descriptive centered1s energy, not transfer function or proof of periodic origin',
        waveform='0.5/1s local initialization; all windows retained, measurability only tags timing diagnostics; peak-time difference not actuator delay'))
if __name__=='__main__':main()
