"""Train/validation Stabilized manual-roll joint-command audit; no inference."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/flap-paper-mpl')
from pathlib import Path
import json,subprocess
import numpy as np
import pandas as pd
from pyulog import ULog
from audit_angular_rate_measurement import sha,past_index
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'dataset/trajectory_v3_september_expanded';OUT=ROOT/'docs/analysis/results/stabilized_roll_chain_v1'

def align(data,key,ref,age):
    if data is None or key not in data:return np.full(len(ref),np.nan),np.zeros(len(ref),bool)
    ix=past_index(data['timestamp'],ref);safe=np.maximum(ix,0);delta=(ref-np.asarray(data['timestamp'],np.int64)[safe])*1e-6
    x=np.asarray(data[key],float)[safe];ok=(ix>=0)&(delta<=age)&np.isfinite(x)
    return x,ok

def states(x):
    result=[];last=0
    for v in x:
        if abs(v)<=.05:last=0
        elif abs(v)>=.1:last=int(np.sign(v))
        result.append(last)
    return np.array(result)

def euler(q):
    w,x,y,z=q.T
    return np.c_[np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)),np.arcsin(np.clip(2*(w*y-z*x),-1,1))]*180/np.pi

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'checks.json').exists()
    manifest=json.loads((DATA/'manifest.json').read_text());roots=list(map(Path,manifest['source']['roots']))
    entries=[]
    for split in ['train','validation']:
        for flight in manifest['split_contract']['assignments'][split]:
            matches=[p/flight for p in roots if (p/flight).exists()];assert matches
            entries.append(dict(split=split,flight_id=flight,path=str(matches[0]),sha256=manifest['source']['ulog_sha256'][flight]))
    pins={str(DATA/f'samples_{s}.parquet'):sha(DATA/f'samples_{s}.parquet') for s in ['train','validation']}
    for p,h in pins.items():assert h==manifest['artifact_sha256'][Path(p).name]
    (OUT/'protocol.json').write_text(json.dumps(dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),logs=entries,pins=pins,
      scope='all58 train/validation flights; no heldout, training or model inference',
      selection='cached valid_core and nav_state==15 (STAB), confirmed manual+attitude+rates enabled and climb_rate disabled; publication past-only joins',
      freshness_s={'manual':.2,'attitude_setpoint':.05,'rates_setpoint':.05,'control_mode':2.},
      activity='descriptive fixed thresholds: stick yaw neutral abs<=0.05; roll hysteresis active abs>=0.10, neutral abs<=0.05. No threshold tuning or model-error screening.',
      episode='Continuous fresh valid Stabilized rows, gap<=50ms; active same-sign roll runs duration>=0.2s. Full cycle requires adjacent neutral pre/post blocks each>=0.2s. Not causal step or independence.',
      control_coordinates='common=(left+right)/2; differential=(left-right)/2; rudder fourth model input; postallocation normalized commands, not measured deflections.',
      pilot_access='Sep18 first admitted log inspected for topic schema before protocol; no blind preregistration claim.'),ensure_ascii=False,indent=2))
    cached={s:pd.read_parquet(DATA/f'samples_{s}.parquet') for s in ['train','validation']}
    rows=[];episodes=[];traces=[];params=[];firmware=set();origins=[]
    for entry in entries:
        assert sha(entry['path'])==entry['sha256']
        with (OUT/'access_log.jsonl').open('a') as f:f.write(json.dumps(dict(time=pd.Timestamp.now(tz='UTC').isoformat(),flight=entry['flight_id']))+'\n')
        u=ULog(entry['path'],message_name_filter_list=['manual_control_setpoint','vehicle_control_mode','vehicle_attitude_setpoint','vehicle_rates_setpoint'])
        fw=str(u.msg_info_dict.get('ver_sw','unknown'));firmware.add(fw);ds={d.name:d.data for d in u.data_list if d.multi_id==0}
        par=u.initial_parameters
        for k in ['FW_RLL_TO_YAW_FF','FW_MAN_R_MAX','FW_MAN_YR_MAX','FW_YR_P','FW_YR_I','FW_YR_FF']:
            params.append(dict(flight_id=entry['flight_id'],firmware=fw,parameter=k,initial_value=par.get(k),changes=json.dumps([(int(t),v) for t,name,v in u.changed_parameters if name==k])))
        f=cached[entry['split']];f=f[f.log_id==entry['flight_id']].reset_index(drop=True);ref=f.timestamp_us.to_numpy(np.int64);dt=np.r_[0,np.diff(ref)*1e-6]
        rawstab=f.valid_core.to_numpy(bool)&(f.nav_state.to_numpy()==15)
        ok=rawstab.copy();vals={}
        for name,expected in [('flag_control_manual_enabled',1),('flag_control_attitude_enabled',1),('flag_control_rates_enabled',1),('flag_control_climb_rate_enabled',0)]:
            v,k=align(ds.get('vehicle_control_mode'),name,ref,2.);ok&=k&(v==expected)
        for axis in ['roll','pitch','yaw','valid']:
            v,k=align(ds.get('manual_control_setpoint'),axis,ref,.2);vals['stick_'+axis]=v;ok&=k
        ok&=vals['stick_valid']>0
        qsp=[]
        for j in range(4):
            v,k=align(ds.get('vehicle_attitude_setpoint'),f'q_d[{j}]',ref,.05);qsp.append(v);ok&=k
        for axis in ['roll','pitch','yaw']:
            v,k=align(ds.get('vehicle_rates_setpoint'),axis,ref,.05);vals['rate_sp_'+axis]=v;ok&=k
        spangle=euler(np.stack(qsp,axis=1));actual=euler(f[['attitude_q_w','attitude_q_x','attitude_q_y','attitude_q_z']].to_numpy())
        vals['roll_sp_deg']=spangle[:,0];vals['roll_deg']=actual[:,0];vals['pitch_deg']=actual[:,1]
        for axis,k in zip('pqr','xyz'):vals[axis]=f[f'angular_velocity_body_rad_s_{k}'].to_numpy()
        left=f.control_left_elevon_normalized.to_numpy();right=f.control_right_elevon_normalized.to_numpy()
        vals['common']=(left+right)/2;vals['differential']=(left-right)/2;vals['rudder']=f.control_rudder_normalized.to_numpy()
        vals['drive']=f.control_flap_motor_normalized.to_numpy();neutral=np.abs(vals['stick_yaw'])<=.05
        expected=vals['stick_roll']*float(par.get('FW_MAN_R_MAX',np.nan));err=vals['roll_sp_deg']-expected
        r=dict(split=entry['split'],flight_id=entry['flight_id'],firmware=fw,valid_rows=int(f.valid_core.sum()),stabilized_rows=int(rawstab.sum()),fresh_chain_rows=int(ok.sum()),
          stabilized_duration_s=float(dt[rawstab].sum()),fresh_duration_s=float(dt[ok].sum()),yaw_neutral_rows=int((ok&neutral).sum()),roll_active_rows=int((ok&(np.abs(vals['stick_roll'])>=.1)).sum()),
          roll_setpoint_formula_rmse_deg=float(np.sqrt(np.mean(err[ok]**2))) if ok.any() else np.nan,
          rudder_rms_yaw_neutral=float(np.sqrt(np.mean(vals['rudder'][ok&neutral]**2))) if (ok&neutral).any() else np.nan)
        # Continuous usable pieces; retain all coverage losses in rows table.
        seg=np.cumsum(np.r_[True,(~ok[:-1])|(~ok[1:])|(dt[1:]>.05)])
        for sid in np.unique(seg[ok]):
            ix=np.flatnonzero(ok&(seg==sid));tt=ref[ix]*1e-6;state=states(vals['stick_roll'][ix]);cuts=np.r_[0,np.flatnonzero(np.diff(state)!=0)+1,len(ix)]
            for j in range(len(cuts)-1):
                lo,hi=cuts[j:j+2];duration=tt[hi-1]-tt[lo]
                if state[lo]==0 or duration<.2:continue
                before=(j>0 and state[cuts[j-1]]==0 and tt[lo-1]-tt[cuts[j-1]]>=.2)
                after=(j<len(cuts)-2 and state[hi]==0 and tt[cuts[j+2]-1]-tt[hi]>=.2)
                active=ix[lo:hi];eid=f"{entry['flight_id']}:{int(ref[active[0]])}"
                pre=max(0,lo-25);end=min(len(ix),hi+25);region=ix[pre:end]
                row=dict(split=entry['split'],flight_id=entry['flight_id'],event_id=eid,start_us=int(ref[active[0]]),end_us=int(ref[active[-1]]),duration_s=duration,sign=int(state[lo]),neutral_before=before,neutral_after=after,full_cycle=before and after,
                  yaw_neutral_fraction=float(neutral[active].mean()),roll_stick_mean=float(vals['stick_roll'][active].mean()),roll_change_deg=float(vals['roll_deg'][active[-1]]-vals['roll_deg'][active[0]]),
                  differential_range=float(np.ptp(vals['differential'][active])),rudder_range=float(np.ptp(vals['rudder'][active])),yaw_rate_sp_range=float(np.ptp(vals['rate_sp_yaw'][active])))
                episodes.append(row)
                for z in region:traces.append(dict(event_id=eid,elapsed_s=float((ref[z]-ref[active[0]])*1e-6),active=bool(z in active),**{k:float(v[z]) for k,v in vals.items()}))
        rows.append(r)
        w=pd.read_csv(ROOT/f'docs/analysis/results/paper_baseline_comparison_v1/{entry["split"]}_origins.csv')
        for _,win in w[w.log_id==entry['flight_id']].iterrows():
            start=np.flatnonzero((f.segment_id==win.segment_id)&(f.sample_in_segment==win.start_sample_in_segment));assert len(start)==1
            sel=np.arange(start[0],start[0]+51);origins.append(dict(split=entry['split'],flight_id=entry['flight_id'],window_id=win.window_id,all51_stabilized=bool(rawstab[sel].all()),all51_fresh_chain=bool(ok[sel].all())))
        print(entry['flight_id'],r['stabilized_rows'],flush=True)
    for name,data in [('flight_coverage.csv',rows),('episodes.csv',episodes),('traces.csv',traces),('parameters.csv',params),('origin_coverage.csv',origins)]:pd.DataFrame(data).to_csv(OUT/name,index=False)
    evidence=[]
    for fw in sorted(firmware):
        for path in ['msg/versioned/VehicleStatus.msg','src/modules/fw_att_control/FixedwingAttitudeControl.cpp','src/modules/fw_att_control/fw_yaw_controller.cpp','src/modules/fw_rate_control/FixedwingRateControl.cpp']:
            result=subprocess.run(['git','-C','/home/zn/PX4-Autopilot','show',f'{fw}:{path}'],capture_output=True,text=True)
            lines=[f'{i+1}: {line.strip()}' for i,line in enumerate(result.stdout.splitlines()) if any(word in line for word in ['STABILIZED mode','NAVIGATION_STATE_STAB','roll_body =','body_rates_setpoint(2) +=','_euler_rate_setpoint = tanf','yaw_body_rate_setpoint_raw','_param_fw_rll_to_yaw_ff.get()','Special case yaw in Acro'])]
            evidence.append(dict(firmware=fw,path=path,available=result.returncode==0,lines=lines))
    (OUT/'firmware_evidence.json').write_text(json.dumps(evidence,indent=2))
    assert all(sha(p)==h for p,h in pins.items())
    (OUT/'checks.json').write_text(json.dumps(dict(logs=len(entries),heldout_accessed=False,cache_hashes_unchanged=True,no_training=True,no_inference=True),indent=2))

if __name__=='__main__':run()
