"""Recorded-state attitude-loop replay and B2b conditional arithmetic audit."""
import numpy as np
import pandas as pd
from audit_control_readiness_stage2 import ROOT, LOGS, PX4, SHA, hold, cols, metric, sha, dump
import subprocess
from pyulog import ULog

OUT=ROOT/'docs/analysis/results/control_readiness_stage2'


def euler(q):
    w,x,y,z=q.T
    return np.column_stack([np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)),np.arcsin(np.clip(2*(w*y-z*x),-1,1))])


def main():
    assert not (OUT/'outer_loop_metrics.csv').exists()
    hashes={}
    for path in ['src/modules/fw_att_control/fw_pitch_controller.cpp','src/modules/fw_att_control/fw_roll_controller.cpp','src/modules/fw_att_control/fw_yaw_controller.cpp','src/modules/fw_att_control/FixedwingAttitudeControl.cpp']:
        p=OUT/'firmware_sources'/path; p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(subprocess.check_output(['git','-C',str(PX4),'show',f'{SHA}:{path}'])); hashes[path]=sha(p)
    results=[]
    for path in sorted((LOGS/'9.18数据').glob('*.ulg')):
        if path.name.startswith('._'): continue
        u=ULog(str(path),message_name_filter_list=['vehicle_attitude','vehicle_attitude_setpoint','vehicle_rates_setpoint','vehicle_status','vehicle_control_mode','airspeed_validated','rate_ctrl_status','rate_ctrl_terms','vehicle_torque_setpoint','autotune_attitude_control_status'])
        assert u.msg_info_dict['ver_sw']==SHA and not u.changed_parameters
        d={x.name:x.data for x in u.data_list if x.multi_id==0}; par=u.initial_parameters
        rate=d['vehicle_rates_setpoint']; t=rate['timestamp'].astype(np.int64)
        q,age=hold(d['vehicle_attitude'],[f'q[{i}]' for i in range(4)],t)
        qs,ages=hold(d['vehicle_attitude_setpoint'],[f'q_d[{i}]' for i in range(4)],t)
        att=euler(q); sp=euler(qs)
        cas,airage=hold(d['airspeed_validated'],['calibrated_airspeed_m_s'],t)
        status,_=hold(d['vehicle_status'],['nav_state','is_vtol','is_vtol_tailsitter'],t)
        mode,_=hold(d['vehicle_control_mode'],['flag_control_attitude_enabled','flag_control_manual_enabled'],t)
        assert not np.any(status[:,1:]>0)
        pred=np.full((len(t),3),np.nan); yaw_euler=0.; yaw_body=0.
        for i,((roll,pitch),(rs,ps)) in enumerate(zip(att,sp)):
            if not np.all(np.isfinite([roll,pitch,rs,ps])) or mode[i,0]!=1: continue
            pitch_euler=(ps-pitch)/par['FW_P_TC']
            pred[i,0]=np.clip((rs-roll)/par['FW_R_TC']-np.sin(pitch)*yaw_euler,-np.deg2rad(par['FW_R_RMAX']),np.deg2rad(par['FW_R_RMAX']))
            pred[i,1]=np.clip(np.cos(roll)*pitch_euler+np.cos(pitch)*np.sin(roll)*yaw_euler,-np.deg2rad(par['FW_P_RMAX_NEG']),np.deg2rad(par['FW_P_RMAX_POS']))
            if abs(roll)<np.pi/2:
                rc=np.clip(np.clip(roll,-np.deg2rad(80),np.deg2rad(80)),-abs(rs),abs(rs))
                speed=cas[i,0] if par['FW_USE_AIRSPD'] and airage[i]<1 and np.isfinite(cas[i,0]) else par['FW_AIRSPD_TRIM']
                speed=np.clip(max(.5,speed),par['FW_AIRSPD_STALL'],par['FW_AIRSPD_MAX'])
                yaw_euler=np.tan(rc)*np.cos(pitch)*9.80665/speed
                yaw_body=np.clip(-np.sin(roll)*pitch_euler+np.cos(roll)*np.cos(pitch)*yaw_euler,-np.deg2rad(par['FW_Y_RMAX']),np.deg2rad(par['FW_Y_RMAX']))
            pred[i,2]=yaw_body
        mask=(status[:,0]==3)&(mode[:,0]==1)&(mode[:,1]==0)&(age<=.05)&(ages<=.05)
        if 'autotune_attitude_control_status' in d:
            auto,aa=hold(d['autotune_attitude_control_status'],['state'],t)
            # Conservative exclusion: only score states with absent/stale or idle autotune.
            mask &= (aa>=1)|(auto[:,0]==0)
        for j,axis in enumerate(['roll','pitch','yaw']): metric(results,path.name,'outer_auto_'+axis,pred[:,j],rate[axis],mask,unit='rad/s')
        pd.DataFrame(dict(timestamp_us=t,roll_pred=pred[:,0],pitch_pred=pred[:,1],yaw_pred=pred[:,2],roll_recorded=rate['roll'],pitch_recorded=rate['pitch'],yaw_recorded=rate['yaw'],scored=mask)).to_csv(OUT/f'{path.stem}_outer_replay.csv',index=False)
        # Internal-state conditional audit only; no claim of B2b state reconstruction.
        terms=d['rate_ctrl_terms']; ref=terms['timestamp'].astype(np.int64)
        st,sa=hold(d['rate_ctrl_status'],['flap_b2b_g_current','flap_b2b_transferred_raw'],ref)
        predicted=np.clip(st[:,0]*(terms['output[0]']+st[:,1]),-1,1)
        assert par['TRIM_ROLL']==0 and par['FW_DTRIM_R_VMIN']==0 and par['FW_DTRIM_R_VMAX']==0
        tor=d['vehicle_torque_setpoint']; ix=np.searchsorted(tor['timestamp_sample'],terms['timestamp_sample']); ix=np.minimum(ix,len(tor['timestamp_sample'])-1)
        exact=tor['timestamp_sample'][ix]==terms['timestamp_sample']
        metric(results,path.name,'roll_conditional_logged_gain_transfer_and_PID',predicted,tor['xyz[0]'][ix],exact&(sa<=.03))
    pd.DataFrame(results).to_csv(OUT/'outer_loop_metrics.csv',index=False)
    dump(OUT/'outer_loop_manifest.json',dict(script_sha256=sha(ROOT/'scripts/audit_control_readiness_stage2_outer.py'),source_hashes=hashes,scope='same three Sep18 logs; AUTO attitude replay and conditional Roll B2b arithmetic; not full controller equivalence'))
    print(pd.DataFrame(results).to_string(index=False))

if __name__=='__main__': main()
