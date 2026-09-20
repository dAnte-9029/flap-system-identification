"""Per-stage recorded-input checks, not a hidden-state/full-autopilot equivalence claim."""
from pathlib import Path
import sys,json,subprocess
import numpy as np
import pandas as pd
from pyulog import ULog
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from audit_control_readiness_stage2 import hold,metric,sha
from audit_control_readiness_stage2_outer import euler
from system_identification.control.px4_flight_chain import FlightAttitudeController,FlightAllocator
from system_identification.control.px4_pitch import FIRMWARE_COMMIT
OUT=ROOT/'docs/analysis/results/flight_chain_alignment'

def main():
    entry=json.loads((ROOT/'docs/analysis/results/control_readiness_stage3/run_manifest.json').read_text())['raw_logs'][0]
    path=Path(entry['path']);assert sha(path)==entry['sha256']
    names=['vehicle_attitude','vehicle_attitude_setpoint','vehicle_rates_setpoint','vehicle_torque_setpoint','actuator_servos','airspeed_validated','vehicle_control_mode','vehicle_status','vehicle_land_detected','autotune_attitude_control_status']
    u=ULog(str(path),message_name_filter_list=names);p=u.initial_parameters
    assert u.msg_info_dict['ver_sw']==FIRMWARE_COMMIT and p['FLAP_SLOW_EN']==0 and not u.changed_parameters
    assert p==json.loads((OUT/'flight_parameters.json').read_text())
    d={x.name:x.data for x in u.data_list if x.multi_id==0};sp=d['vehicle_rates_setpoint'];t=sp['timestamp'].astype(np.int64)
    q,qa=hold(d['vehicle_attitude'],[f'q[{i}]' for i in range(4)],t);qs,qsa=hold(d['vehicle_attitude_setpoint'],[f'q_d[{i}]' for i in range(4)],t)
    a=euler(q);s=euler(qs);air,aa=hold(d['airspeed_validated'],['calibrated_airspeed_m_s'],t)
    mode,ma=hold(d['vehicle_control_mode'],['flag_control_attitude_enabled','flag_control_manual_enabled'],t)
    status,_=hold(d['vehicle_status'],['nav_state'],t)
    auto=np.zeros((len(t),1));autage=np.full(len(t),np.inf)
    if 'autotune_attitude_control_status' in d:auto,autage=hold(d['autotune_attitude_control_status'],['state'],t)
    c=FlightAttitudeController(p);pred=np.full((len(t),3),np.nan)
    for i in range(len(t)):
        if mode[i,0]==1 and np.isfinite(np.r_[a[i],s[i]]).all():pred[i]=c.step(*a[i],*s[i],air[i,0],aa[i])
    mask=(mode[:,0]==1)&(mode[:,1]==0)&(status[:,0]==3)&(qa<.05)&(qsa<.05)&((autage>=1)|(auto[:,0]==0))
    rows=[]
    for j,axis in enumerate(['roll','pitch','yaw']):metric(rows,path.name,'attitude_'+axis,pred[:,j],sp[axis],mask,'rad/s')
    allocator=FlightAllocator(p);sv=d['actuator_servos'];t=sv['timestamp'].astype(np.int64)
    tor,age=hold(d['vehicle_torque_setpoint'],[f'xyz[{j}]' for j in range(3)],t)
    land,_=hold(d['vehicle_land_detected'],['landed'],t)
    pred=np.full_like(tor,np.nan)
    for i in range(len(t)):
        if np.isfinite(tor[i]).all():pred[i]=allocator.allocate(tor[i],.7).command[1:]
    mask=(age<.025)&(land[:,0]==0)
    for j in range(3):metric(rows,path.name,'allocation_'+str(j),pred[:,j],sv[f'control[{j}]'],mask)
    pd.DataFrame(rows).to_csv(OUT/'recorded_input_replay.csv',index=False)
    source_paths=['src/modules/fw_att_control/'+x for x in ['FixedwingAttitudeControl.cpp','fw_roll_controller.cpp','fw_pitch_controller.cpp','fw_yaw_controller.cpp']]+['src/modules/fw_rate_control/FixedwingRateControl.cpp','src/modules/fw_rate_control/FixedwingRateControl.hpp','src/lib/rate_control/rate_control.cpp','src/lib/rate_control/gain_compression.cpp','src/lib/fw_performance_model/PerformanceModel.cpp','src/modules/fw_lateral_longitudinal_control/FwLateralLongitudinalControl.hpp','src/modules/fw_mode_manager/FixedWingModeManager.cpp']
    hashes={}
    for path in source_paths:
        dest=OUT/'sources'/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(subprocess.check_output(['git','-C','/home/zn/PX4-Autopilot','show',FIRMWARE_COMMIT+':'+path]));hashes[path]=sha(dest)
    manifest=dict(firmware_commit=FIRMWARE_COMMIT,raw_log=entry,parameter_sha256=sha(OUT/'flight_parameters.json'),source_hashes=hashes,script_sha256=sha(Path(__file__)),scope='baseline A FLAP_SLOW_EN=0, autonomous cruise; per-stage recorded-input replay; asynchronous logged topics, not full closed-loop bitwise equality',native_outer_manifest_sha256=sha(ROOT/'artifacts/px4_e624_outer/manifest.json'),test_opened=False)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2));print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
