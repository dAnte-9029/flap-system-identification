"""Measure PX4 clock evidence, audit the reusable pitch controller, and replay.

Uses only the three previously admitted September 18 development logs.
"""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from pyulog import ULog

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from system_identification.control.px4_pitch import PitchConfig,PitchInput,PX4PitchRateController
from audit_control_readiness_stage2 import hold,metric,sha,dump,LOGS,SHA

OUT=ROOT/'docs/analysis/results/control_readiness_stage3'
PREV=ROOT/'docs/analysis/results/control_readiness_stage2'


def gain_reference_check():
    """Compile the exact upstream GainCompression::update method body.
    Minimal shims provide only storage, AlphaFilter update, and math::constrain.
    This verifies one component, not the entire firmware.
    """
    src=PREV/'firmware_sources/src/lib/rate_control/gain_compression.cpp'
    code=src.read_text(); start=code.index('float GainCompression::update(')
    method=code[start:]  # Last method in the frozen source file.
    cpp=code[:code.index('#include')]+'''#include <cmath>
#include <cstdio>
namespace math {float constrain(float x,float lo,float hi){return fminf(fmaxf(x,lo),hi);}}
struct Filter {float s=0,a=0;void update(float x){s+=a*(x-s);}float getState(){return s;}};
class GainCompression {public:
float _compression_gain=1,_compression_gain_min=.3f,_alpha_hpf=0,_hpf=0,_input_prev=0;
Filter _lpf; static constexpr float _kSpectralDamperGain=200,_kLeakageGain=.1f;
float update(float input,float dt);
};
'''+method+'''
int main(){GainCompression g; for(int i=0;i<1200;i++){
float dt=(i%2)?.0025f:.01f;float x=.3f*std::sin(.3f*i)+((i%2)?.2f:-.2f);
if(i==600)g._compression_gain=1;
g._alpha_hpf=1/(1+2*3.14159265358979323846f*10*dt);
g._lpf.a=dt/(1/(2*3.14159265358979323846f*5)+dt);
std::printf("%.9g %.9g %.9g\\n",x,dt,g.update(x,dt));}}
'''
    (OUT/'gain_reference.cpp').write_text(cpp)
    with tempfile.TemporaryDirectory(prefix='flap-gain-reference-') as temp:
        binary=Path(temp)/'gain_reference'
        subprocess.run(['g++','-std=c++17','-O2',str(OUT/'gain_reference.cpp'),'-o',str(binary)],check=True,capture_output=True)
        data=np.array([[float(v) for v in row.split()] for row in subprocess.check_output([str(binary)],text=True).splitlines()])
    # Exercise the module's gain update with P=1, other terms=0 and feed in
    # the error producing exactly the desired PRE-compression effort.
    c=PX4PitchRateController(PitchConfig(1,0,0,0,0,8,3,4,20,False,False,True,.3))
    predicted=[]
    for i,(effort,dt,expected) in enumerate(data):
        if i==600:c.state.compression_gain=1
        c.step(PitchInput(0,effort/c.state.cached_gain,0,dt,8,0))
        predicted.append(c.state.cached_gain)
    maximum=float(np.max(abs(np.array(predicted)-data[:,2])))
    assert maximum<3e-6, maximum
    result=dict(source_sha256=sha(src),component='exact GainCompression::update body with minimal filter/math shims',samples=len(data),maximum_gain_error=maximum,tolerance=3e-6)
    dump(OUT/'gain_reference_check.json',result)
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    dump(OUT/'scope.json',dict(raw_logs='stage2 admitted September18 logs only',sealed_test_opened=False,september19_opened=False,firmware=SHA,controller='pitch rate only; no roll B2b',performance_tuned=False))
    gain_reference_check()
    previous=json.loads((PREV/'run_manifest.json').read_text())
    clock=[]; rows=[]; hashes=[]; point_errors=[]
    names=['rate_ctrl_terms','rate_ctrl_status','vehicle_rates_setpoint','vehicle_angular_velocity','vehicle_torque_setpoint','airspeed_validated','gain_compression','vehicle_control_mode','vehicle_land_detected','vehicle_status','control_allocator_status']
    for entry in previous['raw_inputs']:
        path=Path(entry['path']); assert path.parent==LOGS/'9.18数据' and sha(path)==entry['sha256']
        hashes.append(entry)
        ul=ULog(str(path),message_name_filter_list=names); ds={d.name:d.data for d in ul.data_list if d.multi_id==0}; p=ul.initial_parameters
        assert ul.msg_info_dict['ver_sw']==SHA and not ul.changed_parameters
        rs=ds['rate_ctrl_status']; error=rs['rollspeed_error'].astype(float)
        denominator=np.maximum(0,1-(error/np.deg2rad(400))**2)*p['FW_RR_I']*error
        mask=(abs(denominator)>.005)&rs['rollspeed_integ_update_enabled'].astype(bool)
        inferred=rs['rollspeed_integ_delta_raw'][mask]/denominator[mask]
        assert len(inferred)>100 and np.all(np.isfinite(inferred))
        dt_native=float(np.median(inferred))
        assert .002<=dt_native<=.04
        clock.append(dict(log=path.name,n=len(inferred),dt_p01_s=float(np.quantile(inferred,.01)),dt_median_s=dt_native,dt_p99_s=float(np.quantile(inferred,.99)),inferred_hz=1/dt_native,method='roll raw I delta / (FW_RR_I * error * nonlinear I factor); update_enabled bool mask'))
        gyro=ds['vehicle_angular_velocity']; term=ds['rate_ctrl_terms']; tor=ds['vehicle_torque_setpoint']
        old=pd.read_csv(PREV/f'{path.stem}_pitch_replay.csv')
        assert np.array_equal(old.timestamp_us,tor['timestamp'])
        native_us=int(round(dt_native*1e6))
        first=int(gyro['timestamp'][np.searchsorted(gyro['timestamp'],term['timestamp'][0])])
        predictions={'stage2_legacy':old.pitch_torque_replay.to_numpy()}
        base_mask=old.scored.to_numpy(dtype=bool)
        for variant in ['exact_reset_gyro_grid','native_cadence_hold']:
            times=gyro['timestamp'].astype(np.int64) if variant=='exact_reset_gyro_grid' else np.arange(first,int(gyro['timestamp'][-1])+1,native_us,dtype=np.int64)
            g,ga=hold(gyro,['xyz[1]','xyz_derivative[1]','timestamp_sample'],times)
            sp,sa=hold(ds['vehicle_rates_setpoint'],['pitch','reset_integral'],times)
            air,aa=hold(ds['airspeed_validated'],['calibrated_airspeed_m_s'],times)
            mode,ma=hold(ds['vehicle_control_mode'],['flag_control_rates_enabled'],times)
            landed,la=hold(ds['vehicle_land_detected'],['landed'],times)
            sat,sta=hold(ds['control_allocator_status'],['unallocated_torque[1]'],times)
            controller=PX4PitchRateController(PitchConfig.from_parameters(p))
            controller.warm_start(float(term['i_term[1]'][0]),p['FW_AIRSPD_TRIM'])
            pred=np.full(len(times),np.nan); previous_sample=None; snapshots=[]
            for i,t in enumerate(times):
                if t<term['timestamp'][0] or not np.isfinite(sp[i,0]) or not np.isfinite(g[i,:2]).all():continue
                if variant=='exact_reset_gyro_grid':
                    dt=.01 if previous_sample is None else float(np.clip((g[i,2]-previous_sample)/1e6,.002,.04))
                    previous_sample=g[i,2]
                else: dt=native_us/1e6
                result=controller.step(PitchInput(float(g[i,0]),float(sp[i,0]),float(g[i,1]),dt,float(air[i,0]),float(aa[i]),landed=landed[i,0]!=0,rates_enabled=mode[i,0]==1,reset_integral=sp[i,1]==1,saturation_positive=sat[i,0]>np.finfo(np.float32).eps,saturation_negative=sat[i,0]<-np.finfo(np.float32).eps))
                if result.torque is not None: pred[i]=result.torque
                if i%400==0: snapshots.append(dict(timestamp_us=int(t),**controller.snapshot()))
            values,ages=hold(dict(timestamp=times,output=pred),['output'],tor['timestamp'])
            predictions[variant]=values[:,0]
            dump(OUT/f'{path.stem}_{variant}_state_snapshots.json',snapshots)
        common=base_mask.copy()
        for pred in predictions.values():common &= np.isfinite(pred)
        status,_=hold(ds['vehicle_status'],['nav_state'],tor['timestamp'])
        for label,pred in predictions.items():
            metric(rows,path.name,label,pred,tor['xyz[1]'],common)
            metric(rows,path.name,label+'_AUTO',pred,tor['xyz[1]'],common&(status[:,0]==3))
        trace=pd.DataFrame(dict(timestamp_us=tor['timestamp'],recorded=tor['xyz[1]'],scored_common=common,nav_state=status[:,0],**predictions))
        trace.to_csv(OUT/f'{path.stem}_comparison.csv',index=False)
        print(path.name,dt_native,flush=True)
    pd.DataFrame(clock).to_csv(OUT/'controller_clock_evidence.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT/'replay_comparison.csv',index=False)
    dump(OUT/'run_manifest.json',dict(completed=True,script_sha256=sha(Path(__file__)),controller_sha256=sha(ROOT/'src/system_identification/control/px4_pitch.py'),raw_logs=hashes,stage2_manifest_sha256=sha(PREV/'run_manifest.json'),initialization='first logged I once; air filter=trim; gain/HPF/LPF default; unknown boot history',native_grid='rounded median inferred controller dt, causal zero-order hold of available logged sensors; missing samples NOT reconstructed',test_opened=False,september19_opened=False))
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()
