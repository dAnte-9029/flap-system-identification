"""Build unchanged e624 guidance/TECS algorithms as a local C ABI library, no SITL."""
from pathlib import Path
import argparse, hashlib, json, re, subprocess, tarfile, io
ROOT=Path(__file__).resolve().parents[1]
SHA='e624a99f2955addbf76681e636c44162a0c03055'

def build(px4, out, parameters):
    out.mkdir(parents=True,exist_ok=True)
    p=json.loads(parameters.read_text())
    dirs=['tecs','npfg','motion_planning','mathlib','geo','matrix']
    archive=subprocess.check_output(['git','-C',str(px4),'archive',SHA,*['src/lib/'+d for d in dirs]])
    with tarfile.open(fileobj=io.BytesIO(archive)) as t: t.extractall(out,filter='data')
    def write(name,content):
        path=out/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content)
    write('shim/px4_platform_common/defines.h','#pragma once\n#include <cmath>\n#include <cfloat>\n#define PX4_ISFINITE(x) std::isfinite(x)\n#define __EXPORT\n#define M_PI_F 3.14159265358979323846f\n#define M_PI_2_F 1.57079632679489661923f\n#define M_TWOPI_F 6.28318530717958647692f\n#include <cstdio>\n#define PX4_WARN(...) std::fprintf(stderr, __VA_ARGS__)\n')
    write('shim/drivers/drv_hrt.h','#pragma once\n#include <cstdint>\nusing hrt_abstime=uint64_t;\nextern thread_local uint64_t flap_clock;\ninline uint64_t hrt_absolute_time(){return flap_clock;}\nnamespace time_literals {constexpr uint64_t operator"" _s(unsigned long long x){return x*1000000;} constexpr uint64_t operator"" _ms(unsigned long long x){return x*1000;} }\n')
    for n in ['uORB/Publication.hpp','uORB/topics/tecs_status.h','uORB/uORB.h']:write('shim/'+n,'#pragma once\n// No uORB symbols are used by the native TECS algorithm.\n')
    setters={'max_climb_rate':'FW_T_CLMB_MAX','max_sink_rate':'FW_T_SINK_MAX','min_sink_rate':'FW_T_SINK_MIN',
      'equivalent_airspeed_trim':'FW_AIRSPD_TRIM','equivalent_airspeed_min':'FW_AIRSPD_MIN','equivalent_airspeed_max':'FW_AIRSPD_MAX',
      'throttle_damp':'FW_T_THR_DAMPING','integrator_gain_throttle':'FW_T_THR_INTEG','integrator_gain_pitch':'FW_T_I_GAIN_PIT',
      'throttle_slewrate':'FW_THR_SLEW_MAX','vertical_accel_limit':'FW_T_VERT_ACC','roll_throttle_compensation':'FW_T_RLL2THR',
      'pitch_damping':'FW_T_PTCH_DAMP','altitude_error_time_constant':'FW_T_ALT_TC','fast_descend_altitude_error':'FW_T_F_ALT_ERR',
      'altitude_rate_ff':'FW_T_HRATE_FF','airspeed_error_time_constant':'FW_T_TAS_TC','ste_rate_time_const':'FW_T_STE_R_TC',
      'seb_rate_ff_gain':'FW_T_SEB_R_FF','airspeed_measurement_std_dev':'FW_T_SPD_STD',
      'airspeed_rate_measurement_std_dev':'FW_T_SPD_DEV_STD','airspeed_filter_process_std_dev':'FW_T_SPD_PRC_STD','speed_weight':'FW_T_SPDWEIGHT'}
    setup='\n'.join(f't.set_{method}({float(p[key]):.17e}f);' for method,key in setters.items())
    guidance={'setPeriod':'NPFG_PERIOD','setDamping':'NPFG_DAMPING','enablePeriodLB':'NPFG_LB_PERIOD',
       'enablePeriodUB':'NPFG_UB_PERIOD','setRollTimeConst':'NPFG_ROLL_TC','setSwitchDistanceMultiplier':'NPFG_SW_DST_MLT','setPeriodSafetyFactor':'NPFG_PERIOD_SF'}
    setup+='\n'+'\n'.join(f'g.{method}({float(p[key]):.17e}f);' for method,key in guidance.items())
    setup+=f'\nt.enable_airspeed({int(p["FW_USE_AIRSPD"])});'
    # Scalars in the ABI retain wrapper responsibilities: trim/limits/airspeed adaptation.
    write('wrapper.cpp','''#include <tecs/TECS.hpp>
#include <npfg/DirectionalGuidance.hpp>
#include <npfg/AirspeedDirectionController.hpp>
#include <npfg/CourseToAirspeedRefMapper.hpp>
thread_local uint64_t flap_clock=0;
struct Chain {TECS t; DirectionalGuidance g; AirspeedDirectionController h; CourseToAirspeedRefMapper c;
Chain(){'''+setup+'''} };
extern "C" {
void* create_chain(){return new Chain;}
void destroy_chain(void* ptr){delete static_cast<Chain*>(ptr);}
void outer_step(void* ptr, uint64_t us, const float* x, float* y){
 auto &s=*static_cast<Chain*>(ptr); flap_clock=us;
 // x: pitch, altitude, altitude_sp, eas_sp, eas, eas2tas, thr_min,max,trim,
 // pitch_min,max,climb,sink,height_rate,load_factor,eas_min,
 // north,east,vn,ve,windn,winde,tangentn,tangente,origin_n,origin_e
 s.t.set_load_factor(x[14]); s.t.set_equivalent_airspeed_min(x[15]);
 s.t.update(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10],x[11],x[12],0.f,x[13]);
 auto o=s.g.guideToPath({x[16],x[17]},{x[18],x[19]},{x[20],x[21]},{x[22],x[23]},{x[24],x[25]},0.f);
 float heading_sp=s.c.mapCourseSetpointToHeadingSetpoint(o.course_setpoint,{x[20],x[21]},x[3]);
 matrix::Vector2f v{x[18]-x[20],x[19]-x[21]};
 y[0]=s.t.get_pitch_setpoint(); y[1]=s.t.get_throttle_setpoint();
 y[2]=s.h.controlHeading(heading_sp,atan2f(v(1),v(0)),v.norm())+o.lateral_acceleration_feedforward;
 y[3]=o.course_setpoint; y[4]=s.g.getAdaptedPeriod();
 y[5]=s.t.getStatus().true_airspeed_filtered; y[6]=s.t.getStatus().altitude_reference;
}
}
''')
    sources=['tecs/TECS.cpp','motion_planning/VelocitySmoothing.cpp','motion_planning/ManualVelocitySmoothingZ.cpp',
        'npfg/DirectionalGuidance.cpp','npfg/AirspeedDirectionController.cpp','npfg/CourseToAirspeedRefMapper.cpp']
    cmd=['g++','-std=c++17','-O2','-shared','-fPIC',*[f'-I{out/d}' for d in ['shim','src','src/lib','src/lib/matrix']],str(out/'wrapper.cpp'),*[str(out/'src/lib'/s) for s in sources],'-o',str(out/'libpx4_outer.so')]
    subprocess.run(cmd,check=True)
    hashes={str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in out.rglob('*') if f.is_file() and f.name!='manifest.json'}
    manifest=dict(firmware_commit=SHA,parameter_sha256=hashlib.sha256(parameters.read_bytes()).hexdigest(),parameters=str(parameters.resolve()),sources=hashes,compile_command=cmd,compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],scope='unchanged guidance and TECS libraries; deterministic clock and unused uORB headers shimmed; not SITL',setters=setters,guidance_setters=guidance)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(out/'libpx4_outer.so')
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--px4',type=Path,default=Path('/home/zn/PX4-Autopilot'));a.add_argument('--output',type=Path,default=ROOT/'artifacts/px4_e624_outer');a.add_argument('--parameters',type=Path,default=ROOT/'docs/analysis/results/flight_chain_alignment/flight_parameters.json');args=a.parse_args();build(args.px4,args.output,args.parameters)
