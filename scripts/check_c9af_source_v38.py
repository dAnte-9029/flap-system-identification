"""Compile unmodified c9af component function bodies with minimal host shims."""
from pathlib import Path
import subprocess
import numpy as np
import train_control_windows_v2 as base
from train_joint_increment_v34 import read
from system_identification.control.px4_c9af import C9Controller

SOURCE=base.ROOT/'docs/analysis/results/control_windows_v37/source'


def body(file, signature):
    text=(SOURCE/file).read_text();start=text.index(signature);opening=text.index('{',start);depth=1;end=opening+1
    while depth:
        depth+=(text[end]=='{')-(text[end]=='}');end+=1
    return text[start:end]


def check(out):
    out.mkdir(parents=True,exist_ok=True)
    license=(SOURCE/'gain_compression.cpp').read_text().split('#include')[0]
    cpp=license+r'''
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#define PX4_ISFINITE std::isfinite
#define CONSTANTS_ONE_G 9.80665f
namespace math {template<class T>T constrain(T x,T a,T b){return std::min(std::max(x,a),b);} template<class T>T max(T a,T b){return std::max(a,b);}template<class T>T min(T a,T b){return std::min(a,b);} float radians(float x){return x*3.14159265358979323846f/180.f;}}
struct V {float v[3]{};float& operator()(int i){return v[i];}float operator()(int i)const{return v[i];} V emult(const V& b)const{V r;for(int i=0;i<3;i++)r.v[i]=v[i]*b.v[i];return r;}void copyTo(float* a)const{for(int i=0;i<3;i++)a[i]=v[i];}};
V operator+(V a,V b){for(int i=0;i<3;i++)a.v[i]+=b.v[i];return a;} V operator-(V a,V b){for(int i=0;i<3;i++)a.v[i]-=b.v[i];return a;}
V operator-(V a){for(int i=0;i<3;i++)a.v[i]=-a.v[i];return a;}
using Vector3f=V;
struct rate_ctrl_terms_s{float p_term[3],i_term[3],d_term[3],ff_term[3],output[3];};
struct RateControl{V _gain_p,_gain_i,_gain_d,_gain_ff,_rate_int,_lim_int,_control_allocator_saturation_positive,_control_allocator_saturation_negative; V update(const V&,const V&,const V&,float,bool,rate_ctrl_terms_s*);void updateIntegral(V&,float);};
struct LPF{float state{},alpha{};float update(float v){state+=alpha*(v-state);return state;}float getState(){return state;}};
struct GainCompression{float _hpf{},_alpha_hpf{},_input_prev{},_compression_gain{},_compression_gain_min{},_kSpectralDamperGain{200.f},_kLeakageGain{.1f};LPF _lpf;float update(float,float);};
struct RollController{float _tc{},_max_rate{},_euler_rate_setpoint{},_body_rate_setpoint{};float control_roll(float,float,float,float);};
struct PitchController{float _tc{},_max_rate_pos{},_max_rate_neg{},_euler_rate_setpoint{},_body_rate_setpoint{};float control_pitch(float,float,float,float);};
struct YawController{float _max_rate{},_euler_rate_setpoint{},_body_rate_setpoint{};float control_yaw(float,float,float,float,float);};
'''
    for file,signature in [('rate_control.cpp','Vector3f RateControl::update('),('rate_control.cpp','void RateControl::updateIntegral('),
            ('gain_compression.cpp','float GainCompression::update('),('fw_roll_controller.cpp','float RollController::control_roll('),
            ('fw_pitch_controller.cpp','float PitchController::control_pitch('),('fw_yaw_controller.cpp','float YawController::control_yaw(')]:cpp+='\n'+body(file,signature)
    cpp+=r'''
int main(){std::cout<<std::setprecision(10);float x[26];while(std::cin>>x[0]){for(int k=1;k<26;k++)std::cin>>x[k];
RateControl r;GainCompression g;V rate,sp,acc;
for(int j=0;j<3;j++){r._gain_p(j)=x[0];r._gain_i(j)=x[1];r._gain_d(j)=x[2];r._gain_ff(j)=x[3];r._lim_int(j)=x[4];r._rate_int(j)=x[5];r._control_allocator_saturation_positive(j)=x[10];r._control_allocator_saturation_negative(j)=x[11];rate(j)=x[6];sp(j)=x[7];acc(j)=x[8];}
rate_ctrl_terms_s t;V effort=r.update(rate,sp,acc,x[9],false,&t);
g._compression_gain=x[12];g._compression_gain_min=.3f;g._hpf=x[13];g._lpf.state=x[14];g._input_prev=x[15];g._alpha_hpf=1.f/(1.f+2.f*3.14159265358979323846f*10.f*x[9]);g._lpf.alpha=x[9]/(1.f/(2.f*3.14159265358979323846f*5.f)+x[9]);g.update(x[16],x[9]);
RollController rc;PitchController pc;YawController yc;rc._tc=pc._tc=.4f;rc._max_rate=math::radians(60.f);pc._max_rate_pos=math::radians(60.f);pc._max_rate_neg=math::radians(60.f);yc._max_rate=math::radians(50.f);yc._euler_rate_setpoint=x[21];yc._body_rate_setpoint=x[22];
rc.control_roll(x[19],yc._euler_rate_setpoint,x[17],x[18]);pc.control_pitch(x[20],yc._euler_rate_setpoint,x[17],x[18]);yc.control_yaw(x[19],pc._euler_rate_setpoint,x[17],x[18],x[23]);
std::cout<<effort(0)<<" "<<r._rate_int(0)<<" "<<g._hpf<<" "<<g._lpf.state<<" "<<g._compression_gain<<" "<<rc._body_rate_setpoint<<" "<<pc._body_rate_setpoint<<" "<<yc._body_rate_setpoint<<" "<<yc._euler_rate_setpoint<<"\n";}}
'''
    (out/'reference.cpp').write_text(cpp)
    subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off',str(out/'reference.cpp'),'-o',str(out/'reference')],check=True,capture_output=True)
    rng=np.random.default_rng(38);x=np.zeros((512,26),np.float32)
    x[:,:5]=[.06,.1,0,.5,.4];x[:,5]=rng.uniform(-.3,.3,512);x[:,6:9]=rng.uniform(-1,1,(512,3));x[:,9]=rng.choice([.0025,.02],512)
    x[:,10:12]=rng.integers(0,2,(512,2));x[:,12]=rng.uniform(.3,1,512);x[:,13]=rng.uniform(-.2,.2,512);x[:,14]=rng.uniform(0,.2,512);x[:,15:17]=rng.uniform(-1,1,(512,2))
    x[:,17:23]=rng.uniform(-.7,.7,(512,6));x[:,23]=rng.uniform(3,15,512)
    result=subprocess.run([str(out/'reference')],input='\n'.join(' '.join(map(str,row)) for row in x),text=True,capture_output=True,check=True)
    actual=np.loadtxt(result.stdout.splitlines());expected=[]
    p=read(base.ROOT/'docs/analysis/results/control_windows_v37/parameters_by_flight.json')[0]['values'].copy()
    p.update(FW_R_TC=.4,FW_P_TC=.4,FW_R_RMAX=60,FW_P_RMAX_POS=60,FW_P_RMAX_NEG=60,FW_Y_RMAX=50)
    for row in x.astype(float):
        kp,ki,kd,kff,limit,integ,rate,sp,acc,dt,pos,neg,g,h,e,prev,inp,roll,pitch,rs,ps,ye,yb,speed,_,_=row
        c=C9Controller(p);c.state.yaw_euler=ye;c.state.yaw_body=yb
        att=c.attitude(roll,pitch,rs,ps,speed,0)
        # Exercise Python adapter's rate/integral path with gain disabled andscale1.
        rp=p.copy();rp.update(FW_USE_AIRSPD=0,FW_ARSP_SCALE_EN=0,FW_GC_EN=0)
        c2=C9Controller(rp);c2.gains=[[kp,ki,kd,kff,limit]]*3;c2.state.integral=[integ]*3;c2.state.positive=[bool(pos)]*3;c2.state.negative=[bool(neg)]*3
        got=c2.rate([rate]*3,[sp]*3,[acc]*3,8,0,.4,dt)
        # Exercise compression through full rate step configured to give exactly inp.
        rp['FW_GC_EN']=1;rp['FW_GC_GAIN_MIN']=.3;c3=C9Controller(rp);c3.gains=[[0,0,0,1,1]]*3
        c3.state.gain=[g]*3;c3.state.cached_gain=[1]*3;c3.state.hpf=[h]*3;c3.state.energy=[e]*3;c3.state.previous_effort=[prev]*3
        c3.rate([0]*3,[inp]*3,[0]*3,8,0,.4,dt)
        expected.append([got['effort'][0],c2.state.integral[0],c3.state.hpf[0],c3.state.energy[0],c3.state.gain[0],*att,c.state.yaw_euler])
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=2e-6)
    base.write(out/'verification.json',dict(status='passed',cases=512,actual_source_function_bodies=6,
        max_abs_error=float(np.max(abs(actual-np.asarray(expected)))),tolerance=dict(rtol=1e-5,atol=2e-6),
        limit='Extracted scalar/vector component bodies with host shims; not full PX4/uORB or bit-exact firmware replay.',source_sha256=base.sha(out/'reference.cpp')))
    return read(out/'verification.json')


if __name__=='__main__':print(check(base.ROOT/'docs/analysis/results/control_windows_v38/source_check'))
