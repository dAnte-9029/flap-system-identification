import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_ab import cell,fit_common,inject,residual_design,select_injection,conditions


def cycles(n=151):
    result=[]
    for i in range(n):
        t=(i+np.array([.05,.14,.26,.38,.52,.65,.77,.91]))*.25
        p=(t/.25-i)*2*np.pi
        result.append(cell(t,p,.2*t+np.sin(p),i*.25,(i+1)*.25))
    return result


def test_holdout_no_target_leak():
    cs=cycles();new=[cell(c['t'],c['p'],c['y']+(100 if i%5==2 else 0),c['lo'],c['hi']) for i,c in enumerate(cs)]
    for a,b in zip(fit_common(cs,77,True),fit_common(new,77,True)):
        np.testing.assert_array_equal(a,b)


def test_native_zero_mean_and_conservation():
    cs=cycles();scale=dict(slow_amplitude=.4,fast_rms=.3)
    for case in ['slow_015','slow_030','fast_constant','fast_05s','fast_2s','both_05s','both_2s']:
        for c in cs:
            s,f=inject(c,18.8,case,scale)
            assert abs(c['w']@f)<1e-14
            if case.startswith('slow'):assert np.max(abs(f))==0
    c=cs[77];s,f=inject(c,19.2,'both_05s',scale)
    assert np.max(s)>0 and np.max(abs(f))>0


def test_phase_randomization_and_nuisance():
    c=cycles()[20];rec=dict(c=c)
    u,z=residual_design(rec,3,0,True);b=np.column_stack([u*0+1,u,u*u])
    np.testing.assert_allclose(b.T@(c['w'][:,None]*z),0,atol=1e-14)
    _,shifted=residual_design(rec,3,.71,False)
    _,original=residual_design(rec,3,0,False)
    assert not np.allclose(original,shifted)
    # Circular offset preserves temporal order and phase increments.
    np.testing.assert_allclose(np.diff(c['p']+.71),np.diff(c['p']))


def test_full_window_support_and_same_center():
    cs=cycles();bank={('a','vD'):[cs],('a','q'):[cs],('short','vD'):[cs[:65]],('short','q'):[cs[:65]]}
    selected=select_injection(bank,7)
    assert len(selected)==1
    _,iv,iq,t0=selected[0]
    assert cs[25]['lo']<=t0-7 and cs[-26]['hi']>=t0+9
    assert conditions(cs[70],np.array([cs[70]['tau']]))=='changing'
    assert conditions(cs[70],np.array([]))=='stable'
