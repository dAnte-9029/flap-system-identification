from dataclasses import fields
import numpy as np
import torch
from system_identification.models.phase_reference import circular,harmonic_design,interval_design,rotate_coefficients,estimate_windows,PhaseReferenceSimulator
from system_identification.models.main_v2_simulator import SimulatorState
from test_main_v2_simulator import make_simulator,make_inputs


def test_harmonic_rotation_matches_shift_and_native_interval_integral():
    rng=np.random.default_rng(6);c=rng.normal(size=(6,2));phi=np.linspace(-4,9,100)
    np.testing.assert_allclose(harmonic_design(phi)@rotate_coefficients(c,.7),harmonic_design(phi+.7)@c,atol=1e-14)
    numerical=np.array([harmonic_design(np.linspace(a,b,10001)).mean(0) for a,b in zip(phi[:-1],phi[1:])])
    np.testing.assert_allclose(interval_design(phi),numerical,atol=2e-6)


def test_past_only_estimator_offset_invariant_and_ignores_future_nan():
    t=np.cumsum(.02+.003*np.sin(np.arange(160)));phi=2*np.pi*3.2*t;c=np.array([[1.,2.],[.3,.6],[.4,.2],[.2,.1],[.2,.1],[.1,.2]])
    delta=.7;y=interval_design(phi)@rotate_coefficients(c,delta)
    w=np.zeros((160,3));v=w.copy();w[1:,1]=np.cumsum(y[:,0]*np.diff(t));v[1:,2]=np.cumsum(y[:,1]*np.diff(t))
    a=estimate_windows(t,phi,w,v,np.array([100]),50,c,np.ones(2))
    assert abs(circular(a['offset_rad'][0]-delta))<np.pi/180
    changed_w=w.copy();changed_v=v.copy();changed_w[101:]*=100;changed_v[101:]-=400
    batch=estimate_windows(t,phi,changed_w,changed_v,np.array([100,150]),50,c,np.ones(2))
    # Later reset rows may exist in the same batch; they cannot alter earlier estimates.
    np.testing.assert_array_equal(a['offset_rad'][0],batch['offset_rad'][0])
    for x in [phi,w,v]:x[101:]=np.nan
    t[101:]=-1
    b=estimate_windows(t,phi,w,v,np.array([100]),50,c,np.ones(2))
    for key in a:np.testing.assert_array_equal(a[key],b[key])


def test_canonical_snapshot_continuation_is_bitwise_without_reestimation():
    original=make_simulator();sim=PhaseReferenceSimulator(original.model);x=make_inputs()
    kw={k:v for k,v in x.items() if k not in ['future_controls','dt_s']}
    with torch.inference_mode():
        state=sim.reset_phase(offset_rad=torch.tensor([.3,1.7]),**kw);states=[state]
        for k in range(250):
            state,_=sim.step(state,x['future_controls'][:,k],x['dt_s'][:,k]);states.append(state)
        restored=sim.reset(state=SimulatorState.from_snapshot(states[100].snapshot()))
        sim.model.base_model._encode_history=lambda *a: (_ for _ in ()).throw(AssertionError('history re-read'))
        for k in range(100,250):
            restored,_=sim.step(restored,x['future_controls'][:,k],x['dt_s'][:,k])
            for f in fields(restored):torch.testing.assert_close(getattr(restored,f.name),getattr(states[k+1],f.name),rtol=0,atol=0)


def test_offline_lag_and_harmonic_error_decomposition_have_known_sign():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from diagnose_main_v2_phase_reference import alignment
    phi=np.arange(500)*6*np.pi/500
    result,_=alignment(phi,2*np.sin(phi-.6)+3,np.sin(phi))
    assert result['identifiable'] and result['lag_supported']
    assert abs(result['best_circular_lag_rad']-.6)<np.pi/360
    np.testing.assert_allclose(result['amplitude_mse'],.5,atol=1e-12)
    np.testing.assert_allclose(result['phase_mse'],2*(1-np.cos(.6)),atol=1e-12)
    np.testing.assert_allclose(result['dc_mse'],9,atol=1e-12)
