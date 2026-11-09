"""Targeted CIR verification using frozen train beliefs and optional completed artifacts."""
import json
import numpy as np
import pandas as pd
import torch
from pacld_cir_v1 import OUT,FIELDS,BatchedState,load_model,simulate,take,pulse,phase_bin

def main():
    model,bundle=load_model('cuda:1' if torch.cuda.device_count()>1 else 'cuda:0')
    cache=torch.load(OUT/'train_beliefs.pt',weights_only=False,map_location='cpu');states=cache['states'];tapes=cache['tapes'];a=json.load(open(OUT/'protocol.json'))['amplitude'];device=next(model.parameters()).device
    belief=BatchedState(**{k:torch.stack([s[k] for s in states]).to(device) for k in FIELDS});small,_=simulate(model,belief,tapes,a)
    ix=np.arange(64)%len(states);large,_=simulate(model,take(belief,ix),[tapes[i] for i in ix],a)
    err=max(float(np.max(abs(s['x']-l['x'][:,:len(states)]))) for s,l in zip(small,large));assert err<1e-5
    gain_error=max(float(np.max(abs((s['x'][1,:,10]-s['x'][2,:,10]-l['x'][1,:len(states),10]+l['x'][2,:len(states),10])/(2*a)))) for s,l in zip(small,large));assert gain_error<1e-5
    for mult in [.5,1,2]:
        for tape in tapes:
            before=tape.values.clone();plus=pulse(tape,a*mult);minus=pulse(tape,-a*mult);base=pulse(tape,0)
            assert torch.equal(tape.values,before)
            t=np.array(base.times);active=(t>=0)&(t<.1)
            for branch,sign in [(plus,1),(minus,-1)]:
                change=(branch.values-base.values).numpy()
                assert np.allclose((change[active,2]-change[active,1])/2,sign*a*mult)
                assert np.array_equal(change[~active],np.zeros_like(change[~active]))
    assert phase_bin(0)==phase_bin(2*np.pi)==0
    assert phase_bin(np.nextafter(2*np.pi,0))==3 and phase_bin(np.nextafter(2*np.pi,np.inf))==0
    result=dict(batch3_vs64_max_physical_state_error=err,batch3_vs64_gain_max_error=gain_error,all_amplitudes_denominator_pass=True,source_tapes_unchanged=True,near_phase_boundary_pass=True)
    path=OUT/'validation_responses.csv'
    if path.exists():
        d=pd.read_csv(path);s=pd.read_csv(OUT/'selected_validation.csv');assert s.groupby('flight').size().max()<=8
        assert not s.duplicated(['flight','t']).any()
        assert np.allclose(d.G_p,(d.p_plus-d.p_minus)/(2*d.amplitude),atol=1e-12)
        assert np.allclose(d.G_r,(d.r_plus-d.r_minus)/(2*d.amplitude),atol=1e-12)
        at0=d[d.time==0];assert (abs(at0.G_p)<1e-12).all() and (abs(at0.G_r)<1e-12).all()
        at100=d[d.time==.1];assert np.allclose(at100.actuator_plus,at100.amplitude*(1-np.exp(-.1/.06)),atol=1e-12)
        assert np.max(abs(d.phase_difference))==0 and np.max(abs(d.frequency_difference))==0
        result.update(actual_gain_denominators_pass=True,initial_zero_response_pass=True,actuator_closed_form_pass=True,motor_equal_phase_frequency_pass=True,selected_windows=len(s))
    from system_identification.data.pacld import open_windows
    native=open_windows(bundle['config'],'train');lookup={wid:i for i,wid in enumerate(native.index.window_id)}
    cases=pd.read_csv(OUT/'train_candidates.csv');cases=cases[cases.eligible]
    for r in cases.itertuples():
        original=native[lookup[r.a2_window_id]];extended=native[lookup[r.window_id]]
        assert len(original.history)==len(extended.history)
        for a,b in zip(original.history,extended.history):
            assert (a.measurement,a.availability,a.source)==(b.measurement,b.availability,b.source)
            assert a.availability<=0
            for field in ['value','mask','quality','variance']:assert torch.equal(getattr(a,field),getattr(b,field))
        for field in ['x','h','f','phi','actuator','variance','phase_quality','phase_covariance']:assert torch.equal(getattr(original.initial,field),getattr(extended.initial,field))
    result['longer_support_same_raw_initial_and_history']=True
    (OUT/'verification.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()
