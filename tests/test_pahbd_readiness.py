"""Scientific contract regression checks; no formal research training."""
import copy
from dataclasses import replace
import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from system_identification.models.pahbd import PAHBD,Config,ControlTape,Observation
from system_identification.models.pahbd.dynamics import PhasePropagator
from system_identification.data.pahbd_synthetic import controlled_periodic_window
from system_identification.training.pahbd import FlightBalancedSampler,train_steps,save_checkpoint,load_checkpoint,formal_fit
from system_identification.evaluation.pahbd import phase_scenarios,FrozenRollResponse

torch.set_num_threads(1)


def model_window(horizon=.03):
    torch.manual_seed(17);cfg=Config(width=16,latent=4)
    m=PAHBD(cfg).double();w,_=controlled_periodic_window(cfg,horizon)
    w.metadata['horizon_us']=round(horizon*1e6)
    return m,w


def test_delayed_replay_known_controlled_motion_and_arrival_order():
    m,w=model_window();start=-.25
    # Known linear translational system embedded in the SAME production Lie flow.
    # Acceleration is zero and initial velocity is known: exact p(t)=v*(t-start).
    velocity=w.initial.x.new_tensor([2.,-1.,.5])
    x=w.initial.x.clone();x[3:6]=velocity
    initial=w.initial.with_(x=x)
    m.flow.derivative=lambda s:(torch.zeros_like(s.x[:3]),torch.zeros_like(s.x[:3]),torch.zeros_like(s.h))
    def reading(t,arrival):
        val=torch.cat((x.clone(),initial.f,initial.phi));val[:3]=velocity*(t-start)
        mask=torch.zeros(14,dtype=x.dtype);mask[:6]=1
        return Observation(t,arrival,val,mask,torch.ones_like(mask),torch.ones_like(mask),'analytic')
    older=reading(-.2,-.01);newer=reading(-.08,-.04)
    seen=[]
    def exact_update(s,e,dt):
        # Residual MUST be against the state at the measurement time.
        seen.append(float((s.x[:6]-e.value[:6]).abs().max()))
        y=s.x.clone();y[:6]=e.value[:6]
        return s.with_(x=y)
    m.belief.update=exact_update
    poison=replace(older,availability=.01,value=torch.full_like(older.value,float('nan')))
    expired=reading(-.3,-.02)
    result=m.belief(m.flow,initial,start,[newer,older,poison,expired],w.controls)
    assert max(seen)<1e-12
    assert torch.allclose(result.x[:3],velocity*.25,atol=1e-12)
    assert m.belief.last_replay['accepted']==2 and m.belief.last_replay['rejected']==1
    # Compare actual changing ZOH acceleration, with a perfect delayed full-state correction.
    def accel(s):return torch.stack((s.actuator[0],s.actuator[0]*0,s.actuator[0]*0)),torch.zeros_like(s.x[:3]),torch.zeros_like(s.h)
    m.flow.derivative=accel
    m.config.max_step=.0005  # resolve midpoint truncation against the analytic reference
    # analytic actuator relaxation integral after a command switch
    tape=ControlTape([start,-.1],x.new_tensor([[.5,0,0,0],[.5,1,0,0]]))
    initial=initial.with_(actuator=x.new_zeros(3))
    event=reading(-.15,-.02)
    result=m.belief(m.flow,initial,start,[event],tape)
    tau=m.config.actuator_tau;duration=.1
    expected_v=velocity[0]+duration-tau*(1-math.exp(-duration/tau))
    expected_p=velocity[0]*.25+duration**2/2-tau*duration+tau**2*(1-math.exp(-duration/tau))
    assert abs(float(result.x[0]-expected_p))<2e-6
    assert abs(float(result.x[3]-expected_v))<2e-6


def test_joint_covariance_diffusion_semigroup_harmonic_attenuation():
    p=PhasePropagator(Config().frequency_coefficients)
    c=torch.tensor([[.09,.03],[.03,.25]],dtype=torch.float64);q=torch.diag(c.new_tensor([.04,.01]))
    whole=p.covariance(c,2.,q);parts=p.covariance(p.covariance(c,.317,q),1.683,q)
    assert torch.allclose(whole,parts,atol=1e-11,rtol=1e-11)
    assert torch.linalg.eigvalsh(whole).min()>0 and whole[0,1]>0
    m,w=model_window();s=w.initial;context,_=m.flow.slow(s)
    _,det=m.flow.mixer(context,s.actuator,s.phi,s.phase_quality)
    variance=s.phi.new_tensor([.8])
    _,unc=m.flow.mixer(context,s.actuator,s.phi,s.phase_quality,variance)
    _,wrapped=m.flow.mixer(context,s.actuator,s.phi+2*math.pi,s.phase_quality,variance)
    assert torch.allclose(unc['forcing'],wrapped['forcing'],atol=1e-12)
    assert not torch.allclose(det['forcing'],unc['forcing'])
    # Isolate each harmonic and verify exact wrapped-normal characteristic function.
    with torch.no_grad():
        m.flow.mixer.forcing.weight.zero_();m.flow.mixer.forcing.bias.zero_()
        m.flow.mixer.forcing.bias.reshape(3,2,6)[:,0,0]=1
    phase=s.phi*0
    _,terms=m.flow.mixer(context,s.actuator,phase,s.phase_quality,variance)
    expected=sum(math.exp(-.5*k*k*.8) for k in (1,2,3))
    assert abs(float(terms['forcing'][0])-expected)<1e-12
    assert math.exp(-.5*9*.8)<math.exp(-.5*.8)


def test_scenarios_preserve_joint_initial_moments_and_analytic_mean():
    m,w=model_window();s=w.initial.with_(phase_covariance=torch.tensor([[.04,.02],[.02,.3]],dtype=torch.float64))
    r=phase_scenarios(m,s,[.03],w.controls)
    offsets=r['initial_offsets'];weights=r['weights']
    assert torch.allclose(offsets.T@torch.diag(weights)@offsets,s.phase_covariance,atol=1e-12)
    fs=torch.stack([a[-1].f for a in r['states']]);ps=torch.stack([a[-1].phi for a in r['states']])
    assert torch.allclose(fs.mean(0),r['mean_phase'][-1].f,atol=1e-12)
    assert torch.allclose(ps.mean(0),r['mean_phase'][-1].phi,atol=1e-12)


def test_sampler_flight_horizon_balance_resume_and_changed_index():
    rows=[]
    for f,n in [('a',1),('b',100),('c',7)]:
        for h in [200000,500000,1000000,2000000]:
            rows.extend(dict(window_id=f'{f}:{h}:{i}',flight_id=f,horizon_us=h,split='train') for i in range(n))
    index=pd.DataFrame(rows);a=FlightBalancedSampler(index,17);b=FlightBalancedSampler(index,17)
    first=[a.draw() for _ in range(12)];assert first==[b.draw() for _ in range(12)]
    sampled=index.iloc[first]
    assert sampled.groupby(['flight_id','horizon_us']).size().eq(1).all()
    state=a.state_dict();future=[a.draw() for _ in range(100)]
    b.load_state_dict(state);assert future==[b.draw() for _ in range(100)]
    with pytest.raises(ValueError):FlightBalancedSampler(index.iloc[:-1],17).load_state_dict(state)


def test_phase_reference_admission_uses_interval_evidence(tmp_path):
    from system_identification.data.pahbd_phase import annotate_phase
    pd.DataFrame([dict(flight_id='logA:1',hall_rows=30)]).to_csv(tmp_path/'phase_per_flight.csv',index=False)
    pd.DataFrame([dict(flight_id='logA:1',timestamp_us=150,reset_correction_rad=.2)]).to_csv(tmp_path/'phase_resets.csv.gz',index=False)
    w=pd.DataFrame([dict(flight_id='logA:1',support_start_us=a,support_end_us=b,complete_phase_frequency=True,review=False) for a,b in [(100,200),(300,400)]])
    out=annotate_phase(w,{'logA:1':dict(reset_times_us=[350])},tmp_path,
                       hall_events=pd.DataFrame([dict(flight_id='logA:1',timestamp_us=130)]))
    assert out.relative_phase_eligible.tolist()==[True,False]
    assert out.reference_updates.tolist()==[1,0]
    assert not out.shared_periodic_eligible.any() and out.hall_event_traceable.tolist()==[True,False]
    m,window=model_window();s=window.initial.with_(phase_quality=window.initial.phase_quality*0)
    c,_=m.flow.slow(s)
    a,parts=m.flow.mixer(c,s.actuator,s.phi,s.phase_quality)
    b,_=m.flow.mixer(c,s.actuator,s.phi+1,s.phase_quality)
    assert torch.equal(a,b) and parts['interaction'].abs().sum()==0


def test_checkpoint_sampler_optimizer_exact_resume(tmp_path):
    m,w=model_window(.02);windows=[w];index=pd.DataFrame([w.metadata])
    opt=torch.optim.Adam(m.parameters(),lr=1e-4);sampler=FlightBalancedSampler(index)
    train_steps(m,windows,opt,1,sampler=sampler)
    save_checkpoint(tmp_path/'resume.pt',m,opt,{},1,training_state={'sampler':sampler.state_dict()})
    restored,ro,bundle=load_checkpoint(tmp_path/'resume.pt',optimizer=True)
    rs=FlightBalancedSampler(index);rs.load_state_dict(bundle['training_state']['sampler'])
    train_steps(m,windows,opt,1,sampler=sampler);train_steps(restored,windows,ro,1,sampler=rs)
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in m.state_dict().items())


def test_formal_entry_schedule_checkpoint_validation_smoke(tmp_path):
    # Two synthetic updates only; checks orchestration, not research training.
    m,w=model_window(.02);v=copy.deepcopy(w);v.metadata['split']='validation'
    opt=torch.optim.Adam(m.parameters(),lr=1e-4)
    meta={'config':{'purpose':'synthetic_protocol_check'}}
    result=formal_fit(m,[w],[v],opt,1,tmp_path/'first',meta,validation_interval=1,patience=2)
    restored,ro,bundle=load_checkpoint(tmp_path/'first/step_00000001.pt',optimizer=True)
    result2=formal_fit(restored,[w],[v],ro,2,tmp_path/'resume',meta,validation_interval=1,patience=2,resume=bundle)
    assert result['schedule']['step']==1 and result2['schedule']['step']==2
    assert result2['coverage']['draws']==2


FROZEN=Path('/home/zn/flap-system-identification/.worktrees/local-control-response-rollout-v1/experiments/local_control_response_rollout_v1')


def test_frozen_response_48_pairs_11_components_and_metric_oracle():
    if not FROZEN.exists():pytest.skip('frozen external evidence unavailable')
    adapter=FrozenRollResponse(FROZEN);z=adapter.load_truth()
    lookup={r.window_id:i for i,r in adapter.windows.iterrows()}
    def label_oracle(wid,times):
        # Numerical adapter test ONLY. Never a deployed predictor or training feature.
        i=lookup[wid];meta=adapter.windows.iloc[i];n=int(z['future_count'][i])
        t=z['native_time_us'][i,:n+1]*1e-6
        y=np.vstack((z['truth_initial'][i],z['truth'][i,:n]))
        query=np.asarray(times)-(int(meta.initial_measurement_us)-int(meta.decision_us))*1e-6
        states,valid=adapter.analysis.interpolate(t,y,query);assert valid.all()
        return y[0],states
    result=adapter.evaluate(label_oracle,'train')
    assert result['rows'].pair_uid.nunique()==48
    assert result['rows'].dependency_component.nunique()==11
    assert max(r['rmse'] for r in result['summary'])<1e-10


def test_shared_reference_certificate_is_causal_and_uses_per_log_count_anchor(tmp_path):
    import json
    from system_identification.data.pahbd_phase import apply_shared_reference
    base=pd.DataFrame([dict(flight_id=f,phase_reference_id=f,relative_phase_eligible=True,
                           support_start_us=100,support_end_us=200,shared_periodic_eligible=False)
                       for f in ('a','b')])
    provenance={f:dict(start_us=0,end_us=300) for f in ('a','b')}
    entries=[dict(flight_id=f,start_us=90,end_us=250,reference_count=count,reference_measurement_us=70,reference_availability_us=80,
                  reference_id='synthetic_shared_sensor',sensor_index_shared_verified=True,
                  causal_publication_verified=True,continuity_verified=True,mechanical_zero_verified=False,
                  evidence=['synthetic mathematical fixture only']) for f,count in [('a',1000.),('b',28000.)]]
    path=tmp_path/'reference.json'
    def write():path.write_text(json.dumps(dict(schema='pahbd_phase_reference_v1',certified_intervals=entries)))
    write();out=apply_shared_reference(base,provenance,path)
    assert out.shared_periodic_eligible.all() and not out.mechanical_zero_verified.any()
    assert out.reference_count.tolist()==[1000.,28000.]
    # Independent anchors, never a single fitted global gauge offset.
    from system_identification.data.pahbd import NativeWindows
    adapter=NativeWindows.__new__(NativeWindows);adapter.dtype=torch.float64;adapter.device='cpu'
    phases=[]
    for count in out.reference_count:
        row=pd.Series(dict(timestamp_us=120,timestamp_sample_us=0,availability_us=125,total_count=count+4096*8/4))
        obs=adapter._observation('encoder_count',row,150,dict(phase_origin_count=count,transmission_ratio=8.,review_intervals_us=[]))
        phases.append(float(obs.value[14]))
    assert np.allclose(phases,[math.pi/2]*2)
    entries[0]['reference_availability_us']=101;write()
    with pytest.raises(ValueError):apply_shared_reference(base,provenance,path)


def test_sampler_checkpoint_serializes_unfinished_flight_round(tmp_path):
    m,w=model_window();index=pd.DataFrame([dict(w.metadata,flight_id=str(i),window_id=str(i)) for i in range(5)])
    sampler=FlightBalancedSampler(index);sampler.draw()
    save_checkpoint(tmp_path/'round.pt',m,None,{},0,training_state={'sampler':sampler.state_dict()})
    _,_,bundle=load_checkpoint(tmp_path/'round.pt')
    resumed=FlightBalancedSampler(index);resumed.load_state_dict(bundle['training_state']['sampler'])
    assert [sampler.draw() for _ in range(15)]==[resumed.draw() for _ in range(15)]


def test_real_single_log_periodic_graph_and_reference_admission():
    from system_identification.data.pahbd import NativeWindows,configure_latent,ConfiguredWindows,DEFAULT_ROOT
    if not (Path(DEFAULT_ROOT)/'windows_train.parquet').exists():pytest.skip('frozen native train absent')
    reference='37ff9f8ea4e2e378fc13394288aa145a462259e2f53fa15af9511082f0e6d558'
    d=NativeWindows(horizon_us=500000,phase_policy='relative_single',relative_reference_id=reference)
    m,_=model_window();w=configure_latent(d[0],m.config)
    assert w.metadata['periodic_enabled'] and not w.metadata['shared_periodic_eligible']
    assert w.metadata['phase_reference_id']==reference and w.metadata['cycle_pairs']
    from system_identification.training.pahbd import window_loss
    loss,*_=window_loss(m,w);loss.backward()
    assert torch.isfinite(loss)
    assert m.flow.mixer.forcing.weight.grad.abs().sum()>0
    assert m.flow.mixer.control.weight.grad.abs().sum()>0
    # This dataset cannot silently admit any other unknown log origin.
    assert d.windows.phase_reference_id.nunique()==1


def test_formal_scientific_admission_cannot_silently_train_disabled_ab():
    from system_identification.training.pahbd import require_training_admission
    cfg=dict(training=dict(objective='shared_periodic_AB'),data=dict(phase_policy='shared_or_average'))
    empty=pd.DataFrame(dict(shared_periodic_eligible=[False],phase_reference_id=['unknown_log']))
    with pytest.raises(ValueError):require_training_admission(cfg,Config(),empty)
    certified=pd.DataFrame(dict(shared_periodic_eligible=[True],phase_reference_id=['sensor']))
    require_training_admission(cfg,Config(),certified,certified)
    with pytest.raises(ValueError):require_training_admission(cfg,Config(phase_mode='disabled'),certified)
    with pytest.raises(ValueError):require_training_admission(cfg,Config(),certified,empty)
    cfg['training']['objective']='average_dynamics'
    with pytest.raises(ValueError):require_training_admission(cfg,Config(),empty)
    cfg['data']['phase_policy']='average';require_training_admission(cfg,Config(),empty)
