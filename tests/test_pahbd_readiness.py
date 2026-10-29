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


def test_sampler_checkpoint_serializes_unfinished_flight_round(tmp_path):
    m,w=model_window();index=pd.DataFrame([dict(w.metadata,flight_id=str(i),window_id=str(i)) for i in range(5)])
    sampler=FlightBalancedSampler(index);sampler.draw()
    save_checkpoint(tmp_path/'round.pt',m,None,{},0,training_state={'sampler':sampler.state_dict()})
    _,_,bundle=load_checkpoint(tmp_path/'round.pt')
    resumed=FlightBalancedSampler(index);resumed.load_state_dict(bundle['training_state']['sampler'])
    assert [sampler.draw() for _ in range(15)]==[resumed.draw() for _ in range(15)]


def test_formal_scientific_admission_cannot_silently_train_disabled_ab():
    from system_identification.training.pahbd import require_training_admission
    cfg=dict(training=dict(objective='shared_periodic_AB'),data=dict(phase_policy='shared_or_average',horizon_us=[200000]))
    empty=pd.DataFrame(dict(shared_periodic_eligible=[False],phase_reference_id=['unknown_log']))
    with pytest.raises(ValueError):require_training_admission(cfg,Config(),empty)
    certified=pd.DataFrame(dict(shared_periodic_eligible=[True],phase_reference_id=['sensor'],horizon_us=[200000]))
    require_training_admission(cfg,Config(),certified,certified)
    with pytest.raises(ValueError):require_training_admission(cfg,Config(phase_mode='disabled'),certified)
    with pytest.raises(ValueError):require_training_admission(cfg,Config(),certified,empty)
    cfg['training']['objective']='average_dynamics'
    with pytest.raises(ValueError):require_training_admission(cfg,Config(),empty)
    cfg['data']['phase_policy']='average';require_training_admission(cfg,Config(),empty)


def test_hall_wrap_reset_gap_and_causal_recovery():
    import json
    from system_identification.data.pahbd_phase import prepare_phase
    c=json.loads(Path('configs/pahbd/phase_reference.json').read_text())
    ts=np.arange(200)*10000;angle=ts*1e-6*2*np.pi*4
    table=pd.DataFrame(dict(timestamp_us=ts,phase_rad=angle%(2*np.pi),
                           phase_unwrapped_rad=angle%(2*np.pi),encoder_total_count=angle*4096*8/(2*np.pi),
                           phase_valid=True,in_reviewed_flight=True))
    good=prepare_phase(table,8,c)
    assert good.phase_usable.iloc[50:].all()  # wraps and Hall zero resets are normal
    assert np.all(good.availability_us==ts+20000)
    table.loc[100:,'phase_rad']=(table.loc[100:,'phase_rad']+1)%(2*np.pi)
    bad=prepare_phase(table,8,c)
    assert bad.phase_anomaly.iloc[100] and not bad.phase_usable.iloc[100:150].any()
    assert bad.phase_usable.iloc[150:].all()
    table.loc[170:,'encoder_total_count']-=100000
    assert prepare_phase(table,8,c).phase_anomaly.iloc[170]
