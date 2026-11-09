"""Frozen PACLD controlled pulse response: encode once, clone belief, autonomous flow."""
import argparse
from pathlib import Path
from dataclasses import replace
import json,hashlib,pickle
import numpy as np
import pandas as pd
import torch
from system_identification.data.pacld import open_windows
from system_identification.data.pacld_batch import compile_windows,steps,pad_steps
from system_identification.models.pacld import Config
from system_identification.models.pacld.batch import BatchedPACLD,FIELDS
from system_identification.models.pacld.batched_types import BatchedState
from system_identification.models.pahbd.types import ControlTape
from system_identification.models.pacld.dynamics import physical_features
from evaluate_pacld_a2_v2_gpu import CHECKPOINT,EXPECTED,batches

OUT=Path('experiments/phase_anchored_coupled_latent_v1/controlled_response_v1')
PREV=Path('/home/zn/flap-system-identification/.worktrees/pacld-control-effectiveness-v2/experiments/phase_anchored_coupled_latent_v1/control_effectiveness_a2_v2')
TRAIN=Path('/home/zn/flap-system-identification/.worktrees/pacld-control-identifiability-v1/experiments/phase_anchored_coupled_latent_v1/control_identifiability')
TIMES=[0.,.02,.05,.10,.15,.20,.30,.50]

_WORKER=None

def prepare_cir(payload):
    global _WORKER
    cfg,split,cohort=payload
    if _WORKER is None:
        torch.set_num_threads(1);_WORKER=open_windows(cfg,split)
    index=_WORKER.index;ws=[]
    for r in cohort.itertuples():
        group=index[(index.flight_id==r.flight)&(index.decision_us==r.t)]
        original=group[group.horizon_us==500000].iloc[0]
        choices=group[(group.last_target_measurement_us-group.decision_us>=500000)&(group.start_sample==original.start_sample)&(group.support_start_us==original.support_start_us)]
        if choices.empty:ws.append(None);continue
        ix=choices.sort_values(['horizon_us','window_id']).index[0];w=_WORKER[int(ix)]
        w.metadata['a2_window_id']=original.window_id
        ws.append(w)
    return pickle.dumps((cohort,ws),protocol=5)

def cir_batches(frame,cfg,split,workers):
    from concurrent.futures import ProcessPoolExecutor
    from collections import deque
    import multiprocessing
    payloads=iter((cfg,split,frame.iloc[i:i+128]) for i in range(0,len(frame),128))
    if workers==0:
        for payload in payloads:yield pickle.loads(prepare_cir(payload))
        return
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        q=deque()
        for _ in range(workers*2):
            payload=next(payloads,None)
            if payload is not None:q.append(pool.submit(prepare_cir,payload))
        while q:
            yield pickle.loads(q.popleft().result())
            payload=next(payloads,None)
            if payload is not None:q.append(pool.submit(prepare_cir,payload))

def phase_bin(phi):
    return np.floor(np.remainder(phi,2*np.pi)/(np.pi/2)).astype(int)%4

def take(state,indices):
    return state.with_(**{k:getattr(state,k)[indices].clone() for k in FIELDS})

def pulse(tape,amount):
    times=sorted(set(tape.times+[0.,.1,.5]))
    ix=np.searchsorted(tape.times,times,side='right')-1
    if (ix<0).any():raise ValueError('control support')
    values=tape.values[ix].clone();active=(np.array(times)>=0)&(np.array(times)<.1)
    values[active,1]-=amount;values[active,2]+=amount
    return ControlTape(times,values)

def feasible(tape,amount):
    for signed in [amount,-amount]:
        t=pulse(tape,signed);mask=(np.array(t.times)>=0)&(np.array(t.times)<=.5);v=t.values[mask]
        if not torch.isfinite(v).all():return False,'nonfinite_command'
        if ((v[:,1:] < -1)|(v[:,1:] > 1)).any():return False,'servo_range'
        if ((v[:,0]<0)|(v[:,0]>1)).any():return False,'motor_range'
    return True,''

@torch.inference_mode()
def simulate(model,belief,tapes,amount):
    B=len(tapes);state=belief.with_(**{k:torch.cat([getattr(belief,k).clone() for _ in range(3)],0) for k in FIELDS})
    for k in FIELDS:
        assert torch.equal(getattr(state,k)[:B],getattr(state,k)[B:2*B]) and torch.equal(getattr(state,k)[:B],getattr(state,k)[2*B:])
    controls=[pulse(t,sign*amount) for sign in [0,1,-1] for t in tapes]
    snapshots=[];alphas=[]
    for j,time in enumerate(TIMES):
        if j:
            rows=[steps(TIMES[j-1],time,t,model.config.max_step) for t in controls];dt,u=pad_steps(rows)
            # Same event/grid subdivision in each branch, including exactly .1 s.
            assert torch.equal(dt[:,:B],dt[:,B:2*B]) and torch.equal(dt[:,:B],dt[:,2*B:])
            assert dt.max()<=model.config.max_step+1e-12
            state=model.propagate_plan(state,(dt.to(state.x),u.to(state.x)))
        command=torch.stack([t.values[np.searchsorted(t.times,time,side='right')-1] for t in controls]).to(state.x)
        alpha=model.vector_field(state,command)[1]
        snapshots.append({k:getattr(state,k).detach().cpu().numpy().reshape(3,B,*getattr(state,k).shape[1:]) for k in FIELDS})
        alphas.append(alpha.detach().cpu().numpy().reshape(3,B,3))
    return snapshots,np.stack(alphas)

def load_model(device):
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()==EXPECTED
    bundle=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    assert bundle['config']['model']==json.load(open('configs/pacld/default.json'))['model']
    model=BatchedPACLD(Config(**bundle['config']['model']),bundle['precision']).to(device).eval();model.load_state_dict(bundle['model']);torch.set_num_threads(1)
    return model,bundle

def freeze():
    OUT.mkdir(exist_ok=True)
    r=np.load(PREV/'predictions.npz')['residual'][:,5];a=float(np.sqrt(np.mean(r*r)))
    train=pd.read_pickle(TRAIN/'features.pkl');boundary=float(np.median(np.stack(train.X)[:,0]))
    protocol=dict(baseline='ee1faa2ee42c2a7c2aac85c79cfcfaa205bdc581',checkpoint_sha256=EXPECTED,amplitude=a,amplitude_source='A2-v2 train leave-date-out 100ms differential residual RMS, 11221 decisions',multipliers=[.5,1.,2.],speed_boundary_mps=boundary,speed_boundary_source='train predecision body-forward velocity median',times=TIMES,pulse_interval='[0,0.1) seconds',range_servo=[-1,1],range_motor=[0,1],slew='CA_SV0_SLEW=CA_SV1_SLEW=0 in frozen train/validation metadata; no measured physical rate limit claimed',selection='A2-v2 identities, known nav_state 3/15, same decision/history with shortest frozen horizon supporting >=0.5s (may use 1s/2s window), primary-amplitude bidirectionally feasible; encode all candidates once; one earliest decision per causal-belief phase quadrant x train-speed-half per flight, at most 8; missing cells not backfilled',phase='causal encoded belief modulo 2pi; 2pi maps to 0',continuation='validation previously used; frozen-model continuing mechanism analysis, not untouched test',branch_contract='one encoded belief, all FIELDS exact copies, original future ZOH plus symmetric differential pulse; no future observation update',inference='original mixed precision, original max_step=.005 and tau=.06',mechanism='decoder alpha and all state traces; no channel ablation or modified dynamics',interpretation='model conditional open-loop intervention; no PX4 feedback rerun')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));return protocol

def audit_identity(windows,bundle,split):
    identity=windows.identity()
    for k in ['manifest_sha256','index_sha256','windows','split']:assert identity[k]==bundle['data_identity'][split][k]
    return identity

def encode_candidates(model,bundle,frame,split,workers):
    identity=audit_identity(open_windows(bundle['config'],split),bundle,split)
    rows=[];states=[];tapes=[];a=json.load(open(OUT/'protocol.json'))['amplitude']
    for cohort,ws in cir_batches(frame,bundle['config'],split,workers):
        eligible=[]
        for r,w in zip(cohort.itertuples(),ws):
            if w is None:
                rows.append(dict(flight=r.flight,date=r.date,t=int(r.t),window_id='',eligible=False,reason='no_frozen_0.5s_support_with_same_history'));continue
            reason='';ok=True
            if r.mode not in [3,15]:ok=False;reason='mode_absent_from_train'
            if w.metadata['actual_horizon_s']<.5:ok=False;reason='insufficient_0.5s_support'
            good,why=feasible(w.controls,a)
            if not good:ok=False;reason=why
            rows.append(dict(flight=r.flight,date=r.date,t=int(r.t),window_id=w.metadata['window_id'],a2_window_id=w.metadata['a2_window_id'],eligible=ok,reason=reason))
            if ok:eligible.append((r,w))
        if eligible:
            # Compiler requires target bookkeeping, but strip it before encoding.
            # encode_batch receives only initial state and causal history plans.
            batch=compile_windows([w for _,w in eligible],model.config).to(next(model.parameters()).device,torch.float64)
            batch.future=[];batch.targets={};batch.pairs={};batch.times=[]
            with torch.inference_mode():belief=model.encode_batch(batch)
            for i,(r,w) in enumerate(eligible):
                state={k:getattr(belief,k)[i].detach().cpu() for k in FIELDS};states.append(state);tapes.append(w.controls)
                row=next(x for x in reversed(rows) if x['flight']==r.flight and x['t']==r.t)
                row.update(cache_index=len(states)-1,phase=float(belief.phi[i].item()),phase_bin=int(phase_bin(belief.phi[i].item())),speed=float(r.X[0]),mode=int(r.mode),freq=float(belief.f[i].item()))
        print('encoded',split,len(rows),'/',len(frame),flush=True)
    pd.DataFrame(rows).to_csv(OUT/f'{split}_candidates.csv',index=False)
    torch.save(dict(states=states,tapes=tapes),OUT/f'{split}_beliefs.pt')
    (OUT/f'{split}_identity.json').write_text(json.dumps(identity,indent=2))
    return pd.DataFrame(rows),states,tapes

def select_windows(candidates,protocol):
    c=candidates[candidates.eligible].copy();c['speed_bin']=(c.speed>=protocol['speed_boundary_mps']).astype(int)
    return c.sort_values(['flight','t','window_id']).groupby(['flight','phase_bin','speed_bin'],sort=True).head(1).reset_index(drop=True)

def run_selected(model,selected,states,tapes,split):
    protocol=json.load(open(OUT/'protocol.json'));rows=[];excluded=[];internals=[]
    representative=set(selected.sort_values(['date','flight','t']).groupby(['phase_bin','speed_bin']).head(1).window_id)
    for mult in protocol['multipliers']:
        amount=protocol['amplitude']*mult
        valid=[]
        for r in selected.itertuples():
            ok,why=feasible(tapes[int(r.cache_index)],amount)
            if ok:valid.append(r)
            else:excluded.append(dict(window_id=r.window_id,flight=r.flight,multiplier=mult,reason=why))
        for begin in range(0,len(valid),64):
            rs=valid[begin:begin+64];indices=[int(r.cache_index) for r in rs]
            belief=BatchedState(**{k:torch.stack([states[i][k] for i in indices]).to(next(model.parameters()).device) for k in FIELDS})
            trajectory,alpha=simulate(model,belief,[tapes[i] for i in indices],amount)
            for j,time in enumerate(TIMES):
                s=trajectory[j]
                for i,r in enumerate(rs):
                    p=s['x'][:,i,10];yaw=s['x'][:,i,12]
                    row=dict(window_id=r.window_id,flight=r.flight,date=r.date,mode=r.mode,phase_bin=r.phase_bin,speed_bin=r.speed_bin,phase=r.phase,speed=r.speed,freq=r.freq,multiplier=mult,amplitude=amount,time=time,p0=p[0],p_plus=p[1],p_minus=p[2],r0=yaw[0],r_plus=yaw[1],r_minus=yaw[2],G_p=(p[1]-p[2])/(2*amount),G_r=(yaw[1]-yaw[2])/(2*amount),R_plus=p[1]-p[0],R_minus=p[2]-p[0],R_central=(p[1]-p[2])/2,asymmetry_abs=p[1]+p[2]-2*p[0],actuator_plus=(s['actuator'][1,i,1]-s['actuator'][1,i,0]-s['actuator'][0,i,1]+s['actuator'][0,i,0])/2,actuator_minus=(s['actuator'][2,i,1]-s['actuator'][2,i,0]-s['actuator'][0,i,1]+s['actuator'][0,i,0])/2,slow_plus_norm=np.linalg.norm(s['z_s'][1,i]-s['z_s'][0,i]),fast_plus_norm=np.linalg.norm(s['c'][1,i]-s['c'][0,i]),phase_difference=s['phi'][1,i,0]-s['phi'][0,i,0],frequency_difference=s['f'][1,i,0]-s['f'][0,i,0],alpha_p0=alpha[j,0,i,0],alpha_p_plus=alpha[j,1,i,0],alpha_p_minus=alpha[j,2,i,0],alpha_r0=alpha[j,0,i,2],alpha_r_plus=alpha[j,1,i,2],alpha_r_minus=alpha[j,2,i,2])
                    rows.append(row)
                    if r.window_id in representative and mult==1:
                        for b,name in enumerate(['factual','positive','negative']):
                            trace=dict(window_id=r.window_id,time=time,branch=name,alpha_p=alpha[j,b,i,0],alpha_r=alpha[j,b,i,2])
                            for k in FIELDS:
                                trace.update({k+'_'+str(l):v for l,v in enumerate(s[k][b,i].ravel())})
                            internals.append(trace)
            print('responses',split,mult,min(begin+64,len(valid)),'/',len(valid),flush=True)
    pd.DataFrame(rows).to_csv(OUT/f'{split}_responses.csv',index=False);pd.DataFrame(excluded,columns=['window_id','flight','multiplier','reason']).to_csv(OUT/f'{split}_infeasible.csv',index=False);pd.DataFrame(internals).to_csv(OUT/f'{split}_representative_internal.csv',index=False)

def checks(model,bundle,protocol):
    all_train=pd.read_pickle(TRAIN/'features.pkl');frame=all_train.groupby('flight',sort=True).head(1).iloc[:3]
    candidates,states,tapes=encode_candidates(model,bundle,frame,'train',0)
    selected=select_windows(candidates,protocol);assert len(selected)>=2
    B=len(states);belief=BatchedState(**{k:torch.stack([s[k] for s in states]).to(next(model.parameters()).device) for k in FIELDS})
    zero,az=simulate(model,belief,tapes,0.)
    zero_error=max(float(np.max(abs(s[k][1:]-s[k][0:1]))) for s in zero for k in FIELDS);assert zero_error<1e-12
    amount=protocol['amplitude'];response,alpha=simulate(model,belief,tapes,amount)
    singles=[]
    for i in range(B):singles.append(simulate(model,take(belief,[i]),[tapes[i]],amount)[0])
    batch_error=max(float(np.max(abs(response[j]['x'][:,i]-singles[i][j]['x'][:,0]))) for i in range(B) for j in range(len(TIMES)));assert batch_error<1e-5
    for tape in tapes:
        changed=pulse(tape,amount);base=pulse(tape,0.)
        t=np.array(changed.times);v=changed.values.numpy()-base.values.numpy();active=(t>=0)&(t<.1)
        assert np.array_equal(v[~active],np.zeros_like(v[~active]));assert np.allclose(v[:,0],0) and np.allclose(v[:,3],0)
        assert np.allclose(v[:,1]+v[:,2],0) and np.allclose((v[active,2]-v[active,1])/2,amount)
        plan=steps(0,.5,changed,model.config.max_step);ends=np.cumsum([x[0] for x in plan]);assert np.any(abs(ends-.1)<1e-12)
        assert not np.any((np.r_[0,ends[:-1]]<.1-1e-12)&(ends>.1+1e-12))
    assert np.array_equal(phase_bin(np.array([0.,2*np.pi,-2*np.pi,np.pi/2,np.pi,3*np.pi/2])),[0,0,0,1,2,3])
    assert np.max(abs(alpha[0,1:]-alpha[0,:1]))==0 # no direct tail command decoder input
    expected=amount*(1-np.exp(-.1/model.config.actuator_tau));s=response[TIMES.index(.1)];actual=(s['actuator'][1,:,1]-s['actuator'][1,:,0]-s['actuator'][0,:,1]+s['actuator'][0,:,0])/2;assert np.allclose(actual,expected,atol=1e-12)
    result=dict(train_windows=B,zero_amplitude_all_fields_error=zero_error,batch_vs_single_state_max_error=batch_error,branch_initial_equal_all_fields=True,predecision_controls_unchanged=True,pulse_end_grid_boundary=True,no_future_observations_in_encoding=True,phase_boundary_pass=True,actual_differential_denominator_pass=True,decoder_immediate_tail_command_effect=0.,actuator_100ms_expected=float(expected),actuator_100ms_max_error=float(np.max(abs(actual-expected))))
    (OUT/'checks.json').write_text(json.dumps(result,indent=2));run_selected(model,selected,states,tapes,'train');print(result,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','validation']);p.add_argument('--device',default='cuda:0');p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    OUT.mkdir(exist_ok=True);model,bundle=load_model(args.device)
    if args.stage=='preflight':protocol=freeze();checks(model,bundle,protocol);return
    assert (OUT/'checks.json').exists();protocol=json.load(open(OUT/'protocol.json'))
    frame=pd.read_pickle(PREV/'validation_features/features.pkl')
    candidates,states,tapes=encode_candidates(model,bundle,frame,'validation',args.workers)
    selected=select_windows(candidates,protocol);selected.to_csv(OUT/'selected_validation.csv',index=False)
    cells=[]
    for fid in frame.flight.unique():
        for phase in range(4):
            for speed in range(2):cells.append(dict(flight=fid,phase_bin=phase,speed_bin=speed,selected=int(((selected.flight==fid)&(selected.phase_bin==phase)&(selected.speed_bin==speed)).sum())))
    pd.DataFrame(cells).to_csv(OUT/'phase_speed_coverage.csv',index=False);run_selected(model,selected,states,tapes,'validation')
if __name__=='__main__':main()
