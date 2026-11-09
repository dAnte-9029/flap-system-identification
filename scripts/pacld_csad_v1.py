"""Frozen dual-entry attribution; no optimizer, observation update or model writes."""
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
import torch
from pacld_cir_v1 import load_model, pulse, take, FIELDS, BatchedState, steps, pad_steps, PREV, TRAIN
from pacld_csad_flow import DualPACLD
from system_identification.models.pahbd import geometry as so3

CIR=Path('/home/zn/flap-system-identification/.worktrees/pacld-controlled-response-v1/experiments/phase_anchored_coupled_latent_v1/controlled_response_v1')
OUT=Path('experiments/phase_anchored_coupled_latent_v1/control_sensitivity_attribution_v1')
TIMES=[0.,.02,.05,.1,.15,.2]
AMPLITUDE=json.loads((CIR/'protocol.json').read_text())['amplitude']
BRANCHES=[(0,0),(1,0),(0,1),(1,1),(-1,0),(0,-1),(-1,-1)]

def cached(split):
    cache=torch.load(CIR/f'{split}_beliefs.pt',weights_only=False,map_location='cpu')
    frame=pd.read_csv(CIR/('selected_validation.csv' if split=='validation' else 'train_candidates.csv'))
    if split=='train':frame=frame[frame.eligible].copy()
    return frame,cache

def state_for(frame,cache,device):
    ids=frame.cache_index.astype(int).tolist()
    s=BatchedState(**{k:torch.stack([cache['states'][i][k] for i in ids]).to(device) for k in FIELDS})
    return s,[cache['tapes'][i] for i in ids]

@torch.no_grad()
def run(model,belief,tapes):
    B=len(tapes)
    s=belief.with_(**{k:torch.cat([getattr(belief,k).clone() for _ in BRANCHES]) for k in FIELDS})
    for k in FIELDS:
        for b in range(7):assert torch.equal(getattr(s,k)[b*B:(b+1)*B],getattr(belief,k))
    ta=[pulse(t,a*AMPLITUDE) for a,f in BRANCHES for t in tapes]
    tf=[pulse(t,f*AMPLITUDE) for a,f in BRANCHES for t in tapes]
    snapshots=[];commands=[];step_error=0.
    for j,t in enumerate(TIMES):
        if j:
            dt,ua=pad_steps([steps(TIMES[j-1],t,x,model.config.max_step) for x in ta])
            dtf,uf=pad_steps([steps(TIMES[j-1],t,x,model.config.max_step) for x in tf]);assert torch.equal(dt,dtf)
            dt,ua,uf=[v.to(s.x) for v in [dt,ua,uf]]
            for n in range(len(dt)):
                # Exact per-step reference on the factual branch, same batch width.
                ref=model.tensor_step(s,ua[n],dt[n])
                new=model.dual_step(s,ua[n],uf[n],dt[n])
                for k in FIELDS:
                    for b in [0,3,6]:
                        step_error=max(step_error,float((getattr(ref,k)[b*B:(b+1)*B]-getattr(new,k)[b*B:(b+1)*B]).abs().max()))
                assert step_error==0.,step_error
                s=new
        snapshots.append(s.with_(**{k:getattr(s,k).detach().clone() for k in FIELDS}))
        commands.append(torch.stack([x.values[np.searchsorted(x.times,t,side='right')-1] for x in tf]).to(s.x))
    return snapshots,commands,step_error

def decoder(model,factual,positive,u):
    """Right-body SO(3) tangent Jacobian at factual; mixed neural precision retained."""
    B=len(u); device=u.device
    q=torch.zeros((B,31),dtype=torch.float64,device=device,requires_grad=True)
    # tangent = [differential actuator, c6, slow12, position3, NED velocity3, body rotation3, FRD rate3]
    basis=q.new_tensor([-1.,1.,0.])
    def shifted(v):
        return factual.with_(actuator=factual.actuator+v[:,:1]*basis,c=factual.c+v[:,1:7],z_s=factual.z_s+v[:,7:19],x=so3.retract(factual.x,v[:,19:31]))
    def alpha(v):return model.vector_field(shifted(v),u)[1][:,0]
    y=alpha(q);jac=torch.autograd.grad(y.sum(),q)[0]
    d=torch.cat((((positive.actuator-factual.actuator)[:,1:2]-(positive.actuator-factual.actuator)[:,:1])/2,positive.c-factual.c,positive.z_s-factual.z_s,so3.state_residual(factual.x,positive.x)),1).detach()
    contrib=torch.stack([(jac[:,a:b]*d[:,a:b]).sum(1) for a,b in [(0,1),(1,7),(7,19),(19,31)]],1)
    full=model.vector_field(positive,u)[1][:,0]-y.detach()
    # independent directional central differences; moderate epsilon avoids fp32 roundoff.
    eps=.1
    fd=(alpha(eps*d)-alpha(-eps*d))/(2*eps)
    error=(fd-(jac*d).sum(1)).detach()
    # Explicit phase-carrier part: J wrt c with raw c path held constant.
    dyn=model.flow.dynamics;dtype=next(dyn.parameters()).dtype
    from system_identification.models.pacld.dynamics import phase_carrier
    low=factual.with_(**{k:getattr(factual,k).to(dtype) for k in FIELDS});cc=low.c.detach().clone().requires_grad_(True)
    zf=phase_carrier(low.phi,cc)*low.phase_quality
    yy=dyn.decoder(torch.cat((dyn.context(low,u.to(dtype)),low.c,zf),-1))[:,3]
    jc=torch.autograd.grad(yy.sum(),cc)[0].double()
    carrier=(jc*d[:,1:7]).sum(1)
    return [v.detach().cpu().numpy() for v in [jac,d,contrib,full,full-contrib.sum(1),error,carrier]]

def analyze(model,frame,cache,label):
    rows=[];decs=[];maxerr=0.
    for start in range(0,len(frame),48):
        part=frame.iloc[start:start+48];s,tapes=state_for(part,cache,'cuda:0');B=len(part)
        snapshots,commands,err=run(model,s,tapes);maxerr=max(maxerr,err)
        for j,t in enumerate(TIMES):
            st=snapshots[j];p=st.x[:,10].reshape(7,B).cpu().numpy()
            ra=.5*((p[1]-p[0])+(p[3]-p[2]));rf=.5*((p[2]-p[0])+(p[3]-p[1]))
            na=.5*((p[4]-p[0])+(p[6]-p[5]));nf=.5*((p[5]-p[0])+(p[6]-p[4]))
            assert np.max(abs(ra+rf-(p[3]-p[0])))<1e-12
            assert np.max(abs(na+nf-(p[6]-p[0])))<1e-12
            jac,d,c,full,res,fd,carrier=decoder(model,take(st,list(range(B))),take(st,list(range(3*B,4*B))),commands[j][:B])
            for i,r in enumerate(part.itertuples()):
                ident=dict(window_id=r.window_id,flight=r.flight,date=r.date,phase_bin=r.phase_bin,time=t)
                rows.append(dict(**ident,G_total=(p[3,i]-p[6,i])/(2*AMPLITUDE),G_A=(ra[i]-na[i])/(2*AMPLITUDE),G_F=(rf[i]-nf[i])/(2*AMPLITUDE),R_A_plus=ra[i],R_F_plus=rf[i],R_A_minus=na[i],R_F_minus=nf[i],p0=p[0,i],pplus=p[3,i],pminus=p[6,i],p_Aplus=p[1,i],p_Fplus=p[2,i],p_Aminus=p[4,i],p_Fminus=p[5,i],interaction_plus=p[3,i]-p[1,i]-p[2,i]+p[0,i],interaction_minus=p[6,i]-p[4,i]-p[5,i]+p[0,i]))
                rec=dict(**ident,alpha_difference=full[i],A=c[i,0],F=c[i,1],S=c[i,2],X=c[i,3],nonlinear_residual=res[i],directional_fd_error=fd[i],carrier_contribution=carrier[i],raw_fast_contribution=c[i,1]-carrier[i])
                rec.update({f'J_{k}':v for k,v in enumerate(jac[i])});rec.update({f'delta_{k}':v for k,v in enumerate(d[i])});decs.append(rec)
        print(label,start+len(part),'/',len(frame),flush=True)
    df=pd.DataFrame(rows);dd=pd.DataFrame(decs)
    assert dd.directional_fd_error.abs().max()<2e-5
    df.to_csv(OUT/f'{label}_paths.csv',index=False);dd.to_csv(OUT/f'{label}_decoder.csv',index=False)
    return df,dd,maxerr

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    base,bundle=load_model('cuda:0');model=DualPACLD(base.config,base.precision).cuda().eval();model.load_state_dict(base.state_dict())
    for p in model.parameters():p.requires_grad_(False)
    train,tc=cached('train');tr,td,err=analyze(model,train,tc,'train_check')
    # Verify the prior CIR output before any validation computation.
    prior=pd.read_csv(CIR/'train_responses.csv');prior=prior[(prior.multiplier==1)&prior.time.isin(TIMES)]
    cmp=tr.merge(prior,on=['window_id','time']);train_error=float((cmp.G_total-cmp.G_p).abs().max());assert train_error<2e-5
    assert json.loads((OUT/'verification.json').read_text())['jacobian_all_31_coordinates_train_fd_max_error']<5e-4, 'Run verify_pacld_csad_v1.py before validation'
    frame,cache=cached('validation');assert len(frame)==306 and frame.flight.nunique()==39
    reps=frame.sort_values(['date','phase_bin','flight','t']).groupby(['date','phase_bin']).head(1)
    reps.to_csv(OUT/'tau_representatives.csv',index=False)
    df,dd,err=analyze(model,frame,cache,'validation')
    prior=pd.read_csv(CIR/'validation_responses.csv');prior=prior[(prior.multiplier==1)&prior.time.isin(TIMES)]
    cmp=df.merge(prior,on=['window_id','time']);cir_error=float((cmp.G_total-cmp.G_p).abs().max());assert cir_error<2e-5
    cir_state_error=max(float((cmp.pplus-cmp.p_plus).abs().max()),float((cmp.pminus-cmp.p_minus).abs().max()),float((cmp.p0_x-cmp.p0_y).abs().max()));assert cir_state_error<1e-6
    taus=[]
    for tau in [.03,.06,.12]:
        model.config=replace(base.config,actuator_tau=tau);model.flow.dynamics.cfg=model.config
        f,d,e=analyze(model,reps,cache,f'tau_{tau}');f['tau']=tau;taus.append(f)
    pd.concat(taus).to_csv(OUT/'tau_paths.csv',index=False)
    checks=dict(checkpoint_sha256=json.loads((CIR/'protocol.json').read_text())['checkpoint_sha256'],amplitude=AMPLITUDE,train_windows=len(train),validation_windows=len(frame),step_equivalence_max=err,train_CIR_gain_error=train_error,validation_CIR_gain_error=cir_error,validation_CIR_p_all_branches_max_error=cir_state_error,shapley_sum_error=float((df.G_total-df.G_A-df.G_F).abs().max()),jacobian_directional_fd_max=float(dd.directional_fd_error.abs().max()),initial_fields_exact=True,history_reencoded=False,future_observation_updates=False,precision=base.precision,jacobian_coordinates='a_d,c[6],z_s[12],position_NED_m[3],velocity_NED_m/s[3],right_body_rotation_rad[3],omega_FRD_rad/s[3]',tau_selection='earliest flight/time per date x natural phase bin, fixed before analysis',baseline='7962138c6655734def4c490649d39d58eec4233f')
    (OUT/'checks.json').write_text(json.dumps(checks,indent=2));print(checks,flush=True)
if __name__=='__main__':main()
