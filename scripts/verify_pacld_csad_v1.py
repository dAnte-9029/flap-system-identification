"""Targeted train-only numerical and contract checks, independent finite differences."""
import json
import numpy as np
import torch
from pacld_csad_v1 import *
from pacld_cir_v1 import audit_identity, feasible
from system_identification.data.pacld import open_windows

def main():
    base,bundle=load_model('cuda:0');m=DualPACLD(base.config,base.precision).cuda().eval();m.load_state_dict(base.state_dict())
    for p in m.parameters():p.requires_grad_(False)
    for split in ['train','validation']:
        identity=audit_identity(open_windows(bundle['config'],split),bundle,split)
        old=json.loads((CIR/f'{split}_identity.json').read_text());assert identity==old
    from system_identification.training.pacld_gpu import BucketSampler
    training=open_windows(bundle['config'],'train');sampler=BucketSampler(training.index,17,32)
    log=Path('/home/zn/flap-system-identification/.worktrees/pacld-gpu-batched-v1/runs/pacld_gpu_b32_seed17_10944windows/training.jsonl')
    records=[json.loads(line) for line in log.read_text().splitlines()]
    assert len(records)==342
    for r in records:assert sampler.draw()==r['global_window_indices']
    frame,cache=cached('train');s,tapes=state_for(frame,cache,'cuda:0');B=len(frame)
    snapshots,commands,_=run(m,s,tapes)
    symmetry=0.;acterr=0.;phaseerr=0.
    for j,t in enumerate(TIMES):
        ss=snapshots[j]
        for b,(a,f) in enumerate(BRANCHES):
            delta=ss.actuator[b*B:(b+1)*B]-ss.actuator[:B]
            expected=a*AMPLITUDE*((1-np.exp(-t/.06)) if t<=.1 else (1-np.exp(-.1/.06))*np.exp(-(t-.1)/.06))
            acterr=max(acterr,float((delta-delta.new_tensor([-expected,expected,0.])).abs().max()))
            symmetry=max(symmetry,float((delta[:,0]+delta[:,1]).abs().max()),float(delta[:,2].abs().max()))
            for k in ['phi','f','phase_covariance','phase_quality']:
                phaseerr=max(phaseerr,float((getattr(ss,k)[b*B:(b+1)*B]-getattr(ss,k)[:B]).abs().max()))
    assert acterr<1e-12 and symmetry<1e-12 and phaseerr==0
    for tape in tapes:
        assert feasible(tape,AMPLITUDE)[0]
        for sign in [1,-1]:
            plus=pulse(tape,sign*AMPLITUDE);zero=pulse(tape,0.);delta=plus.values-zero.values;active=(np.array(plus.times)>=0)&(np.array(plus.times)<.1)
            assert torch.equal(delta[~active],torch.zeros_like(delta[~active]))
            assert torch.equal(delta[:,[0,3]],torch.zeros_like(delta[:,[0,3]]))
            assert torch.allclose(delta[active,2]-delta[active,1],delta.new_full((sum(active),),2*sign*AMPLITUDE),atol=1e-14,rtol=0)
    # No-control perturbation restores original flow (same motor and tail both entries).
    with torch.no_grad():
        u=commands[0][:B];dt=s.x.new_full((B,1),.005)
        aa=m.dual_step(s,u,u,dt);bb=base.tensor_step(s,u,dt)
        assert all(torch.equal(getattr(aa,k),getattr(bb,k)) for k in FIELDS)
    ss=take(snapshots[3],list(range(B)));pos=take(snapshots[3],list(range(3*B,4*B)));u=commands[3][:B]
    jac=decoder(m,ss,pos,u)[0];eps=1e-3;fd=[]
    def shifted(q):
        return ss.with_(actuator=ss.actuator+q[:,:1]*q.new_tensor([-1.,1.,0.]),c=ss.c+q[:,1:7],z_s=ss.z_s+q[:,7:19],x=so3.retract(ss.x,q[:,19:31]))
    with torch.no_grad():
        for k in range(31):
            q=ss.x.new_zeros((B,31));q[:,k]=eps
            fd.append(((m.vector_field(shifted(q),u)[1][:,0]-m.vector_field(shifted(-q),u)[1][:,0])/(2*eps)).cpu().numpy())
    fd=np.stack(fd,1);error=float(abs(fd-jac).max());assert error<5e-4,error
    result=dict(actuator_analytic_max_error=acterr,symmetric_rudder_actuator_error=symmetry,phase_contract_error=phaseerr,jacobian_all_31_coordinates_train_fd_max_error=error,fd_epsilon=eps,zero_equal_entry_original_exact=True,split_identity_verified=True,all_10944_draw_indices_reproduced=True,commands_and_pulse_interval_verified=True)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()
