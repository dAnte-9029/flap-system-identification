"""History-anchored slow motion with causal native-segment harmonic reconstruction."""
import math
import torch
from workspace import *
from harmonic import SegmentHarmonics
from dynamics import rotate, clock, inverse, prefix_start, qexp, prefix_rotations, unit, qmul


class MeanMotion(SegmentHarmonics):
    def __init__(self, mean, scale):
        super().__init__(mean, scale, True)
        self.relaxation = torch.nn.Parameter(torch.full((6,),math.log(1/.35)))

    def mean_target(self,anchor,hidden,encoded,body,gravity):
        return anchor + self.trend(hidden)

    def transport_drive(self,body,hidden):
        return 0.

    def forward(self,history,initial,tape):
        """tape[B,20ms steps,segments,5] holds motor/left/right/rudder/duration_us.

        Constant-grid MPC candidates use one segment of 20000 us per step.
        Event-packed candidates use right-padded zero durations.
        """
        if tape.ndim!=4 or tape.shape[-1]!=5 or tape.shape[1]==0:
            raise ValueError('expected a positive-length batched four-channel control tape')
        normalized = (history-self.mean)/self.scale
        _,encoded = self.encoder(normalized)
        hidden = torch.tanh(self.init_hidden(encoded[0]))
        end,all_dt,freq,basis,_ = clock(history,tape)
        u = tape[...,:4].flatten(1,2)
        width = self.block*tape.shape[2]
        q = initial[:,6:10]
        motion = torch.cat([initial[:,3:6],initial[:,10:13]],-1)
        prev_command = history[:,-1,9:13]
        observed = torch.cat([history[:,:,:3],history[:,:,3:6]],-1)
        masks = torch.cat([(history[:,:,16:17]*history[:,:,18:19]).expand(-1,-1,3),
                           history[:,:,17:18].expand(-1,-1,3)],-1)
        anchor = (observed*masks).sum(1)/masks.sum(1).clamp_min(1)/self.axis_scale
        coefficient_basis, coefficient_slope = self.interval_bases(u[...,0:1], all_dt, freq, basis)
        coeff = self.initial_amplitude(encoded[0])
        r0 = (coeff.reshape(-1,6,4)*basis[:,0,None]).sum(-1)*self.axis_scale
        slow = motion-torch.cat([rotate(q,r0[:,:3]),r0[:,3:]],-1)
        positions,quats,motions,trends,slopes,slow_slopes = [],[],[],[],[],[]
        coeff_starts, coeff_rates, orientations = [], [], []
        p = initial[:,:3]
        for start in range(0,u.size(1),width):
            sl = slice(start,start+width)
            dt,cmd = all_dt[:,sl],u[:,sl]
            body = torch.cat([rotate(inverse(q),slow[:,:3]),slow[:,3:]],-1)/self.axis_scale
            context = [body,rotate(inverse(q),self.down),prev_command,cmd[:,0],freq[:,start]/5.]
            inputs = torch.cat(context,-1)
            hidden = self.cell(inputs,hidden)
            # A changed command has a direct differentiable entry within this block.
            change = cmd-cmd[:,:1]
            target_motion = self.mean_target(anchor,hidden,encoded[0],body,context[1])
            drive=(target_motion-body)*self.relaxation.exp() + self.transport_drive(body,hidden)
            base=drive[:,None]+self.control_rate(change)
            target = self.amplitude(torch.cat([hidden,coeff],-1))[:,None]+self.control_amplitude(change)
            dc = 2.*(target-coeff[:,None])
            c = coeff[:,None]+prefix_start(dc*dt)
            form = (*c.shape[:2],6,4)
            harmonic = (dc.reshape(form)*coefficient_basis[:,sl,None]+c.reshape(form)*coefficient_slope[:,sl,None]).sum(-1)
            accel = base+harmonic
            coeff_starts.append(c)
            coeff_rates.append(dc.expand_as(c))
            orientations.append(q[:,None].expand(-1,c.shape[1],-1))
            coeff = coeff+(dc*dt).sum(1)
            def physical(rate):
                rate = rate*self.axis_scale
                return torch.cat([rotate(q[:,None],rate[...,:3]),rate[...,3:]],-1)
            acc = physical(accel)
            dv = dt*acc
            path = motion[:,None]+prefix_start(dv)
            slow_acc = physical(base)
            ds = dt*slow_acc
            trends.append(slow[:,None]+prefix_start(ds))
            slow_slopes.append(slow_acc)
            slow = slow+ds.sum(1)
            dp = dt*(path[...,:3]+.5*dv[...,:3])
            dq = qexp(dt*(path[...,3:]+.5*dv[...,3:]))
            rel = prefix_rotations(dq)
            absolute = unit(qmul(q[:,None],rel))
            positions.append(p[:,None]+prefix_start(dp))
            quats.append(torch.cat([q[:,None],absolute[:,:-1]],1))
            motions.append(path)
            slopes.append(acc)
            p,q = p+dp.sum(1),absolute[:,-1]
            motion = motion+dv.sum(1)
            prev_command = (cmd*dt).sum(1)/dt.sum(1).clamp_min(1e-8)
        state = torch.cat([torch.cat(positions,1),torch.cat(motions,1)[...,:3],torch.cat(quats,1),
                           torch.cat(motions,1)[...,3:]],-1)
        trend = (torch.cat(trends,1),torch.cat(slow_slopes,1))
        result = end,state,torch.cat(slopes,1),trend
        terms = (u[...,0:1],freq,basis,torch.cat(coeff_starts,1),torch.cat(coeff_rates,1),
                 torch.cat(orientations,1),trend[1])
        return (*result,self.partial_acceleration,terms)


class HistoryConditionedMotion(MeanMotion):
    def __init__(self,mean,scale):
        super().__init__(mean,scale)
        self.history_field=torch.nn.Sequential(torch.nn.Linear(64+48+6+3,64),torch.nn.SiLU(),torch.nn.Linear(64,6))
        torch.nn.init.normal_(self.history_field[-1].weight,std=.005)
        torch.nn.init.zeros_(self.history_field[-1].bias)

    def mean_target(self,anchor,hidden,encoded,body,gravity):
        history_term=self.history_field(torch.cat([encoded,hidden,body,gravity],-1))
        return anchor+self.trend(hidden)+history_term


class TurningMotion(HistoryConditionedMotion):
    def __init__(self,mean,scale):
        super().__init__(mean,scale)
        self.turn_response=torch.nn.Linear(3,6,bias=False)
        self.turn_gate=torch.nn.Linear(48,6)
        with torch.no_grad():
            self.turn_response.weight.zero_()
            self.turn_response.weight[:3].copy_(torch.eye(3)*.25)
            self.turn_gate.weight.zero_()
            self.turn_gate.bias.zero_()

    def transport_drive(self,body,hidden):
        motion=body*self.axis_scale
        transport=torch.linalg.cross(motion[:,3:],motion[:,:3])/self.axis_scale[:3]
        return self.turn_response(transport)*torch.sigmoid(self.turn_gate(hidden))


def build(name,mean,scale):
    if name=='s3c':
        return TurningMotion(mean,scale)
    if name=='s3a':
        from control_memory import ControlMemoryMotion
        return ControlMemoryMotion(mean,scale)
    if name in ('s2b','s2c','s3b'):
        model=HistoryConditionedMotion(mean,scale)
        if name=='s2c':
            model.block=2
        return model
    if name == 's2a':
        return MeanMotion(mean,scale)
    raise ValueError(name)
