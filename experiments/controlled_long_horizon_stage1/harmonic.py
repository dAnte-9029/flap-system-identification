"""C2 integrates c(t)*B(phase(t)) by its endpoints on each native command segment."""
import math
import torch
from common import make_model
from dynamics import ControlledGRU, rotate, clock, inverse, prefix_start, qexp, prefix_rotations, unit, qmul
from stage_query import query


class SegmentHarmonics(ControlledGRU):

    def partial_acceleration(self, dt, motor, freq, basis, coeff, rate, orientation, base):
        # A dense query uses only time elapsed; a later event may shorten this segment.
        end, chord = self.interval_bases(motor, dt, freq, basis)
        shape = (*coeff.shape[:2], 6, 4)
        periodic = (rate.reshape(shape) * end[..., None, :] +
                    coeff.reshape(shape) * chord[..., None, :]).sum(-1) * self.axis_scale
        return base + torch.cat([rotate(orientation, periodic[..., :3]), periodic[..., 3:]], -1)

    def interval_bases(self, motor, dt, frequency, basis):
        # Same train-fitted motor/frequency ODE as the shared control clock.
        b0, bu, bf = 3.6354511983566393, 23.393570068972757, -5.126725395181777
        step = dt.double()
        f0 = frequency.double()
        stationary = -(b0 + bu * motor.double()) / bf
        cycles = stationary * step + (f0 - stationary) * torch.expm1(bf * step) / bf
        advance = (2 * math.pi * cycles).to(basis.dtype)
        ends, slopes = [], []
        for k in (1, 2):
            c, s = basis[..., 2*k-2:2*k-1], basis[..., 2*k-1:2*k]
            a = k * advance
            ends.extend((c*a.cos() - s*a.sin(), s*a.cos() + c*a.sin()))
            half = a * .5
            speed = a / dt.clamp_min(1e-7) * torch.sinc(half/math.pi)
            slopes.extend((-speed * (s*half.cos() + c*half.sin()),
                            speed * (c*half.cos() - s*half.sin())))
        # For linearly evolving c, Δ(cB)/dt = dc/dt * B_end + c_start * ΔB/dt.
        return torch.cat(ends, -1), torch.cat(slopes, -1)

    def rollout(self, history, initial, commands):
        """MPC interface: [batch, horizon, 4] commands -> [batch, horizon+1, 13].

        Commands are ZOH on 20 ms intervals. All candidate values remain differentiable.
        Physical state is [p_NED, v_NED, Hamilton q_NB, omega_FRD].
        """
        if commands.ndim != 3 or commands.shape[-1] != 4:
            raise ValueError('expected [batch, horizon, 4] motor/left/right/rudder')
        duration = torch.full_like(commands[..., :1], 20000.)
        tape = torch.cat([commands, duration], -1).unsqueeze(2)
        roll = self(history, initial, tape)
        horizon = commands.size(1)
        times = torch.arange(1, horizon+1, device=commands.device, dtype=commands.dtype)*.02
        times = times.expand(commands.size(0), -1)
        path = torch.cat([query(roll,times,'pv'),query(roll,times,'q'),query(roll,times,'w')], -1)
        return torch.cat([initial[:,None],path],1)

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
            base = self.trend(hidden)[:,None]+self.control_rate(change)
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


def candidate_model(name, mean, scale):
    if name == 'c1':
        return make_model('slowfast', mean, scale)
    if name == 'c2':
        return SegmentHarmonics(mean, scale, True)
    raise ValueError(name)
