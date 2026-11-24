"""Control-prefix model. Translational motion is expressed in the frame fixed at T."""
import torch
from torch import nn
from workspace import *
from harmonic import SegmentHarmonics
from dynamics import rotate, inverse, clock, qexp, qmul, unit, prefix_start, prefix_rotations


def filter_starts(initial, forcing, durations, rate):
    """Exact ZOH first-order filter at the start of every interval; zero duration is identity."""
    step=durations.double()
    start=prefix_start(step)
    mass=-torch.expm1(-rate*step)
    source=forcing.double()*mass*torch.exp(rate*(start+step))
    return ((initial[:,None].double()+prefix_start(source))*torch.exp(-rate*start)).float()


def grid_starts(initial, increments, gain):
    """Scan x[k+1]=(1-0.02*gain)*x[k]+increments[k] in causal order."""
    powers=(1-.02*gain)**torch.arange(increments.size(1)+1,device=increments.device,dtype=torch.float64)
    return ((initial[:,None].double()+prefix_start(increments.double()/powers[None,1:,None]))*
            powers[None,:-1,None]).float()


class ControlMemoryMotion(SegmentHarmonics):
    def __init__(self,mean,scale):
        nn.Module.__init__(self)
        self.structured=True
        self.encoder=nn.GRU(len(mean),64,batch_first=True)
        self.init_hidden=nn.Linear(64,64)
        self.slow_net=nn.Sequential(nn.Linear(64+6+3+4+12+2,112),nn.SiLU(),
                                    nn.Linear(112,96),nn.SiLU(),nn.Linear(96,30))
        self.initial_amplitude=nn.Linear(64,24)
        self.control_rate=nn.Linear(4,30,bias=False)
        for layer in [self.initial_amplitude,self.slow_net[-1],self.control_rate]:
            nn.init.normal_(layer.weight,std=.005)
            if layer.bias is not None: nn.init.zeros_(layer.bias)
        self.register_buffer('mean',mean)
        self.register_buffer('scale',scale)
        self.register_buffer('axis_scale',scale[:6].clamp_min(.1))
        self.register_buffer('down',torch.tensor([0.,0.,1.]))

    def forward(self,history,initial,tape):
        if tape.ndim!=4 or tape.size(-1)!=5 or not tape.size(1):
            raise ValueError('four-channel native control intervals required')
        batch,steps,events,_=tape.shape
        _,encoded=self.encoder((history-self.mean)/self.scale)
        h=torch.tanh(self.init_hidden(encoded[0]))
        q0=initial[:,6:10]
        local=torch.cat((rotate(inverse(q0),initial[:,3:6]),initial[:,10:]),-1)/self.axis_scale
        gravity=rotate(inverse(q0),self.down)
        end,dt,f,basis,_=clock(history,tape)
        cmd=tape[...,:4].flatten(1,2)
        prev=history[:,-1,9:13]
        memories=[filter_starts(prev,cmd,dt,rate)[:,::events] for rate in (2.,5.,12.)]
        time=(end-tape[...,4].flatten(1))[:,::events,None]*1e-6
        fixed=torch.cat([h,local,gravity],-1)[:,None].expand(-1,steps,-1)
        x=torch.cat([fixed,tape[:,:,0,:4],*memories,f[:,::events]/5,time/(1+time)],-1)
        target=self.slow_net(x)
        coeff0=self.initial_amplitude(h)
        fast0=(coeff0.reshape(batch,6,4)*basis[:,0,None]).sum(-1)
        initial_latent=torch.cat([local-fast0,coeff0],-1)
        # The predicted mean is a change from the initial motion in the frame fixed at T.
        target=torch.cat([local[:,None]+target[...,:6],target[...,6:]],-1)
        change=self.control_rate(tape[...,:4]-tape[:,:,:1,:4])
        dur=dt.reshape(batch,steps,events,1)
        gain=3.
        increment=.02*gain*target+(change*dur).sum(2)
        start=grid_starts(initial_latent,increment,gain)
        slope=(gain*(target-start))[:,:,None]+change
        displacement=slope*dur
        before=torch.cat([torch.zeros_like(displacement[:,:,:1]),displacement.cumsum(2)[:,:,:-1]],2)
        latent=(start[:,:,None]+before).flatten(1,2)
        slopes=slope.flatten(1,2)
        c,dc=latent[...,6:],slopes[...,6:]
        def physical(a):
            z=a*self.axis_scale
            return torch.cat([rotate(q0[:,None],z[...,:3]),z[...,3:]],-1)
        slow=physical(latent[...,:6])
        base=physical(slopes[...,:6])
        form=(batch,steps*events,6,4)
        motion=slow+physical((c.reshape(form)*basis[...,None,:]).sum(-1))
        orientation=q0[:,None].expand(-1,steps*events,-1)
        terms=(cmd[...,:1],f,basis,c,dc,orientation,base)
        acceleration=self.partial_acceleration(dt,*terms)
        mid=motion+.5*dt*acceleration
        p=initial[:,None,:3]+prefix_start(dt*mid[...,:3])
        dq=qexp(dt*mid[...,3:])
        qs=unit(qmul(q0[:,None],prefix_rotations(dq)))
        q=torch.cat([q0[:,None],qs[:,:-1]],1)
        state=torch.cat([p,motion[...,:3],q,motion[...,3:]],-1)
        return end,state,acceleration,(slow,base),self.partial_acceleration,terms
