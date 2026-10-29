"""Atomic terminal-readable progress; no monitor process required."""
import json
import math
import os
from pathlib import Path
import time
from collections import deque


def atomic_json(path, value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


class RunStatus:
    def __init__(self,output,metadata,total,step,interval):
        self.path=Path(output)/'status.json';self.started=time.time();self.times=deque(maxlen=100)
        self.validation_times=[];self.interval=interval
        self.value=dict(pid=os.getpid(),device=metadata.get('device','cpu'),seed=metadata.get('seed'),
                        variant=metadata.get('model_config',{}).get('mode'),step=step,total=total,
                        phase='training',state='running',started=self.started,last_update=None)
        self.flush()

    def flush(self):
        self.value.update(heartbeat=time.time(),elapsed_seconds=time.time()-self.started)
        rate=sum(self.times)/len(self.times) if self.times else None
        remaining=self.value['total']-self.value['step']
        val=sum(self.validation_times)/len(self.validation_times) if self.validation_times else None
        # Full ETA needs measured training AND validation costs.
        eta=remaining*rate+math.ceil(remaining/self.interval)*val if rate is not None and val is not None else None
        self.value.update(recent_step_seconds=rate,validation_seconds=val,eta_seconds=eta)
        atomic_json(self.path,self.value)

    def update(self,draw):
        self.times.append(draw['update_seconds'])
        self.value.update(step=draw['step'],loss=draw['total'],gradient_norm=draw['gradient_norm'],
                          data_seconds=draw['data_seconds'],last_update=time.time(),phase='training')
        if draw['step']<=5 or draw['step']%25==0:self.flush()

    def validation_start(self):
        self.val_start=time.monotonic();self.value.update(phase='validation',validation_done=0);self.flush()

    def validation_window(self,done,total):
        self.value.update(validation_done=done,validation_total=total)
        if done<=3 or done%5==0 or done==total:self.flush()

    def validation_end(self):
        self.validation_times.append(time.monotonic()-self.val_start);self.flush()

    def finish(self):
        self.value.update(state='completed',phase='completed',ended=time.time());self.flush()
