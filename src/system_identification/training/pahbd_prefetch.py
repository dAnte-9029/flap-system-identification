"""Bounded CPU lookahead; only consumed updates advance the checkpoint sampler."""
import copy
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import os
import time
from system_identification.data.pahbd_transfer import cpu_windows,window_to_device


class WindowStream:
    def __init__(self,windows,sampler,steps,prefetch=False,cpu=None):
        self.windows=windows;self.sampler=sampler;self.remaining=steps
        self.pending=deque();self.executor=None
        if prefetch:
            self.loader=cpu_windows(windows)
            self.planned=copy.copy(sampler)
            self.planned.rng=copy.deepcopy(sampler.rng)
            self.planned.load_state_dict(sampler.state_dict())
            def affinity():
                if cpu is not None:os.sched_setaffinity(0,{cpu})
            self.executor=ThreadPoolExecutor(max_workers=1,initializer=affinity)
            for _ in range(min(2,steps)):self._submit()

    def _submit(self):
        if self.remaining:
            index=self.planned.draw();self.remaining-=1
            def load():
                start=time.monotonic();w=self.loader[index]
                return w,time.monotonic()-start
            self.pending.append((index,self.executor.submit(load)))

    def next(self):
        index=self.sampler.draw()
        if self.executor is None:
            start=time.monotonic();w=self.windows[index]
            return index,w,time.monotonic()-start
        planned,future=self.pending.popleft()
        if planned!=index:raise RuntimeError('prefetch sampling order diverged')
        w,load_seconds=future.result();self._submit()
        return index,window_to_device(w,self.windows.dataset.device),load_seconds

    def close(self):
        if self.executor:self.executor.shutdown(wait=True,cancel_futures=True)
