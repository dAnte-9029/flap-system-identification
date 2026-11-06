"""Rank-zero plain text monitor. Timing is observational, never part of the loss."""
from collections import deque
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import math
import statistics
import subprocess
import time


def duration(seconds):
    if seconds is None: return 'N/A'
    seconds=max(0,int(seconds)); h,seconds=divmod(seconds,3600);m,s=divmod(seconds,60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def number(value, suffix='', digits=3):
    return 'N/A' if value is None else f'{value:9.{digits}f}{suffix}'


def read_gpu(device):
    try:
        values=subprocess.check_output(['nvidia-smi','-i',str(device),'--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True,timeout=2).strip()
        usage,memory=map(float,values.split(','))
        return dict(utilization_percent=usage,memory_gib=memory/1024)
    except (OSError,ValueError,subprocess.SubprocessError):
        return dict(utilization_percent=None,memory_gib=None)


class Telemetry:
    """External process runs in a worker; the optimizer never waits for it."""
    def __init__(self,device):
        self.device=device;self.pool=ThreadPoolExecutor(max_workers=1);self.pending=None
        self.latest=dict(utilization_percent=None,memory_gib=None)
    def sample(self,request=False):
        if self.pending is not None and self.pending.done():
            self.latest=self.pending.result();self.pending=None
        if request and self.pending is None:self.pending=self.pool.submit(read_gpu,self.device)
        return dict(self.latest)
    def close(self):self.pool.shutdown(wait=True)


class Monitor:
    def __init__(self,path,seed,device,batch,total,interval,schedule):
        self.file=path.open('a',buffering=1);self.seed=seed;self.device=device;self.batch=batch
        self.total=total;self.interval=interval;self.schedule=schedule;self.start=time.perf_counter()
        previous=schedule.get('monitor',{})
        self.elapsed_before=previous.get('elapsed_seconds',0.)
        self.times=deque(previous.get('update_seconds',[]),maxlen=40)
        self.val_times=deque(previous.get('validation_seconds',[]),maxlen=10)
        self.last=previous.get('last_validation',{});self.stats={};self.gpu={}
        self.session_updates=0
    def write(self,text):
        self.file.write(text+'\n');self.file.flush();print(text,flush=True)
    def observe(self,stats,gpu):
        self.stats=stats;self.gpu=gpu;self.session_updates+=1
        # First update in each process includes compilation / graph loading.
        if self.session_updates>1:self.times.append(stats['end_to_end_seconds'])
    def snapshot(self):
        return dict(elapsed_seconds=self.elapsed_before+time.perf_counter()-self.start,
                    update_seconds=list(self.times),validation_seconds=list(self.val_times),last_validation=self.last)
    def status(self,status):
        s=self.schedule;n=s['optimizer_steps'];remaining=self.total-n
        avg=statistics.mean(self.times) if self.times else None
        stable=self.session_updates>0 and len(self.times)>=20 and statistics.pstdev(self.times)/avg<.75
        count=(self.total//self.interval-n//self.interval)+int(self.total%self.interval!=0)
        if n==self.total:count=0
        if n>0 and (n%self.interval==0 or n==self.total) and self.last.get('step')!=n:count+=1
        update_eta=remaining*avg if stable else None
        val_eta=count*statistics.mean(self.val_times) if self.val_times else None
        eta=update_eta+val_eta if update_eta is not None and val_eta is not None else None
        if status=='FINISHED':eta=0
        finish=(datetime.now()+timedelta(seconds=eta)).strftime('%Y-%m-%d %H:%M') if eta is not None else 'N/A'
        next_val=min((n//self.interval+1)*self.interval,self.total) if remaining else None
        st=self.stats;g=self.gpu
        self.write('\n'.join([
            '='*62,f' PACLD GPU TRAINING | seed={self.seed} | GPU {self.device} | Batch={self.batch}', '='*62,
            f' Status        : {status}'+(' | WARMUP / ETA CALIBRATING' if eta is None else ''),
            f' Progress      : {n:4d} / {self.total} steps [{100*n/self.total:5.1f}%]',
            f" Windows       : {s['windows_seen']} / {s['windows_seen']+remaining*self.batch}",
            f' Time elapsed  : {duration(self.snapshot()["elapsed_seconds"])}',
            f' ETA remaining : {duration(eta)} (dynamic estimate)',
            f'   Updates     : {duration(update_eta)}',
            f'   Validation  : {duration(val_eta)}'+(' (pending calibration)' if not self.val_times else ''),
            f' Finish around : {finish} (server local time)', '', ' Performance',
            f'   Batch time  : {number(avg," s/step")}',
            f'   Throughput  : {number(self.batch/avg if avg else None," windows/s")}',
            f'   GPU usage   : {number(g.get("utilization_percent"),"%",0)}',
            f'   GPU memory  : {number(g.get("memory_gib")," GiB")}', '', ' Training Loss',
            *[f'   {label:12s}: {number(st.get(key))}' for label,key in [('FULL','full'),('Macro','macro'),('Total','total'),('Grad norm','grad_norm')]],
            f'   Clipped     : {st.get("clipped","N/A")}', '', ' Validation',
            f'   Last step   : {self.last.get("step","N/A")}',
            f'   FULL score  : {number(self.last.get("score"))}',
            f'   Best score  : {number(s["best_score"] if math.isfinite(s["best_score"]) else None)} (step {s.get("best_step","N/A")})',
            f'   Next val    : {next_val if next_val is not None else "N/A"}',
            f' Checkpoint    : {s.get("last_checkpoint") or "N/A"}', '='*62]))
    def validation_end(self,step,score,seconds,best):
        prior=self.last.get('score');delta=score-prior if prior is not None else None
        self.val_times.append(seconds);self.last=dict(step=step,score=score)
        self.write(f'[VALIDATED] step={step} | FULL={score:.6f} | change={number(delta)} | new best={best} | {seconds:.2f} s')
    def close(self):self.file.close()
