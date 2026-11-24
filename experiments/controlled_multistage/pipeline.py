"""Rebuild real histories before and after removal of raw post-decision publications."""
import json
import numpy as np
import pandas as pd
from workspace import *
from evaluate import restore
from train_three import data_three
from data3 import long_control
import argparse
import torch
from prepare import TOPICS, history, native_topic
from controls import native_controls, pack


def inputs(three):
    dataset = ROOT/'dataset/velocity_rate_multistep_v1'
    all_ids = pd.read_parquet(HISTORY/'validation_windows.parquet')
    admitted = pd.read_parquet(TAPES/'validation_ids.parquet')
    ids = admitted.merge(all_ids.reset_index(names='history_index'), validate='one_to_one')
    if three:
        ids=pd.read_parquet(CACHE3/'validation_windows.parquet')
    selected = ids.groupby('flight_id', sort=False).head(1)
    windows = pd.read_parquet(dataset/'windows_validation.parquet', columns=['window_id','support_start_us'],
                              filters=[('history_us','==',500000),('horizon_us','==',2000000)])
    selected = selected.join(windows.set_index('window_id'), on='window_id', validate='one_to_one')
    ratio = pd.read_csv(dataset/'flights.csv',usecols=['flight_id','transmission_ratio']).set_index('flight_id')
    tables = {n:pd.read_parquet(dataset/f'native_{n}_validation.parquet').groupby('flight_id') for n in TOPICS}
    cache = {k:np.load(HISTORY/f'validation_{k}.npy',mmap_mode='r') for k in ('initial','history')}
    tapes = np.load((CACHE3 if three else TAPES)/'validation_commands.npy',mmap_mode='r')
    full_hist, raw_hist, full_init, raw_init, factual, plans, changed = [], [], [], [], [], [], []
    removed = {n:0 for n in TOPICS}
    for i,w in selected.iterrows():
        t = w.decision_us
        frame = {n:g.get_group(w.flight_id) for n,g in tables.items()}
        before = {n:a[a.get('availability_us',a.timestamp_us)<=t] for n,a in frame.items()}
        for n in TOPICS:
            removed[n] += len(frame[n])-len(before[n])
        def rebuild(frames):
            topics = {n:native_topic(a,n) for n,a in frames.items()}
            return history(topics,selected.loc[[i]],w.sampling_group,ratio.loc[w.flight_id,'transmission_ratio'])
        h,s,ok = rebuild(frame)
        ht,st,okt = rebuild(before)
        assert ok.all() and okt.all()
        np.testing.assert_array_equal(s,st)  # including absolute NED origin
        s[:,:3] = st[:,:3] = 0
        for key,x,xt in [('history',h,ht),('initial',s,st)]:
            np.testing.assert_array_equal(x,xt)
            np.testing.assert_array_equal(x[0].astype(np.float32),cache[key][w.history_index])
        full_hist.append(h);raw_hist.append(ht);full_init.append(s);raw_init.append(st)
        factual.append(tapes[i:i+1])
        clock,u,_ = native_controls(frame['actuator_motors'],frame['actuator_servos'])
        # MPC plan pair shares all original events up to an inserted change inside a 20 ms step.
        cutoff = t+(2667000 if three else 667000)
        keep = clock<cutoff
        event_t = np.r_[clock[keep],cutoff]
        value = u[keep][-1].copy()
        value[0] = .15 if value[0]>.5 else .85
        value[1:] = np.where(value[1:]>0,-.5,.5)
        origin,stop = np.array([t]),np.array([t+(3000000 if three else 2000000)])
        packer=long_control if three else pack
        plans.append(packer(clock,u,origin,stop))
        changed.append(packer(event_t,np.vstack([u[keep],value]),origin,stop))
    def tensor(xs):
        return torch.from_numpy(np.concatenate(xs).astype(np.float32)).cuda()
    def padded(xs,width):
        return tensor([np.pad(x,((0,0),(0,0),(0,width-x.shape[2]),(0,0))) for x in xs])
    width=max(x.shape[2] for x in plans+changed)
    assert len(selected)==39 and min(removed.values())>0
    return [tensor(z) for z in (full_hist,raw_hist,full_init,raw_init,factual)],padded(plans,width),padded(changed,width),removed


@torch.no_grad()
def check(setting,names):
    setup()
    three=setting!='2s'
    (h,ht,s,st,tape),plan,changed,removed=inputs(three)
    cutoff=2667000 if three else 667000
    ticks=torch.arange(0,3000001 if three else 2000001,5000,device='cuda')
    ticks=torch.cat((ticks,torch.tensor([cutoff-1,cutoff,cutoff+1],device='cuda'))).sort().values
    t=(ticks.float()*1e-6).expand(len(s),-1)
    results={}
    for name in names:
        m,_=restore(name,setting=='trained3')
        original,trimmed=m(h,s,tape),m(ht,st,tape)
        a,b=m(h,s,plan),m(h,s,changed)
        short=m(h,s,tape[:,:100]) if three else None
        prefix_max=obs_max=post_max=trunc_max=0.
        for c in ('pv','q','w'):
            x,y=query(original,t,c),query(trimmed,t,c)
            torch.testing.assert_close(x,y,atol=0,rtol=0)
            obs_max=max(obs_max,(x-y).abs().max().item())
            if three:
                common=ticks<=2000000
                early=query(short,t[:,common],c)
                torch.testing.assert_close(early,x[:,common],atol=1e-5,rtol=1e-6)
                trunc_max=max(trunc_max,(early-x[:,common]).abs().max().item())
            x,y=query(a,t,c),query(b,t,c)
            prefix=ticks<=cutoff
            torch.testing.assert_close(x[:,prefix],y[:,prefix],atol=1e-5,rtol=0)
            prefix_max=max(prefix_max,(x[:,prefix]-y[:,prefix]).abs().max().item())
            post_max=max(post_max,(x[:,~prefix]-y[:,~prefix]).abs().max().item())
        assert post_max>1e-4
        results[name]=dict(observation_removal=obs_max,control_prefix=prefix_max,post_change=post_max,cut_to_2s=trunc_max)
    record=dict(windows=len(s),removed_raw_rows=removed,intervention_us=cutoff,checks=results)
    (RUN/f'{setting}_pipeline.json').write_text(json.dumps(record,indent=2))
    print(results)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('setting',choices=['2s','extrapolated3','trained3'])
    p.add_argument('names',nargs='+')
    a=p.parse_args()
    check(a.setting,a.names)
