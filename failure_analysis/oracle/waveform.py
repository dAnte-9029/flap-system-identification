"""Held-target-beat high-order waveform transport, with descriptive covariates."""
import numpy as np
import pandas as pd
from run import ROOT,HERE,wave,load,oracle


def main():
    d,f,ids=load('validation')
    _,ref,t,pool,bins=wave.native_cache(ROOT,'cpu')
    ix=ref['window_idx']
    phi,ok=oracle('validation',f[ix],ids[ix],t.numpy(),d['hist'][ix,:180].numpy())
    mem=ROOT/'outputs/history_slow_fast_20261119_cache'
    meta=pd.read_parquet(mem/'validation_windows.parquet').reset_index(drop=True).set_index('window_id')
    loc=meta.index.get_indexer(ids)
    def mmap(key):
        return np.load(mem/f'validation_{key}.npy',mmap_mode='r')
    init,raw,times,valid=map(mmap,['initial','y','t','valid'])
    beats={}
    for jj,wt,q,lead in bins:
        i=jj[0]; w=ix[i]; c=ref['ch'][i]; sec=ref['time'][jj]
        if q is None or not ok[jj,0,0].all() or not d['coef'][w,c].abs().any():continue
        p=phi[jj,0,0].numpy().astype(float)
        cols=[np.ones(len(jj)),(sec-sec.mean())/np.ptp(sec)]
        for k in (1,2,3):
            cols.extend([np.cos(k*p),np.sin(k*p)])
        a=np.column_stack(cols)*np.sqrt(wt)[:,None]
        if np.linalg.cond(a)>100:continue
        co=np.linalg.lstsq(a,ref['data'][jj]*np.sqrt(wt[:,None]),rcond=None)[0][4:].T.reshape(3,2,2)
        idx=loc[w]
        changes=[]
        for stream,field in [(0,slice(3,6)),(1,slice(6,10))]:
            nearest=np.argmin(np.where(valid[idx,:,stream],abs(times[idx,:,stream]-lead),np.inf))
            if not valid[idx,nearest,stream] or abs(times[idx,nearest,stream]-lead)>.04:
                changes.append(np.nan);continue
            x=init[idx,field]; y=raw[idx,nearest,field]
            changes.append(np.linalg.norm(y-x) if stream==0 else 2*np.arccos(
                np.clip(abs(np.dot(x,y))/(np.linalg.norm(x)*np.linalg.norm(y)),0,1)))
        cmd=pool[jj[len(jj)//2],0,0,8:12].numpy()/(-np.expm1(-sec[len(jj)//2]))
        freq=(p[-1]-p[0])/(2*np.pi*np.ptp(sec))
        cov=[abs(freq-float(d['hist'][w,165])),*changes,np.linalg.norm(cmd-d['u0'][w].numpy())]
        beats.setdefault((w,c),[]).append((sec,co,cov,lead))
    records=[]
    for (w,c),items in beats.items():
        items.sort(key=lambda v:v[3])
        past=d['coef'][w,c].numpy()
        for n in range(1,len(items)-1):
            sec,co,cov,lead=items[n]
            prev,after=items[n-1],items[n+1]
            if not(0<sec[0]-prev[0][-1]<.04 and 0<after[0][0]-sec[-1]<.04):continue
            neighbor=(prev[1]+after[1])/2
            for axis in range(3):
                for order in range(2):
                    hist,y,nb,pr=(x[axis,order] for x in (past,co,neighbor,prev[1]))
                    records.append([f[w],w,c,lead,axis,order+2,np.linalg.norm(hist),np.linalg.norm(y),
                        *hist,*y,((hist-y)**2).sum()/2,((nb-y)**2).sum()/2,((pr-y)**2).sum()/2,
                        (y**2).sum()/2,*cov])
    df=pd.DataFrame(records,columns='flight window stream lead axis order hist_amp target_amp hist_a hist_b target_a target_b hist_mse neighbor_mse prev_mse zero_mse dfreq dspeed datt dcommand'.split())
    df['period']=pd.cut(df.lead,[0,.5,1,np.inf],labels=['early','middle','late'])
    def summarize(g):
        ret={k:g[k].mean() for k in ('hist_mse','neighbor_mse','prev_mse','zero_mse')}
        def corr(x,y):
            xx=np.asarray(x).ravel()
            yy=np.asarray(y).ravel()
            ok=np.isfinite(xx)&np.isfinite(yy)
            if len(g)<6 or ok.sum()<6 or np.ptp(xx[ok])<1e-10 or np.ptp(yy[ok])<1e-10:
                return np.nan
            return np.corrcoef(xx[ok],yy[ok])[0,1]
        ret['r_amplitude']=corr(g.hist_amp,g.target_amp)
        ret['r_shape']=corr(g[['hist_a','hist_b']],g[['target_a','target_b']])
        for k in ['dfreq','dspeed','datt','dcommand']:
            ret['r_'+k]=corr(g[k],g.hist_mse)
        return pd.Series(dict(n=len(g),**ret))
    df.groupby(['flight','stream','period','order','axis'],observed=True).apply(summarize,include_groups=False).to_csv(HERE/'waveform.csv')
    print('target beats',len(df)//6,'windows',df.window.nunique(),'flights',df.flight.nunique())


if __name__=='__main__':
    main()
