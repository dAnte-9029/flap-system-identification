"""Score component forecasts and future-carrier interventions against unseen native motions."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from prepare_reference import HERE, PROJECT, OLD, load_module

FEATURES = PROJECT / 'outputs/autopilot_periodic_motion_20261125/validation_features.npy'
TAPES = PROJECT / 'outputs/controlled_slow_fast_20261119_tapes/validation_commands.npy'
POOL = PROJECT / '.worktrees/periodic-motion-20261125-protocol/experiments/periodic_motion/prepare_native.py'


def predict(ref, source, checkpoint):
    torch.set_num_threads(2)
    mod = load_module('evaluated_model', source)
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')
    model = mod.FlightPredictor(state['mean'], state['std'])
    model.load_state_dict(state)
    model.eval()
    pool = load_module('periodic_native_pool', POOL).pool_commands
    ix, ch, t = ref['window_idx'], ref['ch'], ref['time'].astype('float32')
    n = len(ix)
    warm = {}
    with np.load(OLD) as old:
        for key in ('hist', 'u0', 'time', 'pooled'):
            warm[key] = old[key]
        np.testing.assert_array_equal(old['flight'][ix], ref['flight_ids'][ref['flight_index']])
    warm['hist'] = np.c_[warm['hist'], np.load(FEATURES)]
    test_windows = np.unique(ix)
    test_windows = test_windows[np.linspace(0,len(test_windows)-1,24).astype(int)]
    # These rows use the same per-query path as the component probe. Compare them to
    # the original full eight-step grid and its stored pooled commands below.
    ii,jj,cc=np.indices((len(test_windows),8,2)).reshape(3,-1)
    wi=test_windows[ii]
    ix,ch,t=np.r_[ix,wi],np.r_[ch,cc],np.r_[t,warm['time'][wi,jj,cc]]
    origins=np.vstack([ref['origin'],np.zeros((len(wi),3))])
    tape = np.load(TAPES, mmap_mode='r')
    # Duplicate only the current endpoint in the unused other stream, whose recurrence
    # is entirely independent. Preceding queries preserve their exact native grid.
    raw_pool = np.empty((len(t), 16), np.float32)
    for i in np.unique(ix):
        pick = np.flatnonzero(ix == i)
        raw_pool[pick] = pool(tape[i:i+1], t[None,pick])[0]
    preceding = (warm['time'][ix, :, ch] < t[:,None] - 1e-6).sum(1)
    results = {n:np.empty((len(t),3), np.float64) for n in ('slow_abs','fast_abs','origin_fast')}
    with torch.inference_mode():
        for q in np.unique(preceding):
            subset = np.flatnonzero(preceding==q)
            for start in range(0,len(subset),384):
                ids = subset[start:start+384]
                w = ix[ids]
                stream = ch[ids]
                # The unused clock uses the same native cadence in this evaluation;
                # the selected stream then exactly matches its original prefix.
                clock = warm['time'][w,:q,stream]
                clock = np.concatenate((clock, t[ids,None]),axis=1)
                pf = warm['pooled'][w,:q,stream,:]
                pf = np.concatenate((pf,raw_pool[ids,None,:]),axis=1)
                times = np.repeat(clock[...,None],2,axis=2)
                pooling = np.repeat(pf[:,:,None,:],2,axis=2)
                args = [torch.from_numpy(a) for a in (warm['hist'][w],pooling,warm['u0'][w],times)]
                slow, fast, p0 = model.components(*args)
                row = np.arange(len(ids))
                slow, fast, p0 = [a[row,-1,stream,:].numpy() for a in (slow,fast,p0)]
                results['slow_abs'][ids] = origins[ids] + slow - p0
                results['fast_abs'][ids] = fast
                results['origin_fast'][ids] = p0
                if start==0:
                    check = model(*args)[row,-1,stream,:].numpy()
                    np.testing.assert_allclose(check, slow + fast - p0, atol=1e-6)
        official_args=[torch.from_numpy(warm[k][test_windows]) for k in ('hist','pooled','u0','time')]
        official=model(*official_args).numpy()[ii,jj,cc]
        probe=results['slow_abs'][n:]+results['fast_abs'][n:]
        np.testing.assert_allclose(probe,official,atol=5e-5,rtol=3e-6)
    assert all(np.isfinite(v).all() for v in results.values())
    return {k:v[:n] for k,v in results.items()},float(np.max(abs(probe-official)))


def compare(byflight, reference, proposed):
    a,b = byflight[reference].to_numpy(),byflight[proposed].to_numpy()
    rng = np.random.default_rng(17)
    ix = rng.integers(len(a),size=(10000,len(a)))
    gain = np.sqrt(a[ix].mean(1)) - np.sqrt(b[ix].mean(1))
    r,p = np.sqrt(a.mean()),np.sqrt(b.mean())
    return dict(reference_rmse=r, model_rmse=p, gain=r-p,
                interval95=np.quantile(gain,[.025,.975]).tolist(),
                percent=float(100*(1-p/r)) if r>0 else None,
                improved=int((b<a).sum()), flights=len(a))


def block_values(ref, pred):
    starts = np.r_[0,1+np.flatnonzero(np.diff(ref['block']))]
    b = ref['block']
    wt = ref['weight']
    def mean(x):
        scale = wt.reshape((-1,) + (1,)*(x.ndim-1))
        return np.add.reduceat(x*scale,starts)
    np.testing.assert_allclose(mean(np.ones(len(b))), 1.,atol=1e-9)
    def center(x):
        return x-mean(x)[b]
    def mse(x):
        return mean(np.sum(x*x,axis=1))
    slow,fast,meas=ref['slow'],ref['fast'],ref['data']
    body = dict(block=ref['block'][starts],ch=ref['ch'][starts],
                flight=ref['flight_index'][starts],window=ref['window_idx'][starts],
                lead=mean(ref['time']),
                trend_only=mse(meas-ref['trend_only']), trend_wave=mse(meas-slow-fast),
                zero_periodic=mse(fast),centered_zero=mse(center(fast)))
    if pred is None:
        return pd.DataFrame(body)
    s,p=pred['slow_abs'],pred['fast_abs']
    # These terms compare branches after removing the arbitrary common offset in each beat.
    body.update(periodic_error=mse(p-fast),centered_fast_error=mse(center(p)-center(fast)),
                swapped_fast_error=mse(center(s)-center(fast)), full_observation=mse(meas-s-p),
                drop_future_wave=mse(meas-s),flip_future_wave=mse(meas-s+p),
                absolute_slow_error=mse(s-slow),absolute_periodic_rms=mse(p))
    for label, arr in [('slow_ref',slow),('slow_pred',s),('fast_pred',p),('observation',meas)]:
        for axis, val in enumerate(mean(arr).T):
            body[f'{label}{axis}']=val
    df=pd.DataFrame(body)
    # Each horizon is compared with the same model's and reference's first full beat.
    keys=['flight','ch','window']
    for label in ('slow_ref','slow_pred','fast_pred','observation'):
        columns=[f'{label}{axis}' for axis in range(3)]
        delta=df[columns]-df.groupby(keys,sort=False)[columns].transform('first')
        df[[f'd_{n}' for n in columns]]=delta
    delta=lambda label: df[[f'd_{label}{axis}' for axis in range(3)]].to_numpy()
    df['zero_slow_change']=(delta('slow_ref')**2).sum(1)
    df['slow_change_error']=((delta('slow_pred')-delta('slow_ref'))**2).sum(1)
    df['swapped_slow_error']=((delta('fast_pred')-delta('slow_ref'))**2).sum(1)
    df['is_later']=df.groupby(keys).cumcount()>0
    return df


def summarize(df):
    result={}
    for c,stream in enumerate(('velocity','angular_rate')):
        r={}
        for lower,upper,label in [(0,2.,'all'),(0,.5,'0-0.5'),(.5,1.,'0.5-1'),(1.,2.,'1-2')]:
            frame=df[(df.ch==c)&(df.lead>lower)&(df.lead<=upper)]
            aggregate=lambda f:f.groupby(['flight','window']).mean(numeric_only=True).groupby('flight').mean()
            g=aggregate(frame)
            section=dict(windows=int(frame.window.nunique()),beat_window_pairs=len(frame),
                         reference_skill=compare(g,'trend_only','trend_wave'))
            if 'centered_fast_error' in g:
                later=aggregate(frame[frame.is_later])
                section.update(
                    wave_beats_zero=compare(g,'zero_periodic','periodic_error'),
                    centered_wave_beats_zero=compare(g,'centered_zero','centered_fast_error'),
                    centered_wave_beats_swapped_slow=compare(g,'swapped_fast_error','centered_fast_error'),
                    slow_beats_persistence=compare(later,'zero_slow_change','slow_change_error'),
                    slow_beats_swapped_wave=compare(later,'swapped_slow_error','slow_change_error'),
                    future_wave_beats_drop=compare(g,'drop_future_wave','full_observation'),
                    future_wave_beats_flip=compare(g,'flip_future_wave','full_observation'),
                    slow_absolute_rmse=float(np.sqrt(g.absolute_slow_error.mean())),
                    wave_rms=float(np.sqrt(g.absolute_periodic_rms.mean())))
            r[label]=section
        result[stream]=r
    return result


def main():
    arg=argparse.ArgumentParser(description=__doc__)
    arg.add_argument('--model-file',type=Path)
    arg.add_argument('--checkpoint',type=Path)
    arg.add_argument('--name',required=True)
    cfg=arg.parse_args()
    assert bool(cfg.model_file)==bool(cfg.checkpoint)
    with np.load(HERE/'reference.npz') as table:
        ref=dict(table)
    pred,error=predict(ref,cfg.model_file,cfg.checkpoint) if cfg.model_file else (None,None)
    if pred is not None:
        np.savez_compressed(HERE/f'{cfg.name}_points.npz',**{k:v.astype('float32') for k,v in pred.items()})
    df=block_values(ref,pred)
    df.to_parquet(HERE/f'{cfg.name}_beats.parquet',index=False)
    results=summarize(df)
    (HERE/f'{cfg.name}_scores.json').write_text(json.dumps(dict(model_file=str(cfg.model_file),checkpoint=str(cfg.checkpoint),
         eight_endpoint_max_error=error,results=results),indent=2)+'\n')
    for name,x in results.items():
        print(name,x['all'])


if __name__=='__main__':
    main()
