"""Measure the motion-branch response to paired, equal-RMS perturbations of past observations."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
PROJECT = ROUND.parents[2]
OUT = PROJECT / 'outputs'
MODELPATH = 'experiments/autopilot_flight_dynamics/model.py'
PREP = PROJECT/'.worktrees/periodic-motion-20261125-protocol/experiments/periodic_motion/prepare_native.py'
ARMS = {'full': 'actual-phase-carrier', 'matched': 'matched-slow-basis'}
STREAMS = {'down_velocity': (0,2), 'pitch_rate': (1,1)}
SNAPS = [0,5,10,15,20,25]
N = 16


def module(path):
    spec=importlib.util.spec_from_file_location('periodic_intervention_module', path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_data():
    history = OUT/'history_slow_fast_20261119_cache'
    old=OUT/'autopilot_flight_dynamics_20261124/validation.npz'
    with np.load(old,allow_pickle=False) as src:
        data={k:src[k] for k in ('window','flight','hist','pooled','time','u0','mask')}
    meta=pd.read_parquet(history/'validation_windows.parquet').set_index('window_id', verify_integrity=True)
    raw=np.load(history/'validation_history.npy',mmap_mode='r')
    data['orig_ix']=meta.index.get_indexer(data['window'])
    assert (data['orig_ix']>=0).all()
    meta=meta.iloc[data['orig_ix']]
    np.testing.assert_array_equal(data['flight'],meta.flight_id)
    data['main']=meta.sampling_group.to_numpy()=='main_50hz'
    data['decision']=meta.decision_us.to_numpy()
    data['features']=np.load(OUT/'autopilot_periodic_motion_20261125/validation_features.npy',mmap_mode='r')
    return data,raw


def select_windows(data,raw,stream):
    phase_ok=(raw[:,:,21]>0).all(1)&(raw[:,-1,22]>0)
    if stream==0: state_ok=(raw[:,:,16]>0).all(1)&(raw[:,:,18]>0).all(1)
    else: state_ok=(raw[:,:,17]>0).all(1)
    usable=(phase_ok&state_ok)[data['orig_ix']]&(data['features'][:,-2+stream]>0)
    if stream==0: usable&=data['main']
    usable &= data['mask'][:,:,stream].any(1)
    order=np.argsort(data['decision'],kind='stable')
    chosen=[]
    for flight in np.unique(data['flight']):
        part=order[usable[order] & (data['flight'][order]==flight)]
        if len(part):
            chosen.extend(part[np.linspace(0,len(part)-1,min(len(part),N)).round().astype(int)])
    return np.asarray(chosen)


def interventions(data,raw,wi,stream,axis):
    histories=np.array(raw[data['orig_ix'][wi]],copy=True)
    base=histories[:,SNAPS].reshape(len(wi),-1)
    ls=module(PREP).harmonic_history
    features=ls(histories,data['main'][wi])
    np.testing.assert_array_equal(base,data['hist'][wi])
    np.testing.assert_array_equal(features,data['features'][wi])
    yidx=3*stream+axis
    shape={'phase':histories[:,:,13].astype(np.float64),'ramp':np.broadcast_to(np.linspace(-1,0,26),(len(wi),26))}
    ret={}
    for name,curve in shape.items():
        curve=curve-curve[:,-1:]
        rms=np.sqrt(np.mean(curve**2,axis=1,keepdims=True))
        assert np.all(rms>1e-6)
        delta=.05*curve/rms
        inputs=[]
        for sign in (-1,1):
            changed=histories.copy()
            changed[:,:,yidx]+=sign*delta
            actual=changed[:,:,yidx]-histories[:,:,yidx]
            np.testing.assert_allclose(np.sqrt(np.mean(actual**2,axis=1)),.05,atol=3e-7,rtol=0)
            np.testing.assert_array_equal(changed[:,-1],histories[:,-1])
            summary=ls(changed,data['main'][wi])
            np.testing.assert_array_equal(summary[:,-2:],features[:,-2:])
            inputs.append(np.column_stack([changed[:,SNAPS].reshape(len(wi),-1),summary]))
        ret[name]=inputs
    return ret


def source_and_weights(seed,arm):
    studies=json.loads((ROUND/'studies.json').read_text())
    folder=PROJECT/'research/experiments'/studies[str(seed)]
    state=json.loads((folder/'state.json').read_text())
    found=[v for v in state['experiments'].values() if v['hypothesis']['hypothesis_id']==ARMS[arm]]
    if len(found)!=1: raise FileNotFoundError(f'missing registered {seed} {arm}')
    item=found[0]
    done=[a for a in item['attempts'] if a.get('directory') and (folder/a['directory']/f'seed-{seed}/output/step49152.pt').is_file()]
    if not done: raise FileNotFoundError(f'missing common checkpoint for {seed} {arm}')
    attempt=done[-1]
    ckpt=folder/attempt['directory']/f'seed-{seed}/output/step49152.pt'
    return item['id'],attempt['commit'],ckpt


def response(seed,arm,data,batches):
    eid,commit,path=source_and_weights(seed,arm)
    with tempfile.TemporaryDirectory() as td:
        src=Path(td)/'model.py'
        src.write_bytes(subprocess.check_output(['git','-C',str(PROJECT),'show',f'{commit}:{MODELPATH}']))
        model_class=module(src).FlightPredictor
    sd=torch.load(path,weights_only=True,map_location='cpu')
    model=model_class(sd['mean'],sd['std']).eval()
    model.load_state_dict(sd)
    rows=[]
    with torch.inference_mode():
        for stream_name,(wi,treatments) in batches.items():
            si,axis=STREAMS[stream_name]
            args=[torch.from_numpy(data[k][wi]) for k in ('pooled','u0','time')]
            eligible=data['mask'][wi,:,si]
            for shape,inputs in treatments.items():
                pair=[model.components(torch.from_numpy(v),*args) for v in inputs]
                delta=[.5*(b-a).numpy()[:,:,si,axis].astype(np.float64) for a,b in zip(*pair)]
                slow,fast,f0=delta
                f0=np.broadcast_to(f0,slow.shape)
                total=slow+fast-f0
                calc=.5*(model(torch.from_numpy(inputs[1]),*args)-model(torch.from_numpy(inputs[0]),*args)).numpy()[:,:,si,axis]
                np.testing.assert_allclose(total,calc,rtol=2e-5,atol=9e-7)
                energies={name:(np.sum((v**2)*eligible,axis=1)/eligible.sum(1))
                    for name,v in {'slow_delta':slow,'future_wave':fast,'origin_wave':f0,'wave_delta':fast-f0,
                                   'absolute_slow':slow-f0,'output':total}.items()}
                temp=pd.DataFrame(dict(seed=seed,arm=arm,case=shape,stream=stream_name,flight=data['flight'][wi],**energies))
                byflight=temp.groupby(['seed','arm','case','stream','flight'],as_index=False).mean()
                byflight['window_count']=temp.groupby('flight',sort=True).size().to_numpy()
                byflight['experiment']=eid
                byflight['commit']=commit
                rows.append(byflight)
    return pd.concat(rows,ignore_index=True)


def aggregate(df):
    names=['slow_delta','future_wave','origin_wave','wave_delta','absolute_slow','output']
    out={}
    for key in STREAMS:
        d=df[df.stream==key]
        flights=sorted(d.flight.unique())
        ix=np.random.default_rng(17).integers(0,len(flights),(10000,len(flights)))
        out[key]={'flights':len(flights),'windows':int(d[d.seed.eq(17)&d.arm.eq('full')&d.case.eq('phase')].window_count.sum()),'arms':{}}
        contrasts={}
        for arm in ARMS:
            out[key]['arms'][arm]={}
            seed_scores=[]
            seed_components=[]
            for seed in (17,29,43):
                energies={}
                for shape in ('phase','ramp'):
                    group=d[(d.arm==arm)&(d.seed==seed)&(d.case==shape)].set_index('flight').loc[flights,names]
                    arr=group.to_numpy()
                    means=np.vstack((arr.mean(0)[None],arr[ix].mean(1)))
                    energy={k:means[:,i] for i,k in enumerate(names)}
                    energy['share']=energy['future_wave']/(energy['future_wave']+energy['absolute_slow'])
                    energies[shape]=energy
                diffs={'share_difference':energies['phase']['share']-energies['ramp']['share'],
                       'phase_advantage_future_wave_rms':np.sqrt(energies['phase']['future_wave'])-np.sqrt(energies['ramp']['future_wave']),
                       'ramp_advantage_absolute_slow_rms':np.sqrt(energies['ramp']['absolute_slow'])-np.sqrt(energies['phase']['absolute_slow'])}
                component={shape:{**{k:float(np.sqrt(v[0])) for k,v in energy.items() if k!='share'},
                                  'future_wave_energy_share':float(energy['share'][0])} for shape,energy in energies.items()}
                seed_scores.append(diffs)
                seed_components.append(component)
                out[key]['arms'][arm][seed]={'RMS':component,'contrast':{k:float(v[0]) for k,v in diffs.items()}}
            score={k:np.mean([r[k] for r in seed_scores],axis=0) for k in seed_scores[0]}
            contrasts[arm]=score
            def with_ci(nums):
                return dict(mean=float(nums[0]),ci95=np.quantile(nums[1:],[.025,.975]).tolist())
            out[key]['arms'][arm]['average']={k:with_ci(v) for k,v in score.items()}
            out[key]['arms'][arm]['mean_component_responses']={
                shape:{k:float(np.mean([row[shape][k] for row in seed_components]))
                       for k in seed_components[0][shape]} for shape in ('phase','ramp')}
        out[key]['full_minus_matched']={k:with_ci(contrasts['full'][k]-contrasts['matched'][k]) for k in contrasts['full']}
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--seed',type=int,choices=(17,29,43))
    parser.add_argument('--arm',choices=tuple(ARMS))
    parser.add_argument('--summarize',action='store_true')
    opts=parser.parse_args()
    if opts.summarize:
        df=pd.concat([pd.read_csv(HERE/f'seed{s}_{a}.csv') for s in (17,29,43) for a in ARMS])
        (HERE/'results.json').write_text(json.dumps(aggregate(df),indent=2)+'\n')
        return
    if not opts.prepare and (opts.arm is None or opts.seed is None):parser.error('Supply both seed and arm')
    torch.set_num_threads(2)
    data,raw=read_data()
    batches={}
    coverage={}
    for name,(stream,axis) in STREAMS.items():
        wi=select_windows(data,raw,stream)
        batches[name]=wi,interventions(data,raw,wi,stream,axis)
        coverage[name]={'flights':len(np.unique(data['flight'][wi])),'windows':len(wi),
                        'window_ids':data['window'][wi].tolist()}
    if opts.prepare:
        (HERE/'window_selection.json').write_text(json.dumps(coverage,indent=2)+'\n')
        print({k:(v['flights'],v['windows']) for k,v in coverage.items()})
    else:
        frozen=json.loads((HERE/'window_selection.json').read_text())
        assert coverage==frozen
        response(opts.seed,opts.arm,data,batches).to_csv(HERE/f'seed{opts.seed}_{opts.arm}.csv',index=False)


if __name__=='__main__':
    main()
