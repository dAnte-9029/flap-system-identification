"""Apply frozen v42 gates to every conditional/coupled v43 OOF trajectory."""
import argparse
import numpy as np
import pandas as pd
import train_control_windows_v2 as base
from train_attitude_v42 import OUT, PREP, DATA, read
from evaluate_actuator_v30 import errors as state_errors
from audit_attitude_kinematics_v42 import errors as angle_errors
from coupled_replay_v40 import VARIANTS

V35=OUT.parent/'control_windows_v35'
V40=OUT.parent/'control_windows_v40'
SEEDS=[17,23,42]


def metrics(pred,true,h):
    wave,end=state_errors(pred,true,h)
    sq=angle_errors(pred['quaternion'],true['quaternion'])
    for j,name in enumerate(['roll','pitch','tilt']):
        wave[name]=sq[:,:h,j].mean(1);end[name]=sq[:,h-1,j]
    return wave,end


def protections(pair):
    return pair[((pair.kind=='wave')&(pair.horizon==50)&pair.metric.isin(['p','q','r','roll','pitch']))|
                ((pair.kind=='endpoint')&pair.horizon.isin([25,50])&pair.metric.isin(['velocity','attitude']))].copy()


def response_gate(flight,metric):
    f=flight[(flight.horizon==50)&(flight.kind=='wave')&(flight.metric==metric)]
    s=f.groupby(['seed','arm']).rmse.mean().unstack()
    byflight=f.groupby(['fold','arm']).rmse.mean().unstack()
    improvement=float(100*(1-s.candidate.mean()/s.baseline.mean()))
    seeds=int((s.candidate<s.baseline).sum());flights=int((byflight.candidate<byflight.baseline).sum())
    return dict(improvement_percent=improvement,improving_seeds=seeds,improving_flights=flights,
                passed=bool(improvement>=5 and seeds==3 and flights>=4))


def score(mode):
    assert read(OUT/'launch/status.json')['status']=='training_complete_evaluation_pending'
    dest=OUT/mode;dest.mkdir(exist_ok=True)
    windows=pd.read_csv(DATA/'windows.csv');rows=[];actions=[];support=[];micro=[];teachers=[]
    with np.load(DATA/'windows.npz') as z:
        truth={k[6:]:z[k].astype(float) for k in z.files if k.startswith('label_')}
        logged=z['input_future_command'].astype(float)
    variants=['logged_command_conditioned'] if mode=='conditional' else [v[0] for v in VARIANTS]
    for fold in range(5):
        ids=read(DATA/f'fold{fold}/manifest.json')['query_indices'];true={k:v[ids] for k,v in truth.items()}
        for variant in variants:
            if mode=='coupled':
                with np.load(V40/f'fold{fold}/{variant}_teacher.npz') as z:teacher=z['commands']
                for j in range(4):
                    for i,m in zip(ids,((teacher[:,:,j]-logged[ids,:,j])**2).mean(1)):
                        teachers.append(dict(variant=variant,fold=fold,array_index=i,axis=j,mse=m))
            for seed in SEEDS:
                for arm in ['baseline','candidate']:
                    if mode=='conditional':
                        root=V35/f'fold{fold}/joint_increment/seed{seed}' if arm=='baseline' else OUT/f'fold{fold}/seed{seed}'
                        path=root/'oof_predictions.npz'
                    else:
                        path=V40/f'fold{fold}/{variant}_v35_seed{seed}.npz' if arm=='baseline' else dest/f'fold{fold}/{variant}_seed{seed}.npz'
                    with np.load(path) as z:a={k:z[k] for k in z.files}
                    np.testing.assert_array_equal(a['array_index'],ids)
                    assert all(np.isfinite(v).all() for v in a.values())
                    pred={k:a[k].astype(float) for k in truth}
                    common=dict(variant=variant,fold=fold,seed=seed,arm=arm)
                    for h in [25,50]:
                        wave,end=metrics(pred,true,h)
                        for kind,values in [('wave',wave),('endpoint',end)]:
                            for metric,ms in values.items():
                                rows.extend(dict(**common,array_index=i,horizon=h,kind=kind,metric=metric,mse=m) for i,m in zip(ids,ms))
                    if mode=='coupled':
                        for comparison,target in [('log',logged[ids]),('matched_teacher',teacher)]:
                            for j in range(4):
                                ms=((a['commands'][:,:,j]-target[:,:,j])**2).mean(1)
                                actions.extend(dict(**common,array_index=i,comparison=comparison,axis=j,mse=m) for i,m in zip(ids,ms))
                        for j in range(4):
                            ms=((a['micro_commands'][:,:,:,j]-a['commands'][:,:,None,j])**2).mean((1,2))
                            micro.extend(dict(**common,array_index=i,axis=j,mse=m) for i,m in zip(ids,ms))
                        for n,i in enumerate(ids):
                            support.append(dict(**common,array_index=i,
                                command_outside_fraction=float(a['command_outside'][n].any(-1).mean()),
                                state_outside_fraction=float(a['state_outside'][n].any(-1).mean()),
                                surface_clip_fraction=float(a['surface_clipped'][n].any(-1).mean())))
    frame=pd.DataFrame(rows);frame.to_csv(dest/'state_per_window.csv',index=False)
    keys=['variant','seed','arm','horizon','kind','metric']
    flight=frame.groupby(keys+['fold']).mse.mean().pow(.5).reset_index(name='rmse');flight.to_csv(dest/'state_per_flight.csv',index=False)
    for _,g in frame[(frame.horizon==50)&(frame.kind=='wave')&(frame.metric=='tilt')].groupby(['variant','seed','arm']):
        np.testing.assert_array_equal(np.sort(g.array_index),np.arange(407))
    summary=flight.groupby(keys).rmse.mean().reset_index()
    pair=summary.pivot(index=['variant','seed','horizon','kind','metric'],columns='arm',values='rmse').reset_index()
    pair['improvement_percent']=100*(1-pair.candidate/pair.baseline);pair.to_csv(dest/'comparison.csv',index=False)
    active=frame[(frame.horizon==25)&(frame.kind=='endpoint')&(frame.arm=='candidate')&frame.array_index.isin(windows[windows.active_reference].array_index)]
    absolute=active.groupby(['variant','seed','metric','fold']).mse.mean().pow(.5).groupby(['variant','seed','metric']).mean().reset_index(name='rmse')
    absolute['limit']=absolute.metric.map(dict(p=.15,q=.15,attitude=np.deg2rad(3),velocity=.5))
    absolute=absolute[absolute.limit.notna()].copy();absolute['passed']=absolute.rmse<=absolute.limit
    absolute.to_csv(dest/'absolute_gates.csv',index=False)
    strat=frame.merge(windows[['array_index','mode']],on='array_index',validate='many_to_one')
    strat.groupby(keys+['mode','fold']).mse.mean().pow(.5).rename('rmse').to_csv(dest/'state_by_mode.csv')
    if mode=='coupled':
        af=pd.DataFrame(actions);af.to_csv(dest/'action_per_window.csv',index=False)
        af=af.groupby(['variant','seed','arm','comparison','axis','fold']).mse.mean().pow(.5).reset_index(name='rmse')
        af.to_csv(dest/'action_per_flight.csv',index=False)
        af.groupby(['variant','seed','arm','comparison','axis']).rmse.mean().to_csv(dest/'action_summary.csv')
        sp=pd.DataFrame(support);sp.to_csv(dest/'support_per_window.csv',index=False)
        cols=['command_outside_fraction','state_outside_fraction','surface_clip_fraction']
        sp.groupby(['variant','arm','seed','fold'])[cols].mean().groupby(['variant','arm'])[cols].mean().to_csv(dest/'support_summary.csv')
        pd.DataFrame(micro).to_csv(dest/'micro_per_window.csv',index=False)
        pd.DataFrame(teachers).groupby(['variant','axis','fold']).mse.mean().pow(.5).groupby(['variant','axis']).mean().to_csv(dest/'teacher_summary.csv')
    gates=[];protected=[];actionprotect=[]
    for variant in variants:
        f=flight[flight.variant==variant];pr=protections(pair[pair.variant==variant]);assert len(pr)==27
        pr['passed']=pr.improvement_percent>=-5;protected.append(pr)
        ab=absolute[absolute.variant==variant];assert len(ab)==12
        tilt=response_gate(f,'tilt');legacy=response_gate(f,'q')
        gate=dict(variant=variant,tilt=tilt,legacy_q=legacy,protections_passed=int(pr.passed.sum()),protections_total=27,
                  tilt_with_protections=bool(tilt['passed'] and pr.passed.all()),absolute_passed=int(ab.passed.sum()),absolute_total=12,control_ready=False)
        if mode=='coupled':
            a=af[(af.variant==variant)&(af.comparison=='log')]
            ap=a.groupby(['seed','axis','arm']).rmse.mean().unstack().reset_index()
            ap['passed']=ap.candidate<=1.05*ap.baseline;ap['variant']=variant;assert len(ap)==12;actionprotect.append(ap)
            surfaces=a[a.axis.isin([1,2])].groupby(['seed','fold','arm']).rmse.mean().reset_index()
            s=surfaces.groupby(['seed','arm']).rmse.mean().unstack();f=surfaces.groupby(['fold','arm']).rmse.mean().unstack()
            imp=float(100*(1-s.candidate.mean()/s.baseline.mean()));ns=int((s.candidate<s.baseline).sum());nf=int((f.candidate<f.baseline).sum())
            gate.update(surface_improvement_percent=imp,surface_improving_seeds=ns,surface_improving_flights=nf,
                        surface_target_passed=bool(imp>=5 and ns==3 and nf>=4),action_protections_passed=int(ap.passed.sum()),action_protections_total=12)
            gate['coupled_relative_gate']=bool(gate['surface_target_passed'] and ap.passed.all() and pr.passed.all())
        gates.append(gate)
    pd.concat(protected).to_csv(dest/'state_protections.csv',index=False)
    if actionprotect:pd.concat(actionprotect).to_csv(dest/'action_protections.csv',index=False)
    base.write(dest/'gates.json',gates);print(gates,flush=True)
    return gates


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['conditional','coupled'],required=True);score(p.parse_args().mode)
