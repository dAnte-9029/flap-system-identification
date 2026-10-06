"""Replay saved candidates and independently recompute all v43 OOF scores."""
import argparse
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from train_attitude_v42 import OUT,PREP,DATA,read
from evaluate_attitude_v43 import V35,V40,SEEDS
from actuator_training_v30 import load,model_from
from reference_groupcv_v8 import predict
from prepare_actuator_inputs_v29 import fingerprint
from system_identification.models.actuator_command_gru import ActuatorCommandGRU
from system_identification.models.reference_gru_unit_gravity import unit_state_features
from audit_controller_support_v26 import verify_parent
from coupled_replay_v40 import VARIANTS


def independent_squares(a,t):
    rates=(a['rates_b'].astype(float)-t['rates_b'])**2
    result={k:rates[:,:,j] for j,k in enumerate('pqr')}
    result['velocity']=((a['velocity_n'].astype(float)-t['velocity_n'])**2).mean(-1)
    p=a['quaternion'].astype(float);q=t['quaternion']
    result['attitude']=(2*np.arccos(np.minimum(1,np.abs((p*q).sum(-1)))))**2
    def angles_gravity(v):
        w,x,y,z=np.moveaxis(v/np.linalg.norm(v,axis=-1,keepdims=True),-1,0)
        angles=np.stack([np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)),np.arcsin(np.clip(2*(w*y-z*x),-1,1))],-1)
        g=np.stack([2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],-1)
        return angles,g
    ap,gp=angles_gravity(p);aq,gq=angles_gravity(q)
    d=ap-aq;d=np.arctan2(np.sin(d),np.cos(d))
    result['roll']=d[:,:,0]**2;result['pitch']=d[:,:,1]**2
    result['tilt']=np.arctan2(np.sqrt((np.cross(gp,gq)**2).sum(-1)),(gp*gq).sum(-1))**2
    return result


def verify_predictions():
    x,_=load();runs=read(PREP/'training_plan.json')['runs'];checks=[];fitrows=[]
    with np.load(DATA/'windows.npz') as z:truth={k[6:]:z[k].astype(float) for k in z.files if k.startswith('label_')}
    for run in runs:
        fold,seed=run['fold'],run['seed'];root=OUT/f'fold{fold}/seed{seed}';info=read(DATA/f'fold{fold}/manifest.json')
        status=read(root/'status.json');assert status['status']=='complete' and status['step']==1600
        assert base.sha(root/'final.pt')==status['checkpoint_sha256']
        rp=read(root/'run_protocol.json')
        for k,v in run.items():assert rp[k]==v,k
        assert rp['executed_weight']==run['tilt_weight']
        assert rp['preparation_protocol_sha256']==base.sha(PREP/'protocol.json')
        assert base.sha(base.ROOT/run['sample_plan'])==run['sample_plan_sha256']
        model=model_from(root/'final.pt');ck=torch.load(root/'final.pt',weights_only=False,map_location='cpu')
        assert (ck['fold'],ck['seed'],ck['steps'])==(fold,seed,1600)
        assert ck['tilt_weight']==run['tilt_weight'] and ck['local_weight']==1
        assert ck['protocol_sha256']==base.sha(PREP/'protocol.json')
        with np.load(DATA/f'fold{fold}/normalization.npz') as z:
            stats={k:z[k] for k in z.files}
            for k,v in stats.items():np.testing.assert_array_equal(getattr(model,k).numpy(),v.astype(np.float32))
        torch.manual_seed(seed);init=ActuatorCommandGRU(stats);assert fingerprint(init)==run['initial_parameter_sha256']
        assert sum(p.numel() for p in model.parameters())==40070
        for cohort,key in [('fit','fit_indices'),('oof','query_indices')]:
            ids=info[key];p=predict(model,x,ids)
            with np.load(root/f'{cohort}_predictions.npz') as z:
                np.testing.assert_array_equal(z['array_index'],ids)
                for k,v in p.items():np.testing.assert_array_equal(v,z[k])
            if cohort=='fit':
                target={k:v[ids] for k,v in truth.items()}
                for arm,directory in [('candidate',root),('baseline',V35/f'fold{fold}/joint_increment/seed{seed}')]:
                    with np.load(directory/'fit_predictions.npz') as z:
                        np.testing.assert_array_equal(z['array_index'],ids);sq=independent_squares(z,target)
                    for logfold in range(5):
                        query=set(read(DATA/f'fold{logfold}/manifest.json')['query_indices']);mask=np.asarray([i in query for i in ids])
                        if not mask.any():continue
                        for metric,v in sq.items():fitrows.append(dict(fold=fold,seed=seed,arm=arm,source_fold=logfold,metric=metric,rmse=float(np.sqrt(v[mask].mean()))))
        ids=info['query_indices'][:2];other=dict(x);other['future_command']=x['future_command'].clone();other['future_command'][:,10:]+=.1
        a=predict(model,x,ids);b=predict(model,other,ids)
        for k in a:np.testing.assert_array_equal(a[k][:,:10],b[k][:,:10])
        assert any(np.any(a[k][:,10:]!=b[k][:,10:]) for k in a)
        checks.append(dict(fold=fold,seed=seed,predictions_exact=True,buffers_exact=True,initialization_exact=True,causal_prefix=True))
    pd.DataFrame(fitrows).to_csv(OUT/'conditional/fit_wave_metrics.csv',index=False)
    return checks


def main(mode):
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    for parent in [PREP,DATA,V35,V40]:verify_parent(parent)
    d=OUT/mode
    checks=verify_predictions() if mode=='conditional' else []
    metrics=pd.read_csv(d/'state_per_window.csv').set_index(['variant','fold','seed','arm','horizon','kind','metric','array_index']).sort_index()
    with np.load(DATA/'windows.npz') as z:raw={k:z[k] for k in z.files}
    windows=pd.read_csv(DATA/'windows.csv');records=[];active_records=[];count=0;actioncount=0;supportcount=0;actionrecords=[]
    if mode=='coupled':ac=pd.read_csv(d/'action_per_window.csv').set_index(['variant','fold','seed','arm','comparison','axis','array_index']).sort_index()
    variants=['logged_command_conditioned'] if mode=='conditional' else [v[0] for v in VARIANTS]
    for fold in range(5):
        info=read(DATA/f'fold{fold}/manifest.json');ids=info['query_indices'];fit=info['fit_indices']
        truth={k[6:]:v[ids].astype(float) for k,v in raw.items() if k.startswith('label_')}
        active=windows.iloc[ids].active_reference.to_numpy(bool)
        bounds=read(V40/f'fold{fold}/support.json')
        for key,value in [('command_min',raw['input_future_command'][fit].min((0,1))),('command_max',raw['input_future_command'][fit].max((0,1))),
                          ('state_min',raw['input_history_state'][fit].min((0,1))),('state_max',raw['input_history_state'][fit].max((0,1)))]:
            np.testing.assert_array_equal(bounds[key],value)
        for variant in variants:
            for seed in SEEDS:
                for arm in ['baseline','candidate']:
                    if mode=='conditional':
                        root=V35/f'fold{fold}/joint_increment/seed{seed}' if arm=='baseline' else OUT/f'fold{fold}/seed{seed}'
                        path=root/'oof_predictions.npz'
                    else:path=V40/f'fold{fold}/{variant}_v35_seed{seed}.npz' if arm=='baseline' else d/f'fold{fold}/{variant}_seed{seed}.npz'
                    with np.load(path) as z:a={k:z[k] for k in z.files}
                    np.testing.assert_array_equal(a['array_index'],ids);sq=independent_squares(a,truth)
                    for metric,v in sq.items():
                        for h in [25,50]:
                            for kind,values in [('wave',v[:,:h].mean(1)),('endpoint',v[:,h-1])]:
                                actual=metrics.loc[(variant,fold,seed,arm,h,kind,metric)].loc[ids,'mse'].to_numpy()
                                np.testing.assert_allclose(actual,values,rtol=1e-9,atol=1e-12);count+=len(ids)
                                records.append(dict(variant=variant,fold=fold,seed=seed,arm=arm,horizon=h,kind=kind,metric=metric,rmse=float(np.sqrt(values.mean()))))
                        if metric in ['p','q','attitude','velocity'] and arm=='candidate':
                            active_records.append(dict(variant=variant,seed=seed,metric=metric,fold=fold,rmse=float(np.sqrt(v[active,24].mean()))))
                    if mode=='coupled':
                        with np.load(V40/f'fold{fold}/{variant}_teacher.npz') as z:teacher=z['commands']
                        for comparison,target in [('log',raw['input_future_command'][ids]),('matched_teacher',teacher)]:
                            for axis in range(4):
                                values=((a['commands'][:,:,axis]-target[:,:,axis])**2).mean(1)
                                actual=ac.loc[(variant,fold,seed,arm,comparison,axis)].loc[ids,'mse'].to_numpy()
                                np.testing.assert_allclose(actual,values,rtol=1e-10,atol=1e-12);actioncount+=len(ids)
                                if comparison=='log':actionrecords.append(dict(variant=variant,fold=fold,seed=seed,arm=arm,axis=axis,rmse=float(np.sqrt(values.mean()))))
                        np.testing.assert_array_equal(a['commands'],a['micro_commands'][:,:,0])
                        np.testing.assert_array_equal(a['command_outside'],(a['commands']<bounds['command_min'])|(a['commands']>bounds['command_max']))
                        q=torch.tensor(np.concatenate([raw['input_initial_quaternion'][ids,None],a['quaternion'][:,:49]],1));q[:,0]/=q[:,0].norm(dim=-1,keepdim=True).clamp_min(1e-8)
                        v=torch.tensor(np.concatenate([raw['input_initial_velocity_n'][ids,None],a['velocity_n'][:,:49]],1))
                        r=torch.tensor(np.concatenate([raw['input_initial_rates_b'][ids,None],a['rates_b'][:,:49]],1))
                        hist=torch.tensor(raw['input_history_state'][ids]);freq=hist[:,-1,11].clone();phase=torch.zeros_like(freq);anchor=phase.clone()
                        for k in range(50):
                            features=torch.cat([unit_state_features(v[:,k],q[:,k],r[:,k],phase,anchor,freq),hist[:,-1,12:14]],-1).numpy()
                            np.testing.assert_array_equal(a['state_outside'][:,k],(features<bounds['state_min'])|(features>bounds['state_max']))
                            phase=phase+2*torch.pi*freq*torch.full_like(freq,.02);supportcount+=len(ids)
    f=pd.DataFrame(records);f.to_csv(d/'independent_flight_metrics.csv',index=False)
    pair=f.groupby(['variant','seed','horizon','kind','metric','arm']).rmse.mean().unstack()
    saved=pd.read_csv(d/'comparison.csv').set_index(['variant','seed','horizon','kind','metric'])
    np.testing.assert_allclose(pair[['baseline','candidate']],saved.loc[pair.index,['baseline','candidate']],rtol=1e-10,atol=1e-12)
    ag=pd.DataFrame(active_records).groupby(['variant','seed','metric']).rmse.mean()
    saved=pd.read_csv(d/'absolute_gates.csv').set_index(['variant','seed','metric'])
    np.testing.assert_allclose(ag,saved.loc[ag.index,'rmse'],rtol=1e-10,atol=1e-12)
    limits=dict(p=.15,q=.15,attitude=np.deg2rad(3),velocity=.5)
    for gate in read(d/'gates.json'):
        variant=gate['variant'];p=pair.loc[variant].reset_index()
        mask=((p.kind=='wave')&(p.horizon==50)&p.metric.isin(['p','q','r','roll','pitch']))|((p.kind=='endpoint')&p.metric.isin(['velocity','attitude']))
        pr=p[mask];assert len(pr)==27
        n=int((pr.candidate<=1.05*pr.baseline).sum());assert n==gate['protections_passed']
        absn=sum(value<=limits[metric] for (seed,metric),value in ag.loc[variant].items());assert absn==gate['absolute_passed']
        for metric,key in [('tilt','tilt'),('q','legacy_q')]:
            s=p[(p.horizon==50)&(p.kind=='wave')&(p.metric==metric)]
            by=f[(f.variant==variant)&(f.horizon==50)&(f.kind=='wave')&(f.metric==metric)].groupby(['fold','arm']).rmse.mean().unstack()
            imp=float(100*(1-s.candidate.mean()/s.baseline.mean()));ns=int((s.candidate<s.baseline).sum());nf=int((by.candidate<by.baseline).sum())
            g=gate[key];np.testing.assert_allclose(imp,g['improvement_percent'],atol=1e-8);assert (ns,nf)==(g['improving_seeds'],g['improving_flights'])
            assert g['passed']==(imp>=5 and ns==3 and nf>=4)
        assert gate['tilt_with_protections']==(gate['tilt']['passed'] and n==27)
        if mode=='coupled':
            a=pd.DataFrame(actionrecords);a=a[a.variant==variant]
            ap=a.groupby(['seed','axis','arm']).rmse.mean().unstack();nap=int((ap.candidate<=1.05*ap.baseline).sum())
            assert nap==gate['action_protections_passed']
            s=a[a.axis.isin([1,2])].groupby(['seed','arm']).rmse.mean().unstack();by=a[a.axis.isin([1,2])].groupby(['fold','arm']).rmse.mean().unstack()
            imp=float(100*(1-s.candidate.mean()/s.baseline.mean()));ns=int((s.candidate<s.baseline).sum());nf=int((by.candidate<by.baseline).sum())
            np.testing.assert_allclose(imp,gate['surface_improvement_percent'],atol=1e-8)
            assert (ns,nf)==(gate['surface_improving_seeds'],gate['surface_improving_flights'])
            assert gate['surface_target_passed']==(imp>=5 and ns==3 and nf>=4)
            assert gate['coupled_relative_gate']==(gate['surface_target_passed'] and n==27 and nap==12)
    for parent in [PREP,DATA,V35,V40]:verify_parent(parent)
    base.write(d/'verification.json',dict(status='passed',checkpoint_checks=checks,state_metric_values_recomputed=count,
        action_metric_values_recomputed=actioncount,causal_support_points_recomputed=supportcount,
        all_gates_independently_verified=True,all407_retained=True,validation_accessed=False,sealed_test_accessed=False,reserved_accessed=False))
    print(f'{mode}: independent verification passed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['conditional','coupled'],required=True);main(parser.parse_args().mode)
