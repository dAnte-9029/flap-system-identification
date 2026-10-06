"""Same frozen state gates, paired actions and causal marginal support audit."""
import numpy as np
import pandas as pd
from pyulog import ULog
import train_control_windows_v2 as base
from run_coupled_replay_v40 import OUT,DATA,PARENTS,V37,read
from coupled_replay_v40 import VARIANTS
from audit_controller_support_v26 import verify_parent
from evaluate_actuator_v30 import errors


def main():
    assert read(OUT/'status.json')['status']=='replay_complete_evaluation_pending'
    protocol=read(OUT/'protocol.json');w=pd.read_csv(DATA/'windows.csv')
    with np.load(DATA/'windows.npz') as z:
        truth={k[6:]:z[k].astype(float) for k in z.files if k.startswith('label_')}
        logged=z['input_future_command'].astype(float)
    states=[];actions=[];support=[];teachers=[];micro=[];seen={};first={}
    for fold in range(5):
        ids=read(DATA/f'fold{fold}/manifest.json')['query_indices'];dest=OUT/f'fold{fold}'
        assert read(dest/'status.json')['status']=='complete'
        target={k:v[ids] for k,v in truth.items()}
        for name,_,_ in VARIANTS:
            with np.load(dest/f'{name}_teacher.npz') as z:teacher={k:z[k] for k in z.files}
            np.testing.assert_array_equal(teacher['array_index'],ids)
            for j in range(4):
                err=teacher['commands'][:,:,j]-logged[ids,:,j]
                for i,mse in zip(ids,(err**2).mean(1)):teachers.append(dict(fold=fold,variant=name,array_index=i,axis=j,mse=mse))
            for stage in [30,35]:
                for seed in [17,23,42]:
                    with np.load(dest/f'{name}_v{stage}_seed{seed}.npz') as z:a={k:z[k] for k in z.files}
                    np.testing.assert_array_equal(a['array_index'],ids)
                    assert all(np.isfinite(v).all() for v in a.values())
                    assert (a['commands'][:,:,0]>=0).all() and (a['commands'][:,:,0]<=1).all()
                    assert (np.abs(a['commands'][:,:,1:])<=1).all()
                    key=(fold,name)
                    if key not in first:first[key]=a['commands'][:,0].copy()
                    np.testing.assert_array_equal(a['commands'][:,0],first[key])
                    seen.setdefault((stage,seed,name),[]).extend(ids)
                    common=dict(fold=fold,variant=name,stage=stage,seed=seed)
                    pred={k:a[k].astype(float) for k in truth}
                    for horizon in [25,50]:
                        wave,end=errors(pred,target,horizon)
                        for kind,values in [('wave',wave),('endpoint',end)]:
                            for metric,values in values.items():
                                for i,value in zip(ids,values):states.append(dict(**common,array_index=i,horizon=horizon,kind=kind,metric=metric,mse=value))
                    for comparison,t in [('log',logged[ids]),('matched_teacher',teacher['commands'])]:
                        for j in range(4):
                            squared=(a['commands'][:,:,j]-t[:,:,j])**2
                            for i,value in zip(ids,squared.mean(1)):actions.append(dict(**common,array_index=i,comparison=comparison,axis=j,mse=value))
                    # Deviation of internal400Hz requests from the command actually used by50Hz model.
                    for j in range(4):
                        squared=(a['micro_commands'][:,:,:,j]-a['commands'][:,:,None,j])**2
                        for i,value in zip(ids,squared.mean((1,2))):micro.append(dict(**common,array_index=i,axis=j,mse=value))
                    for index,i in enumerate(ids):
                        support.append(dict(**common,array_index=i,
                            command_outside_fraction=float(a['command_outside'][index].any(-1).mean()),
                            state_outside_fraction=float(a['state_outside'][index].any(-1).mean()),
                            surface_clip_fraction=float(a['surface_clipped'][index].any(-1).mean()),
                            any_command_outside=bool(a['command_outside'][index].any()),
                            any_state_outside=bool(a['state_outside'][index].any()),
                            any_surface_clip=bool(a['surface_clipped'][index].any())))
        # Existing logged-command conditional prediction remains a distinct comparator.
        for spec in [r for r in protocol['models'] if r['fold']==fold]:
            root=(base.ROOT/spec['checkpoint']).parent
            with np.load(root/'oof_predictions.npz') as z:
                np.testing.assert_array_equal(z['array_index'],ids);pred={k:z[k].astype(float) for k in truth}
            for horizon in [25,50]:
                wave,end=errors(pred,target,horizon)
                for kind,values in [('wave',wave),('endpoint',end)]:
                    for metric,values in values.items():
                        for i,value in zip(ids,values):states.append(dict(fold=fold,variant='logged_command_conditioned',stage=spec['stage'],seed=spec['seed'],array_index=i,horizon=horizon,kind=kind,metric=metric,mse=value))
    for values in seen.values():np.testing.assert_array_equal(sorted(values),np.arange(407))
    sm=pd.DataFrame(states);ac=pd.DataFrame(actions);sp=pd.DataFrame(support);tc=pd.DataFrame(teachers);mi=pd.DataFrame(micro)
    for name,frame in [('state_metrics',sm),('action_metrics',ac),('support',sp),('teacher_metrics',tc),('micro_command_metrics',mi)]:
        frame.to_csv(OUT/f'{name}_per_window.csv',index=False)
    keys=['variant','stage','seed','horizon','kind','metric']
    sf=sm.groupby(keys+['fold']).mse.mean().pow(.5).reset_index(name='rmse');sf.to_csv(OUT/'state_metrics_per_flight.csv',index=False)
    summary=sf.groupby(keys).rmse.mean().reset_index();summary.to_csv(OUT/'state_summary.csv',index=False)
    af=ac.groupby(['variant','stage','seed','comparison','axis','fold']).mse.mean().pow(.5).reset_index(name='rmse')
    af.to_csv(OUT/'action_metrics_per_flight.csv',index=False)
    action_summary=af.groupby(['variant','stage','seed','comparison','axis']).rmse.mean().reset_index();action_summary.to_csv(OUT/'action_summary.csv',index=False)
    tf=tc.groupby(['variant','axis','fold']).mse.mean().pow(.5)
    tf.groupby(['variant','axis']).mean().rename('rmse').to_csv(OUT/'teacher_summary.csv')
    mif=mi.groupby(['variant','stage','seed','axis','fold']).mse.mean().pow(.5)
    mif.groupby(['variant','stage','axis']).mean().rename('rmse').to_csv(OUT/'micro_command_summary.csv')
    measures=['command_outside_fraction','state_outside_fraction','surface_clip_fraction','any_command_outside','any_state_outside','any_surface_clip']
    sp.groupby(['variant','stage','seed','fold'])[measures].mean().groupby(['variant','stage'])[measures].mean().to_csv(OUT/'support_summary.csv')
    # Also retain actual counts, rather than only equal-flight averages.
    sp.groupby(['variant','stage'])[measures[3:]].sum().to_csv(OUT/'support_window_counts.csv')
    mode=sm.merge(w[['array_index','mode']],on='array_index',validate='many_to_one')
    mode.groupby(keys+['mode','fold']).mse.mean().pow(.5).rename('rmse').to_csv(OUT/'state_by_mode_flight.csv')
    active=sm[(sm.horizon==25)&(sm.kind=='endpoint')&sm.array_index.isin(w[w.active_reference].array_index)]
    absolute=active.groupby(['variant','stage','seed','metric','fold']).mse.mean().pow(.5).groupby(['variant','stage','seed','metric']).mean().reset_index(name='rmse')
    limits={'p':.15,'q':.15,'attitude':np.deg2rad(3),'velocity':.5}
    absolute['limit']=absolute.metric.map(limits);absolute=absolute[absolute.limit.notna()].copy();absolute['passed']=absolute.rmse<=absolute.limit
    absolute.to_csv(OUT/'absolute_gates.csv',index=False)
    pair=summary.pivot(index=['variant','seed','horizon','kind','metric'],columns='stage',values='rmse').reset_index()
    pair['improvement_percent']=100*(1-pair[35]/pair[30]);pair.to_csv(OUT/'paired_state_comparison.csv',index=False)
    ap=action_summary[action_summary.comparison=='log'].pivot(index=['variant','seed','axis'],columns='stage',values='rmse').reset_index()
    ap['passed']=ap[35]<=1.05*ap[30];ap.to_csv(OUT/'action_protections.csv',index=False)
    gates=[]
    for name,_,_ in VARIANTS:
        p=pair[pair.variant==name];q=p[(p.horizon==50)&(p.kind=='wave')&(p.metric=='q')]
        qf=sf[(sf.variant==name)&(sf.horizon==50)&(sf.kind=='wave')&(sf.metric=='q')].groupby(['fold','stage']).rmse.mean().unstack()
        protect=p[((p.horizon==50)&(p.kind=='wave')&p.metric.isin(['p','r']))|((p.kind=='endpoint')&p.metric.isin(['velocity','attitude']))]
        assert len(protect)==18
        imp=float(100*(1-q[35].mean()/q[30].mean()));positive=int((qf[35]<qf[30]).sum())
        protections=int((protect.improvement_percent>=-5).sum());ag=ap[ap.variant==name]
        abs35=absolute[(absolute.variant==name)&(absolute.stage==35)]
        gates.append(dict(variant=name,q_improvement_percent=imp,all_seeds_q_improve=bool((q[35]<q[30]).all()),
            improving_flights=positive,protections_passed=protections,protections_total=18,
            relative_gate=bool(imp>=5 and (q[35]<q[30]).all() and positive>=4 and protections==18),
            action_protections_passed=int(ag.passed.sum()),action_protections_total=12,
            absolute_passed=int(abs35.passed.sum()),absolute_total=12,control_ready=False))
    base.write(OUT/'gates.json',gates)
    # Independently reconstruct every condition tape from raw reference topics.
    verified=0
    entries=read(V37/'dataset_manifest.json')['sources']
    for entry in read(OUT/'conditions_manifest.json')['flights']:
        src=next(s for s in entries if s['log_id']==entry['log_id'])
        assert base.sha(src['source']['path'])==src['source']['sha256']
        u=ULog(src['source']['path'],message_name_filter_list=['vehicle_attitude_setpoint','manual_control_setpoint'])
        ds={d.name:d.data for d in u.data_list if d.multi_id==0}
        root=OUT/entry['path'];meta=read(root/'provenance.json')
        with np.load(root/'conditions.npz') as z:
            for j,item in enumerate(meta):
                t=item['origin_us']+np.arange(400)*2500
                values=[]
                for topic,index_key in [('vehicle_attitude_setpoint','attitude_reference_indices'),('manual_control_setpoint','manual_reference_indices')]:
                    d=ds[topic];ix=np.asarray(item[index_key]);available=np.maximum(d['timestamp'],d.get('timestamp_sample',d['timestamp']))
                    assert (available[ix]<=t).all()
                    for index,query in zip(ix,t):
                        if index+1<len(available):assert available[index+1]>query
                    if topic=='vehicle_attitude_setpoint':
                        q=np.column_stack([d[f'q_d[{k}]'][ix] for k in range(4)]).astype(float)
                        ww,xx,yy,zz=q.T
                        values.extend([np.arctan2(2*(ww*xx+yy*zz),1-2*(xx*xx+yy*yy)),np.arcsin(np.clip(2*(ww*yy-zz*xx),-1,1)),d['thrust_body[0]'][ix]])
                    else:values.append(d['yaw'][ix])
                np.testing.assert_array_equal(np.column_stack(values),z['tape'][j]);verified+=len(t)
                assert item['air_publication_us']<=item['origin_us']
    assert verified==407*400
    for path,digest in protocol['code'].items():assert base.sha(base.ROOT/path)==digest
    for parent in PARENTS:verify_parent(parent)
    base.write(OUT/'verification.json',dict(status='passed',all407_per_stage_seed_variant=True,
        coupled_window_runs=407*6*4,matched_teacher_window_runs=407*4,
        finite_trajectories=407*6*4,first_actions_identical_across_models=True,
        condition_points_independently_rebuilt=verified,
        measured_future_state_only_in_named_teacher_and_scoring=True,
        validation_accessed=False,sealed_test_accessed=False,reserved_accessed=False))
    base.write(OUT/'status.json',dict(status='evaluated_report_pending'))
    print(gates,flush=True)


if __name__=='__main__':main()
