"""All-window action oracle metrics and exact two-factor decomposition."""
import numpy as np
import pandas as pd
import train_control_windows_v2 as base
from run_action_propagation_v41 import OUT,PARENT,DATA,PARENTS,read
from action_propagation_v41 import BRANCHES,SCHEMA
from coupled_replay_v40 import VARIANTS
from audit_controller_support_v26 import verify_parent

HORIZONS=[5,25,50]
TARGETS=['log','teacher']


def main():
    assert read(OUT/'status.json')['status']=='probes_complete_evaluation_pending'
    w=pd.read_csv(DATA/'windows.csv')
    with np.load(DATA/'windows.npz') as z:logged=z['input_future_command'].astype(float)
    metric_rows=[];strata_rows=[];factor_rows=[];history_rows=[];time_rows=[];seen={}
    snapshots=0;identities=0
    labels=[f'{name}{axis}' if width>1 else name for name,width in SCHEMA for axis in range(width)]
    for fold in range(5):
        ids=read(DATA/f'fold{fold}/manifest.json')['query_indices'];modes=w.iloc[ids]['mode'].to_numpy()
        for variant,_,_ in VARIANTS:
            with np.load(PARENT/f'fold{fold}/{variant}_teacher.npz') as z:teacher=z['commands']
            with np.load(OUT/f'fold{fold}/{variant}_teacher_states.npz') as z:ht=z['state']
            for stage in [30,35]:
                for seed in [17,23,42]:
                    common=dict(fold=fold,variant=variant,stage=stage,seed=seed)
                    path=OUT/f'fold{fold}/{variant}_v{stage}_seed{seed}.npz'
                    with np.load(path) as z:a={k:z[k] for k in z.files}
                    with np.load(PARENT/f'fold{fold}/{variant}_v{stage}_seed{seed}.npz') as z:
                        original=z['commands'];state_out=z['state_outside'].any(-1);command_out=z['command_outside'].any(-1)
                    np.testing.assert_array_equal(a['array_index'],ids)
                    command=a['commands'];assert command.shape==(len(ids),50,14,4) and np.isfinite(command).all()
                    np.testing.assert_array_equal(command[:,:,0],original);np.testing.assert_array_equal(command[:,:,3],teacher)
                    seen.setdefault((variant,stage,seed),[]).extend(ids)
                    # Complete per-window metrics are stored in compact arrays.
                    error=np.stack([command-logged[ids,:,None,:],command-teacher[:,:,None,:]],axis=2)
                    per_window=np.stack([(error[:,:h]**2).mean(1) for h in HORIZONS],axis=1)
                    np.savez_compressed(path.with_name(path.stem+'_metrics.npz'),array_index=ids,mse=per_window,
                        surface_clip_fraction=a['surface_clipped'].any(-1).mean(1))
                    for hi,h in enumerate(HORIZONS):
                        rmse=np.sqrt(per_window[:,hi].mean(0))
                        for ti,target in enumerate(TARGETS):
                            for bi,branch in enumerate(BRANCHES):
                                for axis in range(4):metric_rows.append(dict(**common,horizon=h,target=target,branch=branch,axis=axis,rmse=rmse[ti,bi,axis],windows=len(ids)))
                    for target_i,target in enumerate(TARGETS):
                        curve=np.sqrt((error[:,:,target_i]**2).mean(0))
                        for k in range(50):
                            for bi,branch in enumerate(BRANCHES):
                                for axis in range(4):time_rows.append(dict(**common,target=target,step=k,branch=branch,axis=axis,rmse=curve[k,bi,axis]))
                    # Disjoint per-time coverage; source flags remain the v40 factual ones.
                    flags=state_out.astype(int)+2*command_out.astype(int)
                    groups=[('mode',str(mode),np.broadcast_to((modes==mode)[:,None],flags.shape)) for mode in sorted(set(modes))]
                    groups += [('support',str(j),flags==j) for j in range(4)]
                    total_sq=np.zeros((2,14,4));total_points=0
                    for family,label,mask in groups:
                        points=int(mask.sum())
                        if not points:continue
                        sq=(error[mask]**2).sum(0)
                        if family=='support':total_sq+=sq;total_points+=points
                        for ti,target in enumerate(TARGETS):
                            for bi,branch in enumerate(BRANCHES):
                                for axis in range(4):strata_rows.append(dict(**common,family=family,layer=label,target=target,branch=branch,axis=axis,points=points,mse=sq[ti,bi,axis]/points))
                    assert total_points==len(ids)*50
                    np.testing.assert_allclose(total_sq,(error**2).sum((0,1)),rtol=1e-12,atol=1e-10)
                    mm,mt,tm,tt=[command[:,:,j] for j in range(4)]
                    d=mm-tt;plant=mm-mt;history=mt-tt
                    reverse_plant=tm-tt;reverse_history=mm-tm;interaction=mm-mt-tm+tt
                    np.testing.assert_allclose(d,plant+history,rtol=0,atol=1e-12)
                    np.testing.assert_allclose(d,reverse_plant+reverse_history,rtol=0,atol=1e-12)
                    np.testing.assert_allclose(plant-reverse_plant,interaction,rtol=0,atol=1e-12)
                    moments=dict(delta_sq=d*d,plant_sq=plant*plant,history_sq=history*history,
                        plant_history=plant*history,plant_delta=plant*d,history_delta=history*d,
                        reverse_plant_delta=reverse_plant*d,reverse_history_delta=reverse_history*d,interaction_sq=interaction*interaction)
                    for h in HORIZONS:
                        vals={k:v[:,:h].mean((0,1)) for k,v in moments.items()}
                        np.testing.assert_allclose(vals['delta_sq'],vals['plant_sq']+vals['history_sq']+2*vals['plant_history'],rtol=1e-12,atol=1e-12)
                        for axis in range(4):factor_rows.append(dict(**common,horizon=h,axis=axis,**{k:v[axis] for k,v in vals.items()}))
                    diff=a['model_controller_state']-ht
                    # Fixed air proxy and clock make this filter history identical.
                    np.testing.assert_array_equal(diff[:,:,labels.index('airspeed_filter')],0.)
                    for j,label in enumerate(labels):history_rows.append(dict(**common,field=label,rmse=float(np.sqrt((diff[:,:,j]**2).mean()))))
                    snapshots+=len(ids)*50;identities+=d.size
    for values in seen.values():np.testing.assert_array_equal(sorted(values),np.arange(407))
    metric=pd.DataFrame(metric_rows);metric.to_csv(OUT/'metrics_per_flight.csv',index=False)
    metric.groupby(['variant','stage','horizon','target','branch','axis']).rmse.mean().to_csv(OUT/'summary_rmse.csv')
    # This preserves per-seed values for direction consistency and flight-level checks.
    metric.groupby(['variant','stage','seed','horizon','target','branch','axis']).rmse.mean().to_csv(OUT/'summary_by_seed.csv')
    pd.DataFrame(strata_rows).to_csv(OUT/'strata_per_flight.csv',index=False)
    pd.DataFrame(factor_rows).to_csv(OUT/'factor_moments_per_flight.csv',index=False)
    pd.DataFrame(history_rows).to_csv(OUT/'controller_history_rmse.csv',index=False)
    pd.DataFrame(time_rows).to_csv(OUT/'time_curves_per_flight.csv',index=False)
    # Reproduce v40 absolute factual action metrics exactly at same aggregation.
    v40=pd.read_csv(PARENT/'action_metrics_per_flight.csv')
    for target,comparison in [('log','log'),('teacher','matched_teacher')]:
        a=metric[(metric.horizon==50)&(metric.branch=='mm')&(metric.target==target)].set_index(['variant','stage','seed','fold','axis']).sort_index()
        b=v40[v40.comparison==comparison].set_index(['variant','stage','seed','fold','axis']).sort_index()
        np.testing.assert_allclose(a.rmse,b.rmse,rtol=1e-12,atol=1e-12)
    for root in PARENTS:verify_parent(root)
    for name,digest in read(OUT/'protocol.json')['code'].items():assert base.sha(base.ROOT/name)==digest
    base.write(OUT/'verification.json',dict(status='passed',factual_model_runs=9768,teacher_runs=1628,
        model_controller_states_reconstructed=snapshots,action_scalar_identities_checked=identities,
        factual_all_micro_actions_torques_refs_limits_exact=True,all407_preserved=True,
        v40_action_metrics_reproduced=True,support_strata_recompose_exactly=True,airspeed_filter_history_identical=True,
        oracle_probes_not_propagated=True,model_promoted=False,validation_accessed=False,sealed_test_accessed=False,reserved_accessed=False))
    base.write(OUT/'status.json',dict(status='evaluated_independent_verification_pending'))
    print('v41 metrics and factorial identities passed',flush=True)


if __name__=='__main__':main()
