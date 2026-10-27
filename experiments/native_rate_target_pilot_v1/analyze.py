"""Stitch exactly one assigned fold per held pair; retain seed and dependence units."""
from common import *
from evaluate import old

def aggregate():
    facts=pd.read_csv(OUT/'factual_points.csv.gz');effects=pd.read_csv(OUT/'internal_points.csv.gz');pairs=pd.read_csv(OUT/'pairs.csv');mem=pd.read_csv(OUT/'cohorts.csv');assign=pd.read_csv(OUT/'holdout_pair_assignments.csv')
    lookup=dict(zip(assign.pair_uid,assign.fold));main=set(lookup)
    for d in [facts,effects]:
        selected=d[d.pair_uid.isin(main)]
        assert (selected.fold==selected.pair_uid.map(lookup)).all()
    primary=facts[(facts.pair_uid.isin(main))&(facts.time_label=='300')&(facts.method=='D')&(facts.variable=='p')]
    assert len(primary)==48*2 and primary.supported.all()
    assert not primary.duplicated(['pair_uid','model','seed']).any()
    held=facts[facts.role=='heldout'];internal=effects[effects.role=='heldout'];summ=[];ints=[];pergroup=[]
    for cohort,members in mem[mem.split=='train'].groupby('cohort'):
        requested=pairs[pairs.pair_uid.isin(members.pair_uid)]
        for scope in (['all_heldout','fold0','fold1','fold2','outside_fold0'] if cohort=='main' else ['all_heldout']):
            for frame,isinternal in [(held,False),(internal,True)]:
                selected=frame[frame.pair_uid.isin(members.pair_uid)&frame.supported]
                if scope.startswith('fold'):selected=selected[selected.fold==int(scope[-1])]
                if scope=='outside_fold0':selected=selected[selected.fold!=0]
                keys=['model','variable','time_label']+([] if isinternal else ['method']);metric=old.sm.metrics if isinternal else old.tm.metric
                for key,g in selected.groupby(keys,sort=False):
                    info=dict(zip(keys,key));pp=requested[requested.pair_uid.isin(g.pair_uid)]
                    for seed in [17]:
                        z=g if seed=='pooled' else g[g.seed==seed]
                        for weighting in ['pair','flight','component'] if info['time_label']=='300' else ['flight']:
                            result=dict(scope=scope,cohort=cohort,**info,seed=seed,weighting=weighting,frozen_cohort_pairs=len(requested),effective_flights=len(set(pp.flight_a)|set(pp.flight_b)),effective_components=len(set(old.stats.labels(pp).values())),**metric(z,z.pair_uid.map(old.stats.weights(pp,weighting))));(ints if isinternal else summ).append(result)
    pd.DataFrame(summ).to_csv(OUT/'factual_summary.csv',index=False);pd.DataFrame(ints).to_csv(OUT/'internal_summary.csv',index=False)
    keys=sorted(assign.component_key.unique());component={uid:f'C{keys.index(c):02d}' for uid,c in zip(assign.pair_uid,assign.component_key)}
    assign.assign(component_id=assign.pair_uid.map(component)).to_csv(OUT/'component_assignments.csv',index=False)
    pp=pairs[pairs.pair_uid.isin(main)]
    groups=[('component',c,pp[pp.pair_uid.map(component)==c]) for c in sorted(set(component.values()))]
    for field in ['flight','date','controller']:
        for value in sorted(set(pp[field+'_a'])|set(pp[field+'_b'])):groups.append((field,value,pp[(pp[field+'_a']==value)|(pp[field+'_b']==value)]))
    for kind,value,cp in groups:
        for frame,isinternal in [(held,False),(internal,True)]:
            d=frame[(frame.time_label=='300')&frame.pair_uid.isin(cp.pair_uid)&frame.variable.isin(['p','q','r'])&frame.supported]
            keys=['model','variable']+([] if isinternal else ['method']);metric=old.sm.metrics if isinternal else old.tm.metric
            for key,g in d.groupby(keys):
                for seed in [17]:
                    z=g if seed=='pooled' else g[g.seed==seed]
                    pergroup.append(dict(grouping=kind,group=value,kind='internal' if isinternal else 'factual',**dict(zip(keys,key)),seed=seed,folds=','.join(map(str,sorted(z.fold.unique()))),**metric(z,np.ones(len(z)))))
    pd.DataFrame(pergroup).to_csv(OUT/'per_group_300.csv',index=False)
    # Comparisons are paired by fold/pair/seed/time; never average trajectories before direction scoring.
    facts['architecture']=facts.model.str[:2]
    keys=['architecture','fold','pair_uid','seed','role','variable','method','time_label'];base=facts[facts.model.str.endswith('A')];delta=facts[facts.model.str.endswith('B')].merge(base,on=keys,suffixes=('','_A'),validate='one_to_one')
    np.testing.assert_allclose(delta.real,delta.real_A,atol=0,rtol=0,equal_nan=True)
    active=abs(delta.real)>.1;good=(delta.real*delta.pred>0)&(abs(delta.pred)>1e-8);oldgood=(delta.real*delta.pred_A>0)&(abs(delta.pred_A)>1e-8)
    delta['direction_active']=active;delta['gained']=active&good&~oldgood;delta['lost']=active&oldgood&~good
    delta['absolute_error_change']=abs(delta.error)-abs(delta.error_A);delta['amplitude_error_change']=abs(abs(delta.pred)-abs(delta.real))-abs(abs(delta.pred_A)-abs(delta.real));delta.to_csv(OUT/'paired_changes.csv.gz',index=False)
    changes=[]
    for scope,ids in [('all_heldout',main),('outside_fold0',set(assign[assign.fold!=0].pair_uid))]+[(f'fold{i}',set(assign[assign.fold==i].pair_uid)) for i in range(3)]+[(c,{u for u,v in component.items() if v==c}) for c in sorted(set(component.values()))]:
        d=delta[(delta.time_label=='300')&delta.pair_uid.isin(ids)&delta.supported]
        for (name,var,method),g in d.groupby(['model','variable','method']):
            for seed in [17]:
                z=g if seed=='pooled' else g[g.seed==seed];cp=pairs[pairs.pair_uid.isin(z.pair_uid)]
                for weighting in ['pair','flight','component']:
                    w=z.pair_uid.map(old.stats.weights(cp,weighting)).to_numpy();w=w/w.sum();mask=z.direction_active.to_numpy()
                    difference=float(np.dot(w[mask],(z.gained.astype(int)-z.lost.astype(int)).to_numpy()[mask])/w[mask].sum()) if any(mask) else np.nan
                    changes.append(dict(scope=scope,model=name,variable=var,method=method,seed=seed,weighting=weighting,pairs=z.pair_uid.nunique(),active_pairs=z.loc[mask,'pair_uid'].nunique(),pair_seed_rows=len(z),gained_pair_seed_rows=int(z.gained.sum()),lost_pair_seed_rows=int(z.lost.sum()),direction_difference=difference,mean_absolute_error_change=float(np.dot(w,z.absolute_error_change)),mean_amplitude_error_change=float(np.dot(w,z.amplitude_error_change))))
    pd.DataFrame(changes).to_csv(OUT/'paired_summary.csv',index=False)
    # Peak diagnostics use the same supported grid and mark boundary extrema.
    curves=pd.read_csv(OUT/'factual_curves.csv.gz');peak=[]
    for key,g in curves[curves.supported&curves.variable.isin(['p','q','r'])].groupby(['fold','model','seed','pair_uid','role','variable','method']):
        g=g.sort_values('time_ms').drop_duplicates('time_ms');r=g.real.to_numpy();p=g.pred.to_numpy();t=g.time_ms.to_numpy();ri=int(np.argmax(abs(r)));pi=int(np.argmax(abs(p)))
        peak.append(dict(zip(['fold','model','seed','pair_uid','role','variable','method'],key),real_peak=r[ri],pred_peak=p[pi],real_peak_time_ms=t[ri],pred_peak_time_ms=t[pi],lag_ms=t[pi]-t[ri],peak_active=bool(abs(r[ri])>.1),peak_sign_correct=bool(r[ri]*p[pi]>0 and abs(p[pi])>1e-8 and abs(r[ri])>.1),real_boundary=ri in [0,len(t)-1],pred_boundary=pi in [0,len(t)-1]))
    pd.DataFrame(peak).to_csv(OUT/'peak_metrics.csv.gz',index=False)
    pd.DataFrame(summ).query("scope=='all_heldout' and cohort=='main' and variable=='p' and time_label=='300' and weighting=='flight'").to_csv(OUT/'main_p_300.csv',index=False)
    validation=facts[facts.role=='selection_exposed_case'];validation.to_csv(OUT/'validation_case_points.csv.gz',index=False)
    print(pd.read_csv(OUT/'main_p_300.csv')[['model','seed','method','direction_accuracy','scale_ratio','signed_gain']].to_string(index=False))
if __name__=='__main__':aggregate()
