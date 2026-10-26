"""Freeze three group-disjoint folds by graph size only; never train these folds here."""
from common import *
def main():
    pairs=pd.read_csv(SWAP/'results/pairs.csv');members=pd.read_csv(SWAP/'results/cohorts.csv')
    main=pairs[pairs.pair_uid.isin(members.query("split=='train' and cohort=='main'").pair_uid)]
    train=pd.read_parquet(TRAIN/'train_windows.parquet');val=pd.read_parquet(TRAIN/'validation_windows.parquet')
    nodes=set(main.flight_a)|set(main.flight_b);adj={f:set() for f in nodes}
    for p in main.itertuples():adj[p.flight_a].add(p.flight_b);adj[p.flight_b].add(p.flight_a)
    # Same source ULog/boot is also indivisible, even when flight segment IDs differ.
    for f in nodes:adj[f].update(g for g in nodes if g.split(':')[0]==f.split(':')[0])
    seen=set();components=[]
    for f in sorted(nodes):
        if f in seen:continue
        todo=[f];component=set()
        while todo:
            node=todo.pop()
            if node in component:continue
            component.add(node);todo.extend(adj[node]-component)
        seen.update(component);pp=main[main.flight_a.isin(component)&main.flight_b.isin(component)]
        components.append(dict(flights=sorted(component),pairs=sorted(pp.pair_uid),key=min(component)))
    components.sort(key=lambda x:(-len(x['pairs']),x['key']))
    folds=[[] for _ in range(3)];loads=[0]*3
    for c in components:
        fold=min(range(3),key=lambda i:(loads[i],i));folds[fold].append(c);loads[fold]+=len(c['pairs'])
    truth=pd.read_csv(ABLATION/'results/factual_points.csv.gz');truth=truth[(truth.variant=='V0')&(truth.model=='GRU-Control')&(truth.seed==17)&(truth.variable=='p')&(truth.time_label=='300')]
    plans=[];coverage=[];indices={};assignments=[]
    for fi,cs in enumerate(folds):
        selected={f for c in cs for f in c['flights']};boots={f.split(':')[0] for f in selected}
        held={f for f in train.flight_id.unique() if f.split(':')[0] in boots};remaining=set(train.flight_id)-held
        assert not held&remaining and not boots&{f.split(':')[0] for f in remaining}
        eligible=pairs[(pairs.split=='train')&pairs.flight_a.isin(held)&pairs.flight_b.isin(held)]
        evalmain=main[main.flight_a.isin(held)&main.flight_b.isin(held)]
        fit=np.flatnonzero(train.flight_id.isin(remaining));hold=np.flatnonzero(train.flight_id.isin(held));indices[f'fold{fi}_train_indices']=fit;indices[f'fold{fi}_held_indices']=hold
        plans.append(dict(fold=fi,train_flights=sorted(remaining),held_flights=sorted(held),validation_flights=sorted(val.flight_id.unique()),train_windows=len(fit),held_windows=len(hold),main_pairs=sorted(evalmain.pair_uid),eligible_all_frozen_pairs=sorted(eligible.pair_uid),components=len(cs)))
        for c in cs:
            for uid in c['pairs']:assignments.append(dict(fold=fi,component_key=c['key'],pair_uid=uid))
        for method in ['A','B','D']:
            r=truth[(truth.method==method)&truth.pair_uid.isin(evalmain.pair_uid)].real.to_numpy();assert len(r)==len(evalmain)
            coverage.append(dict(fold=fi,method=method,pairs=len(r),flights=len(held),components=len(cs),active=int((abs(r)>.1).sum()),positive=int((r>.1).sum()),negative=int((r<-.1).sum()),real_rms=float(np.sqrt(np.mean(r*r))),min_real=float(r.min()),max_real=float(r.max()),differential_rms_min=float(evalmain.difference_differential_rms.min()),differential_rms_max=float(evalmain.difference_differential_rms.max()),duration_min_s=float(evalmain.difference_differential_longest_us.min()/1e6),duration_max_s=float(evalmain.difference_differential_longest_us.max()/1e6),dates=len(set(evalmain.date_a)|set(evalmain.date_b)),configs=len(set(evalmain.controller_a)|set(evalmain.controller_b)),same_config_pairs=int(evalmain.same_controller.sum())))
    pd.DataFrame(assignments).to_csv(OUT/'holdout_pair_assignments.csv',index=False);pd.DataFrame(coverage).to_csv(OUT/'holdout_coverage.csv',index=False);np.savez_compressed(OUT/'holdout_window_indices.npz',**indices)
    save(OUT/'holdout_protocol.json',dict(folds=plans,assignment='descending component pair count then minimum flight ID; place in smallest current pair-count fold, tie fold index; merge same-ULog segments',chosen_without_model_predictions=True,coverage_descriptive_only=True,test_read=False,executed=False,seeds=[17,29,43],models=['M0','M1','M2'],epochs=24,early_stop=False,optimizer='same AdamW and cosine T_max24 for all',training_runs=27,normalization='Refit input/output normalization and loss scales from each fold training flights only; identical fold-specific values for all models; never reuse global statistics containing held flights.',checkpoint_selection='Original validation 27 flights, equal-flight multistep loss; never use held-fold pair results. Validation:44 remains selection-exposed case only.',evaluation='Every primary pair evaluated only in its held component fold; aggregate per flight/component, report giant component separately; retain frozen all-background pairs only if both flights held. No replacement matches.',limitations=['Original frozen matching rules were developed using original train flights, including held folds: prediction holdout is genuine, selection is not a prospectively independent experiment.','Closed-loop observational differences are not physical counterfactual supervision.','Few components and one giant component preclude strong generalization or reliable independent-window inference.']))
    print(pd.DataFrame(coverage).to_string(index=False))
if __name__=='__main__':main()
