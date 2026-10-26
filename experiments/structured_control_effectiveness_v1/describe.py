from common import *
def main():
    historical=pd.read_csv(ABLATION/'results/factual_summary.csv')
    historical.query("variant=='V0' and split=='train' and cohort=='main' and variable=='p' and time_label=='300' and weighting=='flight'").to_csv(OUT/'historical_24epoch_reference.csv',index=False)
    f=pd.read_csv(OUT/'factual_points.csv.gz');e=pd.read_csv(OUT/'internal_points.csv.gz');s=pd.read_csv(OUT/'factual_summary.csv');pairs=pd.read_csv(SWAP/'results/pairs.csv');mem=pd.read_csv(SWAP/'results/cohorts.csv')
    ids=set(mem.query("split=='train' and cohort=='main'").pair_uid)
    changes=[]
    for var in ['p','q','r']:
        for method in ['A','B','D']:
            q=f[f.pair_uid.isin(ids)&(f.variable==var)&(f.method==method)&(f.time_label=='300')].set_index(['model','pair_uid'])
            a=q.loc['M0']
            for name in ['M1','M2']:
                b=q.loc[name].loc[a.index];active=abs(a.real)>.1;correct0=(a.real*a.pred>0);correct1=(b.real*b.pred>0)
                changes.append(dict(model=name,variable=var,method=method,pairs=len(a),active=int(active.sum()),directions_gained=int((active&~correct0&correct1).sum()),directions_lost=int((active&correct0&~correct1).sum()),amplitude_error_better=int((abs(abs(b.pred)-abs(b.real))<abs(abs(a.pred)-abs(a.real))).sum()),prediction_change_rms=float(np.sqrt(np.mean((b.pred-a.pred)**2))),mean_absolute_error_change=float(np.mean(abs(b.error)-abs(a.error)))))
    pd.DataFrame(changes).to_csv(OUT/'paired_change_summary.csv',index=False)
    r=e[e.pair_uid.isin(ids)&(e.variable=='p')&(e.time_label=='300')];effect=[]
    for name,g in r.groupby('model'):effect.append(dict(model=name,C_near_pairs=int((abs(g.C)<=.1).sum()),C_opposite_CA_CB_numerical=int((g.CA*g.CB<0).sum()),CA_CB_both_active=int(((abs(g.CA)>.1)&(abs(g.CB)>.1)).sum()),CA_CB_opposite_both_active=int(((abs(g.CA)>.1)&(abs(g.CB)>.1)&(g.CA*g.CB<0)).sum())))
    save(OUT/'internal_counts.json',effect)
    examples=f[f.pair_uid.isin(['train:124','train:336','train:412','validation:44'])&(f.time_label=='300')&(f.variable=='p')]
    examples.to_csv(OUT/'example_300.csv',index=False)
    history=pd.read_csv(OUT/'training_history.csv');plan=json.loads((OUT/'holdout_protocol.json').read_text());budget=[]
    for fold in plan['folds']:
        fraction=fold['train_windows']/107920
        for name,g in history.groupby('model'):
            # Epoch time includes validation; bracket fixed validation overhead conservatively.
            sec=float(g.seconds.mean())
            budget.append(dict(fold=fold['fold'],model=name,seeds=3,epochs=24,train_windows=fold['train_windows'],updates_per_epoch=int(np.ceil(fold['train_windows']/1024)),low_seconds=3*24*sec*fraction,high_seconds=3*24*sec))
    pd.DataFrame(budget).to_csv(OUT/'next_stage_budget.csv',index=False)
    save(OUT/'report_numbers.json',dict(training_seconds=float(history.seconds.sum()),next_stage_hours_lower=sum(x['low_seconds'] for x in budget)/3600,next_stage_hours_upper=sum(x['high_seconds'] for x in budget)/3600,actual_training_runs=3,actual_epochs=18,actual_optimizer_updates=1908,smoke_updates=9,next_stage_executed=False))
    print(json.dumps(json.loads((OUT/'report_numbers.json').read_text()),indent=2))
if __name__=='__main__':main()
