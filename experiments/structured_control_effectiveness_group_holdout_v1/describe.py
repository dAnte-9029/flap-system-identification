"""Readable tables and numerical research notes; no automatic model selection."""
from common import *

def main():
    s=pd.read_csv(OUT/'factual_summary.csv');i=pd.read_csv(OUT/'internal_summary.csv');p=pd.read_csv(OUT/'paired_summary.csv');a=pd.read_csv(OUT/'component_assignments.csv');g=pd.read_csv(OUT/'per_group_300.csv',dtype={'folds':str});t=pd.read_csv(OUT/'trajectory_summary.csv');points=pd.read_csv(OUT/'factual_points.csv.gz');peak=pd.read_csv(OUT/'peak_metrics.csv.gz');mainids=set(a.pair_uid)
    tables={}
    main=s.query("cohort=='main' and scope=='all_heldout' and time_label=='300' and variable=='p'")
    cols=['model','seed','method','pairs','active_pairs','near_zero_fraction','direction_accuracy','scale_ratio','signed_gain','rmse','mae']
    tables['main_flight_equal']=main[main.weighting=='flight'][cols]
    tables['weight_sensitivity']=main[main.seed=='pooled'][['weighting']+cols]
    tables['fold_sensitivity']=s.query("cohort=='main' and time_label=='300' and variable=='p' and weighting=='flight' and seed=='pooled'")[['scope']+cols]
    tables['cohort_sensitivity']=s.query("scope=='all_heldout' and time_label=='300' and variable=='p' and weighting=='flight' and seed=='pooled' and method=='D'")[['cohort','frozen_cohort_pairs','effective_flights','effective_components']+cols]
    tables['signed_amplitude_bias']=main.query("weighting=='flight' and seed=='pooled'")[['model','method','positive_pairs','negative_pairs','positive_bias','negative_bias','positive_amplitude_bias','negative_amplitude_bias']]
    tables['threshold_sensitivity']=main.query("weighting=='flight' and seed=='pooled'")[['model','method','active_pairs_0p05','direction_0p05','active_pairs','direction_accuracy','active_pairs_0p2','direction_0p2']]
    tables['internal']=i.query("cohort=='main' and scope=='all_heldout' and time_label=='300' and variable=='p' and weighting=='flight'")[['model','seed','pairs','CA_rms','CB_rms','C_rms','H_rms','C_over_H','both_active_pairs','both_active_n','same_direction','abstention_fraction','C_signed_gain','CH_opposite_fraction','cancellation']]
    tables['paired']=p.query("scope=='all_heldout' and variable=='p' and weighting=='flight'")[['model','method','seed','pairs','active_pairs','gained_pair_seed_rows','lost_pair_seed_rows','direction_difference','mean_absolute_error_change','mean_amplitude_error_change']]
    tables['components']=g.query("grouping=='component' and kind=='factual' and seed=='pooled' and variable=='p' and method=='D'")[['group','folds','model','pairs','active_pairs','direction_accuracy','scale_ratio','signed_gain','rmse']]
    tables['coupling']=s.query("scope=='all_heldout' and cohort=='main' and time_label=='300' and weighting=='flight' and seed=='pooled' and method=='D'")[['model','variable','active_pairs','direction_accuracy','scale_ratio','signed_gain','rmse']]
    tables['trajectory']=t
    cases=json.loads((OUT/'illustration_selection.json').read_text());uids=list(dict.fromkeys(cases['fixed']+cases.get('posthoc_direction_change_cases',[])+[cases['posthoc_best_absolute_error_change'],cases['posthoc_worst_absolute_error_change']]))
    tables['cases']=points[(points.pair_uid.isin(uids))&(points.time_label=='300')&(points.variable=='p')][['pair_uid','fold','role','model','seed','method','real','pred','error']]
    peak['abs_lag_ms']=abs(peak.lag_ms);peak['absolute_peak_amplitude_error']=abs(abs(peak.pred_peak)-abs(peak.real_peak))
    tables['peaks']=peak[(peak.pair_uid.isin(mainids))].groupby(['model','seed','method','variable'])[['lag_ms','abs_lag_ms','absolute_peak_amplitude_error','real_boundary','pred_boundary','peak_active','peak_sign_correct']].mean().reset_index()
    numbers={name:df.to_dict('records') for name,df in tables.items()}
    comparisons=[]
    for model in ['M1','M2']:
        for method in ['B','D']:
            d=p[(p.scope.str.startswith('C'))&(p.model==model)&(p.variable=='p')&(p.method==method)&(p.weighting=='pair')]
            pooled=d[d.seed=='pooled'];seed=d[d.seed!='pooled'];allseed=seed.groupby('scope').direction_difference
            comparisons.append(dict(model=model,method=method,components_with_active_pairs=int((pooled.active_pairs>0).sum()),pooled_improved_components=int((pooled.direction_difference>1e-12).sum()),pooled_worse_components=int((pooled.direction_difference< -1e-12).sum()),all_three_seeds_improve_components=int((allseed.min()>1e-12).sum()),all_three_seeds_worsen_components=int((allseed.max()< -1e-12).sum())))
    numbers['component_consistency']=comparisons
    # Physical-unit trajectory ratios use equal-flight summaries, never normalized loss.
    tr=t.groupby('model').mean(numeric_only=True);numbers['trajectory_changes_percent']={m:{c:float((tr.loc[m,c]/tr.loc['M0',c]-1)*100) for c in tr.columns if c.endswith(('path_rms','endpoint_norm'))} for m in ['M1','M2']}
    save(OUT/'report_numbers.json',numbers)
    def fmt(v):
        if isinstance(v,(float,np.floating)):return f'{v:.5g}' if np.isfinite(v) else 'NA'
        return str(v)
    text=['# 自动生成的描述性结果表\n\n种子分别计分；pooled 行不是新增独立样本。完整定义见 README。\n']
    for name,df in tables.items():
        text.extend(['\n## '+name+'\n','| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |'])
        text.extend('| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))
    (HERE/'result_tables.md').write_text('\n'.join(text)+'\n')
    print('Main p B/D, pooled seed scores, flight equal:')
    print(main.query("weighting=='flight' and seed=='pooled' and method in ['B','D']")[cols].to_string(index=False))
    print('Component consistency:',json.dumps(comparisons))

if __name__=='__main__':main()
