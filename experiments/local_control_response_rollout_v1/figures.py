"""Deterministic illustrations and compact tables; never changes cohort membership."""
from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import labels
HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,default=HERE/'results');ap.add_argument('--figures',type=Path,default=HERE/'figures');args=ap.parse_args()
out=args.results;figs=args.figures;figs.mkdir(parents=True,exist_ok=True)
s=pd.read_csv(out/'summary.csv'); c=pd.read_csv(out/'curve_metrics.csv'); cs=pd.read_csv(out/'curve_summary.csv')
r=pd.read_csv(out/'response_curves.csv.gz'); p=pd.read_csv(out/'pairs.csv'); mem=pd.read_csv(out/'cohorts.csv')
main=mem.query("cohort=='main' and split=='train'").pair_uid
val=mem.query("cohort=='main' and split=='validation'").pair_uid.iloc[0]
models=['GRU-Control','GRU-NoControl','Hold','LinearTrend']; colors=dict(zip(models,['#1565c0','#ef6c00','#555555','#8e24aa']))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':120})
def save(fig,name):
    fig.tight_layout();fig.savefig(figs/(name+'.png'),dpi=180);plt.close(fig)
def response(ax,uid,var='p',allmodels=False):
    d=r[(r.pair_uid==uid)&(r.variable==var)&r.supported].sort_values('time_ms')
    truth=d[d.model=='Hold'].drop_duplicates('time_ms');ax.plot(truth.time_ms,truth.real,'k-',lw=2,label='Recorded')
    for model in models if allmodels else models[:3]:
        for seed,g in d[d.model==model].groupby('seed'):
            g=g.drop_duplicates('time_ms');ax.plot(g.time_ms,g.pred,color=colors[model],alpha=.65,lw=1,
                label=f'{model} s{seed}' if model.startswith('GRU') else model)
    ax.axhline(0,color='gray',lw=.6);ax.set(xlabel='Time (ms)',ylabel=f'D {var} ('+('rad/s)' if var in ['p','q','r'] else 'm/s)'))
# Examples are selected only for illustration using the declared rules.
cc=c[c.pair_uid.isin(main)].groupby(['pair_uid','model']).agg(curve_rmse=('curve_rmse','mean'),real_rms=('real_rms','mean')).reset_index()
ctrl=cc[cc.model=='GRU-Control'].set_index('pair_uid');hold=cc[cc.model=='Hold'].set_index('pair_uid')
success=ctrl[ctrl.real_rms>=.1].curve_rmse.idxmin();failure=(ctrl.curve_rmse-hold.curve_rmse).idxmax()
e=r[(r.pair_uid.isin(main))&(r.variable=='p')&(r.time_kind=='supported_endpoint')&(r.model=='GRU-Control')]
large=e.groupby('pair_uid').error.mean().abs().idxmax()
examples={'lowest_curve_error':success,'worst_vs_hold':failure,'largest_endpoint_bias':large,'validation_case':val}
(out/'examples.json').write_text(json.dumps(examples,indent=2)+'\n')
fig,axes=plt.subplots(2,2,figsize=(13,8))
for ax,(title,uid) in zip(axes.flat,examples.items()):response(ax,uid);ax.set_title(title.replace('_',' ')+' | '+uid)
axes.flat[0].legend(fontsize=8,ncol=2);save(fig,'fig1_response_examples')
# Scatter retains every seed and every fixed pair; exact-500 panel is explicitly smaller.
fig,axes=plt.subplots(2,3,figsize=(13,8))
for ax,point in zip(axes.flat,[100,200,300,400,500,'supported_endpoint']):
    d=r[(r.pair_uid.isin(main))&(r.variable=='p')&(r.model=='GRU-Control')&r.supported]
    d=d[d.time_kind=='supported_endpoint'] if isinstance(point,str) else d[(d.time_kind=='grid')&np.isclose(d.time_ms,point)]
    lim=max(abs(d.real).max(),abs(d.pred).max(),.2)*1.1
    ax.axvspan(-.1,.1,color='gray',alpha=.15);ax.plot([-lim,lim],[-lim,lim],'k--',lw=.8)
    for seed,g in d.groupby('seed'):ax.scatter(g.real,g.pred,s=17,alpha=.6,label=str(seed))
    ax.axhline(0,color='gray',lw=.5);ax.axvline(0,color='gray',lw=.5)
    ax.set(xlim=(-lim,lim),ylim=(-lim,lim),xlabel='Recorded D p (rad/s)',ylabel='Predicted D p (rad/s)',
        title=f'{point}'+(' ms' if not isinstance(point,str) else ' (489-500 ms)')+f' | n={d.pair_uid.nunique()}')
axes.flat[0].legend(title='Seed');save(fig,'fig2_direction_amplitude')
# Time comparison excludes incomplete exact-500 from connected full-cohort curves.
base=s[(s.split=='train')&(s.cohort=='main')&(s.variable=='p')&(s.weighting=='flight')]
fig,axes=plt.subplots(1,3,figsize=(15,4.6))
for ax,metric in zip(axes[:2],['rmse','mae']):
    for model in models:
        d=base[(base.model==model)&(base.seed=='pooled')&base.time_label.isin(['100','200','300','400'])].copy()
        ax.plot(d.time_label.astype(int),d[metric],'-o',color=colors[model],label=model)
    ax.set(xlabel='Time (ms); n=48 at every point',ylabel=f'p difference {metric.upper()} (rad/s)')
for j,model in enumerate(models):
    d=base[(base.model==model)&(base.time_label=='supported_endpoint')]
    pooled=d[d.seed=='pooled'].iloc[0];axes[2].bar(j,pooled.rmse,color=colors[model],alpha=.7)
    axes[2].scatter(np.full(sum(d.seed!='pooled'),j),d[d.seed!='pooled'].rmse,c='black',s=22,zorder=5)
axes[2].set(xticks=range(4),xticklabels=['Control','NoControl','Hold','Trend'],ylabel='p endpoint RMSE (rad/s)',title='Supported endpoint: n=48\nDots = individual seeds')
axes[0].legend(fontsize=8);fig.suptitle('Equal-flight weighting; GRU pooled errors, not ensemble predictions',y=1.02);save(fig,'fig3_model_baselines')
subsets=['main','flight_disjoint','same_firmware','same_controller','strict','small_pair']
fig,axes=plt.subplots(1,2,figsize=(14,5))
for ax,table,metric in [(axes[0],s[(s.variable=='p')&(s.time_label=='supported_endpoint')],'rmse'),(axes[1],cs,'curve_rmse')]:
    for j,model in enumerate(models[:3]):
        d=table[(table.split=='train')&(table.weighting=='flight')&(table.seed=='pooled')&(table.model==model)].set_index('cohort').loc[subsets]
        ax.bar(np.arange(len(subsets))+(j-1)*.25,d[metric],width=.24,color=colors[model],label=model)
    counts=[len(mem[(mem.split=='train')&(mem.cohort==co)]) for co in subsets]
    ax.set(xticks=range(len(subsets)),xticklabels=[f'{co}\nn={n}' for co,n in zip(['All','Disjoint\nflights','Same\nfirmware','Same\nconfig','Strict','Small\ncontrol'],counts)],ylabel='p RMSE (rad/s)',title='Supported endpoint' if metric=='rmse' else 'Full supported curve')
axes[0].legend(fontsize=8);save(fig,'fig4_subset_sensitivity')
fig,axes=plt.subplots(2,3,figsize=(14,8))
for ax,var in zip(axes.flat,['p','q','r','vx','vy','vz']):response(ax,val,var);ax.set_title(val+' | '+var)
axes.flat[0].legend(fontsize=7,ncol=2);save(fig,'fig5_validation_coupling')
# Background response/error distributions are descriptive, not causal bounds.
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
coh=['main','small_pair','both_future_quiet'];values=[[],[]]
for co in coh:
    ids=mem[(mem.split=='train')&(mem.cohort==co)].pair_uid
    d=r[(r.pair_uid.isin(ids))&(r.variable=='p')&(r.time_kind=='supported_endpoint')]
    values[0].append(d[d.model=='Hold'].real.abs().to_numpy())
    values[1].append(d[d.model=='GRU-Control'].assign(abs_error=lambda x:x.error.abs()).groupby('pair_uid').abs_error.mean().to_numpy())
for ax,v,title in zip(axes,values,['Recorded |D p|','Control mean-seed |error|']):
    ax.boxplot(v,tick_labels=['Main (48)','Small-control (29)','Both quiet (30)'],showfliers=True)
    ax.set(ylabel='rad/s',title=title+'; pair distributions')
save(fig,'fig6_background')
g=pd.read_csv(out/'per_group.csv');g=g[(g.split=='train')&(g.cohort=='main')&(g.seed=='pooled')]
fig,axes=plt.subplots(1,2,figsize=(13,4.5))
for ax,grouping in zip(axes,['date','component']):
    d=g[g.grouping==grouping].pivot(index='group',columns='model',values='rmse')
    for model in models[:3]:ax.plot(range(len(d)),d[model],'-o',label=model,color=colors[model])
    ax.set(xticks=range(len(d)),xticklabels=[str(x)[:10] for x in d.index],ylabel='p endpoint RMSE (rad/s)',title=grouping+' groups (descriptive, dependent)')
    ax.tick_params(axis='x',rotation=35)
axes[0].legend(fontsize=8);save(fig,'fig7_cross_flight')
# Compact report tables are exported directly from the full verified tables.
base.to_csv(out/'main_p_time_seed_table.csv',index=False)
s[(s.time_label=='supported_endpoint')&(s.weighting=='flight')&(s.seed=='pooled')].to_csv(out/'endpoint_table.csv',index=False)
cs[(cs.weighting=='flight')&(cs.seed=='pooled')].to_csv(out/'curve_table.csv',index=False)
# Dependency summaries and counts of wins retain the pair, flight and component distinction.
deps=[]
for (split,co),m in mem.groupby(['split','cohort']):
    q=p[p.pair_uid.isin(m.pair_uid)];flights=pd.Series(np.r_[q.flight_a,q.flight_b]);lab=labels(q)
    deps.append(dict(split=split,cohort=co,pairs=len(q),flights=flights.nunique(),components=len(set(lab.values())),max_pairs_per_flight=int(flights.value_counts().max()),
        unique_windows=len(set(q.window_a)|set(q.window_b)),max_pairs_per_component=max(pd.Series(lab).value_counts())))
pd.DataFrame(deps).to_csv(out/'cohort_dependencies.csv',index=False)
end=r[(r.pair_uid.isin(main))&(r.variable=='p')&(r.time_kind=='supported_endpoint')]
pe=end.assign(sq=lambda x:x.error**2).groupby(['pair_uid','model']).sq.mean().unstack()
pe.to_csv(out/'pair_endpoint_mse.csv')
lo=pd.read_csv(out/'leave_one_flight_out.csv').pivot(index='removed_flight',columns='model',values='rmse')
stats={'pair_wins_vs_hold':int((pe['GRU-Control']<pe.Hold).sum()),'pair_wins_vs_nocontrol':int((pe['GRU-Control']<pe['GRU-NoControl']).sum()),
    'leave_one_flight_out_control_rmse_range':[float(lo['GRU-Control'].min()),float(lo['GRU-Control'].max())],
    'leave_one_flight_out_control_minus_hold_range':[float((lo['GRU-Control']-lo.Hold).min()),float((lo['GRU-Control']-lo.Hold).max())],
    'leave_one_flight_out_control_minus_nocontrol_range':[float((lo['GRU-Control']-lo['GRU-NoControl']).min()),float((lo['GRU-Control']-lo['GRU-NoControl']).max())]}
(out/'descriptive_diagnostics.json').write_text(json.dumps(stats,indent=2)+'\n')
print(json.dumps(dict(examples=examples,diagnostics=stats),indent=2))
# Validation A/B trajectories expose cancellation that response differences can hide.
z=np.load(out/'native_rollouts.npz');pair=p.set_index('pair_uid').loc[val]
fig,axes=plt.subplots(1,2,figsize=(13,4.5))
for ax,side in zip(axes,['a','b']):
    i=int(pair[side]);n=int(z['future_count'][i]);t=z['native_time_us'][i,:n+1]/1000
    ax.plot(t,np.r_[z['truth_initial'][i,10],z['truth'][i,:n,10]],'k-',lw=2,label='Recorded aligned p')
    ax.scatter([0],[z['initial'][i,10]],marker='x',c='red',label='Causal initial')
    for model in models[:3]:
        for seed in ([17,29,43] if model.startswith('GRU') else [0]):
            ax.plot(t,np.r_[z['initial'][i,10],z[f'{model}_{seed}'][i,:n,10]],color=colors[model],alpha=.65,lw=1,label=f'{model} s{seed}')
    ax.set(title=f'{val} | trajectory {side.upper()}',xlabel='Native time (ms)',ylabel='p (rad/s)')
axes[0].legend(fontsize=7,ncol=2);save(fig,'fig8_validation_trajectories')
# Additional descriptive median examples, explicitly post-analysis illustrations.
margin=ctrl.curve_rmse-hold.curve_rmse
better=margin[margin<0].sort_values();worse=margin[margin>=0].sort_values()
typical={'median_improvement_vs_hold':better.index[len(better)//2],'median_deterioration_vs_hold':worse.index[len(worse)//2]}
(out/'descriptive_examples.json').write_text(json.dumps({'rule':'Post-analysis illustration only: median signed Control-minus-Hold curve RMSE within improved / worsened fixed main pairs. No cohort or metric changes.','pairs':typical},indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(13,4.5))
for ax,(name,uid) in zip(axes,typical.items()):response(ax,uid);ax.set_title(name.replace('_',' ')+' | '+uid)
axes[0].legend(fontsize=7,ncol=2);save(fig,'fig9_median_examples')
