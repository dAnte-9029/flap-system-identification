"""All-cohort context and predeclared cases; no success-only selection."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 figdir=HERE/'figures' if OUT==HERE/'results' else OUT.parent/'figures';figdir.mkdir(exist_ok=True);cfg=json.loads((HERE/'config.json').read_text());d=pd.read_csv(OUT/'point_decomposition.csv.gz');curves=pd.read_csv(OUT/'decomposition_curves.csv.gz');mem=pd.read_csv(OUT/'cohorts.csv');mainids=set(mem.query("split=='train' and cohort=='main'").pair_uid);m=d[(d.pair_uid.isin(mainids))&(d.variable=='p')&(d.model=='GRU-Control')&(d.time_label=='300')];s=pd.read_csv(OUT/'summary.csv');colors={17:'#0072B2',29:'#D55E00',43:'#009E73'}
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':120})
 def finish(fig,name):fig.tight_layout();fig.savefig(figdir/name,dpi=180);plt.close(fig)
 fig,ax=plt.subplots(1,3,figsize=(14,4))
 for seed,g in m.groupby('seed'):
  ax[0].scatter(g.CA,g.CB,s=22,alpha=.7,color=colors[seed],label=str(seed));ax[1].scatter(g.g,g.C,s=22,alpha=.7,color=colors[seed]);ax[2].scatter(abs(g.H),abs(g.C),s=22,alpha=.7,color=colors[seed])
 lim=max(abs(m.CA).max(),abs(m.CB).max())*1.1;ax[0].plot([-lim,lim],[-lim,lim],'k:',lw=1);ax[0].axvspan(-.1,.1,color='gray',alpha=.1);ax[0].axhspan(-.1,.1,color='gray',alpha=.1);ax[0].set(xlabel='CA: fixed hA response (rad/s)',ylabel='CB: fixed hB response (rad/s)',title='Same control contrast, two histories');ax[0].legend(title='Seed')
 ax[1].set(xlabel='Applied signed mean differential gap',ylabel='C (rad/s)',title='Internal association, not physical gain');ax[1].axhline(0,c='k',lw=.6);ax[1].axvspan(-.02,.02,color='gray',alpha=.15)
 top=max(abs(m.H).max(),abs(m.C).max())*1.05;ax[2].plot([0,top],[0,top],'k:');ax[2].set(xlabel='|H| (rad/s)',ylabel='|C| (rad/s)',title='48 pairs × 3 seeds, not 144 interventions');finish(fig,'fig1_direction_amplitude.png')
 selected=s[(s.split=='train')&(s.cohort=='main')&(s.variable=='p')&(s.seed=='pooled')&(s.weighting=='flight')];fig,ax=plt.subplots(1,2,figsize=(11,4))
 for model,style in [('GRU-Control','-'),('GRU-NoControl','--')]:
  g=selected[selected.model==model];g=g[g.time_label.isin(['100','200','300','400'])].copy();g['t']=g.time_label.astype(int);g=g.sort_values('t')
  for key,color in [('C','#0072B2'),('H','#D55E00'),('F','#009E73')]:ax[0].plot(g.t,g[key+'_rms'],style+'o',color=color,label=model+' '+key)
  ax[1].plot(g.t,g.both_near_fraction,style+'o',label=model+' both near')
 ax[0].set(xlabel='Elapsed model time (ms)',ylabel='p response RMS (rad/s)',title='Flight-equal, seeds pooled descriptively');ax[0].legend(fontsize=8);ax[1].set(xlabel='Elapsed model time (ms)',ylabel='Weighted fraction',ylim=(-.03,1.05),title='Both |CA| and |CB| ≤ 0.1 rad/s');ax[1].legend();finish(fig,'fig2_contribution_time.png')
 cohorts=['main','flight_disjoint','same_firmware','same_firmware_controller','strict','small_pair','both_future_quiet'];g=s[(s.split=='train')&(s.cohort.isin(cohorts))&(s.variable=='p')&(s.seed=='pooled')&(s.weighting=='flight')&(s.time_label=='300')&(s.model=='GRU-Control')].set_index('cohort').loc[cohorts];fig,ax=plt.subplots(1,2,figsize=(13,5));pos=np.arange(len(g));labels=[f'{c}\nn={int(r.pairs)}, flights={int(r.flights)}' for c,r in g.iterrows()]
 ax[0].barh(pos-.16,g.C_rms,.3,label='C');ax[0].barh(pos+.16,g.H_rms,.3,label='H');ax[0].set_yticks(pos,labels);ax[0].invert_yaxis();ax[0].set_xlabel('p RMS (rad/s)');ax[0].legend();ax[1].barh(pos,g.both_active_fraction,label='Both active');ax[1].barh(pos,g.both_near_fraction,left=g.both_active_fraction,label='Both near');ax[1].barh(pos,1-g.both_active_fraction-g.both_near_fraction,left=g.both_active_fraction+g.both_near_fraction,label='One near');ax[1].invert_yaxis();ax[1].set_yticks(pos,[]);ax[1].set_xlabel('Weighted fraction at 300 ms');ax[1].legend();finish(fig,'fig3_cohort_sensitivity.png')
 fig,axes=plt.subplots(4,3,figsize=(14,13),sharex=True)
 for rr,uid in enumerate(cfg['examples']):
  g=curves[(curves.pair_uid==uid)&(curves.variable=='p')&(curves.model=='GRU-Control')&curves.supported].sort_values('time_ms')
  for seed,ss in g.groupby('seed'):
   ss=ss.drop_duplicates('time_ms');axes[rr,0].plot(ss.time_ms,ss.CA,color=colors[seed],label=f'CA seed {seed}');axes[rr,0].plot(ss.time_ms,ss.CB,'--',color=colors[seed],label=f'CB seed {seed}')
   for k,style in [('C','-'),('H','--')]:axes[rr,1].plot(ss.time_ms,ss[k],style,color=colors[seed],label=f'{k} seed {seed}')
   axes[rr,2].plot(ss.time_ms,ss.F,color=colors[seed],label=f'F seed {seed}')
  ss=g[g.seed==17].drop_duplicates('time_ms');axes[rr,2].plot(ss.time_ms,ss.real_F,'k',lw=2,label='Observed factual difference')
  for a in axes[rr]:a.axhline(0,c='gray',lw=.6);a.axvline(300,c='gray',ls=':',lw=.6);a.set_title(uid);a.set_ylabel('p difference (rad/s)')
 for j,title in enumerate(['Fixed-history control effects','Symmetric C (solid), H (dashed)','Factual F versus recorded difference']):axes[0,j].set_title(title+'\n'+cfg['examples'][0]);axes[0,j].legend(fontsize=7);axes[-1,j].set_xlabel('Elapsed model time (ms)')
 finish(fig,'fig4_fixed_cases.png')
 fig,ax=plt.subplots(1,2,figsize=(12,4));ss=s[(s.split=='train')&(s.cohort=='main')&(s.seed=='pooled')&(s.weighting=='flight')&(s.time_label=='300')&(s.model=='GRU-Control')].set_index('variable').loc[VARS];pos=np.arange(6);ax[0].bar(pos-.16,ss.C_rms,.3,label='C');ax[0].bar(pos+.16,ss.H_rms,.3,label='H');ax[0].set_xticks(pos,VARS);ax[0].set_ylabel('RMS: m/s for v; rad/s for p/q/r');ax[0].legend();ax[0].set_title('Full four-channel swap: coupled outputs')
 consistency=pd.read_csv(OUT/'seed_consistency.csv');ss=consistency[(consistency.pair_uid.isin(mainids))&(consistency.variable=='p')&(consistency.time_label=='300')&(consistency.model=='GRU-Control')&(consistency.effect=='C')].sort_values('mean');ax[1].vlines(np.arange(len(ss)),ss.minimum,ss.maximum,color='#0072B2');ax[1].scatter(np.arange(len(ss)),ss['mean'],s=12);ax[1].axhspan(-.1,.1,color='gray',alpha=.15);ax[1].set(xlabel='All 48 fixed pairs, sorted for display only',ylabel='C across seeds (rad/s)',title='Seed range; gray = direction abstention band');finish(fig,'fig5_coupling_seed.png')
 for uid in cfg['examples']:
  g=curves[(curves.pair_uid==uid)&(curves.variable.isin(['p','q','r']))&(curves.model=='GRU-Control')&curves.supported];fig,axes=plt.subplots(3,3,figsize=(13,9))
  for ri,var in enumerate(['p','q','r']):
   for si,seed in enumerate([17,29,43]):
    gg=g[(g.variable==var)&(g.seed==seed)].sort_values('time_ms').drop_duplicates('time_ms')
    for col in ['YAA','YAB','YBA','YBB']:axes[ri,si].plot(gg.time_ms,gg[col],label=col)
    axes[ri,si].set(title=f'{var}, seed {seed}',xlabel='ms',ylabel='increment (rad/s)');axes[ri,si].axhline(0,c='gray',lw=.5)
  axes[0,0].legend();fig.suptitle(uid+' — all four predictions, no swapped truth',y=1.01);finish(fig,'case_'+uid.replace(':','_')+'_four_cells.png')
 print('Saved',len(list(figdir.glob('*.png'))),'PNG figures')
if __name__=='__main__':main()
