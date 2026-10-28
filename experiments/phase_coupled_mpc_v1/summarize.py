from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
a=pd.read_csv(OUT/'phase_per_flight.csv');r=pd.read_csv(OUT/'phase_resets.csv.gz')
r['correction_deg']=np.rad2deg(np.angle(np.exp(1j*r.reset_correction_rad)))
r=r.merge(a[['flight_id','split','date']],on='flight_id',validate='many_to_one')
date=a.groupby(['split','date']).agg(flights=('flight_id','count'),hall_flights=('hall_rows',lambda x:int((x>0).sum())),valid_phase_flights=('valid_rows',lambda x:int((x>0).sum())),median_resultant=('relative_vs_hall_resultant','median'))
date.to_csv(OUT/'phase_by_date.csv')
f=pd.read_csv(OUT/'phase_forecast_windows.csv.gz');s=pd.read_csv(OUT/'phase_forecast_summary.csv')
bydate=f.assign(sq=f.circular_error_rad**2).groupby(['split','model','horizon_s','date','flight_id']).sq.mean().groupby(level=[0,1,2,3]).mean().pow(.5)*180/np.pi
bydate.rename('flight_equal_phase_rmse_deg').to_csv(OUT/'phase_forecast_by_date.csv')
coherence=[]
for (split,model,h),g in f.groupby(['split','model','horizon_s']):
    item=dict(split=split,model=model,horizon_s=h)
    for n in [1,2,3]:
        v=np.exp(1j*n*g.circular_error_rad.to_numpy())
        means=pd.DataFrame(dict(flight_id=g.flight_id,real=v.real,imag=v.imag)).groupby('flight_id')[['real','imag']].mean()
        item[f'h{n}_coherence']=float(abs(means.real.mean()+1j*means.imag.mean()))
    coherence.append(item)
pd.DataFrame(coherence).to_csv(OUT/'phase_forecast_harmonic_coherence.csv',index=False)
stats={}
for split,g in r.groupby('split'):
    q=np.quantile(abs(g.correction_deg),[.5,.9,.95,.99,.999])
    stats[split]=dict(reference_updates=len(g),abs_wrapped_correction_deg_quantiles=q.tolist(),over10deg=int((abs(g.correction_deg)>10).sum()),over30deg=int((abs(g.correction_deg)>30).sum()))
(OUT/'reference_jump_summary.json').write_text(json.dumps(stats,indent=2))
fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
for split,color in [('train','tab:blue'),('validation','tab:orange')]:
    g=a[a.split==split];axes[0].scatter(pd.to_datetime(g.date),g.relative_vs_hall_resultant,s=12,alpha=.6,label=split,color=color)
axes[0].set(ylabel='Within-flight circular consistency R1',title='Relative encoder vs Hall reference');axes[0].tick_params(axis='x',rotation=90);axes[0].legend()
for model in ['held_frequency','motor_first_order']:
    g=s[(s.split=='validation')&(s.model==model)];axes[1].plot(g.horizon_s,g.circular_rmse_deg,'o-',label=model)
axes[1].set(xlabel='Autonomous horizon (s)',ylabel='Circular phase RMSE (deg)',title='Validation, flight-equal');axes[1].legend()
c=pd.DataFrame(coherence);c=c[(c.split=='validation')&(c.model=='motor_first_order')]
for h in [1,2,3]:axes[2].plot(c.horizon_s,c[f'h{h}_coherence'],'o-',label=f'h={h}')
axes[2].set(xlabel='Autonomous horizon (s)',ylabel='Phase-error harmonic coherence',title='Not a control-effect accuracy metric',ylim=(0,1));axes[2].legend()
fig.savefig(OUT/'phase_evidence.png',dpi=180)
print(json.dumps(stats,indent=2));print(c.to_string(index=False))
