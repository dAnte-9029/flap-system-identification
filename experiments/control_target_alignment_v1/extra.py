"""Influence, failure flags and timestamp availability, without redefining cohorts."""
from common import *
z,n,w,p,coh=read_assets();native=pd.read_csv(TIMING/'results/native_rate_extract.csv.gz');clock=pd.read_csv(OUT/'identity_clock_audit.csv');out=[]
for r in clock.itertuples():
 g=native[native.window_id==r.window_id];t=g.measurement_us.to_numpy();j=np.searchsorted(t,r.origin_us);right=j;left=j if t[j]==r.origin_us else j-1
 assert 0<=left<=right<len(t)
 out.append(dict(window_id=r.window_id,left_measurement_us=int(t[left]),right_measurement_us=int(t[right]),left_publication_us=int(g.publication_us.iloc[left]),right_publication_us=int(g.publication_us.iloc[right]),bracket_available_at_decision=bool(max(g.publication_us.iloc[left],g.publication_us.iloc[right])<=r.decision_us)))
clock.merge(pd.DataFrame(out),on='window_id',validate='one_to_one').to_csv(OUT/'origin_support_availability.csv',index=False)
ids=set(coh.query("cohort=='main' and split=='train'").pair_uid);pp=p[p.pair_uid.isin(ids)];a=pd.read_csv(OUT/'oracle_points.csv.gz');a=a[(a.time_label=='300')&a.pair_uid.isin(ids)&(a.model=='O0')&(a.variable=='p')];dec=pd.read_csv(OUT/'pair_decomposition.csv.gz');dec=dec[(dec.time_label=='300')&(dec.variable=='p')&dec.pair_uid.isin(ids)];leave=[]
for f in sorted(set(pp.flight_a)|set(pp.flight_b)):
 cp=pp[(pp.flight_a!=f)&(pp.flight_b!=f)]
 for method in ['B','D']:
  g=a[(a.method==method)&a.pair_uid.isin(cp.pair_uid)];wm=stats.weights(cp,'flight');ww=g.pair_uid.map(wm).to_numpy();gg=dec[(dec.method==method)&dec.pair_uid.isin(cp.pair_uid)];wd=gg.pair_uid.map(wm).to_numpy();leave.append(dict(removed_flight=f,method=method,**metric(g,ww),origin_rms=np.sqrt(np.dot(wd,gg.origin**2)),future_rms=np.sqrt(np.dot(wd,gg.future**2))))
pd.DataFrame(leave).to_csv(OUT/'leave_one_flight_out.csv',index=False)
models=pd.read_csv(OUT/'model_error_decomposition.csv.gz');models=models[(models.time_label=='300')&models.pair_uid.isin(ids)];fl=[]
for key,g in models.groupby(['budget_epochs','model','seed','method','variable']):
 for f in sorted(set(pp.flight_a)|set(pp.flight_b)):
  u=pp[(pp.flight_a==f)|(pp.flight_b==f)].pair_uid;gg=g[g.pair_uid.isin(u)];active=gg.direction_active
  fl.append(dict(zip(['budget_epochs','model','seed','method','variable'],key),flight_id=f,pairs=len(gg),label_error_rms=np.sqrt(np.mean(gg.label_error**2)),model_error_rms=np.sqrt(np.mean(gg.total_error**2)),active_pairs=int(active.sum()),direction_error_rate=float((~gg.loc[active,'model_correct']).mean()) if active.any() else np.nan))
f=pd.DataFrame(fl);f.to_csv(OUT/'model_label_per_flight.csv',index=False)
a=[]
for key,g in f.groupby(['budget_epochs','model','seed','method','variable']):
 good=g.direction_error_rate.notna();a.append(dict(zip(['budget_epochs','model','seed','method','variable'],key),flights=len(g),corr_label_rms_error_rms=g.label_error_rms.corr(g.model_error_rms),corr_label_rms_direction_error=g.loc[good,'label_error_rms'].corr(g.loc[good,'direction_error_rate'])))
pd.DataFrame(a).to_csv(OUT/'model_label_flight_associations.csv',index=False)
# Near-zero prediction flags do not alter the frozen direction threshold.
d=pd.read_csv(OUT/'all_main_oracle_directions.csv');ab=[]
for key,g in d.groupby(['model','method','variable']):
 active=g.direction_active;ab.append(dict(zip(['model','method','variable'],key),pairs=len(g),active=int(active.sum()),wrong=int(g.wrong.sum()),pred_abs_le_01_in_active=int((active&g.near_zero_prediction).sum()),pred_abs_le_1e8_in_active=int((active&(abs(g.pred)<=1e-8)).sum()),true_near_zero=int((~active).sum())))
pd.DataFrame(ab).to_csv(OUT/'oracle_direction_counts.csv',index=False)
print('Added fixed origin availability, 46 flight influence checks and descriptive model associations')
