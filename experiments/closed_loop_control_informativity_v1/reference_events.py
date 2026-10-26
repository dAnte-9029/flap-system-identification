"""Control- and mission-defined event association; not delay or IV estimation."""
from extract import *
def main():
 m=pd.read_csv(RES/'windows.csv.gz');a=dict(np.load(RES/'diagnostic_arrays.npz'));out=[]
 for row in m[m.nonoverlap].itertuples():
  i=row.Index;wp=a['diagnostic_reference'][i,:,3:6];valid=np.isfinite(wp).all(1);change=(wp[1:]!=wp[:-1]).any(1)&valid[1:]&valid[:-1]
  for k in np.where(change)[0]+1:
   if k<5 or k+5>=len(wp):continue
   d=a['control'][i,:,2];p=a['motion'][i,:,0];sp=a['diagnostic_reference'][i,:,0]
   out.append(dict(split=row.split,flight_id=row.flight_id,window_id=row.window_id,reference_detection_us=int(row.decision_us+a['grid_us'][k]),detection_quantization_us=20000,delta_d_before100=d[k]-d[k-5],delta_d_after100=d[k+5]-d[k],delta_p_before100=p[k]-p[k-5],delta_p_after100=p[k+5]-p[k],delta_rate_sp_before100=sp[k]-sp[k-5],delta_rate_sp_after100=sp[k+5]-sp[k]))
 pd.DataFrame(out).to_csv(RES/'mission_event_timing.csv',index=False)
 pd.DataFrame(out).groupby('split').agg(events=('flight_id','size'),flights=('flight_id','nunique'),mean_abs_d_before=('delta_d_before100',lambda x:abs(x).mean()),mean_abs_d_after=('delta_d_after100',lambda x:abs(x).mean()),mean_abs_p_before=('delta_p_before100',lambda x:abs(x).mean()),mean_abs_p_after=('delta_p_after100',lambda x:abs(x).mean())).reset_index().to_csv(RES/'mission_event_summary.csv',index=False)
 # PSD summaries first average intervals within flight, then flights, never concatenate boundaries.
 ps=pd.read_csv(RES/'native_psd.csv.gz');out=[]
 for (split,ch),g in ps.groupby(['split','coordinate']):
  spec=g.groupby(['flight_id','hz']).power.mean().groupby('hz').mean();freq=spec.index.to_numpy();power=spec.to_numpy();total=np.sum(power)
  out.append(dict(split=split,coordinate=ch,flights=g.flight_id.nunique(),peak_nonzero_hz=float(freq[1:][np.argmax(power[1:])]),fraction_0p25_2p5=float(power[(freq>=.25)&(freq<2.5)].sum()/total),fraction_2p5_5p5=float(power[(freq>=2.5)&(freq<=5.5)].sum()/total)))
 pd.DataFrame(out).to_csv(RES/'native_spectral_bands.csv',index=False)
if __name__=='__main__':main()
