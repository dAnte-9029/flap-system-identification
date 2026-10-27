from common import *

def main():
 t=pd.read_csv(OUT/'trajectory_summary.csv');f=pd.read_csv(OUT/'factual_summary.csv');d=pd.read_csv(OUT/'paired_summary.csv');c=pd.read_csv(OUT/'prespecified_cases_300.csv');i=pd.read_csv(OUT/'internal_summary.csv')
 def show(name,x):print('\n'+name+'\n'+x.to_string(index=False))
 show('Trajectory: common prefix, flight equal',t[t.support=='common_B_prefix'][['model','horizon_ms','reference','position_m_rms','velocity_mps_rms','attitude_rad_rms','p_rms','q_rms','r_rms']])
 show('Response: main p, flight equal',f[(f.scope=='all_heldout')&(f.cohort=='main')&(f.variable=='p')&(f.time_label=='300')&(f.weighting=='flight')][['model','method','active_pairs','direction_accuracy','scale_ratio','signed_gain','positive_amplitude_bias','negative_amplitude_bias']])
 show('Transitions',d[(d.scope=='all_heldout')&(d.variable=='p')&(d.weighting=='pair')][['method','gained_pair_seed_rows','lost_pair_seed_rows','mean_amplitude_error_change']])
 show('Components',d[(d.scope.str.startswith('C'))&(d.variable=='p')&(d.weighting=='pair')&d.method.isin(['B','D'])][['scope','method','active_pairs','gained_pair_seed_rows','lost_pair_seed_rows','mean_amplitude_error_change']])
 show('Cases',c[['pair_uid','method','real','pred_A','pred','gained','lost']])
 show('Internal',i[(i.scope=='all_heldout')&(i.cohort=='main')&(i.variable=='p')&(i.time_label=='300')&(i.weighting=='flight')])
 show('Axes',f[(f.scope=='all_heldout')&(f.cohort=='main')&f.variable.isin(['q','r'])&(f.time_label=='300')&(f.weighting=='flight')&f.method.isin(['B','D'])][['model','method','variable','active_pairs','direction_accuracy','scale_ratio','signed_gain']])
if __name__=='__main__':main()
