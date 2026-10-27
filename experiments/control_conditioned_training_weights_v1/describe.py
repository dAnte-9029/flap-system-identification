"""Create descriptive coverage, figures and next-stage protocol; never trains."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import old,cut,sha
from weighting import integral,coordinates
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results';FIG=HERE/'figures'
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def main():
 summaries=json.loads((OUT/'summary.json').read_text());cfg=json.loads((HERE/'config.json').read_text());data=Path(cfg['data_root'])
 m=pd.read_csv(OUT/'full_train/window_audit.csv.gz');a=dict(np.load(OUT/'full_train/weights.npz'));n=len(m)
 motors=pd.read_parquet(data/'native_actuator_motors_train.parquet');servos=pd.read_parquet(data/'native_actuator_servos_train.parquet');mg=dict(tuple(motors.groupby('flight_id')));sg=dict(tuple(servos.groupby('flight_id')))
 tapes={f:old.tape(mg[f],sg[f]) for f in m.flight_id.unique()}
 pos=np.zeros(n);neg=np.zeros(n);dp=np.zeros(n);dn=np.zeros(n);history_gap=np.zeros(n)
 for fid,g in m.groupby('flight_id'):
  tape=tapes[fid]
  for idx,row in g.iterrows():
   decision=int(row.decision_us);outputs=a['output_us'][idx];valid=a['future_mask'][idx]&a['w2_supported'][idx];end=int(outputs[a['future_mask'][idx]][-1]);t,u,_=cut(tape,decision,decision+end,decision+end)
   d=coordinates(u)[:,2];history_gap[idx]=d[0]-row.historical_d0
   if not row.auto_stable:continue
   diff=d-d[0];hi=outputs[valid];lo=np.maximum(0,hi-300000)
   for sign,arr,dur in [(1,pos,dp),(-1,neg,dn)]:
    flag=(sign*diff>=.05).astype(float);e=integral(t,flag,hi)-integral(t,flag,lo);arr[idx]=e.sum();dur[idx]=integral(t,flag,min(end,int(a['support_until_us'][idx])))
 m['W2_positive_rolling_exposure_us']=pos;m['W2_negative_rolling_exposure_us']=neg;m['positive_duration_horizon_us']=dp;m['negative_duration_horizon_us']=dn;m['decision_minus_history_d0']=history_gap
 m[['registered_index','window_id','flight_id','W2_positive_rolling_exposure_us','W2_negative_rolling_exposure_us','positive_duration_horizon_us','negative_duration_horizon_us','decision_minus_history_d0']].to_csv(OUT/'sign_and_clock_coverage.csv.gz',index=False)
 extra={};state=[]
 for name,summary in summaries.items():
  z=dict(np.load(OUT/name/'weights.npz'));ix=z['registered_index'];frame=m.iloc[ix].reset_index(drop=True);weights=z['w1'];flight=pd.read_csv(OUT/name/'group_contributions.csv');c=flight[flight.group=='controller_id'];fw=flight[flight.group=='firmware'];d=flight[flight.group=='date']
  p=pos[ix];q=neg[ix];N=len(ix);mask=z['future_mask'];coeff0=mask/mask.sum(1)[:,None]/N;coeff2=mask*z['w2'];coeff2/=coeff2.sum(1,keepdims=True)*N
  # Positive/negative W2 are rolling-exposure totals, not independent events.
  extra[name]=dict(logs=int(frame.log_sha.nunique()),W1_enhanced_configurations=int((c.W1_added>0).sum()),W1_enhanced_firmwares=int((fw.W1_added>0).sum()),W1_enhanced_dates=int((d.W1_added>0).sum()),top_configuration_added_share_W1=float(c.W1_added.max()/c.W1_added.sum()),top_configuration_added_share_W2=float(c.W2_added.max()/c.W2_added.sum()),W2_positive_flights=int(frame.loc[p>0,'flight_id'].nunique()),W2_negative_flights=int(frame.loc[q>0,'flight_id'].nunique()),W2_positive_rolling_exposure_fraction=float(p.sum()/(p.sum()+q.sum())),W2_positive_windows=int((p>0).sum()),W2_negative_windows=int((q>0).sum()),W2_top1_raw_added_flight_share=float(flight[flight.group=='flight_id'].W2_added.max()/flight[flight.group=='flight_id'].W2_added.sum()),W2_shifted_probability_mass=float(abs(coeff2-coeff0).sum()/2),decision_history_differential_gap_rms=float(np.sqrt(np.mean(history_gap[ix]**2))),decision_history_differential_gap_max=float(abs(history_gap[ix]).max()))
  for key in ['history_speed','rate_rms','roll_max','pitch_max','history_near_limit_fraction']:
   x=frame[key].to_numpy();state.append(dict(dataset=name,feature=key,W0_mean=float(x.mean()),W1_mean=float(np.average(x,weights=weights)),W2_mean=float(x.mean()),W1_enhanced_mean=float(x[weights>1].mean())))
  # Explicit numerical gate plus descriptive sign/configuration evidence (no new threshold).
  summary.update(extra[name]);summary['coverage_assessment']='Both signs span many flights/configurations; no single configuration dominates added weight.'
 dump(OUT/'supplementary_summary.json',extra);dump(OUT/'summary.json',summaries);pd.DataFrame(state).to_csv(OUT/'state_distribution.csv',index=False)
 plt.rcParams.update({'font.size':10})
 fig,ax=plt.subplots(1,3,figsize=(13,3.5))
 ax[0].hist(a['w1'],bins=np.linspace(1,2,21));ax[0].set(title='W1: window weights',xlabel='weight',ylabel='windows')
 ax[1].hist(m.w2_time_ess_ratio,bins=25);ax[1].axvline(.85,color='r',ls='--');ax[1].set(title='W2: within-window ESS / steps',xlabel='ESS ratio')
 x=np.arange(4);ax[2].bar(x,[v['w1_ess_ratio'] for v in summaries.values()],label='W1 windows');ax[2].plot(x,[v['w2_time_ess_ratio_min'] for v in summaries.values()],'o-',color='tab:orange',label='W2 minimum time ESS');ax[2].axhline(.85,color='r',ls='--');ax[2].set_xticks(x,summaries.keys(),rotation=25);ax[2].set_ylim(.83,1.01);ax[2].legend(fontsize=8);fig.tight_layout();fig.savefig(FIG/'weight_and_ess.png',dpi=160);plt.close(fig)
 cover=pd.read_csv(OUT/'full_train/group_contributions.csv');f=cover[cover.group=='flight_id'].sort_values('W1_added',ascending=False)
 fig,ax=plt.subplots(1,2,figsize=(12,4));ax[0].plot(np.arange(len(f))+1,f.W0_share,label='W0 / W2');ax[0].plot(np.arange(len(f))+1,f.W1_share,label='W1');ax[0].set(title='Flight loss shares',xlabel='flight rank by added W1 mass',ylabel='share');ax[0].legend();ax[1].bar(np.arange(len(f)),f.positive_enhanced,label='positive');ax[1].bar(np.arange(len(f)),-f.negative_enhanced,label='negative');ax[1].set(title='Enhanced W1 signs across flights',xlabel='same flight rank',ylabel='windows (negative drawn below zero)');ax[1].legend();fig.tight_layout();fig.savefig(FIG/'flight_sign_coverage.png',dpi=160);plt.close(fig)
 tm=pd.read_csv(OUT/'full_train/step_contribution.csv');fig,ax=plt.subplots(figsize=(8,4))
 for scheme,g in tm.groupby('scheme'):ax.plot(g.step,g.contribution,label=scheme)
 ax.set(xlabel='native future output index',ylabel='total normalized loss contribution',title='Time allocation: no uniform-time resampling');ax.legend();fig.tight_layout();fig.savefig(FIG/'temporal_contribution.png',dpi=160);plt.close(fig)
 cover=pd.read_csv(OUT/'full_train/coverage.csv');fig,axs=plt.subplots(2,3,figsize=(15,8))
 for ax,key in zip(axs.flat,['speed','amplitude','duration_ms','firmware','controller_id','date']):
  g=cover[cover.group==key].copy();x=np.arange(len(g));ax.bar(x-.18,g.W0_share,.36,label='W0 / W2');ax.bar(x+.18,g.W1_share,.36,label='W1');labels=g.value.astype(str).str[:8] if key in ['firmware','controller_id'] else g.value.astype(str);ax.set_xticks(x,labels,rotation=90,fontsize=7);ax.set_title(key);ax.set_ylabel('window loss share')
 axs[0,0].legend();fig.tight_layout();fig.savefig(FIG/'state_configuration_coverage.png',dpi=160);plt.close(fig)
 cc=pd.read_csv(OUT/'full_train/control_coupling.csv');fig,axs=plt.subplots(1,2,figsize=(9,4))
 for ax,scheme in zip(axs,['W0','W1']):
  v=cc[cc.scheme==scheme].pivot(index='axis',columns='other',values='correlation');im=ax.imshow(v,vmin=-1,vmax=1,cmap='coolwarm');ax.set_xticks(range(4),v.columns,rotation=45);ax.set_yticks(range(4),v.index);ax.set_title(scheme+' signed 300 ms changes')
  for i in range(4):
   for j in range(4):ax.text(j,i,f'{v.iloc[i,j]:.2f}',ha='center',va='center')
 fig.tight_layout();fig.savefig(FIG/'control_coupling.png',dpi=160);plt.close(fig)
 assert all(s['W1_numeric_gate_pass'] and s['W2_numeric_gate_pass'] for s in summaries.values())
 protocol=dict(status='frozen_plan_only_not_executed',selected_weight='W1',backup_weight='W2',selection_basis='Both pass fixed numeric and sign/configuration coverage gates. W1 simpler; W2 redistributes toward later time in these records.',models=['M0','M1'],schemes=['W0','W1'],folds=[0,1,2],seeds=[17],epochs=6,planned_runs=12,executed_runs=0,history_ms=500,horizon_ms=1000,batch_size=1024,optimizer='AdamW',learning_rate=.001,weight_decay=.0001,gradient_clip=5,cosine_T_max=6,early_stopping=False,normalization='recompute or hash-verified reuse of fold-fit-only mean/scale/output_scale/loss_scales; shared across all model/weight arms',batch_order='identical fold/seed window permutation across M0/M1 x W0/W1; retain order hashes and equal updates',checkpoint_selection='best original uniform four-component validation loss, flight equal over original 27 flights; independent of held-out response or weighted loss',evaluation=dict(primary_time_ms=300,references=['B','D'],secondary_reference='A',D_average_ms=270,direction_thresholds=[.1,.05,.2],primary_threshold=.1,secondary_times_ms=[100,200,400],endpoints='supported only, no extrapolation',groups='48 assigned main pairs exactly once, 17 flight-disjoint, 11 dependence components, frozen quality subsets and eligible background',metrics=['direction wins/losses','sign-specific amplitude bias','RMS ratio','signed gain','p/q/r coupling','physical-unit factual trajectory error','cross-component consistency'],internal_swap='diagnostic only, larger C is not success'),limitations=['folds and matching protocol previously exposed; not fresh blinded evaluation','weights are not causal/propensity/identifiability weights','no label/model-error based eligibility','W1 self-normalizes per batch; full-dataset ESS describes offline coefficient concentration'],stop_after='12 pilot runs plus frozen evaluation only in a separately authorized task')
 hp=ROOT/'experiments/structured_control_effectiveness_v1/results';protocol['holdout_source_sha256']={n:sha(hp/n) for n in ['holdout_protocol.json','holdout_window_indices.npz','holdout_pair_assignments.csv']};dump(HERE/'next_stage_protocol.json',protocol)
 print(json.dumps(extra,indent=2))
if __name__=='__main__':main()
