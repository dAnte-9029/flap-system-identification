"""Train-only finite-error profile; no physical noise bound is inferred from residuals."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path('/home/zn/flap-system-identification')
OUT=Path(__file__).resolve().parents[1]/'experiments/px4_partial_identifiability_c_v1'
SHARED=ROOT/'.worktrees/px4-identifiability-contract-v1/experiments/px4_identifiability_v1'
BASE='experiments/phase_anchored_coupled_latent_v1'
F=ROOT/'.worktrees/pacld-control-identifiability-v1'/BASE/'control_identifiability/features.pkl'
W=ROOT/'.worktrees/pacld-control-residual-moment-v1'/BASE/'control_residual_moment_v1/windows.csv'

def rms(a,w):return float(np.sqrt(w@np.square(a)))
def weights(df):
    return (1/(df.date.nunique()*df.groupby('date').flight.transform('nunique')*df.groupby(['date','flight']).flight.transform('size'))).to_numpy()
def profile(df,Z,y):
    w=weights(df); sw=np.sqrt(w)
    G=Z.T@(w[:,None]*Z); eig=np.linalg.eigvalsh(G)
    b=np.linalg.lstsq(Z*sw[:,None],y*sw,rcond=None)[0]; e=y-Z@b
    s0=rms(y,w);smin=rms(e,w);half=float(np.sqrt(max(0,s0*s0-smin*smin)*np.ones(4)@np.linalg.pinv(G)@np.ones(4)))
    # The sigma=s0 set is illustrative only: no evidence warrants this bound.
    return dict(n=len(df),dates=df.date.nunique(),flights=df.flight.nunique(),b_observational=b.tolist(),constant_waveform_gain=float(b.sum()),gram_eigenvalues=eig.tolist(),rank=int(np.linalg.matrix_rank(G)),condition_number=float(np.linalg.cond(G)),minimum_rms=smin,zero_gain_rms=s0,sign_reversed_rms=rms(y+Z@b,w),conditional_sigma_equals_zero_gain_rms_interval=[float(b.sum()-half),float(b.sum()+half)],data_only_gain_set='R',data_only_kernel_set='R^4',robust_signed_action_pairs=0)

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    meta=json.loads((SHARED/'train_flights.json').read_text()); assert all(m['split']=='train' for m in meta)
    firm=json.loads((SHARED/'firmware_train.json').read_text());assert all(m['match'] and m['local_git_object'] for m in firm)
    df=pd.read_pickle(F).merge(pd.read_csv(W).drop(columns=['date','mode','Y']),on=['flight','t'],validate='one_to_one')
    assert set(df.flight)=={m['flight_id'] for m in meta}; assert len(df)==11221
    Y=np.stack(df.Y)[:,0]; np.testing.assert_allclose(Y,df.E_increment+df.Y_hat,atol=1e-12)
    servo=pd.read_parquet(ROOT/'dataset/velocity_rate_multistep_v1/native_actuator_servos_train.parquet')
    Z=np.zeros((len(df),4));max_gap=0;parity=[]
    for fid,g in df.groupby('flight',sort=False):
        s=servo[servo.flight_id==fid].sort_values('timestamp_us'); ts=s.timestamp_us.to_numpy();d=(s['control[1]'].to_numpy(float)-s['control[0]'].to_numpy(float))/2
        for i,r in g.iterrows():
            t=int(r.t); end=t+200000+int(r.offset[0]);assert end>t+150000
            j=np.searchsorted(ts,t,side='left')-1; assert j>=0
            base=d[j]; assert t-ts[j]<=160000 and np.max(abs(d))<=1.000001
            for k,(lo,hi) in enumerate(zip([t,t+50000,t+100000,t+150000],[t+50000,t+100000,t+150000,end])):
                edges=np.r_[lo,ts[(ts>lo)&(ts<hi)],hi];idx=np.searchsorted(ts,edges[:-1],side='right')-1
                assert (idx>=0).all() and s.iloc[idx].in_reviewed_flight.all()
                assert (edges[1:]-ts[idx]<=160000).all()
                max_gap=max(max_gap,int(np.max(np.diff(edges))))
                Z[i,k]=np.sum((d[idx]-base)*np.diff(edges))/(hi-lo)
            parity.append(abs((Z[i,0]+Z[i,1])/2-r.U[5]))
    assert max(parity)<1e-12
    y=Y-df.M0_prediction.to_numpy()
    allp=profile(df,Z,y);rows=[]
    for field in ['date','mode','speed_group','phase_group']:
        for group,gg in df.groupby(field):
            p=profile(gg,Z[gg.index],y[gg.index]); rows.append(dict(group_type=field,group=str(group),**p))
    (OUT/'group_profiles.json').write_text(json.dumps(rows,indent=2))
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,list)} for r in rows]).to_csv(OUT/'group_summary.csv',index=False)
    # Exact observational equivalence: measured endpoints unchanged under kernel changes.
    b=np.array(allp['b_observational']); equivalent=[]
    for name,beta in [('zero',b*0),('observational',b),('opposite',-b)]:
        e=y-Z@beta;err=float(np.max(abs(df.M0_prediction+Z@beta+e-Y)))
        equivalent.append(dict(kernel=name,b=beta.tolist(),gain=float(beta.sum()),residual_rms=rms(e,weights(df)),endpoint_reconstruction_max_error=err))
        assert err<1e-12
    # Dynamics null in unbounded disturbance model: [Z I] [v; -Zv] = 0.
    null_error=float(np.max(abs(Z@b+(-Z@b))));assert null_error==0
    action_witnesses=[]
    for k,v in enumerate(np.vstack([np.eye(4),np.ones(4)])):
        plus=v/(v@v); minus=-plus
        action_witnesses.append(dict(direction=k,positive_response=float(v@plus),negative_response=float(v@minus),positive_exact_fit_error=float(np.max(abs(Z@plus+(y-Z@plus)-y))),negative_exact_fit_error=float(np.max(abs(Z@minus+(y-Z@minus)-y)))))
    result=dict(mpc_signed_response_witnesses=action_witnesses,baseline='9c15fffd24bfcacc7c57cd980cc149098cb83f82',split='train',profile=allp,firmware_versions=len(set(m['logged'] for m in firm)),raw_control_parity_max_error=max(parity),largest_bin_hold_us=max_gap,native_horizon_range_s=[float(df.query_s.min()),float(df.query_s.max())],measurement_discrepancies={n:dict(rms=rms(df[n].to_numpy(),weights(df)),q95_abs=float(np.quantile(abs(df[n]),.95)),meaning='prediction/belief discrepancy, NOT independently measured sensor error or hard bound') for n in ['E_initial','E_increment','E_endpoint']},observational_equivalence=equivalent,augmented_design_null_error=null_error,verdict='NO-GO for data-proven finite plant response bounds or robust MPC ranking; GO for conditional sensitivity profiling',physical_constraints_used=['normalized command limits; no torque/actual servo position bound inferred','causal horizon and native sample timing; no future measured states used as predictors'],extra_assumptions_for_finite_set=['fixed nuisance m(H) and local four-bin linear kernel','known finite RMS bound on combined omitted dynamics, disturbances, measurement and nuisance error'],no_neural_training=True)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
