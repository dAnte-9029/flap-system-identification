"""Descriptive matching coverage and initial-alignment sensitivity; no cohort changes."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate import OUT,CHANNELS,write_json

def main():
    (OUT/'figures').mkdir(parents=True,exist_ok=True)
    pairs=pd.read_parquet(OUT/'cohorts.parquet').query('cohort=="signal" and split=="validation"')
    r=pd.read_parquet(OUT/'response_metrics.parquet').query('cohort=="signal" and split=="validation" and point_ms==horizon_ms')
    rows=[];coverage=[]
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ii,h in enumerate([500,1000]):
        p=pairs[pairs.horizon_ms==h];z=np.load(OUT/f'predictions_{h}ms.npz')
        ix=np.array([np.flatnonzero(z['pair_index']==n)[0] for n in p.pair_index]);a=z['i'][ix];b=z['j'][ix]
        w=p.flight_weight.to_numpy();w/=w.sum()
        meta=pd.read_parquet(OUT/f'windows_{h}ms.parquet');win=meta.iloc[np.unique(np.r_[a,b])]
        coverage.append(dict(horizon_ms=h,pairs=len(p),flights=len(set(p.flight_a)|set(p.flight_b)),components=p.component.nunique(),
            nav_states=win.nav_state.value_counts().to_dict(),review_pairs=int(p.any_review.sum()),high_dynamic_pairs=int(p.high_dynamics.sum()),
            independent_channel_pairs={c:int(p['isolated_'+c].sum()) for c in CHANNELS},
            match_ranges={c:[float(p[c].min()),float(p[c].median()),float(p[c].max())] for c in ['score','v_rmse','R_rmse_deg','w_rmse','f_rmse']},
            history_ranges={c:[float(win[c].min()),float(win[c].median()),float(win[c].max())] for c in ['history_speed','history_rate','history_frequency']},
            control_rms_ranges={c:[float(p['future_gap_'+c].min()),float(p['future_gap_'+c].median()),float(p['future_gap_'+c].max())] for c in CHANNELS},
            max_pair_time_difference_ms=float(1000*np.abs(z['actual_time_s'][a]-z['actual_time_s'][b]).max()),
            max_nominal_time_error_ms=float(1000*np.abs(z['actual_time_s'][np.r_[a,b]]-z['point_ms'][None]/1000).max())))
        for jj,(metric,cols) in enumerate([('velocity',slice(3,6)),('rate',slice(10,13))]):
            # Alternative anchoring is explicitly a sensitivity diagnostic, not a replacement label.
            for seed in [17,29,43]:
                for model in ['GRU-Control','GRU-NoControl']:
                    pred=z[f'{model}_{seed}'][:,-1,cols]-z['initial'][:,cols]
                    for anchor in ['native_target_initial','causal_input_initial']:
                        zero=z['truth_initial'] if anchor=='native_target_initial' else z['initial']
                        truth=z['truth'][:,-1,cols]-zero[:,cols]
                        error=(pred[a]-pred[b])-(truth[a]-truth[b])
                        rows.append(dict(horizon_ms=h,metric=metric,model=model,seed=seed,truth_anchor=anchor,
                            response_rmse=float(np.sqrt(np.sum(w*np.sum(error**2,axis=1))))))
            g=r[(r.horizon_ms==h)&(r.metric==metric)]
            means=g.groupby(['pair_uid','model']).agg(mse=('squared_error','mean'),score=('score','first')).reset_index()
            ax=axes[ii,jj]
            for model in ['GRU-Control','GRU-NoControl']:
                d=means[means.model==model];ax.scatter(d.score,np.sqrt(d.mse),label=model,alpha=.8)
            ax.set(xlabel='Archived historical match score (lower is closer)',ylabel='Response error (m/s)' if metric=='velocity' else 'Response error (rad/s)',title=f'{h} ms: {metric}, n={len(p)}')
            ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'figures/matching_quality.png',dpi=150);plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT/'initial_alignment_sensitivity.csv',index=False)
    write_json(OUT/'coverage_diagnostics.json',coverage)
    print('Matching coverage and initial-alignment diagnostics saved; original cohorts unchanged')

if __name__=='__main__':main()
