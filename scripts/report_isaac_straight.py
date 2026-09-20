"""Report straight-flight results at paired common horizons, retaining failures."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args();p=args.root
    s=pd.read_csv(p/'summary.csv');t=pd.read_csv(p/'traces.csv');rows=[]
    for cid,g in t.groupby('case'):
        end=g.groupby('mode').time_s.max().min()
        for mode,h in g[g.time_s<=end].groupby('mode'):
            rows.append(dict(case=int(cid),mode=mode,common_duration_s=float(end),**{c:float(np.sqrt(np.mean(h[c]**2))) for c in ['height_error_m','cross_track_m','speed_error_m_s']},surface_limit_fraction=float(h.surface_limit.mean()),motor_limit_fraction=float(h.motor_limit.mean())))
    r=pd.DataFrame(rows);r.to_csv(p/'paired_common_duration.csv',index=False)
    report=dict(runs=len(s),completed_by_mode=s.groupby('mode').termination.apply(lambda x:int((x=='completed').sum())).to_dict(),duration_s=s.groupby('mode').duration_s.agg(['min','median','max']).to_dict(),termination_counts={str(k):int(v) for k,v in s.groupby(['mode','termination']).size().items()},common_horizon_equal_case_mean=r.groupby('mode').mean(numeric_only=True).to_dict())
    (p/'aggregate.json').write_text(json.dumps(report,indent=2))
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for mode,g in t.groupby('mode'):
        for cid,h in g.groupby('case'):
            for ax,col in zip(axs.flat,['cross_track_m','height_error_m','speed_error_m_s','left']):
                ax.plot(h.time_s,h[col],color='tab:blue' if mode=='closed' else 'tab:orange',alpha=.4,label=mode if cid==0 else None);ax.set(xlabel='Time (s)',ylabel=col);ax.grid(alpha=.2)
    axs[0,0].legend();fig.suptitle('Straight-line hold: traces stop at diagnostic limits');fig.tight_layout();fig.savefig(p/'straight_flight.png',dpi=160)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
