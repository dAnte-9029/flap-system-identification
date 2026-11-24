"""Paper-scale error curves, generated only from the common 3 s validation cohort."""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from workspace import RUN,BASELINES


def plot():
    a=pd.read_csv(RUN/'extrapolated3_comparison.csv')
    b=pd.read_csv(RUN/'trained3_comparison.csv')
    rep=a.representative.iloc[0]
    assert rep==b.representative.iloc[0] and a.windows.equals(b.windows)
    metrics=['position_m','velocity_m_s','attitude_deg','rate_rad_s']
    labels=['Position (m)','Velocity (m/s)','Attitude (deg)','Angular rate (rad/s)']
    colors=['#777777','#947eba','#008b85','#cc7733','#14395a']
    marks=['+','x','s','^','o']
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(2,4,figsize=(7.15,3.9),sharex=True,sharey='col')
    for i,(tab,name) in enumerate([(a,'2 s training'),(b,'3 s training')]):
        for j,metric in enumerate(metrics):
            ax=axes[i,j]
            rows=tab[tab.metric==metric].sort_values('horizon_s')
            for model,color,marker in zip([*BASELINES,rep],colors,marks):
                ax.plot(rows.horizon_s,rows[model],color=color,marker=marker,markersize=3.4,
                        linewidth=1.65 if model==rep else .95,
                        label='Proposed' if model==rep else ('Transformer' if model=='transformer' else model.upper()))
            ax.set_xticks([.5,1,2,3],['0.5','1','2','3'])
            ax.grid(axis='y',lw=.5,alpha=.3)
            if not i: ax.set_title(labels[j],fontsize=9)
            if not j: ax.set_ylabel(name+'\nRMSE')
            if i: ax.set_xlabel('Horizon (s)')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=5,frameon=False)
    fig.tight_layout(rect=(0,0,1,.92),w_pad=.65,h_pad=.75)
    for suffix in ('pdf','png'):
        fig.savefig(RUN/f'rmse_horizons.{suffix}',dpi=220,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__': plot()
