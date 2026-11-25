"""Display one prespecified development window without choosing it by network error."""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from prepare_reference import HERE


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('full');cli.add_argument('control')
    cli.add_argument('--output',default='component_example')
    cli.add_argument('--full-label',default='翼拍频率载波')
    a=cli.parse_args()
    with np.load(HERE/'reference.npz') as f: ref=dict(f)
    with np.load(HERE/f'{a.full}_points.npz') as f: full=dict(f)
    with np.load(HERE/f'{a.control}_points.npz') as f: ctrl=dict(f)
    for fi in np.unique(ref['flight_index']):
        eligible=[]
        for wi in np.unique(ref['window_idx'][ref['flight_index']==fi]):
            n=[len(np.unique(ref['block'][(ref['window_idx']==wi)&(ref['ch']==ch)])) for ch in (0,1)]
            if min(n)>=4: eligible.append(int(wi))
        if eligible: break
    wi=eligible[len(eligible)//2]
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Noto Sans CJK SC','Noto Sans CJK JP'],
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(11,5.7),sharex=True,layout='constrained')
    colors=['#202d38','#147c73','#be742b']
    for ch,(axis,label) in enumerate([(2,'起点机体系下向速度 (m/s)'),(1,'体轴俯仰角速度 (rad/s)')]):
        sel=(ref['window_idx']==wi)&(ref['ch']==ch)
        ix=np.flatnonzero(sel)
        # Avoid joining separate accepted intervals across missing beats.
        gap=np.flatnonzero(np.diff(ref['time'][ix])>.045)+1
        for segment in np.split(ix,gap):
            t=ref['time'][segment]
            for col,field in enumerate(['slow','fast']):
                ax=axes[ch,col]
                for color,v,style in [(colors[0],ref[field],'-'),(colors[1],full[field+'_abs'],'-'),
                                      (colors[2],ctrl[field+'_abs'],'--')]:
                    ax.plot(t,v[segment,axis],color=color,linestyle=style,lw=1.3)
        axes[ch,0].set_ylabel(label)
    for ax,title in zip(axes[0],['慢分量','周期分量']):
        ax.set_title(title,loc='left')
    for row in axes:
        for ax in row:
            ax.grid(axis='y',alpha=.2)
    for ax in axes[-1]:
        ax.set_xlabel('从预测起点起的时间 (s)')
    fig.legend([Line2D([],[],color=c,ls=s) for c,s in zip(colors,['-','-','--'])],
               ['测量参照（目标拍留出）',a.full_label,'同容量低频载波对照'],
               loc='outside lower center',ncols=3,frameon=False)
    fig.savefig(HERE/f'{a.output}.png',dpi=165)
    (HERE/f'{a.output}.json').write_text(json.dumps(dict(flight=ref['flight_ids'][fi],cache_window_index=wi,
            channels=['origin FRD downward velocity','body FRD pitch angular rate']),indent=2)+'\n')


if __name__=='__main__':
    main()
