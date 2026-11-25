"""Show the paired accuracy effect with the matched architecture as reference."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

here=Path(__file__).resolve().parent
out=json.loads((here/'results.json').read_text())['original_protocol']
plt.rcParams.update({'font.sans-serif':['Noto Sans CJK SC','Noto Sans CJK JP'],'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
for ax,prefix,title,unit in zip(axes,['v','w'],['三轴速度','三轴角速度'],['m/s','rad/s']):
    for delta,key,color,label in [(-.07,'corrected_vs_strong','#5c6872','对原强基线'),(.07,'corrected_vs_matched','#007769','对同容量慢时基')]:
        rows=[out['comparisons'][key][prefix+t] for t in ('05','1','2')]
        point=np.array([r['gain'] for r in rows])
        interval=np.array([r['interval95'] for r in rows]).T
        ax.errorbar(np.arange(3)+delta,point,yerr=np.stack((point-interval[0],interval[1]-point)),
                    fmt='o',color=color,capsize=3,ms=5,label=label)
    ax.axhline(0,ls='--',lw=1,color='#89929b')
    ax.set(xlabel='预测时域（秒）',ylabel=f'RMSE 减少量（{unit}）',title=title,xticks=np.arange(3),xticklabels=['0.5','1','2'])
    ax.grid(axis='y',alpha=.16)
fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncols=2,frameon=False)
fig.savefig(here/'paired_accuracy.png',dpi=170,facecolor='white')
