"""Plot paired endpoint gains and the independently observed higher-band differences."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
data=json.loads((HERE/"results.json").read_text())
fig,axes=plt.subplots(1,2,figsize=(9.5,3.8),layout="constrained")
colors={"corrected":"#1d5274","global":"#b96b25","sham":"#24856d"}
def point(axis, reference, values, y, height, label):
    v=values["gain"];lo=values["lo"];hi=values["hi"]
    axis.errorbar(v,y+height,xerr=[[v-lo],[hi-v]],fmt="o",color=colors[reference],label=label,
                  ms=4,capsize=2,lw=1.4)
leg={"corrected":"Prior first-harmonic predictor","global":"Time-only circular calibration",
     "sham":"Training conditions shuffled"}
for axis,labels in zip(axes,[["0.5 s","1 s","2 s"],["Native observations","2nd- and 3rd-order band"]]):
    axis.axvline(0,color=".55",lw=1,ls="--")
    axis.grid(axis="x",alpha=.18)
    axis.set_yticks(range(len(labels)),labels)
    axis.invert_yaxis()
    axis.set_ylim(len(labels)-.65,-.35)
    axis.set_xlabel("RMSE reduction (rad/s)")
offsets=[-.16,0,.16]
for j,m in enumerate(("w05","w1","w2")):
    items=data["endpoints"]["contrasts"]
    for (a,label),delta in zip(leg.items(),offsets):
        record=next(x for x in items if x["metric"]==m and x["reference"]==a)
        point(axes[0],a,record,j,delta,label if j==0 else None)
for j,m in enumerate(("measurement","higher")):
    items=data["native"]["1_early"]["contrasts"]
    for (a,label),delta in zip(leg.items(),offsets):
        record=next(x for x in items if x["metric"]==m and x["reference"]==a)
        point(axes[1],a,record,j,delta,label if j==0 else None)
axes[0].set_title("Angular-rate endpoints, 39 flights",fontsize=11)
axes[1].set_title("Early complete wingbeats, 35 flights",fontsize=11)
fig.legend(*axes[0].get_legend_handles_labels(),loc="outside lower center",ncol=3,frameon=False,fontsize=8)
fig.savefig(HERE/"paired_angular_rate.png",dpi=180)
