import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from common_paths import HERE
df=pd.read_csv(HERE/"native_rmse.csv")
fig,axes=plt.subplots(1,2,figsize=(10,3.9),constrained_layout=True)
for ax,ch,label in zip(axes,(0,1),("Velocity (m/s)","Body rate (rad/s)")):
    data=df.query("ch==@ch and period=='0-.5' and seed==17").pivot(index="band",columns="model",values="vector")
    data.loc[["trend","h1","h2","h3","residual"],["transformer","tcn","bare","v2","wingbeat"]].plot.bar(ax=ax,width=.8,rot=0,legend=False)
    ax.set(xlabel="Complete-beat error component",ylabel=label,title="First 0.5 s")
    ax.spines[["right","top"]].set_visible(False)
axes[1].legend(["Transformer","TCN","Bare recurrent","Slow–Fast V2","WingbeatPredictor"],fontsize=8)
fig.savefig(HERE/"early_components.png",dpi=155)
