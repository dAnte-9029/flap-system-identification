from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
raw = pd.read_csv(root / 'per_flight.csv')
fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.5), sharey=True)
for ax, ch in zip(axs, ('q', 'vD')):
    subset = raw.query("channel == @ch and stride == 1 and not local_linear")
    for case, label in [('varying_amplitude','Amplitude ramp'), ('transient','Nonperiodic pulse')]:
        pivot = subset.query("case == @case").pivot(index='flight', columns='beats', values='nrmse')
        x = pivot.columns
        ax.fill_between(x, *np.percentile(pivot,[25,75],axis=0)*100, alpha=.18)
        ax.plot(x,pivot.mean()*100,'o-',label=label)
    ax.set_xticks([5,11,51])
    ax.set_xlabel('Window (wingbeats)')
    ax.set_title('Pitch rate' if ch=='q' else 'Down velocity')
axs[0].set_ylabel('Allocation error (% of injected RMS)')
axs[1].legend()
fig.tight_layout()
fig.savefig(root/'injection.png', dpi=170)
