"""Descriptive flight-paired bootstrap on the historical-anchor support."""
from pathlib import Path
import numpy as np
import pandas as pd

here = Path(__file__).resolve().parent
data = pd.read_csv(here / 'flight_mse.csv').query("horizon == 'all' and subset == 'anchor'")
rows = []
for baseline, candidate in [('linear_motion','linear_motion_cmd'),('last_beat','anchored_B_s17'),
                            ('last_beat_vs_measured','anchored_B_s17_vs_measured'),
                            ('last_beat_evolution','anchored_B_s17_evolution')]:
    values = data[[baseline,candidate]].dropna().to_numpy()
    indices = np.random.default_rng(17).integers(len(values),size=(10000,len(values)))
    sampled = np.sqrt(values[indices].mean(axis=1))
    lower,upper = np.quantile(sampled[:,0]-sampled[:,1],[.025,.975])
    rows.append(dict(baseline=baseline,candidate=candidate,flights=len(values),
                     improved=int((values[:,0]>values[:,1]).sum()),lower=lower,upper=upper))
pd.DataFrame(rows).to_csv(here / 'paired.csv',index=False)
