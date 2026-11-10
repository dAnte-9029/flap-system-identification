"""Information calculation for explicitly unrestricted closed-loop innovations."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
root=Path('/home/zn/flap-system-identification/.worktrees/pacld-control-identifiability-v1/experiments/phase_anchored_coupled_latent_v1/control_identifiability')
f=pd.read_pickle(root/'features.pkl');x=np.stack(f.X);u=np.stack(f.U);y=np.stack(f.Y)[:,0]
# Endpoint model, observed measured rate increments; no claim of plant consistency.
X=np.column_stack([np.ones(len(f)),x[:,3:6],u[:,4:8]])
s=np.std(X,axis=0);s[0]=1;Z=X/s
b=np.linalg.lstsq(Z,y,rcond=None)[0];e=y-Z@b
rows=[]
for d,g in f.groupby('date'):
 z=Z[g.index];sv=np.linalg.svd(z,compute_uv=False)
 rows.append(dict(date=d,n=len(g),design_rank=int(np.linalg.matrix_rank(z)),gram_condition=float((sv[0]/sv[-1])**2)))
pd.DataFrame(rows).to_csv('experiments/px4_b_v1/information_by_date.csv',index=False)
errors=[]
for delta in [-10.,-1.,1.,10.]:
 b2=b.copy();b2[5]+=delta;e2=e-Z[:,5]*delta
 errors.append(float(np.max(abs((Z@b2+e2)-(Z@b+e)))))
Path('experiments/px4_b_v1/information.json').write_text(json.dumps(dict(model='Delta_p=X theta + innovation_i; X=[1,p,q,r,delta_motor,delta_d,delta_s,delta_rudder]',n=len(f),parameter_count=8,design_rank=int(np.linalg.matrix_rank(Z)),gram_condition=float(np.linalg.cond(Z)**2),ols_rmse=float(np.sqrt(np.mean(e**2))),innovation_profile_information='Exactly zero Schur complement X^T(I-I)X when each innovation is unrestricted; independent of full-rank X',compensation_max_errors=errors,interpretation='Unrestricted innovation demonstrates missing assumptions, not impossibility under an independently justified innovation model'),indent=2))
