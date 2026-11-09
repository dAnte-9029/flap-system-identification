"""Focused numerical and extracted-support checks."""
import numpy as np
import pandas as pd
from diagnose_pacld_a2 import Ridge,GroupKFold,transform,OUT
rng=np.random.default_rng(3);x=rng.normal(size=(80,5));y=rng.normal(size=(80,3));m=Ridge(10,False).fit(x,y)
assert np.allclose(m.coef_.T,np.linalg.solve(x.T@x+10*np.eye(5),x.T@y))
g=np.repeat(np.arange(8),10)
for tr,te in GroupKFold(5).split(x,y,g):assert not set(g[tr])&set(g[te])
assert np.allclose(transform(np.array([[.4,-.2,.6,.1]])),[[.4,.4,.2,.1]])
df=pd.read_pickle(OUT/'features.pkl')
assert df.flight.nunique()==114
assert not df.duplicated(['flight','t']).any()
assert all((np.diff(a.t)>=1000000).all() for _,a in df.groupby('flight'))
assert np.isfinite(np.stack(df.X)).all() and np.isfinite(np.stack(df.Y)).all()
assert abs(np.stack(df.offset)).max()<=30000
print('PASS: ridge, grouped folds, channel sign, 114 flights, unique times, 1s spacing, finite values, native target tolerance')
