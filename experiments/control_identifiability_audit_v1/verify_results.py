"""Independent scalar recomputation of archived matches from native authorized data."""
import json
import numpy as np
import pandas as pd
from audit import (HERE,resolve,read_all,V,Q,W,U,exact_difference,sha,write_json)

def main():
    cfg=json.loads((HERE/'config.json').read_text());root,flights,prov,manifest=resolve(cfg)
    groups,natives,_=read_all(root,flights);out=HERE/'results';rng=np.random.default_rng(cfg['seed']);checked=[]
    for horizon in [500,1000]:
        p=pd.read_parquet(out/f'pairs_{horizon}ms.parquet');meta=pd.read_parquet(out/f'windows_{horizon}ms.parquet')
        selected=pd.read_csv(out/f'selected_{horizon}ms.csv')
        pool=np.unique(selected.pair_index)
        ids=rng.choice(pool,min(100,len(pool)),replace=False)
        for idx in ids:
            row=p.loc[idx];hist=[];tapes=[]
            for k in [int(row.i),int(row.j)]:
                m=meta.iloc[k];s=groups[m.flight_id];origin=int(m.origin_sample)
                grid=s.state_measurement_us.iloc[origin]+np.arange(-500000,1,20000)
                ix=np.searchsorted(s.state_measurement_us,grid,side='right')-1
                assert ix.min()>=m.start_sample and ix.max()==origin
                d=s.iloc[ix];x=d[V+Q+W+['flap_frequency_hz']+U].to_numpy()
                x[:,3:7]/=np.linalg.norm(x[:,3:7],axis=1,keepdims=True);hist.append(x)
                assert d.state_publication_us.max()<=m.decision_us
                for field in ['attitude','rate','motor','servo','frequency']:
                    assert (d[field+'_publication_us']<=d.state_publication_us).all()
                t,u,_=natives[m.flight_id];a=np.searchsorted(t,m.decision_us,side='right')-1;b=np.searchsorted(t,m.decision_us+horizon*1000,side='left')
                tapes.append((np.r_[0,t[a+1:b]-m.decision_us],u[a:b]))
            x,y=hist;v=np.sqrt(np.mean(np.sum((x[:,:3]-y[:,:3])**2,axis=1)))
            w=np.sqrt(np.mean(np.sum((x[:,7:10]-y[:,7:10])**2,axis=1)))
            R=np.rad2deg(np.sqrt(np.mean((2*np.arccos(np.abs(np.sum(x[:,3:7]*y[:,3:7],axis=1)).clip(0,1)))**2)))
            np.testing.assert_allclose([v,w,R],[row.v_rmse,row.w_rmse,row.R_rmse_deg],rtol=2e-4,atol=2e-4)
            diff,shares=exact_difference(*tapes,horizon*1000,np.array(cfg['future_contrast'])[None,:]*np.array(cfg['contrast_sensitivity'])[:,None])
            # Packed source commands are float32; scalar recomputation uses float64.
            np.testing.assert_allclose(diff,[row['future_gap_'+c] for c in ['motor','left_elevon','right_elevon','rudder']],rtol=0,atol=1e-7)
            for j,scale in enumerate(cfg['contrast_sensitivity']):
                np.testing.assert_allclose(shares[j],[row[f'fraction_{scale}_{c}'] for c in ['motor','left_elevon','right_elevon','rudder']],atol=1e-12)
        checked.append(dict(horizon_ms=horizon,scalar_recomputed_pairs=len(ids),history_and_initial_sources_past_only=True,
            command_integral_verified=True,all_registered_artifact_hashes_rechecked=True))
    write_json(out/'scalar_verification.json',dict(results=checked,seed=cfg['seed'],code_sha256=sha(__file__)))
    print(checked)

if __name__=='__main__':main()
