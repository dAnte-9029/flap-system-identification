"""Flight-balanced native full-beat weighted joint and orthogonal errors."""
import json
import numpy as np
import pandas as pd
from common_paths import HERE, OUT, ROUND, DIAG, TMP
SEEDS = (17,29,43)
BANDS = ["trend","h1","h2","h3"]

def native():
    ref=dict(np.load(DIAG/"reference.npz"))
    models={("persistence",17):ref["origin"]}
    for name in ("transformer","tcn","v2"):
        models[name,17]=np.load(TMP/f"{name}.npy")
    for name in ("transformer_shift","tcn_shift"):
        models[name,17]=np.load(TMP/f"{name}.npy")
    for seed in SEEDS:
        for name in ("bare","wingbeat"):
            models[name,seed]=np.load(TMP/f"{name}{seed}.npy")
    rows=[]
    for ids in np.split(np.arange(len(ref["time"])),np.flatnonzero(np.diff(ref["block"]))+1):
        phase,t,w=(ref[k][ids] for k in ("phase","time","weight"))
        cols=[np.ones_like(t),(t-t.mean())/np.ptp(t)]
        for k in (1,2,3):
            cols.extend([np.cos(k*phase),np.sin(k*phase)])
        mat=np.column_stack(cols)*np.sqrt(w[:,None])
        if len(w)<10 or np.linalg.cond(mat)>=100:
            continue
        q,r=np.linalg.qr(mat)
        lead=w@t
        i=ids[0]
        period="0-.5" if lead<=.5 else (".5-1" if lead<=1 else "1-2")
        for (name,seed),pred in models.items():
            err=(pred[ids]-ref["data"][ids])*np.sqrt(w[:,None])
            coords=q.T@err
            coef=np.linalg.solve(r,coords)
            residual=np.sum((err-q@coords)**2,0)
            bands=(coords**2).reshape(4,2,3).sum(1)
            total=(err**2).sum(0)
            np.testing.assert_allclose(bands.sum(0)+residual,total,atol=1e-9)
            metrics=dict(zip(BANDS,bands))
            for k,b in enumerate(BANDS):
                sl=slice(2*k,2*k+2)
                metrics["joint_"+b]=((mat[:,sl]@coef[sl])**2).sum(0)
            metrics.update(residual=residual,total=total)
            for band,e in metrics.items():
                rows.append([name,seed,ref["flight_ids"][ref["flight_index"][i]],ref["window_idx"][i],
                             ref["ch"][i],period,band,*e])
    df=pd.DataFrame(rows,columns=["model","seed","flight","window","ch","period","band","x","y","z"])
    axes=["x","y","z"]; keys=["model","seed","flight","ch","period","band"]
    flights=df.groupby(keys+["window"])[axes].mean().groupby(keys).mean().reset_index()
    flights["vector"]=flights[axes].sum(axis=1)
    flights.to_csv(HERE/"native_flight_mse.csv",index=False)
    summary=flights.groupby(["model","seed","ch","period","band"])[axes+["vector"]].mean()**.5
    summary.to_csv(HERE/"native_rmse.csv")
    # Seed-17 pairs are available for all models. Keep same flight draws for each pair.
    paired=[]
    for (ch,period,band),g in flights.query("seed==17").groupby(["ch","period","band"]):
        mat=g.pivot(index="flight",columns="model",values="vector")
        assert not mat.isna().any().any()
        draw=np.random.default_rng(17).integers(len(mat),size=(2000,len(mat)))
        for model in models:
            name,seed=model
            if seed!=17 or name=="wingbeat":continue
            a,b=mat[name].to_numpy(),mat.wingbeat.to_numpy()
            delta=np.sqrt(a[draw].mean(1))-np.sqrt(b[draw].mean(1))
            paired.append([ch,period,band,name,len(a),(a>b).sum(),np.sqrt(a.mean())-np.sqrt(b.mean()),*np.quantile(delta,[.025,.975])])
    pd.DataFrame(paired,columns=["ch","period","band","reference","flights","wins_full","gain_full","lo","hi"]).to_csv(HERE/"paired.csv",index=False)

def endpoints():
    columns=["model","seed","flight","horizon","channel","mse"]
    rows=[]
    for name,filename in [("transformer","controlled_baselines_20261119_run1/per_flight.csv"),("v2","transformer_slowfast_latent_v2_20261124/latent_flights.csv")]:
        data=pd.read_csv(OUT/filename)
        if name=="transformer":data=data.query("model=='transformer' and control=='logged'")
        for row in data.itertuples():
            if row.metric in ("velocity_m_s","rate_rad_s") and row.horizon_s in (.5,1,2):
                rows.append([name,17,row.flight_id,row.horizon_s,row.metric,getattr(row,"mse",getattr(row,"mse_or_score",None))])
    full=pd.read_csv(ROUND/"combined/flight_mse.csv").query("arm=='full'")
    tcn=json.loads((ROUND/"external_baselines/best_flight_mse.json").read_text())
    for seed in SEEDS:
        bare=json.loads((DIAG.parent/f"readout/original_runs/incumbent{seed}/flight_mse.json").read_text())["strong"]
        for j,h in enumerate((.5,1,2)):
            for index,c in enumerate(("velocity_m_s","rate_rad_s")):
                k=("v","w")[index]+("05","1","2")[j]
                arms=[("bare",bare[k]),("wingbeat",full.query("seed==@seed").set_index("flight")[k])]
                if seed==17: arms.append(("tcn",tcn[k]))
                for name,values in arms:
                    rows.extend([name,seed,f,h,c,e] for f,e in values.items())
    table=pd.DataFrame(rows,columns=columns)
    assert table.groupby(["model","seed","horizon","channel"]).size().eq(39).all()
    assert len(table)==39*6*9,table.groupby("model").size()
    table.to_csv(HERE/"endpoint_flight_mse.csv",index=False)
    (table.groupby(["model","seed","horizon","channel"]).mse.mean()**.5).rename("rmse").to_csv(HERE/"endpoint_rmse.csv")
if __name__=="__main__":
    endpoints()
    native()
    print("Endpoint and orthogonal native score checks passed.")
