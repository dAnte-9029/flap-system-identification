"""Independent response reconstruction and aggregation; no analysis imports."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
VARS=['vx','vy','vz','p','q','r'];COLS=[3,4,5,10,11,12]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def linear(t,values,at):
    # Independent bracket formula, rather than the main np.interp implementation.
    t=np.asarray(t,float);out=np.full((len(at),values.shape[1]),np.nan)
    for i,q in enumerate(at):
        if q<t[0]-1e-10 or q>t[-1]+1e-10:continue
        j=int(np.searchsorted(t,q,side='right'))
        if j==len(t):out[i]=values[-1];continue
        if j==0:out[i]=values[0];continue
        alpha=(q-t[j-1])/(t[j]-t[j-1]);out[i]=values[j-1]*(1-alpha)+values[j]*alpha
    return out


def component_sets(p):
    adjacency={}
    for r in p.itertuples():
        adjacency.setdefault(r.flight_a,set()).add(r.flight_b)
        adjacency.setdefault(r.flight_b,set()).add(r.flight_a)
    groups=[];unseen=set(adjacency)
    while unseen:
        todo=[min(unseen)];seen=set()
        while todo:
            x=todo.pop()
            if x in seen:continue
            seen.add(x);todo.extend(adjacency[x]-seen)
        unseen-=seen;groups.append(seen)
    return [set(p.loc[p.flight_a.isin(g),'pair_uid']) for g in groups]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,default=HERE/'results');a=ap.parse_args();out=a.results
    run=json.loads((out/'run_manifest.json').read_text());assert sha(out/'native_rollouts.npz')==run['rollout_sha256']
    selection=json.loads((out/'selection_manifest.json').read_text())
    for p,h in selection['source_sha256'].items():
        # Resolve the historical checkout path against this checkout for portability.
        rel='experiments/local_control_response_scope_v1/'+p.split('/experiments/local_control_response_scope_v1/',1)[1]
        assert sha(HERE.parents[1]/rel)==h
    for name in ['windows','pairs','cohorts']:assert sha(out/(name+'.csv'))==selection[name+'_sha256']
    receipt=json.loads((out/'input_receipt.json').read_text())
    assert sha(HERE/'config.json')==receipt['config_sha256']
    for p,h in receipt['code_sha256'].items():assert sha(HERE.parents[1]/p)==h
    assert not run['test_opened'] and not run['trained'] and not run['future_motion_in_forward']
    pairs=pd.read_csv(out/'pairs.csv').set_index('pair_uid',drop=False)
    windows=pd.read_csv(out/'windows.csv');mem=pd.read_csv(out/'cohorts.csv')
    assert mem.query('split=="train" and cohort=="main"').shape[0]==48
    assert mem.query('split=="validation" and cohort=="main"').shape[0]==1
    assert mem.query('split=="train" and cohort=="flight_disjoint"').shape[0]==17
    assert mem.query('split=="train" and cohort=="same_controller"').shape[0]==11
    assert set(windows.split)=={'train','validation'} and windows.window_id.is_unique
    z=np.load(out/'native_rollouts.npz');d=pd.read_csv(out/'response_curves.csv.gz')
    maximum=0.;checked=0
    for (uid,model,seed),g in d.groupby(['pair_uid','model','seed'],sort=False):
        pair=pairs.loc[uid];i,j=int(pair.a),int(pair.b);n,m=int(z['future_count'][i]),int(z['future_count'][j])
        t1=z['native_time_us'][i,:n+1]*1e-6;t2=z['native_time_us'][j,:m+1]*1e-6
        at=g.loc[g.variable=='vx','time_ms'].to_numpy()*1e-3
        y1=np.vstack([z['truth_initial'][i],z['truth'][i,:n]])[:,COLS]
        y2=np.vstack([z['truth_initial'][j],z['truth'][j,:m]])[:,COLS]
        p1=np.vstack([z['initial'][i],z[f'{model}_{seed}'][i,:n]])[:,COLS].astype(float)
        p2=np.vstack([z['initial'][j],z[f'{model}_{seed}'][j,:m]])[:,COLS].astype(float)
        r1=linear(t1,y1,at);r2=linear(t2,y2,at);f1=linear(t1,p1,at);f2=linear(t2,p2,at)
        real=(r1-y1[0])-(r2-y2[0]);pred=(f1-p1[0])-(f2-p2[0])
        expected={'real':real,'pred':pred,'error':pred-real,
                  'absolute_error_a':f1-r1,'absolute_error_b':f2-r2,
                  'increment_error_a':(f1-p1[0])-(r1-y1[0]),'increment_error_b':(f2-p2[0])-(r2-y2[0]),
                  'causal_real':(r1-p1[0])-(r2-p2[0])}
        for col,arr in expected.items():
            saved=g[col].to_numpy().reshape(-1,6)
            np.testing.assert_allclose(saved,arr,atol=2e-12,rtol=1e-10,equal_nan=True)
            maximum=max(maximum,float(np.nanmax(np.abs(saved-arr))));checked+=np.isfinite(saved).sum()
    points=pd.read_csv(out/'point_responses.csv.gz');summary=pd.read_csv(out/'summary.csv')
    cfg=json.loads((HERE/'config.json').read_text());aggmax=0.;nrows=0
    groups={key:g for key,g in points.groupby(['model','variable','time_label'],sort=False)}
    cohorts={(sp,c):set(g.pair_uid) for (sp,c),g in mem.groupby(['split','cohort'])}
    for r in summary.itertuples():
        g=groups[r.model,r.variable,r.time_label]
        g=g[g.pair_uid.isin(cohorts[r.split,r.cohort])&g.supported]
        if str(r.seed)!='pooled':g=g[g.seed==int(r.seed)]
        pp=pairs.loc[g.pair_uid.unique()]
        assert len(pp)==r.supported_pairs
        # Seed mean first, then direct equal pair/flight/component reductions.
        err=g.pred-g.real;active=g.real.abs()>cfg['direction_floors'][r.variable]
        correct=(np.sign(g.pred)==np.sign(g.real))&(g.pred.abs()>1e-8)&active
        measures=pd.DataFrame(dict(pair_uid=g.pair_uid,e2=err**2,ae=err.abs(),bias=err,
            real2=g.real**2,pred2=g.pred**2,active=active.astype(float),correct=correct.astype(float),
            traj2=(g.absolute_error_a**2+g.absolute_error_b**2)/2,causal2=(g.pred-g.causal_real)**2)).groupby('pair_uid').mean()
        if r.weighting=='pair':avg=measures.mean()
        elif r.weighting=='flight':
            avg=pd.DataFrame([measures.loc[pp.loc[(pp.flight_a==f)|(pp.flight_b==f),'pair_uid']].mean()
                for f in sorted(set(pp.flight_a)|set(pp.flight_b))]).mean()
        else:avg=pd.DataFrame([measures.loc[list(u)].mean() for u in component_sets(pp)]).mean()
        values=dict(rmse=np.sqrt(avg.e2),mae=avg.ae,bias=avg.bias,real_rms=np.sqrt(avg.real2),pred_rms=np.sqrt(avg.pred2),
                    trajectory_rmse=np.sqrt(avg.traj2),causal_origin_rmse=np.sqrt(avg.causal2),near_zero_fraction=1-avg.active,
                    direction_accuracy=avg.correct/avg.active if avg.active>0 else np.nan)
        for k,val in values.items():
            saved=getattr(r,k)
            assert np.isclose(val,saved,atol=2e-12,rtol=1e-9,equal_nan=True),(r.Index,k,val,saved)
            if np.isfinite(val):aggmax=max(aggmax,abs(val-saved))
        nrows+=1
    result=dict(passed=True,response_scalar_checks=int(checked),response_max_abs_difference=maximum,
                summary_rows=nrows,aggregate_max_abs_difference=aggmax,main_train_pairs=48,main_validation_pairs=1,
                verifier_sha256=sha(Path(__file__)),test_opened=False)
    (out/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
