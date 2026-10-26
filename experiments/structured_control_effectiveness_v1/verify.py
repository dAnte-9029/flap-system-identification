"""Independent point interpolation, integration, graph weighting and SO(3) errors."""
from common import *
from scipy.spatial.transform import Rotation
ind=legacy(ABLATION/'verify.py','independent_previous')

def weights(pp,kind):
    adj={f:set() for f in set(pp.flight_a)|set(pp.flight_b)}
    for p in pp.itertuples():adj[p.flight_a].add(p.flight_b);adj[p.flight_b].add(p.flight_a)
    labels={};n=0
    for node in adj:
        if node in labels:continue
        todo=[node]
        while todo:
            f=todo.pop()
            if f in labels:continue
            labels[f]=n;todo.extend(adj[f]-set(labels))
        n+=1
    degree={f:sum(p.flight_a==f or p.flight_b==f for p in pp.itertuples()) for f in adj}
    size={c:sum(labels[p.flight_a]==c for p in pp.itertuples()) for c in set(labels.values())}
    return {p.pair_uid:1. if kind=='pair' else (1/degree[p.flight_a]+1/degree[p.flight_b]) if kind=='flight' else 1/size[labels[p.flight_a]] for p in pp.itertuples()}

def main():
    x=np.load(SWAP/'results/inputs.npz');pred=np.load(OUT/'predictions.npz');pairs=pd.read_csv(SWAP/'results/pairs.csv');mem=pd.read_csv(SWAP/'results/cohorts.csv');start={p.pair_uid:i*4 for i,p in enumerate(pairs.itertuples())};channels=dict(zip(['vx','vy','vz','p','q','r'],[3,4,5,10,11,12]))
    facts=pd.read_csv(OUT/'factual_points.csv.gz');internal=pd.read_csv(OUT/'internal_points.csv.gz');err=0.;ierr=0.;count=0
    for r in facts[(facts.time_label=='300')&facts.supported].itertuples():
        values=[]
        for ci in [start[r.pair_uid],start[r.pair_uid]+3]:
            n=int(x['valid_steps'][ci]);t=x['native_time_us'][ci,:n+1]*1e-6;y=np.vstack([x['initial'][ci],pred[r.model][ci,:n]]).astype(float);y-=y[0].copy()
            values.append((ind.average(t,y,.3,.27) if r.method=='D' else ind.linear(t,y,.3))[channels[r.variable]])
        err=max(err,abs(values[0]-values[1]-r.pred));count+=1
    assert err<1e-11,err
    for r in internal[(internal.time_label=='300')&internal.supported].itertuples():
        values=[]
        for ci in range(start[r.pair_uid],start[r.pair_uid]+4):
            n=int(x['valid_steps'][ci]);t=x['native_time_us'][ci,:n+1]*1e-6;y=np.vstack([x['initial'][ci],pred[r.model][ci,:n]]).astype(float);values.append((ind.linear(t,y,.3)-y[0])[channels[r.variable]])
        a,b,c,d=values;v=np.array([a-b,c-d,(a-b+c-d)/2,(a-c+b-d)/2]);ierr=max(ierr,float(abs(v-[r.CA,r.CB,r.C,r.H]).max()))
    closure=float(abs(internal.F-internal.C-internal.H).max());assert max(ierr,closure)<1e-12
    truth=pd.read_csv(ABLATION/'results/factual_points.csv.gz');truth=truth[(truth.variant=='V0')&(truth.model=='GRU-Control')&(truth.seed==17)]
    joined=facts.merge(truth[['pair_uid','variable','method','time_label','real','supported']],on=['pair_uid','variable','method','time_label'],suffixes=('','_old'),validate='many_to_one')
    np.testing.assert_array_equal(joined.supported,joined.supported_old);np.testing.assert_allclose(joined.real,joined.real_old,atol=0,rtol=0,equal_nan=True)
    maxstat=0.;summ=pd.read_csv(OUT/'factual_summary.csv');summ=summ[summ.time_label=='300'];weight_cache={}
    def cached_weights(pp,kind):
        key=(tuple(pp.pair_uid),kind)
        if key not in weight_cache:weight_cache[key]=weights(pp,kind)
        return weight_cache[key]
    primary=facts[(facts.time_label=='300')&facts.supported]
    grouped=dict(tuple(primary.groupby(['model','variable','method'])))
    for r in summ.itertuples():
        ids=mem[(mem.split==r.split)&(mem.cohort==r.cohort)].pair_uid;source=grouped[(r.model,r.variable,r.method)];g=source[source.pair_uid.isin(ids)]
        pp=pairs[pairs.pair_uid.isin(g.pair_uid)];w=g.pair_uid.map(cached_weights(pp,r.weighting)).to_numpy();w=w/w.sum();a=g.real.to_numpy();b=g.pred.to_numpy();active=abs(a)>.1
        computed=[np.sqrt(sum(w*b*b)/sum(w*a*a)),sum(w*a*b)/sum(w*a*a),sum(w[active]*((a[active]*b[active]>0)&(abs(b[active])>1e-8)))/sum(w[active]) if any(active) else np.nan]
        for expected,actual in zip(computed,[r.scale_ratio,r.signed_gain,r.direction_accuracy]):
            if np.isfinite(expected):maxstat=max(maxstat,abs(expected-actual))
    assert maxstat<1e-11,maxstat
    istat=0.;isumm=pd.read_csv(OUT/'internal_summary.csv');isumm=isumm[isumm.time_label=='300']
    igrouped=dict(tuple(internal[(internal.time_label=='300')&internal.supported].groupby(['model','variable'])))
    for r in isumm.itertuples():
        ids=mem[(mem.split==r.split)&(mem.cohort==r.cohort)].pair_uid;source=igrouped[(r.model,r.variable)];g=source[source.pair_uid.isin(ids)]
        pp=pairs[pairs.pair_uid.isin(g.pair_uid)];w=g.pair_uid.map(cached_weights(pp,r.weighting)).to_numpy();w=w/w.sum()
        for key in ['CA','CB','C','H']:istat=max(istat,abs(float(np.sqrt(sum(w*g[key].to_numpy()**2)))-getattr(r,key+'_rms')))
    assert istat<1e-12,istat
    # Independent SciPy quaternion geodesic and double-precision Euclidean reductions.
    sample=np.load(OUT/'trajectory_verification_samples.npz');table=pd.read_csv(OUT/'validation_trajectory.csv.gz').set_index(['model','window_index']);terr=0.
    for key in sample.files:
        if not key.startswith(('M0_','M1_','M2_')):continue
        model,idx=key.split('_');i=int(idx);p=sample[key].astype(float);y=sample[f'target_{i}'].astype(float);mask=sample[f'mask_{i}'];p=p[mask];y=y[mask]
        attitude=(Rotation.from_quat(p[:,[7,8,9,6]]).inv()*Rotation.from_quat(y[:,[7,8,9,6]])).magnitude()
        e=np.column_stack([np.linalg.norm(p[:,:3]-y[:,:3],axis=1),np.linalg.norm(p[:,3:6]-y[:,3:6],axis=1),attitude,np.linalg.norm(p[:,10:13]-y[:,10:13],axis=1)])
        r=table.loc[(model,i)]
        for j,var in enumerate(['position_m','velocity_mps','attitude_rad','rate_radps']):terr=max(terr,abs(np.sqrt(np.mean(e[:,j]**2))-r[var+'_path_rms']),abs(e[-1,j]-r[var+'_endpoint_norm']))
    assert terr<5e-6,terr
    history=pd.read_csv(OUT/'training_history.csv');assert history.groupby('epoch').batch_order_sha256.nunique().eq(1).all();assert history.groupby('model').updates.max().eq(636).all();assert len(history)==18
    plan=json.loads((OUT/'holdout_protocol.json').read_text());held=set();covered=[]
    for fold in plan['folds']:
        assert not set(fold['held_flights'])&set(fold['train_flights']);assert not held&set(fold['held_flights']);held.update(fold['held_flights']);covered.extend(fold['main_pairs'])
    assert len(covered)==len(set(covered))==48 and not plan['executed']
    result=dict(passed=True,factual_primary_rows=count,factual_max_error=err,internal_max_error=ierr,closure_max_error=closure,summary_rows=len(summ),summary_max_error=maxstat,internal_summary_rows=len(isumm),internal_summary_max_error=istat,trajectory_scipy_windows_per_model=128,trajectory_max_error=terr,truth_and_support_identical=True,batch_orders_identical=True,updates_per_model=636,fold_pair_coverage=48,fold_training_executed=False,test_read=False)
    save(OUT/'independent_verification.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
