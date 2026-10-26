"""Independent training budget, fold provenance, interpolation and metric audit."""
from common import *
import torch
from scipy.spatial.transform import Rotation
ind=legacy(PILOT/'verify.py','independent_pilot')

def training_audit():
    manifest=DATA/'dataset_manifest.json';expected=json.loads((TRAIN/'data_audit.json').read_text())['manifest_sha256'];actual=sha(manifest);assert actual==expected
    save(OUT/'dataset_registry_check.json',dict(dataset_id='velocity_rate_multistep_v1',manifest_path=str(manifest),manifest_sha256=actual,registry_path='configs/data/velocity_rate_dataset_registry.yaml',registry_sha256=sha(ROOT/'configs/data/velocity_rate_dataset_registry.yaml'),partitions_read=['train','validation'],test_artifact_read_or_hashed=False,passed=True))
    plan=json.loads((OUT/'holdout_protocol.json').read_text());logs=[];summaries=[];indices=np.load(OUT/'holdout_window_indices.npz')
    for f in plan['folds']:
        fold=f['fold'];normalization=json.loads((HERE/'folds'/f'fold{fold}'/'normalization.json').read_text());nh=sha(HERE/'folds'/f'fold{fold}'/'normalization.json')
        for model in ['M0','M1','M2']:
            for seed in [17,29,43]:
                path=HERE/'folds'/f'fold{fold}'/'training'/model/f'seed{seed}';summary=json.loads((path/'summary.json').read_text());summaries.append(summary)
                lines=[json.loads(l) for l in (path/'training.jsonl').read_text().splitlines()];assert [r['epoch'] for r in lines]==list(range(1,25));logs.extend(lines)
                assert summary['updates']==24*int(np.ceil(f['train_windows']/1024)) and summary['normalization_sha256']==nh
                best=float('inf');best_epoch=0
                for r in lines:
                    assert abs(r['lr']-.001*(1+np.cos(np.pi*(r['epoch']-1)/24))/2)<1e-14
                    if r['validation_flight_loss']<best-.0001:best=r['validation_flight_loss'];best_epoch=r['epoch']
                    if model!='M0':assert r['control_gradient_norm_min']>0 and np.isfinite(r['control_gradient_norm_mean'])
                assert summary['best_epoch']==best_epoch
                for tag in ['best','last']:
                    assert sha(path/f'{tag}.pt')==summary[f'{tag}_sha256'];ck=torch.load(path/f'{tag}.pt',map_location='cpu',weights_only=False)
                    assert ck['normalization']==normalization and ck['source_sha256']==sha(PILOT/'model.py')
                    assert tuple(ck['input_keys'])==training.INPUT_KEYS and 'target' not in ck['input_keys'] and 'future_phase' not in ck['input_keys']
                    for field in ['mean','scale','output_scale']:torch.testing.assert_close(ck['model'][field],torch.tensor(normalization[field]),rtol=0,atol=0)
                    if model!='M0':
                        scale=np.array(normalization['scale'])[7:11];expected=np.array([scale[0],.5*np.linalg.norm(scale[1:3]),.5*np.linalg.norm(scale[1:3]),scale[3]])
                        np.testing.assert_allclose(ck['model']['coordinate_scale'],expected,rtol=2e-7,atol=1e-8)
                assert ck['epoch']==24
                save(path/'effective_run_config.json',dict(fold=fold,model=model,seed=seed,epochs=24,history_ms=500,horizon_ms=1000,batch_size=1024,optimizer='AdamW',learning_rate=.001,weight_decay=.0001,gradient_clip=5,cosine_T_max=24,early_stopping=False,input_contract='V0',normalization='fold-local fit windows only',normalization_sha256=nh,normalization_path=f'folds/fold{fold}/normalization.json',train_indices_sha256=summary['train_indices_sha256'],config_sha256=summary['config_sha256'],updates=summary['updates'],parameters=summary['parameters'],selection='original validation 27 flights equal-flight loss, minimum improvement 1e-4',model_source_sha256=sha(PILOT/'model.py'),input_keys=training.INPUT_KEYS,test_read=False))
    history=pd.DataFrame(logs);assert len(history)==27*24
    assert history.groupby(['fold','seed','epoch']).batch_order_sha256.nunique().eq(1).all()
    assert history.groupby(['fold','seed','epoch']).updates.nunique().eq(1).all()
    for (fold,seed,epoch),g in history.groupby(['fold','seed','epoch']):
        ids=indices[f'fold{fold}_train_indices'];order=torch.randperm(len(ids),generator=torch.Generator().manual_seed(int(seed)*10000+int(epoch))).numpy()
        assert g.batch_order_sha256.iloc[0]==hashlib.sha256(ids[order].tobytes()).hexdigest()
    history.to_csv(OUT/'training_history.csv',index=False);pd.DataFrame(summaries).to_csv(OUT/'training_summary.csv',index=False)
    save(OUT/'training_completion.json',dict(passed=True,completed=27,expected=27,epochs_each=24,epoch_records=len(history),optimizer_updates=int(sum(r['updates'] for r in summaries)),total_epoch_gpu_seconds=float(history.seconds.sum()),normalization_fold_local=True,selection_uses_original_validation_only=True,resumed_runs=sum(r['resumed'] for r in summaries),test_read=False))

def evaluation_audit():
    x=np.load(SWAP/'results/inputs.npz');pairs=pd.read_csv(OUT/'pairs.csv');mem=pd.read_csv(OUT/'cohorts.csv');assign=pd.read_csv(OUT/'holdout_pair_assignments.csv');plan=json.loads((OUT/'holdout_protocol.json').read_text());pred={i:np.load(OUT/f'predictions_fold{i}.npz') for i in range(3)};lookup={i:{ci:j for j,ci in enumerate(p['case_ids'])} for i,p in pred.items()};start={p.pair_uid:i*4 for i,p in enumerate(pairs.itertuples())};channels=dict(zip(['vx','vy','vz','p','q','r'],[3,4,5,10,11,12]))
    facts=pd.read_csv(OUT/'factual_points.csv.gz');internal=pd.read_csv(OUT/'internal_points.csv.gz');foldmap=dict(zip(assign.pair_uid,assign.fold))
    main=facts[(facts.pair_uid.isin(foldmap))&(facts.time_label=='300')&(facts.variable=='p')&(facts.method=='D')];assert len(main)==432 and main.supported.all();assert not main.duplicated(['model','seed','pair_uid']).any();assert (main.fold==main.pair_uid.map(foldmap)).all()
    pairindex=pairs.set_index('pair_uid')
    for fold,g in facts[facts.role=='heldout'].groupby('fold'):
        pp=pairindex.loc[g.pair_uid.unique()];allowed=set(plan['folds'][fold]['held_flights']);assert (set(pp.flight_a)|set(pp.flight_b))<=allowed
    err=0.;count=0
    for r in facts[(facts.time_label=='300')&facts.supported].itertuples():
        values=[]
        for ci in [start[r.pair_uid],start[r.pair_uid]+3]:
            n=int(x['valid_steps'][ci]);t=x['native_time_us'][ci,:n+1]*1e-6;y=np.vstack([x['initial'][ci],pred[r.fold][f'{r.model}_{r.seed}'][lookup[r.fold][ci],:n]]).astype(float);y-=y[0].copy()
            values.append((ind.ind.average(t,y,.3,.27) if r.method=='D' else ind.ind.linear(t,y,.3))[channels[r.variable]])
        err=max(err,abs(values[0]-values[1]-r.pred));count+=1
    assert err<1e-11,err
    ierr=0.
    for r in internal[(internal.time_label=='300')&internal.supported].itertuples():
        values=[]
        for ci in range(start[r.pair_uid],start[r.pair_uid]+4):
            n=int(x['valid_steps'][ci]);t=x['native_time_us'][ci,:n+1]*1e-6;y=np.vstack([x['initial'][ci],pred[r.fold][f'{r.model}_{r.seed}'][lookup[r.fold][ci],:n]]).astype(float);values.append((ind.ind.linear(t,y,.3)-y[0])[channels[r.variable]])
        a,b,c,d=values;ierr=max(ierr,float(abs(np.array([a-b,c-d,(a-b+c-d)/2,(a-c+b-d)/2])-[r.CA,r.CB,r.C,r.H]).max()))
    closure=float(abs(internal.F-internal.C-internal.H).max());assert max(ierr,closure)<1e-12
    truth=pd.read_csv(ABLATION/'results/factual_points.csv.gz').query("variant=='V0' and model=='GRU-Control' and seed==17")
    joined=facts.merge(truth[['pair_uid','variable','method','time_label','real','supported']],on=['pair_uid','variable','method','time_label'],suffixes=('','_old'),validate='many_to_one');np.testing.assert_array_equal(joined.supported,joined.supported_old);np.testing.assert_allclose(joined.real,joined.real_old,rtol=0,atol=0,equal_nan=True)
    weight_cache={}
    def weights(pp,kind):
        key=tuple(pp.pair_uid),kind
        if key not in weight_cache:weight_cache[key]=ind.weights(pp,kind)
        return weight_cache[key]
    maximum=0.;imax=0.;counts=[]
    for filename,data,isinternal in [('factual_summary.csv',facts,False),('internal_summary.csv',internal,True)]:
        summaries=pd.read_csv(OUT/filename);summaries=summaries[summaries.time_label=='300'];counts.append(len(summaries));keys=['model','variable']+([] if isinternal else ['method']);grouped=dict(tuple(data[(data.time_label=='300')&data.supported&(data.role=='heldout')].groupby(keys)))
        for r in summaries.itertuples():
            ids=mem[(mem.split=='train')&(mem.cohort==r.cohort)].pair_uid;source=grouped[(r.model,r.variable) if isinternal else (r.model,r.variable,r.method)];g=source[source.pair_uid.isin(ids)]
            if r.scope.startswith('fold'):g=g[g.fold==int(r.scope[-1])]
            if r.scope=='outside_fold0':g=g[g.fold!=0]
            if r.seed!='pooled':g=g[g.seed==int(r.seed)]
            pp=pairs[pairs.pair_uid.isin(g.pair_uid)];w=g.pair_uid.map(weights(pp,r.weighting)).to_numpy();w=w/w.sum()
            if isinternal:
                for key in ['CA','CB','C','H']:imax=max(imax,abs(np.sqrt(sum(w*g[key].to_numpy()**2))-getattr(r,key+'_rms')))
            else:
                a=g.real.to_numpy();b=g.pred.to_numpy();active=abs(a)>.1
                for expected,actual in [(np.sqrt(sum(w*b*b)/sum(w*a*a)),r.scale_ratio),(sum(w*a*b)/sum(w*a*a),r.signed_gain),(sum(w[active]*((a[active]*b[active]>0)&(abs(b[active])>1e-8)))/sum(w[active]) if any(active) else np.nan,r.direction_accuracy)]:
                    if np.isfinite(expected):maximum=max(maximum,abs(expected-actual))
    assert maximum<1e-11 and imax<1e-12,(maximum,imax)
    # Independent paired direction changes, not a second call to the production metric function.
    paired=pd.read_csv(OUT/'paired_changes.csv.gz');valid=paired[(paired.time_label=='300')&paired.supported&(paired.role=='heldout')];assert ((valid.gained.astype(int)-valid.lost.astype(int))==(((valid.real*valid.pred>0)&(abs(valid.pred)>1e-8)).astype(int)-((valid.real*valid.pred_M0>0)&(abs(valid.pred_M0)>1e-8)).astype(int))*(abs(valid.real)>.1)).all()
    # SciPy physical-unit trajectory check on 64 prespecified held windows per fold/model/seed.
    sample=np.load(OUT/'trajectory_verification_samples.npz');table=pd.read_csv(OUT/'heldout_trajectory.csv.gz').set_index(['fold','model','seed','window_index']);terr=0.;tn=0
    for key in sample.files:
        if key.startswith(('target','mask')):continue
        fold,name,seed,idx=key.split('_');i=int(idx);p=sample[key].astype(float);y=sample[f'target_{i}'].astype(float);mask=sample[f'mask_{i}'];p=p[mask];y=y[mask]
        angle=(Rotation.from_quat(p[:,[7,8,9,6]]).inv()*Rotation.from_quat(y[:,[7,8,9,6]])).magnitude();er=np.column_stack([np.linalg.norm(p[:,:3]-y[:,:3],axis=1),np.linalg.norm(p[:,3:6]-y[:,3:6],axis=1),angle,np.linalg.norm(p[:,10:13]-y[:,10:13],axis=1)]);row=table.loc[(int(fold),name,int(seed),i)]
        for j,var in enumerate(['position_m','velocity_mps','attitude_rad','rate_radps']):terr=max(terr,abs(np.sqrt(np.mean(er[:,j]**2))-row[var+'_path_rms']),abs(er[-1,j]-row[var+'_endpoint_norm']))
        for var,ch in channels.items():terr=max(terr,abs(np.sqrt(np.mean((p[:,ch]-y[:,ch])**2))-row[var+'_path_rms']))
        tn+=1
    assert tn==27*64 and terr<5e-6,(tn,terr)
    save(OUT/'independent_verification.json',dict(passed=True,factual_primary_rows=count,factual_max_error=err,internal_max_error=ierr,closure_max_error=closure,summary_rows=counts[0],summary_max_error=maximum,internal_summary_rows=counts[1],internal_summary_max_error=imax,trajectory_checked_windows=tn,trajectory_scipy_max_error=terr,truth_support_identical=True,main_pairs_once_per_model_seed=48,components=11,test_read=False))
    print(json.dumps(json.loads((OUT/'independent_verification.json').read_text()),indent=2))
if __name__=='__main__':training_audit();evaluation_audit()
