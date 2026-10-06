"""Train-only, held-out-log predictability audit; residuals are not instruments."""
from pathlib import Path
import sys,json,time
import numpy as np,pandas as pd,torch,yaml
from scipy.signal import welch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_isaac_surrogate_straight import sha
from system_identification.models.trajectory_main_v1 import _state_features
from system_identification.evaluation.trajectory import VELOCITY_COLUMNS,QUATERNION_COLUMNS,BODY_RATE_COLUMNS
from system_identification.data.trajectory_dataset import CONTROL_COLUMNS
OUT=ROOT/'docs/analysis/results/conditional_excitation_v1'
CHANNELS=['motor','common','differential','rudder']

def build_rows(samples):
    """Current/past state only; history-control variant never sees the target action."""
    xs=[];xh=[];ys=[];ids=[]
    for (log,seg),g in samples[samples.valid_core].groupby(['log_id','segment_id'],sort=True):
        g=g.sort_values('sample_in_segment');numbers=g.sample_in_segment.to_numpy();n=len(g)
        if n<11:continue
        f=_state_features(torch.tensor(g[list(VELOCITY_COLUMNS)].to_numpy(),dtype=torch.float32),torch.tensor(g[list(QUATERNION_COLUMNS)].to_numpy(),dtype=torch.float32),torch.tensor(g[list(BODY_RATE_COLUMNS)].to_numpy(),dtype=torch.float32),torch.tensor(g.relative_flap_phase_rad.to_numpy(),dtype=torch.float32),torch.zeros(n),torch.tensor(g.flap_frequency_hz.to_numpy(),dtype=torch.float32)).numpy()
        u=g[list(CONTROL_COLUMNS)].to_numpy();u=np.column_stack([u[:,0],(u[:,1]+u[:,2])/2,(u[:,1]-u[:,2])/2,u[:,3]])
        for i in range(10,n,5):
            if not np.all(np.diff(numbers[i-10:i+1])==1):continue
            timestamps=g.timestamp_us.to_numpy()[i-10:i+1]
            if np.any(np.diff(timestamps)<=0) or np.any(np.diff(timestamps)>50000):continue
            state=np.concatenate([f[i],f[i-5],f[i-10]])
            history=np.concatenate([u[i-1],u[i-5],u[i-10]])
            if not np.isfinite(np.r_[state,history,u[i]]).all():continue
            xs.append(state);xh.append(np.r_[state,history]);ys.append(u[i]);ids.append((log,int(seg),int(numbers[i]),int(g.iloc[i].timestamp_us)))
    return np.asarray(xs),np.asarray(xh),np.asarray(ys),pd.DataFrame(ids,columns=['log_id','segment_id','sample_in_segment','timestamp_us'])

def main():
    OUT.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    registry=yaml.safe_load((ROOT/'configs/data/trajectory_dataset_registry.yaml').read_text());did=registry['default_dataset_id'];entry=registry['datasets'][did];mp=ROOT/entry['manifest_path'];assert sha(mp)==entry['manifest_sha256'];m=json.loads(mp.read_text());root=mp.parent
    sp=root/'samples_train.parquet';assert sha(sp)==m['artifact_sha256']['samples_train.parquet']
    contract=dict(dataset_id=did,path=str(root.relative_to(ROOT)),manifest_sha256=sha(mp),sample_artifact_hashes={'samples_train.parquet':sha(sp)},partitions=['train'],phase_contract=m['phase_contract'],frequency_contract=m['frequency_contract'])
    manifest=dict(dataset=contract,script_sha256=sha(__file__),method='5-fold balanced whole-log holdout; PyTorch MLP64x64 tanh,25epochs plus Ridge(alpha=100) diagnostic',
        state_lags_s=[0,.1,.2],control_history_lags_s=[.02,.1,.2],stride_samples=5,seed=17,test_opened=False,
        caveat='Out-of-fold prediction residual includes predictor error, unobserved controller state and measurement noise; not independent excitation or a valid causal instrument')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2));s=pd.read_parquet(sp)
    assert set(s.loc[s.valid_core,'log_id'])==set(m['split_contract']['assignments']['train'])
    xs,xh,y,identity=build_rows(s);groups=identity.log_id.to_numpy();allmetrics=[];allpred=[];folds=[]
    loads=np.zeros(5);assignment={}
    for log,count in identity.log_id.value_counts().items():
        fold=int(np.argmin(loads));assignment[log]=fold;loads[fold]+=count
    assigned=np.array([assignment[g] for g in groups]);splits=[(np.flatnonzero(assigned!=f),np.flatnonzero(assigned==f)) for f in range(5)];start=time.monotonic()
    for name,x in [('state_history',xs),('state_and_control_history',xh)]:
        for learner in ['ridge','mlp']:
            prediction=np.full_like(y,np.nan);foldid=np.full(len(y),-1)
            for fold,(train,held) in enumerate(splits):
                assert not set(groups[train])&set(groups[held])
                mean=x[train].mean(0);std=x[train].std(0).clip(.01)
                xt=torch.tensor((x[train]-mean)/std,dtype=torch.float32);xh=torch.tensor((x[held]-mean)/std,dtype=torch.float32)
                ym=y[train].mean(0);ys=y[train].std(0).clip(.01);yt=torch.tensor((y[train]-ym)/ys,dtype=torch.float32)
                if learner=='ridge':
                    xt=torch.cat([xt,torch.ones(len(xt),1)],1);xh=torch.cat([xh,torch.ones(len(xh),1)],1)
                    penalty=100*torch.eye(xt.shape[1]);penalty[-1,-1]=0
                    beta=torch.linalg.solve(xt.T@xt+penalty,xt.T@yt)
                    prediction[held]=(xh@beta).numpy()*ys+ym
                else:
                    torch.manual_seed(17)
                    model=torch.nn.Sequential(torch.nn.Linear(xt.shape[1],64),torch.nn.Tanh(),torch.nn.Linear(64,64),torch.nn.Tanh(),torch.nn.Linear(64,4))
                    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
                    for epoch in range(25):
                        for batch in torch.randperm(len(xt)).split(1024):
                            optimizer.zero_grad();loss=(model(xt[batch])-yt[batch]).square().mean();loss.backward();optimizer.step()
                    with torch.inference_mode():prediction[held]=model(xh).numpy()*ys+ym
                foldid[held]=fold
                folds.append(dict(features=name,learner=learner,fold=fold,train_logs=sorted(set(groups[train])),held_logs=sorted(set(groups[held]))))
                print(name,learner,fold,'elapsed',time.monotonic()-start,flush=True)
            assert np.isfinite(prediction).all() and np.all(foldid>=0)
            residual=y-prediction;data=identity.copy();data['features']=name;data['learner']=learner;data['fold']=foldid
            for j,c in enumerate(CHANNELS):data[c]=y[:,j];data[c+'_pred']=prediction[:,j];data[c+'_residual']=residual[:,j]
            allpred.append(data)
            for log in ['ALL',*sorted(set(groups))]:
                idx=np.ones(len(y),dtype=bool) if log=='ALL' else groups==log
                for j,c in enumerate(CHANNELS):
                    v=y[idx,j];e=residual[idx,j];var=float(np.var(v));mse=float(np.mean(e*e))
                    allmetrics.append(dict(features=name,learner=learner,log_id=log,channel=c,n=int(idx.sum()),action_std=np.sqrt(var),residual_rmse=np.sqrt(mse),residual_rms_to_action_std=np.sqrt(mse/max(var,1e-12)),r2=1-mse/max(var,1e-12)))
    pred=pd.concat(allpred);pred.to_csv(OUT/'crossfit_predictions.csv',index=False);pd.DataFrame(allmetrics).to_csv(OUT/'per_log_metrics.csv',index=False);(OUT/'folds.json').write_text(json.dumps(folds,indent=2))
    spectra=[];correlations={}
    for name,g in pred[pred.learner=='mlp'].groupby('features'):
        r=g[[c+'_residual' for c in CHANNELS]].to_numpy();corr=np.corrcoef(r.T);correlations[name]=dict(correlation=corr.tolist(),eigenvalues=np.linalg.eigvalsh(corr).tolist())
        for (log,seg),h in g.groupby(['log_id','segment_id']):
            h=h.sort_values('timestamp_us');indices=np.flatnonzero(np.r_[True,np.diff(h.sample_in_segment.to_numpy())!=5]);ends=np.r_[indices[1:],len(h)]
            for begin,end in zip(indices,ends):
                if end-begin<128:continue
                hh=h.iloc[begin:end];fs=1/float(np.median(np.diff(hh.timestamp_us))*1e-6)
                for c in CHANNELS:
                    freq,psd=welch(hh[c+'_residual'].to_numpy(),fs=fs,nperseg=128)
                    for f,v in zip(freq,psd):spectra.append(dict(features=name,log_id=log,segment_id=seg,block_start_sample=int(hh.sample_in_segment.iloc[0]),channel=c,frequency_hz=float(f),power=float(v)))
    pd.DataFrame(spectra).to_csv(OUT/'residual_psd.csv',index=False);(OUT/'residual_channel_correlation.json').write_text(json.dumps(correlations,indent=2))
    manifest.update(completed=True,rows=len(y),logs=len(set(groups)),wall_time_s=time.monotonic()-start);(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(pd.DataFrame(allmetrics).query("log_id=='ALL'").to_string(index=False),flush=True)
if __name__=='__main__':main()
