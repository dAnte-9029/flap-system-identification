"""Fixed-regularization causal endpoint baseline and train-flight OOF errors."""
import numpy as np
import pandas as pd
from control_windows_v3 import OUT,OLD,base


def features(x,step):
    h=x['history_state'].numpy();r=x['history_reference'].numpy()
    future=x['future_reference'].numpy()[:,:step]
    return np.concatenate([h[:,-1],h[:,-10:].mean(1),r[:,-1],r[:,-1]-r[:,-10],
        x['history_actuator'].numpy()[:,-1],x['current_mode'].numpy(),
        x['initial_quaternion'].numpy(),x['initial_velocity_n'].numpy(),
        future.mean(1)-r[:,-1],future[:,-1]-r[:,-1]],axis=1)


def fit_predict(features,target,train,query,logs,alpha=10.):
    mean=features[train].mean(0);scale=features[train].std(0).clip(1e-4)
    f=np.c_[np.ones(len(features)),(features-mean)/scale]
    weight=np.ones(len(train))
    for log in np.unique(logs[train]):
        mask=logs[train]==log;weight[mask]=len(train)/(len(np.unique(logs[train]))*mask.sum())
    penalty=np.eye(f.shape[1])*alpha;penalty[0,0]=0
    coef=np.linalg.solve(f[train].T@(weight[:,None]*f[train])+penalty,
                         f[train].T@(weight[:,None]*target[train]))
    return f[query]@coef,dict(mean=mean,scale=scale,coefficient=coef)


def main():
    path=OUT/'ridge';path.mkdir(exist_ok=False)
    base.write(path/'protocol.json',dict(alpha=10.,horizons_steps=[5,10,25,50],
        outputs='delta p/q/r and delta velocity NED; separate endpoint fits, not a rollout simulator',
        inputs='current/past features and references up to each target only; no later reference access',
        fit='train-only normalization, flight-equal least squares, unpenalized intercept',
        calibration='five train-flight leave-one-out predictions, not validation fitting',sealed_test_accessed=False))
    x,y,stats=base.load();w=pd.read_csv(OLD/'windows_v2.csv');logs=w.log_id.to_numpy()
    tr=w[w.partition=='train'].array_index.to_numpy();val=w[w.partition=='validation'].array_index.to_numpy()
    initial=np.c_[x['initial_rates_b'].numpy(),x['initial_velocity_n'].numpy()]
    rows=[];errors=[]
    for step in [5,10,25,50]:
        target=np.c_[y['rates_b'][:,step-1].numpy(),y['velocity_n'][:,step-1].numpy()]-initial
        f=features(x,step)
        pred,model=fit_predict(f,target,tr,np.arange(len(w)),logs)
        np.savez(path/f'k{step}.npz',**model,prediction=pred,truth=target)
        oof=np.full_like(target,np.nan)
        for log in np.unique(logs[tr]):
            fit=tr[logs[tr]!=log];query=tr[logs[tr]==log]
            oof[query],_=fit_predict(f,target,fit,query,logs)
        for partition,indices,prediction in [('train',tr,pred),('validation',val,pred),('train_oof',tr,oof)]:
            err=prediction[indices]-target[indices]
            frame=w.iloc[indices][['window_id','log_id','mode','quality']].copy()
            frame['partition']=partition;frame['horizon_s']=step*.02
            for j,name in enumerate(['p','q','r','vx','vy','vz']):frame[name]=err[:,j]**2
            errors.append(frame)
            for log in np.unique(logs[indices]):
                mask=logs[indices]==log;rmse=np.sqrt((err[mask]**2).mean(0))
                rows.append(dict(partition=partition,log_id=log,horizon_s=step*.02,**dict(zip(['p','q','r','vx','vy','vz'],rmse))))
    pd.DataFrame(rows).to_csv(path/'per_flight_rmse.csv',index=False)
    pd.concat(errors).to_csv(path/'per_window_squared_errors.csv',index=False)
    summary=pd.DataFrame(rows).groupby(['partition','horizon_s'])[['p','q','r','vx','vy','vz']].mean()
    summary.to_csv(path/'summary.csv');print(summary.round(4))


if __name__=='__main__':main()
