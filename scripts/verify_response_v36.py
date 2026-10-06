"""Independent input-anchor and response recomputation, without v36 metric helpers."""
import numpy as np
import pandas as pd
from audit_response_v36 import OUT, DATA, CONTROL, V35, read
import train_control_windows_v2 as base


def main():
    events=pd.read_csv(OUT/'events.csv');recorded=pd.read_csv(OUT/'event_response.csv')
    thresholds=pd.read_csv(OUT/'event_thresholds.csv')
    with np.load(DATA/'windows.npz') as z:a={k:z[k] for k in z.files}
    w=pd.read_csv(DATA/'windows.csv');logs=w.log_id.to_numpy()
    def modal(x):return np.stack([x[:,:,0],(x[:,:,1]+x[:,:,2])/2,(x[:,:,2]-x[:,:,1])/2,x[:,:,3]],-1)
    h=modal(a['input_history_command'].astype(float));f=modal(a['input_future_command'].astype(float))
    prior=np.concatenate([h[:,-6:-1],f],axis=1)
    contrast=np.empty((407,25,4))
    for k in range(25):contrast[:,k]=f[:,k:k+5].sum(1)/5-prior[:,k:k+5].sum(1)/5
    offsets=abs(contrast).argmax(1)
    maxima=np.take_along_axis(contrast,offsets[:,None],1)[:,0]
    count=0
    for fold in range(5):
        info=read(DATA/f'fold{fold}/manifest.json');fit=np.array(info['fit_indices']);ids=np.array(info['query_indices'])
        selected=[]
        for j,channel in enumerate(['motor','common','diff','yaw']):
            values=abs(maxima[fit,j]);order=values.argsort(kind='stable')
            weights=np.array([1/(4*np.sum(logs[fit]==log)) for log in logs[fit]])
            cut=values[order[np.searchsorted(np.cumsum(weights[order]),.75)]]
            np.testing.assert_allclose(cut,thresholds[(thresholds.fold==fold)&(thresholds.channel==channel)].threshold.item(),atol=1e-14)
            for idx in ids:
                if abs(maxima[idx,j])>=cut and abs(maxima[idx,j])>1e-12:
                    selected.append((int(idx),channel,int(offsets[idx,j])))
        actual=events[events.fold==fold]
        assert sorted(selected)==sorted(actual[['array_index','channel','offset']].itertuples(index=False,name=None))
        for seed in [17,23,42]:
            for arm,root in [('baseline',CONTROL/f'fold{fold}/seed{seed}'),('joint',V35/f'fold{fold}/joint_increment/seed{seed}')]:
                with np.load(root/'oof_predictions.npz') as z:pred=z['rates_b'].astype(float)
                for e in actual.itertuples():
                    idx=e.array_index;ix=list(ids).index(idx);k=e.offset
                    truth=a['label_rates_b'][idx].astype(float);hist=a['input_history_state'][idx,-5:,3:6].astype(float)
                    pi=np.concatenate([hist,pred[ix]]);ti=np.concatenate([hist,truth])
                    p=pi[k+5:k+30]-pi[k:k+5].mean(0);t=ti[k+5:k+30]-ti[k:k+5].mean(0)
                    pm,tm=p[5:].mean(0),t[5:].mean(0)
                    group=recorded[(recorded.fold==fold)&(recorded.seed==seed)&(recorded.arm==arm)&
                        (recorded.array_index==idx)&(recorded.channel==e.channel)].set_index('axis').loc[list('pqr')]
                    np.testing.assert_allclose(group.response_mse,((p-t)**2).mean(0),rtol=1e-10,atol=1e-12)
                    np.testing.assert_allclose(group.predicted_sustained,pm,rtol=1e-10,atol=1e-12)
                    np.testing.assert_allclose(group.truth_sustained,tm,rtol=1e-10,atol=1e-12)
                    sign=lambda x:np.where(abs(x)<=1e-12,0,np.sign(x))
                    np.testing.assert_array_equal(group.direction_match,(sign(pm)==sign(tm)).astype(float))
                    count+=1
    base.write(OUT/'independent_verification.json',dict(status='passed',all_event_selections_recomputed=True,
        fit_only_thresholds_recomputed=True,response_traces_recomputed=count,event_rows=len(events),
        same_parents_retained=407,response_mse_and_sustained_signs_match=True))
    print(f'Independent checks passed: {len(events)} anchors, {count} model-event traces',flush=True)


if __name__=='__main__':main()
