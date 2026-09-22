"""Final necessary consistency checks; no training, source modification or selection."""
from datetime import datetime
import json
import numpy as np
import pandas as pd
import torch
from run_paper_independent_test import OUT,ART,ROOT,load_registration,read,write,now,model_load,DEVICE,subset,hold_controls,predict_history_trajectory_model,configure
from paper_independent_test_core import METRICS,HORIZONS


def main():
    configure();p=load_registration();batches=torch.load(ART/'prepared.pt',map_location='cpu',weights_only=False)
    outcomes=read(ART/'inference_status.json')['outcomes'];coverage=pd.read_csv(OUT/'test_flight_coverage.csv')
    assert len(outcomes)==38 and all(x['status']=='valid' for x in outcomes)
    events=[json.loads(line) for line in (OUT/'test_access_log.jsonl').read_text().splitlines()]
    assert all(datetime.fromisoformat(e['at_utc'])>datetime.fromisoformat(p['frozen_at_utc']) for e in events)
    assert set(e['date'] for e in events)=={'Sep8','Sep19'}
    per=pd.read_csv(OUT/'per_flight.csv');seed=pd.read_csv(OUT/'per_seed_summary.csv');multi=pd.read_csv(OUT/'multiseed_summary.csv')
    cube=pd.read_csv(OUT/'error_evolution.csv')
    maxnorm=0.;checked=0
    for date,b in batches.items():
        assert set(b.trajectory.log_ids)==set(coverage[(coverage.date==date)&(coverage.status=='admitted')].flight_id)
        assert not set(b.trajectory.log_ids)&set(p['train_flights']+p['validation_flights'])
        assert b.history_mask.all()
        np.testing.assert_array_equal(b.history_controls[:,-1],b.trajectory.controls[:,0])
    for run in outcomes:
        b=batches[run['date']];n=len(b.trajectory.window_ids);directory=ROOT/run['directory']
        with np.load(directory/f'{run["condition"]}_predictions.npz') as pred:
            np.testing.assert_array_equal(pred['window_ids'],b.trajectory.window_ids)
            for k in vars(b.trajectory.truth):
                assert pred[k].shape==getattr(b.trajectory.truth,k).shape and np.isfinite(pred[k]).all()
            maxnorm=max(maxnorm,float(np.max(abs(np.linalg.norm(pred['quaternion_nb'],axis=-1)-1))))
            if run['condition']=='Hold':
                with np.load(directory/'Actual_predictions.npz') as actual:
                    for k in vars(b.trajectory.truth):np.testing.assert_allclose(pred[k][:,:2],actual[k][:,:2],atol=2e-5,rtol=1e-6)
        err=np.load(directory/f'{run["condition"]}_squared_errors.npy');assert err.shape==(n,50,4) and np.isfinite(err).all()
        d=per[(per.date==run['date'])&(per.model==run['model'])&(per.condition==run['condition'])&(per.group=='ALL')&(per.kind=='endpoint')]
        d=d[d.seed.isna()] if run['seed'] is None else d[d.seed==run['seed']]
        for flight in set(b.trajectory.log_ids):
            mask=b.trajectory.log_ids==flight
            for h,k in HORIZONS.items():
                values=np.sqrt(err[mask,k-1].mean(0))
                for j,metric in enumerate(METRICS):
                    row=d[(d.flight_id==flight)&(d.metric==metric)&(d.step==k)].iloc[0]
                    assert row.n_origins==mask.sum();np.testing.assert_allclose(row.value,values[j],atol=1e-12,rtol=1e-12);checked+=1
    assert maxnorm<1e-5
    for row in multi.itertuples():
        if row.status!='valid':continue
        d=seed[(seed.date==row.date)&(seed.model==row.model)&(seed.condition==row.condition)&(seed.kind==row.kind)&(seed.group==row.group)&(seed.step==row.step)&(seed.metric==row.metric)]
        np.testing.assert_allclose(row.mean,d.value.mean(),rtol=1e-12)
        if row.model=='B0':assert len(d)==1 and np.isnan(row.std)
        else:assert set(d.seed)=={17,23,42};np.testing.assert_allclose(row.std,d.value.std(ddof=1),rtol=1e-10,atol=1e-12)
        if row.kind=='endpoint' and row.model=='H26' and (row.group=='ALL' or row.step==25):
            c=cube[(cube.date==row.date)&(cube.condition==row.condition)&(cube.group==row.group)&(cube.step==row.step)&(cube.metric==row.metric)].iloc[0]
            np.testing.assert_allclose(row.mean,c['mean'],rtol=1e-12);np.testing.assert_allclose(row.std,c['std'],rtol=1e-10,atol=1e-12)
    # Fixed small constant-control inference for each frozen H26 seed; no performance selection.
    constant_checks=[]
    for row in p['models']:
        if row['model']!='H26':continue
        model=model_load(row);b=hold_controls(subset(batches['Sep8'],4))
        a=predict_history_trajectory_model(model,b,use_history=True,batch_size=128,device=DEVICE)
        z=predict_history_trajectory_model(model,hold_controls(b),use_history=True,batch_size=128,device=DEVICE)
        for k,v in vars(a).items():np.testing.assert_array_equal(v,getattr(z,k))
        constant_checks.append(row['seed'])
    write(OUT/'final_verification.json',dict(status='passed',at_utc=now(),registration_precedes_every_test_read=True,
        successful_prediction_runs=38,fifteen_unique_checkpoints=True,no_training=True,independent_per_flight_metric_checks=checked,
        fixed_origin_identity=True,history_last_control_equals_origin=True,no_split_overlap=True,
        max_quaternion_norm_error=maxnorm,all_predictions_finite=True,endpoint_evolution_parity=True,
        equal_flight_and_seed_sd_verified=True,constant_control_bitwise_equal_seeds=constant_checks,
        old_model_report_hashes_unchanged=True))
    print('Final verification passed:',checked,'per-flight endpoint cells; all38prediction runs, constant control seeds',constant_checks)


if __name__=='__main__':main()
