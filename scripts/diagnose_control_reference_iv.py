"""Nested whole-flight partial-linear IV diagnostic, not causal certification."""
from pathlib import Path
import numpy as np
import pandas as pd

import run_paper_rollout_horizon_ablation as source
from identify_control_dynamics_local import design,fit,predict

m=source.m
BASE=m.ROOT/'docs/analysis/results/control_dynamics_realdata_v1'
OUT=BASE/'reference_iv_diagnostic'


def flight_weights(ids):
    _,inv,count=np.unique(ids,return_inverse=True,return_counts=True)
    w=1/count[inv];return w/w.sum()


def iv_gain(u,y,z,ids):
    """Fixed ridge projection of residual input onto residual references."""
    w=flight_weights(ids);scale=np.sqrt(w@(z*z)).clip(1e-6);zz=z/scale
    gamma=np.linalg.solve(zz.T@(zz*w[:,None])+.01*np.eye(zz.shape[1]),zz.T@(w*u))
    instrument=zz@gamma;den=float(np.sum(w*instrument*u))
    if abs(den)<1e-10:return np.nan,den,float(np.sqrt(np.sum(w*instrument**2)))
    return float(np.sum(w*instrument*y)/den),den,float(np.sqrt(np.sum(w*instrument**2)))


def run():
    OUT.mkdir(parents=True,exist_ok=False);m.configure()
    parent=BASE/'reference_semantics';c=m.read_json(parent/'completion.json');m.verify_pins(c['outputs'])
    ep=m.read_json(parent/'protocol.json');m.verify_pins(ep['pins'])
    _,batches,_=source.inputs();b=batches['train']
    with np.load(parent/'references.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['window_ids'],b.trajectory.window_ids);ref=z['past_current_reference'];admit=z['admitted']
    ids=b.trajectory.log_ids[admit];folds=m.read_json(BASE/'local_identification/protocol.json')['folds'];fold=np.array([folds[g] for g in ids])
    pins=dict(ep['pins'])
    for p in [Path(__file__),parent/'references.npz',BASE/'local_identification/protocol.json']:
        pins[m.rel(p)]=m.file_hash(p)
    m.write_json(OUT/'protocol.json',dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),pins=pins,
        heldout_accessed=False,partitions=['train'],train_only=True,
        target='body q increment100/200ms',treatment='known future dt-weighted common command mean',
        nuisance='current/past states and strictly past commands + known future mean drive,differential,rudder',
        instruments=['current/past100/200ms manual pitch','current/past100/200ms attitude pitch'],
        excluded='rate setpoint is explicitly feedback dependent; not treated as an instrument',
        fit='outer5 whole-flight holdout; inner4 whole-flight crossfit nuisance regressions on outer training only; fixed ridge0.01; flight-equal weights',
        iv='project residual common onto residual three reference values with ridge0.01; beta=E[Uhat*Yres]/E[Uhat*Ures]; no sign constraint',
        inference='reference not required: nuisance Y(X)+beta*(known_future_common-nuisance_U(X)); no future measured reference/state input',
        comparators=['state/history/other commands without common','direct ridge with future common'],
        diagnostics='fold coefficient sign/range, denominator, projected input RMS, out-of-fold forecast error; unstable or worse fits not promoted',
        gate='all5 coefficients finite and same sign + positive OOF q improvement vs no-common on>=60%flights + no worse than direct ridge; necessary only',
        causal_boundary='conditional reference exogeneity/exclusion not established; even a passed diagnostic gate would not prove causal gain or control readiness'))
    rows=[];coeff=[];counter=0
    for steps in [5,10]:
        raw=b.trajectory.controls[admit,:steps];dt=b.trajectory.dt_s[admit,:steps]
        means=np.sum(raw*dt[:,:,None],axis=1)/dt.sum(1)[:,None]
        u=(means[:,1]+means[:,2])/2
        other=np.column_stack([means[:,0],(means[:,1]-means[:,2])/2,means[:,3]])
        x=np.concatenate([design(b,steps)[0][admit],other],1)
        y=b.trajectory.truth.angular_velocity_b[admit,steps,1]-b.trajectory.truth.angular_velocity_b[admit,0,1]
        predictions={name:np.full(len(y),np.nan) for name in ['no_common','direct','iv_manual','iv_attitude']}
        for outer in range(5):
            tr=fold!=outer;ho=~tr
            xy=fit(x[tr],np.column_stack([u[tr],y[tr]]),ids[tr]);held_nuisance=predict(xy,x[ho])
            predictions['no_common'][ho]=held_nuisance[:,1]
            direct=fit(np.column_stack([x[tr],u[tr]]),y[tr,None],ids[tr]);predictions['direct'][ho]=predict(direct,np.column_stack([x[ho],u[ho]]))[:,0]
            for j,name in enumerate(['iv_manual','iv_attitude']):
                z=ref[admit,:,j]
                residual=np.full((len(y),5),np.nan)
                values=np.column_stack([u,y,z])
                for inner in range(5):
                    if inner==outer:continue
                    fitmask=(fold!=outer)&(fold!=inner);held=(fold==inner)
                    assert not set(ids[fitmask])&(set(ids[held])|set(ids[ho]))
                    nuisance=fit(x[fitmask],values[fitmask],ids[fitmask])
                    residual[held]=values[held]-predict(nuisance,x[held])
                assert np.isfinite(residual[tr]).all()
                beta,den,signal=iv_gain(residual[tr,0],residual[tr,1],residual[tr,2:],ids[tr])
                coeff.append(dict(steps=steps,variant=name,fold=outer,beta=beta,denominator=den,instrument_rms=signal,
                    conditional_input_rms=float(np.sqrt(np.mean(residual[tr,0]**2)))))
                predictions[name][ho]=held_nuisance[:,1]+beta*(u[ho]-held_nuisance[:,0])
            print('outer fold',steps,outer,'complete',flush=True)
        for name,p in predictions.items():
            for log in sorted(set(ids)):
                take=ids==log
                rows.append(dict(steps=steps,variant=name,log_id=log,n=int(take.sum()),finite=bool(np.isfinite(p[take]).all()),
                    rmse=float(np.sqrt(np.mean((p[take]-y[take])**2))) if np.isfinite(p[take]).all() else None))
        np.savez_compressed(OUT/f'oof_k{steps}.npz',**predictions,truth=y,common=u,window_ids=b.trajectory.window_ids[admit].astype(str),log_ids=ids.astype(str))
    f=pd.DataFrame(rows);f.to_csv(OUT/'per_flight.csv',index=False);pd.DataFrame(coeff).to_csv(OUT/'coefficients.csv',index=False)
    summary=f.groupby(['steps','variant']).rmse.mean().reset_index();summary.to_csv(OUT/'summary.csv',index=False)
    gates=[]
    for steps in [5,10]:
        p=f[f.steps==steps].pivot(index='log_id',columns='variant',values='rmse')
        for variant in ['iv_manual','iv_attitude']:
            beta=np.array([c['beta'] for c in coeff if c['steps']==steps and c['variant']==variant])
            stable=bool(np.isfinite(beta).all() and ((beta>0).all() or (beta<0).all()))
            frac=float((p[variant]<p.no_common).mean());improvement=float(1-p[variant].mean()/p.no_common.mean())
            passed=stable and frac>=.6 and improvement>0 and p[variant].mean()<=p.direct.mean()
            gates.append(dict(steps=steps,variant=variant,sign_stable=stable,improved_flight_fraction=frac,relative_gain=improvement,gate=bool(passed)))
    m.write_json(OUT/'gates.json',gates);m.verify_pins(pins)
    m.write_json(OUT/'completion.json',dict(status='complete',heldout_accessed=False,plant_promoted=False,instrument_validated=False,
        diagnostic_gates=gates,outputs={m.rel(p):m.file_hash(p) for p in OUT.iterdir() if p.is_file()}))


if __name__=='__main__':run()
