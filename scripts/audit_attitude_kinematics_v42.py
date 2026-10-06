"""Native kinematic parity and whole-window rate oracles, no model rerollout."""
import itertools
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from run_coupled_replay_v40 import DATA,read
from coupled_replay_v40 import VARIANTS,angles
from actuator_diagnostics_v31 import integrate_rates
from attitude_supervision_v42 import gravity_direction

OUT=base.ROOT/'docs/analysis/results/control_windows_v42'
V40=OUT.parent/'control_windows_v40'


def errors(quaternion,truth):
    qp=torch.as_tensor(quaternion,dtype=torch.float64);qt=torch.as_tensor(truth,dtype=torch.float64)
    gp=gravity_direction(qp).numpy();gt=gravity_direction(qt).numpy()
    tilt=np.arctan2(np.linalg.norm(np.cross(gp,gt),axis=-1),(gp*gt).sum(-1))
    qp=qp.numpy()/np.linalg.norm(qp.numpy(),axis=-1,keepdims=True)
    qt=qt.numpy()/np.linalg.norm(qt.numpy(),axis=-1,keepdims=True)
    shape=qp.shape[:2];ap=angles(qp.reshape(-1,4)).reshape(*shape,2);at=angles(qt.reshape(-1,4)).reshape(*shape,2)
    delta=np.arctan2(np.sin(ap-at),np.cos(ap-at))
    return np.stack([delta[:,:,0],delta[:,:,1],tilt],-1)**2


def main():
    torch.set_num_threads(1)
    assert read(OUT/'protocol.json')['status']=='frozen_before_audit_and_calibration'
    with np.load(DATA/'windows.npz') as z:
        iq=z['input_initial_quaternion'];ir=z['input_initial_rates_b'];dt=z['input_dt_s'];rates=z['label_rates_b'];truth=z['label_quaternion']
    names=['native','true_p','true_q','true_r','true_pq','true_pr','true_qr','true_pqr','remove_window_rate_bias']
    axis_sets=[(),(0,),(1,),(2,),(0,1),(0,2),(1,2),(0,1,2),None]
    rows=[];count=0
    for fold in range(5):
        ids=read(DATA/f'fold{fold}/manifest.json')['query_indices']
        dest=OUT/f'kinematics/fold{fold}';dest.mkdir(parents=True)
        for variant,_,_ in VARIANTS:
            for stage in [30,35]:
                for seed in [17,23,42]:
                    source=V40/f'fold{fold}/{variant}_v{stage}_seed{seed}.npz'
                    with np.load(source) as z:pred_rates=z['rates_b'];pred_q=z['quaternion']
                    all_metrics=[]
                    for name,axes in zip(names,axis_sets):
                        mixed=pred_rates.copy()
                        if axes is None:mixed=mixed-(mixed-rates[ids]).mean(1,keepdims=True)
                        else:
                            for axis in axes:mixed[:,:,axis]=rates[ids,:,axis]
                        integrated=integrate_rates(iq[ids],ir[ids],mixed,dt[ids])
                        if name=='native':np.testing.assert_array_equal(integrated,pred_q);count+=len(ids)
                        squared=errors(integrated,truth[ids]);all_metrics.append(squared)
                        for h in [25,50]:
                            for kind,val in [('wave',squared[:,:h].mean(1)),('endpoint',squared[:,h-1])]:
                                for axis,label in enumerate(['roll','pitch','tilt']):rows.append(dict(fold=fold,variant=variant,stage=stage,seed=seed,
                                    oracle=name,horizon=h,kind=kind,metric=label,rmse=float(np.sqrt(val[:,axis].mean()))))
                    np.savez_compressed(dest/f'{variant}_v{stage}_seed{seed}.npz',array_index=ids,angle_squared=np.stack(all_metrics,2))
        print(f'kinematics fold{fold}: native quaternion parity and all9 rate tapes complete',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'kinematics_per_flight.csv',index=False)
    frame.groupby(['variant','stage','oracle','horizon','kind','metric']).rmse.mean().to_csv(OUT/'kinematics_summary.csv')
    a=frame[(frame.variant=='warm2_plus')&(frame.stage==35)&(frame.metric=='tilt')&(frame.horizon==50)&(frame.kind=='wave')]
    means=a.groupby('oracle').rmse.mean();byflight=a.groupby(['fold','oracle']).rmse.mean().unstack()
    ratio=float(means.true_pqr/means.native);flights=int((byflight.true_pqr<byflight.native).sum())
    passed=ratio<=.5 and flights>=4
    base.write(OUT/'kinematics_verification.json',dict(status='passed',native_quaternions_exact=count,
        oracle_tapes_per_trajectory=9,whole_true_rate_tilt_to_native_ratio=ratio,true_rate_improving_flights=flights,
        preparation_gate=passed,gate='Predeclared primary warm2_plus v35 true-rate tilt RMSE <=half native and >=4/5 flights lower',
        interpretation='Logged rates/quaternions are not perfectly kinematically consistent; true-rate residual is not an irreducible physical noise bound. Oracles do not alter model/controller trajectories.'))
    assert passed,'Resolve kinematic/label consistency before choosing supervision.'


if __name__=='__main__':main()
