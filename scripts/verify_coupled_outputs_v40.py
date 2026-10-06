"""Recompute saved state/action metrics and causal support flags independently."""
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from run_coupled_replay_v40 import OUT,DATA,read
from coupled_replay_v40 import VARIANTS
from system_identification.models.reference_gru_unit_gravity import unit_state_features


def main():
    torch.set_num_threads(1)
    assert read(OUT/'verification.json')['status']=='passed'
    sm=pd.read_csv(OUT/'state_metrics_per_window.csv').set_index(['fold','variant','stage','seed','horizon','kind','metric','array_index']).sort_index()
    am=pd.read_csv(OUT/'action_metrics_per_window.csv').set_index(['fold','variant','stage','seed','comparison','axis','array_index']).sort_index()
    with np.load(DATA/'windows.npz') as z:
        x={k:z[k] for k in z.files}
    metric_values=0;support_points=0
    for fold in range(5):
        ids=read(DATA/f'fold{fold}/manifest.json')['query_indices'];fit=read(DATA/f'fold{fold}/manifest.json')['fit_indices']
        s=read(OUT/f'fold{fold}/support.json')
        for key,expected in [('command_min',x['input_future_command'][fit].min((0,1))),('command_max',x['input_future_command'][fit].max((0,1))),
                             ('state_min',x['input_history_state'][fit].min((0,1))),('state_max',x['input_history_state'][fit].max((0,1)))]:
            np.testing.assert_array_equal(s[key],expected)
        for variant,_,_ in VARIANTS:
            with np.load(OUT/f'fold{fold}/{variant}_teacher.npz') as z:teacher=z['commands']
            for stage in [30,35]:
                for seed in [17,23,42]:
                    with np.load(OUT/f'fold{fold}/{variant}_v{stage}_seed{seed}.npz') as z:a={k:z[k] for k in z.files}
                    squared={axis:(a['rates_b'].astype(float)[:,:,j]-x['label_rates_b'][ids,:,j].astype(float))**2 for j,axis in enumerate('pqr')}
                    squared['velocity']=((a['velocity_n'].astype(float)-x['label_velocity_n'][ids].astype(float))**2).mean(-1)
                    dot=(a['quaternion'].astype(float)*x['label_quaternion'][ids].astype(float)).sum(-1)
                    squared['attitude']=(2*np.arccos(np.minimum(1,np.abs(dot))))**2
                    for horizon in [25,50]:
                        for metric,sq in squared.items():
                            for kind,values in [('wave',sq[:,:horizon].mean(1)),('endpoint',sq[:,horizon-1])]:
                                actual=sm.loc[(fold,variant,stage,seed,horizon,kind,metric)].loc[ids,'mse'].to_numpy()
                                np.testing.assert_allclose(actual,values,rtol=1e-10,atol=1e-12);metric_values+=len(ids)
                    for comparison,target in [('log',x['input_future_command'][ids]),('matched_teacher',teacher)]:
                        for axis in range(4):
                            values=((a['commands'][:,:,axis]-target[:,:,axis])**2).mean(1)
                            actual=am.loc[(fold,variant,stage,seed,comparison,axis)].loc[ids,'mse'].to_numpy()
                            np.testing.assert_allclose(actual,values,rtol=1e-10,atol=1e-12);metric_values+=len(ids)
                    # State at timek uses only initial state or the preceding generated output.
                    q=torch.tensor(np.concatenate([x['input_initial_quaternion'][ids,None],a['quaternion'][:,:49]],1))
                    q[:,0]=q[:,0]/q[:,0].norm(dim=-1,keepdim=True).clamp_min(1e-8)
                    v=torch.tensor(np.concatenate([x['input_initial_velocity_n'][ids,None],a['velocity_n'][:,:49]],1))
                    r=torch.tensor(np.concatenate([x['input_initial_rates_b'][ids,None],a['rates_b'][:,:49]],1))
                    hist=torch.tensor(x['input_history_state'][ids]);freq=hist[:,-1,11].clone();phase=torch.zeros_like(freq);anchor=phase.clone()
                    for k in range(50):
                        f=torch.cat([unit_state_features(v[:,k],q[:,k],r[:,k],phase,anchor,freq),hist[:,-1,12:14]],-1).numpy()
                        outside=(f<s['state_min'])|(f>s['state_max'])
                        np.testing.assert_array_equal(outside,a['state_outside'][:,k])
                        phase=phase+2*torch.pi*freq*torch.full_like(freq,.02)
                        support_points+=len(ids)
                    np.testing.assert_array_equal((a['commands']<s['command_min'])|(a['commands']>s['command_max']),a['command_outside'])
                    # Applied action is exactly the left endpoint; no output clipping in evaluator.
                    np.testing.assert_array_equal(a['commands'],a['micro_commands'][:,:,0])
    base.write(OUT/'independent_verification.json',dict(status='passed',per_window_metric_values_recomputed=metric_values,
        causal_generated_support_points_recomputed=support_points,fit_only_bounds_exact=True,
        applied_command_is_interval_left_endpoint=True,all_windows_retained=True))
    print('Independent saved-output metrics and causal support verification passed',flush=True)


if __name__=='__main__':main()
