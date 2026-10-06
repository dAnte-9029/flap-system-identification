"""Freeze and execute OOF state/hidden tracing for v23 and v30 models."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from audit_controller_support_v26 import verify_parent
from actuator_training_v30 import model_from as actuator_model
from reference_groupcv_v8 import model_from as reference_model
from rollout_trace_v32 import trace,accumulated_rates,VARIANTS

OUT=base.ROOT/'docs/analysis/results/control_windows_v32'
DATA=OUT.parent/'control_windows_v29';V23=OUT.parent/'control_windows_v23'
V30=OUT.parent/'control_windows_v30';V31=OUT.parent/'control_windows_v31'


def read(path):return json.loads(path.read_text())


def load(path):
    with np.load(path) as z:
        x={k[6:]:torch.tensor(z[k]) for k in z.files if k.startswith('input_')}
        y={k[6:]:torch.tensor(z[k]) for k in z.files if k.startswith('label_')}
    return x,y


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    for root in [DATA,V23,V30,V31]:verify_parent(root)
    names=['scripts/audit_rollout_trace_v32.py','scripts/rollout_trace_v32.py','tests/test_rollout_trace_v32.py',
           'src/system_identification/models/reference_gru_unit_gravity.py','src/system_identification/models/actuator_command_gru.py',
           'src/system_identification/models/trajectory_main_v1.py','scripts/actuator_training_v30.py','scripts/reference_groupcv_v8.py']
    pins={name:base.sha(base.ROOT/name) for name in names}
    OUT.mkdir(exist_ok=False)
    base.write(OUT/'protocol.json',dict(stage='v32 frozen OOF state/hidden rollout localization',status='frozen_before_diagnostic',code=pins,
        parents={str(r.relative_to(base.ROOT)):base.sha(r/'delivery_manifest.json') for r in [DATA,V23,V30,V31]},
        scope='All407 original train parents, 5folds x3seeds, v23 andv30 OOF checkpoints only; no model/data/timing/normalization changes, no fit training or validation/sealed/reserved access.',
        variants=dict(free='Exact original recursive physical and hidden states.',
            true_state_free_hidden='At each step, substitute previous true velocity/quaternion/rates in features AND integration anchor; branch from current free hidden before GRUCell, discard branch hidden after recording.',
            true_state_recursive_hidden='Same true physical state substitution, but recursively maintain a separate hidden state updated on that true-state path.'),
        held_context='Original histories, future references/commands, initial mode, held frequency, synthetic held-frequency phase schedule and held airspeed unchanged in all variants. No future actual phase/frequency/airspeed. At k=0 allbranches identical. At k>0 oracles use labels[k-1], never labels[k] to predict stepk.',
        tracing='Save encoder hidden, 50xhidden64, normalized recurrentinput21, rawhead6 before clamp, derivative6 after clamp, predicted velocity/quaternion/rates for each branch. Six rawhead components clipped only at[-6,6] with derivative_scale[10,10,10,20,20,20]. abs(hidden)>=.99 fraction is a descriptive near-bound statistic, not GRU gate saturation.',
        metrics='Per-step q/r derivatives vs20ms finite differences of logged rates (interval average, not instantaneous acceleration truth); rate error, derivative error, state-input divergence, hidden divergence andclipping. p also saved for completeness. Equal-flight then equal-seed summaries.',
        local='Oracle outputs reset state at every20ms step. Compare their rate one-step errors with true-previous-rate persistence, not with1s forecast accuracy. First step is common to allbranches.',
        accumulated='Separately sum each branch predicted angular accelerations from original initialrate without state resets, compare to recorded rates. Free sum reproduces stored trajectory withinfloat32 accumulation tolerance. Oracle sums expose signed increment accumulation on an observed-state/hidden path, are NOT full physically consistent or deployable rollouts.',
        contrasts='Free-vs-statebranch derivative difference conditions on same free pre-update hidden. Statebranch-vs-recursiveoracle conditions on same truephysical input but different hidden history. Nonlinear/path-dependent decomposition, not independent causal attribution or proof of unique rootcause.',
        focus='Highlight previously identified window97 without selecting/deleting data; also summarize remaining77 log18 and all407. Existing originalmodel metrics andshape contracts retained.',
        tests='Exactfree implementation; generatedtruth makes allbranches coincide; perturb futuretruth cannot alter free path or earlier oracle outputs; rawclamp andintegral unit tests. Independent actual checkpoint hooks verify traces, hashes and unchanged parameters.',
        training_started=False,model_promoted=False,baseline_retained='A1600',
        phase_contract=read(DATA/'dataset_manifest.json')['phase_contract'],frequency_contract=read(DATA/'dataset_manifest.json')['frequency_contract'],
        validation_accessed=False,sealed_test_accessed=False,reserved_accessed=False))
    xa,y=load(DATA/'windows.npz');xb,yb=load(V23/'windows.npz')
    for k in y:np.testing.assert_array_equal(y[k],yb[k])
    w=pd.read_csv(DATA/'windows.csv');assert len(w)==407 and w.partition.eq('train').all()
    rows=[];steps=[];checks=[];clipping=[]
    for fold in range(5):
        ids=np.array(read(DATA/f'fold{fold}/manifest.json')['query_indices'])
        for seed in [17,23,42]:
            for arm,root,x,loader in [('baseline',V23,xb,reference_model),('actuator',V30,xa,actuator_model)]:
                path=root/f'fold{fold}/seed{seed}/final.pt';digest=base.sha(path);model=loader(path)
                chunks={}
                for start in range(0,len(ids),64):
                    ii=ids[start:start+64];batch={k:v[ii] for k,v in x.items()};truth={k:v[ii] for k,v in y.items()}
                    result=trace(model,batch,truth)
                    # Independent hooks on the untouched model check raw head and hidden paths.
                    raw=[];hidden=[];recurrent=[]
                    hooks=[model.head.register_forward_hook(lambda m,a,o:raw.append(o.detach().numpy())),
                           model.cell.register_forward_hook(lambda m,a,o:(hidden.append(o.detach().numpy()),recurrent.append(a[0].detach().numpy())) and None)]
                    with torch.no_grad():original=model(**batch)
                    for h in hooks:h.remove()
                    for k in ['rates_b','quaternion','velocity_n']:np.testing.assert_array_equal(result['free_'+k],original[k].numpy())
                    np.testing.assert_array_equal(result['free_raw_head'],np.stack(raw,1))
                    np.testing.assert_array_equal(result['free_hidden'],np.stack(hidden,1))
                    np.testing.assert_array_equal(result['free_recurrent_input'],np.stack(recurrent,1))
                    # All variants receive the identical origin state and encoder output.
                    for variant in VARIANTS[1:]:
                        for k in ['derivative','rates_b','hidden']:np.testing.assert_array_equal(result[variant+'_'+k][:,0],result['free_'+k][:,0])
                    for k,v in result.items():
                        assert np.isfinite(v).all();chunks.setdefault(k,[]).append(v)
                result={k:np.concatenate(v) for k,v in chunks.items()}
                with np.load(root/f'fold{fold}/seed{seed}/oof_predictions.npz') as z:
                    np.testing.assert_array_equal(z['array_index'],ids)
                    for k in ['rates_b','quaternion','velocity_n']:np.testing.assert_array_equal(result['free_'+k],z[k])
                dest=OUT/f'fold{fold}/{arm}/seed{seed}';dest.mkdir(parents=True)
                np.savez_compressed(dest/'trace.npz',array_index=ids,**result)
                true=y['rates_b'][ids].numpy().astype(float);dt=x['dt_s'][ids].numpy().astype(float)
                initial=x['initial_rates_b'][ids].numpy().astype(float)
                previous=np.concatenate([initial[:,None,:],true[:,:-1]],1)
                true_derivative=(true-previous)/dt[:,:,None]
                pers_error=(previous-true)**2
                for variant in VARIANTS:
                    rates=result[variant+'_rates_b'].astype(float);derivative=result[variant+'_derivative'].astype(float)[:,:,3:]
                    accumulated=accumulated_rates(initial,result[variant+'_derivative'],dt)
                    if variant=='free':np.testing.assert_allclose(accumulated,rates,rtol=3e-6,atol=5e-6)
                    else:
                        # Independent local-error/interval-derivative identity with float32 rounding.
                        np.testing.assert_allclose(rates-true,(derivative-true_derivative)*dt[:,:,None],rtol=2e-4,atol=2e-6)
                    normstate=result[variant+'_recurrent_input'][:,:,:14].astype(float)
                    hd=result[variant+'_hidden'].astype(float)-result['free_hidden'].astype(float)
                    hidden_rms=np.sqrt(np.mean(result[variant+'_hidden'].astype(float)**2,2))
                    near=np.mean(abs(result[variant+'_hidden'])>=.99,2)
                    raw=result[variant+'_raw_head'];clipped=abs(raw)>6
                    clipping.append(dict(fold=fold,seed=seed,arm=arm,variant=variant,windows=len(ids),steps=len(ids)*50,
                        clipped_steps=int(clipped.any(2).sum()),clipped_components=int(clipped.sum()),
                        q_clipped=int(clipped[:,:,4].sum()),r_clipped=int(clipped[:,:,5].sum()),
                        max_raw_abs=float(abs(raw).max()),max_hidden_abs=float(abs(result[variant+'_hidden']).max()),
                        near_bound_hidden_fraction=float(near.mean())))
                    for j,axis in enumerate(['p','q','r']):
                        free_deriv=result['free_derivative'][:,:,3+j].astype(float)
                        error=rates[:,:,j]-true[:,:,j];accerror=accumulated[:,:,j]-true[:,:,j]
                        d=w.iloc[ids][['array_index','log_id','mode']].copy().assign(fold=fold,seed=seed,arm=arm,variant=variant,axis=axis)
                        metrics=dict(output_mse=np.mean(error**2,1),output_end_mse=error[:,-1]**2,
                            accumulated_mse=np.mean(accerror**2,1),accumulated_end_mse=accerror[:,-1]**2,
                            accumulated_half_mse=accerror[:,24]**2,accumulated_bias_mse=accerror.mean(1)**2,
                            persistence_mse=pers_error[:,:,j].mean(1),derivative_mse=np.mean((derivative[:,:,j]-true_derivative[:,:,j])**2,1),
                            derivative_bias=(derivative[:,:,j]-true_derivative[:,:,j]).mean(1),
                            hidden_delta_mse=np.mean(hd**2,(1,2)),hidden_rms=hidden_rms.mean(1),
                            state_input_delta_mse=np.mean((normstate-result['free_recurrent_input'][:,:,:14])**2,(1,2)),
                            derivative_change_mse=np.mean((derivative[:,:,j]-free_deriv)**2,1))
                        for k,v in metrics.items():d[k]=v
                        rows.append(d)
                        if axis=='p':continue
                        for n,idx in enumerate(ids):
                            log_id=w.iloc[idx].log_id
                            for k in range(50):steps.append(dict(fold=fold,seed=seed,arm=arm,variant=variant,axis=axis,
                                array_index=int(idx),log_id=log_id,step=k+1,
                                derivative=derivative[n,k,j],true_interval_derivative=true_derivative[n,k,j],
                                output_error=error[n,k],accumulated_error=accerror[n,k],
                                hidden_rms=hidden_rms[n,k],hidden_delta_rms=np.sqrt(np.mean(hd[n,k]**2)),
                                state_input_delta_rms=np.sqrt(np.mean((normstate[n,k]-result['free_recurrent_input'][n,k,:14])**2)),
                                raw_head=float(raw[n,k,3+j]),clipped=bool(clipped[n,k,3+j]),
                                hidden_near_bound_fraction=near[n,k]))
                assert base.sha(path)==digest
                checks.append(dict(fold=fold,seed=seed,arm=arm,checkpoint_sha256=digest,unchanged=True,
                    free_predictions_exact=True,independent_hooks_exact=True,first_step_all_variants_equal=True))
            print(f'fold{fold} seed{seed}: both models traced',flush=True)
    metrics=pd.concat(rows,ignore_index=True);metrics.to_csv(OUT/'per_window.csv',index=False)
    pd.DataFrame(steps).to_csv(OUT/'per_step.csv.gz',index=False,compression='gzip')
    pd.DataFrame(clipping).to_csv(OUT/'clipping.csv',index=False)
    for _,g in metrics.groupby(['seed','arm','variant','axis']):np.testing.assert_array_equal(np.sort(g.array_index),np.arange(407))
    # Reproduce parent free OOF rates under identical flight and seed weighting.
    parent=pd.read_csv(V30/'comparison.csv');parent=parent[(parent.cohort=='oof')&(parent.kind=='wave')&(parent.horizon==50)].set_index(['seed','metric'])
    for arm in ['baseline','actuator']:
        g=metrics[(metrics.arm==arm)&(metrics.variant=='free')].groupby(['seed','axis','log_id']).output_mse.mean().pow(.5).groupby(['seed','axis']).mean()
        np.testing.assert_allclose(g,parent.loc[g.index,arm],rtol=1e-12,atol=1e-12)
    test=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_rollout_trace_v32.py'],text=True,capture_output=True,cwd=base.ROOT)
    base.write(OUT/'test_results.json',dict(returncode=test.returncode,stdout=test.stdout,stderr=test.stderr));assert test.returncode==0,test.stdout+test.stderr
    base.write(OUT/'verification.json',dict(status='passed',checks=checks,tests_passed=3,all407_unique_per_seed=True,
        rate_integral_identities=True,parent_free_rate_metrics_reproduced=True,training_started=False,
        validation_accessed=False,sealed_test_accessed=False))
    for root in [DATA,V23,V30,V31]:verify_parent(root)
    for name,digest in pins.items():assert base.sha(base.ROOT/name)==digest
    print('v32 tracing complete; checkpoints unchanged',flush=True)


if __name__=='__main__':main()
