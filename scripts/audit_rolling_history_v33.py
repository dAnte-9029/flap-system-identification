"""Frozen rolling-H26 one-step evaluation and hidden/context crossed controls."""
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
from rolling_history_v33 import rolling_windows,crossed_step,VARIANTS
from rollout_trace_v32 import accumulated_rates

OUT=base.ROOT/'docs/analysis/results/control_windows_v33'
DATA=OUT.parent/'control_windows_v29';V23=OUT.parent/'control_windows_v23'
V28=OUT.parent/'control_windows_v28';V30=OUT.parent/'control_windows_v30';V32=OUT.parent/'control_windows_v32'


def read(p):return json.loads(p.read_text())


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    parents=[DATA,V23,V28,V30,V32]
    for root in parents:verify_parent(root)
    paths=['scripts/audit_rolling_history_v33.py','scripts/rolling_history_v33.py','tests/test_rolling_history_v33.py',
        'tests/test_rollout_trace_v32.py','src/system_identification/data/reference_windows.py',
        'src/system_identification/models/actuator_command_gru.py','src/system_identification/models/reference_gru_unit_gravity.py',
        'src/system_identification/models/trajectory_main_v1.py','scripts/rollout_trace_v32.py']
    pins={p:base.sha(base.ROOT/p) for p in paths}
    OUT.mkdir(exist_ok=False)
    parentdata=read(DATA/'dataset_manifest.json')
    base.write(OUT/'protocol.json',dict(stage='v33 frozen rolling H26 history diagnostic',status='frozen_before_diagnostic',code=pins,
        parents={str(r.relative_to(base.ROOT)):base.sha(r/'delivery_manifest.json') for r in parents},
        scope='All407 original train parents, same5flight folds and3seeds; v23/v30 frozen models. 50rolling anchors per parent are correlated diagnostic observations, NOT20350 independent training windows. No new partition/filter/normalization/training.',
        extraction='Use pinned original five train contract caches, v23 source_points for rate references, v28 actuator_grid. Reuse extract_window(history26,horizon1) at origin+k,k0..49. Each support lies within original origin-26..+50. Check original validity/reset/mode/gap rules at every anchor. History state/reference/command origin+k-25..origin+k, lagged actuator origin+k-26..origin+k-1; current left-endpoint command atorigin+k, targetorigin+k+1.',
        phase='Only existing cached logged mechanical phase; reanchor all26history phases to each rolling origin. Native decoder starts relativephase0 and holds frequency/airspeed from latest rolling history. Not April ratio8 or newly reconstructed physical phase. Future local observations are available only as rolling time advances, not at original forecast origin.',
        variants=dict(rolling_current='Native rolling H26 re-encoding with latest causal phase/frequency/airspeed context. Deployment-shaped ONE STEP only.',
            legacy_current='Use v32 true-state-recursive pre-update hidden, with native rolling current recurrent input.',
            rolling_legacy='Use rolling H26 encoder hidden with v32 true-state-recursive recurrent input (original held phase/frequency/air).',
            legacy_legacy='Replay v32 true-state-recursive hidden/input pair exactly, giving a bridge to previous stage.'),
        contrasts='Cross hidden-source and decoder-context factors. Hidden-source changes whole encoder history including historyphase/context; cannot interpret as pure abstract memory effect. Other3branches are off-distribution diagnostic combinations, not deployable models.',
        metrics='Local20ms p/q/r MSE versus same-point previous true rate persistence, finite-difference interval derivative bias, and offline accumulated predicted derivative from parent initial rate. Oracle derivative sum is not a self-consistent physical or1s prediction. Equal-flight/seed weighting; no control promotion or new pass thresholds.',
        verification='Every rollingorigin0 input/label equals parent; all50targets equal parent; allcurrent physical states equal previous targets; source/timestamp lineage; originalfirststep exact; native direct forward matches new branch; legacy branch exactly reproducesv32 raw/derivative/rates; native input difference restricted to context apart from quaternion normalization roundoff; unique407perseed; hashes unchanged.',
        phase_contract=parentdata['phase_contract'],frequency_contract=parentdata['frequency_contract'],sources=parentdata['sources'],
        training_started=False,model_promoted=False,baseline_retained='A1600',validation_accessed=False,sealed_test_accessed=False,reserved_accessed=False))
    w=pd.read_csv(DATA/'windows.csv');assert len(w)==407 and w.partition.eq('train').all()
    with np.load(V23/'windows.npz') as z:oldb={k:z[k] for k in z.files}
    with np.load(DATA/'windows.npz') as z:olda={k:z[k] for k in z.files}
    with np.load(V28/'actuator_grid.npz') as z:commands=z['command_grid'];times=z['query_us']
    sourcepoints=pd.read_csv(V23/'source_points.csv')
    inputs={};labels={};lineage=[]
    for source in parentdata['sources']:
        assert source['source']['partition']=='train'
        path=base.ROOT/source['cache'];assert base.sha(path)==source['cache_sha256']
        with np.load(path) as z:cache={k:z[k] for k in z.files}
        points=sourcepoints[sourcepoints.log_id==source['log_id']]
        pi=points.cache_index.to_numpy();np.testing.assert_array_equal(cache['timestamp_us'][pi],points.timestamp_us)
        assert (points.publication_us<=points.timestamp_us).all()
        cache['references']=np.full((len(cache['timestamp_us']),5),np.nan,dtype=np.float32)
        cache['references'][pi]=points[['p_sp','q_sp','r_sp','thrust','reset_integral']].to_numpy(np.float32)
        for row in w[w.log_id==source['log_id']].itertuples():
            ix=row.array_index;origin=row.origin_index
            np.testing.assert_array_equal(cache['timestamp_us'][origin-26:origin+51],times[ix])
            np.testing.assert_array_equal(cache['actuator_commands'][origin-26:origin+51],commands[ix])
            x,y=rolling_windows(cache,origin)
            for k,v in x.items():
                old=olda if 'command' in k else oldb
                reference=old['input_'+k][ix]
                if k.startswith('future_') or k=='dt_s':reference=reference[:1]
                np.testing.assert_array_equal(v[0],reference)
            for k,v in y.items():np.testing.assert_array_equal(v[:,0],oldb['label_'+k][ix])
            for key,label in [('initial_quaternion','quaternion'),('initial_velocity_n','velocity_n'),('initial_rates_b','rates_b')]:
                np.testing.assert_array_equal(x[key][1:],oldb['label_'+label][ix,:-1])
            inputs[ix]=x;labels[ix]=y
            for offset in range(50):lineage.append(dict(array_index=ix,log_id=row.log_id,offset=offset,
                history_start_us=int(times[ix,1+offset]),origin_us=int(times[ix,26+offset]),
                target_us=int(times[ix,27+offset]),source_cache=source['cache'],source_origin_index=origin+offset))
    arrays={k:np.stack([inputs[i][k] for i in range(407)]) for k in inputs[0]}
    truth={k:np.stack([labels[i][k][:,0] for i in range(407)]) for k in labels[0]}
    np.savez_compressed(OUT/'rolling_inputs.npz',**{'input_'+k:v for k,v in arrays.items()},**{'label_'+k:v for k,v in truth.items()})
    pd.DataFrame(lineage).to_csv(OUT/'lineage.csv',index=False)
    base.write(OUT/'dataset_manifest.json',dict(dataset_id='rolling_h26_diagnostic_v33',partition='original_train_only',
        parent_windows=407,rolling_anchors=20350,independent_new_training_windows=0,
        parent_manifest_sha256=base.sha(DATA/'dataset_manifest.json'),arrays_sha256=base.sha(OUT/'rolling_inputs.npz'),
        lineage_sha256=base.sha(OUT/'lineage.csv'),sources=parentdata['sources'],
        phase_contract=parentdata['phase_contract'],frequency_contract=parentdata['frequency_contract']))
    rows=[];clips=[];checks=[]
    for fold in range(5):
        ids=np.array(read(DATA/f'fold{fold}/manifest.json')['query_indices'])
        for seed in [17,23,42]:
            for arm,root,loader in [('baseline',V23,reference_model),('actuator',V30,actuator_model)]:
                checkpoint=root/f'fold{fold}/seed{seed}/final.pt';digest=base.sha(checkpoint);model=loader(checkpoint)
                with np.load(V32/f'fold{fold}/{arm}/seed{seed}/trace.npz') as z:
                    np.testing.assert_array_equal(z['array_index'],ids)
                    oldh=z['true_state_recursive_hidden_hidden'];encoded=z['encoded_hidden'];oldinput=z['true_state_recursive_hidden_recurrent_input']
                    oldrate=z['true_state_recursive_hidden_rates_b'];oldderiv=z['true_state_recursive_hidden_derivative'];oldraw=z['true_state_recursive_hidden_raw_head']
                trace_steps={};physical_roundoff=0.
                for offset in range(50):
                    batches={}
                    for start in range(0,len(ids),64):
                        loc=slice(start,start+64);ii=ids[loc]
                        keys=set(arrays)-({'history_reference','future_reference'} if arm=='actuator' else {'history_command','future_command'})
                        batch={k:torch.tensor(arrays[k][ii,offset]) for k in keys}
                        hidden=encoded[loc] if offset==0 else oldh[loc,offset-1]
                        result=crossed_step(model,batch,torch.tensor(hidden),torch.tensor(oldinput[loc,offset]))
                        np.testing.assert_array_equal(result['legacy_legacy_raw_head'],oldraw[loc,offset])
                        np.testing.assert_array_equal(result['legacy_legacy_derivative'],oldderiv[loc,offset])
                        np.testing.assert_array_equal(result['legacy_legacy_rates_b'],oldrate[loc,offset])
                        # Physical state normalization may differ by tiny quaternion renormalization rounding.
                        diff=abs(result['current_input'][:,:9]-oldinput[loc,offset,:9]);physical_roundoff=max(physical_roundoff,float(diff.max()))
                        np.testing.assert_allclose(result['current_input'][:,:9],oldinput[loc,offset,:9],rtol=2e-6,atol=2e-5)
                        np.testing.assert_array_equal(result['current_input'][:,14:],oldinput[loc,offset,14:])
                        if offset==0:
                            for variant in VARIANTS:
                                for key in ['rates_b','derivative','raw_head']:
                                    np.testing.assert_array_equal(result[variant+'_'+key],result['legacy_legacy_'+key])
                        for k,v in result.items():assert np.isfinite(v).all();batches.setdefault(k,[]).append(v)
                    for k,v in batches.items():trace_steps.setdefault(k,[]).append(np.concatenate(v))
                values={k:np.stack(v,1) for k,v in trace_steps.items()}
                dest=OUT/f'fold{fold}/{arm}/seed{seed}';dest.mkdir(parents=True)
                np.savez_compressed(dest/'trace.npz',array_index=ids,**values)
                actual=truth['rates_b'][ids].astype(float);initial=arrays['initial_rates_b'][ids,0].astype(float)
                dt=arrays['dt_s'][ids,:,0].astype(float);previous=arrays['initial_rates_b'][ids].astype(float)
                measured=(actual-previous)/dt[:,:,None]
                for variant in VARIANTS:
                    prediction=values[variant+'_rates_b'].astype(float)
                    integral=accumulated_rates(initial,values[variant+'_derivative'],dt)
                    raw=values[variant+'_raw_head']
                    clips.append(dict(fold=fold,seed=seed,arm=arm,variant=variant,clipped_components=int((abs(raw)>6).sum()),max_raw_abs=float(abs(raw).max())))
                    for j,axis in enumerate(['p','q','r']):
                        local=prediction[:,:,j]-actual[:,:,j];accum=integral[:,:,j]-actual[:,:,j]
                        derr=values[variant+'_derivative'][:,:,3+j].astype(float)-measured[:,:,j]
                        df=w.iloc[ids][['array_index','log_id','mode']].copy().assign(fold=fold,seed=seed,arm=arm,variant=variant,axis=axis)
                        for k,v in dict(local_mse=(local**2).mean(1),accumulated_mse=(accum**2).mean(1),
                            accumulated_end_mse=accum[:,-1]**2,derivative_bias=derr.mean(1),
                            persistence_mse=((previous[:,:,j]-actual[:,:,j])**2).mean(1)).items():df[k]=v
                        rows.append(df)
                assert base.sha(checkpoint)==digest
                checks.append(dict(fold=fold,seed=seed,arm=arm,checkpoint_sha256=digest,unchanged=True,
                    original_origin_exact=True,all50_legacy_steps_exact=True,native_rolling_forward_exact=True,
                    physical_input_roundoff_max=physical_roundoff))
            print(f'fold{fold} seed{seed}: rolling histories and crossed paths complete',flush=True)
    metrics=pd.concat(rows,ignore_index=True);metrics.to_csv(OUT/'per_window.csv',index=False)
    pd.DataFrame(clips).to_csv(OUT/'clipping.csv',index=False)
    for _,g in metrics.groupby(['seed','arm','variant','axis']):np.testing.assert_array_equal(np.sort(g.array_index),np.arange(407))
    old=pd.read_csv(V32/'per_window.csv');old=old[old.variant=='true_state_recursive_hidden']
    new=metrics[metrics.variant=='legacy_legacy'].merge(old,on=['array_index','log_id','mode','fold','seed','arm','axis'],suffixes=('_new','_old'),validate='one_to_one')
    for a,b in [('local_mse','output_mse'),('accumulated_mse_new','accumulated_mse_old'),('derivative_bias_new','derivative_bias_old')]:
        np.testing.assert_allclose(new[a],new[b],rtol=1e-10,atol=1e-12)
    test=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_rolling_history_v33.py'],text=True,capture_output=True,cwd=base.ROOT)
    base.write(OUT/'test_results.json',dict(returncode=test.returncode,stdout=test.stdout,stderr=test.stderr));assert test.returncode==0,test.stdout+test.stderr
    base.write(OUT/'verification.json',dict(status='passed',checks=checks,tests_passed=3,parent407_windows_preserved=True,
        all20350targets_equal_parent=True,all_rollingorigins_causal_and_within_support=True,
        legacy_v32_metrics_reproduced=True,no_new_training_windows=True,validation_accessed=False,sealed_test_accessed=False))
    for root in parents:verify_parent(root)
    for name,digest in pins.items():assert base.sha(base.ROOT/name)==digest
    print('v33 rolling diagnostic complete; no training',flush=True)


if __name__=='__main__':main()
