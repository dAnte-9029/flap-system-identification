"""Freeze v8 before training; materialize fold-local statistics and phase predictors."""
import json
from pathlib import Path
import numpy as np
import train_control_windows_v2 as base
from reference_groupcv_v8 import OUT,PHASE,SEEDS,load_train,fit_normalization,crossfit_phase,phase_start
from run_reference_generalization_v7 import verify_manifest


def main():
    OUT.mkdir(exist_ok=False)
    parent=OUT.parent/'control_windows_v7'
    verify_manifest(parent/'delivery_manifest.json',['artifacts','code'])
    manifest=json.loads((base.DATA/'manifest.json').read_text())
    sources=[base.DATA/'manifest.json',base.ROOT/'dataset/control_windows_v2/windows.npz',
        parent/'delivery_manifest.json',PHASE,Path(__file__),base.ROOT/'scripts/reference_groupcv_v8.py',
        base.ROOT/'scripts/train_reference_groupcv_v8.py',base.ROOT/'scripts/run_reference_groupcv_v8.py',
        base.ROOT/'src/system_identification/models/reference_gru_phase_schedule.py',
        base.ROOT/'tests/test_reference_groupcv_v8.py']
    protocol=dict(task='whole-flight OOF dynamics and consistently trained phase schedule',
        source_pins={str(p.relative_to(base.ROOT)):base.sha(p) for p in sources},
        data_authority=str((base.DATA/'manifest.json').relative_to(base.ROOT)),phase_contract=manifest['phase_contract'],frequency_contract=manifest['frequency_contract'],
        partitions='CV only original407trainingwindows/5flights; no new splitting or screening; developmentvalidation not used for CV choices; sealed/reserved not accessed',
        arms=['hold','phase'],seeds=SEEDS,folds=5,steps=1600,batch=32,hidden=64,parameters=40070,
        optimizer='AdamW lr3e-4 wd1e-4 clip1; same original normalized rate+velocity+0.2attitude loss',
        sampling='equal flight probability, uniform windows within each fit flight; paired arms same RNG draws, same initialization and fold normalization; fixed final checkpoint',
        normalization='exact v2 unique-row recipe refit on4fitflights; same normalization for paired arms; no heldflight statistic or old GRU warm start',
        phase_method='ridge10 fixed from v7; no new search. For each outerfold training flight, phase predictor fit only other3fitflights. Outerquery predicted from4fitflights. No own-flight future phase in training features.',
        phase_scope='train/inference history-predicted phase schedules; recurrent frequency context still heldcurrent. crossfit stacking uses3flight phase fits forGRUtraining and4forouterquery; not a full frequency-model change',
        selection_limit='ridge10 family informed by prior v7 analysis of these trainingflights; current OOF is development evidence, not pristine untouched validation',
        cv_primary='activep/q endpoints0.5/1s, equal available flights then axis/horizon then seeds; no window counting as independent trials',
        cv_gate='>=5percent mean improvement, all3seedpositive, >=4of5flightpositive, noallmetric>5percent regression .5/1s; phase candidate full refit only if gate passes',
        conditional_full='if CV gate passes train phase all407 x3seed1600 using5flight crossfit schedules; compare fixed developmentvalidation against frozenA1600. Otherwise do not run fullcandidate training.',
        validation_diagnostic='after CV decision is frozen, evaluate15heldfrequencyfoldmodels on same739developmentvalidation windows, to compare heldsame-day vs nextday at matched training size/model; not a causal day effect',
        control_gate='absolute0.5s activep/q<=.15rad/s attitude<=3deg velocity<=.5m/s allseeds plus independentresponse/closedloop evidence. OOF RMSE alone does not establish control usability',
        bootstrap='2000 paired wholeflight resamples seed908; five flights only, exploratory interval',
        sealed_test_accessed=False)
    base.write(OUT/'protocol.json',protocol)
    x,y,w=load_train();h=x['history_state'].numpy();logs=w.log_id.to_numpy();ids=w.array_index.to_numpy()
    cm=json.loads((base.ROOT/'docs/analysis/results/reference_dynamics_data_v1/training_contract/manifest.json').read_text())
    arrays={};cache_pins={}
    for log,g in w.groupby('log_id'):
        rel=g.cache.iloc[0];path=base.ROOT/rel;assert base.sha(path)==cm['cache_sha256'][rel];cache_pins[rel]=base.sha(path)
        with np.load(path) as z:arrays[log]={k:z[k] for k in ['state_features','references','actuator_commands','rates_b']}
    with np.load(PHASE) as z:
        np.testing.assert_array_equal(z['array_index'][ids],ids);phase=z['phase'][ids]
    original=np.load(base.ROOT/'dataset/control_windows_v2/normalization.npz')
    all_stats=fit_normalization(arrays,w,y['velocity_n'].numpy())
    differences={k:float(np.max(abs(all_stats[k]-original[k]))) for k in all_stats};assert max(differences.values())<1e-6,differences
    folds=[]
    for fold,log in enumerate(sorted(set(logs))):
        fit=ids[logs!=log];query=ids[logs==log];dest=OUT/f'fold{fold}';dest.mkdir()
        stats=fit_normalization(arrays,w.iloc[fit],y['velocity_n'].numpy());np.savez(dest/'normalization.npz',**stats)
        a,b,provenance,parameters=crossfit_phase(h,phase,logs,fit,query)
        schedule=np.zeros((len(w),50),np.float32);schedule[fit]=phase_start(a);schedule[query]=phase_start(b)
        np.savez_compressed(dest/'phase_schedules.npz',array_index=ids,phase_start=schedule,fit_indices=fit,query_indices=query)
        for n,(p,pars) in enumerate(zip(provenance,parameters)):
            np.savez_compressed(dest/f'phase_predictor{n}.npz',**p,**pars)
        info=dict(fold=fold,held_flight=log,fit_flights=sorted(set(logs[fit])),fit_indices=fit.tolist(),query_indices=query.tolist(),
            normalization_sha256=base.sha(dest/'normalization.npz'),phase_schedules_sha256=base.sha(dest/'phase_schedules.npz'),phase_provenance=provenance)
        base.write(dest/'manifest.json',info);folds.append(info)
    base.write(OUT/'preparation.json',dict(status='complete',folds=folds,full_normalization_reproduction=differences,
        cache_pins=cache_pins,all_fit_partition='train',sealed_test_accessed=False))
    print('Prepared5folds; normalization reproduction:',differences)

if __name__=='__main__':main()
