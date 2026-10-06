"""Freeze the single-step objective contrast before training."""
import json
from pathlib import Path
import train_control_windows_v2 as base
from reference_single_step_v17 import OUT,V8
from run_reference_generalization_v7 import verify_manifest


def main():
    OUT.mkdir(exist_ok=False)
    v16=OUT.parent/'control_windows_v16'
    for source in [V8,v16]:verify_manifest(source/'delivery_manifest.json',['artifacts','code'])
    authority=base.ROOT/'docs/analysis/results/reference_dynamics_data_v1/training_contract/manifest.json'
    contract=json.loads(authority.read_text())
    paths=[authority,base.DATA/'manifest.json',base.ROOT/'dataset/control_windows_v2/windows.npz',V8/'delivery_manifest.json',v16/'delivery_manifest.json',Path(__file__)]
    paths += [base.ROOT/p for p in ['scripts/reference_single_step_v17.py','scripts/train_reference_single_step_v17.py','scripts/run_reference_single_step_v17.py','tests/test_reference_single_step_v17.py','scripts/train_control_windows_v2.py','scripts/reference_groupcv_v8.py','src/system_identification/models/reference_gru_unit_gravity.py','src/system_identification/models/reference_gru_phase_schedule.py']]
    base.write(OUT/'protocol.json',dict(stage='v17 single-step objective contrast',source_pins={str(p.relative_to(base.ROOT)):base.sha(p) for p in paths},
        phase_contract=contract['phase_contract'],frequency_contract=contract['frequency_contract'],
        data='Same407 originaltrain parents/5flights/H26/K50; only same5 outerfold fit/query IDs. No new origins, no filtering, no development or sealed/reserved evaluation.',
        model='Unchanged ReferenceGRUPhaseSchedule(phase_start=None),40070parameters,units and clamps. Fresh initialization exactly matching each v8hold seed; same fold normalization and recorded1600x32 minibatch index tapes.',
        objective='Original normalized rate+velocity+0.2 quaternion loss, restricted to original first20ms target for ALL states; no extra weight or new target. Encode complete26history, run decoder1step in training. No teacher-forcing laterorigins. At evaluation run normal autonomous50steps.',
        budget='15runs:5folds xseeds17/23/42,1600AdamW updates lr3e-4 wd1e-4 clip1 batch32. Same updates/windowexposures, NOT equalFLOPs, targetpoints or gradientpath. Fixed finalcheckpoint, noearlystopping.',
        implementation_check='Before training eachrun compare full50step-forward prefix loss/parameter gradients to truncated1step forward on samefirstminibatch; initialparameter hash and tapes match v8.',
        metrics='Identical full50future targets; perflight RMSE forfirst20ms and fullprefix .1/.2/.5/1s, plus .5/1s endpoints. Perflight/seed equalweight, repeated fitfolds averaged per sourceflight. Rate increment std/correlation/sign asv16; allstates and perflight/seed retained. Normalize quaternions for angle metric to avoid norm rounding at20ms, same for botharms.',
        diagnostic_gate='Equalflight/axis(p,q)/seed first20ms RMSE improvement>=5percent; all3seed improvements>0 and>=4of5flight improvements>0. Report fit andOOF independently. Not a controller acceptance gate.',
        protection='Eachseed p/q/r/velocity/attitude fullprefix .5and1s RMSE regression<=5percent vs v8hold; finite50step trajectories allwindows. If one-step improves but protection fails, classify as short-horizon versus rollout tradeoff; no model promotion.',
        absolute='Report .5s endpoints for eachseed: p/q/r<=.15rad/s,velocity<=.5m/s,attitude<=3deg. Passing absolute and relative checks is still not independent control-response evidence.',
        inference='History and plannedreference prefix only; no future measuredstates,commands,phase/frequency input. Fit is in-sample; OOF reusedresearch not blind. No causal/noise conclusion from loss contrast alone.',
        after='No automatic all407refit, developmentevaluation or controller optimization in thisstage; freeze OOF decision first.',sealed_test_accessed=False))
    for fold in range(5):(OUT/f'fold{fold}').mkdir()
    print('v17 protocol frozen')


if __name__=='__main__':main()
