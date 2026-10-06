"""Freeze one q-auxiliary coefficient, without a search over reused OOF."""
import json
from pathlib import Path
import train_control_windows_v2 as base
from reference_q_aux_v18 import OUT,V8,AUX_WEIGHT
from run_reference_generalization_v7 import verify_manifest


def main():
    OUT.mkdir(exist_ok=False);v17=OUT.parent/'control_windows_v17'
    for source in [V8,v17]:verify_manifest(source/'delivery_manifest.json',['artifacts','code'])
    authority=base.ROOT/'docs/analysis/results/reference_dynamics_data_v1/training_contract/manifest.json';contract=json.loads(authority.read_text())
    paths=[authority,base.DATA/'manifest.json',base.ROOT/'dataset/control_windows_v2/windows.npz',V8/'delivery_manifest.json',v17/'delivery_manifest.json',Path(__file__)]
    paths += [base.ROOT/p for p in ['scripts/reference_q_aux_v18.py','scripts/train_reference_q_aux_v18.py','scripts/run_reference_q_aux_v18.py','scripts/evaluate_reference_q_aux_v18.py','tests/test_reference_q_aux_v18.py','scripts/reference_single_step_v17.py','scripts/train_control_windows_v2.py','scripts/reference_groupcv_v8.py','src/system_identification/models/reference_gru_unit_gravity.py','src/system_identification/models/reference_gru_phase_schedule.py']]
    base.write(OUT/'protocol.json',dict(stage='v18 full rollout plus first20ms q auxiliary',source_pins={str(p.relative_to(base.ROOT)):base.sha(p) for p in paths},
        phase_contract=contract['phase_contract'],frequency_contract=contract['frequency_contract'],
        data='407 original train parents, same5 outerflight folds and same normalization. No filtering, no new origins, no development or sealed/reserved access.',
        architecture='Unchanged40070parameter v8hold ReferenceGRUPhaseSchedule phase_start=None, H26,50Hz,K50,derivative bounds and gravity units unchanged.',
        objective='L=original full50step normalized rates+velocity+0.2attitude loss + (1/3)*mean_batch(((qpred_first-qtrue_first)/qscale)^2). Original full loss not renormalized or shortened. Shared weights may affect all outputs.',
        auxiliary_weight=AUX_WEIGHT,weight_rationale='One auxiliary firstq term has same coefficient as original time-mean q term. For direct q squared errors firststep coefficient becomes51/150, later49points remain1/150; total nominal qweight doubles. This is one fixed design, not tuned on currentOOF.',
        budget='15candidate runs(5folds x17/23/42) plus one fold0seed17 zeroaux identity.1600updates,batch32,AdamW3e-4wd1e-4clip1. Same v8 initialparameterhash and exact saved minibatch tapes. Finalcheckpoint only.',
        identity='weight0 fulltraining checkpoint must reproduce v8hold fold0seed17 maxabs<=1e-6. Allcandidate losses retain full50step backpropagation.',
        metrics='Equalflight then seed, repeatedfitmodels averaged by sourceflight. First20ms and fullprefix.1/.2/.5/1s plus .5/1s endpoints allstates; qincrement amplitude/correlation/sign asv16; quaternion angle normalized and stable chord formula asv17.',
        diagnostic_gate='OOF first20ms qRMSE gain>=5percent, all3seedsgain>0 and>=4/5flightsgain>0. Fit separately. q-only primary declared fromv17 beforetraining, not same asv17 jointp/q gate.',
        protection='Eachseed fullprefix.5/1s p/q/r/velocity/attitude regression<=5percent, and first20ms p/r/velocity/attitude regression<=5percent. Allpredictionsfinite.',
        reference_protection='Same OOF windows, factual vs all5refs held at first planned value. qfull1s utility=1-factualRMSE/heldRMSE must not drop>.02 perseed vsbaseline; factual-held output RMS ratio must remain[.9,1.1]. This is a model sensitivity guard, not real counterfactual controlvalidation.',
        absolute='Eachseed .5s endpoint p/q/r<=.15rad/s,velocity<=.5m/s,attitude<=3deg; predictiongates alone never constitute causal/closedloop controlvalidation.',
        after='Freeze OOF decision; no automatic full407refit, developmentevaluation, newcoefficient search or controller design. ReusedOOF researchnotblind.',sealed_test_accessed=False))
    for fold in range(5):(OUT/f'fold{fold}').mkdir()
    print('v18 protocol frozen, auxiliary coefficient=1/3')


if __name__=='__main__':main()
