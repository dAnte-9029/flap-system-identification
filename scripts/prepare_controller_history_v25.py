"""Execute the already frozen v24 design; pin implementation before training."""
from pathlib import Path
import json
import numpy as np
import torch
import train_control_windows_v2 as base
from controller_history_training_v25 import OUT,V23,V24,ARMS,load,model,fingerprint
from reference_groupcv_v8 import ReferenceGRUPhaseSchedule


def main():
    torch.set_num_threads(1)
    for root in [V23,V24]:
        d=json.loads((root/'delivery_manifest.json').read_text())
        for f,h in d['artifacts'].items():assert base.sha(root/f)==h
        for f,h in d['code'].items():assert base.sha(base.ROOT/f)==h
    p=json.loads((V24/'training_protocol.json').read_text())
    assert p['runs']==30 and p['steps']==1600 and p['arms']==ARMS
    paths=[Path(__file__),base.ROOT/'scripts/controller_history_training_v25.py',base.ROOT/'scripts/train_controller_history_v25.py',
        base.ROOT/'scripts/run_controller_history_v25.py',base.ROOT/'scripts/evaluate_controller_history_v25.py',
        base.ROOT/'src/system_identification/models/reference_gru_controller_history.py',base.ROOT/'tests/test_controller_history_model_v25.py',
        base.ROOT/'scripts/evaluate_rate_reference_v23.py',base.ROOT/'scripts/rate_reference_v23.py',
        base.ROOT/'src/system_identification/models/reference_gru_unit_gravity.py',base.ROOT/'scripts/train_control_windows_v2.py']
    OUT.mkdir(exist_ok=False)
    base.write(OUT/'protocol.json',dict(stage='v25',frozen_design=p,design_sha256=base.sha(V24/'training_protocol.json'),
        parent_delivery_sha256={str(r.relative_to(base.ROOT)):base.sha(r/'delivery_manifest.json') for r in [V23,V24]},
        code_pins={str(f.relative_to(base.ROOT)):base.sha(f) for f in paths},
        reference_hold='Both arms: hold future rate3/thrust to origin values, preserve reset. Diagnostic only, no real counterfactual or promotion gate.',
        evaluation='Same v24 primary/protections; same original active flag for absolute .5s limits. Mode/fit/OOF and v23 comparison secondary. No full-data candidate refit.',
        phase_contract=json.loads((V24/'dataset_manifest.json').read_text())['phase_contract'],
        frequency_contract=json.loads((V24/'dataset_manifest.json').read_text())['frequency_contract'],sealed_test_accessed=False))
    x,_=load();checks=[]
    for fold in range(5):
        with np.load(V23/f'fold{fold}/normalization.npz') as z:stats={k:z[k] for k in z.files}
        info=json.loads((V23/f'fold{fold}/manifest.json').read_text());ids=info['query_indices'][:2]
        batch={k:v[ids] for k,v in x.items()}
        for seed in [17,23,42]:
            torch.manual_seed(seed);old=ReferenceGRUPhaseSchedule(stats)
            torch.manual_seed(seed);zero=model(fold,ARMS[0])
            torch.manual_seed(seed);observed=model(fold,ARMS[1])
            assert fingerprint(zero)==fingerprint(observed)
            assert fingerprint(zero,True)==fingerprint(old)
            assert sum(p.numel() for p in zero.parameters())==41606
            with torch.no_grad():
                original=old(**{k:v for k,v in batch.items() if k!='history_controller'})
                for m in [zero,observed]:
                    predicted=m(**batch)
                    for k in original:torch.testing.assert_close(predicted[k],original[k],atol=1e-6,rtol=1e-6)
            checks.append(dict(fold=fold,seed=seed,common_initial_parameters_equal=True,both_arms_identical_initial_weights=True,initial_predictions_match_v23=True))
    base.write(OUT/'preflight.json',dict(status='passed',checks=checks,windows=407,history_shape=[407,26,8],
       scope='Only source-verified past controller history appended; inputs/labels otherwise v23; normalized zero arm.',sealed_test_accessed=False))


if __name__=='__main__':main()
