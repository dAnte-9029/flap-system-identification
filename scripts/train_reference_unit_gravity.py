"""Correct the decoder's gravity units, then rerun frozen data/budget factors."""
import json
import time
import shutil
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from control_windows_v3 import OUT as PARENT,SEEDS,predict
from system_identification.models.reference_gru_unit_gravity import ReferenceGRUUnitGravity,unit_state_features

OUT=PARENT/'unit_gravity'
ART=base.ROOT/'artifacts/control_windows_v3_unit_gravity'


def model_from(path):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck.get('feature_contract')!='unit_gravity_v1':raise ValueError('wrong decoder gravity contract')
    keys=['state_mean','state_scale','reference_mean','reference_scale','actuator_mean','actuator_scale','rate_scale','velocity_scale']
    model=ReferenceGRUUnitGravity({k:ck['state_dict'][k] for k in keys})
    model.load_state_dict(ck['state_dict']);model.eval();return model


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    OUT.mkdir(exist_ok=False);(OUT/'training').mkdir()
    x,y,stats=base.load()
    for name in ['arm_A.csv','arm_A_long.csv','arm_B.csv','arm_C.csv','arm_D.csv','evaluation_windows.csv']:
        shutil.copyfile(PARENT/name,OUT/name)
    shutil.copyfile(base.ROOT/'src/system_identification/models/reference_gru.py',OUT/'legacy_reference_gru_snapshot.py')
    h=x['history_state'][:,-1];zero=torch.zeros(len(h))
    corrected=unit_state_features(x['initial_velocity_n'],x['initial_quaternion'],x['initial_rates_b'],zero,zero,h[:,11])
    mismatch=(corrected-h[:,:12]).abs().max().item()
    assert mismatch<1e-5,mismatch
    protocol=json.loads((PARENT/'training/protocol.json').read_text())
    protocol.update(feature_contract='unit_gravity_v1',corrected_decoder='divide gravity acceleration by9.80665 before applying unit-direction normalization',
        previous_experiments='v2 and legacy v3 used mismatched history/decoder gravity units; retained for implementation diagnosis, not an adequate-model/data-insufficiency verdict',
        architecture='same GRU64 and40070parameters; same initialization and loss; only feature units corrected',
        runs='A/B/C/D800 x3seeds plus A_long1600 x3seeds; A800 checkpoint is fixed intermediate of identical A_long trajectory',
        checkpoint='fixed800/1600 only; no validation choice; continuing identical optimizer is equivalent to same1600-step initialized run',
        all_origin_feature_max_abs_difference=mismatch,script_sha256=base.sha(__file__),
        model_sha256=base.sha(base.ROOT/'src/system_identification/models/reference_gru_unit_gravity.py'),
        selection_sha256=base.sha(PARENT/'selection_protocol.json'))
    base.write(OUT/'training/protocol.json',protocol)
    base.write(OUT/'manifest.json',dict(parent_selection_sha256=base.sha(PARENT/'selection_protocol.json'),
        source_dataset_manifest_sha256=base.sha(base.DATA/'manifest.json'),
        arrays='dataset/control_windows_v2/windows.npz',normalization='dataset/control_windows_v2/normalization.npz',
        feature_contract='unit_gravity_v1',sealed_test_accessed=False))
    evalmeta=pd.read_csv(OUT/'evaluation_windows.csv');evaluation=evalmeta.array_index.to_numpy()
    completed=[]
    try:
        for arm,maximum in [('A',1600),('B',800),('C',800),('D',800)]:
            cohort=pd.read_csv(OUT/f'arm_{arm}.csv');indices=cohort.array_index.to_numpy()
            probabilities=cohort.sampling_probability.to_numpy();probabilities/=probabilities.sum()
            for seed in SEEDS:
                torch.manual_seed(seed);rng=np.random.default_rng(seed);model=ReferenceGRUUnitGravity(stats)
                optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
                exposure=np.zeros(len(cohort),dtype=int);curve=[];started=time.monotonic()
                base.write(OUT/'training/status.json',dict(status='running',arm=arm,seed=seed,completed=completed))
                for step in range(1,maximum+1):
                    local=rng.choice(len(indices),32,replace=True,p=probabilities);np.add.at(exposure,local,1);ix=indices[local]
                    model.train();optimizer.zero_grad(set_to_none=True)
                    loss=base.loss_fn(model(**{k:v[ix] for k,v in x.items()}),{k:v[ix] for k,v in y.items()},model)
                    assert torch.isfinite(loss)
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                    if step==1 or step%50==0:
                        row=dict(arm=arm,seed=seed,step=step,loss=float(loss.detach()),gradient_norm=float(norm),elapsed_s=time.monotonic()-started)
                        curve.append(row);print(json.dumps(row),flush=True)
                    if step in [800,1600]:
                        label='A_long' if step==1600 else arm
                        directory=ART/f'{label}_{seed}';directory.mkdir(parents=True,exist_ok=False)
                        checkpoint=directory/'final.pt'
                        torch.save(dict(state_dict=model.state_dict(),feature_contract='unit_gravity_v1',arm=label,seed=seed,steps=step,
                            architecture='ReferenceGRU64UnitGravity',protocol_sha256=base.sha(OUT/'training/protocol.json')),
                            checkpoint)
                        pd.DataFrame(curve).to_csv(directory/'training_curve.csv',index=False)
                        cohort.assign(draws=exposure).to_csv(directory/'exposure.csv',index=False)
                        model.eval();prediction=predict(model,x,evaluation)
                        np.savez_compressed(directory/'validation_predictions.npz',array_index=evaluation,**prediction)
                        manifest=dict(arm=label,seed=seed,steps=step,elapsed_s=time.monotonic()-started,
                            checkpoint=str(checkpoint.relative_to(base.ROOT)),checkpoint_sha256=base.sha(checkpoint),
                            predictions_sha256=base.sha(directory/'validation_predictions.npz'),feature_contract='unit_gravity_v1')
                        base.write(directory/'manifest.json',manifest);completed.append(manifest)
                        base.write(OUT/'training/status.json',dict(status='running',arm=arm,seed=seed,completed=completed))
        base.write(OUT/'training/status.json',dict(status='complete',completed=completed))
    except Exception as exc:
        base.write(OUT/'training/status.json',dict(status='failed',error=repr(exc),completed=completed));raise


if __name__=='__main__':main()
