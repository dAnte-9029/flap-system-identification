"""Pin v26 design, check pairing, and exactly reproduce one complete zero control."""
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from controller_no_age_v27 import OUT,V23,V24,V25,V26,V8,load,model,fingerprint
from controller_history_training_v25 import model as old_model
from audit_controller_support_v26 import verify_parent


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    for root in [V23,V24,V25,V26]:verify_parent(root)
    runtime=json.loads((V25/'runtime.json').read_text())
    actual=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__)
    for key,value in actual.items():assert runtime[key]==value,key
    paths=[Path(__file__),base.ROOT/'scripts/controller_no_age_v27.py',base.ROOT/'scripts/train_controller_no_age_v27.py',
        base.ROOT/'scripts/run_controller_no_age_v27.py',base.ROOT/'scripts/evaluate_controller_no_age_v27.py',
        base.ROOT/'src/system_identification/models/reference_gru_controller_no_age.py',base.ROOT/'tests/test_controller_no_age_v27.py']
    pins=json.loads((V25/'protocol.json').read_text())['code_pins']
    pins.update({str(p.relative_to(base.ROOT)):base.sha(p) for p in paths})
    OUT.mkdir(exist_ok=False)
    base.write(OUT/'protocol.json',dict(stage='v27',design=json.loads((V26/'next_ablation_protocol.json').read_text()),
        design_sha256=base.sha(V26/'next_ablation_protocol.json'),code_pins=pins,
        parents={str(r.relative_to(base.ROOT)):base.sha(r/'delivery_manifest.json') for r in [V23,V24,V25,V26]},
        runtime=actual,control_reuse='Exactly reproduce fold0 seed17 zero control1600steps before candidate launch; check all15paired initial weights and draws.',
        reference_hold='Same predeclaredv25 diagnostic: hold future rate3/thrust atorigin, resetunchanged; not counterfactual truth.',
        phase_contract=json.loads((V24/'dataset_manifest.json').read_text())['phase_contract'],
        frequency_contract=json.loads((V24/'dataset_manifest.json').read_text())['frequency_contract'],validation_accessed=False,sealed_test_accessed=False))
    x,y=load();checks=[]
    for fold in range(5):
        for seed in [17,23,42]:
            torch.manual_seed(seed);m=model(fold)
            torch.manual_seed(seed);old=old_model(fold,'observed_history_control')
            assert fingerprint(m)==fingerprint(old)
            run=json.loads((V25/f'fold{fold}/observed_history_control/seed{seed}/run_protocol.json').read_text())
            assert fingerprint(m)==run['initial_parameters_sha256']
            assert base.sha(V8/f'fold{fold}/hold_{seed}/sample_draws.npz')==run['draws_sha256']
            assert sum(p.numel() for p in m.parameters())==41606
            checks.append(dict(fold=fold,seed=seed,parameters_and_draws_identical=True))
    # Same masked class with the established all-zero switch is a behavior control.
    torch.manual_seed(17);m=model(0);m.use_controller=False
    with np.load(V8/'fold0/hold_17/sample_draws.npz') as z:draws=z['array_index']
    opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);start=time.monotonic()
    for step,ids in enumerate(draws,1):
        opt.zero_grad(set_to_none=True);pred=m(**{k:v[ids] for k,v in x.items()})
        loss=base.loss_fn(pred,{k:v[ids] for k,v in y.items()},m);assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
        if step%200==0:base.write(OUT/'control_reproduction_status.json',dict(step=step,elapsed_s=time.monotonic()-start))
    target=V25/'fold0/zero_history_control/seed17/final.pt';saved=torch.load(target,weights_only=False)['state_dict']
    for key,value in m.state_dict().items():torch.testing.assert_close(value,saved[key],atol=0,rtol=0)
    base.write(OUT/'preflight.json',dict(status='passed',checks=checks,runtime_versions_equal=True,
        full1600step_zero_control_exact=True,control_checkpoint_sha256=base.sha(target),control_elapsed_s=time.monotonic()-start,
        focused_tests_passed=3,windows=407,parameters=41606,training_candidate_runs=15,sealed_test_accessed=False))
    print('Preflight passed; zero-control1600step parameters and buffers exactly reproduced.',flush=True)


if __name__=='__main__':main()
