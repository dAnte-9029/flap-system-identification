"""Read-only Step 1 probes for the frozen August Main V2; not a flight benchmark."""
import copy, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from system_identification.models.trajectory_main_v1 import CausalHistoryTrajectoryModel
from system_identification.models.trajectory_main_v2 import ActuatorAwareTrajectoryModel
from system_identification.training.trajectory_main_v1 import assemble_history_trajectory_windows, _model_call

torch.set_num_threads(1)
p = ROOT / 'artifacts/trajectory_main_v2/models/main_v2_drive_tail_gated.pt'
c = torch.load(p, map_location='cpu', weights_only=False)
s = c['state_dict']
stats = {k: s['base_model.' + k].numpy() for k in ('feature_mean','feature_std','control_mean','control_std','derivative_mean','derivative_std')}
base = CausalHistoryTrajectoryModel(hidden_size=c['base_config']['hidden_size'], use_controls=False, **stats)
model = ActuatorAwareTrajectoryModel(base_model=base, tail_mean=c['tail_mean'], tail_std=c['tail_std'], **{k:c['config'][k] for k in ('use_drive','use_tail','gated_tail','drive_tau_s','tail_tau_s','initial_tail_gate')})
model.load_state_dict(s, strict=True)
model.eval()
d = ROOT / 'dataset/trajectory_v1_august_f5_c4'
w = pd.read_parquet(d / 'windows_validation.parquet').groupby('log_id', sort=False).head(6).groupby('log_id', sort=False).tail(1)
b = assemble_history_trajectory_windows(pd.read_parquet(d / 'samples_validation.parquet'), w, history_steps=26)
i = np.arange(len(w))
with torch.inference_mode():
    pred, _ = _model_call(model, b, i, use_history=True, rollout_steps=100, device=torch.device('cpu'))
    poisoned = copy.deepcopy(b)
    for name in ('position_n','velocity_n','quaternion_nb','angular_velocity_b','relative_phase_rad','flap_frequency_hz'):
        getattr(poisoned.trajectory.truth, name)[:, 1:] = np.nan
    changed, _ = _model_call(model, poisoned, i, use_history=True, rollout_steps=100, device=torch.device('cpu'))
    equal = all(torch.equal(a, z) for a,z in zip(pred, changed))
    args = dict(history_state_features=torch.tensor(b.history_state_features, dtype=torch.float32), history_controls=torch.tensor(b.history_controls, dtype=torch.float32), history_mask=torch.tensor(b.history_mask))
    for name in ('position_n','velocity_n','quaternion_nb','angular_velocity_b','relative_phase_rad','flap_frequency_hz'):
        args[name] = torch.tensor(getattr(b.trajectory.truth, name)[:,0], dtype=torch.float32)
    args['future_controls'] = torch.tensor(b.trajectory.controls[:,:1], dtype=torch.float32).repeat(1,250,1)
    args['dt_s'] = torch.full((len(w),250),0.02)
    long = model(**args)
    finite = all(bool(torch.isfinite(v).all()) for v in long)
result = dict(checkpoint=str(p), checkpoint_sha256=hashlib.sha256(p.read_bytes()).hexdigest(), window_ids=w.window_id.tolist(), log_ids=w.log_id.tolist(), future_truth_nan_prediction_bitwise_equal=equal, fixed_dt_250_step_all_outputs_finite=finite, quaternion_max_norm_error=float((torch.linalg.vector_norm(long.quaternion_nb,dim=-1)-1).abs().max()), command_contract='hold t0 command for 250 steps; causal history initialization; runtime smoke only, no 5s accuracy or envelope claim', sealed_test_opened=False, device='cpu')
assert equal and finite
print(json.dumps(result, indent=2))
