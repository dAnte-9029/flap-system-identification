"""Restore saved candidate heads and check the causal prediction contract."""
import json

import numpy as np
import torch

from fit_flight import SEEDS, load_data
from fit_clock import SOURCE
from module import HERE, MomentPredictor, build_design, causal_conditions


@torch.inference_mode()
def main():
    dev, ids = load_data(SOURCE, 'validation')
    selected = np.arange(0, len(ids), 277)
    b = {k:v[selected].contiguous() for k,v in dev.items()}
    args = [b[k] for k in ('hist','pooled','u0','time')]
    checks = {}
    for seed in SEEDS:
        state = torch.load(HERE / f'conditional_seed{seed}.pt', map_location='cpu', weights_only=True)
        m = MomentPredictor(state['mean'], state['std']).eval()
        m.load_state_dict(state)
        output = m(*args)
        slow, wave, origin = m.components(*args)
        err = float((output - (slow + wave - origin)).abs().max())
        assert err < 2e-6
        altered = b['pooled'].clone()
        altered[:,4:] += 2
        prefix_output = m(b['hist'], altered, b['u0'], b['time'])
        assert torch.equal(prefix_output[:,:4], output[:,:4])
        zero_t = torch.zeros_like(b['time'])
        zero_basis = m.bases(b['hist'], torch.zeros_like(altered), b['u0'], zero_t)
        assert all(torch.count_nonzero(v)==0 for v in zero_basis.values())
        assert torch.isfinite(output).all()
        checks[str(seed)] = dict(max_component_discrepancy=err, shape=list(output.shape),
            earlier_prediction_change_after_future_command_intervention=0.0, origin_readout=0.0)
    (HERE / 'verification.json').write_text(json.dumps(checks, indent=2)+'\n')


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
