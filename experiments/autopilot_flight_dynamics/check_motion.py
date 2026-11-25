"""Check capacity, nested initialization, temporal support and the component API."""
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch

from model import FlightPredictor


HERE = Path(__file__).resolve().parent
LOW = "505a14493a150fec7b61938a9871a80f65ba5452"


def main():
    torch.set_num_threads(2)
    # These inputs exercise the original real clocks and commands. Synthetic
    # appended features exercise the input contract independently of cache fitting.
    train = json.loads((HERE / "data.json").read_text())["split"]["train"]["file"]
    with np.load(train) as src:
        hist, pooled, u0, time = (torch.from_numpy(src[k][:19]) for k in ("hist", "pooled", "u0", "time"))
    noise = torch.randn(len(hist), 14, generator=torch.Generator().manual_seed(9))
    hist = torch.cat((hist, noise), dim=-1)
    mean, std = hist.mean(0), hist.std(0).clamp_min(0.05)
    local = {}
    original = subprocess.check_output(["git", "-C", str(HERE), "show",
        f"{LOW}:experiments/autopilot_flight_dynamics/model.py"], text=True)
    exec(compile(original, "frozen_low_order_model", "exec"), local)
    torch.manual_seed(17)
    baseline = local["FlightPredictor"](mean, std)
    torch.manual_seed(17)
    model = FlightPredictor(mean, std)
    torch.manual_seed(17)
    control = FlightPredictor(mean, std)
    model.carrier_basis, control.carrier_basis = "wingbeat", "slow"
    inputs = hist, pooled, u0, time
    with torch.no_grad():
        expected = baseline(*inputs)
        for candidate in (model, control):
            torch.testing.assert_close(expected, candidate(*inputs), atol=0, rtol=0)
        model.carrier_coefficients.weight.normal_(std=0.03)
        model.carrier_coefficients.bias.normal_(std=0.03)
        control.load_state_dict(model.state_dict())
    counts, components = {}, {}
    for name, candidate in (("wingbeat", model), ("slow", control)):
        counts[name] = sum(p.numel() for p in candidate.parameters())
        slow, wave, wave0 = candidate.components(*inputs)
        pred = candidate(*inputs)
        torch.testing.assert_close(pred, slow + (wave - wave0), atol=0, rtol=0)
        assert wave.shape == pred.shape == (len(hist), 8, 2, 3)
        assert wave0.shape == (len(hist), 1, 2, 3)
        coefficients = candidate.carrier_coefficients
        pred.square().mean().backward()
        assert (coefficients.weight.grad.abs().sum(1) > 0).all()
        assert (candidate.decay_logits.grad.abs() > 0).all()
        empty_pool = torch.cat((torch.zeros_like(pooled[:, :1]), pooled), dim=1)
        origin_query = torch.cat((torch.zeros_like(time[:, :1]), time), dim=1)
        origin_pred = candidate(hist, empty_pool, u0, origin_query)
        torch.testing.assert_close(origin_pred[:, :1], torch.zeros_like(origin_pred[:, :1]), rtol=0, atol=0)
        torch.testing.assert_close(origin_pred[:, 1:], pred, atol=1e-6, rtol=1e-6)
        for split in range(1, time.shape[1]):
            later_pool = pooled.clone()
            later_time = time.clone()
            later_pool[:, split:] += 0.1
            later_time[:, split:] += 0.03
            torch.testing.assert_close(candidate(hist, later_pool, u0, later_time)[:, :split],
                                       pred[:, :split], atol=0, rtol=0)
        components[name] = (slow, wave, wave0)
    assert counts["wingbeat"] == counts["slow"]
    for ix in (0, 2):
        torch.testing.assert_close(components["wingbeat"][ix], components["slow"][ix], rtol=0, atol=0)
    assert (components["wingbeat"][1] - components["slow"][1]).abs().max() > 0.01
    # With known coefficients and no slow contribution, verify the measured
    # origin frequency is transported independently onto each native query clock.
    with torch.no_grad():
        model.carrier_coefficients.weight.zero_()
        model.carrier_coefficients.bias.copy_(torch.arange(1., 13.))
        _, wave, origin = model.components(*inputs)
        coefficients = torch.arange(1., 13., dtype=torch.float64).reshape(2, 3, 2)
        phase = 2 * math.pi * hist[:, None, None, 165].double() * time.double()
        waves64 = (coefficients[..., 0] * phase.cos().unsqueeze(-1)
                   + coefficients[..., 1] * phase.sin().unsqueeze(-1))
        independent = torch.exp(-time.double().unsqueeze(-1)) * waves64
        err = (wave.double() - independent).abs().max().item()
        assert err < 2e-4
        torch.testing.assert_close(origin[0, 0].double(), coefficients[..., 0], rtol=0, atol=0)
    print(json.dumps(dict(train_windows=len(hist), baseline_parameters=sum(p.numel() for p in baseline.parameters()),
        full_parameters=counts["wingbeat"], control_parameters=counts["slow"],
        baseline_initial_max_error=0, prefix_checks_per_variant=7,
        real_query_vs_float64_carrier_max_error=err, origin_query_zero=True,
        quadrature_gradient_rows_per_variant=12, decay_gradients_per_variant=6), indent=2))


if __name__ == "__main__":
    main()
